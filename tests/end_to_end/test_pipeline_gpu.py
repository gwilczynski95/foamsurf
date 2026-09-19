"""Real multi-process warmup/partition/hybrid integration; run through sbatch."""
from pathlib import Path
import json
import subprocess
import sys
import torch
import yaml

ROOT=Path(__file__).resolve().parents[2]


def write_smoke_config(tmp_path):
    from training.end_to_end import EndToEndConfig
    c=EndToEndConfig.load(ROOT/'configs/end_to_end.yaml')
    for config in [c.warmup,c.joint]:
        config.dataset.resolution=32
        config.dataset.gaussian_ground_truth_path=None
        config.dataset.medium_ground_truth_path=None  # handoff must not depend on layer GT
        config.optimization.iterations=4
        config.optimization.rays_per_batch=512
        config.validation.eval_interval=4
        config.validation.log_interval=1
        config.validation.checkpoint_interval=2
        config.validation.max_eval_views=1
        config.validation.image_views=0
        config.wandb.mode='disabled'
    c.warmup.model.medium_points=256
    c.warmup.voronoi_adaptation.densify_from=2
    c.warmup.voronoi_adaptation.densify_until=5
    c.warmup.voronoi_adaptation.min_points=4
    c.warmup.voronoi_adaptation.max_points=512
    c.warmup.voronoi_warmup.milestone_iterations=[0,4]
    c.profile.outer_points=8;c.profile.population_target=512
    c.profile.population_horizon=8;c.profile.population_min_interval=2
    c.profile.error_map_chunk=1024;c.profile.density_warmup=0
    c.profile.resolutions=[1];c.profile.resolution_steps=[0]
    c.joint.gaussian_adaptation.enabled=False
    # Disabled controllers are still validated; keep every window inside the
    # four-step smoke rather than inheriting the production 15k pruning end.
    adaptation=c.joint.gaussian_adaptation
    adaptation.densify_from=1
    adaptation.densify_until=c.joint.optimization.iterations
    adaptation.opacity_prune_from=1
    adaptation.opacity_prune_until=c.joint.optimization.iterations
    adaptation.size_prune_from=1
    c.joint.model.medium_points=32;c.joint.voronoi_adaptation.min_points=32
    c.warmup.save(tmp_path/'warmup.yaml');c.joint.save(tmp_path/'joint.yaml')
    wrapper=dict(output_dir=str(tmp_path/'run'),warmup=dict(config='warmup.yaml',profile=c.profile.as_dict()),
                 joint=dict(config='joint.yaml'),partition=dict(method='otsu_w1',batch_size=128,image_views=1),
                 handoff=dict(gaussian_count=16,pca_neighbors=3,reset_density=1e-8))
    (tmp_path/'pipeline.yaml').write_text(yaml.safe_dump(wrapper))
    path=tmp_path/'pipeline.yaml'
    EndToEndConfig.load(path)  # exact on-disk configuration consumed by workers
    return path


def test_smoke_configuration_roundtrip(tmp_path):
    from training.end_to_end import EndToEndConfig
    c=EndToEndConfig.load(write_smoke_config(tmp_path))
    assert c.warmup.optimization.iterations==c.joint.optimization.iterations==4
    assert c.joint.gaussian_adaptation.opacity_prune_until==4
    assert not c.joint.gaussian_adaptation.enabled
    assert c.warmup.dataset.gaussian_ground_truth_path is None
    assert c.warmup.dataset.medium_ground_truth_path is None


def test_real_native_pipeline_and_resume(tmp_path):
    assert torch.cuda.is_available(), 'GPU test requires sbatch'
    path=write_smoke_config(tmp_path)
    command=[sys.executable,str(ROOT/'scripts/train_end_to_end.py'),'--config',str(path)]
    subprocess.run(command,cwd=ROOT,check=True)
    root=tmp_path/'run'
    for stage in ['warmup','partition','joint']:assert (root/f'{stage}_done.json').is_file()
    start=torch.load(root/'joint/checkpoints/initialization.pt',map_location='cpu',weights_only=False)
    final=torch.load(root/'joint/checkpoints/last.pt',map_location='cpu',weights_only=False)
    assert start['iteration']==0 and final['iteration']==4
    assert not start['gaussian_optimizer']['state']
    assert not start['medium_optimizer']['state']
    assert final['gaussian_optimizer']['state'] and final['medium_optimizer']['state']
    report=json.loads((root/'handoff_report.json').read_text())
    assert report['removed_surface_count']+report['retained_voronoi_count']==report['source_count']
    assert final['model']['medium.sites'].shape[0]==report['retained_voronoi_count']
    assert (root/'transition/view_00_initialization.png').is_file()
    assert (root/'partition/histograms.png').is_file()
    assert (root/'partition/view_00_partition.png').is_file()
    before=(root/'joint/checkpoints/last.pt').stat().st_mtime_ns
    subprocess.run(command+['--resume'],cwd=ROOT,check=True)
    assert (root/'joint/checkpoints/last.pt').stat().st_mtime_ns==before
    # Simulate termination after final checkpoints but before atomic markers.
    (root/'warmup_done.json').unlink()
    (root/'joint_done.json').unlink()
    subprocess.run(command+['--resume'],cwd=ROOT,check=True)
    assert (root/'joint/checkpoints/last.pt').stat().st_mtime_ns==before
    assert (root/'warmup_done.json').is_file() and (root/'joint_done.json').is_file()
