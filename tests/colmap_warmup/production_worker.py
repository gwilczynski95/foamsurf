"""Real initial population, full-resolution images and one-million-ray batches."""
from pathlib import Path
import json
import sys
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from training.end_to_end import EndToEndConfig
from training.warmup_trainer import WarmupTrainer

c=EndToEndConfig.load(ROOT/'configs/end_to_end_iui3.yaml')
c.warmup.output_dir=sys.argv[1]
c.warmup.wandb.mode='disabled'
c.profile.resolution_steps=[0,1,2]
c.warmup.voronoi_adaptation.densify_from=4
trainer=WarmupTrainer(c.warmup,c.profile)
assert trainer.model.medium.point_count==24716
assert trainer.config.optimization.rays_per_batch==1000000
assert (trainer.dataset.width,trainer.dataset.height)==(1383,917)
for iteration in range(1,5):
    loss,*_=trainer._train_step(iteration)
    assert torch.isfinite(loss.total)
assert trainer._training_resolution==1
assert trainer.voronoi_adaptation.total_densified>0
report=dict(steps=4,initial_sites=24716,final_sites=trainer.model.medium.point_count,
            width=trainer.dataset.width,height=trainer.dataset.height,rays_per_batch=1000000,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
(Path(sys.argv[1])/'production_smoke.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
