"""Required RTX4090 integration tests, included by the existing certificate gate."""
from copy import deepcopy
from dataclasses import replace
import torch
import pytest
from test_integration_gpu import tiny_config, sorted_field
from training.warmup_trainer import WarmupTrainer
from training.warmup_sampling import MixedRaySampler
from training.trainer import Trainer
from foamsurf_model.voronoi_adaptation import VoronoiTopologyPlan


def organic_config(tmp_path):
    config,profile=tiny_config(tmp_path)
    config.optimization.rays_per_batch=512
    config.voronoi_adaptation.densify_from=2
    config.voronoi_adaptation.densify_until=9
    config.voronoi_adaptation.max_points=512
    profile=replace(profile, site_initialization="random",ray_sampling="mixed",
        population_policy="radfoam",topology_schedule="radfoam",
        outer_positions_trainable=True,population_target=512,population_horizon=8,
        population_min_interval=2,error_map_chunk=4096,
        quantile_weight=0.,resolutions=[1],resolution_steps=[0])
    config.dataset.resolution=32  # 25x25, all train cameras used in maps
    return config,profile


def test_mixed_step_adapts_resumes_and_exports_canonical_field(tmp_path):
    config,profile=organic_config(tmp_path)
    trainer=WarmupTrainer(config,profile)
    try:
        gaussians={k:v.clone() for k,v in trainer.model.gaussians.state_dict().items()}
        for i in (1,2):
            loss,*_=trainer._train_step(i)
            assert torch.isfinite(loss.total)
            trainer.logger.log_train(trainer._training_row(i,loss,0.,0.,0.))
        assert trainer.voronoi_adaptation.total_densified > 0
        assert trainer.voronoi_adaptation.next_iteration == 4
        assert trainer.warmup_metrics["warmup/error_map_seconds"] > 0
        checkpoint=tmp_path/"organic.pt"
        trainer._checkpoint(checkpoint,2)
        expected=sorted_field(trainer)
        batch=trainer.sampler.next()
        for k,v in trainer.model.gaussians.state_dict().items():
            torch.testing.assert_close(v,gaussians[k],rtol=0,atol=0)
        period=(trainer._topology_period,trainer._topology_since_update)
    finally:
        trainer.logger.close()
    cfg=deepcopy(config);cfg.output_dir=str(tmp_path/"resumed")
    resumed=WarmupTrainer(cfg,profile,resume=checkpoint)
    try:
        torch.testing.assert_close(sorted_field(resumed),expected,rtol=2e-5,atol=2e-6)
        assert resumed.voronoi_adaptation.next_iteration == 4
        assert (resumed._topology_period,resumed._topology_since_update) == period
        other=resumed.sampler.next()
        for a,b in ((batch.target,other.target),(batch.alpha,other.alpha),
                    (batch.rays.directions,other.rays.directions)):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        for i in (3,4):
            loss,*_=resumed._train_step(i)
            assert torch.isfinite(loss.total)
        assert resumed.voronoi_adaptation.revision >= 2
        resumed._checkpoint(tmp_path/"handoff_organic.pt",4)
        canonical=deepcopy(config);canonical.output_dir=str(tmp_path/"canonical")
        old=Trainer(canonical,resume=tmp_path/"handoff_organic.pt")
        try:
            torch.testing.assert_close(sorted_field(old),sorted_field(resumed),rtol=2e-5,atol=2e-6)
        finally:
            old.logger.close()
    finally:
        resumed.logger.close()


def test_error_map_chunking_sums_gradients_before_norm_without_updating_model(tmp_path):
    config,profile=organic_config(tmp_path)
    trainer=WarmupTrainer(config,profile)
    try:
        trainer._train_step(1)
        # Three cameras suffice to verify reduction across views and chunks.
        trainer.sampler.dataset.train=trainer.sampler.dataset.train[:3]
        trainer.sampler=MixedRaySampler(trainer.sampler.dataset,512,config.background)
        before={k:v.detach().clone() for k,v in trainer.model.medium.named_parameters()}
        rng=torch.cuda.get_rng_state_all()
        trainer._collect_population_map()
        a=trainer.voronoi_adaptation.gradient.clone()
        c=trainer.voronoi_adaptation.contribution.clone()
        assert bool((a>0).any()) and bool((c>0).any())
        torch.cuda.set_rng_state_all(rng)
        trainer.profile=replace(profile,error_map_chunk=17)
        trainer._collect_population_map()
        torch.testing.assert_close(trainer.voronoi_adaptation.gradient,a,rtol=2e-3,atol=2e-5)
        torch.testing.assert_close(trainer.voronoi_adaptation.contribution,c,rtol=2e-4,atol=2e-6)
        for k,v in trainer.model.medium.named_parameters():
            torch.testing.assert_close(v,before[k],rtol=0,atol=0)
            assert v.grad is None
    finally:
        trainer.logger.close()


def test_failed_replacement_rolls_back_density_reset(tmp_path,monkeypatch):
    config,profile=organic_config(tmp_path)
    trainer=WarmupTrainer(config,profile)
    try:
        controller=trainer.voronoi_adaptation
        medium=trainer.model.medium
        n=medium.point_count
        controller.set_error_map(torch.ones(n,device=trainer.device),torch.zeros(n,device=trainer.device))
        plan=controller.propose_plan()
        before=medium.raw_density.detach().clone()
        monkeypatch.setattr(trainer.topology,"replace_population",lambda *a,**kw:False)
        result=controller.apply_plan(plan)
        assert result["skipped"] == 1
        torch.testing.assert_close(medium.raw_density,before,rtol=0,atol=0)
        assert controller.total_density_reset == 0
    finally:
        trainer.logger.close()


def test_outer_parent_inheritance_density_reset_and_optimizer_moments(tmp_path):
    config,profile=organic_config(tmp_path)
    trainer=WarmupTrainer(config,profile)
    try:
        trainer._train_step(1)  # Create nonzero Adam moments.
        medium=trainer.model.medium
        ids=torch.arange(medium.point_count,device=trainer.device)
        outer=ids[medium.outer_mask]
        parent=outer[:1]
        drop=outer[1:2]
        keep=ids[ids != drop[0]]
        child=medium.sites[parent].detach()+torch.tensor([[.03,.02,.01]],device=trainer.device)
        plan=VoronoiTopologyPlan(keep,parent,child,medium.point_count)
        old_color=medium.raw_color.detach().clone()
        old_mask=medium.outer_mask.clone()
        old_moments=trainer.medium_optimizer.state[medium.raw_color]["exp_avg"].clone()
        controller=trainer.voronoi_adaptation
        controller.reset_mask=torch.ones(medium.point_count,dtype=torch.bool,device=trainer.device)
        result=controller.apply_plan(plan)
        assert result == {"densify":1,"prune":1,"skipped":0}
        perm=trainer.topology.last_permutation
        torch.testing.assert_close(medium.raw_color,torch.cat((old_color[keep],old_color[parent]))[perm])
        torch.testing.assert_close(medium.outer_mask,torch.cat((old_mask[keep],old_mask[parent]))[perm])
        torch.testing.assert_close(medium.raw_density,torch.full_like(medium.raw_density,-1.),rtol=2e-6,atol=2e-6)
        torch.testing.assert_close(trainer.medium_optimizer.state[medium.raw_color]["exp_avg"],
                                  torch.cat((old_moments[keep],torch.zeros_like(old_moments[parent])))[perm])
    finally:
        trainer.logger.close()


def test_production_batch_budget_and_fit(tmp_path):
    from training.warmup_organic_sweep import configuration
    config,profile,_=configuration(2,tmp_path/"production_smoke",iterations=3)
    # Keep the real 131072 sites and 1M rays. Small images only reduce dataset
    # residency/evaluation time, not training ray count or cell count.
    config.dataset.resolution=8
    config.validation.image_views=0
    config.validation.max_eval_views=1
    config.validation.eval_interval=3
    config.validation.log_interval=1
    config.wandb.mode="disabled"
    trainer=WarmupTrainer(config,profile)
    result=trainer.fit()
    assert result["status"] == "completed"
    assert result["final_iteration"] == 3
    assert result["voronoi_count"] == 131072
    assert trainer.warmup_metrics["warmup/rays"] == 1000000
    assert trainer.warmup_metrics["warmup/mixed_rays"] == 1
    assert trainer.warmup_metrics["warmup/quantile_weight"] > 0
    assert trainer.warmup_metrics["warmup/truncated_fraction"] == 0
    assert (tmp_path/"production_smoke/checkpoints/last.pt").is_file()
