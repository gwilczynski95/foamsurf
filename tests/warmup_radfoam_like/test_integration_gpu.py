from copy import deepcopy
from dataclasses import replace
import random
import numpy as np
import pytest
import torch
from training.warmup_profile import WarmupProfile
from training.warmup_sweep import configuration
from training.warmup_trainer import WarmupTrainer, WarmupTopology
from training.trainer import Trainer


def tiny_config(tmp_path):
    config,_,_=configuration(7,tmp_path/'train')
    config.dataset.resolution=8  # 100px base, 25px at factor4: SSIM remains defined
    config.optimization.iterations=6
    config.optimization.topology_update_interval=2
    config.voronoi_warmup.milestone_iterations=[0,2,4,6]
    config.model.medium_points=128
    config.voronoi_adaptation.min_points=32
    config.voronoi_adaptation.max_points=256
    config.voronoi_adaptation.densify_from=4
    config.voronoi_adaptation.densify_until=7
    config.voronoi_adaptation.interval=4
    config.voronoi_adaptation.densify_fraction=.1
    config.validation.image_views=0
    config.validation.max_eval_views=1
    config.validation.eval_interval=6
    config.wandb.mode='disabled'
    profile=replace(WarmupProfile(),outer_points=8,resolution_steps=[0,2,4],
                    quantile_ramp=0,density_warmup=0)
    return config,profile


def sorted_field(trainer):
    m=trainer.model.medium
    order=torch.argsort(m.sites[:,0])
    return torch.cat((m.sites,m.densities,m.colors,m.guard_mask[:,None]),dim=1)[order].detach()


def test_progressive_adaptation_checkpoint_resume_and_legacy_read(tmp_path):
    config,profile=tiny_config(tmp_path)
    trainer=WarmupTrainer(config,profile)
    try:
        initial_gaussians={k:v.clone() for k,v in trainer.model.gaussians.state_dict().items()}
        outer=trainer.model.medium.sites.detach()[trainer.model.medium.outer_mask].clone()
        for i in (1,2):
            loss,*_=trainer._train_step(i)
            assert torch.isfinite(loss.total)
        checkpoint=tmp_path/'resume.pt';trainer._checkpoint(checkpoint,2)
        # Continue over a resolution switch and topology change.
        for i in (3,4,5):trainer._train_step(i)
        expected=sorted_field(trainer)
        expected_stack=list(trainer.sampler._camera_stack)
        assert trainer._training_resolution==1
        assert trainer.voronoi_adaptation.total_densified>0
        current_outer=trainer.model.medium.sites.detach()[trainer.model.medium.outer_mask]
        torch.testing.assert_close(torch.sort(current_outer,dim=0).values,torch.sort(outer,dim=0).values,rtol=0,atol=0)
        assert int(trainer.model.medium.outer_mask.sum())==8
        for k,v in trainer.model.gaussians.state_dict().items():torch.testing.assert_close(v,initial_gaussians[k],rtol=0,atol=0)
    finally:trainer.logger.close()
    resumed_config=deepcopy(config);resumed_config.output_dir=str(tmp_path/'resumed')
    resumed=WarmupTrainer(resumed_config,profile,resume=checkpoint)
    try:
        for i in (3,4,5):resumed._train_step(i)
        torch.testing.assert_close(sorted_field(resumed),expected,rtol=5e-4,atol=1e-4)
        assert resumed.sampler._camera_stack==expected_stack
        resumed._checkpoint(tmp_path/'handoff.pt',5)
        canonical=deepcopy(config);canonical.output_dir=str(tmp_path/'evaluation')
        old=Trainer(canonical,resume=tmp_path/'handoff.pt')
        try:
            # Standard FoamSurf reads the same physical field using beta1.
            torch.testing.assert_close(sorted_field(old),sorted_field(resumed),rtol=2e-5,atol=2e-6)
        finally:old.logger.close()
    finally:resumed.logger.close()


def test_iteration_zero_checkpoint_field_and_sampling(tmp_path):
    """Check initial resume separately, before Adam magnifies gradient roundoff."""
    config, profile = tiny_config(tmp_path)
    trainer = WarmupTrainer(config, profile)
    checkpoint = tmp_path / "initial.pt"
    try:
        trainer._checkpoint(checkpoint, 0)
        expected = sorted_field(trainer)
        batch = trainer.sampler.next()
        expected_q = torch.rand(len(batch.target), 2, device=trainer.device)
        expected_stack = list(trainer.sampler._camera_stack)
    finally:
        trainer.logger.close()
    resumed_config = deepcopy(config)
    resumed_config.output_dir = str(tmp_path / "initial_resumed")
    resumed = WarmupTrainer(resumed_config, profile, resume=checkpoint)
    try:
        torch.testing.assert_close(sorted_field(resumed), expected, rtol=2e-5, atol=2e-6)
        other = resumed.sampler.next()
        assert other.camera_name == batch.camera_name
        for actual, wanted in ((other.target, batch.target),
                               (other.alpha, batch.alpha),
                               (other.rays.origins, batch.rays.origins),
                               (other.rays.directions, batch.rays.directions)):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
        torch.testing.assert_close(
            torch.rand(len(other.target), 2, device=resumed.device), expected_q, rtol=0, atol=0,
        )
        assert resumed.sampler._camera_stack == expected_stack
        assert all(group["eps"] == profile.adam_eps for group in resumed.medium_optimizer.param_groups)
    finally:
        resumed.logger.close()


def test_opacity_supervision_uses_sampled_alpha_at_black_background(tmp_path, monkeypatch):
    config, profile = tiny_config(tmp_path)
    config.background = [0., 0., 0.]
    profile = replace(profile, quantile_weight=0., opacity_weight=1.)
    trainer = WarmupTrainer(config, profile)
    try:
        # Exercise the real step on one fixed field. Disabling optimizer.step
        # lets us compare gradients without Adam/topology reconstruction noise.
        monkeypatch.setattr(trainer.medium_optimizer, "step", lambda: None)
        medium = trainer.model.medium
        initial = {name: value.detach().clone() for name, value in medium.named_parameters()}
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
               torch.cuda.get_rng_state_all())
        stack = list(trainer.sampler._camera_stack)
        original_next = trainer.sampler.next
        batches = []

        def record_batch():
            batch = original_next()
            batches.append((batch.target.detach().clone(), batch.alpha.detach().clone(),
                            batch.rays.origins.detach().clone(),
                            batch.rays.directions.detach().clone()))
            return batch

        monkeypatch.setattr(trainer.sampler, "next", record_batch)
        results = []
        # At black background, changing alpha metadata while keeping premultiplied
        # foreground fixed leaves the RGB target unchanged. Any changed loss/
        # density gradient must therefore come from opacity supervision.
        for alpha in (0., 0., 1.):
            random.setstate(rng[0])
            np.random.set_state(rng[1])
            torch.set_rng_state(rng[2])
            torch.cuda.set_rng_state_all(rng[3])
            trainer.sampler._camera_stack = list(stack)
            for camera in trainer.sampler.dataset.train:
                camera.alpha.fill_(alpha)
            loss, *_ = trainer._train_step(1)  # no adaptation/topology update at step 1
            gradients = {name: value.grad.detach().clone()
                         for name, value in medium.named_parameters()}
            results.append((loss.total.detach().clone(), gradients, dict(trainer.warmup_metrics)))
            for name, value in medium.named_parameters():
                torch.testing.assert_close(value, initial[name], rtol=0, atol=0)

        # Numerical-repeatability control.
        for actual, expected in zip(batches[1], batches[0]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(results[1][0], results[0][0], rtol=2e-5, atol=2e-6)
        for name in initial:
            torch.testing.assert_close(results[1][1][name], results[0][1][name],
                                       rtol=2e-5, atol=2e-6)

        # Alpha changes, but premultiplied RGB/rays do not.
        torch.testing.assert_close(batches[2][0], batches[0][0], rtol=0, atol=0)
        torch.testing.assert_close(batches[2][2], batches[0][2], rtol=0, atol=0)
        torch.testing.assert_close(batches[2][3], batches[0][3], rtol=0, atol=0)
        assert torch.count_nonzero(batches[0][1]) == 0
        assert torch.count_nonzero(batches[2][1] - 1) == 0
        assert results[0][2]["warmup/target_alpha_mean"] == 0.
        assert results[2][2]["warmup/target_alpha_mean"] == 1.
        for result in results:
            metrics = result[2]
            assert result[0].item() == pytest.approx(
                metrics["warmup/rgb_loss"] + metrics["warmup/weighted_opacity_loss"],
                rel=2e-5, abs=2e-6,
            )
        assert results[0][2]["warmup/opacity_loss"] != pytest.approx(
            results[2][2]["warmup/opacity_loss"], abs=1e-6)
        assert not torch.allclose(results[0][1]["raw_density"],
                                  results[2][1]["raw_density"], rtol=2e-5, atol=2e-6)
    finally:
        trainer.logger.close()


def test_zero_opacity_weight_recovers_rgb_only_objective(tmp_path, monkeypatch):
    config, profile = tiny_config(tmp_path)
    config.background = [0., 0., 0.]
    profile = replace(profile, quantile_weight=0., opacity_weight=0.)
    trainer = WarmupTrainer(config, profile)
    try:
        monkeypatch.setattr(trainer.medium_optimizer, "step", lambda: None)
        loss, *_ = trainer._train_step(1)
        assert loss.total.item() == pytest.approx(
            trainer.warmup_metrics["warmup/rgb_loss"], rel=2e-5, abs=2e-6)
        assert trainer.warmup_metrics["warmup/weighted_opacity_loss"] == 0.
    finally:
        trainer.logger.close()

def test_warmup_rejects_joint_mode_before_initialization(tmp_path):
    config,profile=tiny_config(tmp_path)
    config.representation_mode='hybrid'
    with pytest.raises(ValueError,match='Voronoi-only'):WarmupTrainer(config,profile)


def test_corner_guards_are_triangulable_and_protected(tmp_path):
    config, profile = tiny_config(tmp_path)
    config.output_dir = str(tmp_path / "corners")
    profile = replace(profile, corner_guards=True, outer_points=8,
                      quantile_weight=0., opacity_weight=1.)
    trainer = WarmupTrainer(config, profile)
    try:
        medium = trainer.model.medium
        assert int(medium.outer_mask.sum()) == 8
        assert torch.unique(medium.sites.detach()[medium.outer_mask], dim=0).shape[0] == 8
        before = medium.sites.detach()[medium.outer_mask].clone()
        loss, *_ = trainer._train_step(1)
        assert torch.isfinite(loss.total)
        after = medium.sites.detach()[medium.outer_mask]
        torch.testing.assert_close(
            torch.sort(before, dim=0).values, torch.sort(after, dim=0).values,
            rtol=0, atol=0,
        )
    finally:
        trainer.logger.close()


def test_outer_protection_on_replacement_and_retry(tmp_path,monkeypatch):
    config,profile=tiny_config(tmp_path)
    trainer=WarmupTrainer(config,profile)
    try:
        medium=trainer.model.medium
        n=medium.point_count
        ids=torch.arange(n,device=trainer.device)
        outer=medium.outer_mask.clone()
        with pytest.raises(ValueError,match='cannot be pruned'):
            trainer.topology.replace_population(medium,trainer.medium_optimizer,ids[~outer],ids[:0],medium.sites[:0])
        with pytest.raises(ValueError,match='parents'):
            trainer.topology.replace_population(medium,trainer.medium_optimizer,ids,ids[outer][:1],medium.sites[outer][:1])
        import foamsurf_topology
        original=foamsurf_topology.Triangulation
        calls=[]
        def once(points):
            calls.append(points.clone())
            if len(calls)==1:raise foamsurf_topology.TriangulationFailedError('forced retry')
            return original(points)
        monkeypatch.setattr(foamsurf_topology,'Triangulation',once)
        parent=ids[~outer][:1]
        child=medium.sites[parent].detach()+.001
        before=medium.sites.detach()[outer].clone()
        assert trainer.topology.replace_population(medium,trainer.medium_optimizer,ids,parent,child)
        assert len(calls)==2
        after=medium.sites.detach()[medium.outer_mask]
        torch.testing.assert_close(torch.sort(before,dim=0).values,torch.sort(after,dim=0).values,rtol=0,atol=0)
    finally:trainer.logger.close()
