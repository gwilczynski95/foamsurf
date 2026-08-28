from __future__ import annotations

from dataclasses import asdict

from training import ExperimentConfig
from training.joint_scheduling_sweeps import (
    JOINT_SCHEDULING_SPECS,
    build_joint_scheduling_config,
)


def test_scheduling_variants_change_only_the_declared_controls() -> None:
    base = ExperimentConfig.load("configs/baseline_clean_joint.yaml")
    configs = [
        build_joint_scheduling_config(base, index, f"/tmp/run{index}", "test")
        for index in range(len(JOINT_SCHEDULING_SPECS))
    ]

    for config in configs:
        assert config.seed == 0
        assert config.renderer.gaussian_hit_buffer_size == 64
        assert config.dataset.train_target == "both"
        assert config.optimization.reconstruction_loss == "l1_dssim"
        assert config.optimization.dssim_weight == 0.2
        assert config.model == base.model
        assert config.renderer == base.renderer

    assert configs[0].optimization.gaussian_freeze_until == 2_000
    assert configs[1].voronoi_adaptation.adaptation_offset == 100
    assert configs[2].gaussian_adaptation.densify_until == 10_000
    assert configs[3].voronoi_adaptation.freeze_at_capacity is True

    ignored = {"output_dir", "experiment_name", "wandb"}
    base_dict = asdict(base)
    for config, spec in zip(configs, JOINT_SCHEDULING_SPECS):
        candidate = asdict(config)
        for key in ignored:
            candidate.pop(key)
            base_dict.pop(key, None)
        candidate["optimization"]["gaussian_freeze_until"] = 0
        candidate["gaussian_adaptation"]["densify_until"] = 15_000
        candidate["voronoi_adaptation"]["adaptation_offset"] = 0
        candidate["voronoi_adaptation"]["freeze_at_capacity"] = False
        assert candidate == base_dict, spec.label
