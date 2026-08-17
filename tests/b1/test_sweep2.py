from __future__ import annotations

from pathlib import Path

from training.config import ExperimentConfig
from training.sweep2 import SWEEP2_SPECS, build_sweep2_config


def _configs(tmp_path):
    root = Path(__file__).resolve().parents[2]
    base = ExperimentConfig.load(root / "configs" / "b1_lego_layers.yaml")
    return [
        build_sweep2_config(
            base,
            index,
            tmp_path / spec.label,
            f"test-{spec.label}",
        )
        for index, spec in enumerate(SWEEP2_SPECS)
    ]


def test_sweep2_has_ten_unique_resolved_05_based_configs(tmp_path) -> None:
    configs = _configs(tmp_path)
    assert len(SWEEP2_SPECS) == 10
    assert len({spec.label for spec in SWEEP2_SPECS}) == 10
    for config in configs:
        assert config.optimization.iterations == 30_000
        assert config.optimization.full_image_batch is True
        assert config.optimization.cameras_without_replacement is True
        assert config.gaussian_adaptation.enabled is True
        assert config.gaussian_adaptation.densify_until == 15_000
        assert config.gaussian_adaptation.max_points is None
        assert config.gaussian_adaptation.opacity_reset_interval == 3_000
        assert config.gaussian_adaptation.medium_freeze_after_reset == 0
        assert config.gaussian_adaptation.runaway_max_points == 1_000_000
        assert config.gaussian_adaptation.screen_pruning_enabled is True
        assert config.wandb.project == "foamsurf_sweep2"


def test_sweep2_key_ablations_are_isolated(tmp_path) -> None:
    configs = _configs(tmp_path)
    control, l1, l1dssim, split, clone, both, combined = configs[:7]
    assert control.optimization.reconstruction_loss == "mse"
    assert control.gaussian_adaptation.opacity_reset_value == 0.05
    assert l1.optimization.reconstruction_loss == "l1"
    assert l1dssim.optimization.reconstruction_loss == "l1_dssim"
    assert split.gaussian_adaptation.preserve_split_opacity is True
    assert split.gaussian_adaptation.preserve_clone_opacity is False
    assert clone.gaussian_adaptation.preserve_clone_opacity is True
    assert clone.gaussian_adaptation.preserve_split_opacity is False
    assert both.gaussian_adaptation.preserve_clone_opacity is True
    assert both.gaussian_adaptation.preserve_split_opacity is True
    assert combined.optimization.reconstruction_loss == "l1_dssim"
    assert combined.gaussian_adaptation.preserve_split_opacity is True
    assert combined.gaussian_adaptation.preserve_clone_opacity is False
    assert configs[7].gaussian_adaptation.opacity_reset_value == 0.075
    assert configs[8].gaussian_adaptation.opacity_reset_value == 0.075
    assert configs[9].gaussian_adaptation.opacity_reset_value == 0.10


def test_sweep2_sbatch_labels_match_python_specs() -> None:
    root = Path(__file__).resolve().parents[2]
    script = (root / "scripts" / "sweep2_30k.sbatch").read_text()
    for spec in SWEEP2_SPECS:
        assert f'"{spec.label}"' in script
    assert "#SBATCH --array=0-9" in script
    assert 'WANDB_PROJECT="${WANDB_PROJECT:-foamsurf_sweep2}"' in script
