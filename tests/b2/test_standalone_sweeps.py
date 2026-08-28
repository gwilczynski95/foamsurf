from pathlib import Path

from training.config import ExperimentConfig
from training.standalone_sweeps import (
    GAUSSIAN_SWEEP_SPECS,
    GAUSSIAN_SWEEP_FAMILY_COUNTS,
    VORONOI_INITIAL_POINTS,
    VORONOI_SWEEP_SPECS,
    build_standalone_sweep_config,
)

ROOT = Path(__file__).resolve().parents[2]


def test_standalone_sweep_matrix_is_large_unique_and_contains_16k_32k() -> None:
    assert len(GAUSSIAN_SWEEP_SPECS) == 372
    assert GAUSSIAN_SWEEP_FAMILY_COUNTS == {
        "control": 48, "graphdeco": 144,
        "scale_shape": 72, "flare_movement": 108,
    }
    assert len(VORONOI_SWEEP_SPECS) == 70
    assert len(GAUSSIAN_SWEEP_SPECS) + len(VORONOI_SWEEP_SPECS) == 442
    assert len({spec.label for spec in GAUSSIAN_SWEEP_SPECS}) == 372
    assert len({spec.label for spec in VORONOI_SWEEP_SPECS}) == 70
    assert 16_384 in VORONOI_INITIAL_POINTS
    assert 32_768 in VORONOI_INITIAL_POINTS
    for points in VORONOI_INITIAL_POINTS:
        assert any(spec.initial_points == points and spec.adaptive is False for spec in VORONOI_SWEEP_SPECS)
        assert any(spec.initial_points == points and spec.adaptive is True for spec in VORONOI_SWEEP_SPECS)


def test_every_standalone_sweep_config_validates_and_targets_correct_layer() -> None:
    cases = (
        ("gaussian", "stage1_gaussian_flare_control.yaml", GAUSSIAN_SWEEP_SPECS, "lego"),
        ("voronoi", "stage2_voronoi_only.yaml", VORONOI_SWEEP_SPECS, "clouds"),
    )
    for phase, filename, specs, target in cases:
        base = ExperimentConfig.load(ROOT / "configs" / filename)
        fingerprints = set()
        for index, spec in enumerate(specs):
            config = build_standalone_sweep_config(
                base, phase, index, "/tmp/test", spec.label, "disabled"
            )
            assert config.dataset.train_target == target
            assert config.optimization.iterations == 30_000
            fingerprint = config.as_dict()
            fingerprint.pop("experiment_name")
            fingerprint.pop("output_dir")
            fingerprint.pop("wandb")
            fingerprints.add(str(fingerprint))
            if phase == "gaussian":
                assert config.gaussian_adaptation.max_points is None
                assert config.gaussian_adaptation.runaway_max_points is None
                assert config.validation.eval_interval == 2_000
                assert not config.diagnostics.save_per_primitive
        assert len(fingerprints) == len(specs)


def test_stage12_sbatch_array_bounds_match_sweep_specs() -> None:
    gaussian = (ROOT / "scripts/sweep_stage1_gaussians.sbatch").read_text()
    voronoi = (ROOT / "scripts/sweep_stage2_voronoi.sbatch").read_text()
    smoke = (ROOT / "scripts/smoke_stage12.sbatch").read_text()
    assert "#SBATCH --array=0-371%8" in gaussian
    assert "#SBATCH --array=0-69%8" in voronoi
    assert "#SBATCH --exclude=c22,c11" in gaussian
    assert "stage1_gaussian_flare_control.yaml" in gaussian
    assert "stage1_gaussian_only.yaml" not in gaussian
    assert "#SBATCH --array=0-1%1" in smoke
    assert "foamsurf/foamsurf" in gaussian
    assert "conda/envs/foamsurf" in voronoi
