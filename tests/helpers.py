"""Shared test helpers: paths, data and short training manifests."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SEATHRU = Path(os.environ.get("FS_DATASETS", "/shared/sets/datasets/SeathruNeRF_dataset"))
IUI3 = SEATHRU / "IUI3-RedSea-Undistorted"
FOGGY_GARDEN = Path(os.environ.get("FS_FOGGY", "/shared/sets/datasets/foamsurf/foggy_garden")) / "garden_easy"

LEGO = Path(os.environ.get("FS_SYNTHETIC", "/shared/sets/datasets/foamsurf")) / "lego"  # lego_both / lego_lego / lego_clouds

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
requires_lego = pytest.mark.skipif(not (LEGO / "lego_both").is_dir(), reason="needs the synthetic lego scenes (FS_SYNTHETIC)")
requires_iui3 = pytest.mark.skipif(not IUI3.is_dir(), reason="needs the SeaThru-NeRF IUI3 scene (FS_DATASETS)")


def short_manifest(tmp_path, name, overrides=(), config="configs/seathru.yaml", scene=IUI3, iterations=4):
    """The full method of ``config`` on ``scene`` at 1/16 resolution for a few steps: every schedule pulled to the
    first iterations (densification, opacity pruning and reset, OceanSplat window, priors, SH degree), evaluation
    and checkpoints every 2 steps."""
    from train import resolve_iui3
    m = resolve_iui3(ROOT / config, tmp_path / name, overrides=[
        f"warmup.dataset.source_path={scene}", f"joint.dataset.source_path={scene}",
        "warmup.wandb.mode=disabled", "joint.wandb.mode=disabled", *overrides])
    m["profile"]["spatial_count"] = 128
    for stage in ("warmup", "joint"):
        c = m[stage]
        c["model"]["medium_points"] = 128
        c["dataset"]["resolution"] = 16
        c["validation"].update(eval_interval=2, checkpoint_interval=2, log_interval=1, max_eval_views=1, image_views=1)
    j = m["joint"]
    j["optimization"].update(iterations=iterations, training_downscale_levels=0, gaussian_sh_degree_interval=1)
    if j["optimization"]["ocean_until"] > j["optimization"]["ocean_from"]:
        j["optimization"].update(ocean_from=1, ocean_until=iterations + 1, ocean_every=1, ocean_alpha_until=iterations)
    j["gaussian_adaptation"].update(densify_from=1, densify_until=iterations + 2, densification_interval=1,
                                    opacity_prune_from=1, opacity_prune_ramp=0, opacity_prune_until=iterations,
                                    opacity_prune_interval=1, min_opacity_late_from=iterations + 2)
    if j["gaussian_adaptation"]["opacity_reset_interval"]:
        j["gaussian_adaptation"]["opacity_reset_interval"] = 2
    j["disentanglement"].update(opacity_binarization_start=0, opacity_binarization_ramp=0)
    for prior in ("chroma", "surface_coverage", "water_clear", "camera_water"):
        m["medium_fix"][f"{prior}_start"] = 0
        m["medium_fix"][f"{prior}_ramp"] = 0
    m["medium_fix"]["camera_water_refresh"] = 1
    m["rgb_surface"].update(start=0, ramp_steps=0, rays=16)
    m["rgb_smoothing"]["ramp_steps"] = 1
    return m


COSMETIC = ("experiment_name", "wandb.name", "wandb.project", "wandb.mode", "output_dir", "source_path", "seed")


def comparable(manifest: dict) -> dict:
    """A resolved manifest as flat ``key.path: value`` with the defaults of every stage and medium_fix field filled
    in and run names, W&B, paths and seeds left out: two manifests that train the same thing compare equal."""
    from training.config import ExperimentConfig, drop_legacy_keys, update_config
    from training.losses import MediumFixConfig, drop_legacy_medium_fix
    manifest = dict(manifest)
    for stage in ("warmup", "joint"):
        config = ExperimentConfig()
        update_config(config, drop_legacy_keys(manifest[stage]))
        manifest[stage] = config.as_dict()
    manifest["medium_fix"] = {**MediumFixConfig().as_dict(), **drop_legacy_medium_fix(manifest.get("medium_fix", {}))}

    def flat(d, prefix=""):
        out = {}
        for key, value in d.items():
            if isinstance(value, dict):
                out.update(flat(value, prefix + key + "."))
            else:
                out[prefix + key] = value
        return out
    return {k: v for k, v in flat(manifest).items() if not any(c in k for c in COSMETIC)}
