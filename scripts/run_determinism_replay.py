#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from foamsurf_data import load_dataset
from foamsurf_model.scene import HybridSceneModel
from training import ExperimentConfig
from training.determinism import (
    runtime_provenance,
    seed_everything,
    tensor_difference,
    tensor_digest,
    tensors_digest,
    write_json,
)
from training.losses import b0_reconstruction_loss
from training.sweep2 import build_sweep2_config
from training.trainer import Trainer


def _state_tensors(model) -> dict[str, torch.Tensor]:
    return {name: value.detach() for name, value in model.state_dict().items()}


def _gradient_tensors(trainer, screen_gradient) -> dict[str, torch.Tensor]:
    gradients = {"screen_points": screen_gradient.detach()}
    for prefix, module in (
        ("gaussian", trainer.model.gaussians),
        ("medium", trainer.model.medium),
    ):
        for name, parameter in module.named_parameters():
            if parameter.grad is not None:
                gradients[f"{prefix}.{name}"] = parameter.grad.detach()
    return gradients


def _initialization_digest(
    config: ExperimentConfig, seed: int
) -> str:
    seed_everything(seed)
    device = torch.device(config.device)
    dataset = load_dataset(
        config.dataset.source_path,
        resolution=config.dataset.resolution,
        device=device,
        llffhold=config.dataset.llffhold,
    )
    model = HybridSceneModel.initialize(dataset, config, device)
    digest = tensors_digest(_state_tensors(model))
    del model, dataset
    torch.cuda.empty_cache()
    return digest


def _indices_digest(indices: torch.Tensor) -> str:
    return tensor_digest(indices.to(torch.int64))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay FoamSurf forward/backward/topology determinism."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wandb-name", required=True)
    parser.add_argument("--wandb-project", default="foamsurf_determinism_audit")
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replays", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=2.0e-4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    base = ExperimentConfig.load(args.config)
    config = build_sweep2_config(
        base,
        0,
        output_dir,
        args.wandb_name,
        wandb_mode="disabled",
        seed=args.seed,
    )
    config.experiment_name = "determinism-replay"
    provenance = runtime_provenance(REPOSITORY)
    # Replay the complete load+initialize sequence. Dataset loading may
    # consume RNG, so recreating only the model would be a false comparison.
    initial_a_digest = _initialization_digest(config, args.seed)
    initial_b_digest = _initialization_digest(config, args.seed)
    initialization_equal = initial_a_digest == initial_b_digest

    trainer = Trainer(config)
    write_json(output_dir / "provenance.json", provenance)

    # Reset RNG so the fixed camera is independent of the initialization
    # replay above.
    seed_everything(args.seed)
    batch = trainer.sampler.next()
    rays = torch.cat(
        (batch.rays.origins, batch.rays.directions), dim=-1
    ).contiguous()

    replay_rows = []
    reference_image = None
    reference_gradients = None
    reference_masks = None
    reference_plan = None
    threshold = float(args.threshold)

    for replay_index in range(args.replays):
        trainer.gaussian_optimizer.zero_grad(set_to_none=True)
        trainer.medium_optimizer.zero_grad(set_to_none=True)
        trainer.gaussian_adaptation._reset_statistics()

        gaussian_scene, screen_points = trainer.model.gaussians.training_scene(
            batch.camera
        )
        trainer.renderer.set_gaussian_geometry(gaussian_scene)
        render = trainer.renderer.render(
            rays,
            gaussians=gaussian_scene,
            voronoi=trainer.model.medium.render_scene(),
            background=tuple(config.background),
            collect_diagnostics=True,
            diagnostic_target=batch.target,
        )
        loss = b0_reconstruction_loss(
            render.image,
            batch.target,
            mode="mse",
            image_size=(batch.camera.height, batch.camera.width),
        )
        loss.total.backward()
        torch.cuda.synchronize()
        if screen_points.grad is None or render.diagnostics is None:
            raise RuntimeError("audit did not receive adaptation gradients")

        gradients = _gradient_tensors(trainer, screen_points.grad)
        magnitude = screen_points.grad.detach().norm(dim=-1)
        masks = {
            "threshold": magnitude >= threshold,
            "within_1pct": (magnitude - threshold).abs() <= 0.01 * threshold,
            "within_5pct": (magnitude - threshold).abs() <= 0.05 * threshold,
            "within_10pct": (magnitude - threshold).abs() <= 0.10 * threshold,
        }
        trainer.gaussian_adaptation.collect_statistics(
            screen_points.grad,
            render.diagnostics.gaussian,
            batch.camera,
        )
        plan = trainer.gaussian_adaptation.propose_plan(1_000)
        plan_digests = {
            "clone": _indices_digest(plan.clone_indices),
            "split": _indices_digest(plan.split_indices),
            "keep": _indices_digest(plan.keep_indices),
        }

        row: dict[str, object] = {
            "replay": replay_index,
            "camera": batch.camera_name,
            "loss": float(loss.total.item()),
            "image_sha256": tensor_digest(render.image),
            "gradient_sha256": tensors_digest(gradients),
            "candidate_sha256": tensor_digest(masks["threshold"]),
            "candidate_count": int(masks["threshold"].sum().item()),
            "within_1pct": int(masks["within_1pct"].sum().item()),
            "within_5pct": int(masks["within_5pct"].sum().item()),
            "within_10pct": int(masks["within_10pct"].sum().item()),
            "clone_count": plan.clone_count,
            "split_count": plan.split_count,
            "prune_count": plan.prune_count,
            "plan_sha256": plan_digests,
        }
        if reference_image is None:
            reference_image = render.image.detach().cpu()
            reference_gradients = {
                name: value.detach().cpu() for name, value in gradients.items()
            }
            reference_masks = {
                name: value.detach().cpu() for name, value in masks.items()
            }
            reference_plan = plan_digests
            row.update(
                {
                    "image_max_abs_vs_0": 0.0,
                    "gradient_max_abs_vs_0": 0.0,
                    "gradient_different_elements_vs_0": 0,
                    "candidate_xor_vs_0": 0,
                    "plan_equal_vs_0": True,
                }
            )
        else:
            image_difference = tensor_difference(reference_image, render.image)
            gradient_differences = {
                name: tensor_difference(reference_gradients[name], value)
                for name, value in gradients.items()
            }
            row.update(
                {
                    "image_max_abs_vs_0": image_difference["max_abs"],
                    "gradient_max_abs_vs_0": max(
                        float(value["max_abs"])
                        for value in gradient_differences.values()
                    ),
                    "gradient_different_elements_vs_0": sum(
                        int(value["different_elements"])
                        for value in gradient_differences.values()
                    ),
                    "candidate_xor_vs_0": int(
                        torch.logical_xor(
                            reference_masks["threshold"], masks["threshold"].cpu()
                        ).sum().item()
                    ),
                    "plan_equal_vs_0": plan_digests == reference_plan,
                }
            )
        replay_rows.append(row)

    summary = {
        "initialization": {
            "first_sha256": initial_a_digest,
            "second_sha256": initial_b_digest,
            "equal": initialization_equal,
        },
        "checkpoint_rng_schema": {
            "torch_saved": True,
            "cuda_saved": True,
            "python_random_saved": False,
            "numpy_saved": False,
            "camera_stack_saved": False,
        },
        "threshold": threshold,
        "replays": replay_rows,
        "provenance": provenance,
    }
    write_json(output_dir / "audit.json", summary)

    if args.wandb_mode != "disabled":
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            mode=args.wandb_mode,
            dir=str(output_dir),
            tags=["determinism-audit", "replay", platform_node(provenance)],
            config={
                "seed": args.seed,
                "threshold": threshold,
                "replays": args.replays,
                "provenance": provenance,
            },
        )
        for row in replay_rows:
            payload = {
                key: value
                for key, value in row.items()
                if isinstance(value, (int, float, bool))
            }
            run.log(payload, step=int(row["replay"]))
        run.summary["initialization_equal"] = initialization_equal
        run.summary["forward_exact"] = all(
            float(row["image_max_abs_vs_0"]) == 0.0 for row in replay_rows
        )
        run.summary["candidate_masks_equal"] = all(
            int(row["candidate_xor_vs_0"]) == 0 for row in replay_rows
        )
        run.summary["topology_plans_equal"] = all(
            bool(row["plan_equal_vs_0"]) for row in replay_rows
        )
        run.save(str(output_dir / "audit.json"), base_path=str(output_dir))
        run.finish()

    trainer.logger.close()
    print(summary)
    return 0


def platform_node(provenance: dict[str, object]) -> str:
    return f"node-{provenance.get('hostname', 'unknown')}"


if __name__ == "__main__":
    raise SystemExit(main())
