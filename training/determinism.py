from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
from typing import Mapping

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def tensors_digest(tensors: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        digest.update(name.encode())
        digest.update(tensor_digest(tensors[name]).encode())
    return digest.hexdigest()


def tensor_difference(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, float | int]:
    reference = reference.detach().float().cpu()
    candidate = candidate.detach().float().cpu()
    difference = (candidate - reference).abs()
    scale = reference.abs().clamp_min(1.0e-12)
    return {
        "different_elements": int(torch.count_nonzero(difference).item()),
        "max_abs": float(difference.max().item()) if difference.numel() else 0.0,
        "mean_abs": float(difference.mean().item()) if difference.numel() else 0.0,
        "max_relative": (
            float((difference / scale).max().item())
            if difference.numel()
            else 0.0
        ),
    }


def _command_output(command: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(
            command,
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def runtime_provenance(repository: str | Path) -> dict[str, object]:
    repository = Path(repository).resolve()
    renderer_files = (
        repository / "renderer" / "foamsurf" / "_C.so",
        repository
        / "renderer"
        / "foamsurf"
        / "_optix"
        / "PYOPTIXFLARERENDERER.so",
    )
    renderer_hashes = {}
    for path in renderer_files:
        if path.is_file():
            renderer_hashes[str(path.relative_to(repository))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    dirty = _command_output(["git", "status", "--porcelain"], repository)
    result: dict[str, object] = {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git_commit": _command_output(["git", "rev-parse", "HEAD"], repository),
        "git_dirty": bool(dirty),
        "git_status_sha256": (
            hashlib.sha256((dirty or "").encode()).hexdigest()
        ),
        "renderer_sha256": renderer_hashes,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
    }
    if torch.cuda.is_available():
        result.update(
            {
                "gpu_name": torch.cuda.get_device_name(),
                "gpu_capability": list(torch.cuda.get_device_capability()),
                "gpu_count": torch.cuda.device_count(),
            }
        )
    return result


def write_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
