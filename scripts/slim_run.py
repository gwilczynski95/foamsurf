#!/usr/bin/env python3
"""Shrink one finished run (about 14 GB -> 2 GB): train_seathru.sbatch calls it after the audit.

Keeps metrics, summary, configs, audit/, checkpoints best.pt and last.pt (+ sidecars) and the
evaluation renders of iterations 0, 5000, ..., the last and the best one. Removes the other
evaluation iterations, checkpoints/initialization.pt and the local W&B copy. --keep-until N also
keeps every evaluation up to iteration N (the start of training, frame by frame).

Usage: python scripts/slim_run.py --run outputs/seathru/<job>/<name> [--keep-until N] [--apply]   (default: dry run)
"""
import argparse
import csv
import os
import shutil
from pathlib import Path

EVERY = 5000


def kept_iterations(run, every=EVERY, keep_until=0):
    rows = list(csv.DictReader((run / "joint/metrics/eval.csv").open()))
    iterations = [int(float(r["iteration"])) for r in rows]
    keep = {i for i in iterations if i % every == 0 or i <= keep_until}
    if rows:
        keep.add(iterations[-1])
        keep.add(iterations[max(range(len(rows)), key=lambda k: float(rows[k]["eval/hybrid_psnr"]))])
    return keep


def removable(run, every=EVERY, keep_until=0):
    out = [p for p in (run / "joint/wandb", run / "warmup/wandb", run / "joint/checkpoints/initialization.pt")
           if p.exists()]
    evaluation = run / "joint/evaluation"
    if evaluation.is_dir() and (run / "joint/metrics/eval.csv").is_file():
        keep = kept_iterations(run, every, keep_until)
        out += sorted(p for p in evaluation.iterdir()
                      if p.name.startswith("iteration_") and int(p.name[-6:]) not in keep)
    return out


def size(path):
    if not path.is_dir():
        return path.lstat().st_size
    return sum(os.lstat(os.path.join(folder, name)).st_size
               for folder, _, files in os.walk(path) for name in files)


def remove(paths):
    for path in paths:
        shutil.rmtree(path) if path.is_dir() and not path.is_symlink() else path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="really delete (default: dry run)")
    parser.add_argument("--keep-until", type=int, default=0, help="also keep every evaluation up to this iteration")
    args = parser.parse_args()
    if not (args.run / "joint/summary.json").is_file():
        raise SystemExit(f"{args.run}: training not finished (no joint/summary.json), not slimming")
    paths = removable(args.run, keep_until=args.keep_until)
    freed = sum(size(p) for p in paths)
    if args.apply:
        remove(paths)
    print(f"{args.run}: {len(paths)} paths, {freed / 2**30:.1f} GiB {'freed' if args.apply else 'would be freed'}")


if __name__ == "__main__":
    main()
