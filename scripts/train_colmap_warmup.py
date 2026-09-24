#!/usr/bin/env python3
"""Native COLMAP warmup, independently inspectable before automatic handoff."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.end_to_end import EndToEndConfig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(ROOT/'configs/end_to_end_iui3.yaml'))
    p.add_argument('--output-dir', required=True)
    p.add_argument('--iterations', type=int, default=5000)
    p.add_argument('--resume')
    args = p.parse_args()
    c = EndToEndConfig.load(args.config)
    c.warmup.output_dir = args.output_dir
    c.warmup.optimization.iterations = args.iterations
    c.warmup.voronoi_warmup.milestone_iterations = sorted(set(
        [i for i in [0,1000,2000,3000,5000,7500,10000,11000,15000,20000] if i <= args.iterations]
        + [args.iterations]))
    c.warmup.validate()
    from training.warmup_trainer import WarmupTrainer
    trainer = WarmupTrainer(c.warmup, c.profile, resume=args.resume)
    if args.resume is None:
        trainer._checkpoint(Path(args.output_dir)/'checkpoints/initialization.pt', 0)
    trainer.fit()


if __name__ == '__main__':
    main()
