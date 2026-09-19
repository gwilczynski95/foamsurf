#!/usr/bin/env python3
"""One command: native Voronoi warmup, surface initialization, hybrid training."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training.end_to_end import EndToEndConfig, EndToEndTrainer, run_stage


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(ROOT/'configs/end_to_end.yaml'))
    parser.add_argument('--output-dir')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stage',choices=['warmup','partition','joint'],help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.stage:
        if not args.output_dir: parser.error('--stage requires --output-dir')
        run_stage(args.output_dir,args.stage)
    else:
        config=EndToEndConfig.load(args.config,args.output_dir)
        print(json.dumps(EndToEndTrainer(config).fit(resume=args.resume),indent=2))


if __name__=='__main__': main()
