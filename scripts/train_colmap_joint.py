#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.colmap_joint import fit, run_stage, resolve
import json

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/colmap_joint_stage_a.yaml')
    p.add_argument('--variant',type=int,default=0)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--stage',choices=['warmup','joint'])
    a=p.parse_args()
    if a.stage: run_stage(a.output_dir,a.stage)
    elif a.dry_run: print(json.dumps(resolve(a.config,a.variant,a.output_dir),indent=2))
    else: fit(a.config,a.variant,a.output_dir,a.resume)

if __name__=='__main__': main()
