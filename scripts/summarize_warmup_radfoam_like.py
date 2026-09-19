#!/usr/bin/env python3
"""CPU-only sweep inventory and diagnostic ranking (never an automatic handoff)."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--count',type=int,default=50);a=p.parse_args()
    root=Path(a.root);rows=[];missing=[]
    for i in range(a.count):
        run=root/f'run_{i}';path=run/'warmup_analysis/analysis.json'
        if not path.exists():missing.append(i);continue
        variant=json.loads((run/'sweep_variant.json').read_text())
        for r in json.loads(path.read_text()):
            rows.append({'id':i,'name':variant['name'],'iteration':r['iteration'],
                **r['metrics'], 'oracle_cloud_psnr':r['oracle_metrics']['medium_psnr_vs_cloud'],
                'cloud_contribution_lost_fraction':r['oracle_cloud_contribution_mislabeled_surface']/max(r['total_oracle_cloud_contribution'],1e-20)})
    if rows:
        with (root/'checkpoint_metrics.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    # No single scalar chooses a winner: low leakage with absent clouds is not success.
    text=['# Warmup sweep results','',f'Missing analyses: {missing}', '',
          'Compare full reconstruction, surface leakage, medium reconstruction and oracle gap jointly.',
          'Oracle layer metrics are benchmark diagnostics; they are not deployment selection inputs.', '',
          '| Run | Iteration | Full PSNR | Medium PSNR | Surface→cloud alpha | Cloud contribution lost |',
          '|---:|---:|---:|---:|---:|---:|']
    for r in sorted(rows,key=lambda x:(x['id'],x['iteration'])):
        text.append(f"| {r['id']} | {r['iteration']} | {r['full_psnr']:.3f} | {r['medium_psnr_vs_cloud']:.3f} | {r['surface_alpha_on_cloud_only']:.5f} | {r['cloud_contribution_lost_fraction']:.5f} |")
    (root/'SWEEP_REPORT.md').write_text('\n'.join(text)+'\n')
    print(f'{len(rows)} checkpoints; missing runs: {missing}')


if __name__=='__main__':main()
