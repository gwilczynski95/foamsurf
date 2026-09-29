"""Reproducible, read-only analysis of RGB smoothing and three-loss sweeps.

Run with the foamsurf environment from anywhere. Writes only beside this file.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ASSETS = HERE / "figures"
SMOOTH = ROOT / "outputs/rgb_smooth/513211"
CONFLICT = ROOT / "outputs/rgb_disentangle/513646"
OLDER = ROOT / "outputs/rgb_medium/512923"
VIEWS = ("MTN_5894", "MTN_5903", "MTN_5911", "MTN_5928")
S_GROUPS = ("control", "color_moderate", "color_strong", "all_moderate", "all_strong")
C_GROUPS = ("control", "O", "G", "S", "OG", "OS", "GS", "OGS")
COLOR = {"control":"#606675","O":"#df8f22","G":"#c84242","S":"#488ac7",
         "OG":"#a85652","OS":"#478c76","GS":"#915ba4","OGS":"#865e36",
         "all_strong":"#22a786","color_strong":"#519bd3"}


def n(row, name):
    return float(row[name])


@dataclass
class Run:
    sweep: str
    index: int
    path: Path
    group: str
    strength: float
    seed: int
    eval: pd.DataFrame
    final: pd.Series
    best: pd.Series
    per_view: pd.DataFrame
    summary: dict

    @property
    def psnr(self): return n(self.final, "eval/hybrid_psnr")
    @property
    def direct(self): return n(self.final, "eval/medium_rgb/direct_mean")
    @property
    def back(self): return n(self.final, "eval/medium_rgb/backscatter_mean")
    @property
    def direct_fraction(self): return self.direct / max(self.direct + self.back, 1e-12)


def load_runs(base, sweep, expected):
    result = []
    for index in range(expected):
        path = base / f"run_{index}"
        assert (path / "joint_done.json").exists(), path
        manifest = json.loads((path / "pipeline_config.json").read_text())
        variant = manifest["variant"]
        group = variant.get("group", variant["name"].rsplit("_seed", 1)[0])
        strength = float(variant.get("strength", 0.))
        ev = pd.read_csv(path / "joint/metrics/eval.csv")
        assert int(ev.iloc[-1]["iteration"]) == 30000, path
        best = ev.loc[ev["eval/hybrid_psnr"].idxmax()]
        view = pd.read_csv(path / "joint/evaluation/iteration_030000/per_view.csv")
        assert set(view.view_name) == set(VIEWS)
        summary = json.loads((path / "joint/summary.json").read_text())
        result.append(Run(sweep, index, path, group, strength, int(variant["seed"]), ev,
                          ev.iloc[-1], best, view, summary))
    return result


def aggregate(rows, key):
    values = np.array([key(r) for r in rows], dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if len(values)>1 else 0.


def groups(runs):
    result={}
    for r in runs:result.setdefault((r.group,r.strength),[]).append(r)
    for rows in result.values():
        assert {r.seed for r in rows} == {42,43,44}
    return result


def savefig(fig, name):
    fig.savefig(ASSETS / name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_matrix(cg,sg):
    metrics=[("Hybryda PSNR [dB]",lambda r:r.psnr,"viridis",(14,26)),
             ("Gaussian-only PSNR [dB]",lambda r:n(r.final,"eval/gaussian_only_psnr"),"viridis",(4,18)),
             ("Voronoi-only PSNR [dB]",lambda r:n(r.final,"eval/voronoi_only_psnr"),"viridis",(10,21)),
             ("Gaussian-only alpha, średnia",lambda r:n(r.final,"eval/gaussian_alpha_mean"),"plasma",(0,1))]
    strengths=(.3,1.,3.)
    fig,axs=plt.subplots(2,2,figsize=(12,12),constrained_layout=True)
    for ax,(title,func,cmap,limits) in zip(axs.flat,metrics):
        matrix=np.array([[aggregate(cg[(g,s)],func)[0] for s in strengths] for g in C_GROUPS[1:]])
        im=ax.imshow(matrix,cmap=cmap,vmin=limits[0],vmax=limits[1],aspect='auto')
        ax.set_xticks(range(3),['×0,3','×1','×3']);ax.set_yticks(range(7),C_GROUPS[1:])
        ax.set_title(title)
        for (i,j),val in np.ndenumerate(matrix):
            ax.text(j,i,f'{val:.2f}',
                    ha='center',va='center',fontsize=10,color='white' if (val-limits[0])/(limits[1]-limits[0])<.45 else 'black')
        fig.colorbar(im,ax=ax,shrink=.75)
    savefig(fig,'01_conflict_matrix.png')

    fig,axs=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    x=np.arange(len(C_GROUPS));width=.24
    control=aggregate(cg[('control',0.)],lambda r:r.psnr)[0]
    prior=aggregate(sg[('all_strong',0.)],lambda r:r.psnr)[0]
    for j,s in enumerate(strengths):
        vals=[control]+[aggregate(cg[(g,s)],lambda r:r.psnr)[0] for g in C_GROUPS[1:]]
        errs=[aggregate(cg[('control',0.)],lambda r:r.psnr)[1]]+[aggregate(cg[(g,s)],lambda r:r.psnr)[1] for g in C_GROUPS[1:]]
        axs[0].bar(x+(j-1)*width,vals,width,yerr=errs,capsize=2,label=f'×{s:g}')
    axs[0].axhline(prior,color=COLOR['all_strong'],ls='--',label='poprzedni: mocne TV')
    axs[0].set_xticks(x,C_GROUPS);axs[0].set_ylabel('PSNR przy 30k [dB]')
    axs[0].set_ylim(10,27);axs[0].legend(fontsize=8)
    for g in C_GROUPS:
        s=0. if g=='control' else 1.
        for r in cg[(g,s)]:axs[1].scatter(n(r.final,'eval/gaussian_only_psnr'),r.psnr,c=COLOR[g],s=30,alpha=.65)
        mean=aggregate(cg[(g,s)],lambda r:n(r.final,'eval/gaussian_only_psnr'))[0]
        psnr=aggregate(cg[(g,s)],lambda r:r.psnr)[0]
        axs[1].text(mean,psnr,g,fontsize=9,weight='bold')
    axs[1].scatter([aggregate(sg[('all_strong',0.)],lambda r:n(r.final,'eval/gaussian_only_psnr'))[0]],
                   [prior],marker='*',s=250,c=COLOR['all_strong'],label='poprzedni: mocne TV')
    axs[1].set_xlabel('Gaussian-only PSNR [dB]');axs[1].set_ylabel('Hybryda PSNR [dB]')
    axs[1].legend(loc='lower right')
    savefig(fig,'02_quality_and_components.png')


def plot_timelines(cg,sg):
    chosen=[('control',0.),('O',1.),('G',1.),('S',1.),('OG',1.),('OS',1.),('GS',1.),('OGS',1.)]
    fig,axs=plt.subplots(2,2,figsize=(13,9),sharex=True,constrained_layout=True)
    quantities=[('eval/hybrid_psnr','Hybryda PSNR [dB]'),
                ('eval/gaussian_only_psnr','Gaussian-only PSNR [dB]'),
                ('eval/voronoi_only_psnr','Voronoi-only PSNR [dB]'),
                ('eval/gaussian_alpha_mean','Średnia opacity Gaussian-only')]
    for (field,label),ax in zip(quantities,axs.flat):
        for g,s in chosen:
            rows=cg[(g,s)];array=np.stack([r.eval[field].to_numpy() for r in rows]);steps=rows[0].eval.iteration
            ax.plot(steps,array.mean(0),color=COLOR[g],label=g,linewidth=1.6)
        rows=sg[('all_strong',0.)];array=np.stack([r.eval[field].to_numpy() for r in rows]);steps=rows[0].eval.iteration
        ax.plot(steps,array.mean(0),color=COLOR['all_strong'],label='poprzedni mocne TV',ls='--',linewidth=2)
        ax.set_ylabel(label);ax.grid(alpha=.2)
    axs[0,0].legend(ncol=3,fontsize=8)
    for ax in axs[1]:ax.set_xlabel('Iteracja')
    savefig(fig,'03_learning_curves.png')

    fig,axs=plt.subplots(1,2,figsize=(13,4),constrained_layout=True)
    for g,s in chosen:
        rows=cg[(g,s)];steps=rows[0].eval.iteration
        for ax,field in zip(axs,['model/gaussian_count','model/voronoi_count']):
            array=np.stack([r.eval[field].to_numpy() for r in rows])
            ax.plot(steps,array.mean(0)/1000,label=g,color=COLOR[g])
    for ax,title in zip(axs,['Gaussy','Voronoi']):
        ax.set_title(title);ax.set_ylabel('Liczba prymitywów [tys.]');ax.set_xlabel('Iteracja');ax.grid(alpha=.2)
    axs[0].legend(ncol=4,fontsize=8)
    savefig(fig,'04_population_growth.png')


def plot_per_view(cg,sg):
    chosen=[('control',0.),('O',1.),('G',1.),('S',1.),('OG',1.),('OS',1.),('GS',1.),('OGS',1.)]
    labels=[g for g,_ in chosen]+['TV mocne (poprz.)']
    data=[]
    for g,s in chosen:
        rows=cg[(g,s)];data.append([np.mean([float(r.per_view.loc[r.per_view.view_name==view,'hybrid_psnr'].iloc[0]) for r in rows]) for view in VIEWS])
    rows=sg[('all_strong',0.)]
    data.append([np.mean([float(r.per_view.loc[r.per_view.view_name==view,'hybrid_psnr'].iloc[0]) for r in rows]) for view in VIEWS])
    fig,ax=plt.subplots(figsize=(10,6),constrained_layout=True)
    mat=np.array(data);im=ax.imshow(mat,cmap='viridis',vmin=10,vmax=29,aspect='auto')
    ax.set_yticks(np.arange(len(labels)),labels);ax.set_xticks(np.arange(4),VIEWS)
    for (i,j),v in np.ndenumerate(mat):ax.text(j,i,f'{v:.1f}',ha='center',va='center',color='white' if v<19 else 'black')
    ax.set_title('PSNR dla każdej kamery: średnia z 3 seedów, 30k')
    fig.colorbar(im,ax=ax,label='dB')
    savefig(fig,'05_per_view_psnr.png')


def train_diagnostics(cg,sg):
    targets=[('control',0.),('O',1.),('G',1.),('S',1.),('OG',1.),('OS',1.),('GS',1.),('OGS',1.)]
    cols=['iteration','train/psnr','conflict/overlap_raw','conflict/opacity_raw',
          'conflict/mask_fraction','conflict/near_mass','conflict/gaussian_confidence',
          'conflict/backscatter','rgb_smooth/attenuation','rgb_smooth/backscatter','rgb_smooth/color']
    fig,axs=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    shown=['train/psnr','conflict/overlap_raw','conflict/mask_fraction',
           'conflict/near_mass','conflict/gaussian_confidence','rgb_smooth/backscatter']
    for g,s in targets:
        run=cg[(g,s)][0]
        d=pd.read_csv(run.path/'joint/metrics/train.csv',usecols=cols)
        d=d[d.iteration%250==0]
        for ax,col in zip(axs.flat,shown):
            if col in d:ax.plot(d.iteration,d[col],color=COLOR[g],label=g,lw=1.2)
    prior=sg[('all_strong',0.)][0]
    d=pd.read_csv(prior.path/'joint/metrics/train.csv',usecols=['iteration','train/psnr','rgb_smooth/backscatter'])
    d=d[d.iteration%250==0]
    for ax,col in ((axs[0,0],'train/psnr'),(axs[1,2],'rgb_smooth/backscatter')):
        ax.plot(d.iteration,d[col],color=COLOR['all_strong'],ls='--',label='TV mocne (poprz.)',lw=1.8)
    for ax,col in zip(axs.flat,shown):
        ax.set_title(col);ax.set_xlabel('Iteracja');ax.grid(alpha=.2)
    axs[0,0].legend(ncol=3,fontsize=8)
    savefig(fig,'06_training_diagnostics.png')


def image(run,view,kind):
    idx=VIEWS.index(view)
    folder=run.path/'joint/evaluation/iteration_030000'
    prefix=f'{idx:04d}_{view}'
    if kind=='target':
        panel=Image.open(folder/f'{prefix}_panel.png').convert('RGB')
        width=panel.width//6
        return panel.crop((0,0,width,panel.height))
    return Image.open(folder/f'{prefix}_{kind}.png').convert('RGB')


def render_sheet(rows,views,out,title):
    kinds=('target','hybrid','gaussian_only','voronoi_only','direct_with_background','backscatter')
    w,h=340,236
    canvas=Image.new('RGB',(len(kinds)*w,(len(rows)*len(views))*h+58),(248,248,248))
    draw=ImageDraw.Draw(canvas)
    draw.text((12,12),title,fill=(10,10,10))
    for ri,(label,run) in enumerate(rows):
        for vi,view in enumerate(views):
            row=ri*len(views)+vi;y0=58+row*h
            for ci,kind in enumerate(kinds):
                pic=image(run,view,kind)
                pic.thumbnail((w,h-31),Image.Resampling.LANCZOS)
                canvas.paste(pic,(ci*w+(w-pic.width)//2,y0+28))
                tag=f'{label} · {view} | '+{'target':'GT','hybrid':'HYBRID','gaussian_only':'GAUSSY ONLY','voronoi_only':'VORONOI ONLY','direct_with_background':'DIRECT (+tło)','backscatter':'BACKSCATTER'}[kind]
                draw.text((ci*w+5,y0+7),tag,fill=(15,15,15))
    canvas.save(ASSETS/out,optimize=True)


def make_critical_zoom(run):
    """Exact saved branch renders, with a bottom-region crop for legibility."""
    view='MTN_5911'
    kinds=('target','hybrid','gaussian_only','voronoi_only')
    names=('GT','HYBRYDA','GAUSSY ONLY — DZIURY','VORONOI ONLY')
    w,h=520,350
    canvas=Image.new('RGB',(4*w,2*(h+42)+48),(248,248,248))
    draw=ImageDraw.Draw(canvas)
    draw.text((12,12),'Run 13 TV mocne · MTN_5911 · iteracja 30000 · pełen kadr i powiększenie dna',fill=(10,10,10))
    for col,(kind,name) in enumerate(zip(kinds,names)):
        src=image(run,view,kind)
        for row in (0,1):
            pic=src if row==0 else src.crop((int(.10*src.width),int(.42*src.height),int(.90*src.width),src.height))
            pic=pic.copy();pic.thumbnail((w,h),Image.Resampling.LANCZOS)
            x=col*w+(w-pic.width)//2;y=48+row*(h+42)
            canvas.paste(pic,(x,y+32))
            draw.text((col*w+8,y+8),f'{name} · '+('CAŁY KADR' if row==0 else 'DNO × POWIĘKSZENIE'),fill=(20,20,20))
    canvas.save(ASSETS/'00_run13_floor_failure.png',optimize=True)


def make_galleries(cg,sg):
    # One seed throughout galleries keeps the camera and initialization paired.
    control=cg[('control',0.)][0]
    key=[('Kontrola',control),('TV mocne (poprzedni)',sg[('all_strong',0.)][0]),
         ('Overlap ×0,3',cg[('O',.3)][0]),('Opacity ×1',cg[('G',1.)][0]),
         ('Overlap + TV ×3',cg[('OS',3.)][0]),('Wszystkie ×1',cg[('OGS',1.)][0])]
    for view in VIEWS:
        render_sheet(key,[view],f'07_examples_{view}.png',f'{view} · identyczny seed 42 · iteracja 30000')
    previous=[(f'{g}',sg[(g,0.)][0]) for g in S_GROUPS]
    render_sheet(previous,['MTN_5903','MTN_5911'],'08_smoothing_sweep.png',
                 'Poprzedni sweep wygładzania · seed 42 · iteracja 30000')
    for g in C_GROUPS[1:]:
        rows=[(f'{g} ×{s:g}  PSNR={cg[(g,s)][0].psnr:.1f}',cg[(g,s)][0]) for s in (.3,1.,3.)]
        render_sheet(rows,['MTN_5903','MTN_5911'],f'09_{g}_strengths.png',
                     f'Loss {g} · seed 42 · iteracja 30000 · GT / HYBRID / GAUSSY ONLY / VORONOI ONLY / direct / backscatter')


def texture_metrics(cg,sg):
    # PNG based, display-only diagnostic, NOT a semantic label or exact renderer stat.
    from scipy.ndimage import gaussian_filter
    values=[]
    for run in [r for rs in list(cg.values())+list(sg.values()) for r in rs]:
        for view in ('MTN_5903','MTN_5911'):
            energies=[]
            for kind in ('direct_with_background','backscatter'):
                im=image(run,view,kind).resize((512,340),Image.Resampling.BILINEAR)
                rgb=np.asarray(im,dtype=np.float32)/255
                y=np.dot(rgb,np.array([.2126,.7152,.0722],dtype=np.float32))
                area=y[int(.6*len(y)):]
                energies.append(float(np.abs(area-gaussian_filter(area,2)).mean()))
            values.append(dict(sweep=run.sweep,id=run.index,group=run.group,strength=run.strength,
                seed=run.seed,view=view,direct_HF=energies[0],back_HF=energies[1],
                back_HF_fraction=energies[1]/max(sum(energies),1e-12)))
    frame=pd.DataFrame(values);frame.to_csv(HERE/'png_texture_proxy.csv',index=False)
    groups_for_plot=[('control',0.),('O',1.),('G',1.),('S',1.),('OG',1.),('OS',1.),('GS',1.),('OGS',1.)]
    fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True)
    measurements=[]
    for g,s in groups_for_plot:
        rows=frame[(frame.sweep=='conflict')&(frame.group==g)&(frame.strength==s)]
        measurements.append(rows.back_HF_fraction.mean())
    prev=frame[(frame.sweep=='smooth')&(frame.group=='all_strong')]
    measurements.append(prev.back_HF_fraction.mean())
    ax.bar(range(len(measurements)),measurements,color=[COLOR[g] for g,_ in groups_for_plot]+[COLOR['all_strong']])
    ax.set_xticks(range(len(measurements)),[g for g,_ in groups_for_plot]+['TV mocne\n(poprzedni)'])
    ax.set_ylabel('Udział wysokich częstotliwości w backscatter')
    ax.set_ylim(0,1);ax.set_title('Dolne 40% obrazu, dwa widoki, średnia 3 seedów; diagnostyka PNG')
    savefig(fig,'10_texture_proxy.png')
    return frame


def summaries(cg,sg):
    rows=[]
    for family,data in (('smooth',sg),('conflict',cg)):
        for (g,s),rs in data.items():
            entry=dict(sweep=family,group=g,strength=s,ids=','.join(str(r.index) for r in rs))
            specs={'final_psnr':lambda r:r.psnr,'best_psnr':lambda r:n(r.best,'eval/hybrid_psnr'),
                   'final_ssim':lambda r:n(r.final,'eval/hybrid_ssim'),
                   'gaussian_only_psnr':lambda r:n(r.final,'eval/gaussian_only_psnr'),
                   'voronoi_only_psnr':lambda r:n(r.final,'eval/voronoi_only_psnr'),
                   'proxy_gaussian_weight':lambda r:n(r.final,'eval/proxy_responsibility/gaussian_weight_mean'),
                   'proxy_medium_weight':lambda r:n(r.final,'eval/proxy_responsibility/medium_weight_mean'),
                   'direct':lambda r:r.direct,'backscatter':lambda r:r.back,
                   'direct_share':lambda r:r.direct_fraction,
                   'gaussian_alpha':lambda r:n(r.final,'eval/gaussian_alpha_mean'),
                   'gaussian_count':lambda r:n(r.final,'model/gaussian_count'),
                   'voronoi_count':lambda r:n(r.final,'model/voronoi_count'),
                   'train_seconds':lambda r:r.summary['training_seconds']}
            for name,key in specs.items():entry[name],entry[name+'_sd']=aggregate(rs,key)
            rows.append(entry)
    pd.DataFrame(rows).to_csv(HERE/'variant_summary.csv',index=False)
    runrows=[]
    for row in [r for data in (sg,cg) for rs in data.values() for r in rs]:
        runrows.append(dict(sweep=row.sweep,id=row.index,group=row.group,strength=row.strength,
            seed=row.seed,best_psnr=n(row.best,'eval/hybrid_psnr'),best_iteration=int(row.best.iteration),
            final_psnr=row.psnr,final_ssim=n(row.final,'eval/hybrid_ssim'),
            gaussian_only_psnr=n(row.final,'eval/gaussian_only_psnr'),
            voronoi_only_psnr=n(row.final,'eval/voronoi_only_psnr'),
            proxy_gaussian_weight=n(row.final,'eval/proxy_responsibility/gaussian_weight_mean'),
            proxy_medium_weight=n(row.final,'eval/proxy_responsibility/medium_weight_mean'),
            direct=row.direct,backscatter=row.back,direct_share=row.direct_fraction,
            gaussian_alpha=n(row.final,'eval/gaussian_alpha_mean'),
            gaussians=int(row.final['model/gaussian_count']),voronoi=int(row.final['model/voronoi_count'])))
    pd.DataFrame(runrows).to_csv(HERE/'run_summary.csv',index=False)
    return pd.DataFrame(rows)


def main():
    ASSETS.mkdir(exist_ok=True,parents=True)
    smooth=load_runs(SMOOTH,'smooth',15)
    conflict=load_runs(CONFLICT,'conflict',66)
    assert len(smooth)==15 and len(conflict)==66
    sg=groups(smooth);cg=groups(conflict)
    summary=summaries(cg,sg)
    plot_matrix(cg,sg);plot_timelines(cg,sg);plot_per_view(cg,sg)
    train_diagnostics(cg,sg);make_galleries(cg,sg);make_critical_zoom(sg[('all_strong',0.)][1]);texture_metrics(cg,sg)
    print(summary[['sweep','group','strength','final_psnr','gaussian_only_psnr','voronoi_only_psnr','gaussian_alpha']].to_string(index=False))
    print('ARTIFACTS',HERE)


if __name__=='__main__':main()
