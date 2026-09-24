"""Make a compact cross-view contact sheet from the saved diagnostic arrays."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--renders',type=Path,required=True);a=p.parse_args()
    files=sorted(a.output.glob('*_data.npz'))
    fig,axs=plt.subplots(len(files),4,figsize=(17,3*len(files)),constrained_layout=True)
    for row,path in enumerate(files):
        name=path.stem.removesuffix('_data');data=np.load(path)
        render=next(a.renders.glob(f'*_{name}_hybrid.png'))
        axs[row,0].imshow(Image.open(render));axs[row,0].set_title(name+' | hybrid')
        for col,(key,limit) in enumerate([('intrinsic_source_concentration',1),('gaussian_opacity',1),('suspect_overlap',.25)],1):
            im=axs[row,col].imshow(data[key],cmap='inferno',vmin=0,vmax=limit,interpolation='nearest')
            axs[row,col].set_title(key);fig.colorbar(im,ax=axs[row,col],shrink=.65)
        for ax in axs[row]:ax.axis('off')
    fig.savefig(a.output/'overview.png',dpi=150);plt.close(fig)

if __name__=='__main__':main()
