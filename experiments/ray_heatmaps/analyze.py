"""Read-only CPU ray probes. No training modules patched, no CUDA required."""
from pathlib import Path
import sys, argparse, ctypes, subprocess, json
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.ndimage import uniform_filter1d
from scipy.special import expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from foamsurf_data import ColmapDataset,generate_camera_rays

def concentration(mass,dt):
    """Participation length: uniform -> 0, one occupied bin -> 1-1/N."""
    total=mass.sum(-1)
    leff=total**2/np.maximum((mass**2/dt).sum(-1),1e-30)
    return np.where(total>1e-12,np.clip(1-leff/(mass.shape[-1]*dt),0,1),0)

def metrics(a,b,color,gtau,dt,band):
    # Optical and Gaussian-only transmittance at midpoint of each bin.
    tg=np.exp(-(np.cumsum(gtau,-1)-.5*gtau))
    btau=b*dt;tb=np.exp(-(np.cumsum(btau,axis=1)-btau))
    q=(tg[:,:,None]*tb*color*(-np.expm1(-btau))).mean(-1)
    amass=a.mean(-1)*dt
    gm=np.exp(-(np.cumsum(gtau,-1)-gtau))*(-np.expm1(-gtau))
    gconf=gm.sum(-1)
    n=gtau.shape[1];depth=(np.arange(n)+.5)*dt
    gd=(gm*depth).sum(-1)/np.maximum(gconf,1e-20)
    width=np.sqrt((gm*(depth-gd[:,None])**2).sum(-1)/np.maximum(gconf,1e-20))
    confidence=gconf*np.exp(-(width/band)**2)
    near=np.abs(depth-gd[:,None])<=band
    total=q.sum(-1)
    overlap=(q*near).sum(-1)/np.maximum(total,1e-20)
    c=concentration(q,dt)
    strength=total/(total+.05)
    maps=dict(attenuation_concentration=concentration(amass,dt)*(-np.expm1(-amass.sum(-1))),
              backscatter_concentration=c*strength,
              gaussian_opacity=gconf,gaussian_surface_confidence=confidence,
              medium_near_surface=overlap,
              suspect_overlap=c*strength*overlap*confidence)
    # Intrinsic bB source: unlike q, this does not include visibility falloff.
    source=(b*color).mean(-1)*dt
    maps['intrinsic_source_concentration']=concentration(source,dt)*(source.sum(-1)/(source.sum(-1)+.05))
    return maps,dict(a=a.mean(-1),b=b.mean(-1),q=q/dt,g=gm/dt,t=depth,
                     gdepth=gd,gwidth=width,backscatter=total,gaussian_opacity=gconf)

def geometry(state):
    def v(k):return state['gaussians.'+k].numpy().astype(np.float64)
    q=v('quaternions');rot=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
    scales=np.exp(v('log_scales'))
    return np.ascontiguousarray(np.column_stack([v('means'),rot[:,:,0]/scales[:,0,None],
        rot[:,:,1]/scales[:,1,None],rot[:,:,2],expit(v('opacity_logits')),
        1+np.logaddexp(0,v('raw_kappa'))]))

def native(out):
    src=Path(__file__).with_name('gaussian.cpp');lib=out/'gaussian.so'
    subprocess.run(['g++','-O3','-shared','-fPIC','-fopenmp',str(src),'-o',str(lib)],check=True)
    f=ctypes.CDLL(str(lib)).trace
    arr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    f.argtypes=[ctypes.c_int]*3+[ctypes.c_double]+[arr]*4;f.restype=None
    return f

def probe(cam,state,tree,coeff,colors,geom,f,n,length):
    rb=generate_camera_rays(cam.origin,cam.right,cam.down,cam.forward,cam.width,cam.height,
                            cam.fov_x,cam.fov_y,cam.intrinsics)
    rays=np.ascontiguousarray(rb.directions.numpy().reshape(-1,3),dtype=np.float64)
    origins=np.ascontiguousarray(rb.origins.numpy().reshape(-1,3),dtype=np.float64)
    tau=np.zeros((len(rays),n));f(len(rays),len(geom),n,length,origins,rays,geom,tau)
    t=(np.arange(n)+.5)*length/n
    maps={};profiles=[]
    for start in range(0,len(rays),128):
        sl=slice(start,start+128)
        xyz=origins[sl,None,:]+rays[sl,None,:]*t[None,:,None]
        _,ids=tree.query(xyz.reshape(-1,3),workers=4)
        ids=ids.reshape(len(rays[sl]),n)
        m,p=metrics(coeff[ids,:3],coeff[ids,3:],colors[ids],tau[sl],length/n,.02*length)
        for k,v in m.items():maps.setdefault(k,[]).append(v)
        profiles.append(p)
    maps={k:np.concatenate(v).reshape(cam.height,cam.width) for k,v in maps.items()}
    profiles={k:(profiles[0][k] if k=='t' else np.concatenate([p[k] for p in profiles])) for k in profiles[0]}
    return maps,profiles

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--resolution',type=int,default=16)
    ap.add_argument('--samples',type=int,default=1024)
    ap.add_argument('--range-multiplier',type=float,default=8.)
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    ckpt=args.run/'joint/checkpoints/last.pt';p=torch.load(ckpt,map_location='cpu',weights_only=False)
    if p['model_metadata']['medium']['medium_type']!='rgb_backscatter':raise ValueError('RGB required')
    state=p['model'];cfg=p['config'];ds=ColmapDataset(cfg['dataset']['source_path'],args.resolution,'cpu',cfg['dataset']['llffhold'])
    center=torch.stack([c.origin for c in ds.train]).mean(0)
    extent=1.1*max(float(torch.linalg.vector_norm(c.origin-center)) for c in ds.train)
    length=min(float(cfg['renderer']['far']),args.range_multiplier*extent)
    coeff=np.logaddexp(0,state['medium.raw_density'].numpy())
    raw=state['medium.raw_color'].numpy();activation=p['model_metadata']['medium']['color_activation']
    colors=np.maximum(.5+.28209479177387814*raw,0) if activation=='dc' else expit(raw)
    geom=geometry(state);f=native(args.output);tree=cKDTree(state['medium.sites'].numpy())
    report=dict(checkpoint=str(ckpt.resolve()),iteration=p['iteration'],samples=args.samples,length=length,
                band_half_width=.02*length,resolution=args.resolution,views={})
    for vi,cam in enumerate(ds.test):
        print('probe',cam.name,cam.width,cam.height,flush=True)
        maps,prof=probe(cam,state,tree,coeff,colors,geom,f,args.samples,length)
        np.savez_compressed(args.output/f'{cam.name}_data.npz',**maps,**{'profile_'+k:v for k,v in prof.items()})
        fig,axs=plt.subplots(3,3,figsize=(16,12),constrained_layout=True)
        saved=args.run/f'joint/evaluation/iteration_{p["iteration"]:06d}'
        for ax,kind in zip(axs[0],['hybrid','backscatter','direct_with_background']):
            ax.imshow(Image.open(saved/f'{vi:04d}_{cam.name}_{kind}.png'));ax.set_title(kind);ax.axis('off')
        keys=['attenuation_concentration','backscatter_concentration','intrinsic_source_concentration',
              'gaussian_opacity','gaussian_surface_confidence','suspect_overlap']
        for ax,k in zip(axs[1:].flat,keys):
            vmax=.25 if k=='suspect_overlap' else 1
            im=ax.imshow(maps[k],cmap='inferno',vmin=0,vmax=vmax,interpolation='nearest');ax.set_title(k);ax.axis('off');fig.colorbar(im,ax=ax,shrink=.7)
        fig.suptitle(f'{cam.name} | CPU diagnostic approximation | 0..{length:.2f} scene units | {args.samples} samples')
        fig.savefig(args.output/f'{cam.name}_panel.png',dpi=140);plt.close(fig)
        for k,v in maps.items():plt.imsave(args.output/f'{cam.name}_{k}.png',v,cmap='inferno',vmin=0,vmax=1)
        # Deterministic maximum plus fixed probes: no manual cherry-picking.
        h,w=cam.height,cam.width
        points=[np.unravel_index(np.argmax(maps['suspect_overlap']), (h,w)),(h//4,w//2),(3*h//4,w//2),(h//2,w//4)]
        fig,axs=plt.subplots(4,2,figsize=(14,12),constrained_layout=True)
        for row,(y,x) in enumerate(points):
            j=y*w+x;tt=prof['t']
            ax=axs[row,0];ax.plot(tt,prof['a'][j],label='mean a');ax.plot(tt,prof['b'][j],label='mean b');ax.set_yscale('symlog',linthresh=.01)
            ax.set_title(f'probe {row}: pixel ({x},{y})');ax.legend()
            ax=axs[row,1];ax.plot(tt,prof['q'][j],label='visible backscatter / length');ax.plot(tt,prof['g'][j],label='Gaussian-only weight / length')
            ax.axvline(prof['gdepth'][j],color='gray',ls='--');ax.legend()
            for ax in axs[row]:ax.set_xlabel('ray depth [scene units]')
        fig.savefig(args.output/f'{cam.name}_profiles.png',dpi=140);plt.close(fig)
        fig,ax=plt.subplots();ax.imshow(Image.open(saved/f'{vi:04d}_{cam.name}_hybrid.png'))
        ow,oh=Image.open(saved/f'{vi:04d}_{cam.name}_hybrid.png').size
        for k,(y,x) in enumerate(points):ax.plot((x+.5)*ow/w,(y+.5)*oh/h,'ro');ax.text((x+.5)*ow/w,(y+.5)*oh/h,str(k),color='yellow')
        ax.axis('off');fig.savefig(args.output/f'{cam.name}_probe_locations.png',dpi=150);plt.close(fig)
        # Double depth resolution, every eighth ray in both image axes.
        rb=generate_camera_rays(cam.origin,cam.right,cam.down,cam.forward,w,h,cam.fov_x,cam.fov_y,cam.intrinsics)
        dirs=np.ascontiguousarray(rb.directions.numpy()[::8,::8].reshape(-1,3),dtype=np.float64)
        org=np.ascontiguousarray(np.broadcast_to(cam.origin.numpy(),dirs.shape),dtype=np.float64)
        n=2*args.samples;t=(np.arange(n)+.5)*length/n;gt=np.zeros((len(dirs),n));f(len(dirs),len(geom),n,length,org,dirs,geom,gt)
        _,ids=tree.query((org[:,None,:]+dirs[:,None,:]*t[None,:,None]).reshape(-1,3),workers=4);ids=ids.reshape(len(dirs),n)
        fine,_=metrics(coeff[ids,:3],coeff[ids,3:],colors[ids],gt,length/n,.02*length)
        report['views'][cam.name]={k:dict(mean=float(v.mean()),p95=float(np.quantile(v,.95)),max=float(v.max()),
            refinement_mae=float(np.abs(v[::8,::8].ravel()-fine[k]).mean()),refinement_max=float(np.abs(v[::8,::8].ravel()-fine[k]).max())) for k,v in maps.items()}
        print(cam.name,report['views'][cam.name]['suspect_overlap'],flush=True)
        (args.output/'report.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
