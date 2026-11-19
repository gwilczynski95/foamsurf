"""Visual reef contact proxy, persistent foam and ballistic spray (not a fluid solver)."""
import numpy as np
import torch
import torch.nn.functional as F


def late_wave_strength(time, duration):
    x = min(1., max(0., (time / duration - .55) / .4))
    return 1 + 2*x*x*(3-2*x)


class ReefInteraction:
    def __init__(self, water, camera):
        from scipy.spatial import cKDTree
        self.water,self.camera=water,camera
        self.device=water.up.device
        self.n=256
        self.strength=getattr(water,"impact_strength",1.)
        depth=water.depth
        valid=(depth>0)&torch.isfinite(depth)
        cloud=(water.rays[valid,:3]+water.rays[valid,3:]*depth[valid,None])[::4]
        uv=torch.stack((cloud@water.u,cloud@water.v),-1).cpu().numpy()
        heights=(cloud@water.up).cpu().numpy()
        lower,upper=np.quantile(uv,[.01,.99],axis=0)
        self.lower=torch.tensor(lower,device=self.device,dtype=torch.float32)
        self.extent=torch.tensor(upper-lower,device=self.device,dtype=torch.float32)
        x,y=np.meshgrid(np.linspace(lower[0],upper[0],self.n),np.linspace(lower[1],upper[1],self.n))
        query=np.stack((x,y),-1).reshape(-1,2)
        distance,index=cKDTree(uv).query(query,k=4)
        # Reject sparse holes and discontinuities in the reconstructed reef.
        weights=1/np.maximum(distance,1e-5)
        height=(heights[index]*weights).sum(-1)/weights.sum(-1)
        reliable=(distance[:,0]<water.scale*.035)&(np.ptp(heights[index],axis=1)<water.scale*.15)
        self.terrain=torch.tensor(height.reshape(self.n,self.n),device=self.device,dtype=torch.float32)
        terrain_smooth=F.avg_pool2d(self.terrain[None,None],5,1,2)[0,0]
        gy,gx=torch.gradient(terrain_smooth,spacing=(float(self.extent[1])/(self.n-1),float(self.extent[0])/(self.n-1)))
        self.rock_face=(gx.square()+gy.square()).sqrt()>.35
        self.reliable=torch.tensor(reliable.reshape(self.n,self.n),device=self.device)
        self.world=(torch.tensor(query,device=self.device,dtype=torch.float32)[:,0,None]*water.u+
                    torch.tensor(query,device=self.device,dtype=torch.float32)[:,1,None]*water.v)
        self.foam=torch.zeros((self.n,self.n),device=self.device)
        self.previous_height=None
        self.previous_time=None
        self.particles=torch.empty((0,3),device=self.device)
        self.velocity=torch.empty_like(self.particles)
        self.life=torch.empty(0,device=self.device)
        self.generator=torch.Generator(device=self.device).manual_seed(73)

    def sample(self,field,points):
        uv=torch.stack((points@self.water.u,points@self.water.v),-1)
        grid=((uv-self.lower)/self.extent*2-1)[None,None]
        return F.grid_sample(field[None,None],grid,align_corners=True,padding_mode='zeros')[0,0,0]

    def update(self,level,time):
        w=self.water
        dt=0 if self.previous_time is None else max(0,time-self.previous_time)
        height,normal=w.waves(self.world)
        height=height.reshape(self.n,self.n)+level
        gap=height-self.terrain
        slope=(1-(normal@w.up).reshape(self.n,self.n)).clamp_min(0).sqrt()
        rising=torch.zeros_like(height) if self.previous_height is None or dt==0 else ((height-self.previous_height)/dt).clamp_min(0)/w.scale
        contact=torch.exp(-(gap/(w.scale*.012)).square())*self.reliable
        energy=(rising*2+slope*.3).clamp(0,1)*contact*self.rock_face
        self.foam=(self.foam*np.exp(-dt/.65)+energy*dt*2).clamp(0,.25)
        # Small fragments of spray are emitted only at active contact locations.
        if dt>0:
            candidates=(energy.reshape(-1)>.03).nonzero().flatten()
            if len(candidates):
                ids=candidates[torch.randperm(len(candidates),generator=self.generator,device=self.device)[:32]]
                position=self.world[ids]+height.reshape(-1)[ids,None]*w.up
                noise=torch.randn((len(ids),3),generator=self.generator,device=self.device)
                velocity=w.up*(w.scale*.65)+noise*(w.scale*.16)
                self.particles=torch.cat((self.particles,position))[-1000:]
                self.velocity=torch.cat((self.velocity,velocity))[-1000:]
                self.life=torch.cat((self.life,torch.full((len(ids),),1.1,device=self.device)))[-1000:]
            self.velocity-=w.up*(1.962*w.scale*dt)
            self.particles+=self.velocity*dt
            self.life-=dt
            keep=self.life>0
            self.particles,self.velocity,self.life=self.particles[keep],self.velocity[keep],self.life[keep]
        self.previous_height,self.previous_time=height,time

    def shade(self,image,level,points,visible):
        w=self.water
        foam=self.sample(self.foam,points)
        # World-space bubbles break up the contact strip; no screen-space noise.
        uv=torch.stack((points@w.u,points@w.v),-1)/w.scale
        bubbles=(.65+.35*torch.sin(uv[:,0]*950)*torch.sin(uv[:,1]*837)).clamp(0,1)
        opacity=(foam*bubbles*visible).clamp(0,.25)[:,None]
        return image*(1-opacity)+image.new_tensor([.92,.97,.98])*opacity

    def spray(self,image):
        if not len(self.life):
            return image
        c=self.camera
        delta=self.particles-c.origin
        z=delta@c.forward
        fx,fy,cx,cy=c.intrinsics
        x=(fx*(delta@c.right)/z.clamp_min(1e-6)+cx).long()
        y=(fy*(delta@c.down)/z.clamp_min(1e-6)+cy).long()
        valid=(z>0)&(x>2)&(x<c.width-3)&(y>2)&(y<c.height-3)
        x,y,z=x[valid],y[valid],z[valid]
        if not len(x):
            return image
        pixels=y*c.width+x
        ray_depth=self.water.depth[pixels]
        dist=(self.particles[valid]-c.origin).norm(dim=-1)
        seen=(ray_depth<=0)|(dist<ray_depth)
        x,y=x[seen],y[seen]
        # Subtle antialiased two-pixel highlights rather than opaque white dots.
        for dx,dy,alpha in [(0,0,.85),(1,0,.45),(0,1,.45),(0,-1,.35),(0,-2,.15),(-1,0,.3)]:
            ids=(y+dy)*c.width+x+dx
            image[ids]=image[ids]*(1-alpha)+image.new_tensor([.8,.88,.9])*alpha
        return image
