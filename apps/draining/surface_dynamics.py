"""Linear shallow-water height/velocity solver on a staggered grid.
No-flux rock boundaries reflect waves; a driven sponge supplies incoming waves.
This is 2.5D surface dynamics, without overturning crests or volumetric splashes.
"""
import math
import torch
import torch.nn.functional as F


class ShallowWater:
    def __init__(self, terrain, reliable, lower, extent, scale):
        self.terrain=terrain
        self.reliable=reliable
        self.lower,self.extent,self.scale=lower,extent,scale
        self.ny,self.nx=terrain.shape
        self.dx=float(extent[0])/(self.nx-1)
        self.dy=float(extent[1])/(self.ny-1)
        self.gravity=1.962*scale
        self.h=torch.zeros_like(terrain)
        self.u=terrain.new_zeros(self.ny,self.nx+1)
        self.v=terrain.new_zeros(self.ny+1,self.nx)
        self.wet=torch.ones_like(terrain,dtype=torch.bool)
        self.depth=torch.full_like(terrain,.2*scale)
        self.time=None
        y,x=torch.meshgrid(torch.arange(self.ny,device=terrain.device),torch.arange(self.nx,device=terrain.device),indexing='ij')
        edge=torch.minimum(torch.minimum(x,self.nx-1-x),torch.minimum(y,self.ny-1-y)).float()
        self.sponge=(1-edge/12).clamp(0,1).square()
        self.domain_blend=(edge/6).clamp(0,1)
        self.xy=torch.stack((lower[0]+x*self.dx,lower[1]+y*self.dy),-1).float()
        self.grid=None
        self.steps=0

    def boundaries(self,level):
        solid=self.reliable&(self.terrain>=level)
        # Seal isolated holes in the reconstruction at the collision boundary.
        solid=F.max_pool2d(solid.float()[None,None],3,1,1)[0,0]>.5
        self.wet=~solid
        self.depth=torch.where(self.reliable,(level-self.terrain).clamp(.04*self.scale,.3*self.scale),torch.full_like(self.terrain,.2*self.scale))
        self.open_x=self.wet[:,:-1]&self.wet[:,1:]
        self.open_y=self.wet[:-1]&self.wet[1:]
        self.face_x=.5*(self.depth[:,:-1]+self.depth[:,1:])
        self.face_y=.5*(self.depth[:-1]+self.depth[1:])
        self.h*=self.wet

    def step(self,dt,target=None):
        self.u[:,1:-1]=(self.u[:,1:-1]-self.gravity*dt*(self.h[:,1:]-self.h[:,:-1])/self.dx)*self.open_x
        self.v[1:-1]=(self.v[1:-1]-self.gravity*dt*(self.h[1:]-self.h[:-1])/self.dy)*self.open_y
        flux_x=torch.zeros_like(self.u)
        flux_y=torch.zeros_like(self.v)
        flux_x[:,1:-1]=self.u[:,1:-1]*self.face_x
        flux_y[1:-1]=self.v[1:-1]*self.face_y
        self.h-=dt*((flux_x[:,1:]-flux_x[:,:-1])/self.dx+(flux_y[1:]-flux_y[:-1])/self.dy)
        if target is not None:
            relax=1-torch.exp(-self.sponge*dt*8)
            self.h=self.h*(1-relax)+target*relax
            self.u[:,1:-1]*=torch.exp(-dt*4*.5*(self.sponge[:,:-1]+self.sponge[:,1:]))
            self.v[1:-1]*=torch.exp(-dt*4*.5*(self.sponge[:-1]+self.sponge[1:]))
        self.h*=self.wet
        self.steps+=1

    def update(self,level,time,ocean):
        self.boundaries(level)
        def incoming(t):
            ocean.update(t)
            return ocean.sample(self.xy[...,0].reshape(-1),self.xy[...,1].reshape(-1))[0].reshape(self.terrain.shape)
        if self.time is None:
            self.h=incoming(time)*self.wet
        else:
            interval=time-self.time
            if interval<0:
                raise ValueError('Surface simulation requires monotone frame times')
            stable=.35*min(self.dx,self.dy)/math.sqrt(self.gravity*.3*self.scale)
            substeps=max(1,math.ceil(interval/stable))
            for i in range(substeps):
                target=incoming(self.time+interval*(i+1)/substeps)
                # Keep forcing in the small-disturbance regime of the linear model.
                limit=.25*self.depth
                target=limit*torch.tanh(target/limit)
                self.step(interval/substeps,target)
        ocean.update(time)
        self.time=time
        gy,gx=torch.gradient(self.h,spacing=(self.dy,self.dx))
        self.grid=torch.stack((self.h,gx,gy,self.domain_blend))[None]
        if not bool(torch.isfinite(self.h).all()):
            raise RuntimeError("Non-finite surface dynamics state")
        self.amplitude=float(self.h.abs().max())

    def sample(self,x,y,fallback):
        coordinates=torch.stack(((x-self.lower[0])/self.extent[0]*2-1,(y-self.lower[1])/self.extent[1]*2-1),-1)[None,None]
        values=F.grid_sample(self.grid,coordinates,align_corners=True,padding_mode='zeros')[0,:,0]
        weight=values[3]
        return tuple(values[i]*weight+fallback[i]*(1-weight) for i in range(3))
