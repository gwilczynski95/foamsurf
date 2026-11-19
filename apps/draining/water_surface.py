"""One dielectric bounce against a procedural height field, traced in the learned scene.
World-space waves, exact unpolarized Fresnel, Snell refraction and total internal reflection.
The environment is procedural; the scene was captured underwater and does not contain a sky.
"""
import math
import torch
import torch.nn.functional as F


def dielectric(direction, normal, inside, ior=1.333):
    n = torch.where(inside[:,None], -normal, normal)
    cosine = -(direction*n).sum(-1).clamp(-1,1)
    eta = torch.where(inside, torch.full_like(cosine,ior), torch.full_like(cosine,1/ior))
    sin2 = eta.square()*(1-cosine.square()).clamp_min(0)
    transmitted_cos = (1-sin2).clamp_min(0).sqrt()
    reflected = F.normalize(direction+2*cosine[:,None]*n,dim=-1)
    refracted = F.normalize(eta[:,None]*direction+(eta*cosine-transmitted_cos)[:,None]*n,dim=-1)
    rs = ((eta*cosine-transmitted_cos)/(eta*cosine+transmitted_cos).clamp_min(1e-8)).square()
    rp = ((cosine-eta*transmitted_cos)/(cosine+eta*transmitted_cos).clamp_min(1e-8)).square()
    fresnel = torch.where(sin2 >= 1, torch.ones_like(cosine), (rs+rp)/2).clamp(0,1)
    return reflected,refracted,fresnel


def final_wave_multiplier(time, duration, maximum=1.):
    x=min(1.,max(0.,(time/duration-.7)/.25))
    return 1+(maximum-1)*x*x*(3-2*x)


class WaterSurface:
    def __init__(self, trainer, gaussians, medium, rays, up, scale, depth):
        from scipy.spatial import cKDTree
        self.t,self.g,self.m,self.rays,self.up = trainer,gaussians,medium,rays,up
        self.r = trainer.renderer
        axis = torch.eye(3,device=up.device)[up.abs().argmin()]
        self.u = F.normalize(torch.linalg.cross(up,axis),dim=0)
        self.v = torch.linalg.cross(up,self.u)
        self.scale = scale
        self.depth = depth.reshape(-1)
        self.tree = cKDTree(medium.sites.detach().cpu().numpy())
        self.time = 0.
        from apps.draining.ocean_waves import OceanWaves
        self.ocean = OceanWaves(scale,up.device)

    def waves(self, points):
        self.ocean.update(self.time)
        x,y=points@self.u,points@self.v
        h,dx,dy = self.ocean.sample(x,y)
        if getattr(self,"dynamics",None) is not None and self.dynamics.grid is not None:
            h,dx,dy=self.dynamics.sample(x,y,(h,dx,dy))
        multiplier=getattr(self,"wave_multiplier",1.)
        h,dx,dy=h*multiplier,dx*multiplier,dy*multiplier
        normal = F.normalize(self.up-dx[:,None]*self.u-dy[:,None]*self.v,dim=-1)
        return h,normal

    def intersection(self,rays,level):
        o,d = rays[:,:3],rays[:,3:]
        slope = d@self.up
        safe = torch.where(slope.abs()>1e-7,slope,torch.full_like(slope,1e-7))
        # The sum of wave amplitudes bounds every root. Bisection avoids Newton
        # failures and speckled misses at grazing angles.
        if hasattr(self,"ocean"):
            self.ocean.update(self.time)
        amplitude = self.ocean.amplitude if hasattr(self,"ocean") else .00285*self.scale
        if getattr(self,"dynamics",None) is not None:
            amplitude=max(amplitude,self.dynamics.amplitude)+1e-6
        amplitude*=getattr(self,"wave_multiplier",1.)
        a = (level-o@self.up-amplitude)/safe
        b = (level-o@self.up+amplitude)/safe
        lo = torch.minimum(a,b).clamp(0,self.r.far)
        hi = torch.maximum(a,b).clamp(0,self.r.far)
        def residual(t):
            point = o+t[:,None]*d
            return point@self.up-level-self.waves(point)[0]
        flo,fhi = residual(lo),residual(hi)
        bracket = (flo*fhi<=0)&(hi>lo)&(slope.abs()>1e-7)
        for _ in range(20):
            mid = (lo+hi)/2
            fm = residual(mid)
            left = flo*fm<=0
            hi = torch.where(left,mid,hi)
            lo = torch.where(left,lo,mid)
            flo = torch.where(left,flo,fm)
        distance = (lo+hi)/2
        point = o+distance[:,None]*d
        _,normal = self.waves(point)
        valid = bracket&(distance>1e-4*self.scale)&(distance<self.r.far)
        inside = o@self.up < level+self.waves(o)[0]
        return distance,point,normal,valid,inside

    def intervals(self,rays,level):
        distance,_,_,valid,inside = self.intersection(rays,level)
        start = torch.where(inside,torch.zeros_like(distance),torch.where(valid,distance,torch.full_like(distance,self.r.far)))
        end = torch.where(inside&valid,distance,torch.full_like(distance,self.r.far))
        return torch.stack((start,end),-1).clamp(0,self.r.far).contiguous()

    def sky(self,directions):
        elevation = (directions@self.up).clamp(0,1)
        horizon = directions.new_tensor([.56,.66,.73])
        zenith = directions.new_tensor([.18,.37,.64])
        sky = horizon+(zenith-horizon)*elevation[:,None].sqrt()
        sun = F.normalize(.45*self.u+.25*self.v+.85*self.up,dim=0)
        disc = ((directions@sun).clamp_min(0)**1800)[:,None]
        return sky+disc*directions.new_tensor([3.,2.7,2.2])

    def trace(self,rays,level,gaussians=True,intervals=None,starts=None):
        self.r.water_intervals = self.intervals(rays,level) if intervals is None else intervals
        if starts is None and not torch.equal(rays[:,:3],self.rays[:len(rays),:3]):
            indices = self.tree.query(rays[:,:3].detach().cpu().numpy(),workers=8)[1]
            starts = torch.as_tensor(indices,device=rays.device,dtype=torch.uint32)
        out = self.r.render(rays,gaussians=self.g if gaussians else None,voronoi=self.m,
                            background=(0.,0.,0.),start_points=starts,surface=False,check_inputs=False)
        return out

    def radiance(self,out,rays,environment=True):
        image = out.image
        light = self.t.evaluator.water_light
        if light:
            image = light(image,out.backscatter,rays)
        if environment:
            image = image+out.direct_transmittance*self.sky(rays[:,3:])
        return image

    def draw(self,level,time):
        self.time = time
        self.wave_multiplier=final_wave_multiplier(time,getattr(self,"duration",12.),getattr(self,"final_wave_scale",1.))
        if getattr(self,"breaking",None) is not None:
            from apps.draining.reef_interaction import late_wave_strength
            self.ocean.strength=1+(late_wave_strength(time,self.duration)-1)*getattr(self,"impact_strength",1.)
            if getattr(self,"dynamics",None) is not None:
                self.dynamics.update(level,time,self.ocean)
            self.breaking.update(level,time)
        rays = self.rays
        distance,point,normal,valid,inside = self.intersection(rays,level)
        base = self.trace(rays,level)
        visibility = torch.ones(len(rays),device=rays.device)
        # Respect partial Gaussian coverage in front of the water, rather than
        # treating every faint first hit as an opaque occluder.
        for slot in range(base.hit_indices.shape[0]):
            ids = base.hit_indices[slot].long()
            depths = base.hit_depths[slot]
            before = (ids>=0)&(depths>0)&(depths<distance)
            ids = ids.clamp_min(0)
            q = F.normalize(self.g.quaternions[ids],dim=-1)
            w,x,y,z = q.unbind(-1)
            ax = torch.stack((1-2*(y*y+z*z),2*(x*y+w*z),2*(x*z-w*y)),-1)
            ay = torch.stack((2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)),-1)
            delta = rays[:,:3]+depths[:,None]*rays[:,3:]-self.g.means[ids]
            scale = self.g.log_scales[ids].exp()
            radius = ((delta*ax).sum(-1)/scale[:,0]).square()+((delta*ay).sum(-1)/scale[:,1]).square()
            kappa = 1+F.softplus(self.g.raw_kappa[ids].reshape(-1))
            response = torch.exp(-.5*radius.clamp_min(1e-8).pow(kappa)/kappa)
            alpha = self.g.opacity_logits[ids].reshape(-1).sigmoid()*response
            visibility *= 1-torch.where(before,alpha,torch.zeros_like(alpha))
        valid &= visibility>.001
        image = self.radiance(base,rays)
        indices = valid.nonzero().flatten()
        if not len(indices):
            return self.breaking.spray(image) if getattr(self,"breaking",None) is not None else image
        incident = rays[indices,3:]
        reflected,refracted,fresnel = dielectric(incident,normal[indices],inside[indices])
        eps = self.scale*2e-4
        hit = point[indices]
        bounce_colors = []
        for directions in (reflected,refracted):
            secondary = torch.cat((hit+eps*directions,directions),-1).contiguous()
            out = self.trace(secondary,level)
            bounce_colors.append(self.radiance(out,secondary))
        interface = fresnel[:,None]*bounce_colors[0]+(1-fresnel[:,None])*bounce_colors[1]
        primary = rays[indices].contiguous()
        intervals = torch.stack((torch.zeros_like(distance[indices]),torch.where(inside[indices],distance[indices],torch.zeros_like(distance[indices]))),-1).contiguous()
        prefix = self.trace(primary,level,gaussians=False,intervals=intervals)
        water_image = self.radiance(prefix,primary,environment=False)+prefix.direct_transmittance*interface
        weight = visibility[indices,None]
        image[indices] = image[indices]*(1-weight)+water_image*weight
        if getattr(self,"breaking",None) is not None:
            image=self.breaking.shade(image,level,point,valid.float()*visibility)
            # Also retain foam on the wet reef after a crest has passed.
            terrain_points=rays[:,:3]+self.depth.clamp_min(0)[:,None]*rays[:,3:]
            near_water=(terrain_points@self.up-level).abs()<self.ocean.amplitude+self.scale*.025
            image=self.breaking.shade(image,level,terrain_points,(self.depth>0).float()*near_water.float()*(1-valid.float()))
            image=self.breaking.spray(image)
        return image
