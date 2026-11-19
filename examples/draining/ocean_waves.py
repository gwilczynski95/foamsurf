"""Independent spectral height-field implementation: Phillips wind spectrum,
finite-depth gravity dispersion and inverse FFT. See Blender ocean.cc for the
same model family; this is not a port of Blender's modifier or its choppiness.
Scene scale is artistic (the reconstruction has no calibrated metre scale).
"""
import math
import torch
import torch.nn.functional as F


class OceanWaves:
    def __init__(self, scale, device, resolution=256, seed=42):
        self.scale=scale
        self.length=8*scale
        self.n=resolution
        k=2*math.pi*torch.fft.fftfreq(resolution,d=self.length/resolution,device=device)
        ky,kx=torch.meshgrid(k,k,indexing='ij')
        magnitude=(kx*kx+ky*ky).sqrt().clamp_min(1e-8)
        alignment=(.94*kx+.342*ky)/magnitude
        spectrum=torch.exp(-1/(magnitude*.65*scale).square())*torch.exp(-(magnitude*.025*scale).square())*alignment.square()/magnitude.pow(4)
        spectrum[0,0]=0
        generator=torch.Generator(device=device).manual_seed(seed)
        self.h0=torch.complex(torch.randn((resolution,resolution),device=device,generator=generator),torch.randn((resolution,resolution),device=device,generator=generator))*spectrum.sqrt()
        reverse=(-torch.arange(resolution,device=device))%resolution
        self.mirror=self.h0[reverse][:,reverse].conj()
        self.omega=(1.962*scale*magnitude*torch.tanh(magnitude*scale)).sqrt()
        self.kx,self.ky=kx,ky
        initial=torch.fft.ifft2(self.h0+self.mirror).real
        self.gain=.0045*scale/initial.square().mean().sqrt().clamp_min(1e-12)
        self.strength=1.
        self.cached_time=None
        self.update(0.)

    def update(self,time):
        if self.cached_time==(time,self.strength):
            return
        phase=torch.exp(1j*self.omega*time)
        spectrum=(self.h0*phase+self.mirror*phase.conj())*self.gain*self.strength
        fields=torch.stack([torch.fft.ifft2(spectrum).real,torch.fft.ifft2(1j*self.kx*spectrum).real,torch.fft.ifft2(1j*self.ky*spectrum).real])
        self.swell_weight=min(1.,max(0.,(self.strength-1)/6))
        self.amplitude=float(fields[0].abs().max())+.24*self.scale*self.swell_weight+1e-6
        self.time=time
        self.grid=torch.cat((fields,fields[:,:,:1]),dim=2)
        self.grid=torch.cat((self.grid,self.grid[:,:1,:]),dim=1)[None]
        self.cached_time=(time,self.strength)

    def sample(self,x,y):
        coords=torch.stack(((x/self.length).remainder(1)*2-1,(y/self.length).remainder(1)*2-1),-1)[None,None]
        h,dx,dy=F.grid_sample(self.grid,coords,mode='bilinear',padding_mode='border',align_corners=True)[0,:,0].unbind(0)
        # Long incoming swell: larger crests, rather than magnifying short ripples.
        for angle,wavelength,amplitude,offset in [(.3,2.4,.18,0.),(.65,1.3,.06,1.7)]:
            k=2*math.pi/(wavelength*self.scale)
            omega=math.sqrt(1.962*self.scale*k)
            phase=k*(math.cos(angle)*x+math.sin(angle)*y)-omega*self.time+offset
            a=amplitude*self.scale*self.swell_weight
            h=h+a*phase.sin()
            dx=dx+a*k*math.cos(angle)*phase.cos()
            dy=dy+a*k*math.sin(angle)*phase.cos()
        return h,dx,dy
