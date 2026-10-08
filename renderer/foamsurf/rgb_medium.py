"""Six-pass CUDA composition; OptiX hits/starts are shared across all passes.

Correctness-first backend: reuses native geometry derivatives. Not a fused kernel.
"""
import torch


def compose_cuda(apply, arguments, attenuation, backscatter):
    direct, scattered, transmission, counts = [], [], [], []
    for channel in range(3):
        args = list(arguments)
        args[2] = attenuation[:, channel:channel+1].contiguous()
        args[3] = torch.zeros_like(arguments[3])
        rgba, n = apply(*args)
        direct.append(rgba[:, channel])
        transmission.append(1 - rgba[:, 3])
        counts.append(n)
        args[2] = backscatter[:, channel:channel+1].contiguous()
        args[3] = arguments[3]
        args[13] = torch.zeros_like(arguments[13])
        args[16] = torch.zeros_like(arguments[16])
        rgba, n = apply(*args)
        scattered.append(rgba[:, channel])
        counts.append(n)
    direct = torch.stack(direct, -1)
    scattered = torch.stack(scattered, -1)
    transmission = torch.stack(transmission, -1)
    # Alpha is mean direct opacity, NOT conserved RGB responsibility.
    rgba = torch.cat((direct + scattered, 1-transmission.mean(-1, keepdim=True)), -1)
    count = torch.stack([n.to(torch.int64) for n in counts]).amax(0).to(counts[0].dtype)
    return rgba, count, (direct, scattered, transmission)


def reference_segments(lengths, attenuation, backscatter, colors, alpha, surface_colors, background):
    """Independent autograd recurrence; segment i is followed by Gaussian i.

    alpha=0 denotes no Gaussian at that endpoint. Last endpoint is finite far.
    """
    td = torch.ones_like(background)
    tb = torch.ones_like(background)
    tg = lengths.new_ones(())
    direct = torch.zeros_like(background)
    scattered = torch.zeros_like(background)
    for i in range(len(lengths)):
        scattered = scattered + tg*tb*colors[i]*(-torch.expm1(-backscatter[i]*lengths[i]))
        td = td * torch.exp(-attenuation[i]*lengths[i])
        tb = tb * torch.exp(-backscatter[i]*lengths[i])
        direct = direct + tg*td*alpha[i]*surface_colors[i]
        tg = tg*(1-alpha[i])
    direct = direct + tg*td*background
    return direct+scattered, direct, scattered, tg*td
