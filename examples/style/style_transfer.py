"""CLIPGaussian style objective, ported from FlaRe (../FlaRe/source/style_transfer.py, CLIPGaussianStyleLoss).

The loss is the FlaRe / CLIPGaussian one, term for term:

    lambda_dir     * (1 - cos(E(render) - E(base), E(style) - E(object prompt)))         directional CLIP, whole image
    lambda_patch   * mean over 64 random-perspective 128 px crops of the same term       patch CLIP
    lambda_content * sum over VGG19 conv4_2, conv5_2 of MSE(F(render), F(base))          content
    lambda_bg      * mean |render - background| over background pixels (optional)        silhouette

E = CLIP ViT-B/32 image embedding (unit norm); E(style) is the style image (or the text prompt with the ImageNet
templates); the object prompt "a Photo" with the same templates. The content reference and the CLIP source embedding
are the render of the model before stylization, as in FlaRe (not the dataset photos).

Differences to FlaRe:
- the directional term floors the norm of E(render) - E(base) at DIRECTION_DELTA (see forward): FlaRe's first render
  differs slightly from its reference render (training vs inference renderer settings), here they are identical, and
  the plain cosine of a zero step is 0/0;
- optional, off by default: ``patch_coverage`` > 0 draws the crops only where the object mask covers at least that
  fraction of the crop (the Gaussian-only render of an underwater scene is half black: crops there carry no gradient
  to the Gaussians).

Needs OpenAI CLIP and torchvision (scripts/install.sh --with-apps).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMAGENET_TEMPLATES = (
    'a bad photo of a {}.', 'a sculpture of a {}.', 'a photo of the hard to see {}.',
    'a low resolution photo of the {}.', 'a rendering of a {}.', 'graffiti of a {}.', 'a bad photo of the {}.',
    'a cropped photo of the {}.', 'a tattoo of a {}.', 'the embroidered {}.', 'a photo of a hard to see {}.',
    'a bright photo of a {}.', 'a photo of a clean {}.', 'a photo of a dirty {}.', 'a dark photo of the {}.',
    'a drawing of a {}.', 'a photo of my {}.', 'the plastic {}.', 'a photo of the cool {}.',
    'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a painting of the {}.',
    'a painting of a {}.', 'a pixelated photo of the {}.', 'a sculpture of the {}.', 'a bright photo of the {}.',
    'a cropped photo of a {}.', 'a plastic {}.', 'a photo of the dirty {}.', 'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a rendering of the {}.',
    'a {} in a video game.', 'a photo of one {}.', 'a doodle of a {}.', 'a close-up photo of the {}.',
    'a photo of a {}.', 'the origami {}.', 'the {} in a video game.', 'a sketch of a {}.', 'a doodle of the {}.',
    'a origami {}.', 'a low resolution photo of a {}.', 'the toy {}.', 'a rendition of the {}.',
    'a photo of the clean {}.', 'a photo of a large {}.', 'a rendition of a {}.', 'a photo of a nice {}.',
    'a photo of a weird {}.', 'a blurry photo of a {}.', 'a cartoon {}.', 'art of a {}.', 'a sketch of the {}.',
    'a embroidered {}.', 'a pixelated photo of a {}.', 'itap of the {}.', 'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.', 'a plushie {}.', 'a photo of the nice {}.', 'a photo of the small {}.',
    'a photo of the weird {}.', 'the cartoon {}.', 'art of the {}.', 'a drawing of the {}.',
    'a photo of the large {}.', 'a black and white photo of a {}.', 'the plushie {}.', 'a dark photo of a {}.',
    'itap of a {}.', 'graffiti of the {}.', 'a toy {}.', 'itap of my {}.', 'a photo of a cool {}.',
    'a photo of a small {}.', 'a tattoo of the {}.',
)


DIRECTION_DELTA = 0.05  # floor of |E(render) - E(base)| in the directional term (FlaRe's VIEW_CONS_DELTA)


@dataclass
class StyleLossConfig:
    style_image: str = ""
    style_prompt: str = ""
    object_prompt: str = "a Photo"
    clip_model: str = "ViT-B/32"
    lambda_dir: float = 5.0
    lambda_patch: float = 90.0
    lambda_content: float = 0.8
    lambda_bg: float = 0.0
    background: tuple = (0.0, 0.0, 0.0)
    crop_size: int = 128
    num_crops: int = 64
    patch_coverage: float = 0.0
    vgg_layers: dict = field(default_factory=lambda: {"21": "conv4_2", "31": "conv5_2"})


def load_image_tensor(path: str, device) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def imagenet_normalize(image: torch.Tensor) -> torch.Tensor:
    mean = image.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = image.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (image - mean) / std


def clip_normalize(image: torch.Tensor) -> torch.Tensor:
    image = F.interpolate(image, size=(224, 224), mode="bicubic", align_corners=False)
    mean = image.new_tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = image.new_tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    return (image - mean) / std


def normalize_features(features: torch.Tensor) -> torch.Tensor:
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)


class CLIPBackend:
    def __init__(self, model_name: str, device: torch.device):
        import clip
        self.clip = clip
        self.device = device
        self.model, _ = clip.load(model_name, device=device.type, jit=False)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return normalize_features(self.model.encode_image(clip_normalize(image)))

    def encode_text(self, texts) -> torch.Tensor:
        return normalize_features(self.model.encode_text(self.clip.tokenize(list(texts)).to(self.device)))


def perspective_params(size: int, n: int, distortion_scale: float = 0.5) -> torch.Tensor:
    """Corner end points of n torchvision RandomPerspective(distortion_scale) draws: (n, 4, 2)."""
    half = size // 2
    d = int(distortion_scale * half)
    low = lambda: torch.randint(0, d + 1, (n,))  # noqa: E731
    high = lambda: torch.randint(size - d - 1, size, (n,))  # noqa: E731
    return torch.stack([torch.stack([low(), low()], 1), torch.stack([high(), low()], 1),
                        torch.stack([high(), high()], 1), torch.stack([low(), high()], 1)], 1)


def perspective_grids(endpoints: torch.Tensor, size: int, device) -> torch.Tensor:
    """Sampling grids of torchvision.transforms.functional.perspective for a batch of end points (n, 4, 2)."""
    n = endpoints.shape[0]
    start = torch.tensor([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=torch.float64)
    p1 = endpoints.to(torch.float64)
    a = torch.zeros(n, 8, 8, dtype=torch.float64)
    a[:, 0::2, 0] = p1[..., 0]
    a[:, 0::2, 1] = p1[..., 1]
    a[:, 0::2, 2] = 1
    a[:, 1::2, 3] = p1[..., 0]
    a[:, 1::2, 4] = p1[..., 1]
    a[:, 1::2, 5] = 1
    a[:, 0::2, 6] = -start[:, 0] * p1[..., 0]
    a[:, 0::2, 7] = -start[:, 0] * p1[..., 1]
    a[:, 1::2, 6] = -start[:, 1] * p1[..., 0]
    a[:, 1::2, 7] = -start[:, 1] * p1[..., 1]
    coeffs = torch.linalg.solve(a, start.reshape(8).expand(n, 8)).to(torch.float32).to(device)
    theta1 = coeffs[:, :6].view(n, 2, 3)
    theta2 = torch.stack([coeffs[:, 6], coeffs[:, 7], torch.ones_like(coeffs[:, 6])], 1).view(n, 1, 3).expand(n, 2, 3)
    lin = torch.linspace(0.5, size - 0.5, steps=size, device=device)
    base = torch.stack([lin.view(1, -1).expand(size, size), lin.view(-1, 1).expand(size, size),
                        torch.ones(size, size, device=device)], -1).view(1, size * size, 3)
    scale = torch.tensor([0.5 * size, 0.5 * size], device=device)
    grid1 = base.expand(n, -1, -1).bmm(theta1.transpose(1, 2) / scale)
    grid2 = base.expand(n, -1, -1).bmm(theta2.transpose(1, 2))
    return (grid1 / grid2 - 1.0).view(n, size, size, 2)


def directional_loss(render_clip: torch.Tensor, source: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """1 - cos(E(render) - E(base), style direction) with a floor on the step norm (fp32): the render of the first
    step IS the base render, so the step is 0 and the plain cosine is 0/0 (NaN in fp16, ~1e15 gradients in fp32).
    For |step| >> DIRECTION_DELTA this is the FlaRe term. ``direction`` is unit length."""
    step = (render_clip - source).float()
    step = step / torch.sqrt((step ** 2).sum(-1, keepdim=True) + DIRECTION_DELTA ** 2)
    return (1.0 - (step * direction.float()).sum(-1)).mean()


def crop_positions(h: int, w: int, crop: int, n: int, mask: torch.Tensor | None, coverage: float):
    """Top-left corners of n crops: uniform (FlaRe), or among those whose mask coverage >= coverage."""
    if mask is None or coverage <= 0.0:
        return torch.randint(0, h - crop + 1, (n,)).tolist(), torch.randint(0, w - crop + 1, (n,)).tolist()
    candidates = 16 * n
    ys = torch.randint(0, h - crop + 1, (candidates,), device=mask.device)
    xs = torch.randint(0, w - crop + 1, (candidates,), device=mask.device)
    integral = F.pad(mask[0, 0].float().cumsum(0).cumsum(1), (1, 0, 1, 0))
    covered = (integral[ys + crop, xs + crop] - integral[ys, xs + crop] - integral[ys + crop, xs]
               + integral[ys, xs]) / float(crop * crop)
    good = torch.nonzero(covered >= coverage).flatten()
    if good.numel() >= n:
        pick = good[torch.randperm(good.numel(), device=mask.device)[:n]]
    else:  # not enough covered crops: the best-covered candidates
        pick = torch.argsort(covered, descending=True)[:n]
    return ys[pick].tolist(), xs[pick].tolist()


def batched_perspective_patches(image: torch.Tensor, crop: int, n: int, mask=None, coverage: float = 0.0,
                                out_size: int = 224) -> torch.Tensor:
    """n x [RandomCrop(crop) -> RandomPerspective(fill=0, p=1, distortion_scale=0.5) -> Resize(out_size)]."""
    h, w = image.shape[-2:]
    ys, xs = crop_positions(h, w, crop, n, mask, coverage)
    crops = torch.cat([image[..., y:y + crop, x:x + crop] for y, x in zip(ys, xs)], 0)
    grids = perspective_grids(perspective_params(crop, n), crop, image.device)
    sampled = F.grid_sample(torch.cat([crops, torch.ones_like(crops[:, :1])], 1), grids, mode="bilinear",
                            padding_mode="zeros", align_corners=False)
    warped = sampled[:, :-1] * sampled[:, -1:]
    return F.interpolate(warped, size=(out_size, out_size), mode="bilinear", align_corners=False, antialias=True)


class StyleLoss(torch.nn.Module):
    def __init__(self, config: StyleLossConfig, device: torch.device):
        super().__init__()
        from torchvision import models
        self.config = config
        self.device = device
        self.vgg = None
        if config.lambda_content > 0.0:
            self.vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.eval().to(device)
            for parameter in self.vgg.parameters():
                parameter.requires_grad_(False)
        self.clip_backend = CLIPBackend(config.clip_model, device)
        self.style_direction = self._style_direction()

    @torch.no_grad()
    def _style_direction(self) -> torch.Tensor:
        if self.config.style_image:
            style = self.clip_backend.encode_image(load_image_tensor(self.config.style_image, self.device))
        elif self.config.style_prompt:
            style = self.clip_backend.encode_text([t.format(self.config.style_prompt) for t in IMAGENET_TEMPLATES])
            style = normalize_features(style.mean(0, keepdim=True))
        else:
            raise ValueError("style transfer needs a style image or a style prompt")
        source = self.clip_backend.encode_text([t.format(self.config.object_prompt) for t in IMAGENET_TEMPLATES])
        source = normalize_features(source.mean(0, keepdim=True))
        return normalize_features(style - source).detach()

    def vgg_features(self, image: torch.Tensor) -> dict:
        features = {}
        if self.vgg is None:
            return features
        x = imagenet_normalize(image)
        last = max(int(k) for k in self.config.vgg_layers)
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.config.vgg_layers:
                features[self.config.vgg_layers[name]] = x.float()
            if int(name) >= last:
                break
        return features

    @torch.no_grad()
    def prepare_reference(self, image: torch.Tensor) -> dict:
        return {"vgg": self.vgg_features(image), "clip": self.clip_backend.encode_image(image).detach()}

    def background_loss(self, image: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if self.config.lambda_bg <= 0.0 or mask is None:
            return image.new_zeros(())
        background = image.new_tensor(self.config.background).view(1, 3, 1, 1)
        bg_mask = 1.0 - mask
        return ((image - background).abs() * bg_mask).sum() / bg_mask.sum().clamp_min(1.0)

    def forward(self, image: torch.Tensor, reference: dict, mask: torch.Tensor | None = None):
        """image [1,3,H,W] in [0,1] with gradients; reference from prepare_reference of the base render."""
        content = image.new_zeros(())
        for key, value in self.vgg_features(image).items():
            content = content + F.mse_loss(value, reference["vgg"][key].detach())
        background = self.background_loss(image, mask)

        source = reference["clip"].detach()
        render_clip = self.clip_backend.encode_image(image)
        directional = directional_loss(render_clip, source, self.style_direction)
        patches = batched_perspective_patches(image, self.config.crop_size, max(1, self.config.num_crops),
                                              mask, self.config.patch_coverage)
        patch_features = self.clip_backend.encode_image(patches)
        patch_direction = normalize_features(patch_features - source.repeat(patch_features.shape[0], 1))
        patch = (1.0 - F.cosine_similarity(patch_direction,
                                           self.style_direction.repeat(patch_features.shape[0], 1), dim=-1)).mean()
        total = (self.config.lambda_dir * directional + self.config.lambda_patch * patch
                 + self.config.lambda_content * content + self.config.lambda_bg * background)
        return total, {"total": float(total), "directional": float(directional), "patch": float(patch),
                       "content": float(content), "background": float(background)}
