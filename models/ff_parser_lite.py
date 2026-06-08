import math

import torch
from torch import nn


class SharedRadialBandAttention(nn.Module):
    def __init__(self, num_bands=6, residual_gamma=0.25, eps=1e-6):
        super().__init__()
        if num_bands <= 0:
            raise ValueError(f'num_bands must be positive, got {num_bands}')

        self.num_bands = num_bands
        self.residual_gamma = residual_gamma
        self.eps = eps
        self.band_logits = nn.Parameter(torch.zeros(num_bands))
        self._mask_cache = {}

    @torch.no_grad()
    def _build_masks(self, h, w_rfft, device, dtype):
        key = (h, w_rfft, device.type, device.index, dtype)
        if key in self._mask_cache:
            return self._mask_cache[key]

        full_w = (w_rfft - 1) * 2
        fy = torch.fft.fftfreq(h, d=1.0, device=device)
        fx = torch.fft.rfftfreq(full_w, d=1.0, device=device)
        radius = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
        radius = radius / (radius.max() + self.eps)

        edges = torch.linspace(0.0, 1.0, self.num_bands + 1, device=device)
        masks = []
        for band_idx in range(self.num_bands):
            lower, upper = edges[band_idx], edges[band_idx + 1]
            if band_idx == self.num_bands - 1:
                mask = (radius >= lower) & (radius <= upper)
            else:
                mask = (radius >= lower) & (radius < upper)
            masks.append(mask.to(dtype=dtype))

        masks = torch.stack(masks, dim=0)
        self._mask_cache[key] = masks
        return masks

    def forward(self, x_fft):
        if not torch.is_complex(x_fft):
            raise TypeError('SharedRadialBandAttention expects a complex FFT tensor')

        _, _, h, w_rfft = x_fft.shape
        real_dtype = x_fft.real.dtype
        masks = self._build_masks(h, w_rfft, x_fft.device, real_dtype)
        gains = 1.0 + self.residual_gamma * torch.tanh(self.band_logits).to(real_dtype)
        radial_attention = torch.einsum('k,khw->hw', gains, masks)
        return x_fft * radial_attention[None, None, :, :]


class LiteChannelGate(nn.Module):
    def __init__(self, dim, reduction=8, min_hidden=4):
        super().__init__()
        if dim <= 0:
            raise ValueError(f'dim must be positive, got {dim}')

        hidden = max(dim // reduction, min_hidden)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(dim, hidden, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, dim, kernel_size=1, bias=True)

    def forward(self, x):
        gate = self.pool(x)
        gate = self.fc2(self.act(self.fc1(gate)))
        return torch.sigmoid(gate)


class FFParserLite(nn.Module):
    def __init__(
        self,
        dim,
        num_bands=6,
        reduction=8,
        min_hidden=4,
        fft_norm='ortho',
        residual_gamma=0.25,
        force_fp32_fft=True,
        shared_radial=None,
        post_fuse='none',
        eps=1e-6,
    ):
        super().__init__()
        if post_fuse not in ('none', 'dw_pw'):
            raise ValueError(f'Unsupported post_fuse: {post_fuse}')

        self.dim = dim
        self.fft_norm = fft_norm
        self.force_fp32_fft = force_fp32_fft
        self.radial = shared_radial or SharedRadialBandAttention(
            num_bands=num_bands,
            residual_gamma=residual_gamma,
            eps=eps,
        )
        self.chan_gate = LiteChannelGate(dim, reduction=reduction, min_hidden=min_hidden)
        self.beta = nn.Parameter(torch.zeros(1))

        if post_fuse == 'dw_pw':
            self.post = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            )
        else:
            self.post = nn.Identity()

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f'FFParserLite expects BCHW input, got {tuple(x.shape)}')
        if x.shape[1] != self.dim:
            raise ValueError(f'FFParserLite expects {self.dim} channels, got {x.shape[1]}')
        if not torch.is_floating_point(x):
            raise TypeError('FFParserLite only accepts floating point tensors')

        in_dtype = x.dtype
        fft_input = x.float() if self.force_fp32_fft else x

        autocast_enabled = fft_input.is_cuda
        with torch.cuda.amp.autocast(enabled=False):
            x_fft = torch.fft.rfft2(
                fft_input.float() if autocast_enabled else fft_input,
                dim=(-2, -1),
                norm=self.fft_norm,
            )
            x_fft = self.radial(x_fft)
            x_spec = torch.fft.irfft2(
                x_fft,
                s=x.shape[-2:],
                dim=(-2, -1),
                norm=self.fft_norm,
            )

        x_spec = self.post(x_spec.to(dtype=in_dtype))
        gate = self.chan_gate(x)
        return x + self.beta * gate * (x_spec - x)


def count_ff_parser_lite(module, inputs, output):
    x = inputs[0]
    if x.ndim != 4:
        return

    batch, channels, height, width = x.shape
    spatial = height * width
    fft_cost = 5.0 * batch * channels * spatial * math.log2(max(spatial, 2))
    gate_cost = 0
    if isinstance(module.chan_gate.fc1, nn.Conv2d) and isinstance(module.chan_gate.fc2, nn.Conv2d):
        hidden = module.chan_gate.fc1.out_channels
        gate_cost = batch * channels * hidden + batch * hidden * channels

    post_cost = 0
    if isinstance(module.post, nn.Sequential):
        post_cost = batch * spatial * channels * 9
        post_cost += batch * spatial * channels * channels

    module.total_ops += torch.DoubleTensor([fft_cost + gate_cost + post_cost])
