# models/arl_blocks.py
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_gn(ch: int, groups: int = 4) -> nn.GroupNorm:
    """GroupNorm with valid group count."""
    g = min(groups, ch)
    while ch % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, ch)


def resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] != ref.shape[-2:]:
        x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
    return x


def _save_map_png(t: torch.Tensor, path: str) -> None:
    """Save a single-channel feature map tensor as a grayscale PNG (min-max normalized)."""
    import numpy as np
    from PIL import Image

    a = t.detach().float().cpu().numpy()
    if a.ndim == 3:
        a = a.squeeze(0)
    lo, hi = float(a.min()), float(a.max())
    if hi > lo:
        a = (a - lo) / (hi - lo)
    a = (a * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(a, mode="L").save(path)


class DSConv(nn.Module):
    """Depthwise separable conv."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, stride: int = 1, dilation: int = 1):
        super().__init__()
        padding = ((k - 1) // 2) * dilation
        self.dw = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=k,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_ch,
            bias=False,
        )
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.norm = make_gn(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ChannelGate(nn.Module):
    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        hidden = max(ch // reduction, 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, ch, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class ArtifactCue(nn.Module):
    """
    Laplacian + bright cue + dark cue + learned map -> artifact cue.
    """

    def __init__(self, in_ch: int = 3, cue_ch: int = 8):
        super().__init__()
        lap = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("lap_kernel", lap)

        self.learned = nn.Sequential(
            DSConv(in_ch, cue_ch, k=3, stride=1),
            DSConv(cue_ch, cue_ch, k=3, stride=1),
            nn.Conv2d(cue_ch, 1, kernel_size=1, bias=True),
        )
        self.mix = nn.Conv2d(4, 1, kernel_size=1, bias=True)

        # optional inference-time visualization; shares the same per-sample
        # directories as ARLUNet (see models/arl_unet.py), controlled via:
        #   ARL_VIS_DIR  -- directory to save per-sample PNGs into (disabled when unset)
        #   ARL_VIS_MAX  -- maximum number of samples to save (default 16)
        self._vis_dir = os.environ.get("ARL_VIS_DIR")
        self._vis_max = int(os.environ.get("ARL_VIS_MAX", "16"))
        self._vis_count = 0

    def _gray01(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)
        g_min = gray.amin(dim=(-2, -1), keepdim=True)
        g_max = gray.amax(dim=(-2, -1), keepdim=True)
        gray01 = (gray - g_min) / (g_max - g_min + 1e-6)
        return gray01

    def _save_cues(self, lap, bright, dark, learned):
        """Save per-sample PNGs of the four artifact cues into shared sample dirs."""
        if self._vis_dir is None or self._vis_count >= self._vis_max:
            return
        cues = {"lap": lap, "bright": bright, "dark": dark, "learned": learned}
        batch_size = lap.shape[0]
        for b in range(batch_size):
            if self._vis_count >= self._vis_max:
                return
            sample_dir = os.path.join(self._vis_dir, f"sample_{self._vis_count:04d}")
            os.makedirs(sample_dir, exist_ok=True)
            for name, t in cues.items():
                _save_map_png(t[b], os.path.join(sample_dir, f"cue_{name}.png"))
            self._vis_count += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self._gray01(x)
        lap = F.conv2d(g, self.lap_kernel, padding=1).abs()

        bright = torch.relu(g - 0.75)
        dark = torch.relu(0.25 - g)

        learned = torch.sigmoid(self.learned(x))
        if self._vis_dir is not None and not self.training:
            self._save_cues(lap, bright, dark, learned)
        cue = torch.cat([lap, bright, dark, learned], dim=1)
        cue = torch.sigmoid(self.mix(cue))
        return cue


class ArtifactAwareStem(nn.Module):
    """F' = F - alpha * (A * F), alpha zero-init."""

    def __init__(self, in_ch: int = 3, out_ch: int = 8):
        super().__init__()
        self.feat = DSConv(in_ch, out_ch, k=3, stride=1)
        self.cue = ArtifactCue(in_ch=in_ch, cue_ch=out_ch)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor):
        feat = self.feat(x)
        art = self.cue(x)
        art = resize_like(art, feat)
        feat = feat - self.alpha * feat * art
        return feat, art


class PlainStem(nn.Module):
    """Ablation: single DSConv stem; artifact map is zeros (skip fusion ignores it via (1-A))."""

    def __init__(self, in_ch: int = 3, out_ch: int = 8):
        super().__init__()
        self.feat = DSConv(in_ch, out_ch, k=3, stride=1)

    def forward(self, x: torch.Tensor):
        feat = self.feat(x)
        b, _, h, w = feat.shape
        art = feat.new_zeros(b, 1, h, w)
        return feat, art


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = DSConv(ch, ch, k=3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class UpProject(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            make_gn(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class RobustBlock(nn.Module):
    """depthwise 3x3 + depthwise 5x5 or dilated 3x3 + channel gate + residual."""

    def __init__(self, in_ch: int, out_ch: int, use_dilation: bool = False):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()
        self.conv1 = DSConv(in_ch, out_ch, k=3, stride=1)
        self.conv2 = DSConv(
            out_ch,
            out_ch,
            k=3 if use_dilation else 5,
            stride=1,
            dilation=2 if use_dilation else 1,
        )
        self.gate = ChannelGate(out_ch, reduction=8)
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.gate(y)
        return self.proj(x) + self.beta * y


class PlainBlock(nn.Module):
    """Ablation: single DSConv + residual projection (no second kernel / channel gate)."""

    def __init__(self, in_ch: int, out_ch: int, use_dilation: bool = False):
        super().__init__()
        _ = use_dilation  # kept for API parity with RobustBlock; not used
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()
        self.conv = DSConv(in_ch, out_ch, k=3, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x) + self.conv(x)


class MultiKernelContext(nn.Module):
    """Lightweight bottleneck context."""

    def __init__(self, ch: int):
        super().__init__()
        self.dw3 = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        self.dw5 = nn.Conv2d(ch, ch, 5, padding=2, groups=ch, bias=False)
        self.dwd = nn.Conv2d(ch, ch, 3, padding=2, dilation=2, groups=ch, bias=False)
        self.strip_h = nn.Conv2d(ch, ch, (1, 7), padding=(0, 3), groups=ch, bias=False)
        self.strip_v = nn.Conv2d(ch, ch, (7, 1), padding=(3, 0), groups=ch, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(ch * 5, ch, kernel_size=1, bias=False),
            make_gn(ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.cat(
            [
                self.dw3(x),
                self.dw5(x),
                self.dwd(x),
                self.strip_h(x),
                self.strip_v(x),
            ],
            dim=1,
        )
        return x + self.fuse(y)


class RobustSkipFusion(nn.Module):
    """
    Y = D + S + beta * (S * G * (1 - A)), beta zero-init.
    G = sigmoid(Conv([S, D, M, A])).
    """

    def __init__(self, skip_ch: int, dec_ch: int, out_ch: int, refine_cls=None):
        super().__init__()
        self.skip_proj = nn.Conv2d(skip_ch, out_ch, kernel_size=1, bias=False)
        self.dec_proj = nn.Conv2d(dec_ch, out_ch, kernel_size=1, bias=False)

        gate_ch = max(out_ch // 2, 8)
        self.gate = nn.Sequential(
            nn.Conv2d(out_ch * 2 + 2, gate_ch, kernel_size=3, padding=1, bias=False),
            make_gn(gate_ch),
            nn.GELU(),
            nn.Conv2d(gate_ch, out_ch, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        _refine = refine_cls if refine_cls is not None else RobustBlock
        self.refine = _refine(out_ch, out_ch)
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        dec_feat: torch.Tensor,
        skip_feat: torch.Tensor,
        coarse_mask: torch.Tensor,
        artifact_map: torch.Tensor,
    ):
        dec_feat = resize_like(dec_feat, skip_feat)
        coarse_mask = resize_like(coarse_mask, skip_feat)
        artifact_map = resize_like(artifact_map, skip_feat)

        s = self.skip_proj(skip_feat)
        d = self.dec_proj(dec_feat)

        g = self.gate(torch.cat([s, d, coarse_mask, artifact_map], dim=1))
        y = d + s + self.beta * (s * g * (1.0 - artifact_map))
        y = self.refine(y)
        return y, g


class SoftArtifactGuidedSkipFusion(nn.Module):
    """
    Soft artifact-guided skip fusion.

    Design goals (research-report2.md):
    - Artifact map is only a soft hint: suppression strength gamma is zero-init.
    - Avoid over-suppressing lesion boundaries: clamp gamma to max_art_suppress.

    Forward signature matches RobustSkipFusion for drop-in replacement.
    """

    def __init__(
        self,
        skip_ch: int,
        dec_ch: int,
        out_ch: int,
        refine_cls=None,
        max_art_suppress: float = 0.3,
    ):
        super().__init__()
        self.skip_proj = nn.Conv2d(skip_ch, out_ch, kernel_size=1, bias=False)
        self.dec_proj = nn.Conv2d(dec_ch, out_ch, kernel_size=1, bias=False)

        gate_ch = max(out_ch // 2, 8)
        # decoder-guided semantic gate (uses coarse mask as extra cue)
        self.semantic_gate = nn.Sequential(
            nn.Conv2d(out_ch * 2 + 1, gate_ch, kernel_size=3, padding=1, bias=False),
            make_gn(gate_ch),
            nn.GELU(),
            nn.Conv2d(gate_ch, out_ch, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        # artifact gate: 1-channel artifact cue -> channel-wise hint
        self.artifact_gate = nn.Sequential(
            nn.Conv2d(1, gate_ch, kernel_size=3, padding=1, bias=False),
            make_gn(gate_ch),
            nn.GELU(),
            nn.Conv2d(gate_ch, out_ch, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        _refine = refine_cls if refine_cls is not None else RobustBlock
        self.refine = _refine(out_ch, out_ch)

        self.gamma_art = nn.Parameter(torch.zeros(1))
        self.max_art_suppress = float(max_art_suppress)

    def forward(
        self,
        dec_feat: torch.Tensor,
        skip_feat: torch.Tensor,
        coarse_mask: torch.Tensor,
        artifact_map: torch.Tensor,
    ):
        dec_feat = resize_like(dec_feat, skip_feat)
        coarse_mask = resize_like(coarse_mask, skip_feat)
        artifact_map = resize_like(artifact_map, skip_feat)

        s = self.skip_proj(skip_feat)
        d = self.dec_proj(dec_feat)

        semantic_gate = self.semantic_gate(torch.cat([s, d, coarse_mask], dim=1))
        artifact_gate = self.artifact_gate(artifact_map)

        gamma = torch.clamp(self.gamma_art, 0.0, self.max_art_suppress)

        s_refined = s * (1.0 + semantic_gate)
        s_refined = s_refined * (1.0 - gamma * artifact_gate)

        y = self.refine(d + s_refined)
        # return semantic gate for visualization/compat with caller expecting (y, gate)
        return y, semantic_gate


class PlainSkipFusion(nn.Module):
    """
    Ablation: additive skip fusion D + S + refine; ignores mask / artifact gate.
    Same forward signature as RobustSkipFusion for drop-in use.
    """

    def __init__(self, skip_ch: int, dec_ch: int, out_ch: int, refine_cls=None):
        super().__init__()
        self.skip_proj = nn.Conv2d(skip_ch, out_ch, kernel_size=1, bias=False)
        self.dec_proj = nn.Conv2d(dec_ch, out_ch, kernel_size=1, bias=False)
        _refine = refine_cls if refine_cls is not None else RobustBlock
        self.refine = _refine(out_ch, out_ch)

    def forward(
        self,
        dec_feat: torch.Tensor,
        skip_feat: torch.Tensor,
        coarse_mask: torch.Tensor,
        artifact_map: torch.Tensor,
    ):
        _ = coarse_mask, artifact_map
        dec_feat = resize_like(dec_feat, skip_feat)
        s = self.skip_proj(skip_feat)
        d = self.dec_proj(dec_feat)
        y = self.refine(d + s)
        dummy_g = torch.ones_like(s[:, :1, :, :])
        return y, dummy_g
