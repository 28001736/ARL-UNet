# models/arl_unet.py
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_

from models.ff_parser_lite import FFParserLite, SharedRadialBandAttention
from .arl_blocks import (
    ArtifactAwareStem,
    Downsample,
    MultiKernelContext,
    PlainBlock,
    PlainSkipFusion,
    PlainStem,
    RobustBlock,
    RobustSkipFusion,
    SoftArtifactGuidedSkipFusion,
    UpProject,
)


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


def _save_rgb_png(t: torch.Tensor, path: str) -> None:
    """Save a (C, H, W) image tensor as a PNG (min-max normalized).

    Saves RGB when C == 3; otherwise falls back to the first channel as grayscale.
    """
    import numpy as np
    from PIL import Image

    a = t.detach().float().cpu().numpy()
    if a.ndim != 3 or a.shape[0] != 3:
        _save_map_png(t[0] if a.ndim == 3 else t, path)
        return
    lo, hi = float(a.min()), float(a.max())
    if hi > lo:
        a = (a - lo) / (hi - lo)
    a = (a * 255.0).clip(0, 255).astype(np.uint8)
    a = np.transpose(a, (1, 2, 0))
    Image.fromarray(a, mode="RGB").save(path)


class AuxHead(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ARLUNet(nn.Module):
    """
    Compatible with training loss:
    return (gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1), out
    """

    def __init__(
        self,
        num_classes: int = 1,
        input_channels: int = 3,
        c_list=None,
        deep_supervision: bool = True,
        logger=None,
        ff_parser_cfg=None,
        arl_cfg=None,
    ):
        super().__init__()
        if c_list is None:
            c_list = [8, 16, 24, 36, 48, 72]
        c0, c1, c2, c3, c4, c5 = c_list
        self.deep_supervision = deep_supervision
        self.logger = logger
        self.ff_parser_cfg = ff_parser_cfg or {}
        arl_cfg = arl_cfg or {}
        use_artifact_stem = arl_cfg.get("use_artifact_stem", True)
        use_robust_encoder = arl_cfg.get("use_robust_encoder", True)
        use_skip_fusion = arl_cfg.get("use_skip_fusion", True)
        skip_fusion_mode = arl_cfg.get("skip_fusion_mode", "soft")

        StemCls = ArtifactAwareStem if use_artifact_stem else PlainStem
        BlockCls = RobustBlock if use_robust_encoder else PlainBlock
        if not use_skip_fusion:
            SkipCls = PlainSkipFusion
        else:
            if skip_fusion_mode == "hard":
                SkipCls = RobustSkipFusion
            elif skip_fusion_mode == "soft":
                SkipCls = SoftArtifactGuidedSkipFusion
            else:
                raise ValueError(f"Unknown skip_fusion_mode: {skip_fusion_mode}")
        RefineCls = RobustBlock if use_robust_encoder else PlainBlock

        if self.logger is not None:
            self.logger.info(
                "ARL-UNet ablation: "
                f"artifact_stem={use_artifact_stem}, "
                f"robust_encoder={use_robust_encoder}, "
                f"skip_fusion={use_skip_fusion}, "
                f"skip_fusion_mode={skip_fusion_mode}"
            )

        self.stem = StemCls(input_channels, c0)

        self.down1 = Downsample(c0)
        self.enc1 = BlockCls(c0, c0)

        self.down2 = Downsample(c0)
        self.enc2 = BlockCls(c0, c1)

        self.down3 = Downsample(c1)
        self.enc3 = BlockCls(c1, c2)

        self.down4 = Downsample(c2)
        self.enc4 = BlockCls(c2, c3)

        self.down5 = Downsample(c3)
        self.enc5 = BlockCls(c3, c4)

        self.bridge = nn.Sequential(
            BlockCls(c4, c5, use_dilation=True),
            MultiKernelContext(c5),
            nn.Conv2d(c5, c5, kernel_size=1, bias=False),
            nn.GroupNorm(4 if c5 % 4 == 0 else 1, c5),
            nn.GELU(),
        )

        self._build_ffl_adapters(c_list, self.ff_parser_cfg)

        self.fuse5 = SkipCls(skip_ch=c4, dec_ch=c5, out_ch=c4, refine_cls=RefineCls)
        self.up4 = UpProject(c4, c3)
        self.fuse4 = SkipCls(skip_ch=c3, dec_ch=c3, out_ch=c3, refine_cls=RefineCls)

        self.up3 = UpProject(c3, c2)
        self.fuse3 = SkipCls(skip_ch=c2, dec_ch=c2, out_ch=c2, refine_cls=RefineCls)

        self.up2 = UpProject(c2, c1)
        self.fuse2 = SkipCls(skip_ch=c1, dec_ch=c1, out_ch=c1, refine_cls=RefineCls)

        self.up1 = UpProject(c1, c0)
        self.fuse1 = SkipCls(skip_ch=c0, dec_ch=c0, out_ch=c0, refine_cls=RefineCls)

        self.aux5 = AuxHead(c5)
        self.aux4 = AuxHead(c4)
        self.aux3 = AuxHead(c3)
        self.aux2 = AuxHead(c2)
        self.aux1 = AuxHead(c1)

        self.out_head = nn.Conv2d(c0, num_classes, kernel_size=1)

        self.apply(self._init_weights)

        # optional inference-time visualization controlled via env vars:
        #   ARL_VIS_DIR  -- directory to save per-sample PNGs into (saving disabled when unset)
        #   ARL_VIS_MAX  -- maximum number of samples to save (default 16)
        self._vis_dir = os.environ.get("ARL_VIS_DIR")
        self._vis_max = int(os.environ.get("ARL_VIS_MAX", "16"))
        self._vis_count = 0
        if self._vis_dir is not None:
            os.makedirs(self._vis_dir, exist_ok=True)
            if self.logger is not None:
                self.logger.info(
                    f"ARL-UNet visualization enabled: dir={self._vis_dir}, max={self._vis_max}"
                )

    def _build_ffl_adapters(self, c_list, ff_parser_cfg):
        self.ff_t3 = nn.Identity()
        self.ff_t4 = nn.Identity()
        self.ff_t5 = nn.Identity()

        if not ff_parser_cfg or not ff_parser_cfg.get("enable", False):
            return

        scales = ff_parser_cfg.get("scales", ("s8", "s16", "s32"))
        if isinstance(scales, str):
            scales = [s.strip() for s in scales.split(",") if s.strip()]
        scales = set(scales)

        # EMA repo: s8/s16/s32; report: t3/t4/t5 (same spatial scales)
        use_t3 = "s8" in scales or "t3" in scales
        use_t4 = "s16" in scales or "t4" in scales
        use_t5 = "s32" in scales or "t5" in scales

        share_radial = ff_parser_cfg.get("share_radial", True)
        shared_radial = None
        if share_radial:
            shared_radial = SharedRadialBandAttention(
                num_bands=ff_parser_cfg.get("num_bands", 6),
                residual_gamma=ff_parser_cfg.get("residual_gamma", 0.25),
                eps=ff_parser_cfg.get("eps", 1e-6),
            )

        common_kwargs = {
            "num_bands": ff_parser_cfg.get("num_bands", 6),
            "reduction": ff_parser_cfg.get("reduction", 8),
            "min_hidden": ff_parser_cfg.get("min_hidden", 4),
            "fft_norm": ff_parser_cfg.get("fft_norm", "ortho"),
            "residual_gamma": ff_parser_cfg.get("residual_gamma", 0.25),
            "force_fp32_fft": ff_parser_cfg.get("force_fp32_fft", True),
            "post_fuse": ff_parser_cfg.get("post_fuse", "none"),
            "eps": ff_parser_cfg.get("eps", 1e-6),
        }

        if use_t3:
            self.ff_t3 = FFParserLite(c_list[2], shared_radial=shared_radial, **common_kwargs)
        if use_t4:
            self.ff_t4 = FFParserLite(c_list[3], shared_radial=shared_radial, **common_kwargs)
        if use_t5:
            self.ff_t5 = FFParserLite(c_list[4], shared_radial=shared_radial, **common_kwargs)

        if self.logger is not None:
            active = [n for n, u in (("t3", use_t3), ("t4", use_t4), ("t5", use_t5)) if u]
            self.logger.info(f"ARL-UNet: FF-Parser-Lite on scales: {active}")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _up_to_input(self, x: torch.Tensor, size_hw):
        return torch.sigmoid(F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False))

    def _save_intermediates(self, x, a0, m5, m4, m3, m2, m1, out_logits):
        """Save per-sample PNGs: input image, artifact map, 5 coarse masks, and the
        pre-upsample head output.

        ``out_logits`` is the raw output of ``self.out_head(d1)``; we apply sigmoid before
        saving so it shares the same [0,1] range as the other maps.
        """
        if self._vis_dir is None or self._vis_count >= self._vis_max:
            return

        maps = {
            "a0": a0,
            "m5": m5, "m4": m4, "m3": m3, "m2": m2, "m1": m1,
            "out": torch.sigmoid(out_logits),
        }
        batch_size = x.shape[0]
        for b in range(batch_size):
            if self._vis_count >= self._vis_max:
                return
            sample_dir = os.path.join(self._vis_dir, f"sample_{self._vis_count:04d}")
            os.makedirs(sample_dir, exist_ok=True)
            _save_rgb_png(x[b], os.path.join(sample_dir, "input.png"))
            for name, t in maps.items():
                _save_map_png(t[b], os.path.join(sample_dir, f"{name}.png"))
            self._vis_count += 1

    def forward(self, x: torch.Tensor):
        in_size = x.shape[-2:]

        f0, a0 = self.stem(x)
        t1 = self.enc1(self.down1(f0))
        t2 = self.enc2(self.down2(t1))
        t3 = self.enc3(self.down3(t2))
        t4 = self.enc4(self.down4(t3))
        t5 = self.enc5(self.down5(t4))

        t3 = self.ff_t3(t3)
        t4 = self.ff_t4(t4)
        t5 = self.ff_t5(t5)

        b = self.bridge(t5)

        m5 = torch.sigmoid(self.aux5(b))
        d5, _ = self.fuse5(b, t5, m5, a0)

        m4 = torch.sigmoid(self.aux4(d5))
        d4, _ = self.fuse4(self.up4(d5), t4, m4, a0)

        m3 = torch.sigmoid(self.aux3(d4))
        d3, _ = self.fuse3(self.up3(d4), t3, m3, a0)

        m2 = torch.sigmoid(self.aux2(d3))
        d2, _ = self.fuse2(self.up2(d3), t2, m2, a0)

        m1 = torch.sigmoid(self.aux1(d2))
        d1, _ = self.fuse1(self.up1(d2), t1, m1, a0)

        out = self.out_head(d1)
        if self._vis_dir is not None and not self.training:
            self._save_intermediates(x, a0, m5, m4, m3, m2, m1, out)
        out = self._up_to_input(out, in_size)

        if self.deep_supervision:
            gt_pre = (
                self._up_to_input(m5, in_size),
                self._up_to_input(m4, in_size),
                self._up_to_input(m3, in_size),
                self._up_to_input(m2, in_size),
                self._up_to_input(m1, in_size),
            )
        else:
            gt_pre = (out, out, out, out, out)

        return gt_pre, out
