import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple, trunc_normal_


class _LayerNorm(nn.Module):
    """ConvNeXt-style LayerNorm supporting channels_last and channels_first."""

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ("channels_last", "channels_first"):
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class _DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = F.layer_norm(x, [H, W])
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class _UCMBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, sr_ratio=1, shift_size=5):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.dim = dim

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.dwconv = _DWConv(mlp_hidden_dim)
        self.dwconv1 = _DWConv(mlp_hidden_dim)
        self.act = act_layer()
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.norm2(x)
        B, N, C = x.shape
        x1 = x.clone().detach()

        x = self.fc1(x)
        x = self.dwconv(x, H, W)

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = self.act1(xn)

        x = self.drop(xn)
        x_s = x.reshape(B, C, H * W).contiguous()
        x = x_s.transpose(1, 2)

        x = self.drop(x)
        x = self.fc2(x)
        x = self.dwconv1(x, H, W)
        x = self.drop(x)

        x += x1
        x = x + self.drop_path(x)
        return x


class _OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.H = img_size[0] // patch_size[0]
        self.W = img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=stride)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class UCMNet(nn.Module):
    """UCM-Net: Shifted MLP Mixer segmentation model (arXiv 2310.09457)."""

    def __init__(self, num_classes=1, input_channels=3, deep_supervision=True,
                 img_size=256, embed_dims=None, drop_path_rate=0.,
                 num_heads=None, depths=None, sr_ratios=None,
                 norm_layer=nn.LayerNorm, logger=None, **kwargs):
        super().__init__()
        if embed_dims is None:
            embed_dims = [8, 16, 24, 32, 48, 64, input_channels]
        if num_heads is None:
            num_heads = [1, 2, 4, 8]
        if depths is None:
            depths = [1, 1, 1]
        if sr_ratios is None:
            sr_ratios = [8, 4, 2, 1]

        self.deep_supervision = deep_supervision

        self.encoder1 = nn.Conv2d(embed_dims[-1], embed_dims[0], 3, stride=1, padding=1)
        self.ebn1 = nn.GroupNorm(4, embed_dims[0])

        self.norm1 = norm_layer(embed_dims[1])
        self.norm2 = norm_layer(embed_dims[2])
        self.norm3 = norm_layer(embed_dims[3])
        self.norm4 = norm_layer(embed_dims[4])
        self.norm5 = norm_layer(embed_dims[5])

        self.dnorm2 = norm_layer(embed_dims[4])
        self.dnorm3 = norm_layer(embed_dims[3])
        self.dnorm4 = norm_layer(embed_dims[2])
        self.dnorm5 = norm_layer(embed_dims[1])
        self.dnorm6 = norm_layer(embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        _block_kwargs = dict(qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                             norm_layer=norm_layer, sr_ratio=sr_ratios[0])

        self.block_0_1 = nn.ModuleList([_UCMBlock(embed_dims[1], num_heads[0], mlp_ratio=1,
                                                   drop_path=dpr[0], **_block_kwargs)])
        self.block0 = nn.ModuleList([_UCMBlock(embed_dims[2], num_heads[0], mlp_ratio=1,
                                                drop_path=dpr[0], **_block_kwargs)])
        self.block1 = nn.ModuleList([_UCMBlock(embed_dims[3], num_heads[0], mlp_ratio=1,
                                                drop_path=dpr[0], **_block_kwargs)])
        self.block2 = nn.ModuleList([_UCMBlock(embed_dims[4], num_heads[0], mlp_ratio=1,
                                                drop_path=dpr[1], **_block_kwargs)])
        self.block3 = nn.ModuleList([_UCMBlock(embed_dims[5], num_heads[0], mlp_ratio=1,
                                                drop_path=dpr[1], **_block_kwargs)])

        self.dblock0 = nn.ModuleList([_UCMBlock(embed_dims[4], num_heads[0], mlp_ratio=1,
                                                 drop_path=dpr[0], **_block_kwargs)])
        self.dblock1 = nn.ModuleList([_UCMBlock(embed_dims[3], num_heads[0], mlp_ratio=1,
                                                 drop_path=dpr[0], **_block_kwargs)])
        self.dblock2 = nn.ModuleList([_UCMBlock(embed_dims[2], num_heads[0], mlp_ratio=1,
                                                 drop_path=dpr[1], **_block_kwargs)])
        self.dblock3 = nn.ModuleList([_UCMBlock(embed_dims[1], num_heads[0], mlp_ratio=1,
                                                 drop_path=dpr[1], **_block_kwargs)])
        self.dblock4 = nn.ModuleList([_UCMBlock(embed_dims[0], num_heads[0], mlp_ratio=1,
                                                 drop_path=dpr[1], **_block_kwargs)])

        self.patch_embed1 = _OverlapPatchEmbed(img_size,       3, 2, embed_dims[0], embed_dims[1])
        self.patch_embed2 = _OverlapPatchEmbed(img_size // 2,  3, 2, embed_dims[1], embed_dims[2])
        self.patch_embed3 = _OverlapPatchEmbed(img_size // 4,  3, 2, embed_dims[2], embed_dims[3])
        self.patch_embed4 = _OverlapPatchEmbed(img_size // 8,  3, 2, embed_dims[3], embed_dims[4])
        self.patch_embed5 = _OverlapPatchEmbed(img_size // 16, 3, 2, embed_dims[4], embed_dims[5])

        self.decoder0 = nn.Conv2d(embed_dims[5], embed_dims[4], 1)
        self.decoder1 = nn.Conv2d(embed_dims[4], embed_dims[3], 1)
        self.decoder2 = nn.Conv2d(embed_dims[3], embed_dims[2], 1)
        self.decoder3 = nn.Conv2d(embed_dims[2], embed_dims[1], 1)
        self.decoder4 = nn.Conv2d(embed_dims[1], embed_dims[0], 1)
        self.decoder5 = nn.Conv2d(embed_dims[0], embed_dims[-1], 1)

        self.dbn0 = nn.GroupNorm(4, embed_dims[4])
        self.dbn1 = nn.GroupNorm(4, embed_dims[3])
        self.dbn2 = nn.GroupNorm(4, embed_dims[2])
        self.dbn3 = nn.GroupNorm(4, embed_dims[1])
        self.dbn4 = nn.GroupNorm(4, embed_dims[0])

        self.finalpre0 = nn.Conv2d(embed_dims[4], num_classes, 1)
        self.finalpre1 = nn.Conv2d(embed_dims[3], num_classes, 1)
        self.finalpre2 = nn.Conv2d(embed_dims[2], num_classes, 1)
        self.finalpre3 = nn.Conv2d(embed_dims[1], num_classes, 1)
        self.finalpre4 = nn.Conv2d(embed_dims[0], num_classes, 1)
        self.final = nn.Conv2d(embed_dims[-1], num_classes, 1)

    def forward(self, x):
        B = x.shape[0]

        # Encoder Stage 1 (conv stem)
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out

        # Encoder Stage 2
        out, H, W = self.patch_embed1(out)
        for blk in self.block_0_1:
            out = blk(out, H, W)
        out = self.norm1(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t2 = out

        # Encoder Stage 3
        out, H, W = self.patch_embed2(out)
        for blk in self.block0:
            out = blk(out, H, W)
        out = self.norm2(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t3 = out

        # Encoder Stage 4
        out, H, W = self.patch_embed3(out)
        for blk in self.block1:
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        # Bottleneck 1
        out, H, W = self.patch_embed4(out)
        for blk in self.block2:
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t5 = out

        # Bottleneck 2
        out, H, W = self.patch_embed5(out)
        for blk in self.block3:
            out = blk(out, H, W)
        out = self.norm5(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # Decoder Stage 0
        out = F.relu(F.interpolate(self.dbn0(self.decoder0(out)), scale_factor=2, mode='bilinear'))
        out = torch.add(out, t5)
        if self.deep_supervision:
            pre0 = torch.sigmoid(self.finalpre0(
                F.interpolate(out, scale_factor=32, mode='bilinear', align_corners=True)))
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock0:
            out = blk(out, H, W)

        # Decoder Stage 1
        out = self.dnorm2(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=2, mode='bilinear'))
        out = torch.add(out, t4)
        if self.deep_supervision:
            pre1 = torch.sigmoid(self.finalpre1(
                F.interpolate(out, scale_factor=16, mode='bilinear', align_corners=True)))
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock1:
            out = blk(out, H, W)

        # Decoder Stage 2
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=2, mode='bilinear'))
        out = torch.add(out, t3)
        if self.deep_supervision:
            pre2 = torch.sigmoid(self.finalpre2(
                F.interpolate(out, scale_factor=8, mode='bilinear', align_corners=True)))
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock2:
            out = blk(out, H, W)

        # Decoder Stage 3
        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=2, mode='bilinear'))
        out = torch.add(out, t2)
        if self.deep_supervision:
            pre3 = torch.sigmoid(self.finalpre3(
                F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)))
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock3:
            out = blk(out, H, W)

        # Decoder Stage 4
        out = self.dnorm5(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=2, mode='bilinear'))
        out = torch.add(out, t1)
        if self.deep_supervision:
            pre4 = torch.sigmoid(self.finalpre4(
                F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)))
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock4:
            out = blk(out, H, W)

        # Final output
        out = self.dnorm6(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=2, mode='bilinear'))
        out = torch.sigmoid(self.final(out))

        if self.deep_supervision:
            return (pre0, pre1, pre2, pre3, pre4), out
        return None, out
