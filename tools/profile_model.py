import argparse
import copy
import os
import sys
import time
from types import SimpleNamespace

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config_setting import setting_config
from models import build_model
from models.ff_parser_lite import FFParserLite, count_ff_parser_lite


def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('1', 'true', 'yes', 'y', 'on'):
        return True
    if value in ('0', 'false', 'no', 'n', 'off'):
        return False
    raise argparse.ArgumentTypeError(f'Expected a boolean value, got {value}')


def _parse_shape(value):
    shape = tuple(int(item.strip()) for item in value.split(',') if item.strip())
    if len(shape) != 4:
        raise argparse.ArgumentTypeError('input_shape must be N,C,H,W')
    return shape


def _parse_scales(value):
    return [item.strip() for item in value.split(',') if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description='Profile EMA-U-Net or ARL-UNet with optional FF-Parser-Lite.')
    parser.add_argument('--network', default='emaunet', choices=['emaunet', 'arl_unet'])
    parser.add_argument('--input_shape', type=_parse_shape, default=(1, 3, 256, 256))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--repeat', type=int, default=200)
    parser.add_argument('--enable_ffl', action='store_true')
    parser.add_argument('--ffl_scales', default='s8,s16,s32')
    parser.add_argument('--ffl_num_bands', type=int, default=6)
    parser.add_argument('--ffl_reduction', type=int, default=8)
    parser.add_argument('--ffl_share_radial', type=_str2bool, default=True)
    parser.add_argument('--ffl_force_fp32_fft', type=_str2bool, default=True)
    parser.add_argument('--ffl_post_fuse', choices=['none', 'dw_pw'], default='none')
    parser.add_argument(
        '--disable_artifact_stem',
        action='store_true',
        help='ARL-UNet ablation: disable Artifact-aware Stem.',
    )
    parser.add_argument(
        '--disable_robust_encoder',
        action='store_true',
        help='ARL-UNet ablation: use plain encoder blocks.',
    )
    parser.add_argument(
        '--disable_skip_fusion',
        action='store_true',
        help='ARL-UNet ablation: use plain skip fusion.',
    )
    parser.add_argument(
        '--skip_fusion_mode',
        default='soft',
        choices=['soft', 'hard'],
        help='ARL-UNet skip fusion mode when enabled.',
    )
    return parser.parse_args()


def build_profile_model(args):
    model_cfg = copy.deepcopy(setting_config.model_config)
    ff_parser_cfg = copy.deepcopy(model_cfg.get('ff_parser_cfg', {}))
    ff_parser_cfg.update({
        'enable': args.enable_ffl,
        'scales': _parse_scales(args.ffl_scales),
        'num_bands': args.ffl_num_bands,
        'reduction': args.ffl_reduction,
        'share_radial': args.ffl_share_radial,
        'force_fp32_fft': args.ffl_force_fp32_fft,
        'post_fuse': args.ffl_post_fuse,
    })
    model_cfg['ff_parser_cfg'] = ff_parser_cfg

    arl_defaults = {
        'use_artifact_stem': True,
        'use_robust_encoder': True,
        'use_skip_fusion': True,
        'skip_fusion_mode': 'soft',
    }
    arl_cfg = {**arl_defaults, **copy.deepcopy(model_cfg.get('arl_cfg', {}))}
    if args.disable_artifact_stem:
        arl_cfg['use_artifact_stem'] = False
    if args.disable_robust_encoder:
        arl_cfg['use_robust_encoder'] = False
    if args.disable_skip_fusion:
        arl_cfg['use_skip_fusion'] = False
    if getattr(args, 'skip_fusion_mode', None) is not None:
        arl_cfg['skip_fusion_mode'] = args.skip_fusion_mode
    model_cfg['arl_cfg'] = arl_cfg

    config = SimpleNamespace(network=args.network, model_config=model_cfg)
    return build_model(config)


def profile_thop(model, input_shape):
    try:
        from thop import profile
    except ImportError:
        return None

    model = copy.deepcopy(model).cpu().eval()
    dummy = torch.randn(input_shape)
    macs, params = profile(
        model,
        inputs=(dummy,),
        custom_ops={FFParserLite: count_ff_parser_lite},
        verbose=False,
    )
    return macs, params


def measure_latency(model, input_shape, device, warmup, repeat):
    if device.startswith('cuda') and not torch.cuda.is_available():
        return None

    model = model.to(device).eval()
    dummy = torch.randn(input_shape, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        if device.startswith('cuda'):
            torch.cuda.synchronize()
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            for _ in range(repeat):
                model(dummy)
            ender.record()
            torch.cuda.synchronize()
            return starter.elapsed_time(ender) / repeat

        start = time.perf_counter()
        for _ in range(repeat):
            model(dummy)
        return (time.perf_counter() - start) * 1000.0 / repeat


def main():
    args = parse_args()
    model = build_profile_model(args)
    params = sum(parameter.numel() for parameter in model.parameters())
    thop_result = profile_thop(model, args.input_shape)
    latency = measure_latency(model, args.input_shape, args.device, args.warmup, args.repeat)

    print(f'Network: {args.network}')
    if args.network == 'arl_unet':
        print(
            f"ARL ablation: stem={not args.disable_artifact_stem}, "
            f"REB={not args.disable_robust_encoder}, skip_fusion={not args.disable_skip_fusion}"
        )
    print(f'FF-Parser-Lite enabled: {args.enable_ffl}')
    print(f'Input shape: {args.input_shape}')
    print(f'Params: {params / 1e6:.4f} M')
    if thop_result is not None:
        macs, _ = thop_result
        print(f'THOP GFLOPs: {macs / 1e9:.4f}')
    else:
        print('THOP GFLOPs: unavailable (install thop)')
    if latency is not None:
        print(f'Latency: {latency:.3f} ms ({args.device}, batch={args.input_shape[0]})')
    else:
        print(f'Latency: skipped ({args.device} unavailable)')


if __name__ == '__main__':
    main()
