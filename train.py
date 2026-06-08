import argparse
import copy
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from models import build_model
from models.ff_parser_lite import FFParserLite, count_ff_parser_lite
from dataset.npy_datasets import NPY_datasets
from engine import *


from utils import *
from configs.config_setting import setting_config
from dataset.artifact_curriculum import get_artifact_curriculum_policy
from tools.evaluate_artifact_benchmark import (
    ARTIFACTS,
    SEVERITIES,
    condition_specs,
    evaluate_condition,
)


def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('1', 'true', 'yes', 'y', 'on'):
        return True
    if value in ('0', 'false', 'no', 'n', 'off'):
        return False
    raise argparse.ArgumentTypeError(f'Expected a boolean value, got {value}')


def _parse_gpu_ids(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [int(item.strip()) for item in value.split(',') if item.strip()]


def _parse_scales(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [item.strip() for item in value.split(',') if item.strip()]


def _parse_ints(value):
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(item.strip()) for item in value.split(',') if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description='Train EMA-U-Net or ARL-UNet with optional FF-Parser-Lite.')
    parser.add_argument('--network', default='arl_unet', help="Override config network: emaunet, arl_unet, unet, or ucm_net.")
    parser.add_argument('--datasets', default=None, choices=['isic17', 'isic18'])
    parser.add_argument('--data_path', default=None)
    parser.add_argument('--work_dir', default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--val_interval', type=int, default=None)
    parser.add_argument('--print_interval', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--gpu_ids', default=None, help='Comma-separated physical GPU ids, e.g. 0 or 0,1.')
    parser.add_argument('--amp', type=_str2bool, default=None)
    parser.add_argument('--profile_before_train', type=_str2bool, default=None)
    parser.add_argument(
        '--loss_mode',
        default='base',
        choices=['base', 'boundary_shape'],
        help='base: original BCE+Dice loss; boundary_shape: add conservative boundary and shape losses.',
    )
    parser.add_argument(
        '--train_split',
        default=None,
        choices=['train', 'trainval'],
        help="Training dataset split. Use 'trainval' to concatenate train and val for training.",
    )
    parser.add_argument(
        '--train_split_ratio',
        type=float,
        default=None,
        metavar='RATIO',
        help=(
            'Fraction of the val set (0.0–1.0) to mix into training. '
            '0.0 = train only, 1.0 = full trainval. '
            'Overrides --train_split when provided.'
        ),
    )
    parser.add_argument(
        '--train_data_mode',
        default='clean',
        choices=['clean', 'direct_aug', 'curriculum'],
        help='  clean: clean train only; \
                direct_aug: online artifact augmentation; \
                curriculum: staged policy.\n',
    )
    parser.add_argument('--direct_aug_p', type=float, default=None)
    parser.add_argument(
        '--direct_aug_artifacts',
        default=None,
        help='Comma-separated artifacts: hair,air_bubble,skin_line,highlight.',
    )
    parser.add_argument('--direct_aug_severities', default=None, help='Comma-separated severities for direct_aug, e.g. 1,2. Defaults to (1, 2).')
    parser.add_argument(
        '--artifact_val_benchmark',
        default=None,
        help='Artifact benchmark root used for optional artifact validation logging.',
    )
    parser.add_argument(
        '--artifact_val_interval',
        type=int,
        default=None,
        help='Run optional artifact benchmark validation every N epochs.',
    )

    ffl_group = parser.add_mutually_exclusive_group()
    ffl_group.add_argument('--enable_ffl', action='store_true', help='Enable FF-Parser-Lite.')
    ffl_group.add_argument('--disable_ffl', action='store_true', help='Disable FF-Parser-Lite.')
    parser.add_argument('--ffl_scales', default=None, help='Comma-separated scales: s8,s16,s32.')
    parser.add_argument('--ffl_num_bands', type=int, default=None)
    parser.add_argument('--ffl_reduction', type=int, default=None)
    parser.add_argument('--ffl_min_hidden', type=int, default=None)
    parser.add_argument('--ffl_fft_norm', default=None, choices=['backward', 'forward', 'ortho'])
    parser.add_argument('--ffl_residual_gamma', type=float, default=None)
    parser.add_argument('--ffl_force_fp32_fft', type=_str2bool, default=None)
    parser.add_argument('--ffl_share_radial', type=_str2bool, default=None)
    parser.add_argument('--ffl_post_fuse', default=None, choices=['none', 'dw_pw'])

    parser.add_argument(
        '--disable_artifact_stem',
        action='store_true',
        help='ARL-UNet ablation: disable Artifact-aware Stem (plain stem + zero artifact map).',
    )
    parser.add_argument(
        '--disable_robust_encoder',
        action='store_true',
        help='ARL-UNet ablation: replace Robust Encoder Block with a single DSConv residual block.',
    )
    parser.add_argument(
        '--disable_skip_fusion',
        action='store_true',
        help='ARL-UNet ablation: replace Mask-guided Robust Skip Fusion with plain D+S fusion.',
    )
    parser.add_argument(
        '--skip_fusion_mode',
        default='soft',
        choices=['soft', 'hard'],
        help="ARL-UNet skip fusion mode when skip fusion is enabled: soft (default) or hard (legacy).",
    )
    return parser.parse_args()


def apply_overrides(config, args):
    config.model_config = copy.deepcopy(config.model_config)

    for name in (
        'network',
        'datasets',
        'data_path',
        'epochs',
        'batch_size',
        'val_interval',
        'print_interval',
        'num_workers',
        'seed',
        'amp',
        'profile_before_train',
        'loss_mode',
        'train_split',
        'train_split_ratio',
        'train_data_mode',
        'direct_aug_p',
        'artifact_val_benchmark',
        'artifact_val_interval',
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(config, name, value)

    # Resolve train_split_ratio → train_split override.
    ratio = getattr(config, 'train_split_ratio', None)
    if ratio is not None:
        if not (0.0 <= ratio <= 1.0):
            raise ValueError(f'--train_split_ratio must be in [0.0, 1.0], got {ratio}')
        config.train_split = 'train' if ratio <= 0.0 else 'trainval'

    if args.gpu_ids is not None:
        config.gpu_ids = _parse_gpu_ids(args.gpu_ids)
    if args.direct_aug_artifacts is not None:
        config.direct_aug_artifacts = tuple(_parse_scales(args.direct_aug_artifacts))
    if args.direct_aug_severities is not None:
        config.direct_aug_severities = tuple(_parse_ints(args.direct_aug_severities))
    if getattr(config, 'train_data_mode', 'clean') == 'curriculum':
        config.direct_aug_severities = (1,)
    if getattr(config, 'train_data_mode', 'clean') == 'artifact_consistency':
        config.artifact_consistency_severities = (1,)

    if args.datasets is not None and args.data_path is None:
        if config.datasets == 'isic18':
            config.data_path = './data/isic2018/'
        elif config.datasets == 'isic17':
            config.data_path = './data/isic2017/'

    if args.artifact_val_benchmark is None and not getattr(config, 'artifact_val_benchmark', ''):
        if config.datasets == 'isic18':
            config.artifact_val_benchmark = './data/isic2018_artifact_benchmark'
        elif config.datasets == 'isic17':
            config.artifact_val_benchmark = './data/isic2017_artifact_benchmark'

    if args.work_dir is not None:
        config.work_dir = args.work_dir
    elif (
        any(getattr(args, name) is not None for name in ('network', 'datasets', 'train_split', 'train_data_mode', 'loss_mode'))
        or args.enable_ffl
        or args.disable_artifact_stem
        or args.disable_robust_encoder
        or args.disable_skip_fusion
    ):
        tag = 'ffl' if args.enable_ffl else config.network
        if getattr(config, 'network', None) == 'arl_unet':
            abl_parts = []
            if args.disable_artifact_stem:
                abl_parts.append('noStem')
            if args.disable_robust_encoder:
                abl_parts.append('noREB')
            if args.disable_skip_fusion:
                abl_parts.append('noSkip')
            if abl_parts:
                tag = f"{tag}_{'_'.join(abl_parts)}"
        if config.train_data_mode == 'direct_aug':
            tag = f'{tag}_direct_aug'
        if config.train_data_mode == 'curriculum':
            tag = f'{tag}_curriculum'
        if config.train_data_mode == 'artifact_consistency':
            tag = f'{tag}_artifact_consistency'
        if getattr(config, 'loss_mode', 'base') == 'boundary_shape':
            tag = f'{tag}_boundary_shape'
        _ratio = getattr(config, 'train_split_ratio', None)
        if getattr(config, 'train_split', 'train') != 'train':
            if _ratio is not None and 0.0 < _ratio < 1.0:
                tag = f'{tag}_trainval{_ratio:.2f}'
            else:
                tag = f'{tag}_{config.train_split}'
        config.work_dir = f'results/{tag}_{config.datasets}_{datetime.now().strftime("%Y%m%d-%H%M%S")}/'

    ff_parser_cfg = copy.deepcopy(config.model_config.get('ff_parser_cfg', {}))
    if args.enable_ffl:
        ff_parser_cfg['enable'] = True
    if args.disable_ffl:
        ff_parser_cfg['enable'] = False
    if args.ffl_scales is not None:
        ff_parser_cfg['scales'] = _parse_scales(args.ffl_scales)
    if args.ffl_num_bands is not None:
        ff_parser_cfg['num_bands'] = args.ffl_num_bands
    if args.ffl_reduction is not None:
        ff_parser_cfg['reduction'] = args.ffl_reduction
    if args.ffl_min_hidden is not None:
        ff_parser_cfg['min_hidden'] = args.ffl_min_hidden
    if args.ffl_fft_norm is not None:
        ff_parser_cfg['fft_norm'] = args.ffl_fft_norm
    if args.ffl_residual_gamma is not None:
        ff_parser_cfg['residual_gamma'] = args.ffl_residual_gamma
    if args.ffl_force_fp32_fft is not None:
        ff_parser_cfg['force_fp32_fft'] = args.ffl_force_fp32_fft
    if args.ffl_share_radial is not None:
        ff_parser_cfg['share_radial'] = args.ffl_share_radial
    if args.ffl_post_fuse is not None:
        ff_parser_cfg['post_fuse'] = args.ffl_post_fuse
    config.model_config['ff_parser_cfg'] = ff_parser_cfg

    arl_defaults = {
        'use_artifact_stem': True,
        'use_robust_encoder': True,
        'use_skip_fusion': True,
        'skip_fusion_mode': 'soft',
    }
    arl_cfg = {**arl_defaults, **copy.deepcopy(config.model_config.get('arl_cfg', {}))}
    if args.disable_artifact_stem:
        arl_cfg['use_artifact_stem'] = False
    if args.disable_robust_encoder:
        arl_cfg['use_robust_encoder'] = False
    if args.disable_skip_fusion:
        arl_cfg['use_skip_fusion'] = False
    if args.skip_fusion_mode is not None:
        arl_cfg['skip_fusion_mode'] = args.skip_fusion_mode
    config.model_config['arl_cfg'] = arl_cfg

    if config.train_data_mode == 'curriculum':
        if args.epochs is None:
            config.epochs = getattr(config, 'curriculum_epochs', 300)
        config.sch = getattr(config, 'curriculum_scheduler', 'ArtifactCurriculumLR')
        config.warm_up_epochs = getattr(config, 'curriculum_warm_up_epochs', 20)
        config.eta_min = getattr(config, 'curriculum_eta_min', getattr(config, 'eta_min', 1e-5))

    if config.train_data_mode == 'artifact_consistency':  # artifact_consistency not renamed
        if args.epochs is None:
            config.epochs = getattr(config, 'artifact_consistency_epochs', 320)

    if getattr(config, 'network', 'emaunet') == 'unet':
        config.criterion = SimpleBceDiceLoss(wb=1, wd=1, scale=2.5)

    return config


def _artifact_val_args(config):
    return SimpleNamespace(
        dataset=config.datasets,
        input_size_h=config.input_size_h,
        input_size_w=config.input_size_w,
        batch_size=1,
        num_workers=config.num_workers,
        threshold=config.threshold,
    )


def val_artifact_benchmark(model, config, epoch, logger):
    benchmark_dir = Path(config.artifact_val_benchmark)
    if not benchmark_dir.is_dir():
        raise FileNotFoundError(
            f'Artifact validation benchmark does not exist: {benchmark_dir}'
        )

    device = next(model.parameters()).device
    eval_model = model.module if hasattr(model, 'module') else model
    args = _artifact_val_args(config)
    condition_metrics = {}

    for condition, artifact, severity, image_dir, mask_dir in condition_specs(benchmark_dir):
        metrics = evaluate_condition(eval_model, args, image_dir, mask_dir, device)
        condition_metrics[condition] = metrics

    corrupt_mious = [
        condition_metrics[f'{artifact}_s{severity}']['miou']
        for artifact in ARTIFACTS
        for severity in SEVERITIES
    ]
    clean_miou = condition_metrics['clean']['miou']
    avg_corrupt_miou = float(np.mean(corrupt_mious))
    all_condition_miou = float(np.mean([clean_miou, *corrupt_mious]))

    logger.info(
        f'val_artifact | epoch {epoch:03d} | '
        f'all_miou {all_condition_miou:.4f} | '
        f'clean_miou {clean_miou:.4f} | avg_corrupt_miou {avg_corrupt_miou:.4f}'
    )
    return {
        'artifact_all_miou': all_condition_miou,
        'artifact_clean_miou': clean_miou,
        'artifact_avg_corrupt_miou': avg_corrupt_miou,
    }


def _log_model_profile(model, config, logger):
    params = sum(parameter.numel() for parameter in model.parameters())
    logger.info(f'Params: {params / 1e6:.4f} M')

    try:
        from thop import profile

        input_shape = (
            1,
            config.model_config['input_channels'],
            config.input_size_h,
            config.input_size_w,
        )
        dummy = torch.randn(input_shape)
        macs, _ = profile(
            model.cpu(),
            inputs=(dummy,),
            custom_ops={FFParserLite: count_ff_parser_lite},
            verbose=False,
        )
        logger.info(f'THOP GFLOPs: {macs / 1e9:.4f}')
    except Exception as exc:
        logger.info(f'Profile skipped: {exc}')


def main(config):
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)

    log_section(logger, 'Creating Logger')
    log_config_info(config, logger)

    log_section(logger, 'GPU Init')
    set_seed(config.seed)
    physical_gpu_ids = getattr(config, 'gpu_ids', [0])
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(gpu_id) for gpu_id in physical_gpu_ids)
    gpu_ids = list(range(len(physical_gpu_ids)))
    torch.cuda.empty_cache()

    log_section(logger, 'Preparing Dataset')
    train_dataset = NPY_datasets(
        config.data_path, config, train=True, split=config.train_split,
        val_ratio=getattr(config, 'train_split_ratio', None),
    )
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=config.num_workers)
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=config.num_workers,
                            drop_last=True)

    log_section(logger, 'Preparing Model')
    model = build_model(config, logger=logger)

    if getattr(config, 'profile_before_train', False):
        _log_model_profile(model, config, logger)

    model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])

    log_section(logger, 'Preparing Loss, Optimizer, Scheduler, AMP')
    if getattr(config, 'loss_mode', 'base') == 'boundary_shape':
        config.criterion = BoundaryShapeGT_BceDiceLoss(
            wb=1,
            wd=1,
            boundary_weight=getattr(config, 'boundary_loss_weight', 0.10),
            shape_weight=getattr(config, 'shape_loss_weight', 0.05),
            boundary_kernel_size=getattr(config, 'boundary_kernel_size', 5),
        )
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler()

    log_section(logger, 'Setting Training State')
    start_epoch = 1

    if os.path.exists(resume_model):
        log_section(logger, 'Resuming Model')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        loss = checkpoint.get('loss', float('nan'))

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, loss: {loss:.4f}'
        logger.info(log_info)

    log_section(logger, 'Training')
    for epoch in range(start_epoch, config.epochs + 1):

        if getattr(config, 'train_data_mode', 'clean') == 'curriculum':
            policy = get_artifact_curriculum_policy(epoch)
            aug_policy = {
                'enabled': policy['enabled'],
                'p': policy['p'],
                'artifacts': policy['artifacts'],
                'severities': policy['severities'],
            }
            dataset = train_loader.dataset
            # if wrapped by another dataset (Subset, etc.)
            if hasattr(dataset, "dataset"):
                dataset = dataset.dataset
            if hasattr(dataset, "set_artifact_policy"):
                dataset.set_artifact_policy(**aug_policy)
            logger.info(
                f"[Epoch {epoch:03d}] Artifact curriculum: "
                f"stage={policy['stage']}, enabled={policy['enabled']}, p={policy['p']}, "
                f"artifacts={policy['artifacts']}, severities={policy['severities']}"
                f", lr_scale={policy['lr_scale']}"
            )

        torch.cuda.empty_cache()

        train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            logger,
            config,
            scaler=scaler
        )

        loss, metrics = val_one_epoch(
            val_loader,
            model,
            criterion,
            epoch,
            logger,
            config
        )

        clean_miou = metrics['miou'] if metrics is not None else None
        artifact_metrics = None
        artifact_interval = max(1, int(getattr(config, 'artifact_val_interval', 1)))
        artifact_benchmark = getattr(config, 'artifact_val_benchmark', '')
        artifact_benchmark_exists = bool(artifact_benchmark) and Path(artifact_benchmark).is_dir()
        if artifact_benchmark_exists:
            if epoch % artifact_interval == 0:
                artifact_metrics = val_artifact_benchmark(model, config, epoch, logger)
            else:
                logger.info(
                    f'val_artifact | epoch {epoch:03d} | skipped '
                    f'(artifact_val_interval={artifact_interval})'
                )

        torch.save(
            {
                'epoch': epoch,
                'loss': loss,
                'clean_miou': clean_miou,
                'artifact_metrics': artifact_metrics,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth'))


if __name__ == '__main__':
    args = parse_args()
    config = apply_overrides(setting_config, args)
    main(config)
