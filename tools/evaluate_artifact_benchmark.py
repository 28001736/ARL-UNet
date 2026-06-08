import argparse
import copy
import csv
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from medpy.metric.binary import hd95 as _medpy_hd95

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_model
from models.ff_parser_lite import FFParserLite, count_ff_parser_lite
from tools.robust_inference import predict_with_robust_inference


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
ARTIFACTS = ('hair', 'air_bubble', 'skin_line', 'highlight')
SEVERITIES = (1, 2)
MODEL_CONFIG = {
    'num_classes': 1,
    'input_channels': 3,
    'c_list': [8, 16, 24, 32, 48, 64],
    'arl_c_list': [8, 16, 24, 36, 48, 72],
    'bridge': True,
    'deep_supervision': True,
    'ff_parser_cfg': {
        'enable': False,
        'scales': ['s8', 's16', 's32'],
        'num_bands': 6,
        'reduction': 8,
        'min_hidden': 4,
        'fft_norm': 'ortho',
        'residual_gamma': 0.25,
        'force_fp32_fft': True,
        'share_radial': True,
        'post_fuse': 'none',
        'eps': 1e-6,
    },
    'arl_cfg': {
        'use_artifact_stem': True,
        'use_robust_encoder': True,
        'use_skip_fusion': True,
        'skip_fusion_mode': 'soft',
    },
}
NORMALIZE_STATS = {
    'isic17': {'mean': 148.429, 'std': 25.748},
    'isic18': {'mean': 149.034, 'std': 32.022},
}
METRICS = ['miou', 'dice', 'accuracy', 'specificity', 'sensitivity', 'hd95']
CONDITION_CSV_COLUMNS = ['dataset', 'model', 'params_m', 'gflops', 'num_images'] + METRICS
SUMMARY_CSV_COLUMNS = ['dataset', 'model'] + METRICS


def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('1', 'true', 'yes', 'y', 'on'):
        return True
    if value in ('0', 'false', 'no', 'n', 'off'):
        return False
    raise argparse.ArgumentTypeError(f'Expected a boolean value, got {value}')


def _parse_scales(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [item.strip() for item in value.split(',') if item.strip()]


def _mask_stem_candidates(image_stem: str):
    return (
        image_stem,
        f'{image_stem}_segmentation',
        f'{image_stem}_mask',
    )


def pair_image_masks(image_dir: Path, mask_dir: Path):
    images = {
        path.stem: path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    }
    masks = {
        path.stem: path
        for path in mask_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    }
    pairs = []
    missing_masks = []
    for image_stem in sorted(images):
        mask_path = None
        for mask_stem in _mask_stem_candidates(image_stem):
            if mask_stem in masks:
                mask_path = masks[mask_stem]
                break
        if mask_path is None:
            missing_masks.append(image_stem)
        else:
            pairs.append((images[image_stem], mask_path))

    if missing_masks:
        raise ValueError(
            f'Image/mask stem mismatch for {image_dir} and {mask_dir}: '
            f'missing_masks={missing_masks[:5]}'
        )
    return pairs


class SegmentationFolderDataset(Dataset):
    def __init__(self, image_dir, mask_dir, dataset_name, input_size=(256, 256)):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.input_size = input_size
        self.mean = NORMALIZE_STATS[dataset_name]['mean']
        self.std = NORMALIZE_STATS[dataset_name]['std']

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f'Missing image directory: {self.image_dir}')
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f'Missing mask directory: {self.mask_dir}')

        self.samples = pair_image_masks(self.image_dir, self.mask_dir)
        if not self.samples:
            raise RuntimeError(f'No image/mask pairs found in {self.image_dir}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        size_w, size_h = self.input_size[1], self.input_size[0]
        image = Image.open(image_path).convert('RGB').resize((size_w, size_h), Image.BILINEAR)
        mask = Image.open(mask_path).convert('L').resize((size_w, size_h), Image.NEAREST)

        image = np.asarray(image).astype(np.float32)
        image = (image - self.mean) / self.std
        denom = np.max(image) - np.min(image)
        if denom > 0:
            image = ((image - np.min(image)) / denom) * 255.0

        mask = (np.asarray(mask).astype(np.float32) / 255.0)[..., None]
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).permute(2, 0, 1)
        return image, mask


def build_eval_model(args):
    model_cfg = copy.deepcopy(MODEL_CONFIG)
    ff_parser_cfg = copy.deepcopy(MODEL_CONFIG['ff_parser_cfg'])
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
    if args.ffl_force_fp32_fft is not None:
        ff_parser_cfg['force_fp32_fft'] = args.ffl_force_fp32_fft
    if args.ffl_share_radial is not None:
        ff_parser_cfg['share_radial'] = args.ffl_share_radial
    if args.ffl_post_fuse is not None:
        ff_parser_cfg['post_fuse'] = args.ffl_post_fuse
        
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


def load_weights(model, weights_path, device, strict=True):
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    state_dict = {
        key.replace('module.', '', 1) if key.startswith('module.') else key: value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=strict)


def profile_model(model, input_size):
    params = sum(parameter.numel() for parameter in model.parameters()) / 1e6
    try:
        from thop import profile

        dummy = torch.randn(1, 3, input_size[0], input_size[1])
        model_cpu = model.cpu().eval()
        macs, _ = profile(
            model_cpu,
            inputs=(dummy,),
            custom_ops={FFParserLite: count_ff_parser_lite},
            verbose=False,
        )
        return params, macs / 1e9
    except Exception:
        return params, None


def condition_specs(benchmark_dir: Path):
    clean_mask_dir = benchmark_dir / 'clean/val/masks'
    specs = [('clean', None, None, benchmark_dir / 'clean/val/images', clean_mask_dir)]
    for artifact in ARTIFACTS:
        for severity in SEVERITIES:
            condition = f'{artifact}_s{severity}'
            base = benchmark_dir / 'corrupt/val' / artifact / f'severity_{severity}'
            specs.append((condition, artifact, severity, base / 'images', clean_mask_dir))
    return specs


def _compute_hd95(preds_bin, targets_bin):
    """Average HD95 (px) over valid image pairs. Returns 0.0 when no valid pair exists."""
    scores = []
    for p, g in zip(preds_bin, targets_bin):
        if p.sum() > 0 and g.sum() > 0:
            try:
                scores.append(_medpy_hd95(p, g))
            except RuntimeError:
                pass
    return float(np.mean(scores)) if scores else 0.0


def compute_metrics(preds, targets, threshold=0.5):
    preds_bin = (preds >= threshold).astype(np.uint8)
    targets_bin = (targets >= 0.5).astype(np.uint8)

    tp = np.logical_and(preds_bin == 1, targets_bin == 1).sum()
    tn = np.logical_and(preds_bin == 0, targets_bin == 0).sum()
    fp = np.logical_and(preds_bin == 1, targets_bin == 0).sum()
    fn = np.logical_and(preds_bin == 0, targets_bin == 1).sum()

    eps = 1e-8
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    miou = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    sensitivity = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    return {
        'dice': float(dice),
        'miou': float(miou),
        'accuracy': float(accuracy),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'hd95': _compute_hd95(preds_bin, targets_bin),
    }


def _save_rgb_png(t, path) -> None:
    """Save a (C, H, W) image tensor as a min-max-normalized PNG (RGB when C == 3)."""
    a = t.detach().float().cpu().numpy()
    if a.ndim == 3 and a.shape[0] == 3:
        lo, hi = float(a.min()), float(a.max())
        if hi > lo:
            a = (a - lo) / (hi - lo)
        a = np.transpose((a * 255.0).clip(0, 255).astype(np.uint8), (1, 2, 0))
        Image.fromarray(a, mode='RGB').save(path)
        return
    a = a.squeeze()
    lo, hi = float(a.min()), float(a.max())
    if hi > lo:
        a = (a - lo) / (hi - lo)
    Image.fromarray((a * 255.0).clip(0, 255).astype(np.uint8), mode='L').save(path)


def _save_mask_png(t, path, threshold: float = 0.5) -> None:
    """Save a single-channel mask/probability tensor as a binary (0/255) PNG."""
    a = t.detach().float().cpu().numpy()
    if a.ndim == 3:
        a = a.squeeze(0)
    a = (a >= threshold).astype(np.uint8) * 255
    Image.fromarray(a, mode='L').save(path)


def evaluate_condition(model, args, image_dir, mask_dir, device, condition=None):
    dataset = SegmentationFolderDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        dataset_name=args.dataset,
        input_size=(args.input_size_h, args.input_size_w),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
    )

    save_interval = int(getattr(args, 'save_seg_interval', 0) or 0)
    seg_dir = None
    if save_interval > 0 and condition is not None:
        model_safe = args.model_name.replace(' ', '_').replace('/', '_')
        base = Path(args.save_seg_dir) if args.save_seg_dir else (Path(args.output_dir) / 'seg_samples')
        seg_dir = base / model_safe / condition
        seg_dir.mkdir(parents=True, exist_ok=True)

    preds = []
    targets = []
    sample_idx = 0
    model.eval()
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            outputs = predict_with_robust_inference(
                model,
                images,
                mode=getattr(args, 'robust_inference', 'none'),
                alpha_max=getattr(args, 'robust_inference_alpha', 0.35),
            )

            if seg_dir is not None:
                probs = outputs.detach().cpu()
                for b in range(images.shape[0]):
                    if sample_idx % save_interval == 0:
                        stem = dataset.samples[sample_idx][0].stem
                        prefix = seg_dir / f'{sample_idx:05d}_{stem}'
                        _save_rgb_png(images[b], f'{prefix}_input.png')
                        _save_mask_png(masks[b], f'{prefix}_gt.png', threshold=0.5)
                        _save_mask_png(probs[b], f'{prefix}_pred.png', threshold=args.threshold)
                    sample_idx += 1

            preds.append(outputs.squeeze(1).cpu().numpy())
            targets.append(masks.squeeze(1).numpy())

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    metrics = compute_metrics(preds, targets, threshold=args.threshold)
    metrics['num_images'] = len(dataset)
    return metrics


def fmt(value):
    if value is None:
        return ''
    if isinstance(value, float):
        return f'{value:.6f}'
    return value


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key, '')) for key in fieldnames})


def upsert_csv(path: Path, row, fieldnames, key_fields):
    existing_rows = []
    if path.exists():
        with path.open('r', newline='', encoding='utf-8') as file:
            existing_rows = list(csv.DictReader(file))

    row_key = tuple(str(row.get(key, '')) for key in key_fields)
    kept_rows = [
        existing
        for existing in existing_rows
        if tuple(str(existing.get(key, '')) for key in key_fields) != row_key
    ]
    kept_rows.append(row)
    write_csv(path, kept_rows, fieldnames)


def _avg_metrics(rows):
    return {m: float(np.mean([r[m] for r in rows])) for m in METRICS}


def make_severity_avg_row(condition_rows, severity, args):
    rows = [r for r in condition_rows if r['severity'] == severity]
    return {'dataset': args.dataset, 'model': args.model_name, **_avg_metrics(rows)}


def make_corrupted_avg_row(condition_rows, args):
    rows = [r for r in condition_rows if r['condition'] != 'clean']
    return {'dataset': args.dataset, 'model': args.model_name, **_avg_metrics(rows)}


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate clean val and artifact-corrupted val benchmarks.')
    parser.add_argument('--dataset', required=True, choices=['isic17', 'isic18'])
    parser.add_argument('--benchmark', required=True, help='e.g. data/isic2018_artifact_benchmark')
    parser.add_argument('--weights', required=True, help='Path to best.pth or latest.pth')
    parser.add_argument('--model_name', default='EMA-U-Net')
    parser.add_argument('--output_dir', default='results/artifact_eval')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--input_size_h', type=int, default=256)
    parser.add_argument('--input_size_w', type=int, default=256)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument(
        '--save_seg_interval',
        type=int,
        default=100,
        help='Save input/GT/prediction PNGs every N samples per condition (0 disables).',
    )
    parser.add_argument(
        '--save_seg_dir',
        default=None,
        help='Directory for saved segmentation samples (default: <output_dir>/seg_samples).',
    )
    parser.add_argument('--strict_load', type=_str2bool, default=False)
    parser.add_argument('--network', default='emaunet', choices=['emaunet', 'arl_unet', 'unet', 'ucm_net'])
    parser.add_argument(
        '--robust_inference',
        default='none',
        choices=['none', 'artifact_tta'],
        help='Optional test-time robust inference. none keeps the original single-view prediction.',
    )
    parser.add_argument(
        '--robust_inference_alpha',
        type=float,
        default=0.35,
        help='Maximum artifact-region fusion strength for --robust_inference artifact_tta.',
    )

    ffl_group = parser.add_mutually_exclusive_group()
    ffl_group.add_argument('--enable_ffl', action='store_true', help='Enable FF-Parser-Lite.')
    ffl_group.add_argument('--disable_ffl', action='store_true', help='Disable FF-Parser-Lite.')
    parser.add_argument('--ffl_scales', default=None)
    parser.add_argument('--ffl_num_bands', type=int, default=None)
    parser.add_argument('--ffl_reduction', type=int, default=None)
    parser.add_argument('--ffl_min_hidden', type=int, default=None)
    parser.add_argument('--ffl_force_fp32_fft', type=_str2bool, default=None)
    parser.add_argument('--ffl_share_radial', type=_str2bool, default=None)
    parser.add_argument('--ffl_post_fuse', default=None, choices=['none', 'dw_pw'])
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


def main():
    args = parse_args()
    benchmark_dir = Path(args.benchmark)
    output_dir = Path(args.output_dir)
    device = torch.device(args.device)

    model = build_eval_model(args)
    params_m, gflops = profile_model(model, (args.input_size_h, args.input_size_w))
    model = model.to(device)
    load_weights(model, args.weights, device, strict=args.strict_load)

    condition_rows = []
    for condition, artifact, severity, image_dir, mask_dir in condition_specs(benchmark_dir):
        metrics = evaluate_condition(model, args, image_dir, mask_dir, device, condition=condition)
        row = {
            'dataset': args.dataset,
            'model': args.model_name,
            'condition': condition,
            'artifact': artifact or '',
            'severity': severity,
            **metrics,
        }
        condition_rows.append(row)
        metrics_str = ' | '.join(f'{m}={metrics[m]:.4f}' for m in METRICS)
        print(f'{condition}: {metrics_str}')

    # 9 per-condition CSVs
    by_condition = {r['condition']: r for r in condition_rows}
    all_conditions = ['clean'] + [f'{art}_s{sev}' for art in ARTIFACTS for sev in SEVERITIES]
    for condition in all_conditions:
        r = by_condition[condition]
        row = {
            'dataset': args.dataset,
            'model': args.model_name,
            'params_m': params_m,
            'gflops': gflops,
            'num_images': r['num_images'],
            **{m: r[m] for m in METRICS},
        }
        upsert_csv(output_dir / f'{condition}.csv', row, CONDITION_CSV_COLUMNS, ['dataset', 'model'])

    # severity_1_avg.csv and severity_2_avg.csv
    for severity in SEVERITIES:
        row = make_severity_avg_row(condition_rows, severity, args)
        upsert_csv(output_dir / f'severity_{severity}_avg.csv', row, SUMMARY_CSV_COLUMNS, ['dataset', 'model'])

    # corrupted_avg.csv
    row = make_corrupted_avg_row(condition_rows, args)
    upsert_csv(output_dir / 'corrupted_avg.csv', row, SUMMARY_CSV_COLUMNS, ['dataset', 'model'])

    print(f'Results saved to {output_dir}')


if __name__ == '__main__':
    main()
