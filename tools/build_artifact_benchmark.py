import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
DEFAULT_ARTIFACTS = ['hair', 'air_bubble', 'skin_line', 'highlight']
DEFAULT_SEVERITIES = [1, 2]


def stable_seed(name: str, artifact: str, severity: int, base_seed: int) -> int:
    key = f'{name}_{artifact}_{severity}_{base_seed}'.encode('utf-8')
    digest = hashlib.md5(key).hexdigest()
    return int(digest[:8], 16)


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert('RGB')


def save_rgb(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def copy_if_different(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and src.resolve() == dst.resolve():
        return
    shutil.copy2(src, dst)


def iter_images(image_dir: Path):
    return sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def read_mask_if_exists(mask_dir: Path, image_path: Path):
    mask_path = find_mask_path(mask_dir, image_path)
    if mask_path is not None:
        mask = Image.open(mask_path).convert('L')
        return np.asarray(mask) > 127
    return None


def find_mask_path(mask_dir: Path, image_path: Path):
    candidate_stems = (
        image_path.stem,
        f'{image_path.stem}_segmentation',
        f'{image_path.stem}_mask',
    )
    for stem in candidate_stems:
        for ext in IMAGE_EXTS:
            mask_path = mask_dir / f'{stem}{ext}'
            if mask_path.exists():
                return mask_path
    return None


def artifact_stats(artifact_mask: np.ndarray, lesion_mask):
    artifact_area_ratio = float(artifact_mask.mean())

    if lesion_mask is None or lesion_mask.sum() == 0:
        lesion_overlap_ratio = None
    else:
        lesion_overlap_ratio = float((artifact_mask & lesion_mask).sum() / lesion_mask.sum())

    return artifact_area_ratio, lesion_overlap_ratio


def quadratic_bezier(p0, p1, p2, n=80):
    ts = np.linspace(0.0, 1.0, n)
    points = []
    for t in ts:
        point = (
            (1 - t) ** 2 * np.array(p0)
            + 2 * (1 - t) * t * np.array(p1)
            + t ** 2 * np.array(p2)
        )
        points.append(tuple(point.astype(np.int32)))
    return points


def add_hair(img: Image.Image, severity: int, rng: np.random.Generator):
    width, height = img.size

    if severity == 1:
        num_hairs = int(rng.integers(14, 26))
        thickness_range = (1, 3)
        alpha_range = (122, 188)
        blur_radius = 0.52
    else:
        num_hairs = int(rng.integers(15, 27))
        thickness_range = (1, 3)
        alpha_range = (115, 186)
        blur_radius = 0.55

    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    mask_img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(overlay)
    mask_draw = ImageDraw.Draw(mask_img)

    def random_border_point(side):
        if side == 0:
            return (int(rng.integers(0, width)), 0)
        if side == 1:
            return (width - 1, int(rng.integers(0, height)))
        if side == 2:
            return (int(rng.integers(0, width)), height - 1)
        return (0, int(rng.integers(0, height)))

    for _ in range(num_hairs):
        p0 = random_border_point(int(rng.integers(0, 4)))
        p2 = random_border_point(int(rng.integers(0, 4)))
        p1 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        points = quadratic_bezier(p0, p1, p2, n=80)

        thickness = int(rng.integers(thickness_range[0], thickness_range[1] + 1))
        alpha = int(rng.integers(alpha_range[0], alpha_range[1] + 1))
        color_base = int(rng.integers(5, 45))
        color = (
            color_base,
            max(0, color_base - int(rng.integers(0, 12))),
            max(0, color_base - int(rng.integers(0, 12))),
            alpha,
        )

        draw.line(points, fill=color, width=thickness, joint='curve')
        mask_draw.line(points, fill=255, width=thickness + 2)

    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    out = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
    artifact_mask = np.asarray(mask_img) > 20

    params = {
        'num_hairs': num_hairs,
        'thickness_range': thickness_range,
        'alpha_range': alpha_range,
        'blur_radius': blur_radius,
    }
    return out, artifact_mask, params


def add_highlight(img: Image.Image, severity: int, rng: np.random.Generator):
    width, height = img.size
    min_side = min(width, height)

    if severity == 1:
        num_spots = int(rng.integers(1, 4))
        alpha_range = (76, 136)
        radius_range = (max(5, int(0.035 * min_side)), max(8, int(0.085 * min_side)))
        blur_factor = float(rng.uniform(0.30, 0.48))
    else:
        num_spots = int(rng.integers(2, 5))
        alpha_range = (96, 168)
        radius_range = (max(6, int(0.045 * min_side)), max(9, int(0.115 * min_side)))
        blur_factor = float(rng.uniform(0.34, 0.55))

    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    mask_img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(overlay)
    mask_draw = ImageDraw.Draw(mask_img)

    for _ in range(num_spots):
        rx = int(rng.integers(radius_range[0], radius_range[1] + 1))
        ry = int(rx * float(rng.uniform(0.55, 1.20)))
        cx = int(rng.integers(rx, max(rx + 1, width - rx)))
        cy = int(rng.integers(ry, max(ry + 1, height - ry)))
        alpha = int(rng.integers(alpha_range[0], alpha_range[1] + 1))

        box = [cx - rx, cy - ry, cx + rx, cy + ry]
        draw.ellipse(box, fill=(255, 255, 255, alpha))
        mask_draw.ellipse(box, fill=255)

        core_box = [
            cx - int(rx * 0.28),
            cy - int(ry * 0.28),
            cx + int(rx * 0.28),
            cy + int(ry * 0.28),
        ]
        draw.ellipse(core_box, fill=(255, 255, 255, min(210, alpha + 52)))

    blur_radius = blur_factor * max(radius_range[0], 3)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    out = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
    artifact_mask = np.asarray(mask_img) > 20

    params = {
        'num_spots': num_spots,
        'alpha_range': alpha_range,
        'radius_range': radius_range,
        'blur_radius': blur_radius,
    }
    return out, artifact_mask, params


def add_skin_line(img: Image.Image, severity: int, rng: np.random.Generator):
    width, height = img.size
    min_side = min(width, height)

    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    mask_img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(overlay)
    mask_draw = ImageDraw.Draw(mask_img)

    if severity == 1:
        num_lines = int(rng.integers(18, 30))
        length_range = (int(0.22 * min_side), int(0.55 * min_side))
        alpha_range = (76, 128)
        thickness_range = (1, 2)
        blur_radius = float(rng.uniform(0.12, 0.30))
        shadow_alpha_range = (18, 42)
    else:
        num_lines = int(rng.integers(32, 48))
        length_range = (int(0.30 * min_side), int(0.72 * min_side))
        alpha_range = (96, 160)
        thickness_range = (1, 3)
        blur_radius = float(rng.uniform(0.10, 0.26))
        shadow_alpha_range = (30, 60)

    for _ in range(num_lines):
        length = int(rng.integers(max(8, length_range[0]), max(9, length_range[1]) + 1))
        cx = int(rng.integers(0, width))
        cy = int(rng.integers(0, height))
        angle = float(rng.uniform(0, np.pi))
        dx = np.cos(angle) * length / 2.0
        dy = np.sin(angle) * length / 2.0

        p0 = (int(np.clip(cx - dx, 0, width - 1)), int(np.clip(cy - dy, 0, height - 1)))
        p2 = (int(np.clip(cx + dx, 0, width - 1)), int(np.clip(cy + dy, 0, height - 1)))
        bend = float(rng.uniform(-0.18, 0.18) * length)
        p1 = (
            int(np.clip(cx - np.sin(angle) * bend, 0, width - 1)),
            int(np.clip(cy + np.cos(angle) * bend, 0, height - 1)),
        )
        points = quadratic_bezier(p0, p1, p2, n=50)

        alpha = int(rng.integers(alpha_range[0], alpha_range[1] + 1))
        thickness = int(rng.integers(thickness_range[0], thickness_range[1] + 1))
        if rng.random() < 0.65:
            color_base = int(rng.integers(45, 95))
            color = (
                min(120, color_base + int(rng.integers(10, 30))),
                min(105, color_base + int(rng.integers(2, 18))),
                max(25, color_base - int(rng.integers(8, 28))),
                alpha,
            )
        else:
            color_base = int(rng.integers(150, 205))
            color = (
                min(235, color_base + int(rng.integers(8, 28))),
                min(220, color_base + int(rng.integers(0, 16))),
                max(115, color_base - int(rng.integers(18, 48))),
                int(alpha * 0.75),
            )
        shadow_alpha = int(rng.integers(shadow_alpha_range[0], shadow_alpha_range[1] + 1))
        shadow_color = (35, 28, 22, shadow_alpha)
        shadow_points = [
            (int(np.clip(x + 1, 0, width - 1)), int(np.clip(y + 1, 0, height - 1)))
            for x, y in points
        ]
        draw.line(shadow_points, fill=shadow_color, width=thickness + 1, joint='curve')
        draw.line(points, fill=color, width=thickness, joint='curve')
        mask_draw.line(points, fill=255, width=thickness + 2)

    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    out = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
    artifact_mask = np.asarray(mask_img) > 20

    params = {
        'num_lines': num_lines,
        'length_range': length_range,
        'alpha_range': alpha_range,
        'shadow_alpha_range': shadow_alpha_range,
        'thickness_range': thickness_range,
        'blur_radius': blur_radius,
    }
    return out, artifact_mask, params


def _ensure_uint8_rgb(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError('Input image should be RGB with shape [H, W, 3].')

    return img


def _add_air_bubbles_numpy(
    img: np.ndarray,
    rng: np.random.Generator,
    num_bubbles_range,
    radius_range,
    alpha_range,
    rim_alpha_range,
    blur_sigma_range,
    brightness_gain_range,
    prob_inner_blur,
):
    """Core air-bubble synthesis (uint8 RGB in/out). Returns (out_uint8, bubble_mask_uint8)."""
    img = _ensure_uint8_rgb(img)
    h, w = img.shape[:2]
    out = img.copy().astype(np.float32)

    bubble_mask_total = np.zeros((h, w), dtype=np.float32)

    num_bubbles = int(rng.integers(num_bubbles_range[0], num_bubbles_range[1] + 1))

    for _ in range(num_bubbles):
        radius = int(rng.integers(radius_range[0], radius_range[1] + 1))

        cx = int(rng.integers(radius, max(radius + 1, w - radius)))
        cy = int(rng.integers(radius, max(radius + 1, h - radius)))

        axis_x = radius
        axis_y = int(radius * rng.uniform(0.65, 1.25))
        angle = float(rng.uniform(0, 180))

        alpha = float(rng.uniform(alpha_range[0], alpha_range[1]))
        rim_alpha = float(rng.uniform(rim_alpha_range[0], rim_alpha_range[1]))
        blur_sigma = float(rng.uniform(blur_sigma_range[0], blur_sigma_range[1]))
        brightness_gain = float(rng.uniform(brightness_gain_range[0], brightness_gain_range[1]))

        bubble = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(
            bubble,
            center=(cx, cy),
            axes=(axis_x, axis_y),
            angle=angle,
            startAngle=0,
            endAngle=360,
            color=1.0,
            thickness=-1,
        )

        k = int(blur_sigma * 6 + 1)
        if k % 2 == 0:
            k += 1
        bubble_soft = cv2.GaussianBlur(bubble, (k, k), blur_sigma)
        bubble_soft = np.clip(bubble_soft, 0, 1)

        bubble_region = out.copy()

        local_mean = cv2.GaussianBlur(bubble_region, (0, 0), sigmaX=radius / 2)
        bubble_region = 0.65 * bubble_region + 0.35 * local_mean

        bubble_region = np.clip(bubble_region + brightness_gain, 0, 255)

        if rng.random() < prob_inner_blur:
            blurred = cv2.GaussianBlur(bubble_region, (0, 0), sigmaX=1.2)
            bubble_region = 0.5 * bubble_region + 0.5 * blurred

        mask3 = bubble_soft[..., None]
        out = out * (1 - alpha * mask3) + bubble_region * (alpha * mask3)

        outer = np.zeros((h, w), dtype=np.float32)
        inner = np.zeros((h, w), dtype=np.float32)

        cv2.ellipse(
            outer,
            center=(cx, cy),
            axes=(axis_x, axis_y),
            angle=angle,
            startAngle=0,
            endAngle=360,
            color=1.0,
            thickness=max(1, int(radius * 0.12)),
        )

        cv2.ellipse(
            inner,
            center=(cx, cy),
            axes=(max(1, int(axis_x * 0.72)), max(1, int(axis_y * 0.72))),
            angle=angle,
            startAngle=0,
            endAngle=360,
            color=1.0,
            thickness=max(1, int(radius * 0.06)),
        )

        ring = np.clip(outer + 0.45 * inner, 0, 1)
        ring = cv2.GaussianBlur(ring, (k, k), blur_sigma)
        ring = np.clip(ring, 0, 1)

        white = np.ones_like(out) * 255.0
        ring3 = ring[..., None]
        out = out * (1 - rim_alpha * ring3) + white * (rim_alpha * ring3)

        if rng.random() < 0.8:
            glint = np.zeros((h, w), dtype=np.float32)
            hx = int(cx - axis_x * rng.uniform(0.15, 0.45))
            hy = int(cy - axis_y * rng.uniform(0.15, 0.45))
            hr = max(2, int(radius * rng.uniform(0.08, 0.18)))

            cv2.circle(glint, (hx, hy), hr, 1.0, -1)
            glint = cv2.GaussianBlur(glint, (0, 0), sigmaX=max(1.0, hr / 2))
            glint = np.clip(glint, 0, 1)

            glint3 = glint[..., None]
            out = out * (1 - 0.65 * glint3) + white * (0.65 * glint3)

        bubble_mask_total = np.maximum(bubble_mask_total, bubble_soft)

    out = np.clip(out, 0, 255).astype(np.uint8)
    bubble_mask_total = (np.clip(bubble_mask_total, 0, 1) * 255).astype(np.uint8)
    return out, bubble_mask_total


def add_air_bubbles(img: Image.Image, severity: int, rng: np.random.Generator):
    arr = np.asarray(img)
    if severity == 1:
        kw = {
            'num_bubbles_range': (2, 4),
            'radius_range': (9, 28),
            'alpha_range': (0.14, 0.30),
            'rim_alpha_range': (0.27, 0.52),
            'blur_sigma_range': (1.0, 2.5),
            'brightness_gain_range': (13, 40),
            'prob_inner_blur': 0.62,
        }
    else:
        kw = {
            'num_bubbles_range': (2, 5),
            'radius_range': (11, 38),
            'alpha_range': (0.18, 0.37),
            'rim_alpha_range': (0.34, 0.64),
            'blur_sigma_range': (1.3, 3.2),
            'brightness_gain_range': (18, 48),
            'prob_inner_blur': 0.74,
        }

    out_u8, bubble_mask_u8 = _add_air_bubbles_numpy(arr, rng, **kw)
    artifact_mask = bubble_mask_u8 > 20
    params = {'severity': severity, **{k: list(v) if isinstance(v, tuple) else v for k, v in kw.items()}}
    return Image.fromarray(out_u8), artifact_mask, params


def corrupt_image(img: Image.Image, artifact: str, severity: int, rng: np.random.Generator):
    if artifact == 'hair':
        return add_hair(img, severity, rng)
    if artifact == 'air_bubble':
        return add_air_bubbles(img, severity, rng)
    if artifact == 'skin_line':
        return add_skin_line(img, severity, rng)
    if artifact == 'highlight':
        return add_highlight(img, severity, rng)
    raise ValueError(f'Unknown artifact: {artifact}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build paired corrupted val images for artifact robustness evaluation.'
    )
    parser.add_argument('--dataset', required=True, help='Original dataset directory, e.g. data/isic2018.')
    parser.add_argument(
        '--output',
        required=True,
        help='Artifact benchmark directory, e.g. data/isic2018_artifact_benchmark.',
    )
    parser.add_argument(
        '--split',
        default='val',
        help='Source split to corrupt. This repo uses val as the test split.',
    )
    parser.add_argument('--output_split', default='val', help='Output split name under corrupt/.')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--artifacts', nargs='+', default=DEFAULT_ARTIFACTS)
    parser.add_argument('--severities', nargs='+', type=int, default=DEFAULT_SEVERITIES)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output)
    image_dir = dataset_dir / args.split / 'images'
    mask_dir = dataset_dir / args.split / 'masks'

    if not image_dir.exists():
        raise FileNotFoundError(f'Missing source image directory: {image_dir}')
    if not mask_dir.exists():
        raise FileNotFoundError(f'Missing source mask directory: {mask_dir}')

    unknown_artifacts = sorted(set(args.artifacts) - set(DEFAULT_ARTIFACTS))
    if unknown_artifacts:
        raise ValueError(f'Unknown artifacts: {unknown_artifacts}')
    if any(severity not in DEFAULT_SEVERITIES for severity in args.severities):
        raise ValueError(f'Severities must be one of {DEFAULT_SEVERITIES}: {args.severities}')

    image_paths = iter_images(image_dir)
    if not image_paths:
        raise RuntimeError(f'No images found in {image_dir}')

    clean_image_dir = output_dir / 'clean' / args.output_split / 'images'
    clean_mask_dir = output_dir / 'clean' / args.output_split / 'masks'
    clean_image_dir.mkdir(parents=True, exist_ok=True)
    clean_mask_dir.mkdir(parents=True, exist_ok=True)
    for image_path in image_paths:
        copy_if_different(image_path, clean_image_dir / image_path.name)
        mask_path = find_mask_path(mask_dir, image_path)
        if mask_path is None:
            raise FileNotFoundError(f'Missing mask for image: {image_path}')
        copy_if_different(mask_path, clean_mask_dir / mask_path.name)

    metadata_dir = output_dir / 'metadata'
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / 'corrupt_val_metadata.csv'
    rows = []

    for image_path in image_paths:
        img = load_rgb(image_path)
        lesion_mask = read_mask_if_exists(mask_dir, image_path)

        for artifact in args.artifacts:
            for severity in args.severities:
                seed = stable_seed(image_path.name, artifact, severity, args.seed)
                rng = np.random.default_rng(seed)
                out_img, artifact_mask, params = corrupt_image(img, artifact, severity, rng)

                out_path = (
                    output_dir
                    / 'corrupt'
                    / args.output_split
                    / artifact
                    / f'severity_{severity}'
                    / 'images'
                    / image_path.name
                )
                save_rgb(out_img, out_path)

                artifact_area_ratio, lesion_overlap_ratio = artifact_stats(
                    artifact_mask=artifact_mask,
                    lesion_mask=lesion_mask,
                )
                rows.append({
                    'image_id': image_path.stem,
                    'source_image': str(image_path),
                    'corrupt_image': str(out_path),
                    'artifact_type': artifact,
                    'severity': severity,
                    'seed': seed,
                    'artifact_area_ratio': artifact_area_ratio,
                    'lesion_overlap_ratio': lesion_overlap_ratio,
                    'params': json.dumps(params, ensure_ascii=True),
                })

    with metadata_path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                'image_id',
                'source_image',
                'corrupt_image',
                'artifact_type',
                'severity',
                'seed',
                'artifact_area_ratio',
                'lesion_overlap_ratio',
                'params',
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f'Source split: {image_dir}')
    print(f'Generated {len(rows)} corrupted images.')
    print(f'Metadata saved to {metadata_path}')


if __name__ == '__main__':
    main()
