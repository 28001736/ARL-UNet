import torch
import torch.nn.functional as F


def _box_blur(x, kernel_size):
    pad = kernel_size // 2
    return F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)


def _normalize01(x, eps=1e-6):
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


def _artifact_cues(images):
    x = images / 255.0 if images.max() > 1.5 else images
    x = torch.clamp(x, 0.0, 1.0)
    gray = x.mean(dim=1, keepdim=True)
    gray = _normalize01(gray)

    lap_kernel = gray.new_tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
    ).view(1, 1, 3, 3)
    lap = F.conv2d(gray, lap_kernel, padding=1).abs()
    lap = _normalize01(lap)

    dilated = F.max_pool2d(gray, kernel_size=9, stride=1, padding=4)
    closed = -F.max_pool2d(-dilated, kernel_size=9, stride=1, padding=4)
    blackhat = torch.relu(closed - gray)
    blackhat = _normalize01(blackhat)

    local_mean = _box_blur(gray, 15)
    local_contrast = torch.abs(gray - local_mean)
    local_contrast = _normalize01(local_contrast)

    dark_thin = blackhat * torch.relu(0.45 - gray) / 0.45
    bright = torch.relu(gray - 0.72) / 0.28
    bright_low_texture = bright * torch.relu(0.45 - local_contrast) / 0.45
    bright_rim = bright * local_contrast

    hair_cue = torch.clamp(0.75 * dark_thin + 0.25 * lap, 0.0, 1.0)
    air_bubble_cue = torch.clamp(0.65 * bright_rim + 0.35 * local_contrast * bright, 0.0, 1.0)
    skin_line_cue = torch.clamp(0.55 * local_contrast + 0.45 * lap, 0.0, 1.0)
    highlight_cue = torch.clamp(bright_low_texture, 0.0, 1.0)

    artifact_cue = torch.clamp(
        0.32 * hair_cue + 0.26 * air_bubble_cue + 0.22 * skin_line_cue + 0.20 * highlight_cue,
        0.0,
        1.0,
    )
    artifact_cue = torch.clamp((artifact_cue - 0.12) / 0.35, 0.0, 1.0)
    artifact_cue = _box_blur(artifact_cue, 5)
    return artifact_cue, hair_cue, air_bubble_cue, skin_line_cue, highlight_cue


def _restore_views(images, hair_cue, air_bubble_cue, skin_line_cue, highlight_cue):
    local_3 = _box_blur(images, 3)
    local_7 = _box_blur(images, 7)
    local_15 = _box_blur(images, 15)

    hair_mask = torch.clamp((hair_cue - 0.18) / 0.42, 0.0, 1.0)
    air_mask = torch.clamp((air_bubble_cue - 0.16) / 0.40, 0.0, 1.0)
    skin_line_mask = torch.clamp((skin_line_cue - 0.22) / 0.45, 0.0, 1.0)
    highlight_mask = torch.clamp((highlight_cue - 0.12) / 0.38, 0.0, 1.0)

    hair_view = images * (1.0 - hair_mask) + local_7 * hair_mask
    air_view = images * (1.0 - air_mask) + local_3 * air_mask
    skin_line_view = images * (1.0 - skin_line_mask) + local_3 * skin_line_mask
    highlight_view = images * (1.0 - highlight_mask) + local_15 * highlight_mask
    return [hair_view, air_view, skin_line_view, highlight_view]


def _forward_pred(model, images):
    _, outputs = model(images)
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    return outputs


def predict_artifact_tta(model, images, alpha_max=0.35):
    """Conservative artifact-aware test-time fusion.

    Inputs and outputs follow the repository convention: images are normalized
    tensors and model outputs are probabilities in [0, 1].
    """
    pred_original = _forward_pred(model, images)
    artifact_cue, hair_cue, air_bubble_cue, skin_line_cue, highlight_cue = _artifact_cues(images)
    views = _restore_views(images, hair_cue, air_bubble_cue, skin_line_cue, highlight_cue)

    view_preds = [_forward_pred(model, view) for view in views]
    pred_view = torch.stack(view_preds, dim=0).mean(dim=0)

    uncertainty = 1.0 - torch.abs(pred_original - 0.5) * 2.0
    uncertainty = torch.clamp(uncertainty, 0.0, 1.0)
    agreement = 1.0 - torch.abs(pred_original - pred_view)
    agreement = torch.clamp(agreement, 0.0, 1.0)
    alpha = float(alpha_max) * artifact_cue * (0.50 + 0.35 * uncertainty + 0.15 * agreement)
    alpha = torch.clamp(alpha, 0.0, float(alpha_max))

    pred = pred_original * (1.0 - alpha) + pred_view * alpha
    return torch.clamp(pred, 0.0, 1.0)


def predict_with_robust_inference(model, images, mode='none', alpha_max=0.35):
    if mode == 'none':
        return _forward_pred(model, images)
    if mode == 'artifact_tta':
        return predict_artifact_tta(model, images, alpha_max=alpha_max)
    raise ValueError(f'Unsupported robust_inference mode: {mode}')
