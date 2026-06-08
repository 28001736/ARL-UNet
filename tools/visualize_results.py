import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

result_path = [
    "results/_artifact_eval_isic18_unet/seg_samples/U-Net",
    "results/_artifact_eval_isic18_ucm_net/seg_samples/UCM-Net",
    "results/_artifact_eval_isic18_emaunet/seg_samples/EMA-UNet",
    "results/_artifact_eval_isic18_arl_unet/seg_samples/ARL-UNet",
]

"""
suffix:
    Ground Truth: gt
    Input: input pred
    Model Prediction: pred

Example 1 (e.g., 00000_0_{suffix}.png)
--------------------------------
1. Input(Artifact Type): clean, hair_s1, hair_s2, highlight_s1, highlight_s2, air_bubble_s1, air_bubble_s2, skin_line_s1, skin_line_s2
2. GT (All the same for each input)
3. Pred (ARL-UNet Prediction)
4. Pred (UCM-Net Prediction)
5. Pred (EMA-UNet Prediction)
6. Pred (U-Net Prediction)
"""

# 9 columns: artifact conditions, in display order.
CONDITIONS = [
    "clean",
    "hair_s1", "hair_s2",
    "highlight_s1", "highlight_s2",
    "air_bubble_s1", "air_bubble_s2",
    "skin_line_s1", "skin_line_s2",
]

# Rows 3-6: model predictions, in display order.
PRED_ROW_ORDER = ["ARL-UNet", "UCM-Net", "EMA-UNet", "U-Net"]

OUTPUT_DIR = "results/_seg_grids_isic18"


def _model_paths():
    """Map model display name (folder basename) -> seg_samples path."""
    return {Path(p).name: Path(p) for p in result_path}


def _condition_label(cond):
    if cond == "clean":
        return "Clean"
    artifact, severity = cond.rsplit("_s", 1)
    return f"{artifact.replace('_', ' ').title()} s{severity}"


def _find_shared(model_paths, condition, prefix, suffix):
    """Locate an input/gt file from any model folder (identical across models)."""
    for name in PRED_ROW_ORDER:
        mp = model_paths.get(name)
        if mp is None:
            continue
        f = mp / condition / f"{prefix}_{suffix}.png"
        if f.exists():
            return f
    return None


def _list_sample_prefixes(model_paths):
    """Sample prefixes (e.g. '00000_0') from the reference model's clean folder."""
    for name in PRED_ROW_ORDER:
        ref = model_paths.get(name)
        if ref is not None and (ref / "clean").is_dir():
            return [f.name[: -len("_input.png")] for f in sorted((ref / "clean").glob("*_input.png"))]
    return []


def _show(ax, path, is_rgb):
    if path is None or not Path(path).exists():
        ax.imshow(np.zeros((8, 8)), cmap="gray", vmin=0, vmax=255)
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes, color="red", fontsize=8)
        return
    if is_rgb:
        ax.imshow(np.asarray(Image.open(path).convert("RGB")))
    else:
        ax.imshow(np.asarray(Image.open(path).convert("L")), cmap="gray", vmin=0, vmax=255)


def build_grid(prefix, model_paths, out_path):
    row_labels = ["Input", "GT"] + PRED_ROW_ORDER
    n_rows, n_cols = len(row_labels), len(CONDITIONS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.6, n_rows * 1.6))

    for r in range(n_rows):
        for c, cond in enumerate(CONDITIONS):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                _show(ax, _find_shared(model_paths, cond, prefix, "input"), is_rgb=True)
            elif r == 1:
                _show(ax, _find_shared(model_paths, cond, prefix, "gt"), is_rgb=False)
            else:
                mp = model_paths.get(PRED_ROW_ORDER[r - 2])
                pred = (mp / cond / f"{prefix}_pred.png") if mp is not None else None
                _show(ax, pred, is_rgb=False)
            if r == 0:
                ax.set_title(_condition_label(cond), fontsize=9)
            if c == 0:
                ax.set_ylabel(row_labels[r], fontsize=11)

    fig.suptitle(f"Sample {prefix}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    model_paths = _model_paths()
    prefixes = _list_sample_prefixes(model_paths)
    if not prefixes:
        print("No samples found; run the benchmark eval with --save_seg_interval first.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for prefix in prefixes:
        out_path = os.path.join(OUTPUT_DIR, f"sample_{prefix}.png")
        build_grid(prefix, model_paths, out_path)
        print(f"saved {out_path}")
    print(f"Done. {len(prefixes)} grids saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
