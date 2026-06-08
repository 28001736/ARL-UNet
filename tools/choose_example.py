"""A = {hair, highlight, air_bubble, skin_line}
S = {severity_1, severity_2}

clean: data/isic2017_artifact_benchmark/clean/val/images/ISIC_0012746.jpg
corrupt: data/isic2017_artifact_benchmark/corrupt/val/{A}/{S}/ISIC_0012746.jpg


clean: data/isic2018_artifact_benchmark/clean/val/images/70.png
corrupt: data/isic2018_artifact_benchmark/corrupt/val/{A}/{S}/images/70.png
"""
import os

import matplotlib.pyplot as plt
from PIL import Image

ARTIFACTS = ["hair", "highlight", "air_bubble", "skin_line"]
SEVERITIES = ["severity_1", "severity_2"]

# Each row: (clean image path, corrupt path template with {a} and {s} placeholders)
ROWS = [
    (
        "data/isic2017_artifact_benchmark/clean/val/images/ISIC_0012746.jpg",
        "data/isic2017_artifact_benchmark/corrupt/val/{a}/{s}/images/ISIC_0012746.jpg",
    ),
    (
        "data/isic2018_artifact_benchmark/clean/val/images/70.png",
        "data/isic2018_artifact_benchmark/corrupt/val/{a}/{s}/images/70.png",
    ),
]

OUT_DIR = "results/_vis_arl_unet"


def build_row_paths(clean_path, corrupt_template):
    paths = [clean_path]
    for a in ARTIFACTS:
        for s in SEVERITIES:
            paths.append(corrupt_template.format(a=a, s=s))
    return paths


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    n_rows = len(ROWS)
    n_cols = 1 + len(ARTIFACTS) * len(SEVERITIES)  # 9

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2)
    )

    for r, (clean_path, corrupt_template) in enumerate(ROWS):
        paths = build_row_paths(clean_path, corrupt_template)
        for c, path in enumerate(paths):
            ax = axes[r, c]
            img = Image.open(path).convert("RGB")
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    out_path = os.path.join(OUT_DIR, "artifact_grid.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
