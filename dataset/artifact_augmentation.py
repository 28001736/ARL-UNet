"""
Lesion-overlap-constrained artifact augmentation for training.

Training-time artifacts are produced by the SAME generators used to build the
offline evaluation benchmark (tools/build_artifact_benchmark.py), so severity
levels {1, 2} match the benchmark exactly. The ArtifactAugmentation wrapper
adds training-only behavior on top of those generators: per-sample probability
gating and lesion-overlap / area-ratio constraints that preserve lesion
visibility when a lesion mask is available.

Usage in Dataset:
    img = Image.open(img_path).convert('RGB')
    mask_pil = Image.open(msk_path).convert('L')
    if self.artifact_aug is not None:
        img = self.artifact_aug(img, lesion_mask=mask_pil)
"""

import random
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from tools.build_artifact_benchmark import corrupt_image


DEFAULT_ARTIFACT_PROBS = {
    "hair": 0.34,
    "air_bubble": 0.27,
    "skin_line": 0.24,
    "highlight": 0.15,
}

DEFAULT_MAX_LESION_OVERLAP = {
    "hair": 0.14,
    "air_bubble": 0.14,
    "skin_line": 0.16,
    "highlight": 0.08,
}

DEFAULT_MAX_AREA_RATIO = {
    "hair": 0.14,
    "air_bubble": 0.13,
    "skin_line": 0.10,
    "highlight": 0.09,
}


# -----------------------------
# Utility functions
# -----------------------------

def _pil_to_bool_mask(mask, size: Tuple[int, int]) -> Optional[np.ndarray]:
    """
    Convert PIL/np mask to bool array with shape [H, W].
    size is PIL size: (W, H).
    """
    if mask is None:
        return None

    if isinstance(mask, Image.Image):
        if mask.size != size:
            mask = mask.resize(size, resample=Image.NEAREST)
        arr = np.asarray(mask.convert("L"))
    else:
        arr = np.asarray(mask)
        if arr.ndim == 3:
            arr = arr[..., 0]
        h, w = arr.shape[:2]
        if (w, h) != size:
            arr = cv2.resize(arr.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)

    return arr > 127


def _artifact_stats(artifact_mask: np.ndarray, lesion_mask: Optional[np.ndarray]):
    artifact_mask = artifact_mask.astype(bool)
    artifact_area_ratio = float(artifact_mask.mean())

    if lesion_mask is None or lesion_mask.sum() == 0:
        lesion_overlap_ratio = 0.0
    else:
        lesion_overlap_ratio = float((artifact_mask & lesion_mask).sum() / (lesion_mask.sum() + 1e-6))

    return artifact_area_ratio, lesion_overlap_ratio


def _sample_weighted(items: Sequence[str], weights: Sequence[float]) -> str:
    return random.choices(list(items), weights=list(weights), k=1)[0]


# -----------------------------
# Main augmentation class
# -----------------------------

class ArtifactAugmentation:
    """
    Benchmark-matched + lesion-overlap-constrained artifact augmentation.

    Backward compatible:
        img = aug(img)

    Recommended:
        img = aug(img, lesion_mask=mask_pil)

    Only image is changed. Mask should remain unchanged.
    """

    def __init__(
        self,
        p: float = 0.15,
        artifacts: Sequence[str] = ("hair", "air_bubble", "skin_line", "highlight"),
        severities: Sequence[int] = (1,),
        enabled: bool = True,
        artifact_probs: Optional[Dict[str, float]] = None,
        max_lesion_overlap: Optional[Dict[str, float]] = None,
        max_area_ratio: Optional[Dict[str, float]] = None,
        max_tries: int = 10,
        fallback_to_clean: bool = True,
        return_debug: bool = False,
    ):
        self.p = float(p)
        self.artifacts = tuple(artifacts)
        self.severities = tuple(severities)
        self.enabled = bool(enabled)
        self.max_tries = int(max_tries)
        self.fallback_to_clean = bool(fallback_to_clean)
        self.return_debug = bool(return_debug)

        artifact_probs = artifact_probs or DEFAULT_ARTIFACT_PROBS
        self.base_artifact_probs = {
            k: float(v) for k, v in artifact_probs.items() if v > 0
        }
        self.artifact_probs = {
            k: float(v) for k, v in self.base_artifact_probs.items() if k in self.artifacts and v > 0
        }
        if not self.artifact_probs:
            self.artifact_probs = {k: 1.0 for k in self.artifacts}

        self.max_lesion_overlap = dict(DEFAULT_MAX_LESION_OVERLAP)
        if max_lesion_overlap is not None:
            self.max_lesion_overlap.update(max_lesion_overlap)

        self.max_area_ratio = dict(DEFAULT_MAX_AREA_RATIO)
        if max_area_ratio is not None:
            self.max_area_ratio.update(max_area_ratio)

    def _choose_artifact(self) -> str:
        items = list(self.artifact_probs.keys())
        weights = list(self.artifact_probs.values())
        return _sample_weighted(items, weights)

    def set_policy(
        self,
        enabled=None,
        p=None,
        artifacts=None,
        severities=None,
    ):
        if enabled is not None:
            self.enabled = bool(enabled)
        if p is not None:
            self.p = float(p)
        if artifacts is not None:
            self.artifacts = tuple(artifacts)
        if severities is not None:
            self.severities = tuple(severities)
        self.artifact_probs = {
            k: v for k, v in self.base_artifact_probs.items() if k in self.artifacts
        }
        if not self.artifact_probs:
            self.artifact_probs = {k: 1.0 for k in self.artifacts}

    def _choose_severity(self) -> int:
        if not self.severities:
            return 1
        return int(random.choice(self.severities))

    def _make_one(self, image: Image.Image, artifact: str, severity: int, rng: np.random.Generator):
        # Reuse the benchmark generators so training-time severities {1, 2}
        # match the offline benchmark exactly.
        return corrupt_image(image, artifact, severity, rng)

    def _is_acceptable(self, artifact: str, artifact_mask: np.ndarray, lesion_mask: Optional[np.ndarray]):
        area_ratio, overlap_ratio = _artifact_stats(artifact_mask, lesion_mask)
        max_area = self.max_area_ratio.get(artifact, 1.0)
        max_overlap = self.max_lesion_overlap.get(artifact, 1.0)

        ok = (area_ratio <= max_area) and (overlap_ratio <= max_overlap)
        return ok, area_ratio, overlap_ratio

    def __call__(self, image: Image.Image, lesion_mask=None):
        if not self.enabled or self.p <= 0:
            if self.return_debug:
                return image, {"applied": False, "reason": "disabled"}
            return image

        if random.random() > self.p:
            if self.return_debug:
                return image, {"applied": False, "reason": "probability"}
            return image

        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")
        else:
            image = image.convert("RGB")

        lesion_mask_bool = _pil_to_bool_mask(lesion_mask, image.size)
        artifact = self._choose_artifact()
        severity = self._choose_severity()

        best = None
        best_score = float("inf")

        for _ in range(self.max_tries):
            rng = np.random.default_rng(random.randint(0, 2**31 - 1))
            out, artifact_mask, params = self._make_one(image, artifact, severity, rng)
            ok, area_ratio, overlap_ratio = self._is_acceptable(
                artifact=artifact,
                artifact_mask=artifact_mask,
                lesion_mask=lesion_mask_bool,
            )

            # Keep the least harmful sample as fallback.
            max_overlap = self.max_lesion_overlap.get(artifact, 1.0)
            max_area = self.max_area_ratio.get(artifact, 1.0)
            score = max(0.0, overlap_ratio - max_overlap) + 0.5 * max(0.0, area_ratio - max_area)
            if score < best_score:
                best_score = score
                best = (out, artifact_mask, params, area_ratio, overlap_ratio)

            if ok:
                if self.return_debug:
                    return out, {
                        "applied": True,
                        "artifact": artifact,
                        "severity": severity,
                        "accepted": True,
                        "area_ratio": area_ratio,
                        "lesion_overlap_ratio": overlap_ratio,
                        "params": params,
                    }
                return out

        # If all samples violate constraints, either return clean or the least harmful sample.
        if self.fallback_to_clean or best is None:
            if self.return_debug:
                return image, {
                    "applied": False,
                    "artifact": artifact,
                    "severity": severity,
                    "accepted": False,
                    "reason": "constraint_failed",
                }
            return image

        out, _, params, area_ratio, overlap_ratio = best
        if self.return_debug:
            return out, {
                "applied": True,
                "artifact": artifact,
                "severity": severity,
                "accepted": False,
                "area_ratio": area_ratio,
                "lesion_overlap_ratio": overlap_ratio,
                "params": params,
            }
        return out

    def apply_with_mask(self, image: Image.Image, lesion_mask=None):
        """
        Apply one artifact and return both the corrupted image and artifact mask.

        This keeps __call__ backward-compatible for existing direct_aug and
        curriculum modes. If augmentation is disabled, skipped by
        probability, or rejected with fallback_to_clean=True, the returned
        artifact mask is all zeros.
        """
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")
        else:
            image = image.convert("RGB")

        zero_mask = Image.new("L", image.size, 0)

        if not self.enabled or self.p <= 0:
            return image, zero_mask

        if random.random() > self.p:
            return image, zero_mask

        lesion_mask_bool = _pil_to_bool_mask(lesion_mask, image.size)
        artifact = self._choose_artifact()
        severity = self._choose_severity()

        best = None
        best_score = float("inf")

        for _ in range(self.max_tries):
            rng = np.random.default_rng(random.randint(0, 2**31 - 1))
            out, artifact_mask, params = self._make_one(image, artifact, severity, rng)
            ok, area_ratio, overlap_ratio = self._is_acceptable(
                artifact=artifact,
                artifact_mask=artifact_mask,
                lesion_mask=lesion_mask_bool,
            )

            max_overlap = self.max_lesion_overlap.get(artifact, 1.0)
            max_area = self.max_area_ratio.get(artifact, 1.0)
            score = max(0.0, overlap_ratio - max_overlap) + 0.5 * max(0.0, area_ratio - max_area)
            if score < best_score:
                best_score = score
                best = (out, artifact_mask)

            if ok:
                mask_img = Image.fromarray((artifact_mask.astype(np.uint8) * 255), mode="L")
                return out, mask_img

        if self.fallback_to_clean or best is None:
            return image, zero_mask

        out, artifact_mask = best
        mask_img = Image.fromarray((artifact_mask.astype(np.uint8) * 255), mode="L")
        return out, mask_img
