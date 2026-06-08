"""
Replacement for dataset/npy_datasets.py.
This version passes the lesion mask to ArtifactAugmentation so training-time artifacts can be lesion-overlap constrained.
"""

from torch.utils.data import Dataset
import random
import torch
import numpy as np
import os
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from dataset.artifact_augmentation import ArtifactAugmentation


class NPY_datasets(Dataset):
    def __init__(self, path_Data, config, train=True, split=None, val_ratio=None):
        super().__init__()

        if split is None:
            split = 'train' if train else 'val'
        if split not in ('train', 'val', 'test', 'trainval'):
            raise ValueError(f'Unsupported split: {split}')

        source_splits = ('train', 'val') if split == 'trainval' else (split,)

        # Collect items per source split so val can be subsampled independently.
        items_by_split = {}
        for source_split in source_splits:
            image_dir = os.path.join(path_Data, source_split, 'images')
            mask_dir = os.path.join(path_Data, source_split, 'masks')
            if not os.path.isdir(image_dir):
                raise FileNotFoundError(f'Image directory does not exist: {image_dir}')
            if not os.path.isdir(mask_dir):
                raise FileNotFoundError(f'Mask directory does not exist: {mask_dir}')

            images_list = sorted(os.listdir(image_dir))
            masks_list = sorted(os.listdir(mask_dir))
            if len(images_list) != len(masks_list):
                raise ValueError(
                    f'Image/mask count mismatch for {source_split}: '
                    f'{len(images_list)} images vs {len(masks_list)} masks'
                )

            pairs = [
                [os.path.join(image_dir, img), os.path.join(mask_dir, msk)]
                for img, msk in zip(images_list, masks_list)
            ]
            items_by_split[source_split] = pairs

        # Apply val_ratio: deterministically subsample val items.
        if split == 'trainval' and val_ratio is not None and val_ratio < 1.0:
            val_pairs = items_by_split.get('val', [])
            n_val = max(1, round(len(val_pairs) * val_ratio))
            seed = getattr(config, 'seed', 42) or 42
            rng = random.Random(seed)
            val_pairs = rng.sample(val_pairs, n_val)
            items_by_split['val'] = val_pairs

        self.data = []
        for source_split in source_splits:
            self.data.extend(items_by_split[source_split])

        if len(self.data) == 0:
            raise ValueError(f'No image/mask pairs found for {split} under {path_Data}')

        train_mode = getattr(config, 'train_data_mode', 'clean')
        self.split = split
        self.config = config
        self.train_mode = train_mode
        self.transformer = config.train_transformer if split in ('train', 'trainval') else config.test_transformer
        self.artifact_aug = None

        if split in ('train', 'trainval') and train_mode in ('direct_aug', 'curriculum', 'artifact_consistency'):
            if train_mode == 'artifact_consistency':
                p = getattr(config, 'artifact_consistency_p', 0.15)
                artifacts = getattr(config, 'artifact_consistency_artifacts', ('hair', 'air_bubble', 'skin_line', 'highlight'))
                severities = getattr(config, 'artifact_consistency_severities', (1,))
                enabled = True
            else:
                p = getattr(config, 'direct_aug_p', 0.15)
                artifacts = getattr(config, 'direct_aug_artifacts', ('hair', 'air_bubble', 'skin_line', 'highlight'))
                severities = getattr(config, 'direct_aug_severities', (1,))
                enabled = train_mode == 'direct_aug'

            self.artifact_aug = ArtifactAugmentation(
                p=p,
                artifacts=artifacts,
                severities=severities,
                enabled=enabled,
                artifact_probs=getattr(
                    config,
                    'direct_aug_probs',
                    {
                        'hair': 0.34,
                        'air_bubble': 0.27,
                        'skin_line': 0.24,
                        'highlight': 0.15,
                    },
                ),
                max_lesion_overlap=getattr(
                    config,
                    'direct_aug_max_lesion_overlap',
                    {
                        'hair': 0.15,
                        'air_bubble': 0.15,
                        'skin_line': 0.16,
                        'highlight': 0.08,
                    },
                ),
                max_area_ratio=getattr(
                    config,
                    'direct_aug_max_area_ratio',
                    {
                        'hair': 0.16,
                        'air_bubble': 0.14,
                        'skin_line': 0.10,
                        'highlight': 0.10,
                    },
                ),
                max_tries=getattr(config, 'direct_aug_max_tries', 10),
                fallback_to_clean=getattr(config, 'direct_aug_fallback_to_clean', True),
            )

    def _normalize_image(self, img):
        if self.config.datasets == 'isic18':
            mean, std = 157.561, 26.706
        elif self.config.datasets == 'isic17':
            mean, std = 159.922, 28.871
        else:
            mean, std = 0.0, 1.0

        img = img.astype(np.float32)
        img_normalized = (img - mean) / std
        min_value = np.min(img_normalized)
        max_value = np.max(img_normalized)
        denom = max(max_value - min_value, 1e-6)
        return ((img_normalized - min_value) / denom) * 255.0

    def _to_image_tensor(self, img):
        return torch.tensor(img).permute(2, 0, 1)

    def _to_mask_tensor(self, mask):
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=2)
        return torch.tensor(mask).permute(2, 0, 1).float()

    def _shared_train_transform(self, clean_img, artifact_img, lesion_mask, artifact_mask):
        clean_img = self._to_image_tensor(self._normalize_image(clean_img))
        artifact_img = self._to_image_tensor(self._normalize_image(artifact_img))
        lesion_mask = self._to_mask_tensor(lesion_mask)
        artifact_mask = self._to_mask_tensor(artifact_mask)

        if random.random() < 0.5:
            clean_img = TF.hflip(clean_img)
            artifact_img = TF.hflip(artifact_img)
            lesion_mask = TF.hflip(lesion_mask)
            artifact_mask = TF.hflip(artifact_mask)

        if random.random() < 0.5:
            clean_img = TF.vflip(clean_img)
            artifact_img = TF.vflip(artifact_img)
            lesion_mask = TF.vflip(lesion_mask)
            artifact_mask = TF.vflip(artifact_mask)

        if random.random() < 0.5:
            angle = random.uniform(0, 360)
            clean_img = TF.rotate(clean_img, angle, interpolation=InterpolationMode.BILINEAR)
            artifact_img = TF.rotate(artifact_img, angle, interpolation=InterpolationMode.BILINEAR)
            lesion_mask = TF.rotate(lesion_mask, angle, interpolation=InterpolationMode.NEAREST)
            artifact_mask = TF.rotate(artifact_mask, angle, interpolation=InterpolationMode.NEAREST)

        size = [self.config.input_size_h, self.config.input_size_w]
        clean_img = TF.resize(clean_img, size, interpolation=InterpolationMode.BILINEAR)
        artifact_img = TF.resize(artifact_img, size, interpolation=InterpolationMode.BILINEAR)
        lesion_mask = TF.resize(lesion_mask, size, interpolation=InterpolationMode.NEAREST)
        artifact_mask = TF.resize(artifact_mask, size, interpolation=InterpolationMode.NEAREST)
        artifact_mask = torch.clamp(artifact_mask, 0.0, 1.0)

        return clean_img, artifact_img, lesion_mask, artifact_mask

    def set_artifact_policy(
        self,
        enabled=None,
        p=None,
        artifacts=None,
        severities=None,
    ):
        if self.artifact_aug is None:
            return
        self.artifact_aug.set_policy(
            enabled=enabled,
            p=p,
            artifacts=artifacts,
            severities=severities,
        )

    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = Image.open(img_path).convert('RGB')
        msk_pil = Image.open(msk_path).convert('L')

        if (
            self.split in ('train', 'trainval')
            and self.train_mode == 'artifact_consistency'
            and self.artifact_aug is not None
        ):
            artifact_img, artifact_mask_pil = self.artifact_aug.apply_with_mask(img, lesion_mask=msk_pil)
            clean_np = np.array(img)
            artifact_np = np.array(artifact_img)
            lesion_mask_np = np.expand_dims(np.array(msk_pil), axis=2) / 255.0
            artifact_mask_np = np.expand_dims(np.array(artifact_mask_pil), axis=2) / 255.0
            return self._shared_train_transform(clean_np, artifact_np, lesion_mask_np, artifact_mask_np)

        # Key change: pass the lesion mask into artifact augmentation.
        # Mask itself is unchanged; it is only used to reject overly destructive artifacts.
        if self.artifact_aug is not None:
            img = self.artifact_aug(img, lesion_mask=msk_pil)

        img = np.array(img)
        msk = np.expand_dims(np.array(msk_pil), axis=2) / 255.0
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        return len(self.data)
