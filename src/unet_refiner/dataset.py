"""Dataset that renders corrupted / perfect DXF pairs to image pairs."""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from .render_dxf import render_dxf_to_image


class DXFImageDataset(Dataset):
    """Renders corrupted and perfect DXF files to paired raster images.

    Images are cached as ``.npy`` files under ``<root>/processed_images/<split>/``
    so rendering only happens once.

    Args:
        root:       Project root (contains ``dataset/`` folder).
        split:      ``'train'``, ``'val'``, or ``'test'``.
        image_size: Square image resolution in pixels.
        augment:    Whether to apply random augmentations (flips + 90° rotations).
    """

    def __init__(self, root, split='train', image_size=512, augment=False):
        self.root = root
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.cache_dir = os.path.join(root, 'processed_images', split)
        os.makedirs(self.cache_dir, exist_ok=True)

        dataset_dir = os.path.join(root, 'dataset')
        pair_dirs = sorted(
            glob.glob(os.path.join(dataset_dir, 'pair_*')),
            key=lambda x: int(os.path.basename(x).split('_')[1]),
        )

        # Keep pairs that contain both DXF files
        valid = []
        for p in pair_dirs:
            if (os.path.exists(os.path.join(p, 'not perfict.dxf'))
                    and os.path.exists(os.path.join(p, 'perfect.dxf'))):
                valid.append(p)

        # 80 / 10 / 10 split
        n = len(valid)
        t_end = int(0.8 * n)
        v_end = int(0.9 * n)

        if split == 'train':
            self.pairs = valid[:t_end]
        elif split == 'val':
            self.pairs = valid[t_end:v_end]
        elif split == 'test':
            self.pairs = valid[v_end:]
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

    # ------------------------------------------------------------------
    def _cache_path(self, pair_dir, kind):
        name = os.path.basename(pair_dir)
        return os.path.join(self.cache_dir, f'{name}_{kind}.npy')

    def _load_or_render(self, pair_dir):
        c_path = self._cache_path(pair_dir, 'corrupted')
        p_path = self._cache_path(pair_dir, 'perfect')

        if os.path.exists(c_path) and os.path.exists(p_path):
            return np.load(c_path), np.load(p_path)

        corrupted = render_dxf_to_image(
            os.path.join(pair_dir, 'not perfict.dxf'), self.image_size)
        perfect = render_dxf_to_image(
            os.path.join(pair_dir, 'perfect.dxf'), self.image_size)

        np.save(c_path, corrupted)
        np.save(p_path, perfect)
        return corrupted, perfect

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        corrupted, perfect = self._load_or_render(self.pairs[idx])

        # Random augmentation (applied identically to both images)
        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                corrupted = np.flip(corrupted, axis=1).copy()
                perfect = np.flip(perfect, axis=1).copy()
            # Random vertical flip
            if np.random.rand() > 0.5:
                corrupted = np.flip(corrupted, axis=0).copy()
                perfect = np.flip(perfect, axis=0).copy()
            # Random 90° rotation
            k = np.random.randint(0, 4)
            if k > 0:
                corrupted = np.rot90(corrupted, k).copy()
                perfect = np.rot90(perfect, k).copy()

        # (H, W) → (1, H, W)  float32
        c_tensor = torch.from_numpy(corrupted).unsqueeze(0).float()
        p_tensor = torch.from_numpy(perfect).unsqueeze(0).float()
        return c_tensor, p_tensor

    # ------------------------------------------------------------------
    def process_all(self):
        """Pre-render every pair to the cache (called once before training)."""
        total = len(self.pairs)
        for i, pair_dir in enumerate(self.pairs):
            self._load_or_render(pair_dir)
            if (i + 1) % 100 == 0 or (i + 1) == total:
                print(f'  [{self.split}] Rendered {i + 1}/{total} pairs')
