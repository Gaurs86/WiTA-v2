"""
datasets/augmentations.py — Clip augmentation pipeline.

Composes spatial (PIL-level) and temporal (tensor-level) augmentations
using only the internal transforms.py — no external repo dependency.

Train pipeline
--------------
  PIL list → ClipRandomHorizontalFlip
           → ClipRandomGrayscale
           → ClipColorJitter
           → ClipRandomRotation
           → ClipToTensor          → [C, T, H, W] float32
           → ClipNormalize
           → TemporalCrop          (random sub-sequence + resample)
           → DropFrames            (zero out random frames)

Val / test pipeline
-------------------
  PIL list → ClipToTensor → ClipNormalize
"""

from __future__ import annotations
from typing import Any

import torch
from PIL import Image

from configs.default import AugConfig, DataConfig
from datasets.transforms import (
    Compose,
    ClipToTensor,
    ClipNormalize,
    ClipColorJitter,
    ClipRandomRotation,
    ClipRandomHorizontalFlip,
    ClipRandomGrayscale,
    TemporalCrop,
    DropFrames,
)


class WiTAClipAugmentation:
    """
    Clip augmentation pipeline configured from AugConfig + DataConfig.

    Parameters
    ----------
    aug_cfg  : augmentation hyper-parameters
    data_cfg : supplies img_mean, img_std
    mode     : 'train' | 'val' | 'test'
    """

    def __init__(
        self,
        aug_cfg:  AugConfig,
        data_cfg: DataConfig,
        mode:     str = "train",
    ):
        self.mode = mode

        # Shared normalisation (both train and val)
        self._to_tensor  = ClipToTensor(channel_nb=3)
        self._normalise  = ClipNormalize(
            mean=list(data_cfg.img_mean),
            std=list(data_cfg.img_std),
        )

        # Spatial PIL augmentations (train only)
        self._pil_aug = Compose([
            ClipRandomHorizontalFlip(p=aug_cfg.mirror_prob),
            ClipRandomGrayscale(p=aug_cfg.grayscale_prob),
            ClipColorJitter(
                brightness=aug_cfg.brightness,
                contrast=aug_cfg.contrast,
                saturation=aug_cfg.saturation,
                hue=aug_cfg.hue,
            ),
            ClipRandomRotation(degrees=aug_cfg.rotation_deg),
        ])

        # Temporal tensor augmentations (train only)
        self._tensor_aug: list[Any] = [
            TemporalCrop(
                ratio_range=aug_cfg.temporal_crop_ratio,
                min_frames=aug_cfg.temporal_min_frames,
            ),
            DropFrames(p=aug_cfg.drop_frames_prob),
        ]

    def __call__(self, clip: list[Image.Image]) -> torch.Tensor:
        """
        Parameters
        ----------
        clip : list of PIL.Image.Image  (T frames)

        Returns
        -------
        torch.Tensor  [C, T, H, W]  float32, normalised
        """
        if self.mode == "train":
            clip = self._pil_aug(clip)

        tensor = self._normalise(self._to_tensor(clip))   # [C, T, H, W]

        if self.mode == "train":
            for aug in self._tensor_aug:
                tensor = aug(tensor)

        return tensor
