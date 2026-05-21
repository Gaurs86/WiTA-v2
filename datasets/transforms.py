"""
datasets/transforms.py — Self-contained video / clip transforms.

Re-implements the WiTA baseline's video_transforms.py with:
  • Identical behaviour — clips processed the same way
  • No skimage dependency (replaced with PIL/torchvision equivalents)
  • Proper type hints and compose API

All transforms operate on  list[PIL.Image.Image]  (clip-consistent: the same
random parameters are applied to every frame) or on  torch.Tensor  for the
normalisation step.

Baseline equivalents
--------------------
  ColorJitter(b, c, s, h)           → ClipColorJitter
  RandomRotation(degrees)           → ClipRandomRotation
  ClipToTensor(channel_nb=3)        → ClipToTensor
  Normalize(mean, std)              → ClipNormalize
  Compose(transforms)               → Compose
"""

from __future__ import annotations

import random
import numbers
from typing import Any

import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------

class Compose:
    """Chain a sequence of clip transforms."""
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, clip: Any) -> Any:
        for t in self.transforms:
            clip = t(clip)
        return clip

    def __repr__(self) -> str:
        lines = [f"{self.__class__.__name__}("]
        for t in self.transforms:
            lines.append(f"    {t}")
        lines.append(")")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# PIL-level transforms  (operate on list[Image.Image])
# ---------------------------------------------------------------------------

class ClipColorJitter:
    """
    Randomly jitter brightness, contrast, saturation, hue of all frames.
    Parameters are sampled once and applied identically to every frame
    (preserves temporal colour consistency of the writing gesture).
    """
    def __init__(
        self,
        brightness:  float = 0.0,
        contrast:    float = 0.0,
        saturation:  float = 0.0,
        hue:         float = 0.0,
    ):
        self.brightness  = brightness
        self.contrast    = contrast
        self.saturation  = saturation
        self.hue         = hue

    def _sample(self) -> tuple:
        def _uni(v):
            return random.uniform(max(0.0, 1 - v), 1 + v) if v > 0 else None
        b = _uni(self.brightness)
        c = _uni(self.contrast)
        s = _uni(self.saturation)
        h = random.uniform(-self.hue, self.hue) if self.hue > 0 else None
        return b, c, s, h

    def __call__(self, clip: list[Image.Image]) -> list[Image.Image]:
        b, c, s, h = self._sample()
        out = []
        for img in clip:
            if b is not None: img = TF.adjust_brightness(img, b)
            if c is not None: img = TF.adjust_contrast(img, c)
            if s is not None: img = TF.adjust_saturation(img, s)
            if h is not None: img = TF.adjust_hue(img, h)
            out.append(img)
        return out


class ClipRandomRotation:
    """
    Rotate all frames by the same randomly sampled angle.

    Parameters
    ----------
    degrees : float or (min, max)  — if float, range is (-degrees, +degrees)
    """
    def __init__(self, degrees: float | tuple):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("degrees must be non-negative")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("degrees tuple must have length 2")
            self.degrees = tuple(degrees)

    def __call__(self, clip: list[Image.Image]) -> list[Image.Image]:
        angle = random.uniform(*self.degrees)
        return [img.rotate(angle) for img in clip]


class ClipRandomHorizontalFlip:
    """Flip all frames left-right with probability p."""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, clip: list[Image.Image]) -> list[Image.Image]:
        if random.random() < self.p:
            return [TF.hflip(img) for img in clip]
        return clip


class ClipRandomGrayscale:
    """Convert all frames to 3-channel grayscale with probability p."""
    def __init__(self, p: float = 0.1):
        self.p = p

    def __call__(self, clip: list[Image.Image]) -> list[Image.Image]:
        if random.random() < self.p:
            return [TF.to_grayscale(img, num_output_channels=3) for img in clip]
        return clip


# ---------------------------------------------------------------------------
# ClipToTensor  (PIL list → float tensor)
# ---------------------------------------------------------------------------

class ClipToTensor:
    """
    Convert a list of PIL Images to a float tensor [C, T, H, W] in [0, 1].

    Mirrors WiTA baseline ClipToTensor(channel_nb=3).
    """
    def __init__(self, channel_nb: int = 3, div_255: bool = True):
        self.channel_nb = channel_nb
        self.div_255    = div_255

    def __call__(self, clip: list[Image.Image]) -> torch.Tensor:
        T = len(clip)
        w, h = clip[0].size
        buf = np.zeros((self.channel_nb, T, h, w), dtype=np.float32)
        for t, img in enumerate(clip):
            arr = np.array(img, dtype=np.float32)
            if arr.ndim == 2:
                arr = np.stack([arr] * self.channel_nb, axis=0)
            else:
                arr = arr.transpose(2, 0, 1)          # HWC → CHW
            buf[:, t, :, :] = arr
        tensor = torch.from_numpy(buf)
        if self.div_255:
            tensor = tensor.div(255.0)
        return tensor                                  # [C, T, H, W]


# ---------------------------------------------------------------------------
# ClipNormalize  (tensor in-place)
# ---------------------------------------------------------------------------

class ClipNormalize:
    """
    Normalise a clip tensor [C, T, H, W] by channel mean and std.

    Mirrors WiTA baseline Normalize(mean, std).
    """
    def __init__(self, mean: list[float], std: list[float]):
        self.mean = mean
        self.std  = std

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        # clip: [C, T, H, W]
        mean = torch.tensor(self.mean, dtype=clip.dtype).view(-1, 1, 1, 1)
        std  = torch.tensor(self.std,  dtype=clip.dtype).view(-1, 1, 1, 1)
        return (clip - mean) / std


# ---------------------------------------------------------------------------
# Temporal tensor-level augmentations
# ---------------------------------------------------------------------------

class TemporalCrop:
    """
    Crop a random contiguous sub-sequence and resample back to original T.
    Operates on a tensor [C, T, H, W].
    """
    import torch.nn.functional as _F

    def __init__(self, ratio_range: tuple = (0.75, 1.0), min_frames: int = 8):
        self.ratio_range = ratio_range
        self.min_frames  = min_frames

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        C, T, H, W = clip.shape
        ratio  = random.uniform(*self.ratio_range)
        crop_T = max(self.min_frames, int(T * ratio))
        crop_T = min(crop_T, T)
        if crop_T == T:
            return clip
        start  = random.randint(0, T - crop_T)
        cropped = clip[:, start : start + crop_T]                # [C, crop_T, H, W]
        return F.interpolate(
            cropped.unsqueeze(0), size=(T, H, W),
            mode="trilinear", align_corners=False,
        ).squeeze(0)


class DropFrames:
    """
    Zero out random frames with probability p per frame.
    Operates on tensor [C, T, H, W].  Shape is preserved (no removal).
    """
    def __init__(self, p: float = 0.10):
        self.p = p

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        T    = clip.shape[1]
        mask = torch.rand(T) < self.p
        clip = clip.clone()
        clip[:, mask] = 0.0
        return clip


# ---------------------------------------------------------------------------
# Convenience aliases (keeps the baseline import names working if needed)
# ---------------------------------------------------------------------------
ColorJitter  = ClipColorJitter
RandomRotation = ClipRandomRotation
Normalize    = ClipNormalize
