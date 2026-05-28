"""
datasets/hand_crop.py — MediaPipe-based hand-region cropping.

Motivation
----------
V-L foundation models (CLIP / SigLIP / X-CLIP) tokenize input into 14×14 or
16×16 patches at 224×224 — each patch covers ~256 pixels.  The WiTA fingertip
trail moves ~10–30 pixels per frame on a 224×224 source frame.  Patch-scale
spatial averaging destroys the signal the recognizer needs.

Hand cropping concentrates the writing region inside the 224×224 input fed
to the V-L model.  After cropping a ~100×100-pixel bounding box around the
hand and resizing back to 224×224, each 16×16 patch now covers ~7×7 source
pixels — fine enough to resolve fingertip motion across frames.

Pipeline
--------
For each clip of T frames:
  1. Run MediaPipe Hands on every frame, collect detected bboxes
  2. If at least one detection: union all bboxes (the writing area)
     - This preserves the full trajectory, unlike a per-frame moving crop
  3. Pad the union bbox by `padding_ratio` to give the model some context
  4. Square-up the bbox so we don't distort aspect ratio on resize
  5. Clip to image bounds
  6. Crop every frame to the same bbox, resize to target_size

  If no hand detected in ANY frame: fall back to a centered crop.

Real-world deployment
---------------------
This module is meant to run at BOTH train and inference time.  Same code
path, same MediaPipe model — no train/test mismatch.  MediaPipe Hands runs
at 30+ FPS on CPU, so the added latency is acceptable for real-time use.
"""

from __future__ import annotations
import logging
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class HandCropper:
    """
    Per-clip hand-region cropper.

    Parameters
    ----------
    target_size      : H==W of the cropped+resized output frames (224 for X-CLIP / SigLIP-224)
    padding_ratio    : extra padding around the detected bbox, as fraction of
                       bbox side length.  0.3 ≈ 30% margin on each side.
    min_detection_confidence : MediaPipe Hands hyperparameter; lower = more
                       permissive detection (helps with motion blur).
    max_num_hands    : 1 for single-handed writing (WiTA is one hand).
    require_any_detection : if False and NO hand is detected in any frame
                       of a clip, fall back to a centered crop.  If True,
                       raise instead (useful for debugging data quality).
    """

    def __init__(
        self,
        target_size: int = 224,
        padding_ratio: float = 0.3,
        min_detection_confidence: float = 0.3,
        min_tracking_confidence:  float = 0.3,
        max_num_hands: int = 1,
        require_any_detection: bool = False,
    ):
        try:
            import mediapipe as mp
        except ImportError as e:
            raise ImportError(
                "mediapipe required for hand cropping. "
                "pip install mediapipe"
            ) from e

        self.target_size           = target_size
        self.padding_ratio         = padding_ratio
        self.require_any_detection = require_any_detection
        self._mp                   = mp

        # MediaPipe Hands instance.  static_image_mode=False enables tracking
        # across calls within the same process — faster for sequential frames.
        self.hands = mp.solutions.hands.Hands(
            static_image_mode        = False,
            max_num_hands            = max_num_hands,
            min_detection_confidence = min_detection_confidence,
            min_tracking_confidence  = min_tracking_confidence,
        )

        # Stats — useful diagnostic to print after extracting a whole dataset.
        self.n_clips_seen      = 0
        self.n_clips_no_hand   = 0       # 0 detections in any frame → fallback
        self.n_clips_partial   = 0       # <50% of frames had a detection
        self.n_frames_detected = 0
        self.n_frames_total    = 0

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self.hands is not None:
            self.hands.close()
            self.hands = None

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _detect_bbox(self, frame: Image.Image) -> Optional[tuple[int, int, int, int]]:
        """
        Return (x0, y0, x1, y1) bbox in pixel coords of the largest detected
        hand in `frame`, or None if no hand found.
        """
        rgb = np.asarray(frame.convert("RGB"))
        # MediaPipe expects RGB uint8 — no extra conversion needed.
        results = self.hands.process(rgb)
        if not results.multi_hand_landmarks:
            return None

        h, w = rgb.shape[:2]
        # If multi_num_hands>1, pick the largest bbox.
        best = None
        best_area = -1
        for hand in results.multi_hand_landmarks:
            xs = [lm.x for lm in hand.landmark]
            ys = [lm.y for lm in hand.landmark]
            x0 = int(max(0.0, min(xs)) * w)
            y0 = int(max(0.0, min(ys)) * h)
            x1 = int(min(1.0, max(xs)) * w)
            y1 = int(min(1.0, max(ys)) * h)
            area = max(0, x1 - x0) * max(0, y1 - y0)
            if area > best_area:
                best_area = area
                best = (x0, y0, x1, y1)
        return best

    @staticmethod
    def _union(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
        x0 = min(b[0] for b in boxes)
        y0 = min(b[1] for b in boxes)
        x1 = max(b[2] for b in boxes)
        y1 = max(b[3] for b in boxes)
        return (x0, y0, x1, y1)

    def _expand_and_square(
        self,
        bbox: tuple[int, int, int, int],
        frame_size: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        """
        Pad bbox by padding_ratio (relative to its side), square-up by
        expanding the shorter side to match the longer one, then clip to
        image bounds.
        """
        x0, y0, x1, y1 = bbox
        w_img, h_img = frame_size

        # Pad
        bw = x1 - x0
        bh = y1 - y0
        pad_x = int(bw * self.padding_ratio)
        pad_y = int(bh * self.padding_ratio)
        x0 -= pad_x; x1 += pad_x
        y0 -= pad_y; y1 += pad_y

        # Square-up around the center
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        side = max(x1 - x0, y1 - y0)
        half = side // 2
        x0, x1 = cx - half, cx + half
        y0, y1 = cy - half, cy + half

        # Clip to image
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(w_img, x1)
        y1 = min(h_img, y1)
        # If we got clipped asymmetrically, drop back to whatever square fits.
        side = min(x1 - x0, y1 - y0)
        if side <= 0:
            # Degenerate; fall back to a centered square.
            side = min(w_img, h_img)
            x0 = (w_img - side) // 2
            y0 = (h_img - side) // 2
            x1 = x0 + side
            y1 = y0 + side
        return (x0, y0, x1, y1)

    def _center_square(self, frame_size: tuple[int, int]) -> tuple[int, int, int, int]:
        w_img, h_img = frame_size
        side = min(w_img, h_img)
        x0 = (w_img - side) // 2
        y0 = (h_img - side) // 2
        return (x0, y0, x0 + side, y0 + side)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def crop_clip(self, frames: list[Image.Image]) -> list[Image.Image]:
        """
        Crop every frame in a clip to the same hand-region bbox, then resize
        all frames to (target_size, target_size).

        Returns
        -------
        list[PIL.Image]  — same length as input, all of size (target_size, target_size).
        """
        if not frames:
            return frames

        bboxes: list[Optional[tuple[int, int, int, int]]] = [
            self._detect_bbox(f) for f in frames
        ]
        valid = [b for b in bboxes if b is not None]

        self.n_clips_seen      += 1
        self.n_frames_total    += len(frames)
        self.n_frames_detected += len(valid)

        if not valid:
            # No hand seen anywhere in this clip.
            self.n_clips_no_hand += 1
            if self.require_any_detection:
                raise RuntimeError(
                    "No hand detected in any frame of clip; "
                    "set require_any_detection=False to enable center-crop fallback."
                )
            crop_box = self._center_square(frames[0].size)
        else:
            if len(valid) < max(1, len(frames) // 2):
                self.n_clips_partial += 1
            union = self._union(valid)
            crop_box = self._expand_and_square(union, frames[0].size)

        ts = self.target_size
        return [
            f.crop(crop_box).resize((ts, ts), Image.BILINEAR)
            for f in frames
        ]

    def stats(self) -> dict:
        """Return summary stats for the run."""
        return {
            "clips_seen":      self.n_clips_seen,
            "clips_no_hand":   self.n_clips_no_hand,
            "clips_partial":   self.n_clips_partial,
            "frames_detected": self.n_frames_detected,
            "frames_total":    self.n_frames_total,
            "frame_detect_rate": (
                self.n_frames_detected / self.n_frames_total
                if self.n_frames_total else 0.0
            ),
        }
