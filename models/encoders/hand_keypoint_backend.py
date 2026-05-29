"""
models/encoders/hand_keypoint_backend.py — pluggable hand-keypoint backends.

All backends emit the SAME shape MediaPipe's HandLandmarker emits:
    detect(pil_image) -> np.ndarray [21, 3]  or  None
        x, y : normalised image coords in [0, 1]
        z    : MediaPipe's relative depth (or 0.0 for 2D-only backends)

That makes them drop-in replacements for `LandmarkExtractor.detect` in
`build_clip_features`, which is exactly what the HRNet-swap experiment
needs: substitute the keypoint source, keep the rest of the feature
pipeline identical.

Three backends:
  * MediaPipeBackend(name='mediapipe_default',   min_conf=0.5)
  * MediaPipeBackend(name='mediapipe_sensitive', min_conf=0.2)
  * RTMPoseHandBackend()    — requires mmpose; raises ImportError if absent

The RTMPose backend converts pixel coords to normalised coords; sets z=0
since the 2D model has no depth.  Joint order: RTMPose's hand5 dataset
follows the same 21-keypoint MediaPipe convention (index fingertip = 8).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PIL import Image

from ...datasets.skeleton_cache import LandmarkExtractor  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class KeypointBackend:
    """Common interface used by `build_clip_features` and the eval scripts."""

    name: str = "abstract"

    def detect(self, frame: Image.Image) -> Optional[np.ndarray]:
        """Return [21, 3] (x, y, z) normalised to [0, 1], or None on miss."""
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


# ---------------------------------------------------------------------------
# MediaPipe (default / sensitive)
# ---------------------------------------------------------------------------

class MediaPipeBackend(KeypointBackend):
    """
    Thin wrapper around the existing `LandmarkExtractor` so we can vary
    `min_detection_confidence` / `min_tracking_confidence` without touching
    the original extractor's call sites.

    name='mediapipe_default'   uses Stage 1 v2's exact settings (0.3 / 0.3).
    name='mediapipe_sensitive' bumps recall (0.2 / 0.2).
    """

    def __init__(
        self,
        name:      str = "mediapipe_default",
        det_conf:  float = 0.3,
        track_conf: float = 0.3,
    ):
        super().__init__()
        self.name = name
        self._extractor = LandmarkExtractor(
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
            max_num_hands=1,
        )

    def detect(self, frame: Image.Image) -> Optional[np.ndarray]:
        return self._extractor.detect(frame)

    def close(self) -> None:
        self._extractor.close()


def build_mediapipe(name: str = "mediapipe_default",
                    conf: float = 0.3) -> MediaPipeBackend:
    """Factory for either the default or sensitive MediaPipe backend."""
    return MediaPipeBackend(name=name, det_conf=conf, track_conf=conf)


# ---------------------------------------------------------------------------
# RTMPose (HRNet-equivalent hand model via MMPose)
# ---------------------------------------------------------------------------

class RTMPoseHandBackend(KeypointBackend):
    """
    Top-down hand keypoint detector using MMPose's RTMPose-M hand5 model.

    Two-stage pipeline:
      1. Bounding-box detector — we use the full image as the bbox
         (single-hand assumption, same as MediaPipe in our config).  This
         is suboptimal but cheap; it lets the keypoint model lock onto the
         dominant hand without an external person/hand detector.
      2. Pose estimator — RTMPose-M hand5, 256x256 input, 21 keypoints
         (MediaPipe convention).

    name = 'rtmpose_hand'
    """

    name = "rtmpose_hand"

    def __init__(
        self,
        pose2d_alias:     str = "hand",            # MMPose model alias
        device:           str = "cpu",
        score_threshold:  float = 0.3,
    ):
        super().__init__()
        try:
            from mmpose.apis import MMPoseInferencer  # noqa
        except ImportError as e:
            raise ImportError(
                "RTMPoseHandBackend requires mmpose.  Install with:\n"
                "    pip install -U openmim\n"
                "    mim install mmengine 'mmcv>=2.0.1' 'mmdet>=3.1.0' "
                "'mmpose>=1.3.0'\n"
                "(MMPose's `pose2d=\"hand\"` alias resolves to the recommended "
                "RTMPose-M hand5 model.)"
            ) from e
        self._alias    = pose2d_alias
        self._device   = device
        self._score_thr = score_threshold
        self._inf       = None     # lazy-init on first detect()

    def _ensure_model(self):
        if self._inf is None:
            from mmpose.apis import MMPoseInferencer
            self._inf = MMPoseInferencer(
                pose2d=self._alias, device=self._device,
            )

    def detect(self, frame: Image.Image) -> Optional[np.ndarray]:
        self._ensure_model()
        rgb = np.asarray(frame.convert("RGB"))
        H, W = rgb.shape[:2]
        # MMPoseInferencer accepts a numpy array directly.  Returns a
        # generator of result dicts; consume the first.
        gen = self._inf(rgb, return_vis=False, show=False)
        try:
            out = next(gen)
        except StopIteration:
            return None
        preds = out.get("predictions", [])
        if not preds or not preds[0]:
            return None
        # preds[0] is a list of per-instance dicts.  Pick the highest
        # bbox_score instance (largest hand).
        instances = preds[0]
        best = max(instances, key=lambda d: d.get("bbox_score", 0.0))
        kps_xy   = np.asarray(best.get("keypoints"),         dtype=np.float32)
        kp_score = np.asarray(best.get("keypoint_scores"),   dtype=np.float32)
        if kps_xy.shape[0] != 21 or kp_score.mean() < self._score_thr:
            return None
        # MMPose returns absolute pixel coords; normalise to [0, 1].
        x_norm = kps_xy[:, 0] / max(W, 1)
        y_norm = kps_xy[:, 1] / max(H, 1)
        z      = np.zeros_like(x_norm)            # 2D-only model
        return np.stack([x_norm, y_norm, z], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Factory dict for the orchestration scripts
# ---------------------------------------------------------------------------

def all_default_backends(device: str = "cpu") -> dict[str, KeypointBackend]:
    """
    Return the three backends the HRNet-swap experiment compares.  The
    RTMPose entry is omitted (with a warning logged) when mmpose is not
    importable so cell pipelines can still run a useful 2/3 comparison.
    """
    out: dict[str, KeypointBackend] = {
        "mediapipe_default":   build_mediapipe("mediapipe_default",   0.3),
        "mediapipe_sensitive": build_mediapipe("mediapipe_sensitive", 0.2),
    }
    try:
        out["rtmpose_hand"] = RTMPoseHandBackend(device=device)
    except ImportError as e:
        logger.warning("RTMPose backend unavailable (%s) — continuing with "
                       "MediaPipe-only comparison.", e)
    return out
