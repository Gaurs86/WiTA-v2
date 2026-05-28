"""
datasets/skeleton_cache.py — MediaPipe landmark feature extraction + cache.

Stage 1 of the iterative ablation plan.  Builds a per-clip feature cache
where each clip is a sequence of frame-level landmark descriptors:

  21 hand joints × (x, y, z)            = 63
  + first-difference velocity per dim    = 63
  + second-difference accel per dim      = 63
  + per-frame visibility (worst landmark): 1
  total                                  = 190

The sequence is resampled to a uniform T_native frames per clip via linear
interpolation in time so the downstream Conformer sees consistent shapes.

Cache layout (compatible with the existing CachedFeatureDataset by sharing
the same `feats` / `labels` / `lengths` fields):
  {
    "feats":     list[Tensor [T_native, 190]]  fp16 on CPU
    "labels":    list[str]
    "subjects":  list[str]                     subject ID per clip
    "lengths":   LongTensor [N]                same value (T_native) for all
    "out_dim":   int                           190
    "n_native":  int                           T_native
  }
"""

from __future__ import annotations

import io
import os
import time
import urllib.request
import logging
from typing import Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MediaPipe Tasks API model file (shared with hand_crop.py)
# ---------------------------------------------------------------------------

_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


def _ensure_landmarker_model() -> str:
    cache_dir = os.environ.get(
        "WITA_HAND_LANDMARKER_DIR",
        os.path.expanduser("~/.cache/wita_v2"),
    )
    os.makedirs(cache_dir, exist_ok=True)
    model_path = os.path.join(cache_dir, "hand_landmarker.task")
    if not os.path.exists(model_path):
        print(f"[skeleton_cache] Downloading hand landmarker → {model_path}",
              flush=True)
        urllib.request.urlretrieve(_LANDMARKER_MODEL_URL, model_path)
    return model_path


# ---------------------------------------------------------------------------
# Landmark extractor — emits raw 21-joint (x, y, z) per frame
# ---------------------------------------------------------------------------

N_JOINTS = 21


class LandmarkExtractor:
    """
    Minimal MediaPipe wrapper that returns the 21 hand joints per frame.

    Returns a numpy array of shape [21, 3] (x, y, z in [0, 1]) for the
    largest detected hand, or None if no hand was detected.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.3,
        min_tracking_confidence:  float = 0.3,
        max_num_hands: int = 1,
    ):
        try:
            import mediapipe as mp
        except ImportError as e:
            raise ImportError("mediapipe required. pip install mediapipe") from e

        self._mp = mp
        self._api = None
        self.landmarker = None
        self.hands = None

        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError:
            mp_python = mp_vision = None

        if mp_python is not None and mp_vision is not None:
            try:
                model_path = _ensure_landmarker_model()
                base_options = mp_python.BaseOptions(model_asset_path=model_path)
                options = mp_vision.HandLandmarkerOptions(
                    base_options                  = base_options,
                    num_hands                     = max_num_hands,
                    min_hand_detection_confidence = min_detection_confidence,
                    min_hand_presence_confidence  = min_detection_confidence,
                    min_tracking_confidence       = min_tracking_confidence,
                    running_mode                  = mp_vision.RunningMode.IMAGE,
                )
                self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
                self._api = "tasks"
            except Exception as e:
                logger.warning("[LandmarkExtractor] Tasks API init failed (%s)", e)

        if self._api is None:
            solutions = getattr(mp, "solutions", None)
            if solutions is None or not hasattr(solutions, "hands"):
                raise ImportError(
                    "mediapipe missing both Tasks API and solutions.hands"
                )
            self.hands = solutions.hands.Hands(
                static_image_mode        = False,
                max_num_hands            = max_num_hands,
                min_detection_confidence = min_detection_confidence,
                min_tracking_confidence  = min_tracking_confidence,
            )
            self._api = "solutions"

    def close(self):
        if self._api == "tasks" and self.landmarker is not None:
            self.landmarker.close()
        if self._api == "solutions" and self.hands is not None:
            self.hands.close()

    def detect(self, frame: Image.Image) -> Optional[np.ndarray]:
        """Returns [21, 3] (x, y, z) of the largest detected hand, or None."""
        rgb = np.asarray(frame.convert("RGB"))

        if self._api == "tasks":
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            result = self.landmarker.detect(mp_image)
            hand_lists = result.hand_landmarks
            if not hand_lists:
                return None
            # Pick largest by bbox area.
            best = None
            best_area = -1.0
            for hand in hand_lists:
                xs = np.array([lm.x for lm in hand])
                ys = np.array([lm.y for lm in hand])
                area = (xs.max() - xs.min()) * (ys.max() - ys.min())
                if area > best_area:
                    best_area = area
                    best = hand
            if best is None:
                return None
            arr = np.array(
                [[lm.x, lm.y, lm.z] for lm in best],
                dtype=np.float32,
            )
            return arr  # [21, 3]
        else:
            results = self.hands.process(rgb)
            if not results.multi_hand_landmarks:
                return None
            best = None
            best_area = -1.0
            for hand in results.multi_hand_landmarks:
                xs = np.array([lm.x for lm in hand.landmark])
                ys = np.array([lm.y for lm in hand.landmark])
                area = (xs.max() - xs.min()) * (ys.max() - ys.min())
                if area > best_area:
                    best_area = area
                    best = hand
            if best is None:
                return None
            arr = np.array(
                [[lm.x, lm.y, lm.z] for lm in best.landmark],
                dtype=np.float32,
            )
            return arr


# ---------------------------------------------------------------------------
# Frame-level → clip-level feature construction
# ---------------------------------------------------------------------------

def _resample_uniform(seq: np.ndarray, T_target: int) -> np.ndarray:
    """
    Linear interpolation in time to bring `seq` of shape [T_in, *] to
    [T_target, *].  Used when T_in > 0; if T_target == T_in just returns seq.
    """
    T_in = seq.shape[0]
    if T_in == T_target:
        return seq
    if T_in < 2:
        # Repeat the only frame.
        out = np.broadcast_to(seq, (T_target, *seq.shape[1:])).copy()
        return out
    # Resample by interpolating each spatial channel independently.
    src_idx = np.linspace(0, T_in - 1, T_target)
    src_lo  = np.floor(src_idx).astype(np.int32)
    src_hi  = np.minimum(src_lo + 1, T_in - 1)
    frac    = (src_idx - src_lo)[:, None, None] if seq.ndim == 3 else (src_idx - src_lo)[:, None]
    return seq[src_lo] * (1 - frac) + seq[src_hi] * frac


def build_clip_features(
    frame_bytes: list[bytes],
    extractor:   LandmarkExtractor,
    T_native:    int = 32,
    fallback_zero: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Extract per-frame landmarks, then construct the 190-dim per-frame
    descriptor and resample to T_native steps.

    Returns
    -------
    feats : np.ndarray [T_native, 190]  float32
    stats : dict  with detection-rate diagnostics for this clip
    """
    # 1) Per-frame landmark detection
    raw: list[np.ndarray] = []                  # each [21, 3] or None
    n_detected = 0
    n_total    = len(frame_bytes)
    last_valid: Optional[np.ndarray] = None
    visibility: list[float] = []                # 1.0 if detected, else 0.0

    for b in frame_bytes:
        try:
            img = Image.open(io.BytesIO(b))
        except Exception:
            raw.append(last_valid if last_valid is not None else np.zeros((21, 3),
                       dtype=np.float32))
            visibility.append(0.0)
            continue
        lm = extractor.detect(img)
        if lm is None:
            # Fill-forward with the last valid detection; if none yet, zero.
            if last_valid is not None:
                raw.append(last_valid.copy())
            else:
                raw.append(np.zeros((21, 3), dtype=np.float32))
            visibility.append(0.0)
        else:
            raw.append(lm)
            last_valid = lm
            n_detected += 1
            visibility.append(1.0)

    arr = np.stack(raw, axis=0)                  # [T_in, 21, 3]
    vis = np.array(visibility, dtype=np.float32) # [T_in]

    # If we never saw a hand and not falling back, signal upstream.
    if n_detected == 0 and not fallback_zero:
        raise RuntimeError("No hand detected in any frame and fallback disabled.")

    # 2) Resample landmark sequence to T_native frames
    arr  = _resample_uniform(arr, T_native)      # [T_native, 21, 3]
    vis  = _resample_uniform(vis[:, None], T_native).squeeze(-1)  # [T_native]

    # 3) Compute velocity (first difference) and acceleration (second diff)
    # velocity[t] = arr[t] - arr[t-1], padded at t=0 with zeros
    velocity = np.zeros_like(arr)
    velocity[1:] = arr[1:] - arr[:-1]
    accel = np.zeros_like(arr)
    accel[1:] = velocity[1:] - velocity[:-1]

    # 4) Flatten joints → channels:  [T, 21, 3] → [T, 63]
    pos_flat = arr.reshape(T_native, N_JOINTS * 3)
    vel_flat = velocity.reshape(T_native, N_JOINTS * 3)
    acc_flat = accel.reshape(T_native, N_JOINTS * 3)
    vis_col  = vis[:, None]                              # [T, 1]

    feats = np.concatenate([pos_flat, vel_flat, acc_flat, vis_col],
                            axis=1).astype(np.float32)   # [T_native, 190]

    stats = {
        "n_frames_seen":      n_total,
        "n_frames_detected":  n_detected,
        "detect_rate":        n_detected / max(n_total, 1),
    }
    return feats, stats


# ---------------------------------------------------------------------------
# Cache builder
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_skeleton_features(
    samples:    list[tuple],
    out_path:   str,
    T_native:   int = 32,
    dtype:      torch.dtype = torch.float16,
) -> dict:
    """
    Build a skeleton feature cache.

    Parameters
    ----------
    samples : list of (frame_bytes, label, subject_id) tuples — from
              `subject_splits.stream_and_index_with_subjects`.
    out_path : where to save the .pt cache.
    T_native : uniform sequence length to resample every clip to.

    Returns the same dict that was written to disk.
    """
    extractor = LandmarkExtractor()
    feats_list:    list[torch.Tensor] = []
    labels_list:   list[str]          = []
    subjects_list: list[str]          = []
    lengths_list:  list[int]          = []

    total = len(samples)
    log_every = max(1, total // 100)
    t0 = time.time()
    n_detected_total = 0
    n_frames_total   = 0
    n_skipped        = 0

    print(f"[skeleton_cache] Extracting landmarks for {total} clips, "
          f"T_native={T_native}, dtype={dtype}", flush=True)

    for ci, item in enumerate(samples):
        # Support both 2- and 3-tuple sample lists for safety.
        if len(item) == 3:
            frame_bytes, label, subject = item
        else:
            frame_bytes, label = item
            subject = "UNKNOWN"

        if not frame_bytes:
            n_skipped += 1
            continue

        try:
            feats_np, stats = build_clip_features(
                frame_bytes, extractor, T_native=T_native,
            )
        except Exception as e:
            logger.warning("Skipping clip %d (%s): %s", ci, subject, e)
            n_skipped += 1
            continue

        n_detected_total += stats["n_frames_detected"]
        n_frames_total   += stats["n_frames_seen"]

        feats = torch.from_numpy(feats_np).to(dtype).contiguous()
        feats_list.append(feats)
        labels_list.append(label)
        subjects_list.append(subject)
        lengths_list.append(T_native)

        if (ci + 1) % log_every == 0 or (ci + 1) == total:
            elapsed = time.time() - t0
            rate    = (ci + 1) / max(elapsed, 1e-3)
            eta     = (total - (ci + 1)) / max(rate, 1e-3)
            print(
                f"[skeleton_cache] {ci + 1}/{total}  "
                f"({100*(ci+1)/total:5.1f}%)  "
                f"{rate:.1f} clips/s  ETA {eta/60:5.1f} min  "
                f"detect_rate={n_detected_total/max(n_frames_total,1)*100:.1f}%  "
                f"skipped={n_skipped}",
                flush=True,
            )

    extractor.close()

    cache = {
        "feats":      feats_list,
        "labels":     labels_list,
        "subjects":   subjects_list,
        "lengths":    torch.tensor(lengths_list, dtype=torch.long),
        "out_dim":    190,
        "n_native":   T_native,
        "frame_detect_rate":
            n_detected_total / max(n_frames_total, 1),
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(cache, out_path)
    mb = sum(f.numel() * f.element_size() for f in feats_list) / 1e6
    print(f"[skeleton_cache] Saved {len(feats_list)} clips → {out_path}  "
          f"({mb:.1f} MB, frame_detect_rate="
          f"{n_detected_total/max(n_frames_total,1)*100:.1f}%)", flush=True)
    return cache
