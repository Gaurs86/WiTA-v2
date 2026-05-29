"""
scripts/extract_phw_kim_keypoints.py — HRNet-swap experiment §3.3.

Re-extracts hand keypoints for clips belonging ONLY to PHW and KIM,
using multiple backends (MediaPipe default, MediaPipe sensitive, RTMPose).
Each backend writes its own skeleton-feature cache compatible with the
Stage 1 v2 model — same 190-d per-frame layout, same T_native=32 resample,
same fallback-on-miss policy.

Files written:
    caches/landmarks_mediapipe_default_phw_kim.pt
    caches/landmarks_mediapipe_sensitive_phw_kim.pt
    caches/landmarks_rtmpose_hand_phw_kim.pt   (if mmpose available)

Usage (Kaggle / Colab):
    python scripts/extract_phw_kim_keypoints.py \
        --target-signers PHW KIM \
        --out-dir        /kaggle/working/caches/hrnet_swap \
        --device         cpu

The full WiTA-English manifest is 39 signers x 2 zips = 78 zips.  We
pre-filter to the 2 target signers' 4 zips before downloading anything,
saving ~95% of the streaming time.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import shutil
import sys
import zipfile
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger("extract_phw_kim")


# ---------------------------------------------------------------------------
# Targeted streamer (subset of subject_splits.stream_and_index_with_subjects)
# ---------------------------------------------------------------------------

def stream_signers(
    target_signers: list[str],
    hf_repo_id:     str = "yewon816/WiTA",
    lang:           str = "english",
    max_frames:     int = 64,
    download_dir:   str = "/tmp/wita_hrnet_downloads",
) -> list[tuple[list[bytes], str, str]]:
    """
    Same return shape as stream_and_index_with_subjects, but filters HF
    zip files to only those whose subject prefix is in `target_signers`.
    Network cost is `len(target_signers) * 2` zip downloads, not 78.
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)) + '/..')
    from wita_v2.datasets.dataset         import _index_zip
    from wita_v2.datasets.subject_splits  import subject_id_from_zip
    try:
        from huggingface_hub import list_repo_files, hf_hub_download
    except ImportError as e:
        raise ImportError("pip install huggingface_hub") from e

    target_set = {s.upper() for s in target_signers}

    all_files = list(list_repo_files(hf_repo_id, repo_type="dataset"))
    lang_filters = {
        "english": lambda f: f.endswith(".zip") and "kor" not in f.lower(),
        "korean":  lambda f: f.endswith(".zip") and ("kor" in f.lower() or "korean" in f.lower()),
        "both":    lambda f: f.endswith(".zip"),
    }
    zip_files = sorted(filter(lang_filters[lang], all_files))
    zip_files = [
        z for z in zip_files
        if subject_id_from_zip(z) in target_set
    ]

    logger.info(
        "[stream_signers] %d zips to download for %d target signers (%s)",
        len(zip_files), len(target_set), sorted(target_set),
    )
    os.makedirs(download_dir, exist_ok=True)

    samples: list[tuple[list[bytes], str, str]] = []
    for i, zip_name in enumerate(zip_files):
        subj = subject_id_from_zip(zip_name)
        logger.info("[%d/%d] %s  subject=%s", i + 1, len(zip_files), zip_name, subj)
        local = hf_hub_download(
            repo_id=hf_repo_id, filename=zip_name, repo_type="dataset",
            local_dir=download_dir, local_dir_use_symlinks=False,
        )
        try:
            with zipfile.ZipFile(local, "r") as zf:
                batch = _index_zip(zf, lang, max_frames)
            for frames, label in batch:
                samples.append((frames, label, subj))
            logger.info("  -> %d clips  (total: %d)", len(batch), len(samples))
        except Exception as e:
            logger.error("Failed to process %s: %s", zip_name, e)
        finally:
            if os.path.exists(local):
                os.remove(local)
    return samples


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--target-signers', nargs='+', default=['PHW', 'KIM'])
    p.add_argument('--out-dir', default='caches/hrnet_swap')
    p.add_argument('--hf-repo-id', default='yewon816/WiTA')
    p.add_argument('--hf-lang',    default='english')
    p.add_argument('--max-frames', type=int, default=64)
    p.add_argument('--t-native',   type=int, default=32)
    p.add_argument('--device',     default='cpu',
        help='Device for RTMPose inference (cpu / cuda).')
    p.add_argument('--skip-rtmpose', action='store_true',
        help='Do not attempt the RTMPose backend.')
    p.add_argument('--log-level', default='INFO')
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s  %(levelname)-7s  %(name)s — %(message)s',
    )
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Stream just the target signers' zips once.  All backends consume
    #    the same in-memory `samples` list so we don't re-download.
    samples = stream_signers(
        target_signers=args.target_signers,
        hf_repo_id=args.hf_repo_id,
        lang=args.hf_lang,
        max_frames=args.max_frames,
    )
    logger.info("Got %d clips across signers %s.",
                len(samples), sorted({s[2] for s in samples}))

    # 2) Build the backend list.
    from wita_v2.models.encoders.hand_keypoint_backend import (
        build_mediapipe, RTMPoseHandBackend,
    )
    from wita_v2.datasets.skeleton_cache import extract_skeleton_features

    backends: dict[str, object] = {
        "mediapipe_default":   build_mediapipe("mediapipe_default",   0.3),
        "mediapipe_sensitive": build_mediapipe("mediapipe_sensitive", 0.2),
    }
    if not args.skip_rtmpose:
        try:
            backends["rtmpose_hand"] = RTMPoseHandBackend(device=args.device)
        except ImportError as e:
            logger.warning("Skipping rtmpose_hand: %s", e)

    # 3) For each backend, run the same skeleton-feature pipeline.
    for name, backend in backends.items():
        out_path = os.path.join(args.out_dir, f"landmarks_{name}_phw_kim.pt")
        if os.path.exists(out_path):
            logger.info("[%s] cache exists at %s — skipping.", name, out_path)
            continue
        logger.info("[%s] extracting -> %s", name, out_path)
        extract_skeleton_features(
            samples=samples, out_path=out_path,
            T_native=args.t_native, dtype=torch.float16,
            extractor=backend, backend_name=name,
        )

    # 4) Sanity print: per-backend frame detection rate.
    print('\n=== HRNet-swap extraction summary ===')
    for name in backends:
        out_path = os.path.join(args.out_dir, f"landmarks_{name}_phw_kim.pt")
        if not os.path.exists(out_path):
            continue
        c = torch.load(out_path, map_location='cpu', weights_only=False)
        print(
            f'  {name:<22s}  '
            f'clips={len(c["feats"])}  '
            f'detect_rate={c.get("frame_detect_rate", 0)*100:.1f}%  '
            f'out={out_path}'
        )


if __name__ == '__main__':
    main()
