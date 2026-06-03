"""
build_cache.py — pre-decode WiTA clips into uint8 frame tensors.

CER-NEUTRAL speedup.  The cache stores the EXACT output of the paper's
decode + resize + temporal-cap pipeline (AirTypingDataset.decode_resize_frames).
At train time, read_images loads the cached uint8 array and reconstructs
PIL frames via Image.fromarray (lossless round-trip for uint8 RGB), then
runs the UNCHANGED augmentation pipeline.  Model input is bit-identical
to the JPEG path -> no effect on CER.  It only removes the per-epoch
libjpeg decode + resize that bottlenecks the 4-vCPU data loader.

Usage:
  # Build the cache for all three splits (run once, ~20-30 min):
  python build_cache.py \
      --data_type=english --img_size=112 --max_frames=64 \
      --cache_dir=$HOME/wita-cache \
      --data_path_train=$HOME/wita-data/english/train \
      --data_path_val=$HOME/wita-data/english/val \
      --data_path_test=$HOME/wita-data/english/test \
      --n_workers=4

  # Verify bit-identity on a sample (run before trusting the cache):
  python build_cache.py --verify --cache_dir=$HOME/wita-cache \
      --data_type=english --img_size=112 --max_frames=64 \
      --data_path_train=$HOME/wita-data/english/train ...
"""

import os
import sys
import time
import argparse
import numpy as np

from options import AirTypingOptions
from data    import AirTypingDataset


# ----------------------------------------------------------------------
# Worker globals (fork-initialised)
# ----------------------------------------------------------------------

_DS = None          # AirTypingDataset for the split being built
_CACHE_DIR = None


def _init_worker(opts_dict, data_path, cache_dir):
    global _DS, _CACHE_DIR
    # Rebuild a minimal opts namespace in the worker.
    ns = argparse.Namespace(**opts_dict)
    _DS = AirTypingDataset(ns, data_path)
    _CACHE_DIR = cache_dir


def _build_one(index):
    """Decode+resize+cap one clip and write its uint8 cache.  Returns a
    small status dict."""
    global _DS, _CACHE_DIR
    video_dir = _DS.video_list[index]
    cpath = _DS._cache_path(video_dir)
    if os.path.isfile(cpath):
        return {"existed": True}
    try:
        frames = _DS.decode_resize_frames(index)          # list of PIL
        arr = np.stack([np.asarray(f) for f in frames])    # [T, H, W, 3] uint8
        if arr.ndim != 4 or arr.shape[-1] != 3 or arr.dtype != np.uint8:
            return {"skipped": True,
                    "err": f"unexpected array {arr.shape} {arr.dtype} "
                           f"(non-RGB frame?) for {video_dir}"}
        tmp = cpath + ".tmp.npy"
        np.save(tmp, arr)
        os.replace(tmp, cpath)                             # atomic
        return {"written": True, "T": int(arr.shape[0]),
                "bytes": int(arr.nbytes)}
    except Exception as e:
        return {"skipped": True, "err": f"{type(e).__name__}: {e} @ {video_dir}"}


# ----------------------------------------------------------------------

def _opts_to_dict(opts):
    """Pull the dataset-relevant fields into a picklable dict."""
    keys = ["data_type", "data_augment", "max_frames", "img_size", "cache_dir"]
    d = {}
    for k in keys:
        d[k] = getattr(opts, k, None)
    # AirTypingDataset only reads these; data_augment irrelevant for the
    # decode pipeline but harmless.  Force no-aug so workers don't build
    # the aug transforms (cheaper).
    d["data_augment"] = False
    return d


def build_split(opts, data_path, cache_dir, n_workers, log_every=200):
    import multiprocessing as mp
    os.makedirs(cache_dir, exist_ok=True)
    # Build a dataset once in the main process to count + sanity check.
    probe = AirTypingDataset(argparse.Namespace(**_opts_to_dict(opts)), data_path)
    n = len(probe)
    print(f"[build_cache] {data_path}: {n} clips -> {cache_dir}", flush=True)
    if n == 0:
        return {"n": 0}

    ctx = mp.get_context("fork")
    t0 = time.time()
    n_written = n_existed = n_skipped = 0
    total_bytes = 0
    with ctx.Pool(processes=n_workers, initializer=_init_worker,
                  initargs=(_opts_to_dict(opts), data_path, cache_dir)) as pool:
        for i, r in enumerate(pool.imap_unordered(_build_one, range(n), chunksize=8)):
            if r.get("existed"): n_existed += 1
            elif r.get("written"):
                n_written += 1; total_bytes += r.get("bytes", 0)
            else:
                n_skipped += 1
                if r.get("err"):
                    print(f"  skip: {r['err']}", flush=True)
            done = i + 1
            if done % log_every == 0 or done == n:
                el = time.time() - t0
                rate = done / max(el, 1e-3)
                eta = (n - done) / max(rate, 1e-3) / 60.0
                print(f"  [{done}/{n}] {rate:.1f} clips/s  ETA {eta:5.1f} min  "
                      f"written={n_written} reused={n_existed} skipped={n_skipped}  "
                      f"{total_bytes/1e9:.1f} GB", flush=True)
    print(f"[build_cache] done {data_path}: written={n_written} "
          f"reused={n_existed} skipped={n_skipped} "
          f"elapsed={time.time()-t0:.0f}s", flush=True)
    return {"n": n, "written": n_written, "existed": n_existed,
            "skipped": n_skipped, "bytes": total_bytes}


# ----------------------------------------------------------------------
# Verification: cached frames must be bit-identical to JPEG decode
# ----------------------------------------------------------------------

def verify_split(opts, data_path, cache_dir, n_check=20):
    from PIL import Image
    ds_dict = _opts_to_dict(opts)
    ds_dict["cache_dir"] = cache_dir
    ds = AirTypingDataset(argparse.Namespace(**ds_dict), data_path)
    n = len(ds)
    idxs = list(range(0, n, max(n // n_check, 1)))[:n_check]
    print(f"[verify] checking {len(idxs)} clips from {data_path}", flush=True)
    all_ok = True
    for idx in idxs:
        # Ground truth: fresh JPEG decode.
        gt_frames = ds.decode_resize_frames(idx)
        gt = np.stack([np.asarray(f) for f in gt_frames])
        # Cached path.
        cpath = ds._cache_path(ds.video_list[idx])
        if not os.path.isfile(cpath):
            print(f"  idx {idx}: NO CACHE FILE ({cpath})", flush=True)
            all_ok = False; continue
        cached = np.load(cpath)
        # Reconstruct exactly what read_images would feed the augmentation.
        recon = np.stack([np.asarray(Image.fromarray(cached[i]))
                          for i in range(cached.shape[0])])
        if gt.shape != recon.shape:
            print(f"  idx {idx}: SHAPE MISMATCH jpeg={gt.shape} cache={recon.shape}",
                  flush=True)
            all_ok = False; continue
        if not np.array_equal(gt, recon):
            ndiff = int((gt != recon).sum())
            print(f"  idx {idx}: PIXEL MISMATCH ({ndiff} differing values)",
                  flush=True)
            all_ok = False; continue
        print(f"  idx {idx}: OK  shape={gt.shape}", flush=True)
    print(("[verify] ALL CLIPS BIT-IDENTICAL -- cache is CER-neutral."
           if all_ok else
           "[verify] MISMATCHES FOUND -- do NOT train on this cache."),
          flush=True)
    return all_ok


# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Extend the paper's options with a couple of build-only flags.
    base = AirTypingOptions()
    base.parser.add_argument("--n_workers", type=int, default=4,
                             help="parallel decode workers for cache build")
    base.parser.add_argument("--verify", action="store_true",
                             help="verify cache == JPEG decode, then exit")
    base.parser.add_argument("--n_check", type=int, default=20,
                             help="clips per split to check in --verify mode")
    opts = base.parse()

    assert opts.cache_dir, "--cache_dir is required"

    splits = [p for p in (opts.data_path_train, opts.data_path_val,
                          opts.data_path_test) if p and os.path.isdir(p)]
    if not splits:
        print("No valid split dirs found.", file=sys.stderr); sys.exit(2)

    if opts.verify:
        ok = True
        for p in splits:
            ok &= verify_split(opts, p, opts.cache_dir, n_check=opts.n_check)
        sys.exit(0 if ok else 1)

    grand = {"written": 0, "existed": 0, "skipped": 0, "bytes": 0}
    for p in splits:
        r = build_split(opts, p, opts.cache_dir, opts.n_workers)
        for k in ("written", "existed", "skipped", "bytes"):
            grand[k] += r.get(k, 0)
    print(f"\n[build_cache] GRAND TOTAL  written={grand['written']} "
          f"reused={grand['existed']} skipped={grand['skipped']}  "
          f"{grand['bytes']/1e9:.1f} GB")
    print("Next: verify, then train with --cache_dir set.")
    print(f"  python build_cache.py --verify --cache_dir={opts.cache_dir} ...")
