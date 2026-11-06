#!/usr/bin/env python3
"""Compare a depth backend with torch-fp32 on the same gray frames, through the full live pipeline.

Each frame goes through ucslam.live_depth.LiveDepth as the depth node runs it: net_input, the network of the
backend, downscale_map (block average, holes as 0, float16). The maps of the backend (b) and of torch-fp32 (a)
are compared over the pixels valid in both:

    d = log z_b - log z_a          median d, median |d|, p99 |d|, max |d|
    mask agreement                 share of map pixels where both are valid or both are holes
    time per frame                 net_ms of LiveDepth (processor, network, resample) per backend, after warm-up

Pass limits (docs/ros2_design.md, backend parity): |median d| <= 0.005 and p99 |d| <= 0.02, and a mask agreement
of at least 0.999 on every frame (a backend that makes NaN or inf turns them into holes, which d does not see).
A backend that fails them needs its own depth constants: refit the domain with that backend.

The two backends run one after the other (reference first, then freed), so one model is in GPU memory at a
time and the timings do not disturb each other.

    python scripts/jetson/depth_parity.py --backend trt-fp16 --engine da2b_518x910_fp16n.engine \\
        --images data/openloris_raw/office1-1/color --n 30 --json parity_trt_fp16n.json

Exit status: 0 pass, 1 parity fail, 2 nothing compared (usage error, missing images or engine, a backend that
could not load or run).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import sys
import time
from typing import Dict, List, Optional

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

MEDIAN_LIMIT = 0.005
P99_LIMIT = 0.02
MASK_LIMIT = 0.999
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".pgm")


def list_frames(paths: List[str], n: int) -> List[str]:
    """The image files of folders and file arguments, sorted. n > 0: n frames evenly spaced over the list."""
    files: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(f for f in glob.glob(os.path.join(p, "*")) if f.lower().endswith(IMAGE_EXT))
        elif os.path.isfile(p):
            files.append(p)
        else:
            raise FileNotFoundError(p)
    if n > 0 and len(files) > n:
        idx = np.linspace(0, len(files) - 1, n).round().astype(int)
        files = [files[i] for i in idx]
    return files


def read_gray(path: str) -> np.ndarray:
    """The frame as the SLAM node sends it: 8-bit gray (cv2 conversion of a color frame, as the depth cache)."""
    import cv2
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def run_backend(backend: str, engine: Optional[str], imgs: List[np.ndarray], model: str, device: str,
                warmup: int) -> Dict:
    import torch

    from ucslam.live_depth import LiveDepth
    live = LiveDepth(model or None, device, backend=backend, engine=engine)
    for _ in range(warmup):
        live(imgs[0])
    maps, net_ms = [], []
    for g in imgs:
        zmap, desc, ms = live(g)
        maps.append(zmap)
        net_ms.append(ms)
    out = {"backend": backend, "engine": engine, "model": live.model, "source": str(getattr(live.net, "source", "")),
           "image_wh": [int(imgs[0].shape[1]), int(imgs[0].shape[0])], "map_hw": list(maps[0].shape),
           "maps": maps, "net_ms": net_ms}
    if hasattr(live.net, "input_hw"):
        out["input_hw"] = list(live.net.input_hw)
    if hasattr(live.net, "sidecar"):   # the build record of the engine (precision, tf32, TensorRT version)
        out["engine_record"] = live.net.sidecar
    out["processor"] = type(getattr(live.net, "processor", None)).__name__
    out["processor_warning"] = live.processor_warning   # both backends share the processor, so parity cannot see it
    del live
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def compare(a: np.ndarray, b: np.ndarray) -> Dict:
    za, zb = a.astype(np.float32), b.astype(np.float32)
    va, vb = za > 0, zb > 0
    both = va & vb
    d = np.log(zb[both]) - np.log(za[both])
    return {"n_valid": int(both.sum()), "mask_agree": float(np.mean(va == vb)), "d": d}


def stats(d: np.ndarray) -> Dict:
    if d.size == 0:
        return {"median_d": float("nan"), "median_abs_d": float("nan"), "p99_abs_d": float("nan"),
                "max_abs_d": float("nan"), "mean_d": float("nan")}
    ad = np.abs(d)
    return {"median_d": float(np.median(d)), "median_abs_d": float(np.median(ad)),
            "p99_abs_d": float(np.percentile(ad, 99)), "max_abs_d": float(ad.max()), "mean_d": float(np.mean(d))}


def pct(v: List[float], q: float) -> float:
    return float(np.percentile(np.asarray(v, np.float64), q)) if v else float("nan")


def passes(pooled: Dict, mask_agree_min: float) -> bool:
    """The pass test. NaN (no valid pixel) fails."""
    return bool(abs(pooled["median_d"]) <= MEDIAN_LIMIT and pooled["p99_abs_d"] <= P99_LIMIT
                and mask_agree_min >= MASK_LIMIT)


def usage_error(args) -> Optional[str]:
    """The argument problems found before any model loads, or None."""
    from ucslam.trt_depth import BACKENDS
    if args.backend not in BACKENDS or args.backend == "torch-fp32":
        return f"--backend must be one of {BACKENDS[1:]}"
    if args.backend.startswith("trt-"):
        if not args.engine:
            return f"--backend {args.backend} needs --engine (a TensorRT engine file)"
        if not os.path.isfile(args.engine):
            return f"engine file not found: {args.engine}"
    elif args.engine:
        return f"--engine is only for trt-* backends, not {args.backend}"
    return None


def main(argv=None) -> int:
    """Exit status 0 pass, 1 parity fail, 2 any error (also one this script did not foresee)."""
    try:
        return _main(argv)
    except Exception:
        import traceback
        traceback.print_exc()
        print("error: nothing or not all compared", file=sys.stderr)
        return 2


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--backend", required=True, help="backend to test: torch-fp16, trt-fp16 or trt-fp32")
    ap.add_argument("--engine", default="", help="TensorRT engine file for trt-*")
    ap.add_argument("--images", nargs="+", required=True, help="image folders or files (gray or color)")
    ap.add_argument("--n", type=int, default=30, help="frames evenly spaced over the list, 0: all")
    ap.add_argument("--model", default="", help="model key, empty: ucslam.depth_model of configs/default.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=int, default=3, help="untimed runs per backend before the frames")
    ap.add_argument("--json", default="", help="write the result here")
    args = ap.parse_args(argv)

    err = usage_error(args)
    if err:
        print(err, file=sys.stderr)
        return 2
    try:   # all frames are read before a model loads
        frames = list_frames(args.images, args.n)
        imgs = [read_gray(f) for f in frames]
    except FileNotFoundError as e:
        print(f"image folder or file not found or not readable: {e}", file=sys.stderr)
        return 2
    if not frames:
        print("no images found", file=sys.stderr)
        return 2
    t0 = time.time()
    try:   # a backend that cannot load or run (wrong engine size, bad image) is an error, not a parity fail
        ref = run_backend("torch-fp32", None, imgs, args.model, args.device, args.warmup)
        test = run_backend(args.backend, args.engine or None, imgs, args.model, args.device, args.warmup)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"nothing compared: {e}", file=sys.stderr)
        return 2
    per_frame, ds, agree, nvalid = [], [], [], 0
    for f, a, b in zip(frames, ref["maps"], test["maps"]):
        c = compare(a, b)
        s = stats(c["d"])
        per_frame.append({"frame": os.path.basename(f), "n_valid": c["n_valid"], "mask_agree": c["mask_agree"],
                          **{k: s[k] for k in ("median_d", "p99_abs_d", "max_abs_d")}})
        ds.append(c["d"])
        agree.append(c["mask_agree"])
        nvalid += c["n_valid"]
    pooled = stats(np.concatenate(ds))
    ok = passes(pooled, float(np.min(agree)))
    import torch
    res = {
        "backend": args.backend, "reference": "torch-fp32", "engine": args.engine or None, "model": ref["model"],
        "source": {"reference": ref["source"], "backend": test["source"]},
        "frames": len(frames), "first": frames[0], "last": frames[-1], "image_wh": ref["image_wh"],
        "map_hw": ref["map_hw"], "input_hw": test.get("input_hw"), "n_valid": nvalid,
        "pooled": pooled, "mask_agree_mean": float(np.mean(agree)), "mask_agree_min": float(np.min(agree)),
        "worst_frame_p99_abs_d": float(max(p["p99_abs_d"] for p in per_frame)),
        "net_ms": {"torch-fp32": {"p50": pct(ref["net_ms"], 50), "p90": pct(ref["net_ms"], 90)},
                   args.backend: {"p50": pct(test["net_ms"], 50), "p90": pct(test["net_ms"], 90)}},
        "limits": {"abs_median_d": MEDIAN_LIMIT, "p99_abs_d": P99_LIMIT, "mask_agree_min": MASK_LIMIT},
        "pass": bool(ok), "engine_record": test.get("engine_record"), "processor": test.get("processor"),
        "processor_warning": test.get("processor_warning"),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "python": platform.python_version(), "torch": torch.__version__, "wall_s": round(time.time() - t0, 1),
        "per_frame": per_frame,
    }
    print(f"{args.backend} vs torch-fp32: {len(frames)} frames {ref['image_wh'][0]}x{ref['image_wh'][1]}, "
          f"map {ref['map_hw'][1]}x{ref['map_hw'][0]}, {nvalid} valid pixels")
    print(f"  median d log z {pooled['median_d']:+.2e} (limit {MEDIAN_LIMIT}), mean d {pooled['mean_d']:+.2e}, "
          f"median |d| {pooled['median_abs_d']:.2e}, p99 |d| {pooled['p99_abs_d']:.2e} (limit {P99_LIMIT}), "
          f"max |d| {pooled['max_abs_d']:.2e}")
    print(f"  mask agreement mean {res['mask_agree_mean']:.6f}, min {res['mask_agree_min']:.6f} (limit {MASK_LIMIT})")
    for k, v in res["net_ms"].items():
        print(f"  net_ms {k}: p50 {v['p50']:.1f}, p90 {v['p90']:.1f}")
    if res["processor_warning"]:
        print(f"  warning: {res['processor_warning']}")
    print(f"  {'PASS' if ok else 'FAIL: refit the depth constants for this backend'}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)
        print(f"  written {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
