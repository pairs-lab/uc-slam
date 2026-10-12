#!/usr/bin/env python
"""Precompute the PriorNet depth cache for the ORB-SLAM3 fork.

The paper's PriorNet thread runs a metric monocular depth network on
keyframes. For the offline benchmark the network output is computed once,
here, for every cam0 frame, and the C++ thread (``cpp/ucslam/src/PriorNet.cc``)
reads it by timestamp. The estimator input is then identical from run to run,
and the network latency is measured here, separately, for the runtime table.
This is a Python cache of released third-party weights. It is not the C++
system.

Layout::

    data/depth/<dataset>/<seq>/<model>/<timestamp_ns>.npy   float16 (H/d, W/d) metres, 0 = no prediction
    data/depth/<dataset>/<seq>/<model>/manifest.json        geometry, counts, latency, provenance

Network input, the convention of ``experiments/depth_prior_bench.py``. An
EuRoC cam0 frame (752 x 480, radial-tangential) is fed as recorded, gray
replicated to three channels, with the distortion left in. A TUM-VI cam0
frame (512 x 512 equidistant fisheye) is first rectified to a pinhole view
of ``depth.tumvi_rect.fov_deg`` degrees and ``depth.tumvi_rect.size`` pixels
with ``cv2.fisheye.initUndistortRectifyMap`` and ``R = I``. The rectifier is
the bench's own function, imported from it. The manifest records which of
the two views the map lives in (``view`` is ``camera`` or ``pinhole``) and,
for the pinhole view, its intrinsics ``K_view`` at the full resolution.

Storage. The network predicts at the input resolution. The map is averaged
over ``downscale x downscale`` blocks (``cv2.INTER_AREA``) and stored as
float16. Pixel centres stay aligned, so a network input pixel ``u`` reads the
map at ``u_map = (u + 0.5) * map_w / full_w - 0.5``. A prediction that is not
finite or is at or below ``z_min_m`` is stored as 0, which means no
prediction. Depths are clipped at ``z_max_m`` before the float16 cast.

Latency. ``MetricDepth.predict`` times its own call, host to device copy and
the resample back included, with a ``torch.cuda.synchronize`` on each side.
The first ``latency_warmup`` frames of every sequence are excluded from the
statistics. The pipeline time per frame (image read, predict, downscale,
save) is recorded next to it.

Constants come from the ``ucslam.precompute`` block of ``configs/default.yaml``.

Usage::

    uv run python experiments/precompute_depth.py                       # all 17 sequences
    uv run python experiments/precompute_depth.py --seqs euroc/MH_01_easy
    uv run python experiments/precompute_depth.py --limit 20            # smoke test
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import datetime as dt
import json
import os
import platform
import re
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from ucslam import calib                                   # noqa: E402
from ucslam.calib import fisheye_rectifier                 # noqa: E402

DEFAULT_CONFIG = os.path.join(REPO, "configs", "default.yaml")
DEFAULT_DATA_ROOT = os.path.join(REPO, "data")
DEFAULT_OUT_ROOT = os.path.join(REPO, "data", "depth")
DEFAULT_SUMMARY_DIR = os.path.join(REPO, "experiments", "out")

EUROC_SEQS = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult",
              "V1_01_easy", "V1_02_medium", "V1_03_difficult", "V2_01_easy", "V2_02_medium",
              "V2_03_difficult"]
TUMVI_ROOMS = [f"room{i}" for i in range(1, 7)]
ALL_KEYS = [f"euroc/{s}" for s in EUROC_SEQS] + [f"tumvi/{s}" for s in TUMVI_ROOMS]

PIXEL_CONVENTION = "u_map = (u_full + 0.5) * map_w / full_w - 0.5, pixel centres aligned"
PROVENANCE = ("Python cache of released third-party depth weights, written by "
              "experiments/precompute_depth.py. Not the C++ system. The C++ PriorNet thread reads it.")

_POINTER = re.compile(r"^[a-z_][a-z0-9_]*(\.[a-z_][a-z0-9_]*)+$")


# ------------------------------------------------------------------ config --

def load_config(path: str = DEFAULT_CONFIG) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(cfg: dict, value: Any) -> Any:
    """Follow a ``block.key`` pointer string into ``cfg``. Other values pass through."""
    for _ in range(8):
        if not (isinstance(value, str) and _POINTER.match(value)):
            return value
        node: Any = cfg
        for part in value.split("."):
            if not isinstance(node, dict) or part not in node:
                return value
            node = node[part]
        value = node
    return value


def precompute_params(cfg: dict) -> dict:
    """The resolved ``ucslam.precompute`` block plus the model name."""
    p = dict(cfg["ucslam"]["precompute"])
    for k in list(p):
        p[k] = resolve(cfg, p[k])
    p["model"] = cfg["ucslam"]["depth_model"]
    p["downscale"] = int(p["downscale"])
    p["latency_warmup"] = int(p["latency_warmup"])
    return p


# ---------------------------------------------------------------- geometry --

def make_view(dataset: str, cam0, cfg: dict) -> dict:
    """How the network sees a cam0 frame of ``dataset``.

    Returns a dict with ``view`` (``camera`` or ``pinhole``), ``full_size``
    (w, h) of the network input, and for the pinhole view ``K_view``
    (fx, fy, cx, cy), ``rect`` (fov_deg, f_px) and the two remap arrays under
    ``maps``. The pinhole view is the bench's rectified TUM-VI view.
    """
    if dataset == "tumvi":
        rc = resolve(cfg, cfg["ucslam"]["precompute"]["tumvi_rect"])
        rect = fisheye_rectifier(cam0, float(rc["fov_deg"]), rc["size"])
        K = rect["K"]
        return {"view": "pinhole", "full_size": [int(rect["size"][0]), int(rect["size"][1])],
                "K_view": [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])],
                "rect": {"fov_deg": float(rect["fov_deg"]), "f_px": float(rect["f"])},
                "maps": (rect["map1"], rect["map2"])}
    if dataset in ("euroc", "openloris"):
        # OpenLORIS (A14): the D435i color frame converted to gray, pinhole with zero distortion, fed as recorded.
        return {"view": "camera", "full_size": [int(cam0.width), int(cam0.height)], "maps": None}
    raise ValueError(f"unknown dataset {dataset!r}")


def net_input(img_gray: np.ndarray, view: dict) -> np.ndarray:
    """The frame as the network sees it. Rectified for the pinhole view, as recorded otherwise."""
    import cv2
    if view["maps"] is None:
        return img_gray
    m1, m2 = view["maps"]
    return cv2.remap(img_gray, m1, m2, cv2.INTER_LINEAR)


def downscale_map(z: np.ndarray, factor: int, z_min: float, z_max: float) -> np.ndarray:
    """Block-average a depth map and cast it to float16. 0 marks no prediction.

    Non-finite or too small predictions are set to NaN before the average, so
    a block touching one spreads the mark instead of averaging it in. Depths
    are clipped at ``z_max`` before the cast. The output shape is
    ``(H // factor, W // factor)``.
    """
    import cv2
    z = np.asarray(z, np.float32)
    bad = ~np.isfinite(z) | (z <= z_min)
    z = np.where(bad, np.nan, np.minimum(z, np.float32(z_max))).astype(np.float32)
    h, w = z.shape
    if factor > 1:
        z = cv2.resize(z, (w // factor, h // factor), interpolation=cv2.INTER_AREA)
    z = np.where(np.isfinite(z) & (z > z_min), z, 0.0).astype(np.float32)
    return z.astype(np.float16)


def map_filename(t_ns: int) -> str:
    return f"{int(t_ns)}.npy"


# ------------------------------------------------------------------- stats --

def latency_stats(latency_s: List[float], warmup: int) -> dict:
    """Median, p90, p99, mean and max in ms after dropping the first ``warmup`` values."""
    v = np.asarray(latency_s[warmup:], float) * 1e3
    if v.size == 0:
        return {"n": 0, "warmup_discarded": min(warmup, len(latency_s))}
    return {"n": int(v.size), "warmup_discarded": min(warmup, len(latency_s)),
            "median_ms": float(np.median(v)), "p90_ms": float(np.percentile(v, 90)),
            "p99_ms": float(np.percentile(v, 99)), "mean_ms": float(np.mean(v)),
            "max_ms": float(np.max(v))}


def _jsonable(o: Any) -> Any:
    """Plain JSON that OpenCV's FileStorage reader also accepts: no null, no bool, no NaN."""
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items() if v is not None}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, bool):
        return int(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        if not np.isfinite(f):
            raise ValueError("a manifest value is not finite")
        return f
    return o


def write_manifest(path: str, manifest: dict) -> None:
    with open(path, "w") as f:
        json.dump(_jsonable(manifest), f, indent=1)


# ---------------------------------------------------------------- sequence --

def process_sequence(dataset: str, seq: str, net, cfg: dict, data_root: str = DEFAULT_DATA_ROOT,
                     out_root: str = DEFAULT_OUT_ROOT, force: bool = False,
                     limit: Optional[int] = None, prefetch: int = 8, device_name: str = "") -> dict:
    """Write the cache of one sequence and return its manifest.

    ``net`` needs ``predict(img) -> float32 (H, W)``, ``last_latency_s``,
    ``name`` and ``source``, the interface of ``ucslam.depthnet.MetricDepth``.
    Existing maps are kept unless ``force``. ``limit`` caps the frame count
    for smoke tests and is recorded in the manifest.
    """
    import cv2
    p = precompute_params(cfg)
    seq_dir = os.path.join(data_root, dataset, seq)
    cam0 = (calib.load_euroc_camera(seq_dir, "cam0") if dataset in ("euroc", "openloris")
            else calib.load_tumvi_camera(seq_dir, "cam0"))
    t_ns, paths = calib.image_index(seq_dir, "cam0")
    if limit is not None:
        t_ns, paths = t_ns[:limit], paths[:limit]
    view = make_view(dataset, cam0, cfg)
    out_dir = os.path.join(out_root, dataset, seq, net.name)
    os.makedirs(out_dir, exist_ok=True)

    z_min, z_max, d = float(p["z_min_m"]), float(p["z_max_m"]), int(p["downscale"])
    latency, pipeline, n_written, n_skipped, n_zero = [], [], 0, 0, 0
    map_size = None
    t_start = time.time()

    def _read(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        return img

    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        pending: collections.deque = collections.deque()
        todo = []
        for t, path in zip(t_ns, paths):
            target = os.path.join(out_dir, map_filename(int(t)))
            if os.path.exists(target) and not force:
                n_skipped += 1
                continue
            todo.append((int(t), path, target))
        idx = 0
        while idx < len(todo) or pending:
            while idx < len(todo) and len(pending) < prefetch:
                t, path, target = todo[idx]
                pending.append((t, target, ex.submit(_read, path)))
                idx += 1
            t, target, fut = pending.popleft()
            t0 = time.perf_counter()
            img = fut.result()
            z = net.predict(net_input(img, view))
            latency.append(float(net.last_latency_s))
            zmap = downscale_map(z, d, z_min, z_max)
            if map_size is None:
                map_size = [int(zmap.shape[1]), int(zmap.shape[0])]
            n_zero += int(np.count_nonzero(zmap == 0))
            np.save(target, zmap)
            pipeline.append(time.perf_counter() - t0)
            n_written += 1

    if map_size is None:
        probe = sorted(f for f in os.listdir(out_dir) if f.endswith(".npy"))
        if probe:
            a = np.load(os.path.join(out_dir, probe[0]), mmap_mode="r")
            map_size = [int(a.shape[1]), int(a.shape[0])]
    manifest = {
        "dataset": dataset, "seq": seq, "model": net.name, "model_source": getattr(net, "source", ""),
        "view": view["view"], "full_size": view["full_size"], "map_size": map_size,
        "downscale": d, "dtype": str(p["dtype"]), "z_min_m": z_min, "z_max_m": z_max,
        "K_view": view.get("K_view"), "rect": view.get("rect"),
        "pixel_convention": PIXEL_CONVENTION,
        "input": "grayscale replicated to three channels" + (", rectified fisheye" if view["view"] == "pinhole" else ", as recorded"),
        "n_frames": int(len(t_ns)), "n_written": n_written, "n_skipped_existing": n_skipped,
        "limit": limit, "zero_pixel_fraction": (n_zero / (n_written * map_size[0] * map_size[1])) if n_written and map_size else 0.0,
        "t_ns_first": int(t_ns[0]) if len(t_ns) else 0, "t_ns_last": int(t_ns[-1]) if len(t_ns) else 0,
        "latency": latency_stats(latency, p["latency_warmup"]),
        "pipeline": latency_stats(pipeline, p["latency_warmup"]),
        "wall_s": time.time() - t_start, "device_name": device_name,
        "date": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "code": "experiments/precompute_depth.py",
        "provenance": PROVENANCE,
    }
    write_manifest(os.path.join(out_dir, "manifest.json"), manifest)
    return manifest


def summarize(manifests: List[dict]) -> dict:
    """Per-dataset latency over the frames of several manifests, count weighted."""
    out: Dict[str, Any] = {"sequences": {}, "datasets": {}}
    per_ds: Dict[str, List[dict]] = {}
    for m in manifests:
        out["sequences"][f"{m['dataset']}/{m['seq']}"] = {
            "n_frames": m["n_frames"], "n_written": m["n_written"], "map_size": m["map_size"],
            "latency": m["latency"], "pipeline": m["pipeline"], "wall_s": m["wall_s"]}
        per_ds.setdefault(m["dataset"], []).append(m)
    for ds, ms in per_ds.items():
        lat = [m["latency"] for m in ms if m["latency"].get("n", 0) > 0]
        n = sum(l["n"] for l in lat)
        out["datasets"][ds] = {
            "n_sequences": len(ms), "n_frames": sum(m["n_frames"] for m in ms),
            "n_written": sum(m["n_written"] for m in ms), "n_timed": n,
            "latency_median_ms_of_sequence_medians": float(np.median([l["median_ms"] for l in lat])) if lat else None,
            "latency_p99_ms_max_over_sequences": float(max(l["p99_ms"] for l in lat)) if lat else None,
            "latency_mean_ms": float(sum(l["mean_ms"] * l["n"] for l in lat) / n) if n else None,
            "wall_s": sum(m["wall_s"] for m in ms)}
    return out


# --------------------------------------------------------------------- CLI --

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--seqs", nargs="*", default=None, help="dataset/seq keys. Default: 11 EuRoC and 6 TUM-VI rooms")
    ap.add_argument("--model", default=None, help="override ucslam.depth_model")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true", help="rewrite maps that exist")
    ap.add_argument("--limit", type=int, default=None, help="frames per sequence, for smoke tests")
    ap.add_argument("--summary-dir", default=DEFAULT_SUMMARY_DIR,
                    help="where precompute_depth_<model>.json with the latency summary is written")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import torch
    from ucslam.depthnet import MetricDepth
    cfg = load_config(args.config)
    if args.model:
        cfg["ucslam"]["depth_model"] = args.model
    model = cfg["ucslam"]["depth_model"]
    device = args.device if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device.startswith("cuda") else platform.processor()
    keys = list(args.seqs or ALL_KEYS)
    print(f"loading {model} on {device} ({device_name})", flush=True)
    net = MetricDepth(model, device=device)
    manifests = []
    t_all = time.time()
    for key in keys:
        dataset, seq = key.split("/")
        t0 = time.time()
        m = process_sequence(dataset, seq, net, cfg, args.data_root, args.out_root, args.force,
                             args.limit, device_name=device_name)
        manifests.append(m)
        lat = m["latency"]
        print(f"  {key}: {m['n_written']} written, {m['n_skipped_existing']} kept, map {m['map_size']}, "
              f"latency median {lat.get('median_ms', float('nan')):.1f} ms p99 {lat.get('p99_ms', float('nan')):.1f} ms, "
              f"{time.time() - t0:.0f} s", flush=True)
    summary = summarize(manifests)
    summary["meta"] = {"model": model, "model_source": net.source, "device": device, "device_name": device_name,
                       "date": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "wall_s": time.time() - t_all,
                       "limit": args.limit, "n_sequences": len(manifests),
                       "precompute": {k: v for k, v in precompute_params(cfg).items() if k != "tumvi_rect"},
                       "provenance": PROVENANCE}
    os.makedirs(args.summary_dir, exist_ok=True)
    path = os.path.join(args.summary_dir, f"precompute_depth_{model}.json")
    with open(path, "w") as f:
        json.dump(_jsonable(summary), f, indent=1)
    print(f"wrote {path}\n{(time.time() - t_all) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
