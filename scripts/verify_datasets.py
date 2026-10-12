#!/usr/bin/env python
"""Verify the on-disk datasets against scripts/dataset_manifest.yaml.

For every requested sequence directory data/<dataset>/<sequence>/ this checks
  * layout: mav0/cam0, mav0/cam1, mav0/imu0 (data.csv each), the ground-truth csv named in the
    manifest, gt.tum, SOURCE.json; plus per layout the calibration files: EuRoC ASL ships
    mav0/*/sensor.yaml + mav0/body.yaml, the TUM-VI 512_16 euroc export ships NO sensor.yaml but
    dso/camchain.yaml, dso/imu_config.yaml and dso/cam{0,1}/camera.txt;
  * images: number of files in cam{0,1}/data equals the rows of cam{0,1}/data.csv and every
    listed file exists; cam0 and cam1 stamps agree (stereo pairs);
  * timestamps strictly increasing for cam0, cam1, imu0 and ground truth;
  * rates: IMU ~200 Hz, camera ~20 Hz, ground-truth rate reported; largest gap reported;
  * gt.tum: row count equals the ground-truth csv rows, unit quaternions, first stamp matches;
  * archive hash (with --hash): sha256 of data/_downloads/<dataset>/<archive> equals the manifest
    value (and the server-published md5 when the manifest carries one);
  * size on disk (size_GB) and provenance (sha256, source URL) from SOURCE.json, for the inventory.

Prints one table row per sequence and exits 1 when any check fails.

    uv run python scripts/verify_datasets.py                    # everything found on disk
    uv run python scripts/verify_datasets.py --dataset euroc --sequences V1_01_easy --hash
    uv run python scripts/verify_datasets.py --json /tmp/inv.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "scripts" / "dataset_manifest.yaml"
CHUNK = 1 << 22


def _hash(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _stamps_and_names(csv: Path) -> tuple[np.ndarray, list[str]]:
    ts, names = [], []
    with open(csv) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split(",")
            ts.append(int(parts[0]))
            names.append(parts[1].strip())
    return np.asarray(ts, dtype=np.int64), names


def _rate(ts: np.ndarray) -> tuple[float, float, float]:
    """(mean rate Hz, duration s, max gap s)."""
    if ts.size < 2:
        return float("nan"), 0.0, float("nan")
    dt = np.diff(ts) / 1e9
    dur = (ts[-1] - ts[0]) / 1e9
    return (ts.size - 1) / dur, dur, float(dt.max())


def verify_sequence(dataset: str, name: str, ds: dict, seq: dict, root: Path, do_hash: bool) -> dict:
    d = root / dataset / name
    mav0 = d / "mav0"
    r = {"dataset": dataset, "sequence": name, "problems": []}
    prob = r["problems"].append
    if not mav0.is_dir():
        prob("mav0 missing")
        return r

    # --- required files
    gt_rel = ds["gt_csv"]
    gt_dir = (mav0.parent / gt_rel).parent
    required = [mav0 / "cam0" / "data.csv", mav0 / "cam1" / "data.csv", mav0 / "imu0" / "data.csv",
                d / gt_rel, d / "gt.tum", d / "SOURCE.json"]
    layout = ds["layout"]
    if layout == "euroc_asl":
        required += [mav0 / "cam0" / "sensor.yaml", mav0 / "cam1" / "sensor.yaml", mav0 / "imu0" / "sensor.yaml",
                     gt_dir / "sensor.yaml", mav0 / "body.yaml"]
    elif layout == "tumvi_euroc_export":
        required += [d / "dso" / "camchain.yaml", d / "dso" / "imu_config.yaml",
                     d / "dso" / "cam0" / "camera.txt", d / "dso" / "cam1" / "camera.txt"]
    else:
        prob(f"no layout rules for {layout}")
    for p in required:
        if not p.exists():
            prob(f"missing {p.relative_to(d)}")
    r["size_bytes"] = sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
    r["size_GB"] = r["size_bytes"] / 1e9
    src_path = d / "SOURCE.json"
    if src_path.exists():
        rec = json.load(open(src_path))
        r["sha256"] = rec.get("sha256")
        r["md5_server"] = rec.get("md5_server")
        r["source_url"] = rec.get("source", {}).get("url")
        r["source_type"] = rec.get("source", {}).get("type")
        r["archive"] = rec.get("archive")
        r["archive_size"] = rec.get("archive_size")
        if not rec.get("source"):
            prob("SOURCE.json has an empty source record")

    # --- cameras
    cam_ts = {}
    for cam in ("cam0", "cam1"):
        csv = mav0 / cam / "data.csv"
        if not csv.exists():
            continue
        ts, names = _stamps_and_names(csv)
        cam_ts[cam] = ts
        files = {p.name for p in (mav0 / cam / "data").iterdir()} if (mav0 / cam / "data").is_dir() else set()
        r[f"{cam}_n"] = len(names)
        r[f"{cam}_files"] = len(files)
        missing = [n for n in names if n not in files]
        if missing:
            prob(f"{cam}: {len(missing)} listed images missing (first {missing[0]})")
        if len(files) != len(names):
            prob(f"{cam}: {len(files)} files on disk vs {len(names)} csv rows")
        if ts.size and not (np.diff(ts) > 0).all():
            prob(f"{cam}: timestamps not strictly increasing")
        hz, dur, gap = _rate(ts)
        r[f"{cam}_hz"] = hz
        if cam == "cam0":
            r["dur_s"] = dur
            r["cam_gap_s"] = gap
    if "cam0" in cam_ts and "cam1" in cam_ts:
        a, b = cam_ts["cam0"], cam_ts["cam1"]
        common = np.intersect1d(a, b).size
        r["stereo_pairs"] = int(common)
        if common < 0.95 * min(a.size, b.size):
            prob(f"only {common} stereo pairs share a stamp (cam0 {a.size}, cam1 {b.size})")

    # --- IMU
    imu_csv = mav0 / "imu0" / "data.csv"
    if imu_csv.exists():
        imu = np.loadtxt(imu_csv, delimiter=",", comments="#", ndmin=2)
        ts = imu[:, 0].astype(np.int64)
        r["imu_n"] = int(ts.size)
        if imu.shape[1] != 7:
            prob(f"imu0/data.csv has {imu.shape[1]} columns, expected 7")
        if not (np.diff(ts) > 0).all():
            prob("imu0: timestamps not strictly increasing")
        hz, _, gap = _rate(ts)
        r["imu_hz"] = hz
        r["imu_gap_s"] = gap
        if not (150 <= hz <= 250):
            prob(f"imu rate {hz:.1f} Hz outside 150-250")
        if not np.isfinite(imu).all():
            prob("imu0: non-finite values")

    # --- ground truth csv and gt.tum
    gt_csv = d / gt_rel
    if gt_csv.exists():
        gt = np.loadtxt(gt_csv, delimiter=",", comments="#", usecols=range(8), ndmin=2)
        ts = gt[:, 0].astype(np.int64)
        r["gt_n"] = int(ts.size)
        if not (np.diff(ts) > 0).all():
            prob("ground truth: timestamps not strictly increasing")
        hz, gdur, gap = _rate(ts)
        r["gt_hz"] = hz
        r["gt_gap_s"] = gap
        qn = np.linalg.norm(gt[:, 4:8], axis=1)
        if not np.allclose(qn, 1.0, atol=1e-3):
            prob("ground truth: non-unit quaternions")
        tum = d / "gt.tum"
        if tum.exists():
            arr = np.loadtxt(tum, comments="#", ndmin=2)
            r["tum_n"] = int(arr.shape[0])
            if arr.shape != (ts.size, 8):
                prob(f"gt.tum shape {arr.shape} vs csv {(ts.size, 8)}")
            else:
                if abs(arr[0, 0] - ts[0] / 1e9) > 1e-6:
                    prob("gt.tum first stamp differs from csv")
                if not np.allclose(arr[:, 1:4], gt[:, 1:4], atol=1e-6):
                    prob("gt.tum positions differ from csv")
                # csv is w,x,y,z ; tum is x,y,z,w
                if not np.allclose(arr[:, 4:8], gt[:, [5, 6, 7, 4]], atol=1e-6):
                    prob("gt.tum quaternion order/values differ from csv")
                if not np.allclose(np.linalg.norm(arr[:, 4:8], axis=1), 1.0, atol=1e-3):
                    prob("gt.tum: non-unit quaternions")
        # overlap between GT and camera stream
        if "cam0" in cam_ts and cam_ts["cam0"].size:
            c = cam_ts["cam0"]
            ov = (min(c[-1], ts[-1]) - max(c[0], ts[0])) / 1e9
            r["gt_overlap_s"] = ov
            if ov < 0.5 * r.get("dur_s", 0):
                prob(f"ground truth covers only {ov:.1f} s of {r.get('dur_s', 0):.1f} s")

    # --- archive hash
    if do_hash:
        archive = root / "_downloads" / dataset / seq["archive"]
        if not archive.exists():
            r["hash"] = "archive absent"
        else:
            sha = _hash(archive, "sha256")
            want = seq.get("sha256")
            if want is None:
                r["hash"] = "no manifest sha256"
                prob("manifest has no sha256 to check against")
            elif sha != want:
                r["hash"] = "MISMATCH"
                prob(f"sha256 mismatch {sha[:12]}.. vs manifest {want[:12]}..")
            else:
                r["hash"] = "sha256 ok"
            if seq.get("md5_server"):
                md5 = _hash(archive, "md5")
                if md5 != seq["md5_server"]:
                    prob("md5 differs from server-published value")
                elif r["hash"] == "sha256 ok":
                    r["hash"] = "sha256+md5 ok"
    else:
        src = d / "SOURCE.json"
        if src.exists():
            rec = json.load(open(src))
            r["hash"] = "recorded" if rec.get("sha256") == seq.get("sha256") else "SOURCE.json sha != manifest"
            if r["hash"] != "recorded":
                prob("SOURCE.json sha256 differs from manifest")
    return r


def fmt_table(rows: list[dict]) -> str:
    cols = [("dataset", 7), ("sequence", 16), ("cam0_n", 6), ("cam1_n", 6), ("stereo_pairs", 6), ("imu_n", 7),
            ("imu_hz", 7), ("gt_n", 7), ("gt_hz", 6), ("dur_s", 6), ("cam_gap_s", 8), ("imu_gap_s", 8),
            ("gt_gap_s", 8), ("tum_n", 7), ("size_GB", 7), ("hash", 14), ("status", 6)]
    head = " ".join(f"{c:<{w}}" for c, w in cols)
    lines = [head, "-" * len(head)]
    for r in rows:
        r = dict(r)
        r["status"] = "OK" if not r["problems"] else "FAIL"
        cells = []
        for c, w in cols:
            v = r.get(c, "-")
            if isinstance(v, float):
                v = f"{v:.2f}" if v < 100 else f"{v:.1f}"
            cells.append(f"{str(v):<{w}}")
        lines.append(" ".join(cells))
        for p in r["problems"]:
            lines.append(f"    ! {p}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--dataset", choices=["euroc", "tumvi"])
    ap.add_argument("--sequences", nargs="*", default=[])
    ap.add_argument("--hash", action="store_true", help="re-hash the archives (slow)")
    ap.add_argument("--json", type=Path, help="also write the rows to this JSON file")
    args = ap.parse_args(argv)

    man = yaml.safe_load(open(args.manifest))
    root = REPO / man.get("root", "data")
    rows = []
    for dname, ds in man["datasets"].items():
        if ds["layout"] == "rosbag":
            continue
        if args.dataset and dname != args.dataset:
            continue
        for sname, seq in ds["sequences"].items():
            if args.sequences and sname not in args.sequences:
                continue
            if not args.sequences and not (root / dname / sname).is_dir():
                continue
            rows.append(verify_sequence(dname, sname, ds, seq, root, args.hash))
    if not rows:
        print("nothing to verify (no sequence directories found)")
        return 1
    print(fmt_table(rows))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)
    bad = [r for r in rows if r["problems"]]
    print(f"\n{len(rows) - len(bad)}/{len(rows)} sequences verified OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
