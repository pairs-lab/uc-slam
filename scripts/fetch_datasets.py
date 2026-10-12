#!/usr/bin/env python
"""Fetch, integrity-check and lay out the datasets listed in scripts/dataset_manifest.yaml.

Layout produced (relative to the repo root, see `root:` in the manifest):

    data/_downloads/<dataset>/<archive>           the verified archive (ASL zip / TUM tar)
    data/<dataset>/<sequence>/mav0/...            cam0/, cam1/, imu0/, ground truth (+ sensor.yaml for EuRoC)
    data/<dataset>/<sequence>/dso/...             TUM-VI only: Kalibr camchain.yaml, imu_config.yaml, pcalib, vignette
    data/<dataset>/<sequence>/gt.tum              "timestamp[s] tx ty tz qx qy qz qw" (evo TUM format)
    data/<dataset>/<sequence>/SOURCE.json         URL used, sha256, date, mirror

Stages per sequence, each skipped when its output already exists:
    download -> sha256 (recorded or checked) -> extract -> gt.tum -> SOURCE.json

Sources are tried in manifest order.  A `file` source is a plain (resumable) HTTP download.  A
`zip_member` source HTTP-Range-downloads one deflate-stored member out of a bundle zip and inflates
it locally; the result must match the bundle's central-directory CRC32 and size.  This is how the
EuRoC ASL zips are pulled out of the HuggingFace bundles without downloading the rosbags.  The raw
member bytes are kept in <archive>.deflate until the inflated zip passed the CRC/size check; a
complete .deflate left behind by an interrupted run is inflated offline on the next run.

Every HTTP request carries `Accept-Encoding: identity`, and a response that still declares a
Content-Encoding is rejected, so no transfer-encoded body is ever written to disk as an archive.

TUM-VI (`tumvi_euroc_export`): the 512_16 tars hold <name>/mav0/{cam0,cam1,imu0,mocap0}/data.csv
and <name>/dso/{camchain.yaml,imu_config.yaml,cam*/camera.txt,...}; there are NO sensor.yaml files
in this export.  Both mav0/ and dso/ (minus the dso/cam*/images symlinks) are extracted.

Examples:
    uv run python scripts/fetch_datasets.py --list
    uv run python scripts/fetch_datasets.py --dataset euroc --sequences V1_01_easy MH_01_easy V2_01_easy
    uv run python scripts/fetch_datasets.py --dataset tumvi --sequences room1 --record
    uv run python scripts/fetch_datasets.py --dataset euroc --all --download-only
    uv run python scripts/fetch_datasets.py --dataset euroc --sequences V1_01_easy --stage extract

`--record` writes the sha256 of a freshly downloaded archive into the manifest when it has none (it
 re-dumps the YAML and keeps only the leading comment block; run it from ONE process at a time, never
 from concurrent downloads -- record afterwards from the <archive>.sha256 sidecars instead).
Without `--record`, an archive whose manifest sha256 is null is accepted with a warning.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import struct
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "scripts" / "dataset_manifest.yaml"
CHUNK = 1 << 22  # 4 MiB
USER_AGENT = "ucslam-fetch/1.0 (+https://github.com/; python-urllib)"


# ----------------------------------------------------------------------------- manifest


def load_manifest(path: Path = MANIFEST) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_manifest(man: dict, path: Path = MANIFEST) -> None:
    """Rewrite the manifest, preserving only the leading '#' comment block."""
    header = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                header.append(line)
            else:
                break
    body = yaml.safe_dump(man, sort_keys=False, width=110, allow_unicode=True)
    tmp = path.with_suffix(".yaml.tmp")
    with open(tmp, "w") as f:
        f.writelines(header)
        f.write(body)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- hashing


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- HTTP


def _open(url: str, headers: dict | None = None, method: str = "GET", timeout: float = 60):
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity", **(headers or {})}, method=method)
    return urllib.request.urlopen(req, timeout=timeout)


def probe(url: str, timeout: float) -> int | None:
    """HEAD the URL; return Content-Length or None when unreachable."""
    try:
        with _open(url, method="HEAD", timeout=timeout) as r:
            return int(r.headers.get("Content-Length", "0"))
    except Exception as e:  # noqa: BLE001
        print(f"    probe failed ({type(e).__name__}: {e})")
        return None


def http_download(url: str, dst: Path, start: int | None = None, end: int | None = None,
                  timeout: float = 60, attempts: int = 5, label: str = "") -> int:
    """Resumable download of [start, end] (inclusive; None = whole file) to `dst`.

    Returns the number of bytes now in `dst`.  Resumes from the current size of `dst`.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    base = start or 0
    expected = None if end is None else end - base + 1
    for attempt in range(1, attempts + 1):
        have = dst.stat().st_size if dst.exists() else 0
        if expected is not None and have >= expected:
            return have
        req_start = base + have
        rng = f"bytes={req_start}-" if end is None else f"bytes={req_start}-{end}"
        try:
            with _open(url, headers={"Range": rng}, timeout=timeout) as r, open(dst, "ab") as f:
                status = r.status
                enc = (r.headers.get("Content-Encoding") or "identity").strip().lower()
                if enc not in ("", "identity"):
                    raise RuntimeError(f"{url}: server answered with Content-Encoding={enc} although we asked for "
                                       "identity; refusing to save an encoded body (pick another mirror)")
                if have and status != 206:
                    # server ignored the Range header; start over
                    f.truncate(0)
                    have = 0
                t0 = time.time()
                got = 0
                last = t0
                for chunk in iter(lambda: r.read(CHUNK), b""):
                    f.write(chunk)
                    got += len(chunk)
                    if time.time() - last > 5:
                        rate = got / max(time.time() - t0, 1e-6) / 1e6
                        print(f"    {label} {(have + got) / 1e6:9.1f} MB  {rate:6.1f} MB/s", flush=True)
                        last = time.time()
                rate = got / max(time.time() - t0, 1e-6) / 1e6
                print(f"    {label} done: +{got / 1e6:.1f} MB in {time.time() - t0:.1f} s ({rate:.1f} MB/s)")
            have = dst.stat().st_size
            if expected is None or have >= expected:
                return have
            print(f"    short read ({have}/{expected}); retrying")
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            print(f"    attempt {attempt}/{attempts} failed: {e}")
            time.sleep(min(5 * attempt, 30))
    raise RuntimeError(f"download failed after {attempts} attempts: {url}")


def inflate_raw(src: Path, dst: Path) -> tuple[int, str]:
    """Inflate a raw-deflate stream; return (size, crc32 hex)."""
    d = zlib.decompressobj(-15)
    crc = 0
    n = 0
    with open(src, "rb") as f, open(dst, "wb") as g:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            out = d.decompress(chunk)
            g.write(out)
            crc = zlib.crc32(out, crc)
            n += len(out)
        out = d.flush()
        g.write(out)
        crc = zlib.crc32(out, crc)
        n += len(out)
    return n, f"{crc & 0xFFFFFFFF:08x}"


# ----------------------------------------------------------------------------- stages


def fetch_archive(ds: dict, seq: dict, dl_dir: Path, keep_raw: bool) -> tuple[Path, dict]:
    """Download the archive for one sequence; return (path, source-record)."""
    archive = dl_dir / seq["archive"]
    if archive.exists() and archive.stat().st_size == seq.get("size", archive.stat().st_size):
        rec = _read_source_record(archive)
        print(f"  archive present: {archive} ({archive.stat().st_size} B)")
        return archive, rec
    # A complete raw range download left by an interrupted run: inflate it without touching the network.
    for src in seq["sources"]:
        if src["type"] != "zip_member":
            continue
        raw = archive.with_suffix(archive.suffix + ".deflate")
        s, e = src["range"]
        if raw.exists() and raw.stat().st_size == e - s + 1:
            print(f"  complete raw member present: {raw} ({raw.stat().st_size} B); inflating offline")
            rec = _inflate_member(ds, seq, src, raw, archive, keep_raw)
            if rec is not None:
                return archive, rec
    for src in seq["sources"]:
        typ = src["type"]
        if typ == "file":
            url = src["url"]
            timeout = float(src.get("timeout", 60))
            print(f"  trying file source {url}")
            if probe(url, timeout) is None:
                continue
            part = archive.with_suffix(archive.suffix + ".part")
            n = http_download(url, part, timeout=timeout, label=seq["archive"])
            if "size" in seq and n != seq["size"]:
                print(f"    size mismatch {n} != {seq['size']}; discarding")
                part.unlink(missing_ok=True)
                continue
            os.replace(part, archive)
            rec = {"type": "file", "url": url}
        elif typ == "zip_member":
            url = ds["bundles"][src["bundle"]]
            s, e = src["range"]
            print(f"  trying zip_member source {url}  bytes {s}-{e}")
            if probe(url, float(src.get("timeout", 60))) is None:
                continue
            raw = archive.with_suffix(archive.suffix + ".deflate")
            n = http_download(url, raw, start=s, end=e, label=seq["archive"])
            if n != e - s + 1:
                print(f"    raw size mismatch {n} != {e - s + 1}; discarding")
                raw.unlink(missing_ok=True)
                continue
            rec = _inflate_member(ds, seq, src, raw, archive, keep_raw)
            if rec is None:
                continue
            return archive, rec
        else:
            raise ValueError(f"unknown source type {typ}")
        rec["date"] = _dt.date.today().isoformat()
        _write_source_record(archive, rec)
        return archive, rec
    raise RuntimeError(f"no source reachable for {seq['archive']}")


def _inflate_member(ds: dict, seq: dict, src: dict, raw: Path, archive: Path, keep_raw: bool) -> dict | None:
    """Inflate the raw deflate member `raw` into `archive`; check size + CRC32 against the manifest.

    Returns the source record (also written next to the archive) or None when the check fails, in
    which case both the temporary output and the raw file are removed so the next run re-downloads.
    """
    url = ds["bundles"][src["bundle"]]
    s, e = src["range"]
    tmp = archive.with_suffix(archive.suffix + ".part")
    t0 = time.time()
    size, crc = inflate_raw(raw, tmp)
    if size != seq["size"] or crc != seq["crc32"]:
        print(f"    inflate check failed: size {size} (want {seq['size']}), crc32 {crc} (want {seq['crc32']})")
        tmp.unlink(missing_ok=True)
        raw.unlink(missing_ok=True)
        return None
    print(f"    inflated {size} B in {time.time() - t0:.1f} s, crc32 {crc} OK")
    os.replace(tmp, archive)
    if not keep_raw:
        raw.unlink(missing_ok=True)
    rec = {"type": "zip_member", "url": url, "member": src["member"], "range": [s, e],
           "date": _dt.date.today().isoformat()}
    _write_source_record(archive, rec)
    return rec


def _read_source_record(archive: Path) -> dict:
    p = archive.with_suffix(archive.suffix + ".source.json")
    return json.load(open(p)) if p.exists() else {}


def _write_source_record(archive: Path, rec: dict) -> None:
    with open(archive.with_suffix(archive.suffix + ".source.json"), "w") as f:
        json.dump(rec, f, indent=1)


def check_hash(seq: dict, archive: Path, record: bool, man: dict, manifest_path: Path) -> str:
    sha_path = archive.with_suffix(archive.suffix + ".sha256")
    if sha_path.exists():
        sha = sha_path.read_text().split()[0]
    else:
        print(f"  sha256 {archive.name} ...", flush=True)
        sha = sha256_file(archive)
        sha_path.write_text(f"{sha}  {archive.name}\n")
    want = seq.get("sha256")
    if want:
        if sha != want:
            raise RuntimeError(f"sha256 MISMATCH for {archive}: {sha} != manifest {want}")
        print(f"  sha256 OK {sha}")
    else:
        if record:
            seq["sha256"] = sha
            save_manifest(man, manifest_path)
            print(f"  sha256 recorded into manifest: {sha}")
        else:
            print(f"  WARNING: manifest has no sha256 for {archive.name}; got {sha} (use --record)")
    md5_want = seq.get("md5_server")
    if md5_want:
        md5_path = archive.with_suffix(archive.suffix + ".md5.local")
        md5 = md5_path.read_text().split()[0] if md5_path.exists() else md5_file(archive)
        md5_path.write_text(f"{md5}  {archive.name}\n")
        if md5 != md5_want:
            raise RuntimeError(f"md5 MISMATCH for {archive}: {md5} != server {md5_want}")
        print(f"  md5 matches server-published {md5}")
    return sha


def extract(layout: str, archive: Path, seq_dir: Path) -> Path:
    mav0 = seq_dir / "mav0"
    if mav0.is_dir() and (mav0 / "cam0" / "data.csv").exists():
        print(f"  extracted already: {mav0}")
        return mav0
    seq_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if layout == "euroc_asl":
        # zip contains mav0/... (+ __MACOSX junk)
        with zipfile.ZipFile(archive) as z:
            members = [m for m in z.infolist() if m.filename.startswith("mav0/")]
            for m in members:
                z.extract(m, seq_dir)
        print(f"  extracted {len(members)} entries in {time.time() - t0:.1f} s")
    elif layout == "tumvi_euroc_export":
        # tar contains dataset-<name>_512_16/{mav0,dso}/... ; strip the first component.  dso/ holds the
        # Kalibr calibration (camchain.yaml, imu_config.yaml, cam*/camera.txt, pcalib, vignette); its
        # cam*/images entries are symlinks to ../../mav0/cam*/data and are skipped.
        n = 0
        with tarfile.open(archive) as t:
            for m in t:
                parts = Path(m.name).parts
                if len(parts) < 2 or parts[1] not in ("mav0", "dso"):
                    continue
                if m.issym() or m.islnk():
                    continue
                m.name = str(Path(*parts[1:]))
                t.extract(m, seq_dir, filter="data")
                n += 1
        print(f"  extracted {n} entries in {time.time() - t0:.1f} s")
    else:
        raise ValueError(f"no extractor for layout {layout}")
    if not (mav0 / "cam0" / "data.csv").exists():
        raise RuntimeError(f"extraction of {archive} did not produce {mav0}/cam0/data.csv")
    return mav0


def write_gt_tum(seq_dir: Path, gt_csv_rel: str) -> Path:
    """Convert an ASL ground-truth csv (ns, p_xyz, q_wxyz, ...) to evo TUM format (s, xyz, q_xyzw)."""
    out = seq_dir / "gt.tum"
    if out.exists():
        print(f"  gt.tum present: {out}")
        return out
    csv = seq_dir / gt_csv_rel
    arr = np.loadtxt(csv, delimiter=",", comments="#", usecols=range(8), ndmin=2)
    ts = arr[:, 0].astype(np.int64)
    with open(csv) as f:
        header = f.readline().strip()
    if not ts.size or not (np.diff(ts) > 0).all():
        raise RuntimeError(f"ground-truth timestamps not strictly increasing in {csv}")
    p = arr[:, 1:4]
    q_wxyz = arr[:, 4:8]
    q_xyzw = q_wxyz[:, [1, 2, 3, 0]]
    nrm = np.linalg.norm(q_xyzw, axis=1)
    if not np.allclose(nrm, 1.0, atol=1e-3):
        raise RuntimeError(f"non-unit quaternions in {csv} (min/max norm {nrm.min():.4f}/{nrm.max():.4f})")
    tmp = out.with_suffix(".tum.tmp")
    with open(tmp, "w") as f:
        f.write(f"# timestamp[s] tx ty tz qx qy qz qw  (converted from {gt_csv_rel}; source header: {header})\n")
        for t, pp, qq in zip(ts, p, q_xyzw):
            f.write(f"{t // 1_000_000_000}.{t % 1_000_000_000:09d} "
                    f"{pp[0]:.7f} {pp[1]:.7f} {pp[2]:.7f} {qq[0]:.7f} {qq[1]:.7f} {qq[2]:.7f} {qq[3]:.7f}\n")
    os.replace(tmp, out)
    print(f"  wrote {out} ({len(ts)} poses)")
    return out


def write_source_json(seq_dir: Path, dataset: str, name: str, seq: dict, archive: Path, rec: dict, sha: str) -> None:
    info = {
        "dataset": dataset,
        "sequence": name,
        "archive": archive.name,
        "archive_size": archive.stat().st_size,
        "sha256": sha,
        "md5_server": seq.get("md5_server"),
        "crc32_bundle": seq.get("crc32"),
        "source": rec,
        "fetched_by": "scripts/fetch_datasets.py",
        "written": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    with open(seq_dir / "SOURCE.json", "w") as f:
        json.dump(info, f, indent=1)


# ----------------------------------------------------------------------------- driver


def run_sequence(man: dict, dataset: str, name: str, args) -> None:
    ds = man["datasets"][dataset]
    seq = ds["sequences"][name]
    root = REPO / man.get("root", "data")
    dl_dir = root / "_downloads" / dataset
    seq_dir = root / dataset / name
    print(f"== {dataset}/{name}")
    archive, rec = fetch_archive(ds, seq, dl_dir, args.keep_raw)
    if args.stage == "download":
        return
    sha = check_hash(seq, archive, args.record, man, args.manifest)
    if args.stage == "hash" or args.download_only:
        return
    if ds["layout"] == "rosbag":
        print("  rosbag layout: no extraction/gt.tum stage (needs a bag converter)")
        return
    extract(ds["layout"], archive, seq_dir)
    if args.stage == "extract":
        return
    write_gt_tum(seq_dir, ds["gt_csv"])
    write_source_json(seq_dir, dataset, name, seq, archive, rec, sha)
    if args.delete_archive:
        archive.unlink()
        print(f"  deleted archive {archive}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--dataset", choices=["euroc", "tumvi", "viode"])
    ap.add_argument("--sequences", nargs="*", default=[])
    ap.add_argument("--all", action="store_true", help="every sequence of --dataset")
    ap.add_argument("--list", action="store_true", help="print the manifest and exit")
    ap.add_argument("--stage", choices=["download", "hash", "extract", "all"], default="all")
    ap.add_argument("--download-only", action="store_true", help="download + hash, no extraction")
    ap.add_argument("--record", action="store_true", help="write missing sha256 values into the manifest")
    ap.add_argument("--keep-raw", action="store_true", help="keep the .deflate range download")
    ap.add_argument("--delete-archive", action="store_true", help="remove the archive after extraction")
    args = ap.parse_args(argv)

    man = load_manifest(args.manifest)

    if args.list:
        for dname, ds in man["datasets"].items():
            print(f"{dname}: {ds['description']}")
            for sname, s in ds["sequences"].items():
                st = "sha256 recorded" if s.get("sha256") else "no sha256 yet"
                print(f"  {sname:<18} {s['archive']:<32} {s.get('size', '?'):>12}  {st}")
        return 0
    if not args.dataset:
        ap.error("--dataset is required (or --list)")
    names = list(man["datasets"][args.dataset]["sequences"]) if args.all else args.sequences
    if not names:
        ap.error("give --sequences ... or --all")
    unknown = [n for n in names if n not in man["datasets"][args.dataset]["sequences"]]
    if unknown:
        ap.error(f"unknown sequences for {args.dataset}: {unknown}")
    failures = []
    for n in names:
        try:
            run_sequence(man, args.dataset, n, args)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED {args.dataset}/{n}: {e}")
            failures.append(n)
    if failures:
        print(f"FAILED: {failures}")
        return 1
    print("all requested sequences done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
