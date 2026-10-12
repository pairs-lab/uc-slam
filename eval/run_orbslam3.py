r"""Batch runner and scorer for the stock ORB-SLAM3 baseline (milestone M2a).

What runs
---------
The upstream ORB-SLAM3 examples built by scripts/build_orbslam3.sh, with their
stock settings files, on the sequences on disk:

    dataset  mode             executable                                        settings
    euroc    mono_inertial    Examples/Monocular-Inertial/mono_inertial_euroc   Monocular-Inertial/EuRoC.yaml
    euroc    stereo_inertial  Examples/Stereo-Inertial/stereo_inertial_euroc    Stereo-Inertial/EuRoC.yaml
    tumvi    mono_inertial    Examples/Monocular-Inertial/mono_inertial_tum_vi  Monocular-Inertial/TUM-VI.yaml

Upstream ships the image timestamp lists and the TUM-VI IMU files the examples
read. They were checked against the data on disk (every listed image exists,
the IMU line counts agree), so the runner uses them as they are.

ORB-SLAM3 takes no random seed. Its RANSAC draws from a clock-seeded generator
and its threads race, so two runs of one sequence differ. The run index k in
run<k> is the only "seed" there is. It labels the run and controls nothing.

Layout of one run
-----------------
    runs/orbslam3/<mode>/<seq>/run<k>.tum         keyframe trajectory, TUM format, seconds
    runs/orbslam3/<mode>/<seq>/run<k>_frames.tum  frame trajectory, TUM format, seconds
    runs/orbslam3/<mode>/<seq>/run<k>.log         stdout and stderr of the process
    runs/orbslam3/<mode>/<seq>/run<k>.json        the scored record of the run
    runs/orbslam3/<mode>/<seq>/kf_run<k>.txt      raw saver output (stamps in nanoseconds)
    runs/orbslam3/<mode>/<seq>/f_run<k>.txt       raw saver output (stamps in nanoseconds)

The ORB-SLAM3 EuRoC saver writes 1e9 times the time in seconds. The runner
divides by 1e9 when a stamp exceeds stamp_ns_threshold, so evo can associate
the estimate with data/<dataset>/<seq>/gt.tum.

Scoring
-------
ATE is the RMSE of evo_ape tum --align --t_max_diff 0.02 (SE(3), keyframe
trajectory), protocol Section 3. The same call on the frame trajectory is
stored as a secondary number, because the published ORB-SLAM3 EuRoC numbers
are frame-trajectory numbers (Table II caption of arXiv 2007.11898). The
Sim(3) scale error |1 - s| is computed with evo's Umeyama on the same
associated keyframe poses.

Failure rule, protocol Section 2, as far as it can be read from outside the
binary:
  * ATE above 1.0 m on EuRoC, or above 5 percent of the path length on TUM-VI.
    Room path lengths come from the ground truth. Corridor and magistrale path
    lengths come from the estimated frame trajectory, because their ground
    truth covers only the start and the end.
  * A new Atlas map: more than one "Creation of new map with id" line in the
    log. The first one is the map every run starts with.
  * Tracking lost for more than 1.0 s: either a "Track Lost..." line, which the
    code prints after 5 s in the RECENTLY_LOST state, or a gap longer than
    1.0 s plus one frame period between consecutive saved frames. The saver
    skips frames whose state was LOST and frames that belong to another map.
Not checkable from outside: a RECENTLY_LOST spell between 1 s and 5 s. During
it the IMU predicts the pose, the frame is saved as tracked, and the code
prints "Lost for a short time" once per frame without a duration. The count
of those lines is stored so the reader can see it, but it cannot be turned
into seconds without instrumenting the binary. An active-map reset before IMU
initialization ("Reseting active map") discards the early trajectory without
opening a new map. It is counted and reported, and it does not fail a run
under the letter of the rule.

TUM-VI corridor and magistrale, protocol Section 3: the estimate is aligned
with SE(3) on the ground-truth start segment only, and the end drift is the
position error at the last keyframe that has ground truth in the end segment,
in metres and in percent of the estimated path length. The ATE over the
start and end segments together (the TUM-VI convention) is stored as well.

Usage
-----
    uv run python -m eval.run_orbslam3 --dataset euroc --mode mono_inertial
    uv run python -m eval.run_orbslam3 --dataset euroc --mode stereo_inertial
    uv run python -m eval.run_orbslam3 --dataset tumvi --mode mono_inertial --seqs room1 room2 ...
    uv run python -m eval.run_orbslam3 --dataset tumvi --mode mono_inertial --seqs corridor1 ... --runs 3

Every run that already has a run<k>.json is skipped unless --force is given,
so an interrupted batch resumes. The aggregate goes to
eval/out/orbslam3/scores_<dataset>_<mode>.json and is what
eval/emit_baseline_tables.py reads.

Fork runs
---------
    uv run python -m eval.run_orbslam3 --dataset euroc --mode mono_inertial --seqs MH_01_easy \
        --tree cpp/UC_SLAM3 --settings configs/orbslam3/euroc_mono_inertial_ucslam.yaml --tag fork --runs 1

--tree picks the tree whose example binary runs (default the stock tree).
--settings replaces the stock settings file. --tag <name> puts the run files
in runs/orbslam3/<mode>/<seq>/<tag>/ and the aggregate in
scores_<dataset>_<mode>_<tag>.json, so fork runs land beside the stock ones
and never overwrite them. The timestamp lists and the TUM-VI IMU files always
come from the stock tree, which is the only tree that ships them. The meta
block of the aggregate names the tree and the settings file, and the stock
provenance string is written only for the stock tree.

Pseudo-RGB-D runs (protocol H3)
-------------------------------
    uv run python -m eval.run_orbslam3 --dataset euroc --mode rgbd_inertial --seqs MH_01_easy --runs 1

Mode rgbd_inertial runs cpp/pseudo_rgbd/build/rgbd_inertial_offline, an offline
driver linked against the stock library (System::IMU_RGBD), with
configs/orbslam3/pseudo_rgbd_<dataset>.yaml. It reads the depth PNGs of
experiments/make_pseudo_depth.py from data/pseudo_depth/<dataset>/<seq>/depth.
EuRoC frames come from the recorded cam0 folder and the IMU from
mav0/imu0/data.csv. TUM-VI frames come from the rectified folder
data/pseudo_depth/tumvi/<seq>/cam0_rect and the IMU from the stock TUM_IMU file.
The timestamp lists are the stock mono-inertial ones. The command is built by
build_command_rgbd. Scoring and the failure rule are the ones above. Runs land
in runs/orbslam3/rgbd_inertial/<seq>/[<tag>/] and the aggregate in
scores_<dataset>_rgbd_inertial[_<tag>].json. --tree has no effect in this
mode. --settings replaces the default settings file.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORB = os.path.join(REPO, "third_party", "ORB_SLAM3")
VOCAB = os.path.join(ORB, "Vocabulary", "ORBvoc.txt")
RUNS_ROOT = os.path.join(REPO, "runs", "orbslam3")
OUT_DIR = os.path.join(REPO, "eval", "out", "orbslam3")
CONFIG = os.path.join(REPO, "configs", "default.yaml")

# EuRoC sequence directory -> key of the shipped timestamp file.
EUROC_SEQS: Dict[str, str] = {
    "MH_01_easy": "MH01", "MH_02_easy": "MH02", "MH_03_medium": "MH03",
    "MH_04_difficult": "MH04", "MH_05_difficult": "MH05",
    "V1_01_easy": "V101", "V1_02_medium": "V102", "V1_03_difficult": "V103",
    "V2_01_easy": "V201", "V2_02_medium": "V202", "V2_03_difficult": "V203",
}
TUMVI_ROOMS = [f"room{i}" for i in range(1, 7)]
TUMVI_LONG = [f"corridor{i}" for i in range(1, 6)] + [f"magistrale{i}" for i in range(1, 7)]
TUMVI_SEQS = TUMVI_ROOMS + TUMVI_LONG
# OpenLORIS-Scene (protocol amendment A14), converted to the EuRoC layout by scripts/openloris_convert.py.
OPENLORIS_SCENES: Dict[str, List[str]] = {
    "office": [f"office1-{i}" for i in range(1, 8)],
    "corridor": [f"corridor1-{i}" for i in range(1, 6)],
    "home": [f"home1-{i}" for i in range(1, 6)],
    "cafe": [f"cafe1-{i}" for i in range(1, 3)],
    "market": [f"market1-{i}" for i in range(1, 4)],
}
OPENLORIS_SEQS = [s for v in OPENLORIS_SCENES.values() for s in v]
# A14 note of 2026-10-05: sequences whose ground truth fails the timing check (gt vs gyro yaw rate), dropped for every
# system. Their runs, if any, stay on disk and in the release, and enter no table.
OPENLORIS_DROPPED: Dict[str, str] = {
    "market1-1": "ground-truth time error changes within the sequence (lowest window correlation 0.61, lag spread 0.70 s)",
    "market1-3": "ground-truth time error changes within the sequence (lowest window correlation 0.32, lag spread 2.64 s)",
}
OPENLORIS_KEPT = [s for s in OPENLORIS_SEQS if s not in OPENLORIS_DROPPED]


def openloris_scene(seq: str) -> str:
    """office1-3 -> office."""
    return seq.split("1-")[0]


def openloris_robot(seq: str) -> str:
    """The market sequences come from another robot (Gaussian Robotics), the others from the Tsinghua robot."""
    return "gaussian" if openloris_scene(seq) == "market" else "tsinghua"

MODES = {
    ("euroc", "mono_inertial"): {
        "exe": "Examples/Monocular-Inertial/mono_inertial_euroc",
        "yaml": "Examples/Monocular-Inertial/EuRoC.yaml",
        "stamps": "Examples/Monocular-Inertial/EuRoC_TimeStamps/{key}.txt",
    },
    ("euroc", "stereo_inertial"): {
        "exe": "Examples/Stereo-Inertial/stereo_inertial_euroc",
        "yaml": "Examples/Stereo-Inertial/EuRoC.yaml",
        "stamps": "Examples/Stereo-Inertial/EuRoC_TimeStamps/{key}.txt",
    },
    ("tumvi", "mono_inertial"): {
        "exe": "Examples/Monocular-Inertial/mono_inertial_tum_vi",
        "yaml": "Examples/Monocular-Inertial/TUM-VI.yaml",
        "stamps": "Examples/Monocular-Inertial/TUM_TimeStamps/dataset-{seq}_512.txt",
        "imu": "Examples/Monocular-Inertial/TUM_IMU/dataset-{seq}_512.txt",
    },
}

# OpenLORIS runs the EuRoC example on the converted layout. There is no stock settings file for this camera, so the
# default is the repository file derived from the stock RealSense_D435i.yaml (A14), one per robot: {robot} is
# replaced per sequence. The image list is the one the converter writes.
MODES[("openloris", "mono_inertial")] = {
    "exe": "Examples/Monocular-Inertial/mono_inertial_euroc",
    "yaml": "configs/orbslam3/openloris_mono_inertial_stock_{robot}.yaml",
    "stamps": None,
}

# The pseudo-RGB-D baseline of protocol H3. The binary and the settings files live in this
# repository, not in an ORB-SLAM3 tree, so their paths are relative to REPO. The timestamp lists
# and the TUM-VI IMU files are the stock mono-inertial ones, relative to the stock tree.
PSEUDO_RGBD_MODE = "rgbd_inertial"
PSEUDO_RGBD_EXE = os.path.join(REPO, "cpp", "pseudo_rgbd", "build", "rgbd_inertial_offline")
PSEUDO_DEPTH_ROOT = os.path.join(REPO, "data", "pseudo_depth")
MODES.update({
    ("euroc", PSEUDO_RGBD_MODE): {
        "exe": PSEUDO_RGBD_EXE,
        "yaml": "configs/orbslam3/pseudo_rgbd_euroc.yaml",
        "stamps": "Examples/Monocular-Inertial/EuRoC_TimeStamps/{key}.txt",
    },
    ("tumvi", PSEUDO_RGBD_MODE): {
        "exe": PSEUDO_RGBD_EXE,
        "yaml": "configs/orbslam3/pseudo_rgbd_tumvi.yaml",
        "stamps": "Examples/Monocular-Inertial/TUM_TimeStamps/dataset-{seq}_512.txt",
        "imu": "Examples/Monocular-Inertial/TUM_IMU/dataset-{seq}_512.txt",
    },
})

# Log lines of third_party/ORB_SLAM3/src at the pinned commit (Tracking.cc, Atlas.cc).
LOG_MARKERS = {
    "n_maps_created": "Creation of new map with id:",
    "n_track_lost": "Track Lost...",
    "n_fail_local_map": "Fail to track local map!",
    "n_recently_lost_frames": "Lost for a short time",
    "n_lost_lt_1s": "Track lost for less than one second...",
    "n_resets_imu": "IMU is not or recently initialized. Reseting active map...",
    "n_resets_before_init": "Track lost before IMU initialisation, reseting...",
    "n_resets_active": "Active map Reseting",
    "n_new_map_started": "A new map is started...",
}

_print_lock = threading.Lock()

# Set by configure() from --tree, --settings and --tag. The defaults are the stock baseline.
_TREE: str = ORB
_SETTINGS: Optional[str] = None
_TAG: Optional[str] = None


def configure(tree: Optional[str] = None, settings: Optional[str] = None, tag: Optional[str] = None) -> None:
    """Choose the binary tree, an override settings file and a run tag for this process.

    ``tree`` is the root of an ORB-SLAM3 tree whose Examples/ hold the built
    binaries, relative paths resolve against the repository. ``settings``
    replaces the stock YAML for every run. ``tag`` names the run subdirectory
    and the aggregate file. None restores the stock default for that item.
    """
    global _TREE, _SETTINGS, _TAG
    _TREE = os.path.abspath(os.path.join(REPO, tree)) if tree else ORB
    _SETTINGS = os.path.abspath(os.path.join(REPO, settings)) if settings else None
    if tag is not None and not re.fullmatch(r"[A-Za-z0-9_.-]+", tag):
        raise ValueError(f"tag must be a plain name, got {tag!r}")
    _TAG = tag or None


def current_tree() -> str:
    return _TREE


def is_stock_tree() -> bool:
    return os.path.abspath(_TREE) == os.path.abspath(ORB)


def log(msg: str) -> None:
    with _print_lock:
        print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_cfg(path: str = CONFIG) -> dict:
    """The orbslam3_baseline block of configs/default.yaml, read without torch."""
    with open(path) as f:
        return yaml.safe_load(f)["orbslam3_baseline"]


# ---------------------------------------------------------------------------
# Paths and commands
# ---------------------------------------------------------------------------


def seq_dir(dataset: str, seq: str) -> str:
    return os.path.join(REPO, "data", dataset, seq)


def gt_file(dataset: str, seq: str) -> str:
    return os.path.join(seq_dir(dataset, seq), "gt.tum")


def stamps_file(dataset: str, mode: str, seq: str) -> str:
    if dataset == "openloris":
        return os.path.join(seq_dir(dataset, seq), "mav0", "timestamps.txt")
    spec = MODES[(dataset, mode)]
    key = EUROC_SEQS.get(seq, seq)
    return os.path.join(ORB, spec["stamps"].format(key=key, seq=seq))


_DEPTH_DIR_LINE = re.compile(r'^(UCSLAM\.DepthDir:[ \t]*)"([^"]*)"[ \t]*$', re.M)
_DEPTH_MODEL_LINE = re.compile(r'^UCSLAM\.DepthModel:[ \t]*"([^"]*)"[ \t]*$', re.M)


def _depth_base(depth_dir: str) -> str:
    """UCSLAM.DepthDir as an absolute path. A relative path is relative to the repository root."""
    return depth_dir if os.path.isabs(depth_dir) else os.path.join(REPO, depth_dir)


def check_openloris_depth_cache(text: str, seq: str) -> Optional[str]:
    """Check the depth cache that the settings text ``text`` needs on OpenLORIS sequence ``seq`` (A14).

    Returns the path of the cache's manifest.json, or None when the text has no UCSLAM.DepthDir (a stock run).
    Raises RuntimeError when the cache is missing or incomplete, so that no run starts and no record is written.

    PriorNet looks for the cache once, at the first keyframe. When it finds none, no keyframe of the run gets
    depth, even if the cache appears later (PriorNet::ProcessKeyFrame). experiments/precompute_depth.py writes
    manifest.json after the last map. So the cache counts as complete only when
    <DepthDir>/openloris/<seq>/<UCSLAM.DepthModel>/manifest.json exists and names that model, every frame was
    written or kept (n_written + n_skipped_existing == n_frames), n_frames equals the number of image stamps in
    data/openloris/<seq>/mav0/timestamps.txt, and each of these stamps has its map file.
    """
    m = _DEPTH_DIR_LINE.search(text)
    if m is None:
        return None
    mm = _DEPTH_MODEL_LINE.search(text)
    if mm is None or not mm.group(1):
        raise RuntimeError(f"OpenLORIS {seq}: the settings name UCSLAM.DepthDir but no UCSLAM.DepthModel, "
                           "so the depth cache cannot be checked")
    model = mm.group(1)
    cache = os.path.join(_depth_base(m.group(2)), "openloris", seq, model)
    manifest = os.path.join(cache, "manifest.json")
    hint = ("A UC-SLAM run would get no depth (A14). Complete the cache with experiments/precompute_depth.py "
            "before any UC-SLAM run on this sequence.")
    try:
        with open(manifest) as f:
            man = json.load(f)
    except (OSError, ValueError) as e:
        raise RuntimeError(f"OpenLORIS {seq}: no readable depth-cache manifest {manifest} "
                           f"({e.__class__.__name__}). {hint}") from None
    stamps_path = stamps_file("openloris", "mono_inertial", seq)
    with open(stamps_path) as f:
        stamps = [s.split()[0] for s in (line.strip() for line in f) if s and not s.startswith("#")]
    problems = []
    if man.get("model") != model:
        problems.append(f"the manifest names model {man.get('model')!r}, not UCSLAM.DepthModel {model!r}")
    try:
        n_frames = int(man["n_frames"])
        n_done = int(man["n_written"]) + int(man["n_skipped_existing"])
    except (KeyError, TypeError, ValueError):
        n_frames = n_done = None
        problems.append("the manifest has no n_frames, n_written or n_skipped_existing")
    if n_frames is not None:
        if n_done != n_frames:
            problems.append(f"n_written + n_skipped_existing is {n_done}, not n_frames {n_frames}")
        if n_frames != len(stamps):
            problems.append(f"n_frames is {n_frames}, but {stamps_path} lists {len(stamps)} images")
    names = set(os.listdir(cache))
    missing = [s for s in stamps if f"{s}.npy" not in names]
    if missing:
        problems.append(f"{len(missing)} of the {len(stamps)} image stamps have no map file (first: {missing[0]})")
    if problems:
        raise RuntimeError(f"OpenLORIS {seq}: depth cache {cache} is incomplete: " + "; ".join(problems) +
                           f". {hint}")
    return manifest


def resolve_settings(settings: str, wd: str, dataset: Optional[str] = None, seq: Optional[str] = None) -> str:
    """The settings file the binary reads.

    A relative ``UCSLAM.DepthDir`` is relative to the repository root
    (scripts/gen_ucslam_yaml.py). The binary runs in the run directory, so the
    path is made absolute in a copy written there as settings.yaml. A file
    without the key, or with an absolute path, is used as it is.

    OpenLORIS (A14 note of 2026-10-03): the time ranges of some sequences of different scenes overlap, and the
    fork's PriorNet finds its depth cache by timestamp over every sequence of the dataset (PriorNet::LocateSequence,
    1 ms tolerance). So each OpenLORIS run gets its own depth root, ``<wd>/depth_root/openloris/<seq>``, a symlink
    to the cache of its own sequence only, and the copy points UCSLAM.DepthDir there. Before that,
    check_openloris_depth_cache raises when the cache is missing or incomplete. The caller (run_one) then starts no
    binary and writes no log and no record.
    """
    try:
        with open(settings) as f:
            text = f.read()
    except OSError:
        return settings
    m = _DEPTH_DIR_LINE.search(text)
    if m is None:
        return settings
    base = _depth_base(m.group(2))
    if dataset == "openloris" and seq:
        check_openloris_depth_cache(text, seq)
        root = os.path.join(wd, "depth_root")
        link = os.path.join(root, "openloris", seq)
        os.makedirs(os.path.dirname(link), exist_ok=True)
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(os.path.join(base, "openloris", seq), link)
        target = root
    elif os.path.isabs(m.group(2)):
        return settings
    else:
        target = base
    text = _DEPTH_DIR_LINE.sub(lambda g: f'{g.group(1)}"{target}"', text)
    out = os.path.join(wd, "settings.yaml")
    with open(out, "w") as f:
        f.write(text)
    return out


def build_command(dataset: str, mode: str, seq: str, run_name: str) -> List[str]:
    """Argument list of the example, without the nice prefix.

    The binary comes from the configured tree, the settings file from
    --settings or the stock tree, and the vocabulary, timestamp lists and IMU
    files always from the stock tree.
    """
    if mode == PSEUDO_RGBD_MODE:
        return build_command_rgbd(dataset, seq, run_name)
    spec = MODES[(dataset, mode)]
    exe = os.path.join(_TREE, spec["exe"])
    settings = _SETTINGS or os.path.join(ORB, spec["yaml"])
    if dataset == "euroc":
        return [exe, VOCAB, settings, seq_dir(dataset, seq), stamps_file(dataset, mode, seq), run_name]
    if dataset == "openloris":
        settings = (_SETTINGS or os.path.join(REPO, spec["yaml"])).replace("{robot}", openloris_robot(seq))
        return [exe, VOCAB, settings, seq_dir(dataset, seq), stamps_file(dataset, mode, seq), run_name]
    if dataset == "tumvi":
        imgs = os.path.join(seq_dir(dataset, seq), "mav0", "cam0", "data")
        imu = os.path.join(ORB, spec["imu"].format(seq=seq))
        return [exe, VOCAB, settings, imgs, stamps_file(dataset, mode, seq), imu, run_name]
    raise ValueError(dataset)


def build_command_rgbd(dataset: str, seq: str, run_name: str) -> List[str]:
    """Argument list of the pseudo-RGB-D driver, without the nice prefix.

    Layout: exe vocabulary settings image_dir depth_dir times_file imu_file run_name.
    The settings file sits at index 2 like in the stock commands, so the run
    record reads it from the same place. --settings replaces the default file.
    """
    spec = MODES[(dataset, PSEUDO_RGBD_MODE)]
    settings = _SETTINGS or os.path.join(REPO, spec["yaml"])
    depth = os.path.join(PSEUDO_DEPTH_ROOT, dataset, seq, "depth")
    stamps = stamps_file(dataset, PSEUDO_RGBD_MODE, seq)
    if dataset == "euroc":
        imgs = os.path.join(seq_dir(dataset, seq), "mav0", "cam0", "data")
        imu = os.path.join(seq_dir(dataset, seq), "mav0", "imu0", "data.csv")
    elif dataset == "tumvi":
        imgs = os.path.join(PSEUDO_DEPTH_ROOT, dataset, seq, "cam0_rect")
        imu = os.path.join(ORB, spec["imu"].format(seq=seq))
    else:
        raise ValueError(dataset)
    return [spec["exe"], VOCAB, settings, imgs, depth, stamps, imu, run_name]


def sequence_duration_s(dataset: str, mode: str, seq: str) -> float:
    """Span of the shipped timestamp list in seconds."""
    ts = []
    with open(stamps_file(dataset, mode, seq)) as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                ts.append(float(s.split()[0]))
    if len(ts) < 2:
        raise ValueError(f"timestamp list {stamps_file(dataset, mode, seq)} is too short")
    span = ts[-1] - ts[0]
    return span / 1e9 if span > 1e9 else span


# ---------------------------------------------------------------------------
# Trajectory files
# ---------------------------------------------------------------------------


def read_tum(path: str) -> np.ndarray:
    """(N, 8) array of a TUM file: t x y z qx qy qz qw. Comment lines are skipped."""
    rows = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.replace(",", " ").split()
            if len(parts) < 8:
                continue
            rows.append([float(p) for p in parts[:8]])
    if not rows:
        return np.zeros((0, 8))
    return np.asarray(rows, dtype=float)


def convert_saver_output(src: str, dst: str, stamp_ns_threshold: float) -> int:
    """Rewrite an ORB-SLAM3 EuRoC-saver file as TUM with stamps in seconds.

    Returns the number of poses written. Stamps above stamp_ns_threshold are
    divided by 1e9. A missing or empty source yields an empty file with a
    header so that downstream tools see a trajectory of zero poses.
    """
    arr = read_tum(src) if os.path.exists(src) else np.zeros((0, 8))
    if arr.shape[0] and np.nanmax(arr[:, 0]) > stamp_ns_threshold:
        arr[:, 0] = arr[:, 0] / 1e9
    with open(dst, "w") as f:
        f.write("# timestamp[s] tx ty tz qx qy qz qw  (ORB-SLAM3 output, stamps converted to seconds "
                f"from {os.path.basename(src)})\n")
        for r in arr:
            f.write(f"{r[0]:.9f} " + " ".join(f"{v:.9f}" for v in r[1:]) + "\n")
    return int(arr.shape[0])


def largest_internal_gap_s(times: np.ndarray) -> float:
    """Largest gap between consecutive stamps. 0 for fewer than two stamps."""
    if times.size < 2:
        return 0.0
    return float(np.max(np.diff(np.sort(times))))


def path_length_m(times: np.ndarray, xyz: np.ndarray, resample_s: float) -> float:
    """Length of the polyline through the poses after resampling to resample_s.

    The resampling keeps the first pose and every later pose at least
    resample_s after the previously kept one. It suppresses the spurious
    length that 120 Hz mocap noise or 20 Hz pose noise adds.
    """
    if times.size < 2:
        return 0.0
    order = np.argsort(times)
    t, p = times[order], xyz[order]
    keep = [0]
    for i in range(1, t.size):
        if t[i] - t[keep[-1]] >= resample_s:
            keep.append(i)
    q = p[keep]
    return float(np.sum(np.linalg.norm(np.diff(q, axis=0), axis=1)))


# ---------------------------------------------------------------------------
# evo
# ---------------------------------------------------------------------------


def evo_ape_bin() -> str:
    cand = os.path.join(os.path.dirname(sys.executable), "evo_ape")
    if os.path.exists(cand):
        return cand
    found = shutil.which("evo_ape")
    if found is None:
        raise FileNotFoundError("evo_ape not found in the project environment")
    return found


def evo_ape(gt: str, est: str, t_max_diff: float, results_zip: str,
            correct_scale: bool = False) -> Tuple[Optional[dict], str]:
    """Run `evo_ape tum gt est --align --t_max_diff T --save_results zip`.

    Returns (stats, message). stats is the stats.json of the result archive
    (rmse, mean, median, std, min, max, sse) plus n_pairs, the number of
    associated poses read from the archive's error array, or None when evo
    refused, and message holds evo's last lines in that case.
    """
    cmd = [evo_ape_bin(), "tum", gt, est, "--align", "--t_max_diff", str(t_max_diff),
           "--save_results", results_zip, "--no_warnings"]
    if correct_scale:
        cmd.append("--correct_scale")
    if os.path.exists(results_zip):
        os.remove(results_zip)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0 or not os.path.exists(results_zip):
        tail = "\n".join(proc.stdout.strip().splitlines()[-4:])
        return None, tail
    with zipfile.ZipFile(results_zip) as z:
        stats = json.loads(z.read("stats.json").decode())
        try:
            import io
            n_pairs = int(np.load(io.BytesIO(z.read("error_array.npy"))).size)
        except KeyError:
            n_pairs = -1
    stats = {k: float(v) for k, v in stats.items()}
    stats["n_pairs"] = n_pairs
    return stats, ""


def associate(gt: np.ndarray, est: np.ndarray, t_max_diff: float) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest-stamp association of est to gt within t_max_diff, one gt pose per estimate.

    Returns index arrays (i_gt, i_est) of the matched rows. It mirrors evo's
    association closely enough for the scale error and the end drift; the
    reported ATE itself always comes from the evo_ape command line.
    """
    if gt.shape[0] == 0 or est.shape[0] == 0:
        return np.zeros(0, int), np.zeros(0, int)
    tg = gt[:, 0]
    order = np.argsort(tg)
    tg_sorted = tg[order]
    i_gt, i_est = [], []
    for j, t in enumerate(est[:, 0]):
        k = np.searchsorted(tg_sorted, t)
        best, bestd = -1, t_max_diff
        for cand in (k - 1, k):
            if 0 <= cand < tg_sorted.size:
                d = abs(tg_sorted[cand] - t)
                if d <= bestd:
                    best, bestd = cand, d
        if best >= 0:
            i_gt.append(order[best])
            i_est.append(j)
    return np.asarray(i_gt, int), np.asarray(i_est, int)


def umeyama(x: np.ndarray, y: np.ndarray, with_scale: bool) -> Tuple[np.ndarray, np.ndarray, float]:
    """evo's Umeyama alignment of x onto y (both (3, N)): y ~ c R x + t."""
    from evo.core.geometry import umeyama_alignment
    r, t, c = umeyama_alignment(x, y, with_scale)
    return r, t, float(c)


def sim3_scale_error_pct(gt: np.ndarray, est: np.ndarray, t_max_diff: float) -> float:
    """|1 - s| in percent from a Sim(3) Umeyama fit of the associated positions."""
    i_gt, i_est = associate(gt, est, t_max_diff)
    if i_gt.size < 3:
        return float("nan")
    _, _, c = umeyama(est[i_est, 1:4].T, gt[i_gt, 1:4].T, True)
    return float(abs(1.0 - c) * 100.0)


def gt_segments(gt_times: np.ndarray, gap_split_s: float) -> Optional[Tuple[float, float]]:
    """(end of start segment, begin of end segment) around the largest gap, if it exceeds gap_split_s."""
    if gt_times.size < 2:
        return None
    t = np.sort(gt_times)
    d = np.diff(t)
    i = int(np.argmax(d))
    if d[i] < gap_split_s:
        return None
    return float(t[i]), float(t[i + 1])


def end_drift(gt: np.ndarray, est: np.ndarray, t_max_diff: float, gap_split_s: float) -> dict:
    """SE(3) alignment on the start segment, position error on the end segment.

    Returns a dict with end_drift_m (error at the last matched end-segment
    keyframe), end_rmse_m (RMSE over the end segment), n_start, n_end, and the
    segment boundaries. NaN entries when a segment has no matches.
    """
    out = {"end_drift_m": float("nan"), "end_rmse_m": float("nan"), "n_start": 0, "n_end": 0,
           "t_split": None}
    seg = gt_segments(gt[:, 0], gap_split_s)
    if seg is None:
        return out
    t_a, t_b = seg
    out["t_split"] = [t_a, t_b]
    i_gt, i_est = associate(gt, est, t_max_diff)
    if i_gt.size == 0:
        return out
    t_match = est[i_est, 0]
    m_start = t_match <= t_a
    m_end = t_match >= t_b
    out["n_start"] = int(m_start.sum())
    out["n_end"] = int(m_end.sum())
    if out["n_start"] < 3 or out["n_end"] < 1:
        return out
    r, t, _ = umeyama(est[i_est[m_start], 1:4].T, gt[i_gt[m_start], 1:4].T, False)
    p_est = (r @ est[i_est[m_end], 1:4].T + t[:, None]).T
    err = np.linalg.norm(p_est - gt[i_gt[m_end], 1:4], axis=1)
    last = int(np.argmax(t_match[m_end]))
    out["end_drift_m"] = float(err[last])
    out["end_rmse_m"] = float(np.sqrt(np.mean(err ** 2)))
    return out


# ---------------------------------------------------------------------------
# Log parsing and the failure rule
# ---------------------------------------------------------------------------


_PRIORNET_LINE = re.compile(r"^\[UCSLAM\] priornet keyframes_received=(\d+) with_depth=(\d+) (.*)$", re.M)


def priornet_counters(text: str) -> Optional[Dict[str, int]]:
    """The keyframe counters PriorNet prints at shutdown, or None when PriorNet was off."""
    m = _PRIORNET_LINE.search(text)
    if m is None:
        return None
    out = {"keyframes_received": int(m.group(1)), "with_depth": int(m.group(2))}
    for key, val in re.findall(r"(\w+)=(\d+)", m.group(3)):
        out[key] = int(val)
    return out


def priornet_warnings(counters: Optional[Dict[str, int]]) -> List[str]:
    """A run whose keyframes got no network depth ran as the system without the depth factor.

    This is a warning, not part of the failure rule of protocol Section 2. The usual cause is a
    wrong UCSLAM.DepthDir or a sequence without a depth cache (experiments/precompute_depth.py).
    """
    if not counters:
        return []
    warn = []
    if counters.get("keyframes_received", 0) > 0 and counters.get("with_depth", 0) == 0:
        warn.append("PriorNet received keyframes but found depth for none of them")
    if counters.get("no_sequence", 0) > 0:
        warn.append(f"PriorNet found no depth cache for {counters['no_sequence']} keyframe(s)")
    return warn


# PriorNet prints this once, on stderr, when its first depth-cache lookup fails (PriorNet::ProcessKeyFrame).
_PRIORNET_NO_CACHE = "[UCSLAM] PriorNet: no depth cache holds a map"


def openloris_pipeline_errors(counters: Optional[Dict[str, int]], n_kf: int, text: str) -> List[str]:
    """Why an OpenLORIS run ran without its depth prior, or [] (A14).

    Such a run is a pipeline error, not a result of the method. score_run stores the list as rec["pipeline_error"]
    and leaves "failed" to the failure rule. scripts/diag/openloris_pairs.sh stops on a non-empty list.
    The causes:
    - PriorNet gave depth to no keyframe, although the run made keyframes (``n_kf`` saved, or some received).
    - PriorNet found no depth cache for some keyframe (no_sequence above 0).
    - The run printed no counters (killed or crashed before shutdown), and its log has PriorNet's message that it
      found no depth cache.
    A run without any keyframe fails under the failure rule and gets no mark. Stock ORB-SLAM3 prints neither the
    counters nor the message.
    """
    errs = []
    if counters:
        received = counters.get("keyframes_received", 0)
        if counters.get("with_depth", 0) == 0 and (received > 0 or n_kf > 0):
            errs.append(f"PriorNet gave depth to none of the {received} keyframes it received ({n_kf} saved)")
        if counters.get("no_sequence", 0) > 0:
            errs.append(f"PriorNet found no depth cache for {counters['no_sequence']} keyframe(s)")
    elif _PRIORNET_NO_CACHE in text:
        errs.append("the log says PriorNet found no depth cache, and the run printed no counters")
    return errs


def parse_log(text: str) -> Dict[str, int]:
    """Count the ORB-SLAM3 log lines that the failure rule can read."""
    return {key: text.count(marker) for key, marker in LOG_MARKERS.items()}


def _isnan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def openloris_coverage(gt: np.ndarray, kf: np.ndarray) -> dict:
    """Share of the ground-truth span that the keyframe trajectory spans (A14).

    A system that initializes late, or whose saved map starts late after a reset, covers less. The numerator is
    the overlap of [first keyframe, last keyframe] with [first, last] ground-truth stamp.
    """
    if gt.shape[0] < 2:
        return {"gt_span_s": float("nan"), "coverage": float("nan")}
    g0, g1 = float(gt[0, 0]), float(gt[-1, 0])
    if kf.shape[0] < 2:
        return {"gt_span_s": g1 - g0, "coverage": 0.0}
    k0, k1 = float(kf[0, 0]), float(kf[-1, 0])
    return {"gt_span_s": g1 - g0, "coverage": max(0.0, min(k1, g1) - max(k0, g0)) / (g1 - g0),
            "kf_start_delay_s": k0 - g0}


def openloris_record_fields(seq: str, gt: np.ndarray, traj: np.ndarray, cfg: dict) -> dict:
    """The fields the OpenLORIS failure rule reads, for any runner (A14): scene, ground-truth path length and the
    coverage of ``traj`` (the scored trajectory: keyframes, or the output trajectory of a system without them)."""
    out = {"scene": openloris_scene(seq), "path_len_source": "ground truth",
           "path_len_m": path_length_m(gt[:, 0], gt[:, 1:4], cfg["path_resample_s"])}
    out.update(openloris_coverage(gt, traj))
    return out


def apply_failure_rule(rec: dict, dataset: str, cfg: dict) -> Tuple[bool, List[str], List[str]]:
    """(failed, reasons, unchecked) for one scored run record."""
    reasons: List[str] = []
    unchecked = ["RECENTLY_LOST spells between 1 s and 5 s are carried by IMU prediction and "
                 "saved as tracked; only spells over 5 s print 'Track Lost...'"]
    if rec.get("timed_out"):
        reasons.append("process timed out")
    if rec.get("returncode", 0) != 0 and not rec.get("timed_out"):
        reasons.append(f"process exited with code {rec.get('returncode')}")
    if rec.get("n_kf", 0) == 0:
        reasons.append("no keyframe trajectory written")
    ate = rec.get("ate_kf_m")
    if _isnan(ate):
        reasons.append("evo could not score the keyframe trajectory")
    else:
        if dataset == "euroc" and ate > cfg["fail_ate_euroc_m"]:
            reasons.append(f"ATE {ate:.3f} m above {cfg['fail_ate_euroc_m']} m")
        if dataset == "tumvi":
            path = rec.get("path_len_m")
            if _isnan(path) or path <= 0:
                unchecked.append("path length unavailable, 5 percent rule not applied")
            else:
                ref = rec.get("ate_segments_m", ate) if rec.get("long_sequence") else ate
                if ref > cfg["fail_ate_tumvi_frac"] * path:
                    reasons.append(f"ATE {ref:.3f} m above {100 * cfg['fail_ate_tumvi_frac']:.0f} percent "
                                   f"of the {path:.1f} m path")
            if rec.get("long_sequence") and _isnan(rec.get("end_drift_m", float("nan"))):
                reasons.append("no keyframe with ground truth in the end segment")
        if dataset == "openloris":
            lim = cfg["fail_ate_openloris_m"][rec.get("scene") or openloris_scene(rec["seq"])]
            if ate > lim:
                reasons.append(f"ATE {ate:.3f} m above {lim} m (A14, the scene threshold of the dataset paper)")
    if dataset == "openloris":
        cov = rec.get("coverage", float("nan"))
        if _isnan(cov) or cov < cfg["min_coverage_openloris"]:
            reasons.append(f"keyframe trajectory covers {cov:.2f} of the ground-truth span, below "
                           f"{cfg['min_coverage_openloris']} (A14)")
    logc = rec.get("log", {})
    if logc.get("n_maps_created", 1) > 1:
        reasons.append(f"{logc['n_maps_created'] - 1} new Atlas map(s)")
    if logc.get("n_track_lost", 0) > 0:
        reasons.append("'Track Lost...' in the log (over 5 s in RECENTLY_LOST)")
    gap = rec.get("frame_max_gap_s", 0.0)
    period = cfg["frame_period_s_openloris"] if dataset == "openloris" else cfg["frame_period_s"]
    if gap - period > cfg["fail_lost_s"]:
        reasons.append(f"gap of {gap:.2f} s in the saved frame trajectory")
    n_resets = logc.get("n_resets_imu", 0) + logc.get("n_resets_before_init", 0)
    if n_resets:
        unchecked.append(f"{n_resets} active-map reset(s) before IMU initialization discarded the early "
                         "trajectory; not a new Atlas map under the rule")
    return bool(reasons), reasons, unchecked


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------


def run_dir(mode: str, seq: str) -> str:
    """runs/orbslam3/<mode>/<seq>, with a <tag>/ subdirectory when a tag is configured."""
    d = os.path.join(RUNS_ROOT, mode, seq)
    return os.path.join(d, _TAG) if _TAG else d


def score_run(dataset: str, mode: str, seq: str, k: int, cfg: dict, rec: dict) -> dict:
    """Convert the saver output, run evo, apply the failure rule. Extends rec in place."""
    d = run_dir(mode, seq)
    name = f"run{k}"
    kf_tum = os.path.join(d, f"{name}.tum")
    fr_tum = os.path.join(d, f"{name}_frames.tum")
    rec["n_kf"] = convert_saver_output(os.path.join(d, f"kf_{name}.txt"), kf_tum, cfg["stamp_ns_threshold"])
    rec["n_frames"] = convert_saver_output(os.path.join(d, f"f_{name}.txt"), fr_tum, cfg["stamp_ns_threshold"])
    gt_path = gt_file(dataset, seq)
    gt = read_tum(gt_path)
    kf = read_tum(kf_tum)
    fr = read_tum(fr_tum)
    rec["long_sequence"] = dataset == "tumvi" and seq in TUMVI_LONG

    t = cfg["t_max_diff_s"]
    stats, msg = (evo_ape(gt_path, kf_tum, t, os.path.join(d, f"{name}_ape_kf.zip"))
                  if rec["n_kf"] else (None, "empty keyframe trajectory"))
    rec["ate_kf_m"] = stats["rmse"] if stats else float("nan")
    rec["ate_kf_stats"] = stats
    rec["evo_msg_kf"] = msg
    fstats, fmsg = (evo_ape(gt_path, fr_tum, t, os.path.join(d, f"{name}_ape_frames.zip"))
                    if rec["n_frames"] else (None, "empty frame trajectory"))
    rec["ate_frame_m"] = fstats["rmse"] if fstats else float("nan")
    rec["evo_msg_frames"] = fmsg
    rec["scale_err_pct"] = sim3_scale_error_pct(gt, kf, t) if rec["n_kf"] else float("nan")
    rec["frame_max_gap_s"] = largest_internal_gap_s(fr[:, 0]) if rec["n_frames"] else 0.0
    rec["kf_span_s"] = float(kf[-1, 0] - kf[0, 0]) if rec["n_kf"] > 1 else 0.0

    if dataset == "tumvi":
        if rec["long_sequence"]:
            rec["path_len_m"] = (path_length_m(fr[:, 0], fr[:, 1:4], cfg["path_resample_s"])
                                 if rec["n_frames"] else float("nan"))
            rec["path_len_source"] = "estimated frame trajectory"
            drift = end_drift(gt, kf, t, cfg["gt_gap_split_s"])
            rec.update(drift)
            rec["end_drift_pct"] = (100.0 * drift["end_drift_m"] / rec["path_len_m"]
                                    if np.isfinite(drift["end_drift_m"]) and rec["path_len_m"] > 0
                                    else float("nan"))
            rec["ate_segments_m"] = rec["ate_kf_m"]
        else:
            rec["path_len_m"] = path_length_m(gt[:, 0], gt[:, 1:4], cfg["path_resample_s"])
            rec["path_len_source"] = "ground truth"
    if dataset == "openloris":
        rec.update(openloris_record_fields(seq, gt, kf, cfg))
    with open(os.path.join(d, f"{name}.log"), errors="replace") as f:
        text = f.read()
    rec["log"] = parse_log(text)
    rec["priornet"] = priornet_counters(text)
    rec["warnings"] = priornet_warnings(rec["priornet"])
    for w in rec["warnings"]:
        log(f"WARNING {mode} {seq} {name}: {w}")
    if dataset == "openloris":
        # A14: a UC-SLAM run without its depth prior is a pipeline error, not a result. The mark leaves "failed"
        # to the failure rule. scripts/diag/openloris_pairs.sh stops on a marked record.
        rec["pipeline_error"] = openloris_pipeline_errors(rec["priornet"], rec["n_kf"], text)
        for e in rec["pipeline_error"]:
            log(f"PIPELINE ERROR {mode} {seq} {name}: {e} (A14: a pipeline error, not a result)")
    failed, reasons, unchecked = apply_failure_rule(rec, dataset, cfg)
    rec["failed"] = failed
    rec["fail_reasons"] = reasons
    rec["unchecked"] = unchecked
    return rec


def run_one(dataset: str, mode: str, seq: str, k: int, cfg: dict, force: bool = False) -> dict:
    d = run_dir(mode, seq)
    os.makedirs(d, exist_ok=True)
    name = f"run{k}"
    rec_path = os.path.join(d, f"{name}.json")
    if os.path.exists(rec_path) and not force:
        with open(rec_path) as f:
            rec = json.load(f)
        log(f"skip {mode} {seq} {name}: already scored (ATE {rec.get('ate_kf_m', float('nan')):.4f} m)")
        return rec
    # Every run gets its own working directory, runs/.../run<k>_wd/, so files the binary writes
    # relative to its working directory (the covariance dump in marginals/) never collide between
    # runs of one cell, even in parallel. The two saver outputs are moved back into the cell after
    # the run, where the scorer reads them.
    wd = os.path.join(d, f"{name}_wd")
    if os.path.isdir(wd):
        shutil.rmtree(wd)
    os.makedirs(wd)
    for stale in (f"kf_{name}.txt", f"f_{name}.txt"):
        p = os.path.join(d, stale)
        if os.path.exists(p):
            os.remove(p)
    duration = sequence_duration_s(dataset, mode, seq)
    timeout = (cfg["timeout_factor_openloris"] if dataset == "openloris" else cfg["timeout_factor"]) * duration + cfg["timeout_extra_s"]
    cmd = ["nice", "-n", str(cfg["nice"])] + build_command(dataset, mode, seq, name)
    rec = {"dataset": dataset, "mode": mode, "seq": seq, "run": k,
           "seed_note": "ORB-SLAM3 takes no seed; the run index labels the run only",
           "command": cmd, "started": dt.datetime.now().isoformat(timespec="seconds"),
           "sequence_duration_s": duration, "timeout_s": timeout,
           "tree": os.path.relpath(_TREE, REPO), "settings": os.path.relpath(cmd[5], REPO), "tag": _TAG,
           "stock_tree": is_stock_tree()}
    cmd[5] = resolve_settings(cmd[5], wd, dataset, seq)   # the record keeps the committed settings path
    log(f"start {mode} {seq} {name} (sequence {duration:.0f} s)")
    t0 = time.time()
    timed_out = False
    with open(os.path.join(d, f"{name}.log"), "w") as logf:
        logf.write("# " + " ".join(cmd) + "\n")
        logf.flush()
        try:
            proc = subprocess.run(cmd, cwd=wd, stdout=logf, stderr=subprocess.STDOUT, timeout=timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out, rc = True, -1
            logf.write("\n# killed by eval/run_orbslam3.py after timeout\n")
    rec["wall_s"] = time.time() - t0
    for out in (f"kf_{name}.txt", f"f_{name}.txt"):
        if os.path.exists(os.path.join(wd, out)):
            os.replace(os.path.join(wd, out), os.path.join(d, out))
    rec["workdir"] = os.path.relpath(wd, REPO)
    rec["returncode"] = rc
    rec["timed_out"] = timed_out
    score_run(dataset, mode, seq, k, cfg, rec)
    with open(rec_path, "w") as f:
        json.dump(rec, f, indent=1, default=_json_default)
    status = "FAIL " + "; ".join(rec["fail_reasons"]) if rec["failed"] else "ok"
    log(f"done  {mode} {seq} {name}: ATE_kf {rec['ate_kf_m']:.4f} m, ATE_frames {rec['ate_frame_m']:.4f} m, "
        f"{rec['n_kf']} KFs, wall {rec['wall_s']:.0f} s, {status}")
    return rec


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(str(type(o)))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def summarize(records: Sequence[dict]) -> dict:
    """Per-sequence summary over the runs of one (dataset, mode)."""
    from eval.stats import median_iqr
    by_seq: Dict[str, List[dict]] = {}
    for r in records:
        by_seq.setdefault(r["seq"], []).append(r)
    out = {}
    for seq, rs in by_seq.items():
        rs = sorted(rs, key=lambda r: r["run"])
        ok = [r for r in rs if not r["failed"]]
        ate = [r["ate_kf_m"] for r in ok]
        med, q1, q3, iqr = median_iqr(ate)
        s = {"n_runs": len(rs), "n_ok": len(ok), "n_fail": len(rs) - len(ok),
             "ate_kf_m_median": med, "ate_kf_m_q1": q1, "ate_kf_m_q3": q3, "ate_kf_m_iqr": iqr,
             "ate_kf_m_all_runs": [r["ate_kf_m"] for r in rs],
             "ate_kf_m_ok_runs": ate,
             "ate_frame_m_median": median_iqr([r["ate_frame_m"] for r in ok])[0],
             "ate_frame_m_q1": median_iqr([r["ate_frame_m"] for r in ok])[1],
             "ate_frame_m_q3": median_iqr([r["ate_frame_m"] for r in ok])[2],
             "ate_frame_m_iqr": median_iqr([r["ate_frame_m"] for r in ok])[3],
             "ate_frame_m_ok_runs": [r["ate_frame_m"] for r in ok],
             "scale_err_pct_median": median_iqr([r["scale_err_pct"] for r in ok])[0],
             "scale_err_pct_ok_runs": [r["scale_err_pct"] for r in ok],
             "wall_s_median": median_iqr([r["wall_s"] for r in rs])[0],
             "wall_s_all_runs": [r["wall_s"] for r in rs],
             "sequence_duration_s": rs[0]["sequence_duration_s"],
             "fail_reasons": {},
             "n_kf_median": median_iqr([r["n_kf"] for r in ok])[0],
             "log_totals": {k: int(sum(r.get("log", {}).get(k, 0) for r in rs)) for k in LOG_MARKERS}}
        for r in rs:
            for reason in r["fail_reasons"]:
                key = re.sub(r"[0-9.]+", "#", reason)
                s["fail_reasons"][key] = s["fail_reasons"].get(key, 0) + 1
        if rs[0].get("long_sequence"):
            s["path_len_m_median"] = median_iqr([r["path_len_m"] for r in ok])[0]
            s["end_drift_m_ok_runs"] = [r["end_drift_m"] for r in ok]
            s["end_drift_pct_ok_runs"] = [r["end_drift_pct"] for r in ok]
            s["end_drift_m_median"], _, _, s["end_drift_m_iqr"] = median_iqr(s["end_drift_m_ok_runs"])
            s["end_drift_pct_median"], _, _, s["end_drift_pct_iqr"] = median_iqr(s["end_drift_pct_ok_runs"])
            s["ate_segments_m_median"] = median_iqr([r["ate_segments_m"] for r in ok])[0]
        elif rs[0]["dataset"] == "tumvi":
            s["path_len_m_gt"] = rs[0].get("path_len_m")
        out[seq] = s
    return out


def write_scores(dataset: str, mode: str, records: Sequence[dict], cfg: dict,
                 path: Optional[str] = None, parallel: Optional[int] = None) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = f"_{_TAG}" if _TAG else ""
    path = path or os.path.join(OUT_DIR, f"scores_{dataset}_{mode}{suffix}.json")
    sha = None
    tree_sha = None
    try:
        sha = subprocess.run(["git", "-C", ORB, "rev-parse", "HEAD"], stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, text=True).stdout.strip() or None
        tree_sha = subprocess.run(["git", "-C", _TREE, "rev-parse", "HEAD"], stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True).stdout.strip() or None
    except OSError:
        pass
    stock = is_stock_tree()
    settings_used = _SETTINGS or "stock settings of the mode"
    provenance = ("stock upstream ORB-SLAM3 binary, not the UC-SLAM system" if stock and _SETTINGS is None else
                  f"binary tree {os.path.relpath(_TREE, REPO)} with settings {os.path.relpath(settings_used, REPO) if _SETTINGS else settings_used}. "
                  "Not the stock baseline" + ("" if stock else ", the fork tree"))
    if mode == PSEUDO_RGBD_MODE:
        provenance = ("pseudo-RGB-D baseline of protocol H3: cpp/pseudo_rgbd/rgbd_inertial_offline linked against the "
                      "stock ORB-SLAM3 library (System::IMU_RGBD), predicted depth from data/pseudo_depth, settings "
                      + (os.path.relpath(_SETTINGS, REPO) if _SETTINGS else
                         MODES[(dataset, PSEUDO_RGBD_MODE)]["yaml"]) + ". Not the UC-SLAM system")
    evo_version = None
    try:
        import evo
        evo_version = evo.__version__
    except Exception:  # pragma: no cover
        pass
    patch_dir = os.path.join(REPO, "cpp", "orbslam3_patches")
    doc = {"meta": {"dataset": dataset, "mode": mode, "date": dt.date.today().isoformat(),
                    "orbslam3_sha": sha, "pinned_sha": cfg["upstream_sha"],
                    "patches": sorted(os.listdir(patch_dir)) if os.path.isdir(patch_dir) else [],
                    "evo_version": evo_version, "t_max_diff_s": cfg["t_max_diff_s"],
                    "alignment": "SE(3)", "trajectory": "keyframe (frame trajectory as secondary)",
                    "realtime": "the examples pace to the image timestamps, so yes",
                    "seed_note": "ORB-SLAM3 takes no seed; run<k> is a label",
                    "provenance": provenance,
                    "tree": os.path.relpath(_TREE, REPO), "tree_git_head": tree_sha,
                    "settings": os.path.relpath(_SETTINGS, REPO) if _SETTINGS else None, "tag": _TAG,
                    "parallel": parallel, "nice": cfg["nice"]},
           "summary": summarize(records),
           "runs": [{k: v for k, v in r.items() if k not in ("command", "ate_kf_stats")} for r in
                    sorted(records, key=lambda r: (r["seq"], r["run"]))]}
    if mode == PSEUDO_RGBD_MODE:
        doc["meta"]["settings"] = (os.path.relpath(_SETTINGS, REPO) if _SETTINGS
                                   else MODES[(dataset, PSEUDO_RGBD_MODE)]["yaml"])
        doc["meta"]["driver"] = os.path.relpath(PSEUDO_RGBD_EXE, REPO)
    with open(path, "w") as f:
        json.dump(doc, f, indent=1, default=_json_default)
    return path


def load_existing_records(dataset: str, mode: str, seqs: Sequence[str]) -> List[dict]:
    recs = []
    for seq in seqs:
        d = run_dir(mode, seq)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if re.fullmatch(r"run\d+\.json", fn):
                with open(os.path.join(d, fn)) as f:
                    r = json.load(f)
                if r.get("dataset") == dataset:
                    recs.append(r)
    return recs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["euroc", "tumvi", "openloris"], required=True)
    ap.add_argument("--mode", choices=["mono_inertial", "stereo_inertial", PSEUDO_RGBD_MODE], required=True)
    ap.add_argument("--seqs", nargs="*", help="sequence directory names; default: all EuRoC or the TUM-VI rooms")
    ap.add_argument("--runs", type=int, default=None, help="runs per cell; default orbslam3_baseline.runs")
    ap.add_argument("--run-index", type=int, default=None,
                    help="run only run<K> of each sequence, for paired schedules that start runs of two arms together")
    ap.add_argument("--parallel", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="rerun cells that already have a run<k>.json")
    ap.add_argument("--rescore", action="store_true", help="only rescore existing runs from their saved output")
    ap.add_argument("--out", default=None, help="aggregate JSON path; default eval/out/orbslam3/scores_<dataset>_<mode>[_<tag>].json")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--tree", default=None,
                    help="root of the ORB-SLAM3 tree whose example binaries run, e.g. cpp/UC_SLAM3; default the stock tree")
    ap.add_argument("--settings", default=None,
                    help="settings YAML that replaces the stock one, e.g. configs/orbslam3/euroc_mono_inertial_ucslam.yaml")
    ap.add_argument("--tag", default=None,
                    help="run files go to runs/orbslam3/<mode>/<seq>/<tag>/ and the aggregate gets the _<tag> suffix")
    a = ap.parse_args(argv)

    try:
        configure(a.tree, a.settings, a.tag)
    except ValueError as e:
        ap.error(str(e))
    if a.settings and "{robot}" not in _SETTINGS and not os.path.exists(_SETTINGS):
        ap.error(f"settings file {_SETTINGS} does not exist")
    if (a.tree or a.settings) and not a.tag:
        ap.error("--tree or --settings needs --tag so that the runs do not overwrite the stock ones")

    cfg = load_cfg(a.config)
    if (a.dataset, a.mode) not in MODES:
        ap.error(f"no stock example for {a.dataset} {a.mode}")
    seqs = a.seqs or (list(EUROC_SEQS) if a.dataset == "euroc" else
                      OPENLORIS_SEQS if a.dataset == "openloris" else TUMVI_ROOMS)
    for s in seqs:
        if not os.path.isdir(seq_dir(a.dataset, s)):
            ap.error(f"missing sequence {seq_dir(a.dataset, s)}")
        if a.settings and "{robot}" in _SETTINGS:
            if a.dataset != "openloris":
                ap.error("a {robot} placeholder in --settings is for OpenLORIS only")
            if not os.path.exists(_SETTINGS.replace("{robot}", openloris_robot(s))):
                ap.error(f"settings file {_SETTINGS.replace('{robot}', openloris_robot(s))} does not exist")
    n_runs = a.runs or cfg["runs"]
    parallel = a.parallel or cfg["parallel"]
    if not a.rescore:
        exe = build_command(a.dataset, a.mode, seqs[0], "x")[0]
        if not os.access(exe, os.X_OK):
            ap.error(f"{exe} is not built; run scripts/build_orbslam3.sh or scripts/build_ucslam3.sh")
        if not os.path.exists(VOCAB):
            ap.error(f"{VOCAB} missing")

    jobs = [(s, k) for k in range(n_runs) for s in seqs]   # run index outer, so every seq gets a run early
    if a.run_index is not None:
        jobs = [(s, a.run_index) for s in seqs]
    log(f"{a.dataset} {a.mode}: {len(seqs)} sequences x {n_runs} runs, {parallel} in parallel, nice {cfg['nice']}")
    if a.mode == PSEUDO_RGBD_MODE:
        log(f"driver {os.path.relpath(PSEUDO_RGBD_EXE, REPO)}, settings "
            f"{os.path.relpath(build_command(a.dataset, a.mode, seqs[0], 'x')[2], REPO)}, "
            f"tag {_TAG}, run dir {os.path.relpath(run_dir(a.mode, seqs[0]), REPO)}")
    elif not is_stock_tree() or _SETTINGS or _TAG:
        log(f"tree {os.path.relpath(_TREE, REPO)}, settings {os.path.relpath(_SETTINGS, REPO) if _SETTINGS else 'stock'}, "
            f"tag {_TAG}, run dir {os.path.relpath(run_dir(a.mode, seqs[0]), REPO)}")
    records: List[dict] = []
    if a.rescore:
        for s, k in jobs:
            rec_path = os.path.join(run_dir(a.mode, s), f"run{k}.json")
            if not os.path.exists(rec_path):
                continue
            with open(rec_path) as f:
                rec = json.load(f)
            score_run(a.dataset, a.mode, s, k, cfg, rec)
            with open(rec_path, "w") as f:
                json.dump(rec, f, indent=1, default=_json_default)
            records.append(rec)
            log(f"rescored {a.mode} {s} run{k}: ATE_kf {rec['ate_kf_m']:.4f} m, failed={rec['failed']}")
    else:
        done = 0
        t_batch = time.time()
        with cf.ThreadPoolExecutor(max_workers=parallel) as ex:
            futs = {ex.submit(run_one, a.dataset, a.mode, s, k, cfg, a.force): (s, k) for s, k in jobs}
            for fut in cf.as_completed(futs):
                s, k = futs[fut]
                try:
                    records.append(fut.result())
                except Exception as e:  # keep the batch alive, record the crash
                    log(f"ERROR {a.mode} {s} run{k}: {e!r}")
                    records.append({"dataset": a.dataset, "mode": a.mode, "seq": s, "run": k, "failed": True,
                                    "fail_reasons": [f"runner exception {e!r}"], "unchecked": [],
                                    "ate_kf_m": float("nan"), "ate_frame_m": float("nan"),
                                    "scale_err_pct": float("nan"), "wall_s": float("nan"), "n_kf": 0,
                                    "n_frames": 0, "log": {}, "frame_max_gap_s": 0.0,
                                    "sequence_duration_s": float("nan")})
                done += 1
                elapsed = (time.time() - t_batch) / 60
                log(f"progress {done}/{len(jobs)} after {elapsed:.1f} min")
    all_recs = load_existing_records(a.dataset, a.mode, seqs)
    if not all_recs:
        all_recs = records
    if a.rescore:
        # A rescore runs nothing. Keep the value the aggregate recorded when the runs were made.
        old = a.out or os.path.join(OUT_DIR, f"scores_{a.dataset}_{a.mode}" + (f"_{_TAG}" if _TAG else "") + ".json")
        try:
            with open(old) as f:
                parallel = json.load(f)["meta"].get("parallel")
        except (OSError, ValueError, KeyError):
            parallel = None
    path = write_scores(a.dataset, a.mode, all_recs, cfg, a.out, parallel=parallel)
    summ = summarize(all_recs)
    log(f"wrote {path}")
    for seq in seqs:
        if seq in summ:
            s = summ[seq]
            log(f"  {seq:16s} ATE_kf median {100 * s['ate_kf_m_median']:.1f} cm (IQR {100 * s['ate_kf_m_iqr']:.1f}) "
                f"ok {s['n_ok']}/{s['n_runs']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
