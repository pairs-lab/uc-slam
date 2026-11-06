#!/usr/bin/env python
"""Generate the UC-SLAM fork settings files from the stock ORB-SLAM3 YAMLs.

ORB-SLAM3 reads one YAML per run through ``Settings::readParameter``. The fork
adds keys under the ``UCSLAM.`` prefix. Their values live in the ``ucslam``
block of ``configs/default.yaml``, the single source. This script copies a
stock camera YAML from ``third_party/ORB_SLAM3/Examples`` verbatim and appends
the ``UCSLAM.*`` block. The stock keys are untouched, so the fork binary reads
the same camera, IMU and ORB parameters as the baseline.

Targets, written to ``configs/orbslam3/``:

    euroc_mono_inertial_ucslam.yaml     from Monocular-Inertial/EuRoC.yaml, domain euroc
    euroc_stereo_inertial_ucslam.yaml   from Stereo-Inertial/EuRoC.yaml,    domain euroc
    tumvi_mono_inertial_ucslam.yaml     from Monocular-Inertial/TUM-VI.yaml, domain tumvi

Pointers. A string value of the form ``block.key`` in the ucslam block is a
pointer into ``configs/default.yaml``. It is resolved here, so sigma_0 and the
commit thresholds are typed once in that file.

Types. OpenCV's YAML reader is typed. ``Settings::readParameter<float>`` aborts
on an integer literal. Every float is therefore written with a decimal point
or an exponent, every int without one, and every string in double quotes.

Paths. ``UCSLAM.DepthDir`` defaults to ``data/depth``, relative to the
repository root, so the committed files hold no machine path. The example
binaries run with the run directory as their working directory, where a
relative path would not resolve. ``eval/run_orbslam3.py`` therefore writes a
copy of the settings file with the path made absolute into each run directory
(``resolve_settings``). An absolute ``--depth-dir`` is written as given.

Usage::

    uv run python scripts/gen_ucslam_yaml.py            # write the three files
    uv run python scripts/gen_ucslam_yaml.py --check    # exit 1 when a file on disk differs

The C++ side is ``cpp/UC_SLAM3/src/Settings.cc``, function ``readUcslam``.
Every key it reads has a default that reproduces stock behaviour when the
key is absent, so the stock YAML still parses unchanged.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Any, Dict, List, Tuple

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(REPO, "configs", "default.yaml")
STOCK_EXAMPLES = os.path.join(REPO, "third_party", "ORB_SLAM3", "Examples")
DEFAULT_OUT_DIR = os.path.join(REPO, "configs", "orbslam3")
DEFAULT_DEPTH_DIR = "data/depth"   # relative to the repository root, resolved per run by eval/run_orbslam3.py

# Output name -> (stock YAML relative to Examples/, sensor domain of the per-domain constants)
TARGETS: Dict[str, Tuple[str, str]] = {
    "euroc_mono_inertial_ucslam.yaml": ("Monocular-Inertial/EuRoC.yaml", "euroc"),
    "euroc_stereo_inertial_ucslam.yaml": ("Stereo-Inertial/EuRoC.yaml", "euroc"),
    "tumvi_mono_inertial_ucslam.yaml": ("Monocular-Inertial/TUM-VI.yaml", "tumvi"),
}

# OpenLORIS-Scene (protocol amendment A14). ORB-SLAM3 ships no file for this recording, so the source is the
# repository file that scripts/make_openloris_settings.py derives from the stock RealSense_D435i.yaml, one per robot,
# and the path is relative to the repository. Only the variants listed are written: the stock-equivalent file and
# the adopted variant of A11. The targets are active once ucslam.per_domain.openloris exists, so --check passes
# before the OpenLORIS depth calibration has run.
# Output name -> (source relative to the repository, domain, variants)
OPENLORIS_TARGETS: Dict[str, Tuple[str, str, Tuple]] = {
    f"openloris_mono_inertial_{robot}_ucslam.yaml": (
        f"configs/orbslam3/openloris_mono_inertial_stock_{robot}.yaml", "openloris", (None, "a11r_capgatex"))
    for robot in ("tsinghua", "gaussian")
}

# (UCSLAM key, path inside the ucslam block, type). ``{domain}`` is replaced
# by the target's domain. The order is the order in the written file.
KEYS: List[Tuple[str, str, type]] = [
    ("PriorNet", "prior_net", int),
    ("DepthDir", "depth_dir", str),
    ("DepthModel", "depth_model", str),
    ("DepthMatchTolMs", "queue_match_tol_ms", float),
    ("UseDepthFactor", "use_depth_factor", int),
    ("SigmaNetC0", "per_domain.{domain}.sigma_net_c0", float),
    ("SigmaNetC1", "per_domain.{domain}.sigma_net_c1", float),
    ("LogBias", "per_domain.{domain}.log_bias", float),
    ("NeffOverN", "per_domain.{domain}.neff_over_n", float),
    ("ZRef", "z_ref_m", float),
    ("Sigma0", "sigma_0", float),
    ("SigmaRW", "sigma_rw", float),
    ("ESSDeflation", "ess_deflation", int),
    ("MaxDepthPriorPoints", "max_depth_prior_points", int),
    ("InitPolicy", "init_policy", str),
    ("InitMinWindow", "init_min_window_s", float),
    ("InitMaxWindow", "init_max_window_s", float),
    ("TauS", "tau_s", float),
    ("TauG", "tau_g_deg", float),
    ("TauBa", "tau_ba", float),
    ("DumpMarginals", "dump_marginals", int),
    ("CovSource", "cov_source", str),
    ("SigmaMerge", "sigma_merge", float),
    ("Diag", "diag", int),
    ("DepthEdgeMinObs", "depth_edge_min_obs", int),
    ("DepthEdgesFixedKFs", "depth_edges_fixed_kfs", int),
    ("ChainFixedBoundary", "chain_fixed_boundary", int),
    ("ImuInitHandoff", "imu_init_handoff", int),
    ("LogScaleFixed", "log_scale_fixed", int),
    ("DepthGateMode", "depth_gate_mode", int),
    ("DepthEdgesStereoPoints", "depth_edges_stereo_points", int),
    ("LogScaleSeed", "log_scale_seed", int),
    ("DepthEdgesStereoObs", "depth_edges_stereo_obs", int),
    ("DiagLbaDelayMs", "diag_lba_delay_ms", int),
    ("ScaleRefineMaxLog", "scale_refine_max_log", float),
]

# Keys added after the benchmark (amendment A11). A key here is written only when a variant sets it, after the
# KEYS block and in this order (the order of the A11 settings files), so the files of every other variant stay
# as they were.
OPTIONAL_KEYS: List[Tuple[str, type]] = [
    ("LogInitState", int),
    ("YoungMapScaleGate", float),
    ("YoungMapBiasGate", float),
    ("DepthEdgeGateAtAdd", int),
    ("LbaMargDiagOnly", int),
    ("DepthEdgeSubsampleEps", float),
    ("DiagMargCheck", int),
    ("DiagSubsampleCheck", int),
    ("YoungMapGateShadow", int),
]

# The keys docs/fork_design.md names. A test checks that every one is emitted.
DESIGN_KEYS = ("DepthDir", "DepthModel", "UseDepthFactor", "SigmaNetC0", "SigmaNetC1", "ZRef",
               "Sigma0", "SigmaRW", "ESSDeflation", "MaxDepthPriorPoints", "InitPolicy",
               "InitMinWindow", "InitMaxWindow", "TauS", "TauG", "TauBa", "DumpMarginals",
               "CovSource", "SigmaMerge", "LogBias")

_POINTER = re.compile(r"^[a-z_][a-z0-9_]*(\.[a-z_][a-z0-9_]*)+$")


def load_config(path: str = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def lookup(cfg: dict, dotted: str) -> Any:
    """Value at a dotted path. Raises KeyError with the full path when absent."""
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(dotted)
        node = node[part]
    return node


def resolve(cfg: dict, value: Any, depth: int = 0) -> Any:
    """Follow a ``block.key`` pointer string into ``cfg``. Other values pass through.

    A string that looks like a pointer but does not exist in ``cfg`` is left
    as a plain string, so a value such as ``pyramid`` is never mistaken for
    one. Pointers may chain, up to a small depth.
    """
    if isinstance(value, str) and _POINTER.match(value) and depth < 8:
        try:
            return resolve(cfg, lookup(cfg, value), depth + 1)
        except KeyError:
            return value
    return value


def fmt_value(value: Any, typ: type) -> str:
    """Render a value for OpenCV's typed YAML reader."""
    if typ is int:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
            raise TypeError(f"expected an integer, got {value!r}")
        return str(int(value))
    if typ is float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"expected a number, got {value!r}")
        s = f"{float(value):.10g}"
        if "." not in s and "e" not in s and "inf" not in s and "nan" not in s:
            s += ".0"
        return s
    if typ is str:
        s = str(value)
        if '"' in s or "\n" in s:
            raise ValueError(f"string value cannot hold a quote or a newline: {s!r}")
        return f'"{s}"'
    raise TypeError(typ)


def ucslam_entries(cfg: dict, domain: str, depth_dir: str = DEFAULT_DEPTH_DIR) -> List[Tuple[str, str]]:
    """(UCSLAM key, rendered value) pairs for one domain, in file order."""
    block = cfg["ucslam"]
    out = [("Dataset", fmt_value(domain, str))]
    for key, path, typ in KEYS:
        path = path.format(domain=domain)
        raw = lookup(block, path)
        value = resolve(cfg, raw)
        if key == "DepthDir":
            value = os.path.normpath(depth_dir)
        out.append((key, fmt_value(value, typ)))
    return out


def apply_overrides(entries: List[Tuple[str, str]], overrides: Dict[str, Any] | None) -> List[Tuple[str, str]]:
    """Replace the rendered value of each overridden UCSLAM key, keeping its type.

    An override names a key that must already exist, so a typo fails loudly
    instead of adding a key the fork never reads."""
    if not overrides:
        return entries
    types = {key: typ for key, _path, typ in KEYS}
    types.update(dict(OPTIONAL_KEYS))
    known = {k for k, _ in entries}
    unknown = sorted(set(overrides) - known - set(types))
    if unknown:
        raise KeyError(f"override of unknown UCSLAM keys: {unknown}")
    out = [(k, fmt_value(overrides[k], types[k]) if k in overrides else v) for k, v in entries]
    return out + [(k, fmt_value(overrides[k], t)) for k, t in OPTIONAL_KEYS if k in overrides and k not in known]


def generate(stock_text: str, cfg: dict, domain: str, stock_rel: str,
             depth_dir: str = DEFAULT_DEPTH_DIR, overrides: Dict[str, Any] | None = None,
             variant: str | None = None, source_label: str | None = None) -> str:
    """The stock YAML text plus the UCSLAM block, with a named variant's overrides applied."""
    if not stock_text.startswith("%YAML:1.0"):
        raise ValueError("the stock file does not start with %YAML:1.0, refusing to extend it")
    if not stock_text.endswith("\n"):
        stock_text += "\n"
    rule = "#" + "-" * 92
    lines = [
        "",
        rule,
        "# UC-SLAM parameters",
        rule,
        "# Generated by scripts/gen_ucslam_yaml.py from the ucslam block of configs/default.yaml",
        f"# for domain {domain}. Do not edit by hand. Regenerate instead. Every key above this",
        (f"# block is the stock ORB-SLAM3 file Examples/{stock_rel}, copied verbatim." if source_label is None else
         f"# block is {source_label}, copied verbatim."),
        "# A float carries a decimal point because Settings::readParameter<float> is typed.",
    ]
    if variant:
        lines.append(f"# Variant {variant} of ucslam.variants in configs/default.yaml overrides: "
                     + ", ".join(f"{k}={v}" for k, v in sorted((overrides or {}).items())))
    for key, value in apply_overrides(ucslam_entries(cfg, domain, depth_dir), overrides):
        lines.append(f"UCSLAM.{key}: {value}")
    return stock_text + "\n".join(lines) + "\n"


def generate_target(name: str, cfg: dict, stock_root: str = STOCK_EXAMPLES,
                    depth_dir: str = DEFAULT_DEPTH_DIR, variant: str | None = None) -> str:
    stock_rel, domain = TARGETS[name]
    with open(os.path.join(stock_root, stock_rel)) as f:
        stock_text = f.read()
    overrides = variant_overrides(cfg, variant) if variant else None
    return generate(stock_text, cfg, domain, stock_rel, depth_dir, overrides, variant)


def openloris_active(cfg: dict) -> bool:
    """True once the OpenLORIS depth calibration has put its constants in ucslam.per_domain (A14)."""
    return "openloris" in (cfg["ucslam"].get("per_domain") or {})


def generate_openloris_target(name: str, cfg: dict, depth_dir: str = DEFAULT_DEPTH_DIR,
                              variant: str | None = None, repo: str = REPO) -> str:
    src_rel, domain, _variants = OPENLORIS_TARGETS[name]
    with open(os.path.join(repo, src_rel)) as f:
        text = f.read()
    overrides = variant_overrides(cfg, variant) if variant else None
    return generate(text, cfg, domain, src_rel, depth_dir, overrides, variant,
                    source_label=f"the repository file {src_rel} (scripts/make_openloris_settings.py)")


def variant_overrides(cfg: dict, variant: str) -> Dict[str, Any]:
    """The overrides of one named variant, from ucslam.variants of the config."""
    variants = cfg["ucslam"].get("variants") or {}
    if variant not in variants:
        raise KeyError(f"unknown variant '{variant}', known: {sorted(variants)}")
    return dict(variants[variant])


def variant_path(out_dir: str, name: str, variant: str | None) -> str:
    """configs/orbslam3/<target stem>_<variant>.yaml, or the target itself without a variant."""
    if not variant:
        return os.path.join(out_dir, name)
    stem, ext = os.path.splitext(name)
    return os.path.join(out_dir, f"{stem}_{variant}{ext}")


def parse_ucslam_keys(text: str) -> Dict[str, str]:
    """``UCSLAM.<Key>`` -> raw value text, from a generated file."""
    out = {}
    for line in text.splitlines():
        m = re.match(r"^UCSLAM\.(\w+):\s*(.*?)\s*$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def write_all(out_dir: str = DEFAULT_OUT_DIR, cfg: dict | None = None, stock_root: str = STOCK_EXAMPLES,
              depth_dir: str = DEFAULT_DEPTH_DIR, check: bool = False) -> List[str]:
    """Write every target and variant. With ``check`` compare instead and return the file names that differ."""
    cfg = cfg or load_config()
    differ: List[str] = []
    os.makedirs(out_dir, exist_ok=True)
    variants = [None] + sorted((cfg["ucslam"].get("variants") or {}).keys())
    for name, variant in ((n, v) for n in TARGETS for v in variants):
        text = generate_target(name, cfg, stock_root, depth_dir, variant)
        path = variant_path(out_dir, name, variant)
        if check:
            on_disk = open(path).read() if os.path.exists(path) else None
            if on_disk != text:
                differ.append(os.path.basename(path))
            continue
        with open(path, "w") as f:
            f.write(text)
    if openloris_active(cfg):
        for name, (_src, _domain, ol_variants) in OPENLORIS_TARGETS.items():
            for variant in ol_variants:
                text = generate_openloris_target(name, cfg, depth_dir, variant)
                path = variant_path(out_dir, name, variant)
                if check:
                    on_disk = open(path).read() if os.path.exists(path) else None
                    if on_disk != text:
                        differ.append(os.path.basename(path))
                    continue
                with open(path, "w") as f:
                    f.write(text)
    return differ


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--stock-root", default=STOCK_EXAMPLES, help="the Examples/ directory of the stock tree")
    ap.add_argument("--depth-dir", default=DEFAULT_DEPTH_DIR, help="written as UCSLAM.DepthDir. A relative path is relative to the repository root")
    ap.add_argument("--check", action="store_true", help="compare the files on disk with what would be written")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    differ = write_all(a.out_dir, cfg, a.stock_root, a.depth_dir, check=a.check)
    if a.check:
        if differ:
            print("out of date: " + ", ".join(differ))
            return 1
        print(f"{len(TARGETS)} files match the generator")
        return 0
    variants = [None] + sorted((cfg["ucslam"].get("variants") or {}).keys())
    for name in TARGETS:
        for v in variants:
            print(f"wrote {variant_path(a.out_dir, name, v)}")
    if openloris_active(cfg):
        for name, (_src, _domain, ol_variants) in OPENLORIS_TARGETS.items():
            for v in ol_variants:
                print(f"wrote {variant_path(a.out_dir, name, v)}")
    else:
        print("OpenLORIS targets skipped: ucslam.per_domain.openloris is not set yet (A14 depth calibration)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
