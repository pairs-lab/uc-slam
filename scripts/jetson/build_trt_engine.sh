#!/usr/bin/env bash
# Build a TensorRT engine of the UC-SLAM depth network from the ONNX file of export_depth_onnx.py.
#
# An engine runs only on the GPU and the TensorRT version it was built with. Build it on the Jetson itself.
# On a Jetson the script uses /usr/src/tensorrt/bin/trtexec. Without trtexec it builds with the TensorRT Python
# API (the desktop test path). It writes <engine>.json next to the engine: the ONNX record plus the precision,
# the FP32 layers, TF32 and the TensorRT version. ucslam/trt_depth.py refuses an engine whose record names
# another model or precision.
#
# usage: build_trt_engine.sh ONNX ENGINE [--fp32] [--plain-fp16] [--workspace MiB] [--python PY] [--no-trtexec]
#   (default)     FP16 engine (backend trt-fp16) that keeps the LayerNormalization and Softmax layers and their
#                 outputs in FP32 (obey precision constraints). This is the tested flight setup. trtexec and the
#                 Python builder get the same layer precisions and layer output types. Needs the tensorrt Python
#                 package (it lists the layer names, also for trtexec)
#   --fp32        strict FP32 engine, TF32 off (backend trt-fp32). Slower
#   --plain-fp16  FP16 everywhere, no FP32 layers. WARNING: this engine failed the parity limit on the 848x480
#                 OpenLORIS office1-1 frames (p99 |d log z| 0.0220, limit 0.02). The default engine passed (0.0096)
#   --fp32-norm   the default, accepted for old command lines
#   --workspace   builder workspace limit in MiB (default 2048). Lower it on an 8 GB Orin NX if the build fails
#   --python      python with the tensorrt package (default: python3)
#   --no-trtexec  use the Python builder even when trtexec exists
# Environment: TRTEXEC overrides the trtexec path.
#
# Any engine must pass scripts/jetson/depth_parity.py on frames of the flight camera before use, and the depth
# constants must be fitted with its backend. Give each build its own file name.
#
# Example on the Jetson (MAXN power mode, SLAM stopped):
#   scripts/jetson/build_trt_engine.sh da2b_518x910.onnx da2b_518x910_fp16n.engine
set -euo pipefail

usage() { sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'; }

PRECISION=fp16
FP32_NORM=1
PLAIN=0
WORKSPACE=2048
PY=python3
USE_TRTEXEC=1
POS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --fp32) PRECISION=fp32 ;;
    --plain-fp16) PLAIN=1 ;;
    --fp32-norm) ;;   # the default
    --workspace) WORKSPACE="$2"; shift ;;
    --python) PY="$2"; shift ;;
    --no-trtexec) USE_TRTEXEC=0 ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "unknown option $1" >&2; usage >&2; exit 2 ;;
    *) POS+=("$1") ;;
  esac
  shift
done
if [ ${#POS[@]} -ne 2 ]; then usage >&2; exit 2; fi
ONNX="${POS[0]}"
ENGINE="${POS[1]}"
if [ ! -f "$ONNX" ]; then echo "ONNX file not found: $ONNX" >&2; exit 2; fi
if [ "$PRECISION" = fp32 ] && [ "$PLAIN" = 1 ]; then echo "--plain-fp16 and --fp32 exclude each other" >&2; exit 2; fi
if [ "$PRECISION" = fp32 ] || [ "$PLAIN" = 1 ]; then FP32_NORM=0; fi
PLAIN_WARNING="WARNING: plain FP16 engine (--plain-fp16). It failed the parity limit on the 848x480 OpenLORIS frames
         (p99 |d log z| 0.0220 > 0.02). Do not fly it unless depth_parity.py passes on D455 frames and the depth
         constants were fitted with it. The default build (FP32 LayerNormalization and Softmax) passed."

TRTEXEC_BIN="${TRTEXEC:-}"
if [ -z "$TRTEXEC_BIN" ]; then
  if [ -x /usr/src/tensorrt/bin/trtexec ]; then TRTEXEC_BIN=/usr/src/tensorrt/bin/trtexec
  elif command -v trtexec >/dev/null 2>&1; then TRTEXEC_BIN="$(command -v trtexec)"; fi
fi
[ "$USE_TRTEXEC" = 1 ] || TRTEXEC_BIN=""

PY_TRT_VERSION="$("$PY" -c 'import tensorrt; print(tensorrt.__version__)' 2>/dev/null || true)"
echo "onnx:      $ONNX"
echo "engine:    $ENGINE"
echo "precision: $PRECISION$([ "$FP32_NORM" = 1 ] && echo ' (LayerNormalization and Softmax in FP32)')$([ "$PLAIN" = 1 ] && echo ' (plain)')"
echo "tensorrt (python $PY): ${PY_TRT_VERSION:-not found}"
if [ "$PLAIN" = 1 ]; then echo "$PLAIN_WARNING" >&2; fi

# Layer names of the LayerNormalization and Softmax layers, as TensorRT names them after parsing the ONNX file.
norm_layers() {
  "$PY" - "$ONNX" <<'EOF'
import sys
import tensorrt as trt
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
flags = 0
if int(trt.__version__.split(".")[0]) < 10:
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(flags)
parser = trt.OnnxParser(network, logger)
if not parser.parse_from_file(sys.argv[1]):
    sys.exit("could not parse " + sys.argv[1])
names = [network.get_layer(i).name for i in range(network.num_layers)
         if network.get_layer(i).type in (trt.LayerType.NORMALIZATION, trt.LayerType.SOFTMAX)]
bad = [n for n in names if "," in n or ":" in n]
if bad:   # the trtexec layer spec splits at "," and ":"
    sys.exit("layer names with ',' or ':' cannot be given to trtexec: %s. Build with --no-trtexec" % bad[:3])
print(",".join(n + ":fp32" for n in names))
EOF
}

START=$(date +%s)
if [ -n "$TRTEXEC_BIN" ]; then
  BUILDER="trtexec"
  ARGS=(--onnx="$ONNX" --saveEngine="$ENGINE" --memPoolSize=workspace:"$WORKSPACE")
  if [ "$PRECISION" = fp16 ]; then ARGS+=(--fp16); else ARGS+=(--noTF32); fi
  if [ "$FP32_NORM" = 1 ]; then
    if [ -z "$PY_TRT_VERSION" ]; then
      echo "the default build (FP32 LayerNormalization and Softmax) needs the tensorrt Python package for $PY to" \
           "list the layer names. On a Jetson: sudo apt install python3-libnvinfer, or pass --python /usr/bin/python3" >&2
      exit 2
    fi
    SPEC="$(norm_layers)"
    if [ -z "$SPEC" ]; then echo "no LayerNormalization or Softmax layer found in $ONNX" >&2; exit 2; fi
    echo "fp32 layers: $(echo "$SPEC" | tr ',' '\n' | grep -c . || true)"
    # The same constraints as the Python builder: layer precision and output type FP32 (each layer has one output).
    ARGS+=(--precisionConstraints=obey --layerPrecisions="$SPEC" --layerOutputTypes="$SPEC")
  fi
  echo "builder:   $TRTEXEC_BIN"
  LOG="$ENGINE.build.log"
  "$TRTEXEC_BIN" "${ARGS[@]}" 2>&1 | tee "$LOG"
  TRT_VERSION="$(grep -m1 -oE 'TensorRT version: [0-9.]+' "$LOG" | awk '{print $3}' || true)"
  [ -n "$TRT_VERSION" ] || TRT_VERSION="${PY_TRT_VERSION:-unknown}"
  if [ "$FP32_NORM" = 1 ] && [ "${TRT_VERSION%.*}" != "${PY_TRT_VERSION%.*}" ]; then
    echo "WARNING: the FP32 layer names came from TensorRT $PY_TRT_VERSION (python), but trtexec is TensorRT" \
         "$TRT_VERSION. The names may not match. Rebuild with --no-trtexec" >&2
  fi
else
  BUILDER="python"
  if [ -z "$PY_TRT_VERSION" ]; then
    echo "no trtexec and no tensorrt Python package for $PY. On a Jetson install nvidia-tensorrt (JetPack)," \
         "on a desktop pip install tensorrt-cu12" >&2
    exit 2
  fi
  echo "builder:   TensorRT Python API ($PY)"
  "$PY" - "$ONNX" "$ENGINE" "$PRECISION" "$FP32_NORM" "$WORKSPACE" <<'EOF'
import sys, time
import tensorrt as trt
onnx_path, engine_path, precision, fp32_norm, workspace = sys.argv[1:6]
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
flags = 0
if int(trt.__version__.split(".")[0]) < 10:
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(flags)
parser = trt.OnnxParser(network, logger)
if not parser.parse_from_file(onnx_path):
    for i in range(parser.num_errors):
        print(parser.get_error(i))
    sys.exit("could not parse " + onnx_path)
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace) << 20)
if precision == "fp16":
    config.set_flag(trt.BuilderFlag.FP16)
else:   # strict FP32: TensorRT turns TF32 on by default on Ampere GPUs (Orin too)
    config.clear_flag(trt.BuilderFlag.TF32)
if fp32_norm == "1":
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    n = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.type in (trt.LayerType.NORMALIZATION, trt.LayerType.SOFTMAX):
            layer.precision = trt.float32
            for k in range(layer.num_outputs):
                layer.set_output_type(k, trt.float32)
            n += 1
    print("fp32 layers:", n)
for i in range(network.num_inputs):
    t = network.get_input(i)
    print("input ", t.name, tuple(t.shape), t.dtype)
for i in range(network.num_outputs):
    t = network.get_output(i)
    print("output", t.name, tuple(t.shape), t.dtype)
t0 = time.time()
plan = builder.build_serialized_network(network, config)
if plan is None:
    sys.exit("engine build failed")
with open(engine_path, "wb") as f:
    f.write(plan)
print("built in %.0f s" % (time.time() - t0))
EOF
  TRT_VERSION="$PY_TRT_VERSION"
fi
if [ ! -s "$ENGINE" ]; then echo "no engine written to $ENGINE" >&2; exit 1; fi

# The build record next to the engine: the ONNX record plus precision, TensorRT version and builder.
"$PY" - "$ONNX" "$ENGINE" "$PRECISION" "$FP32_NORM" "$TRT_VERSION" "$BUILDER" <<'EOF' || echo "warning: no build record written" >&2
import json, os, sys
onnx_path, engine_path, precision, fp32_norm, version, builder = sys.argv[1:7]
rec = {}
if os.path.isfile(onnx_path + ".json"):
    with open(onnx_path + ".json") as f:
        rec = json.load(f)
# TF32: off in a --fp32 build. An FP16 build keeps the TensorRT default (on), which FP32 layers may use.
rec.update({"onnx": os.path.basename(onnx_path), "engine": os.path.basename(engine_path), "precision": precision,
            "fp32_norm": fp32_norm == "1", "tf32": precision != "fp32", "tensorrt": version, "builder": builder})
with open(engine_path + ".json", "w") as f:
    json.dump(rec, f, indent=1)
EOF

echo "TensorRT version: $TRT_VERSION"
echo "engine: $(readlink -f "$ENGINE") ($(du -h "$ENGINE" | cut -f1)), built in $(( $(date +%s) - START )) s"
if [ "$PLAIN" = 1 ]; then echo "$PLAIN_WARNING" >&2; fi
echo "check it first: scripts/jetson/depth_parity.py --backend trt-$PRECISION --engine $(readlink -f "$ENGINE")" \
     "--images <folder of flight camera frames> --n 30 (exit 0 pass, 1 parity fail, 2 error)"
echo "fit the depth constants with backend trt-$PRECISION and this engine (UCSLAM.DepthBackend of the settings)"
echo "then use it with: backend:=trt-$PRECISION engine:=$(readlink -f "$ENGINE") warmup_width:=<W> warmup_height:=<H>"
