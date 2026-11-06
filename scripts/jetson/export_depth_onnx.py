#!/usr/bin/env python3
"""Export the depth network of UC-SLAM to ONNX at the fixed input size of one camera image size.

The model is ucslam.depth_model of configs/default.yaml (da2-metric-indoor-base), loaded by
ucslam.depthnet.MetricDepth from the pinned Hugging Face revision. The input size is the size the HF processor
makes for a width x height image (848x480 gives 518x910, 640x480 gives 518x686). The graph holds the network
only: the processor (resize, normalisation) and the resample back to the image size stay in Python
(ucslam/trt_depth.py), so the TensorRT backend sees the same input as torch-fp32.

The export runs on the CPU in FP32: torch.onnx.export with dynamo=False, opset 17 (one LayerNormalization op per
LayerNorm), constant folding. It writes <out>.json next to the ONNX file (model, revision, input size), which
build_trt_engine.sh carries into the engine record.

Needs the onnx package (pip install onnx). Runs on the desktop or on the Jetson (Python 3.10 or newer).

    python scripts/jetson/export_depth_onnx.py --width 848 --height 480 --out da2b_518x910.onnx
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="Needs the onnx package (pip install onnx) besides torch and transformers. Set HF_HUB_OFFLINE=1 "
               "when the Hugging Face cache already holds the weights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--width", type=int, default=848, help="camera image width in pixels")
    ap.add_argument("--height", type=int, default=480, help="camera image height in pixels")
    ap.add_argument("--model", default="", help="model key, empty: ucslam.depth_model of --config")
    ap.add_argument("--config", default=os.path.join(REPO, "configs", "default.yaml"))
    ap.add_argument("--out", default="", help="ONNX file, empty: <model>_<H>x<W>.onnx in the current folder")
    ap.add_argument("--opset", type=int, default=17)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import onnx
    except ImportError:
        print("this script needs the onnx package: pip install onnx", file=sys.stderr)
        return 2
    import numpy as np
    import torch

    from experiments.precompute_depth import load_config, precompute_params
    from ucslam.depthnet import _DA2_IDS, _DA2_REVISIONS, MetricDepth

    model_key = args.model or precompute_params(load_config(args.config))["model"]
    if model_key not in _DA2_IDS:
        print(f"only Depth Anything V2 models export, got {model_key!r}", file=sys.stderr)
        return 2
    net = MetricDepth(model_key, "cpu")
    rgb = np.zeros((args.height, args.width, 3), np.uint8)
    x = net.processor(images=rgb, return_tensors="pt")["pixel_values"]
    in_h, in_w = int(x.shape[-2]), int(x.shape[-1])
    out = args.out or f"{model_key}_{in_h}x{in_w}.onnx"
    print(f"model {model_key} ({net.source})")
    print(f"image {args.width}x{args.height} -> network input (1, 3, {in_h}, {in_w})")

    class Net(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values):
            return self.m(pixel_values=pixel_values).predicted_depth   # (1, H, W) metres

    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(Net(net.model.eval()), (x,), out, input_names=["pixel_values"],
                          output_names=["predicted_depth"], opset_version=args.opset, dynamo=False,
                          do_constant_folding=True)
    print(f"exported in {time.time() - t0:.1f} s: {out} ({os.path.getsize(out) / 1e6:.1f} MB)")

    m = onnx.load(out, load_external_data=False)
    try:
        m = onnx.shape_inference.infer_shapes(m)
    except Exception as e:  # shape inference is a convenience only
        print(f"shape inference skipped: {e!r}")

    def dims(v):
        return [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?") for d in v.type.tensor_type.shape.dim]

    for v in m.graph.input:
        print(f"input  {v.name} {dims(v)}")
    for v in m.graph.output:
        print(f"output {v.name} {dims(v)}")
    ops = sorted({n.op_type for n in m.graph.node})
    print(f"opset {args.opset}, {len(m.graph.node)} nodes, ops: {' '.join(ops)}")
    onnx.checker.check_model(out)
    hf_id = _DA2_IDS[model_key]
    record = {"model": model_key, "hf_id": hf_id, "revision": _DA2_REVISIONS.get(hf_id), "source": net.source,
              "image_wh": [args.width, args.height], "input_hw": [in_h, in_w], "opset": args.opset,
              "input": "pixel_values", "output": "predicted_depth", "torch": torch.__version__,
              "onnx_version": onnx.__version__}
    with open(out + ".json", "w") as f:
        json.dump(record, f, indent=1)
    print(f"record: {out}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
