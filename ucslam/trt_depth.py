"""Fast depth backends for the live depth node (Jetson): torch FP16 and TensorRT.

The live node must give the map the per-domain constants were fitted on. So both backends keep the cache
pipeline of ucslam/depthnet.py around the network and change only the network itself:

    1. the same Hugging Face processor (pinned revision) makes ``pixel_values`` on the CPU,
    2. the network runs in FP16 weights (TorchFp16MetricDepth) or as a TensorRT engine (TrtMetricDepth),
    3. the same bilinear resample brings the prediction back to the image size, as float32 metres.

Both classes have the interface of ``depthnet.MetricDepth`` that the callers use: ``predict(img)``, ``name`` (the
model key), ``source``, ``device`` and ``last_latency_s`` (wall time of the last call, processor and resample
included, bracketed by CUDA synchronisation).

A TensorRT engine has a fixed input size. ``net_input_hw`` gives the size the processor makes for an image
(848x480 gives 518x910). An image that gives another size is refused, never resized, because a resized input
changes the depth map. Build one engine per stream size with scripts/jetson/export_depth_onnx.py and
scripts/jetson/build_trt_engine.sh.

The engine code uses the named-tensor Python API (``num_io_tensors``, ``set_tensor_address`` and
``execute_async_v3``). TensorRT 8.5 and newer have it, so it serves TensorRT 8.6 (JetPack 6.0) and TensorRT 10
(JetPack 6.1 and 6.2). The bindings API (``execute_async_v2``) is kept only as a fallback for TensorRT older than
8.5, which JetPack 6 does not ship. Torch CUDA tensors are the engine buffers, so neither pycuda nor cuda-python
is needed.

The depth cache was made with the torchvision DPT processor. Without torchvision, transformers silently picks a PIL
processor that makes a slightly different network input. ``check_processor`` warns once per process when that
happens (live_depth.LiveDepth calls it for every backend).

The constants of a depth domain are valid for one backend only. Check a new backend with
scripts/jetson/depth_parity.py against torch-fp32 before use.
"""
from __future__ import annotations

import json
import math
import os
import time
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ucslam.depthnet import _DA2_IDS, _DA2_REVISIONS, MetricDepth, to_rgb

__all__ = ["BACKENDS", "EngineSizeError", "TorchFp16MetricDepth", "TrtEngine", "TrtMetricDepth", "check_processor",
           "make_depth_net", "net_input_hw", "read_sidecar"]

# torch-fp32 is the reference: depthnet.MetricDepth unchanged, the backend of the depth cache.
BACKENDS = ("torch-fp32", "torch-fp16", "trt-fp16", "trt-fp32")

# The DPT processor of Depth Anything V2 (preprocessor_config.json of the pinned revision).
DA2_SIZE = 518
DA2_MULTIPLE = 14


def net_input_hw(width: int, height: int, size: int = DA2_SIZE, multiple: int = DA2_MULTIPLE) -> Tuple[int, int]:
    """(H, W) of ``pixel_values`` for a width x height image: the keep-aspect-ratio rule of the DPT processor.

    Scale as little as possible (fit the side whose scale is closer to 1), then round each side to a multiple of
    14. This is a copy of transformers' rule for planning engines. The processor itself stays the authority:
    TrtMetricDepth checks the shape it really makes.
    """
    def to_multiple(v: float) -> int:
        x = round(v / multiple) * multiple
        if x < 0:
            x = math.ceil(v / multiple) * multiple
        return int(x)

    sh, sw = size / float(height), size / float(width)
    if abs(1 - sw) < abs(1 - sh):
        sh = sw
    else:
        sw = sh
    return to_multiple(sh * height), to_multiple(sw * width)


def read_sidecar(path: str) -> Dict:
    """The JSON written next to an ONNX file or engine by the export and build scripts, {} when missing."""
    side = path + ".json"
    if not os.path.isfile(side):
        return {}
    with open(side) as f:
        return json.load(f)


def _cuda_device(device: str, what: str) -> torch.device:
    dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError(f"{what} runs on CUDA only, got device {device!r}")
    if not torch.cuda.is_available():
        raise RuntimeError(f"{what} needs CUDA, but torch.cuda.is_available() is False")
    return dev


def _da2_id(model: str) -> str:
    if model not in _DA2_IDS:
        raise ValueError(f"backend needs a Depth Anything V2 model, one of {tuple(_DA2_IDS)}, got {model!r}")
    return _DA2_IDS[model]


_PROCESSOR_WARNED = False


def check_processor(processor, backend: str, log: Optional[Callable[[str], None]] = None) -> Optional[str]:
    """Warn once per process when the Hugging Face processor is not the torchvision one of the depth cache.

    transformers 5 falls back to a PIL processor when torchvision is missing. Its input differs a little (p99
    d log z about 1.6e-3 on EuRoC), so the depth maps drift from the fitted constants with no error at all.
    Returns the warning text, or None when the processor is the torchvision one or there is no processor (not a
    Depth Anything V2 model). ``log`` gets the text the first time, default ``warnings.warn``.
    """
    global _PROCESSOR_WARNED
    if processor is None:
        return None
    names = [c.__name__ for c in type(processor).__mro__]
    if "TorchvisionBackend" in names:
        return None
    msg = (f"depth backend {backend}: the Hugging Face processor is {type(processor).__name__}, not the "
           f"torchvision processor (TorchvisionBackend) that made the depth cache. The depth maps differ a little "
           f"from the ones the constants were fitted on. Install the torchvision build of this torch")
    if not _PROCESSOR_WARNED:
        _PROCESSOR_WARNED = True
        if log is None:
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
        else:
            log(msg)
    return msg


class EngineSizeError(ValueError):
    """The image size does not give the fixed input size of a TensorRT engine. Every image of that size fails
    the same way until an engine for it is built."""


# --------------------------------------------------------------------------------------------- torch FP16 --

class TorchFp16MetricDepth(MetricDepth):
    """depthnet.MetricDepth with the weights in FP16 on CUDA. Same processor, output cast to float32 before the
    same resample. The FP32 model is loaded on the CPU first, so no FP32 copy is kept on the GPU."""

    def __init__(self, model: str = "da2-metric-indoor-base", device: str = "cuda"):
        _da2_id(model)
        dev = _cuda_device(device, "backend torch-fp16")
        super().__init__(model, "cpu")
        self.model.half().to(dev).eval()
        self.device = dev
        self.source = self.source + " (fp16 weights)"

    def _predict_da2(self, rgb: np.ndarray, h: int, w: int) -> np.ndarray:
        inputs = self.processor(images=rgb, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, torch.float16)
        pred = self.model(pixel_values=pixel_values).predicted_depth   # (1, h', w') float16
        if pred.ndim == 3:
            pred = pred[:, None]
        pred = F.interpolate(pred.float(), size=(h, w), mode="bilinear", align_corners=False)
        return pred[0, 0].cpu().numpy().astype(np.float32)

    def __repr__(self) -> str:
        return f"TorchFp16MetricDepth(name={self.name!r}, device={str(self.device)!r})"


# --------------------------------------------------------------------------------------------- TensorRT --

def _dtype_name(dt) -> str:
    """'FLOAT' or 'HALF' for a tensorrt.DataType (or any object with that name)."""
    return str(getattr(dt, "name", dt)).split(".")[-1].upper()


_TORCH_DTYPES = {"FLOAT": torch.float32, "HALF": torch.float16}


class TrtEngine:
    """One deserialised engine with one input and one output of fixed shape.

    ``api`` is "tensors" (named-tensor API of TensorRT 8.5 and newer, so TensorRT 8.6 and 10 of JetPack 6) or
    "bindings" (fallback for TensorRT older than 8.5). ``engine`` is a tensorrt.ICudaEngine. Any object with the
    same methods works, so the checks run in tests without TensorRT.
    """

    def __init__(self, engine, label: str = "engine"):
        self.engine = engine
        self.label = label
        self.api = "tensors" if hasattr(engine, "num_io_tensors") else "bindings"
        ios = self._io_tensors() if self.api == "tensors" else self._io_bindings()
        inputs = [t for t in ios if t["input"]]
        outputs = [t for t in ios if not t["input"]]
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(f"{label}: expected one input and one output tensor, got {len(inputs)} inputs "
                             f"{[t['name'] for t in inputs]} and {len(outputs)} outputs "
                             f"{[t['name'] for t in outputs]}")
        self.input, self.output = inputs[0], outputs[0]
        for t in (self.input, self.output):
            if any(int(d) < 0 for d in t["shape"]):
                raise ValueError(f"{label}: tensor {t['name']} has a dynamic shape {t['shape']}. Build the engine "
                                 f"from an ONNX file with a fixed input size (scripts/jetson/export_depth_onnx.py)")
            if t["dtype"] not in _TORCH_DTYPES:
                raise ValueError(f"{label}: tensor {t['name']} has type {t['dtype']}, expected FLOAT or HALF")
        if len(self.input["shape"]) != 4 or self.input["shape"][0] != 1 or self.input["shape"][1] != 3:
            raise ValueError(f"{label}: input {self.input['name']} must be (1, 3, H, W), got {self.input['shape']}")
        oshape = self.output["shape"]
        if len(oshape) not in (3, 4) or oshape[-2:] != self.input["shape"][-2:] or any(d != 1 for d in oshape[:-2]):
            raise ValueError(f"{label}: output {self.output['name']} must be (1, H, W) or (1, 1, H, W) at the input "
                             f"size {self.input['shape'][-2:]}, got {oshape}")
        self.input_hw = (int(self.input["shape"][2]), int(self.input["shape"][3]))
        self.context = engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"{label}: could not create an execution context (out of GPU memory?)")

    def _io_tensors(self) -> List[Dict]:
        ios = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)   # tensorrt.TensorIOMode.INPUT or OUTPUT
            ios.append({"name": name, "input": _dtype_name(mode) == "INPUT",
                        "shape": tuple(int(d) for d in self.engine.get_tensor_shape(name)),
                        "dtype": _dtype_name(self.engine.get_tensor_dtype(name))})
        return ios

    def _io_bindings(self) -> List[Dict]:
        ios = []
        for i in range(self.engine.num_bindings):
            ios.append({"name": self.engine.get_binding_name(i), "input": bool(self.engine.binding_is_input(i)),
                        "shape": tuple(int(d) for d in self.engine.get_binding_shape(i)),
                        "dtype": _dtype_name(self.engine.get_binding_dtype(i)), "index": i})
        return ios

    @property
    def input_dtype(self) -> torch.dtype:
        return _TORCH_DTYPES[self.input["dtype"]]

    @property
    def output_dtype(self) -> torch.dtype:
        return _TORCH_DTYPES[self.output["dtype"]]

    def run(self, x: torch.Tensor, out: torch.Tensor, stream_handle: int) -> None:
        """Enqueue one inference: x and out are contiguous tensors of the engine shapes and types on its GPU."""
        if tuple(x.shape) != tuple(self.input["shape"]) or tuple(out.shape) != tuple(self.output["shape"]):
            raise ValueError(f"{self.label}: buffers {tuple(x.shape)} -> {tuple(out.shape)} do not match the engine "
                             f"{self.input['shape']} -> {self.output['shape']}")
        if self.api == "tensors":
            self.context.set_tensor_address(self.input["name"], int(x.data_ptr()))
            self.context.set_tensor_address(self.output["name"], int(out.data_ptr()))
            ok = self.context.execute_async_v3(stream_handle)
        else:
            bindings = [0] * (max(self.input["index"], self.output["index"]) + 1)
            bindings[self.input["index"]] = int(x.data_ptr())
            bindings[self.output["index"]] = int(out.data_ptr())
            ok = self.context.execute_async_v2(bindings=bindings, stream_handle=stream_handle)
        if ok is False:
            raise RuntimeError(f"{self.label}: TensorRT inference failed")


def load_engine(path: str):
    """Deserialise a TensorRT engine file. Needs the tensorrt Python package of the TensorRT that built it."""
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError("backend trt-* needs the tensorrt Python package (on a Jetson: apt python3-libnvinfer, "
                          "seen by a venv made with --system-site-packages)") from e
    if not os.path.isfile(path):
        raise FileNotFoundError(f"TensorRT engine not found: {path}")
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"could not load {path} with TensorRT {trt.__version__}. An engine runs only with the "
                           f"TensorRT version and GPU it was built on. Rebuild it on this device")
    return engine, runtime, trt.__version__


class TrtMetricDepth:
    """A Depth Anything V2 TensorRT engine behind the MetricDepth interface.

    The HF processor of the pinned revision makes ``pixel_values`` on the CPU exactly as MetricDepth does. The
    engine runs on a torch CUDA tensor and writes a torch CUDA tensor. The same ``F.interpolate`` brings the
    prediction to the image size.

    ``precision`` (fp16 or fp32) is the label of the backend. When the build script left a sidecar
    ``<engine>.json``, its model and precision must agree, else the engine is refused.
    """

    def __init__(self, engine_path: str, model: str = "da2-metric-indoor-base", device: str = "cuda",
                 precision: str = "fp16", _engine=None, _processor=None):
        if precision not in ("fp16", "fp32"):
            raise ValueError(f"precision must be fp16 or fp32, got {precision!r}")
        hf_id = _da2_id(model)
        rev = _DA2_REVISIONS.get(hf_id)
        self.name = model
        self.precision = precision
        self.engine_path = engine_path
        self.needs_intrinsics = False
        self.last_latency_s = float("nan")
        self.sidecar = {} if _engine is not None else read_sidecar(engine_path)
        check_sidecar(self.sidecar, model, precision, engine_path)
        if _engine is None:   # the real path. Tests pass a fake runner and processor instead.
            self.device = _cuda_device(device, f"backend trt-{precision}")
            with torch.cuda.device(self.device):   # TensorRT uses the current CUDA device
                torch.zeros(1, device=self.device)   # make the CUDA context of torch first
                engine, self._runtime, trt_version = load_engine(engine_path)
                self.runner = TrtEngine(engine, os.path.basename(engine_path))
            self._stream = torch.cuda.Stream(self.device)
        else:
            self.device = torch.device(device)
            self.runner, trt_version, self._stream = _engine, "test", None
        if _processor is None:
            from transformers import AutoImageProcessor
            _processor = AutoImageProcessor.from_pretrained(hf_id, revision=rev)
        self.processor = _processor
        self.input_hw = tuple(self.runner.input_hw)
        self.trt_version = trt_version
        hf = f"huggingface:{hf_id}@{rev}" if rev else f"huggingface:{hf_id}"
        self.source = f"{hf} as tensorrt-{trt_version} {precision} engine {os.path.basename(engine_path)}"

    def check_image_size(self, height: int, width: int) -> None:
        """Raise EngineSizeError when a width x height image does not give the engine input size (rule only)."""
        hw = net_input_hw(width, height)
        if hw != self.input_hw:
            raise EngineSizeError(self._size_message(width, height, hw))

    def _size_message(self, width: int, height: int, hw) -> str:
        return (f"a {width}x{height} image gives the network input {hw[1]}x{hw[0]} (W x H), but the engine "
                f"{os.path.basename(self.engine_path)} takes {self.input_hw[1]}x{self.input_hw[0]}. Build an engine "
                f"for this image size: scripts/jetson/export_depth_onnx.py --width {width} --height {height}, "
                f"then scripts/jetson/build_trt_engine.sh")

    @torch.no_grad()
    def predict(self, img: np.ndarray, K: Optional[np.ndarray] = None) -> np.ndarray:
        """float32 (H, W) depth in metres at the image size, as MetricDepth.predict. K is ignored (DA2)."""
        rgb = to_rgb(img)
        h, w = rgb.shape[:2]
        cuda = self.device.type == "cuda"
        if cuda:
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        pixel_values = self.processor(images=rgb, return_tensors="pt")["pixel_values"]
        hw = tuple(int(v) for v in pixel_values.shape[-2:])
        if hw != self.input_hw:
            raise EngineSizeError(self._size_message(w, h, hw))
        with (torch.cuda.stream(self._stream) if cuda else _null_ctx()):
            x = pixel_values.to(self.device, self.runner.input_dtype).contiguous()
            out = torch.empty(tuple(self.runner.output["shape"]), dtype=self.runner.output_dtype, device=self.device)
            handle = self._stream.cuda_stream if cuda else 0
            self.runner.run(x, out, handle)
            pred = out.reshape(1, 1, self.input_hw[0], self.input_hw[1])
            pred = F.interpolate(pred.float(), size=(h, w), mode="bilinear", align_corners=False)
            z = pred[0, 0].cpu().numpy().astype(np.float32)
        if cuda:
            torch.cuda.synchronize(self.device)
        self.last_latency_s = time.perf_counter() - t0
        return z

    def __repr__(self) -> str:
        return (f"TrtMetricDepth(name={self.name!r}, engine={os.path.basename(self.engine_path)!r}, "
                f"input={self.input_hw[1]}x{self.input_hw[0]}, precision={self.precision!r})")


class _null_ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def check_sidecar(side: Dict, model: str, precision: str, engine_path: str) -> None:
    """Refuse an engine whose build record names another model or precision. No record: no check."""
    if not side:
        return
    if side.get("model") and side["model"] != model:
        raise ValueError(f"engine {engine_path} was built from model {side['model']!r}, but the node serves "
                         f"{model!r}. DepthPrior.model would be wrong")
    if side.get("precision") and side["precision"] != precision:
        raise ValueError(f"engine {engine_path} was built in {side['precision']}, but the backend is "
                         f"trt-{precision}. Use backend trt-{side['precision']} or the matching engine")


def make_depth_net(model: str, device: str = "cuda", backend: str = "torch-fp32", engine: Optional[str] = None):
    """The network of a backend. torch-fp32 is depthnet.MetricDepth exactly as the depth cache uses it."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown depth backend {backend!r}, expected one of {BACKENDS}")
    if backend.startswith("trt-"):
        if not engine:
            raise ValueError(f"backend {backend} needs an engine file (parameter 'engine')")
        return TrtMetricDepth(engine, model, device, precision=backend[4:])
    if engine:
        raise ValueError(f"an engine file was given ({engine}), but backend {backend} does not use one. "
                         f"Set backend to trt-fp16 or trt-fp32, or leave engine empty")
    if backend == "torch-fp16":
        return TorchFp16MetricDepth(model, device)
    return MetricDepth(model, device)
