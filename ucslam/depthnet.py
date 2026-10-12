"""Metric monocular depth networks behind one plug, for the depth prior.

The PriorNet thread of the paper (Sec. VI-C) runs a metric monocular depth
network on keyframes and reads the predicted depth at every feature. This
module wraps the networks the characterization of Table tab:depthprior uses
behind one class, ``MetricDepth``, so that the benchmark and any other caller
see the same interface whichever network is loaded.

    net = MetricDepth("da2-metric-indoor-small", device="cuda")
    z = net.predict(img)              # float32 (H, W) metric depth in metres

Supported models, see ``MetricDepth.SUPPORTED``:

    da2-metric-indoor-small   Depth Anything V2, metric, indoor (Hypersim)
                              fine-tune, ViT-S. Hugging Face id
                              depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf,
                              loaded through ``transformers``.
    da2-metric-indoor-base    the ViT-B variant of the same,
                              depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf.
    metric3d-vit-small        Metric3D v2 ViT-S through ``torch.hub``
                              (yvanyin/metric3d, entry ``metric3d_vit_small``).
                              It needs the pinhole intrinsics ``K`` of the
                              input, because the network is trained at a
                              canonical focal length and rescales its output
                              by the ratio of the true focal length to it.

Input handling, which the last column of Table tab:depthprior reports.
Every network here was trained on three-channel perspective images. A
grayscale input (EuRoC, and the rectified TUM-VI frames) is replicated to
three identical channels before it enters the network. No colour is
invented. The network then sees a desaturated image, which is inside the
augmentation range these models train with. Fisheye frames are NOT handled
here. The caller rectifies a fisheye frame to a pinhole view first, with the
field of view and focal length it reports, and passes the rectified frame.

Resolution. Depth Anything V2 resizes the input so that the shorter side is
518 px and both sides are multiples of 14, runs the network, and this class
resamples the prediction bilinearly back to the input size. Metric3D v2
resizes to its canonical 616 x 1064 input with the intrinsics scaled by the
same factor and resamples back the same way. The returned map therefore has
the shape of the input, so a keypoint (u, v) in input pixels reads its depth
at ``z[round(v), round(u)]``, or by bilinear sampling with
``sample_depth``.

Latency. ``predict`` records the wall time of the last call, including the
host to device copy and the resample, in ``last_latency_s``. On CUDA the
timer brackets a ``torch.cuda.synchronize``.

Provenance. This is a Python wrapper around released third-party weights. It
is not the C++ PriorNet thread, which does not exist yet.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["MetricDepth", "sample_depth", "to_rgb"]

_DA2_IDS = {
    "da2-metric-indoor-small": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "da2-metric-indoor-base": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
}
# Hugging Face snapshots the paper's depth caches were made with. Pinned so a new upload
# of the weights cannot change the depth prior silently.
_DA2_REVISIONS = {
    "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf": "8078d68a9c75a972131914f6afd0c1723be0da7f",
    "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf": "c6d9784685727bfc6d0a7b5452ce94afaee1e7f5",
}
_METRIC3D_HUB = ("yvanyin/metric3d", "metric3d_vit_small")
_METRIC3D_CANON_F = 1000.0            # canonical focal length of Metric3D v2 [px]
_METRIC3D_INPUT_HW = (616, 1064)      # ViT input size the released hubconf uses
_METRIC3D_MEAN = (123.675, 116.28, 103.53)
_METRIC3D_STD = (58.395, 57.12, 57.375)


def to_rgb(img: np.ndarray) -> np.ndarray:
    """Return ``img`` as uint8 (H, W, 3).

    A (H, W) grayscale image is replicated to three identical channels. A
    (H, W, 3) image is returned unchanged. A (H, W, 1) image is squeezed and
    replicated. Anything else raises ValueError. The dtype is converted to
    uint8 with clipping, and a float image in [0, 1] is scaled by 255 first.
    """
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    if a.dtype != np.uint8:
        a = a.astype(np.float64)
        if a.size and np.nanmax(a) <= 1.0:
            a = a * 255.0
        a = np.clip(np.round(a), 0, 255).astype(np.uint8)
    if a.ndim == 2:
        return np.repeat(a[:, :, None], 3, axis=2)
    if a.ndim == 3 and a.shape[2] == 3:
        return a
    raise ValueError(f"expected (H,W) or (H,W,3) image, got shape {a.shape}")


def sample_depth(depth: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinearly sample a (H, W) depth map at sub-pixel (u, v) positions.

    Pixel centres sit at integer coordinates, the same convention the
    detector uses for its keypoints. Positions outside the map return NaN.
    A sample whose four neighbours are not all finite and positive returns
    NaN too, so a keypoint on a depth discontinuity next to an invalid pixel
    is not silently averaged.

    Args:
        depth: (H, W) float array.
        uv: (K, 2) array of (u, v) in pixels of ``depth``.

    Returns:
        (K,) float64 depths, NaN where undefined.
    """
    d = np.asarray(depth, np.float64)
    uv = np.atleast_2d(np.asarray(uv, np.float64))
    h, w = d.shape
    out = np.full(uv.shape[0], np.nan)
    if uv.shape[0] == 0:
        return out
    u, v = uv[:, 0], uv[:, 1]
    ok = np.isfinite(u) & np.isfinite(v) & (u >= 0) & (v >= 0) & (u <= w - 1) & (v <= h - 1)
    if not np.any(ok):
        return out
    u, v = u[ok], v[ok]
    x0 = np.clip(np.floor(u).astype(int), 0, w - 1)
    y0 = np.clip(np.floor(v).astype(int), 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    fx, fy = u - x0, v - y0
    q = np.stack([d[y0, x0], d[y0, x1], d[y1, x0], d[y1, x1]], axis=1)
    val = ((1 - fx) * (1 - fy) * q[:, 0] + fx * (1 - fy) * q[:, 1]
           + (1 - fx) * fy * q[:, 2] + fx * fy * q[:, 3])
    good = np.all(np.isfinite(q) & (q > 0), axis=1)
    val[~good] = np.nan
    out[ok] = val
    return out


class MetricDepth:
    """One metric monocular depth network behind a fixed interface.

    Attributes:
        name: the model key that was loaded, one of ``SUPPORTED``.
        device: the torch device the network runs on.
        needs_intrinsics: True when ``predict`` requires ``K``.
        last_latency_s: wall time of the last ``predict`` call in seconds,
            NaN before the first call.
    """

    SUPPORTED = ("da2-metric-indoor-small", "da2-metric-indoor-base",
                 "metric3d-vit-small")

    def __init__(self, model: str = "da2-metric-indoor-small", device: str = "cuda"):
        if model not in self.SUPPORTED:
            raise ValueError(f"unknown model {model!r}, supported: {self.SUPPORTED}")
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.name = model
        self.device = torch.device(device)
        self.last_latency_s = float("nan")
        self.needs_intrinsics = model.startswith("metric3d")
        self._kind = "metric3d" if self.needs_intrinsics else "da2"
        if self._kind == "da2":
            self._load_da2(_DA2_IDS[model])
        else:
            self._load_metric3d()

    # ----------------------------------------------------------- loading --

    def _load_da2(self, hf_id: str) -> None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        rev = _DA2_REVISIONS.get(hf_id)
        self.processor = AutoImageProcessor.from_pretrained(hf_id, revision=rev)
        self.model = AutoModelForDepthEstimation.from_pretrained(hf_id, revision=rev)
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.source = f"huggingface:{hf_id}@{rev}" if rev else f"huggingface:{hf_id}"

    def _load_metric3d(self) -> None:
        repo, entry = _METRIC3D_HUB
        self.model = torch.hub.load(repo, entry, pretrain=True, trust_repo=True)
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.source = f"torch.hub:{repo}/{entry}"

    # --------------------------------------------------------- inference --

    @torch.no_grad()
    def predict(self, img: np.ndarray, K: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict metric depth for one image.

        Args:
            img: uint8 (H, W) grayscale or (H, W, 3) RGB. Grayscale is
                replicated to three channels, see the module docstring.
            K: (3, 3) pinhole intrinsics of ``img``. Required by Metric3D,
                ignored by Depth Anything V2, which predicts metric depth
                without intrinsics.

        Returns:
            float32 (H, W) depth in metres at the input resolution.
        """
        rgb = to_rgb(img)
        h, w = rgb.shape[:2]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        if self._kind == "da2":
            out = self._predict_da2(rgb, h, w)
        else:
            if K is None:
                raise ValueError(f"{self.name} needs the pinhole intrinsics K")
            out = self._predict_metric3d(rgb, np.asarray(K, np.float64), h, w)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.last_latency_s = time.perf_counter() - t0
        return out

    def _predict_da2(self, rgb: np.ndarray, h: int, w: int) -> np.ndarray:
        inputs = self.processor(images=rgb, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        pred = self.model(pixel_values=pixel_values).predicted_depth   # (1, h', w')
        if pred.ndim == 3:
            pred = pred[:, None]
        pred = F.interpolate(pred.float(), size=(h, w), mode="bilinear", align_corners=False)
        return pred[0, 0].cpu().numpy().astype(np.float32)

    def _predict_metric3d(self, rgb: np.ndarray, K: np.ndarray, h: int, w: int) -> np.ndarray:
        # The released hubconf example resizes the image so that it fits the
        # 616 x 1064 canvas, pads the remainder, scales the intrinsics by the
        # same factor and divides the predicted depth by (f_scaled / 1000).
        ih, iw = _METRIC3D_INPUT_HW
        scale = min(ih / h, iw / w)
        rh, rw = int(round(h * scale)), int(round(w * scale))
        x = torch.from_numpy(rgb).permute(2, 0, 1).float()[None]          # (1,3,H,W)
        x = F.interpolate(x, size=(rh, rw), mode="bilinear", align_corners=False)
        pad_h, pad_w = ih - rh, iw - rw
        pad = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        mean = torch.tensor(_METRIC3D_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(_METRIC3D_STD).view(1, 3, 1, 1)
        x = F.pad(x, pad, value=0.0)
        x = ((x - mean) / std).to(self.device)
        pred, _, _ = self.model.inference({"input": x})
        pred = pred[:, :, pad[2]:ih - pad[3], pad[0]:iw - pad[1]]
        f_scaled = 0.5 * (K[0, 0] + K[1, 1]) * scale
        pred = pred * (f_scaled / _METRIC3D_CANON_F)
        pred = F.interpolate(pred.float(), size=(h, w), mode="bilinear", align_corners=False)
        return pred[0, 0].cpu().numpy().astype(np.float32)

    def __repr__(self) -> str:
        return f"MetricDepth(name={self.name!r}, device={str(self.device)!r})"
