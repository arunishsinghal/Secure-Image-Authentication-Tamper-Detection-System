"""
utils.py
--------
Image I/O, imperceptibility metrics, and small helpers.

The metrics here are what turn this from "a script that works" into a project
with defensible results. You need numbers in the report, not screenshots.
"""

from __future__ import annotations

import hashlib
import io
import os
from datetime import datetime

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load_image(path: str) -> np.ndarray:
    """Read an image as uint8 BGR. Alpha is dropped -- it is not authenticated."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def decode_upload(file_bytes: bytes) -> np.ndarray:
    """Decode raw bytes (e.g. a Streamlit upload) into a uint8 BGR array."""
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unsupported or corrupt image file.")
    return img


def save_png(image: np.ndarray, path: str) -> str:
    """Write losslessly as PNG.

    PNG is mandatory, not a preference. Saving a fragile-watermarked image as
    JPEG destroys the LSB plane and every block will fail verification.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not path.lower().endswith(".png"):
        path = os.path.splitext(path)[0] + ".png"
    if not cv2.imwrite(path, image, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise IOError(f"Failed to write {path}")
    return path


def encode_png_bytes(image: np.ndarray) -> bytes:
    """Encode to in-memory PNG bytes, for Streamlit download buttons."""
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise IOError("PNG encoding failed.")
    return buf.tobytes()


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """OpenCV works in BGR; Streamlit and PIL expect RGB."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def file_sha256(image: np.ndarray) -> str:
    """SHA-256 over the encoded PNG bytes -- an integrity fingerprint for the report."""
    return hashlib.sha256(encode_png_bytes(image)).hexdigest()


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# --------------------------------------------------------------------------- #
# Imperceptibility metrics
# --------------------------------------------------------------------------- #
def psnr(original: np.ndarray, modified: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio in dB.

    Interpretation for your report: above ~40 dB the difference is invisible to
    the human eye. A 1-bit LSB watermark should land near 51 dB, because the
    mean squared error is bounded by the fact that each pixel changes by at
    most 1 intensity level.
    """
    a, b = _match_shapes(original, modified)
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def ssim(original: np.ndarray, modified: np.ndarray) -> float:
    """Structural Similarity Index, range [-1, 1]. 1.0 means identical.

    SSIM is reported alongside PSNR because PSNR only measures raw error energy
    and is a poor proxy for perceived quality. SSIM models luminance, contrast,
    and structure, so it is the stronger claim to make about imperceptibility.
    """
    a, b = _match_shapes(original, modified)
    if a.ndim == 3:
        return float(ssim_fn(a, b, channel_axis=2, data_range=255))
    return float(ssim_fn(a, b, data_range=255))


def mae(original: np.ndarray, modified: np.ndarray) -> float:
    """Mean absolute per-pixel change. For 1-bit LSB embedding this sits near 0.5."""
    a, b = _match_shapes(original, modified)
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def max_pixel_change(original: np.ndarray, modified: np.ndarray) -> int:
    """Largest single-pixel intensity change. Must be <= 1 for 1-bit LSB embedding."""
    a, b = _match_shapes(original, modified)
    return int(np.max(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def quality_metrics(original: np.ndarray, watermarked: np.ndarray) -> dict:
    """All imperceptibility metrics in one call. Drop this table into your report."""
    return {
        "PSNR (dB)": round(psnr(original, watermarked), 2),
        "SSIM": round(ssim(original, watermarked), 6),
        "MAE": round(mae(original, watermarked), 4),
        "Max pixel change": max_pixel_change(original, watermarked),
    }


def _match_shapes(a: np.ndarray, b: np.ndarray):
    """Crop both arrays to their common region.

    Needed because embedding crops the image down to whole blocks, so the
    protected image can be a few pixels smaller than the original.
    """
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    return a[:h, :w], b[:h, :w]


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #
def localisation_scores(truth_mask: np.ndarray, detected_grid: np.ndarray,
                        block: int) -> dict:
    """Compare the detector against a known ground-truth tamper mask.

    This is the single most valuable thing you can put in the report. Anyone can
    show a red box on a screenshot; precision/recall proves the detector is
    correct. `truth_mask` is a pixel-level boolean array of what was actually
    edited (attacks.py returns one for every attack).
    """
    gh, gw = detected_grid.shape
    truth = truth_mask[: gh * block, : gw * block]
    # A block counts as truly tampered if any pixel inside it was modified.
    truth_blocks = truth.reshape(gh, block, gw, block).any(axis=(1, 3))

    tp = int(np.logical_and(truth_blocks, detected_grid).sum())
    fp = int(np.logical_and(~truth_blocks, detected_grid).sum())
    fn = int(np.logical_and(truth_blocks, ~detected_grid).sum())
    tn = int(np.logical_and(~truth_blocks, ~detected_grid).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "Accuracy": round((tp + tn) / max(tp + tn + fp + fn, 1), 4),
    }
