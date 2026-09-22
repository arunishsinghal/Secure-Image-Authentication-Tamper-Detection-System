"""
attacks.py
----------
Simulated tampering, with ground-truth masks.

This module is not in your original plan and it is the reason the project would
otherwise fail at demo time. Without it, the only way to show tamper detection
working is to open the protected PNG in an image editor, paint on it, save it
(hoping the editor does not re-encode or strip anything), and re-upload. That is
slow, unreproducible, and gives you no ground truth to score against.

Every function here returns (attacked_image, truth_mask), where truth_mask is a
pixel-level boolean array of exactly which pixels were modified. Feed that into
utils.localisation_scores() to get precision, recall, and F1 -- real numbers for
the report instead of a screenshot with a red box on it.
"""

from __future__ import annotations

import cv2
import numpy as np


def _empty_mask(image: np.ndarray) -> np.ndarray:
    return np.zeros(image.shape[:2], dtype=bool)


# --------------------------------------------------------------------------- #
# Content-changing attacks -- these SHOULD be detected and localised
# --------------------------------------------------------------------------- #
def copy_move(image: np.ndarray, src=(0.10, 0.10), dst=(0.55, 0.55),
              size=(0.20, 0.20)):
    """Copy-move forgery: duplicate a region of the image elsewhere.

    The classic forgery -- cloning out a person, duplicating a crowd. Hard for
    passive (watermark-free) detectors because the copied texture statistics are
    native to the image. Trivial for this scheme, because the pasted block does
    not carry a valid tag for its NEW position.
    """
    out = image.copy()
    mask = _empty_mask(image)
    h, w = image.shape[:2]

    bh, bw = int(h * size[0]), int(w * size[1])
    sy, sx = int(h * src[0]), int(w * src[1])
    dy, dx = int(h * dst[0]), int(w * dst[1])
    bh, bw = min(bh, h - sy, h - dy), min(bw, w - sx, w - dx)
    if bh <= 0 or bw <= 0:
        raise ValueError("Copy-move region falls outside the image.")

    out[dy:dy + bh, dx:dx + bw] = image[sy:sy + bh, sx:sx + bw]
    mask[dy:dy + bh, dx:dx + bw] = True
    return out, mask


def splice(image: np.ndarray, patch: np.ndarray, at=(0.30, 0.30),
           size=(0.25, 0.25)):
    """Splicing: paste content from a DIFFERENT image."""
    out = image.copy()
    mask = _empty_mask(image)
    h, w = image.shape[:2]

    ph, pw = int(h * size[0]), int(w * size[1])
    y, x = int(h * at[0]), int(w * at[1])
    ph, pw = min(ph, h - y), min(pw, w - x)
    if ph <= 0 or pw <= 0:
        raise ValueError("Splice region falls outside the image.")

    resized = cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_AREA)
    if resized.ndim == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    out[y:y + ph, x:x + pw] = resized
    mask[y:y + ph, x:x + pw] = True
    return out, mask


def add_text(image: np.ndarray, text: str = "FAKE", at=(0.45, 0.15),
             scale: float = 2.5):
    """Overlay text -- the "doctored document" scenario.

    Note the mask is derived from actual pixel difference rather than the text
    bounding box, so ground truth follows the glyph strokes exactly.
    """
    out = image.copy()
    h, w = image.shape[:2]
    org = (int(w * at[1]), int(h * at[0]))
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 255), max(int(scale * 2), 2), cv2.LINE_AA)
    mask = np.any(out != image, axis=2) if image.ndim == 3 else (out != image)
    return out, mask


def erase_region(image: np.ndarray, at=(0.35, 0.35), size=(0.20, 0.20),
                 method: str = "blur"):
    """Object removal by blurring or inpainting -- hiding evidence rather than adding it."""
    out = image.copy()
    mask = _empty_mask(image)
    h, w = image.shape[:2]

    rh, rw = int(h * size[0]), int(w * size[1])
    y, x = int(h * at[0]), int(w * at[1])
    rh, rw = min(rh, h - y), min(rw, w - x)

    region = out[y:y + rh, x:x + rw]
    if method == "blur":
        out[y:y + rh, x:x + rw] = cv2.GaussianBlur(region, (31, 31), 0)
    else:
        out[y:y + rh, x:x + rw] = int(region.mean())
    mask[y:y + rh, x:x + rw] = True
    return out, mask


def single_pixel_flip(image: np.ndarray, at=(0.5, 0.5)):
    """Change ONE pixel by ONE intensity level.

    The extreme sensitivity test. A robust watermark ignores this entirely.
    A correct fragile scheme flags exactly the one block containing that pixel
    and nothing else -- which is the whole point. Run this in your demo; it is
    the most convincing single result in the project.

    Caveat: the morphological opening in detector.py will erase a lone block, so
    call detect_tampering(..., denoise=False) for this specific test.
    """
    out = image.copy()
    h, w = image.shape[:2]
    y, x = int(h * at[0]), int(w * at[1])
    mask = _empty_mask(image)
    if out.ndim == 3:
        out[y, x, 0] = np.uint8((int(out[y, x, 0]) + 1) % 256)
    else:
        out[y, x] = np.uint8((int(out[y, x]) + 1) % 256)
    mask[y, x] = True
    return out, mask


# --------------------------------------------------------------------------- #
# Global attacks -- these destroy the watermark everywhere.
# Expected behaviour is the UNVERIFIABLE verdict, NOT localisation.
# Include them in the report as a documented limitation.
# --------------------------------------------------------------------------- #
def jpeg_compress(image: np.ndarray, quality: int = 90):
    """Lossy re-encode. Wipes the LSB plane. Demonstrates the PNG-only constraint."""
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise IOError("JPEG encoding failed.")
    out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    mask = np.ones(image.shape[:2], dtype=bool)
    return out, mask


def gaussian_noise(image: np.ndarray, sigma: float = 2.0):
    """Additive noise across the whole frame."""
    noise = np.random.normal(0, sigma, image.shape)
    out = np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    mask = np.any(out != image, axis=2) if image.ndim == 3 else (out != image)
    return out, mask


def rescale(image: np.ndarray, factor: float = 0.5):
    """Downscale then upscale. Geometric attack; destroys block alignment."""
    h, w = image.shape[:2]
    small = cv2.resize(image, (max(int(w * factor), 1), max(int(h * factor), 1)),
                       interpolation=cv2.INTER_AREA)
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = np.ones(image.shape[:2], dtype=bool)
    return out, mask


ATTACK_REGISTRY = {
    "Copy-move forgery": copy_move,
    "Text overlay": add_text,
    "Region erase (blur)": erase_region,
    "Single pixel flip (+1)": single_pixel_flip,
    "JPEG re-compression (Q90)": jpeg_compress,
    "Gaussian noise (sigma=2)": gaussian_noise,
    "Rescale 0.5x": rescale,
}

# Attacks the scheme is designed to LOCALISE, versus those it can only DETECT.
LOCALISABLE = {
    "Copy-move forgery", "Text overlay",
    "Region erase (blur)", "Single pixel flip (+1)",
}
