"""
detector.py
-----------
Turns the raw per-block verification grid into a usable forensic result:
a cleaned tamper mask, connected-region statistics, and a visual overlay.

Why post-processing is needed
=============================
The raw grid from verify_watermark() is technically complete but forensically
noisy. Two problems:

1. Isolated single-block hits. A genuine edit is spatially contiguous -- an
   attacker pastes a face, adds text, clones a region. One lone flagged block
   surrounded by clean blocks is far more likely to be a stray bit error (bad
   transfer, a stripped alpha channel) than a real forgery. Morphological
   opening removes these.

2. Holes inside a tampered region. If a pasted patch happens to contain a flat
   area whose LSBs coincidentally match, a few interior blocks pass. Closing
   fills them so the reported region is a solid shape, not swiss cheese.

Both operations run on the BLOCK grid, not the pixel grid. Doing this at pixel
resolution would be meaningless -- the detector has no sub-block precision.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from watermark import BLOCK_SIZE, LSB_BITS, verify_watermark

# A region smaller than this many blocks is discarded as noise.
MIN_REGION_BLOCKS = 2


@dataclass
class DetectionResult:
    """Everything the UI and the PDF report need, in one object."""

    authentic: bool
    tampered_blocks: int
    total_blocks: int
    tampered_percent: float
    regions: list = field(default_factory=list)   # bounding boxes in PIXELS
    block_grid: np.ndarray = None                 # bool, post-cleanup
    raw_grid: np.ndarray = None                   # bool, pre-cleanup
    block_size: int = BLOCK_SIZE
    verdict: str = ""
    confidence_note: str = ""


def _clean_grid(grid: np.ndarray) -> np.ndarray:
    """Remove isolated hits and fill interior holes, WITHOUT moving any boundary.

    Implementation note (this was a real bug, keep the comment):
    the obvious approach is morphological opening + closing with a 2x2 kernel.
    Do not do that. OpenCV anchors an even-sized kernel off-centre, so open and
    close each translate the mask by one block. Measured against ground truth,
    that dropped copy-move localisation from precision 1.000 to 0.716 -- it
    manufactured 48 false positives and 48 false negatives purely from the
    kernel geometry.

    Both operations are instead done in a translation-free way:
      * speck removal  -> connected-component analysis with an area threshold
      * hole filling   -> flood fill the background inward from the border;
                          anything unreached is an interior hole
    """
    if not grid.any():
        return grid

    g = grid.astype(np.uint8)

    # --- drop components smaller than the noise threshold ---
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(g, connectivity=8)
    keep = np.zeros(n_labels, dtype=bool)
    for i in range(1, n_labels):
        keep[i] = stats[i, cv2.CC_STAT_AREA] >= MIN_REGION_BLOCKS
    g = keep[labels].astype(np.uint8)

    if not g.any():
        return g.astype(bool)

    # --- fill interior holes ---
    # The flood fill must be seeded on BACKGROUND. Seeding at (0, 0) is a bug:
    # when the top-left block is itself tampered the seed lands on foreground,
    # so no background is ever painted, every background cell looks "unreached",
    # and the whole grid is reported as a hole. That flags 100% of the image and
    # flips the verdict from TAMPERED to UNVERIFIABLE -- a real forgery in the
    # top-left corner would be dismissed as a wrong password. Padding a one-cell
    # background border guarantees a valid seed and keeps every border-connected
    # background cell reachable, so the fill stays translation-free.
    padded = cv2.copyMakeBorder(g, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(padded, ff_mask, (0, 0), 1)   # paint reachable background
    holes = (padded[1:-1, 1:-1] == 0)           # background never reached = hole
    return (g.astype(bool) | holes)


def _regions_from_grid(grid: np.ndarray, block: int) -> list:
    """Connected-component analysis -> pixel-space bounding boxes."""
    if not grid.any():
        return []

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        grid.astype(np.uint8), connectivity=8
    )

    regions = []
    for i in range(1, n_labels):  # label 0 is background
        area_blocks = int(stats[i, cv2.CC_STAT_AREA])
        if area_blocks < MIN_REGION_BLOCKS:
            continue
        x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        regions.append(
            {
                "x": int(x * block),
                "y": int(y * block),
                "w": int(w * block),
                "h": int(h * block),
                "area_blocks": area_blocks,
                "area_px": area_blocks * block * block,
            }
        )

    regions.sort(key=lambda r: r["area_blocks"], reverse=True)
    return regions


def detect_tampering(image: np.ndarray, password: str,
                     block: int = BLOCK_SIZE,
                     lsb_bits: int = LSB_BITS,
                     denoise: bool = True) -> DetectionResult:
    """Full verification pipeline. Returns a DetectionResult.

    A note on the verdict logic: if EVERY block fails, the most likely cause is
    not that someone repainted the whole image -- it is that the password is
    wrong, or the file was re-saved as JPEG and the LSB plane was destroyed.
    Reporting "100% tampered" in that case is technically true but practically
    misleading, so we distinguish the two cases explicitly.
    """
    raw = verify_watermark(image, password, block=block, lsb_bits=lsb_bits)
    grid = _clean_grid(raw) if denoise else raw

    total = int(grid.size)
    n_tampered = int(grid.sum())
    percent = (n_tampered / total * 100.0) if total else 0.0
    regions = _regions_from_grid(grid, block)

    if n_tampered == 0:
        verdict = "AUTHENTIC"
        note = "All blocks verified against the supplied key. No modification detected."
    elif percent > 95.0:
        verdict = "UNVERIFIABLE"
        note = (
            "Nearly every block failed. This pattern indicates a wrong password, "
            "an image that was never watermarked, or lossy re-encoding (JPEG / "
            "resize) that destroyed the LSB plane -- not a localised edit. "
            "Confirm the password and that the file is the original PNG."
        )
    else:
        verdict = "TAMPERED"
        note = (
            f"{n_tampered} of {total} blocks failed authentication across "
            f"{len(regions)} contiguous region(s). Localisation resolution is "
            f"{block}x{block} px."
        )

    return DetectionResult(
        authentic=(n_tampered == 0),
        tampered_blocks=n_tampered,
        total_blocks=total,
        tampered_percent=round(percent, 2),
        regions=regions,
        block_grid=grid,
        raw_grid=raw,
        block_size=block,
        verdict=verdict,
        confidence_note=note,
    )


def draw_overlay(image: np.ndarray, result: DetectionResult,
                 shade: bool = True, boxes: bool = True) -> np.ndarray:
    """Render the tamper map onto the image: red shading + region bounding boxes."""
    h, w = image.shape[:2]
    block = result.block_size
    canvas = image.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    if result.block_grid is None or not result.block_grid.any():
        return canvas

    if shade:
        # Upscale the block grid to pixel resolution with nearest-neighbour so
        # each flagged block becomes an exact block-sized square.
        mask = cv2.resize(
            result.block_grid.astype(np.uint8) * 255,
            (result.block_grid.shape[1] * block, result.block_grid.shape[0] * block),
            interpolation=cv2.INTER_NEAREST,
        )
        mask = cv2.copyMakeBorder(
            mask, 0, h - mask.shape[0], 0, w - mask.shape[1], cv2.BORDER_CONSTANT, value=0
        )
        red = np.zeros_like(canvas)
        red[:, :] = (0, 0, 255)  # BGR
        alpha = (mask.astype(np.float32) / 255.0 * 0.45)[:, :, None]
        canvas = (canvas * (1 - alpha) + red * alpha).astype(np.uint8)

    if boxes:
        for i, reg in enumerate(result.regions, start=1):
            cv2.rectangle(
                canvas,
                (reg["x"], reg["y"]),
                (reg["x"] + reg["w"], reg["y"] + reg["h"]),
                (0, 255, 255), 2,
            )
            cv2.putText(
                canvas, f"R{i}", (reg["x"] + 3, max(reg["y"] - 5, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA,
            )

    return canvas


def heatmap(result: DetectionResult, size: tuple = None) -> np.ndarray:
    """Standalone binary tamper map (white = tampered). Useful as a report figure."""
    if result.block_grid is None:
        return np.zeros((1, 1), np.uint8)
    m = result.block_grid.astype(np.uint8) * 255
    if size:
        m = cv2.resize(m, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    return m
