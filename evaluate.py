"""
evaluate.py
-----------
Automated benchmark. Run this once, paste the output table into your report.

    python evaluate.py --image samples/test.png --password secret123

For every attack in attacks.py it embeds, attacks, detects, and scores the
detector against ground truth. This is the results section of your report,
produced in one command and reproducible by your examiner.
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

from attacks import ATTACK_REGISTRY, LOCALISABLE
from detector import detect_tampering, draw_overlay
from utils import (load_image, localisation_scores, quality_metrics, save_png)
from watermark import capacity_report, embed_watermark


def make_sample(path: str, size=(512, 512)) -> str:
    """Synthesise a test image if the user has none handy."""
    h, w = size
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :, 0] = (xx * 255 // w).astype(np.uint8)
    img[:, :, 1] = (yy * 255 // h).astype(np.uint8)
    img[:, :, 2] = ((np.sin(xx / 18.0) * np.cos(yy / 22.0) + 1) * 110).astype(np.uint8)
    cv2.circle(img, (w // 3, h // 3), min(h, w) // 7, (245, 245, 245), -1)
    cv2.rectangle(img, (int(w * 0.6), int(h * 0.6)), (int(w * 0.88), int(h * 0.88)),
                  (20, 20, 200), -1)
    # Texture matters: a perfectly flat image makes LSB embedding look artificially good.
    rng = np.random.default_rng(0)
    img = np.clip(img.astype(np.int16) + rng.integers(-8, 9, img.shape), 0, 255).astype(np.uint8)
    save_png(img, path)
    return path


def main():
    ap = argparse.ArgumentParser(description="ImageAuth benchmark")
    ap.add_argument("--image", default="samples/test.png")
    ap.add_argument("--password", default="secret123")
    ap.add_argument("--block", type=int, default=8)
    ap.add_argument("--outdir", default="outputs/evaluation")
    args = ap.parse_args()

    if not os.path.exists(args.image):
        print(f"[i] {args.image} not found -- generating a synthetic test image.")
        os.makedirs(os.path.dirname(args.image) or ".", exist_ok=True)
        make_sample(args.image)

    os.makedirs(args.outdir, exist_ok=True)
    original = load_image(args.image)

    # ---- 1. Embed -------------------------------------------------------- #
    protected = embed_watermark(original, args.password, block=args.block)
    save_png(protected, os.path.join(args.outdir, "protected.png"))

    q = quality_metrics(original, protected)
    cap = capacity_report(original, block=args.block)

    print("\n" + "=" * 78)
    print("1. EMBEDDING  (imperceptibility)")
    print("=" * 78)
    print(f"  Image                    : {original.shape[1]}x{original.shape[0]}")
    print(f"  Usable after block crop  : {cap['usable_size'][1]}x{cap['usable_size'][0]}"
          f"  ({cap['cropped_pixels']} px dropped)")
    print(f"  Block grid               : {cap['grid'][0]} x {cap['grid'][1]}")
    print(f"  Tag length per block     : {cap['tag_bits_per_block']} bits")
    print(f"  P(forged block accepted) : {cap['false_accept_probability']:.3e}")
    for k, v in q.items():
        print(f"  {k:25}: {v}")

    # ---- 2. Clean verification ------------------------------------------- #
    clean = detect_tampering(protected, args.password, block=args.block)
    wrong = detect_tampering(protected, "wrong_password", block=args.block)

    print("\n" + "=" * 78)
    print("2. BASELINE VERIFICATION")
    print("=" * 78)
    print(f"  Correct password  -> {clean.verdict:14} "
          f"({clean.tampered_blocks}/{clean.total_blocks} blocks failed)")
    print(f"  Wrong password    -> {wrong.verdict:14} "
          f"({wrong.tampered_percent}% blocks failed)")

    # ---- 3. Attack sweep -------------------------------------------------- #
    print("\n" + "=" * 78)
    print("3. ATTACK SWEEP")
    print("=" * 78)
    hdr = f"{'Attack':<28}{'Verdict':<15}{'Tampered%':>10}{'Prec':>8}{'Recall':>8}{'F1':>8}"
    print(hdr)
    print("-" * 78)

    rows = []
    for name, fn in ATTACK_REGISTRY.items():
        # A single-block edit is erased by morphological opening, so the
        # sensitivity test must run with denoising off.
        denoise = name != "Single pixel flip (+1)"
        try:
            attacked, truth = fn(protected)
        except Exception as exc:
            print(f"{name:<28}ERROR: {exc}")
            continue

        res = detect_tampering(attacked, args.password,
                               block=args.block, denoise=denoise)
        line = f"{name:<28}{res.verdict:<15}{res.tampered_percent:>9.2f}%"

        if name in LOCALISABLE:
            s = localisation_scores(truth, res.block_grid, args.block)
            line += f"{s['Precision']:>8.3f}{s['Recall']:>8.3f}{s['F1']:>8.3f}"
            rows.append((name, res, s))
        else:
            line += f"{'n/a':>8}{'n/a':>8}{'n/a':>8}"
            rows.append((name, res, None))

        print(line)

        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        slug = slug.replace("+", "plus").replace("-", "_").replace("=", "").replace(".", "")
        save_png(attacked, os.path.join(args.outdir, f"{slug}_attacked.png"))
        save_png(draw_overlay(attacked, res),
                 os.path.join(args.outdir, f"{slug}_detected.png"))

    print("-" * 78)
    print("Note: Precision/Recall are reported only for localisable attacks.")
    print("      Global attacks (JPEG, noise, rescale) destroy the whole LSB plane;")
    print("      the correct outcome there is the UNVERIFIABLE verdict, not localisation.")
    print(f"\n[OK] Images written to {args.outdir}/")


if __name__ == "__main__":
    main()
