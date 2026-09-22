"""
app.py
------
Streamlit dashboard for the Image Authentication & Tamper Detection System.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import streamlit as st

from attacks import ATTACK_REGISTRY, LOCALISABLE
from detector import detect_tampering, draw_overlay, heatmap
from report import generate_report
from utils import (bgr_to_rgb, decode_upload, encode_png_bytes, file_sha256,
                   localisation_scores, quality_metrics, timestamp)
from watermark import capacity_report, embed_watermark

st.set_page_config(page_title="ImageAuth", page_icon="🔍", layout="wide")

REPORT_DIR = "reports"
OUTPUT_DIR = "outputs"
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("🔍 ImageAuth")
st.sidebar.caption("Fragile watermarking for image authentication")

password = st.sidebar.text_input(
    "Secret key", type="password", value="",
    help="Required for both protecting and verifying. Same key, or verification fails.")
owner = st.sidebar.text_input("Owner name", value="")
block = st.sidebar.select_slider(
    "Block size (px)", options=[4, 8, 16, 32], value=8,
    help="Smaller blocks localise tampering more precisely but carry a shorter "
         "authentication tag per block.")

st.sidebar.divider()
st.sidebar.markdown(
    "**Protected images must stay as PNG.**\n\n"
    "This is a *fragile* watermark. JPEG re-compression, resizing or rotation "
    "destroy it everywhere and the result becomes UNVERIFIABLE rather than a "
    "localised tamper map. That is intended behaviour, not a defect."
)


def _require_key() -> bool:
    if not password:
        st.warning("Enter a secret key in the sidebar to continue.")
        return False
    return True


def _verdict_banner(result):
    if result.verdict == "AUTHENTIC":
        st.success(f"### ✅ AUTHENTIC\n{result.confidence_note}")
    elif result.verdict == "TAMPERED":
        st.error(f"### ⛔ TAMPERED\n{result.confidence_note}")
    else:
        st.warning(f"### ⚠️ UNVERIFIABLE\n{result.confidence_note}")


def _metric_row(result):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Verdict", result.verdict)
    c2.metric("Tampered area", f"{result.tampered_percent} %")
    c3.metric("Failed blocks", f"{result.tampered_blocks:,} / {result.total_blocks:,}")
    c4.metric("Regions found", len(result.regions))


def _make_report(result, image, overlay, filename, quality=None, scores=None):
    path = os.path.join(
        REPORT_DIR, f"report_{timestamp().replace(':', '-').replace(' ', '_')}.pdf")
    generate_report(path, result, owner, image, overlay, password,
                    filename=filename, quality=quality, scores=scores)
    with open(path, "rb") as fh:
        st.download_button("⬇️ Download PDF report", fh.read(),
                           file_name=os.path.basename(path),
                           mime="application/pdf", width='stretch')


tab_protect, tab_verify, tab_demo, tab_about = st.tabs(
    ["🛡️ Protect", "🔎 Verify", "🧪 Tamper Demo", "ℹ️ How it works"])


# --------------------------------------------------------------------------- #
# Tab 1 — Protect
# --------------------------------------------------------------------------- #
with tab_protect:
    st.header("Protect an image")
    st.write("Embeds an invisible, key-bound authentication watermark. "
             "The output is visually identical to the input.")

    up = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "tif"],
                          key="protect_upload")

    if up and _require_key():
        original = decode_upload(up.getvalue())
        with st.spinner("Embedding watermark…"):
            protected = embed_watermark(original, password, block=block)
            q = quality_metrics(original, protected)
            cap = capacity_report(original, block=block)

        c1, c2 = st.columns(2)
        c1.image(bgr_to_rgb(original), caption="Original", width='stretch')
        c2.image(bgr_to_rgb(protected), caption="Protected (watermarked)",
                 width='stretch')

        st.subheader("Imperceptibility")
        m = st.columns(4)
        m[0].metric("PSNR", f"{q['PSNR (dB)']} dB",
                    help="Above 40 dB is considered invisible to the human eye.")
        m[1].metric("SSIM", q["SSIM"])
        m[2].metric("Mean abs. change", q["MAE"])
        m[3].metric("Max pixel change", q["Max pixel change"])

        with st.expander("Embedding details"):
            st.json({
                "grid": f"{cap['grid'][0]} x {cap['grid'][1]} blocks",
                "authenticated blocks": cap["total_blocks"],
                "tag bits per block": cap["tag_bits_per_block"],
                "P(forged block accepted)": f"{cap['false_accept_probability']:.3e}",
                "pixels cropped to fit grid": cap["cropped_pixels"],
                "localisation resolution": cap["localisation_resolution_px"],
                "SHA-256 of protected file": file_sha256(protected),
            })

        st.download_button(
            "⬇️ Download protected image (PNG)", encode_png_bytes(protected),
            file_name=f"protected_{os.path.splitext(up.name)[0]}.png",
            mime="image/png", width='stretch')
        st.info("Save this as PNG. Re-saving as JPEG will destroy the watermark.")


# --------------------------------------------------------------------------- #
# Tab 2 — Verify
# --------------------------------------------------------------------------- #
with tab_verify:
    st.header("Verify an image")
    st.write("Checks every block against the key. Failed blocks are shaded red.")

    up = st.file_uploader("Upload an image to verify", type=["png", "bmp", "tif"],
                          key="verify_upload")
    denoise = st.checkbox(
        "Filter isolated single-block hits", value=True,
        help="Removes stray one-block detections. Turn OFF to see raw, "
             "unfiltered sensitivity down to a single pixel.")

    if up and _require_key():
        image = decode_upload(up.getvalue())
        with st.spinner("Verifying…"):
            result = detect_tampering(image, password, block=block, denoise=denoise)
            overlay = draw_overlay(image, result)

        _verdict_banner(result)
        _metric_row(result)

        c1, c2 = st.columns(2)
        c1.image(bgr_to_rgb(image), caption="Examined image", width='stretch')
        c2.image(bgr_to_rgb(overlay), caption="Tamper map", width='stretch')

        if result.regions:
            st.subheader("Tampered regions")
            st.dataframe(
                [{"Region": f"R{i}", "X": r["x"], "Y": r["y"],
                  "Width": r["w"], "Height": r["h"], "Area (px)": r["area_px"]}
                 for i, r in enumerate(result.regions, 1)],
                width='stretch', hide_index=True)

            with st.expander("Binary tamper mask"):
                st.image(heatmap(result, size=image.shape[:2]),
                         caption="White = failed authentication",
                         width='stretch')

        st.divider()
        _make_report(result, image, overlay, up.name)


# --------------------------------------------------------------------------- #
# Tab 3 — Tamper Demo
# --------------------------------------------------------------------------- #
with tab_demo:
    st.header("Tamper demo")
    st.write("Protect an image, apply a simulated attack, and score the detector "
             "against ground truth. This is the fastest way to demonstrate the "
             "system without an external image editor.")

    up = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp"],
                          key="demo_upload")
    attack_name = st.selectbox("Attack", list(ATTACK_REGISTRY.keys()))

    if up and _require_key():
        original = decode_upload(up.getvalue())

        if st.button("Run attack and detect", type="primary", width='stretch'):
            protected = embed_watermark(original, password, block=block)
            attacked, truth = ATTACK_REGISTRY[attack_name](protected)

            # A one-block edit is removed by the speck filter, so the
            # sensitivity test must run unfiltered.
            use_denoise = attack_name != "Single pixel flip (+1)"
            result = detect_tampering(attacked, password, block=block,
                                      denoise=use_denoise)
            overlay = draw_overlay(attacked, result)

            _verdict_banner(result)
            _metric_row(result)

            c1, c2, c3 = st.columns(3)
            c1.image(bgr_to_rgb(protected), caption="Protected", width='stretch')
            c2.image(bgr_to_rgb(attacked), caption=f"After: {attack_name}",
                     width='stretch')
            c3.image(bgr_to_rgb(overlay), caption="Detected", width='stretch')

            scores = None
            if attack_name in LOCALISABLE:
                scores = localisation_scores(truth, result.block_grid, block)
                st.subheader("Localisation accuracy vs ground truth")
                s = st.columns(4)
                s[0].metric("Precision", scores["Precision"])
                s[1].metric("Recall", scores["Recall"])
                s[2].metric("F1", scores["F1"])
                s[3].metric("Accuracy", scores["Accuracy"])
                st.caption(f"TP {scores['TP']} · FP {scores['FP']} · "
                           f"FN {scores['FN']} · TN {scores['TN']}")
            else:
                st.info(
                    "This is a global attack. It wipes the LSB plane across the "
                    "whole image, so localisation is not meaningful — the correct "
                    "outcome is the UNVERIFIABLE verdict. Precision and recall are "
                    "not reported.")

            st.divider()
            _make_report(result, attacked, overlay, f"{up.name} + {attack_name}",
                         quality=quality_metrics(original, protected), scores=scores)


# --------------------------------------------------------------------------- #
# Tab 4 — About
# --------------------------------------------------------------------------- #
with tab_about:
    st.header("How it works")
    st.markdown(r"""
### The scheme

This is a **fragile, keyed, block-wise self-authenticating watermark** — the
Wong scheme. Authentication needs a watermark that *breaks* on modification,
which is the opposite of a robust copyright watermark that must survive it.

**Embedding**, per 8×8 block, per colour channel:

1. Zero the LSB plane → this is the block's *content*.
2. Compute `HMAC-SHA256(content ‖ row ‖ col ‖ channel)` with a key derived from
   the password via PBKDF2 (100,000 iterations).
3. Write the first 64 bits of that tag into the block's 64 LSBs.

**Verification** recomputes the tag from the MSBs and compares it to the stored
LSBs. A mismatch flags that block.

### Why the design choices matter

| Choice | Reason |
|---|---|
| Tag stored in LSBs, computed over MSBs | Makes verification **blind** — no original image or side-channel file needed |
| Position `(row, col)` inside the HMAC | Defeats **block-swap attacks**; a valid block only verifies at its own coordinates |
| Keyed HMAC, not a plain hash | A plain hash lets anyone forge a block and recompute a valid tag |
| PBKDF2 key stretching | Short passwords are otherwise brute-forceable |
| Per-channel embedding | Catches single-channel edits a grayscale scheme would miss |

### Security margin

Each block carries a 64-bit tag per channel — 192 bits over three channels.
The probability a randomly altered block still validates is 2⁻¹⁹², around
10⁻⁵⁸. False positives are not a practical concern.

### Known limitations — state these in the report

- **No robustness.** JPEG, resizing, rotation, and cropping invalidate the whole
  image. Fragile watermarking trades robustness for localisation; you cannot
  have both in one watermark.
- **Block-resolution localisation.** Tampering is located to the nearest 8×8
  block, not the exact pixel.
- **Edge cropping.** Rows and columns that do not fill a whole block are dropped
  during embedding.
- **Integrity, not truth.** The system proves the file's bits are unchanged. It
  cannot tell you whether the original photograph was staged or misleading.
""")
