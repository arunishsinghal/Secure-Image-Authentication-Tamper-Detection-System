# ImageAuth — Secure Image Authentication & Tamper Detection

Fragile block-wise watermarking for image integrity verification and tamper
localisation. Detects *whether* an image was modified, *where* it was modified,
and produces a signed PDF verification report.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Reproduce the benchmark (generates a test image if you have none):

```bash
python evaluate.py --image samples/test.png --password secret123
```

## Results

Measured on a 512×512 test image, 8×8 blocks, 1 LSB per pixel.

**Imperceptibility**

| Metric | Value |
|---|---|
| PSNR | 51.14 dB |
| SSIM | 0.9961 |
| Mean absolute pixel change | 0.50 |
| Max single-pixel change | 1 |

**Tamper localisation**

| Attack | Verdict | Tampered | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Copy-move forgery | TAMPERED | 4.13 % | 1.000 | 1.000 | 1.000 |
| Text overlay | TAMPERED | 3.39 % | 1.000 | 1.000 | 1.000 |
| Region erase (blur) | TAMPERED | 4.79 % | 1.000 | 1.000 | 1.000 |
| Single pixel flip (+1) | TAMPERED | 0.02 % | 1.000 | 1.000 | 1.000 |
| JPEG re-compression Q90 | UNVERIFIABLE | 100 % | n/a | n/a | n/a |
| Gaussian noise σ=2 | UNVERIFIABLE | 100 % | n/a | n/a | n/a |
| Rescale 0.5× | UNVERIFIABLE | 100 % | n/a | n/a | n/a |

Wrong password on an untampered image correctly returns UNVERIFIABLE (100 % of
blocks fail), not a false localisation.

## Scheme

Fragile keyed watermark (Wong scheme). Per 8×8 block, per channel:

1. Zero the LSB plane to isolate the block *content*.
2. Compute `HMAC-SHA256(content ‖ row ‖ col ‖ channel)`, key derived from the
   password by PBKDF2-HMAC-SHA256 (100,000 iterations).
3. Store the first 64 bits of the tag in the block's 64 LSBs.

Verification recomputes the tag from the MSBs and compares. Blind — no original
image or side-channel file is needed, only the password.

Binding `(row, col, channel)` into the HMAC defeats block-swap attacks: a valid
block only authenticates at the coordinates it was signed for. Security margin
is 2⁻¹⁹² per block across three channels.

## Limitations

- **Not robust.** JPEG, resize, rotate and crop invalidate the whole image.
  This is inherent: fragility is what buys localisation.
- **Block-resolution localisation** — 8×8 px, not exact pixels.
- **Edge cropping** — partial blocks at the right/bottom edge are dropped.
- **Integrity ≠ truth.** Proves the bits are unchanged; says nothing about
  whether the original photograph was staged.

## Files

| File | Role |
|---|---|
| `app.py` | Streamlit UI — Protect / Verify / Demo / About tabs |
| `watermark.py` | Embedding and verification, PBKDF2 key derivation |
| `detector.py` | Mask cleanup, connected regions, overlay rendering |
| `attacks.py` | Simulated attacks with pixel-level ground-truth masks |
| `utils.py` | I/O, PSNR / SSIM / MAE, precision-recall scoring |
| `report.py` | PDF verification certificate |
| `evaluate.py` | Full benchmark, one command |
