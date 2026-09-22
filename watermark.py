"""
watermark.py
------------
Fragile, keyed, block-wise self-authenticating watermark (Wong-style scheme).

Design rationale
================
Tamper DETECTION and tamper LOCALIZATION require a *fragile* watermark, not a
robust one. A robust DCT/DWT watermark is designed to SURVIVE modification --
which is the exact opposite of what an authentication system needs. We want the
watermark to shatter the moment a single pixel changes, and to shatter *locally*
so we know which region was touched.

Scheme
------
1. Split the image into BxB blocks (default 8x8), per colour channel.
2. For each block, zero out the LSB plane -> this is the "content" (MSB) part.
3. Compute HMAC-SHA256( MSB-content || block_row || block_col || channel )
   using a secret key derived from the owner's password.
4. Write the first (B*B*lsb_bits) bits of that digest into the block's LSBs.

Because the authentication tag is computed over the MSBs only, and stored in
the LSBs only, extraction is *blind*: verification needs the key and nothing
else. No side-channel file, no copy of the original image.

Including (row, col, channel) in the HMAC input defeats the vector-quantisation
/ block-swap attack, where an attacker rearranges authenticated blocks from the
same image. A block only validates at the position it was signed for.

Fragility note (state this in your report and viva)
--------------------------------------------------
This watermark does NOT survive JPEG re-compression, resizing, or any lossy
operation. That is by design. Protected images MUST be stored as PNG.
"""

from __future__ import annotations

import hashlib
import hmac

import numpy as np

BLOCK_SIZE = 8
LSB_BITS = 1
KDF_ITERATIONS = 100_000
KDF_SALT = b"ImageAuth-v1-static-salt"


# --------------------------------------------------------------------------- #
# Key handling
# --------------------------------------------------------------------------- #
def derive_key(password: str) -> bytes:
    """Stretch a human password into a 32-byte HMAC key via PBKDF2-HMAC-SHA256.

    A raw password must never be used directly as a MAC key: short, low-entropy
    passwords are trivially brute-forced. PBKDF2 with 100k iterations makes each
    guess expensive.

    NOTE: the salt is fixed so that the same password always yields the same key
    (required, since the verifier only has the password). In a production system
    the salt would be stored alongside the image in a metadata chunk.
    """
    if not password:
        raise ValueError("Password must not be empty.")
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), KDF_SALT, KDF_ITERATIONS, dklen=32
    )


# --------------------------------------------------------------------------- #
# Bit-level helpers
# --------------------------------------------------------------------------- #
def _hash_bits(key: bytes, message: bytes, n_bits: int) -> np.ndarray:
    """Return exactly n_bits pseudorandom bits from a keyed hash of `message`.

    SHA-256 emits 256 bits per call. If more are needed (large blocks or
    lsb_bits=2), we extend by appending a counter, which is the standard
    MGF1-style construction.
    """
    out = bytearray()
    counter = 0
    while len(out) * 8 < n_bits:
        out += hmac.new(key, message + counter.to_bytes(2, "big"), hashlib.sha256).digest()
        counter += 1
    return np.unpackbits(np.frombuffer(bytes(out), dtype=np.uint8))[:n_bits]


def _pack_bits(bits: np.ndarray, lsb_bits: int) -> np.ndarray:
    """Pack a flat bit array into per-pixel integers of `lsb_bits` width."""
    groups = bits.reshape(-1, lsb_bits)
    values = np.zeros(groups.shape[0], dtype=np.uint8)
    for i in range(lsb_bits):
        values |= (groups[:, i] << (lsb_bits - 1 - i)).astype(np.uint8)
    return values


def _unpack_bits(values: np.ndarray, lsb_bits: int) -> np.ndarray:
    """Inverse of _pack_bits: expand per-pixel integers back into a bit array."""
    values = values.astype(np.uint8).ravel()
    bits = np.zeros((values.size, lsb_bits), dtype=np.uint8)
    for i in range(lsb_bits):
        bits[:, i] = (values >> (lsb_bits - 1 - i)) & 1
    return bits.ravel()


def _block_tag(key: bytes, msb_block: np.ndarray, row: int, col: int,
               channel: int, n_bits: int) -> np.ndarray:
    """Authentication tag for one block, bound to its position and channel."""
    message = (
        msb_block.tobytes()
        + row.to_bytes(4, "big")
        + col.to_bytes(4, "big")
        + channel.to_bytes(1, "big")
    )
    return _hash_bits(key, message, n_bits)


def _crop_to_grid(img: np.ndarray, block: int) -> np.ndarray:
    """Crop the image so both dimensions are exact multiples of the block size.

    Partial edge blocks cannot be authenticated (they hold fewer LSBs than the
    tag needs), so we drop them rather than leave an unprotected border.
    """
    h, w = img.shape[:2]
    return img[: (h // block) * block, : (w // block) * block]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def embed_watermark(image: np.ndarray, password: str,
                    block: int = BLOCK_SIZE,
                    lsb_bits: int = LSB_BITS) -> np.ndarray:
    """Embed the fragile authentication watermark. Returns the protected image.

    Parameters
    ----------
    image : uint8 array, HxW (grayscale) or HxWx3 (BGR/RGB).
    password : owner secret; the same string is required to verify.
    block : block edge length in pixels. Smaller -> finer localisation,
            but fewer LSBs per block, so a shorter (weaker) tag.
    lsb_bits : LSB planes used per pixel. 1 keeps PSNR around 51 dB.
    """
    if image.dtype != np.uint8:
        raise TypeError("Image must be uint8.")
    if lsb_bits not in (1, 2):
        raise ValueError("lsb_bits must be 1 or 2.")

    key = derive_key(password)
    img = _crop_to_grid(image, block).copy()

    if img.ndim == 2:
        img = img[:, :, None]
        was_gray = True
    else:
        was_gray = False

    h, w, n_ch = img.shape
    if h < block or w < block:
        raise ValueError(f"Image too small: needs at least {block}x{block} px.")

    clear_mask = np.uint8(0xFF ^ ((1 << lsb_bits) - 1))
    n_bits = block * block * lsb_bits

    for ch in range(n_ch):
        plane = img[:, :, ch]
        for r in range(0, h, block):
            for c in range(0, w, block):
                region = plane[r:r + block, c:c + block]
                msb = region & clear_mask
                tag = _block_tag(key, msb, r // block, c // block, ch, n_bits)
                payload = _pack_bits(tag, lsb_bits).reshape(block, block)
                plane[r:r + block, c:c + block] = msb | payload

    return img[:, :, 0] if was_gray else img


def verify_watermark(image: np.ndarray, password: str,
                     block: int = BLOCK_SIZE,
                     lsb_bits: int = LSB_BITS) -> np.ndarray:
    """Verify every block. Returns a boolean grid: True = block is TAMPERED.

    Shape of the returned grid is (H // block, W // block). Channels are
    collapsed with a logical OR: if any channel of a block fails, the block is
    flagged. This catches single-channel edits that a grayscale-only scheme
    would miss.
    """
    if image.dtype != np.uint8:
        raise TypeError("Image must be uint8.")

    key = derive_key(password)
    img = _crop_to_grid(image, block)
    if img.ndim == 2:
        img = img[:, :, None]

    h, w, n_ch = img.shape
    clear_mask = np.uint8(0xFF ^ ((1 << lsb_bits) - 1))
    read_mask = np.uint8((1 << lsb_bits) - 1)
    n_bits = block * block * lsb_bits

    tampered = np.zeros((h // block, w // block), dtype=bool)

    for ch in range(n_ch):
        plane = img[:, :, ch]
        for r in range(0, h, block):
            for c in range(0, w, block):
                region = plane[r:r + block, c:c + block]
                msb = region & clear_mask
                expected = _block_tag(key, msb, r // block, c // block, ch, n_bits)
                found = _unpack_bits(region & read_mask, lsb_bits)
                if not np.array_equal(expected, found):
                    tampered[r // block, c // block] = True

    return tampered


def capacity_report(image: np.ndarray, block: int = BLOCK_SIZE,
                    lsb_bits: int = LSB_BITS) -> dict:
    """Diagnostic info for the write-up: grid size, tag length, false-accept rate."""
    img = _crop_to_grid(image, block)
    h, w = img.shape[:2]
    n_ch = 1 if img.ndim == 2 else img.shape[2]
    tag_bits = block * block * lsb_bits
    return {
        "usable_size": (h, w),
        "cropped_pixels": (image.shape[0] * image.shape[1]) - (h * w),
        "grid": (h // block, w // block),
        "total_blocks": (h // block) * (w // block) * n_ch,
        "tag_bits_per_block": tag_bits,
        # Probability a randomly-altered block still produces a matching tag.
        "false_accept_probability": 2.0 ** (-tag_bits * n_ch),
        "localisation_resolution_px": f"{block}x{block}",
    }
