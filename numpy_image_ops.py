
"""numpy_image_ops.py
---------------------------------
A tiny, pure-NumPy collection of common image manipulation & augmentation ops.
- Assumes images are HWC (height, width, channels) unless noted.
- Uses simple kernels (e.g., nearest-neighbor for resize).
- Prefer float32 in [0, 1] for stable math; clip as needed.
- For high-quality resampling / arbitrary-angle rotation, use SciPy or OpenCV.
"""

from __future__ import annotations
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

__all__ = [
    # Resize & crops
    "nn_resize", "center_crop", "random_crop",
    # Flips & rotations
    "hflip", "vflip", "rot90", "rot180", "rot270",
    # Padding
    "pad",
    # Normalization & tone
    "normalize", "minmax", "bright", "contrast", "to_gray", "saturate",
    # Blur & sharpen
    "box_blur", "sharpen",
    # Noise
    "add_gaussian_noise", "salt_pepper",
    # Erasing
    "random_erase",
]

# -----------------------------
# Resize (Nearest-Neighbor)
# -----------------------------
def nn_resize(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Nearest-neighbor resize for HxWxC or HxW arrays.

    Parameters
    ----------
    img : np.ndarray
        Input image (H, W, C) or (H, W).
    out_h : int
        Output height.
    out_w : int
        Output width.

    Returns
    -------
    np.ndarray
        Resized image with shape (out_h, out_w, C) or (out_h, out_w).
    """
    H, W = img.shape[:2]
    # Map output grid to nearest integer source indices
    ys = np.linspace(0, H - 1, out_h).astype(int)
    xs = np.linspace(0, W - 1, out_w).astype(int)
    return img[ys[:, None], xs[None, :]]

# -----------------------------
# Crops
# -----------------------------
def center_crop(img: np.ndarray, ch: int, cw: int) -> np.ndarray:
    """Center crop to (ch, cw)."""
    H, W = img.shape[:2]
    top  = max(0, (H - ch) // 2)
    left = max(0, (W - cw) // 2)
    return img[top:top+ch, left:left+cw]

def random_crop(img: np.ndarray, ch: int, cw: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Random crop to (ch, cw)."""
    H, W = img.shape[:2]
    if ch > H or cw > W:
        raise ValueError(f"Crop size ({ch},{cw}) must be <= image size ({H},{W}).")
    rng = rng or np.random.default_rng()
    top  = int(rng.integers(0, H - ch + 1))
    left = int(rng.integers(0, W - cw + 1))
    return img[top:top+ch, left:left+cw]

# -----------------------------
# Flips
# -----------------------------
def hflip(img: np.ndarray) -> np.ndarray:
    """Horizontal flip (left↔right)."""
    return np.flip(img, axis=1)

def vflip(img: np.ndarray) -> np.ndarray:
    """Vertical flip (top↔bottom)."""
    return np.flip(img, axis=0)

# -----------------------------
# Rotations (multiples of 90°)
# -----------------------------
def rot90(img: np.ndarray) -> np.ndarray:
    """Rotate 90° CCW."""
    return np.rot90(img, 1)

def rot180(img: np.ndarray) -> np.ndarray:
    """Rotate 180°."""
    return np.rot90(img, 2)

def rot270(img: np.ndarray) -> np.ndarray:
    """Rotate 270° CCW (or 90° CW)."""
    return np.rot90(img, 3)

# -----------------------------
# Padding
# -----------------------------
def pad(
    img: np.ndarray,
    top: int = 0, bottom: int = 0, left: int = 0, right: int = 0,
    mode: str = "constant", value: float | int = 0,
) -> np.ndarray:
    """Pad image with given border sizes and mode.

    Modes include: 'constant', 'edge', 'reflect', 'symmetric'.
    - 'constant' uses the `value` as fill.
    - For non-constant modes, `value` is ignored.

    Returns an array with the same number of channels as input.
    """
    pad_width = ((top, bottom), (left, right))
    if img.ndim == 3:
        pad_width = pad_width + ((0, 0),)
    if mode == "constant":
        return np.pad(img, pad_width, mode=mode, constant_values=value)
    else:
        return np.pad(img, pad_width, mode=mode)

# -----------------------------
# Normalization & Tone Ops
# -----------------------------
def normalize(img: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-channel normalize: (img - mean) / std

    mean, std: shape (C,) for HWC input, or scalar for HW.
    """
    if img.ndim == 3:
        mean = np.asarray(mean)[None, None, :]
        std  = np.asarray(std)[None, None, :]
    return (img - mean) / (std + eps)

def minmax(img: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Min-max normalize image to [0, 1]."""
    return (img - img.min()) / (img.ptp() + eps)

def bright(img: np.ndarray, b: float) -> np.ndarray:
    """Adjust brightness by adding offset b (assumes float image)."""
    return img + b

def contrast(img: np.ndarray, c: float) -> np.ndarray:
    """Scale contrast around global (or per-channel) mean by factor c."""
    m = img.mean(axis=(0, 1), keepdims=True)
    return (img - m) * c + m

def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert RGB HWC image to grayscale (3-channel) via luminance."""
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        g = img[..., 0] if img.ndim == 3 else img
    else:
        g = np.dot(img[..., :3], [0.299, 0.587, 0.114])  # (H, W)
    return np.repeat(g[..., None], 3, axis=2)

def saturate(img: np.ndarray, s: float) -> np.ndarray:
    """Interpolate between grayscale (s=0) and original (s=1)."""
    g3 = to_gray(img)
    return g3 * (1 - s) + img * s

# -----------------------------
# Blur & Sharpen
# -----------------------------
def box_blur(img: np.ndarray, k: int = 3) -> np.ndarray:
    """Box blur via sliding window mean (reflect padding).

    k: odd kernel size (e.g., 3, 5, 7).
    """
    if k % 2 == 0:
        raise ValueError("k must be odd.")
    pad_r = k // 2
    x = pad(img, pad_r, pad_r, pad_r, pad_r, mode='reflect')
    if img.ndim == 3:
        # Build (H, W, k, k, C) windows, then mean over spatial window
        patches = sliding_window_view(x, (k, k, 1))[:, :, :, :, 0]
        return patches.mean((2, 3))
    else:
        patches = sliding_window_view(x, (k, k))
        return patches.mean((2, 3))

def sharpen(img: np.ndarray, alpha: float = 1.0, k: int = 3) -> np.ndarray:
    """Unsharp masking: img + alpha * (img - blur)."""
    blur = box_blur(img, k)
    return img + alpha * (img - blur)

# -----------------------------
# Noise
# -----------------------------
def add_gaussian_noise(img: np.ndarray, sigma: float = 0.05, rng: np.random.Generator | None = None) -> np.ndarray:
    """Additive zero-mean Gaussian noise with std=sigma."""
    rng = rng or np.random.default_rng()
    return img + rng.normal(0.0, sigma, img.shape).astype(img.dtype, copy=False)

def salt_pepper(img: np.ndarray, p: float = 0.02, rng: np.random.Generator | None = None) -> np.ndarray:
    """Salt & Pepper noise: with prob p/2 set to 1, with prob p/2 set to 0 (per pixel)."""
    rng = rng or np.random.default_rng()
    H, W = img.shape[:2]
    mask = rng.random((H, W))
    salt = mask < (p / 2)
    pep  = (mask >= (p / 2)) & (mask < p)
    out = img.copy()
    if img.ndim == 3:
        out[salt] = 1.0
        out[pep]  = 0.0
    else:
        out[salt] = 1.0
        out[pep]  = 0.0
    return out

# -----------------------------
# Random Erasing (Cutout)
# -----------------------------
def random_erase(img: np.ndarray, er_h: int, er_w: int, value: float | int = 0.0,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """Erase a random (er_h x er_w) rectangle by filling with `value`."""
    H, W = img.shape[:2]
    er_h = min(er_h, H)
    er_w = min(er_w, W)
    rng = rng or np.random.default_rng()
    top  = int(rng.integers(0, max(1, H - er_h + 1)))
    left = int(rng.integers(0, max(1, W - er_w + 1)))
    out = img.copy()
    if img.ndim == 3:
        out[top:top+er_h, left:left+er_w, :] = value
    else:
        out[top:top+er_h, left:left+er_w] = value
    return out
