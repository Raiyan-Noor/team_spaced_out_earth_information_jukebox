"""Image -> float32 value grid with NaN for no-data.

Primary path: invert a GIBS colormap (exact RGB lookup, nearest-colour fallback). Real GIBS PNGs, the
synthetic demo frames and test images all go through :func:`decode_with_colormap`.
See docs/SONIFICATION_SPEC.md section 3.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from jukebox.colormap import DEFAULT_TOLERANCE, Colormap

BBox = tuple[float, float, float, float]  # (minlon, minlat, maxlon, maxlat)
GLOBAL_BBOX: BBox = (-180.0, -90.0, 180.0, 90.0)

ImageSource = Union[str, Path, bytes, Image.Image, np.ndarray]


@dataclass
class ValueGrid:
    """Decoded data values for one frame on an equirectangular grid (row 0 = north)."""

    values: np.ndarray  # float32[H, W], NaN = no data
    units: str | None
    bbox: BBox
    source_info: dict = field(default_factory=dict)


def load_rgba(source: ImageSource) -> np.ndarray:
    """Load an image (path, PNG bytes, PIL image or array) as ``uint8[H, W, 4]``.

    Palette and greyscale PNGs are converted, so a palette index never leaks through as a colour.
    """
    if isinstance(source, np.ndarray):
        arr = source
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] == 3:
            arr = np.concatenate([arr, np.full(arr.shape[:2] + (1,), 255, dtype=arr.dtype)], axis=-1)
        return np.ascontiguousarray(arr.astype(np.uint8))
    if isinstance(source, Image.Image):
        img = source
    elif isinstance(source, bytes):
        img = Image.open(io.BytesIO(source))
    else:
        img = Image.open(source)
    img.load()
    return np.asarray(img.convert("RGBA"), dtype=np.uint8).copy()


def decode_with_colormap(
    source: ImageSource,
    colormap: Colormap,
    bbox: BBox = GLOBAL_BBOX,
    tolerance: float = DEFAULT_TOLERANCE,
    source_info: dict | None = None,
) -> ValueGrid:
    """Recover data values from a colormapped image. Transparent pixels (alpha 0) become NaN."""
    rgba = load_rgba(source)
    values = colormap.lookup(rgba[..., :3], tolerance=tolerance)
    values[rgba[..., 3] == 0] = np.nan
    info = {"decode": "colormap", "colormap": colormap.name, "approximate": False}
    info.update(source_info or {})
    return ValueGrid(values=values.astype(np.float32), units=colormap.units, bbox=bbox, source_info=info)
