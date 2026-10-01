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


# --- fallbacks for frames without a machine-readable colormap (EIC / SVS images) -----------------------------

GRAY_CHROMA = 12  # max(R,G,B) - min(R,G,B) at or below this counts as "near gray"
CMAP_TOLERANCE = 30.0  # looser than GIBS: these images are often JPEGs or resampled


def gray_mask(rgba: np.ndarray, chroma: int = GRAY_CHROMA) -> np.ndarray:
    """True where a pixel is near-gray (land fill, text, borders, coastlines)."""
    rgb = rgba[..., :3].astype(np.int16)
    return (rgb.max(axis=-1) - rgb.min(axis=-1)) <= chroma


def srgb_to_lstar(rgb: np.ndarray) -> np.ndarray:
    """CIE L* (0-100) of sRGB uint8 colours (D65 white)."""
    c = rgb.astype(np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    y = lin @ np.array([0.2126, 0.7152, 0.0722])
    eps = (6 / 29) ** 3
    f = np.where(y > eps, np.cbrt(y), y / (3 * (6 / 29) ** 2) + 4 / 29)
    return 116.0 * f - 16.0


def decode_luminance(
    source: ImageSource, bbox: BBox = GLOBAL_BBOX, mask_gray: bool = False, source_info: dict | None = None
) -> ValueGrid:
    """Last-resort decode: perceptual lightness L* scaled to [0, 1]. Values are approximate brightness."""
    rgba = load_rgba(source)
    values = (srgb_to_lstar(rgba[..., :3]) / 100.0).astype(np.float32)
    values[rgba[..., 3] == 0] = np.nan
    if mask_gray:
        values[gray_mask(rgba)] = np.nan
    info = {"decode": "luminance", "approximate": True}
    info.update(source_info or {})
    return ValueGrid(values=values, units=None, bbox=bbox, source_info=info)


def matplotlib_colormap(name: str, vmin: float, vmax: float, units: str | None = None) -> Colormap:
    """A 256-entry :class:`Colormap` sampled from a matplotlib colormap, index mapped linearly to [vmin, vmax]."""
    try:
        import matplotlib  # lazy: optional dependency
    except ImportError as exc:
        raise ImportError("--cmap needs matplotlib: pip install matplotlib") from exc
    from jukebox.colormap import ColormapEntry

    try:
        mpl_cmap = matplotlib.colormaps[name]
    except KeyError as exc:
        raise ValueError(f"unknown matplotlib colormap {name!r}") from exc
    rgb = np.round(mpl_cmap(np.linspace(0.0, 1.0, 256))[:, :3] * 255).astype(int)
    vals = np.linspace(vmin, vmax, 256)
    step = (vmax - vmin) / 255.0
    entries, seen = [], set()
    for (r, g, b), v in zip(rgb, vals):
        if (r, g, b) in seen:
            continue
        seen.add((r, g, b))
        entries.append(
            ColormapEntry(rgb=(int(r), int(g), int(b)), value=float(v), lo=v - step / 2, hi=v + step / 2, interval=f"[{v:g}]")
        )
    return Colormap(entries=entries, units=units, title=f"matplotlib {name}", name=f"matplotlib:{name}")


def decode_with_matplotlib_cmap(
    source: ImageSource,
    name: str,
    vmin: float,
    vmax: float,
    bbox: BBox = GLOBAL_BBOX,
    mask_gray: bool = False,
    units: str | None = None,
    source_info: dict | None = None,
) -> ValueGrid:
    """Approximate decode of an image drawn with a known matplotlib colormap over [vmin, vmax]."""
    rgba = load_rgba(source)
    if mask_gray:
        rgba = rgba.copy()
        rgba[gray_mask(rgba), 3] = 0
    vg = decode_with_colormap(rgba, matplotlib_colormap(name, vmin, vmax, units), bbox, tolerance=CMAP_TOLERANCE)
    vg.source_info.update({"decode": "cmap", "approximate": True, "cmap": name, "vmin": vmin, "vmax": vmax})
    vg.source_info.update(source_info or {})
    return vg
