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


def colorbar_colormap(
    rgba: np.ndarray, crop: tuple[int, int, int, int], vmin: float, vmax: float, units: str | None = None
) -> Colormap:
    """Build a colormap by sampling a colour bar inside the image.

    ``crop`` = ``(x0, y0, x1, y1)`` in pixels (x1/y1 exclusive) around the bar only. The long side is the value
    axis: a horizontal bar runs ``vmin`` (left) -> ``vmax`` (right), a vertical bar ``vmin`` (bottom) -> ``vmax``
    (top). Each position takes the median colour across the bar's short side.
    """
    from jukebox.colormap import ColormapEntry

    x0, y0, x1, y1 = crop
    h, w = rgba.shape[:2]
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError(f"--legend-crop {crop} is outside the {w}x{h} image")
    strip = rgba[y0:y1, x0:x1, :3].astype(np.float64)
    horizontal = (x1 - x0) >= (y1 - y0)
    samples = np.median(strip, axis=0) if horizontal else np.median(strip, axis=1)[::-1]
    if len(samples) < 2:
        raise ValueError("--legend-crop must be at least 2 pixels long")
    vals = np.linspace(vmin, vmax, len(samples))
    step = abs(vmax - vmin) / (len(samples) - 1)
    entries, seen = [], set()
    for rgb, v in zip(np.round(samples).astype(int), vals):
        key = tuple(int(c) for c in rgb)
        if key in seen:
            continue
        seen.add(key)
        entries.append(ColormapEntry(rgb=key, value=float(v), lo=v - step / 2, hi=v + step / 2, interval=f"[{v:g}]"))
    return Colormap(entries=entries, units=units, title="colour bar from image", name="legend-crop")


def decode_with_legend_crop(
    source: ImageSource,
    crop: tuple[int, int, int, int],
    vmin: float,
    vmax: float,
    bbox: BBox = GLOBAL_BBOX,
    mask_gray: bool = False,
    units: str | None = None,
    source_info: dict | None = None,
) -> ValueGrid:
    """Approximate decode using the image's own colour bar; the colour bar pixels themselves become no-data."""
    rgba = load_rgba(source)
    cmap = colorbar_colormap(rgba, crop, vmin, vmax, units)
    rgba = rgba.copy()
    x0, y0, x1, y1 = crop
    rgba[y0:y1, x0:x1, 3] = 0
    if mask_gray:
        rgba[gray_mask(rgba), 3] = 0
    vg = decode_with_colormap(rgba, cmap, bbox, tolerance=CMAP_TOLERANCE)
    vg.source_info.update({"decode": "legend-crop", "approximate": True, "crop": list(crop), "vmin": vmin, "vmax": vmax})
    vg.source_info.update(source_info or {})
    return vg
