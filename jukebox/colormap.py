"""GIBS colormap v1.3 XML parser and RGB -> data value lookup.

GIBS publishes one colormap XML per layer. Each ``ColorMapEntry`` maps an RGB triple to a value interval
such as ``[200.00,200.60)``. Inverting that table recovers real data values (e.g. Kelvin) from a rendered PNG.
See docs/SONIFICATION_SPEC.md section 3.
"""

from __future__ import annotations

import math
import re
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: Max Euclidean RGB distance accepted by the nearest-colour fallback.
DEFAULT_TOLERANCE = 12.0


class ColormapError(ValueError):
    """The colormap XML could not be used (malformed, empty, or unsupported)."""


class UnsupportedColormapError(ColormapError):
    """Classification colormaps (labels, no numeric values) are not supported in M1."""


@dataclass(frozen=True)
class ColormapEntry:
    """One colour of a colormap and the data value it stands for."""

    rgb: tuple[int, int, int]
    value: float  # representative value used for sonification
    lo: float  # interval lower bound (may be -inf)
    hi: float  # interval upper bound (may be +inf)
    interval: str  # the original interval text, e.g. "[200.00,200.60)"


@dataclass
class Colormap:
    """A parsed colormap: data entries (no-data entries removed) plus lookup tables."""

    entries: list[ColormapEntry]
    units: str | None = None
    title: str | None = None
    name: str | None = None
    _keys: np.ndarray = field(init=False, repr=False)
    _values: np.ndarray = field(init=False, repr=False)
    _rgbs: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.entries:
            raise ColormapError("colormap has no usable data entries")
        rgbs = np.array([e.rgb for e in self.entries], dtype=np.int64)
        keys = pack_rgb(rgbs)
        order = np.argsort(keys, kind="stable")
        self._keys = keys[order]
        self._values = np.array([e.value for e in self.entries], dtype=np.float64)[order]
        self._rgbs = rgbs[order]

    def value_range(self) -> tuple[float, float]:
        """Finite min/max of the representative values (used by ``--range colormap``)."""
        finite = self._values[np.isfinite(self._values)]
        return float(finite.min()), float(finite.max())

    def lookup(self, rgb: np.ndarray, tolerance: float = DEFAULT_TOLERANCE) -> np.ndarray:
        """Map an ``(..., 3)`` uint8 RGB array to float32 values.

        Exact matches come first. Colours with no exact match take the value of the nearest colormap colour
        (Euclidean distance in RGB) if it is within ``tolerance``; otherwise they become NaN.
        """
        rgb = np.asarray(rgb)
        shape = rgb.shape[:-1]
        flat_keys = pack_rgb(rgb.reshape(-1, 3).astype(np.int64))
        out = np.full(flat_keys.shape, np.nan, dtype=np.float32)

        idx = np.searchsorted(self._keys, flat_keys)
        idx_clipped = np.minimum(idx, len(self._keys) - 1)
        exact = self._keys[idx_clipped] == flat_keys
        out[exact] = self._values[idx_clipped[exact]]

        if not exact.all():
            missing_keys, inverse = np.unique(flat_keys[~exact], return_inverse=True)
            nearest = self._nearest_values(unpack_rgb(missing_keys), tolerance)
            out[~exact] = nearest[inverse]
        return out.reshape(shape)

    def _nearest_values(self, rgbs: np.ndarray, tolerance: float, chunk: int = 4096) -> np.ndarray:
        """Nearest-colour values for a small set of unique off-palette colours."""
        result = np.full(len(rgbs), np.nan, dtype=np.float32)
        palette = self._rgbs.astype(np.float64)
        for start in range(0, len(rgbs), chunk):
            block = rgbs[start : start + chunk].astype(np.float64)
            d2 = ((block[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
            best = d2.argmin(axis=1)
            ok = d2[np.arange(len(block)), best] <= tolerance**2
            vals = self._values[best].astype(np.float32)
            vals[~ok] = np.nan
            result[start : start + chunk] = vals
        return result

    def color_for_values(self, values: np.ndarray) -> np.ndarray:
        """Inverse direction (used by the synthetic demo): value -> RGB of the entry whose interval holds it.

        Returns ``(..., 4)`` uint8 RGBA; NaN values become fully transparent.
        """
        values = np.asarray(values, dtype=np.float64)
        ordered = sorted(self.entries, key=lambda e: (e.lo, e.hi))
        lows = np.array([e.lo for e in ordered])
        rgbs = np.array([e.rgb for e in ordered], dtype=np.uint8)
        idx = np.clip(np.searchsorted(lows, np.nan_to_num(values, nan=0.0), side="right") - 1, 0, len(ordered) - 1)
        rgba = np.zeros(values.shape + (4,), dtype=np.uint8)
        rgba[..., :3] = rgbs[idx]
        rgba[..., 3] = np.where(np.isfinite(values), 255, 0)
        return rgba


def pack_rgb(rgb: np.ndarray) -> np.ndarray:
    """Pack an ``(N, 3)`` integer RGB array into one int64 key per colour."""
    rgb = np.asarray(rgb, dtype=np.int64)
    return (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]


def unpack_rgb(keys: np.ndarray) -> np.ndarray:
    """Inverse of :func:`pack_rgb`."""
    keys = np.asarray(keys, dtype=np.int64)
    return np.stack([(keys >> 16) & 255, (keys >> 8) & 255, keys & 255], axis=-1)


_NUM = r"[+-]?(?:INF|inf|Inf|\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)"
_INTERVAL_RE = re.compile(rf"^\s*([\[(])\s*({_NUM})\s*(?:,\s*({_NUM})\s*)?([\])])\s*$")


def _parse_number(text: str) -> float:
    t = text.strip().upper()
    if t in ("INF", "+INF"):
        return math.inf
    if t == "-INF":
        return -math.inf
    return float(t)


def parse_interval(text: str) -> tuple[float, float, float]:
    """Parse a GIBS value interval and return ``(lo, hi, representative)``.

    Accepts ``[a,b)``, ``(a,b]``, ``[a,b]``, ``(a,b)``, ``[a]`` and a bare number, with ``-INF``/``+INF``/``INF``.
    The representative is the midpoint if both bounds are finite, otherwise the finite bound; ``[a]`` -> ``a``.
    """
    m = _INTERVAL_RE.match(text)
    if m is None:
        try:
            v = _parse_number(text)
        except ValueError as exc:
            raise ColormapError(f"cannot parse value interval {text!r}") from exc
        return v, v, v
    a = _parse_number(m.group(2))
    b = _parse_number(m.group(3)) if m.group(3) is not None else a
    lo, hi = min(a, b), max(a, b)
    if math.isfinite(lo) and math.isfinite(hi):
        rep = (lo + hi) / 2.0
    elif math.isfinite(lo):
        rep = lo
    elif math.isfinite(hi):
        rep = hi
    else:
        raise ColormapError(f"interval {text!r} has no finite bound")
    return lo, hi, rep


def _is_true(attr: str | None) -> bool:
    return attr is not None and attr.strip().lower() == "true"


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_colormap_xml(xml: str | bytes, name: str | None = None) -> Colormap:
    """Parse GIBS colormap v1.3 XML text into a :class:`Colormap`.

    No-data and transparent entries are skipped. Units and title come from the first ``<ColorMap>`` that holds
    data entries. Raises :class:`UnsupportedColormapError` for classification colormaps.
    """
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise ColormapError(f"colormap XML is malformed: {exc}") from exc

    colormaps = [root] if _local(root.tag) == "ColorMap" else [c for c in root if _local(c.tag) == "ColorMap"]
    entries: list[ColormapEntry] = []
    seen: set[tuple[int, int, int]] = set()
    duplicates = 0
    units: str | None = None
    title: str | None = None
    unlabelled = 0

    for cm in colormaps:
        container = next((c for c in cm if _local(c.tag) == "Entries"), cm)
        cm_entries = [e for e in container if _local(e.tag) == "ColorMapEntry"]
        added_here = False
        for e in cm_entries:
            if _is_true(e.get("nodata")) or _is_true(e.get("transparent")):
                continue
            raw = e.get("value") or e.get("sourceValue")
            if raw is None:
                unlabelled += 1
                continue
            try:
                lo, hi, rep = parse_interval(raw)
            except ColormapError:
                unlabelled += 1
                continue
            try:
                rgb = tuple(int(x) for x in e.get("rgb", "").split(","))
            except ValueError as exc:
                raise ColormapError(f"bad rgb attribute {e.get('rgb')!r}") from exc
            if len(rgb) != 3:
                raise ColormapError(f"bad rgb attribute {e.get('rgb')!r}")
            if rgb in seen:
                duplicates += 1
                continue
            seen.add(rgb)  # type: ignore[arg-type]
            entries.append(ColormapEntry(rgb=rgb, value=rep, lo=lo, hi=hi, interval=raw))  # type: ignore[arg-type]
            added_here = True
        if added_here and units is None:
            units = cm.get("units")
            title = cm.get("title")

    if not entries:
        if unlabelled:
            raise UnsupportedColormapError(
                "this looks like a classification colormap (categories, no numeric values); "
                "classification layers are not supported in M1"
            )
        raise ColormapError("colormap has no usable data entries")
    if duplicates:
        warnings.warn(f"colormap {name or ''}: {duplicates} entries reuse an earlier RGB; kept the first", stacklevel=2)
    return Colormap(entries=entries, units=units, title=title, name=name)


def load_colormap(path: str | Path) -> Colormap:
    """Read and parse a colormap XML file; the file name becomes the colormap name."""
    p = Path(path)
    return parse_colormap_xml(p.read_bytes(), name=p.name)
