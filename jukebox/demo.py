"""Synthetic "seasonal planet" for the offline demo and tests.

Frames are rendered to PNG bytes through a GIBS-style colormap XML built in memory, then decoded by the same
:func:`jukebox.decode.decode_with_colormap` path as real NASA frames. See docs/SONIFICATION_SPEC.md section 2.3.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from jukebox.colormap import Colormap, parse_colormap_xml

DEMO_SIZE = (360, 180)  # width, height -> 1 degree pixels
BASELINE_K = 300.0
LAT_SLOPE = 0.6  # K per degree of latitude
SEASON_K = 12.0  # seasonal amplitude
NOISE_K = 1.0
CMAP_LO, CMAP_HI, CMAP_STEP = 220.0, 320.0, 0.5

#: Crude rectangular continents (minlon, minlat, maxlon, maxlat). Deliberately more land in the north, like
#: Earth: with a symmetric mask the opposite-sign seasons of the two hemispheres would cancel in the global mean.
LAND_BOXES = [
    (-165.0, 15.0, -55.0, 70.0),  # "North America"
    (-10.0, 10.0, 140.0, 72.0),  # "Eurasia"
    (-17.0, -35.0, 50.0, 35.0),  # "Africa"
    (-80.0, -55.0, -35.0, 10.0),  # "South America"
    (113.0, -40.0, 153.0, -12.0),  # "Australia"
]

_ANCHORS = np.array(
    [(110, 0, 220), (20, 60, 255), (0, 190, 255), (0, 200, 90), (250, 230, 0), (255, 120, 0), (220, 20, 0)],
    dtype=np.float64,
)


def demo_colormap_xml() -> str:
    """A GIBS v1.3-style colormap XML: 0.5 K bins from 220 to 320 K, open-ended end bins, one no-data entry."""
    edges = np.arange(CMAP_LO, CMAP_HI + CMAP_STEP / 2, CMAP_STEP)
    intervals = ["(-INF,{:.1f})".format(edges[0])]
    intervals += [f"[{a:.1f},{b:.1f})" for a, b in zip(edges[:-1], edges[1:])]
    intervals += ["[{:.1f},+INF)".format(edges[-1])]
    n = len(intervals)
    pos = np.linspace(0, len(_ANCHORS) - 1, n)
    rgbs = np.array([np.interp(pos, np.arange(len(_ANCHORS)), _ANCHORS[:, c]) for c in range(3)]).T
    rgbs = np.round(rgbs).astype(int)
    seen: set[tuple[int, int, int]] = set()
    for i in range(n):  # keep every colour unique so decoding is exact
        while tuple(rgbs[i]) in seen:
            rgbs[i, 2] = (rgbs[i, 2] + 1) % 256
        seen.add(tuple(rgbs[i]))
    rows = [
        f'      <ColorMapEntry rgb="{r},{g},{b}" transparent="false" value="{iv}" ref="{k + 1}"/>'
        for k, ((r, g, b), iv) in enumerate(zip(rgbs, intervals))
    ]
    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            "<ColorMaps>",
            '  <ColorMap title="No Data">',
            "    <Entries>",
            '      <ColorMapEntry rgb="0,0,0" transparent="true" nodata="true" sourceValue="[0]" ref="0"/>',
            "    </Entries>",
            "  </ColorMap>",
            '  <ColorMap title="Synthetic surface temperature" units="K">',
            "    <Entries>",
            *rows,
            "    </Entries>",
            "  </ColorMap>",
            "</ColorMaps>",
        ]
    )


def demo_colormap() -> Colormap:
    """Parsed demo colormap (same parser as real GIBS colormaps)."""
    return parse_colormap_xml(demo_colormap_xml(), name="demo_synthetic.xml")


def _grid(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    lat = 90.0 - (np.arange(height) + 0.5) * 180.0 / height
    lon = -180.0 + (np.arange(width) + 0.5) * 360.0 / width
    return np.meshgrid(lat, lon, indexing="ij")


def land_mask(width: int = DEMO_SIZE[0], height: int = DEMO_SIZE[1]) -> np.ndarray:
    """Boolean land mask (True = land) built from :data:`LAND_BOXES`."""
    lat, lon = _grid(width, height)
    mask = np.zeros(lat.shape, dtype=bool)
    for minlon, minlat, maxlon, maxlat in LAND_BOXES:
        mask |= (lon >= minlon) & (lon <= maxlon) & (lat >= minlat) & (lat <= maxlat)
    return mask


def planet_values(month: int, rng: np.random.Generator, width: int = DEMO_SIZE[0], height: int = DEMO_SIZE[1]) -> np.ndarray:
    """Toy temperature field in K for calendar ``month`` (1-12); ocean is NaN.

    ``value = baseline - 0.6*|lat| + A*cos(2*pi*(month-7)/12)*sign(lat) + noise``.
    """
    lat, _ = _grid(width, height)
    season = SEASON_K * np.cos(2 * np.pi * (month - 7) / 12.0) * np.sign(lat)
    values = BASELINE_K - LAT_SLOPE * np.abs(lat) + season + rng.normal(0.0, NOISE_K, lat.shape)
    return np.where(land_mask(width, height), values, np.nan)


def month_times(n: int, start: str = "2022-01") -> list[str]:
    """``n`` monthly ISO dates (first of month) starting at ``YYYY-MM``."""
    y, m = int(start[:4]), int(start[5:7])
    out = []
    for i in range(n):
        k = (m - 1) + i
        out.append(f"{y + k // 12:04d}-{k % 12 + 1:02d}-01")
    return out


def demo_frames(n: int = 24, start: str = "2022-01", seed: int = 7) -> tuple[list[tuple[str, bytes]], Colormap]:
    """Render ``n`` synthetic monthly frames as PNG bytes. Returns ``([(time, png_bytes)], colormap)``."""
    cmap = demo_colormap()
    rng = np.random.default_rng(seed)
    frames = []
    for t in month_times(n, start):
        rgba = cmap.color_for_values(planet_values(int(t[5:7]), rng))
        buf = io.BytesIO()
        Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
        frames.append((t, buf.getvalue()))
    return frames, cmap
