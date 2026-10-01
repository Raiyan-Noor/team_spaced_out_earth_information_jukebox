"""Per-frame features on a coarse, cos(latitude)-weighted grid.

Everything in :func:`frame_features` depends on one frame only. The single sequence-level quantity,
``extreme_frac`` (needs the 90th percentile over the whole sequence), is filled in by
:func:`apply_extreme_fracs`. See docs/SONIFICATION_SPEC.md section 4.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from jukebox.decode import BBox, ValueGrid

DEFAULT_GRID = (18, 36)  # lat rows x lon cols -> 10 degree cells for a global bbox


@dataclass
class Frame:
    """Features of one frame. Statistics are NaN when the frame has no valid cells."""

    time: str | None
    grid: np.ndarray  # float64[rows, cols] block means, NaN = no data
    lat_c: np.ndarray  # cell-centre latitudes, north to south
    lon_c: np.ndarray  # cell-centre longitudes, west to east
    bbox: BBox
    units: str | None
    mean: float
    std: float
    valid_frac: float
    col_mean: np.ndarray  # float64[cols]
    col_valid: np.ndarray  # float64[cols], weighted fraction of valid cells per column
    hotspot_lon: float
    extreme_frac: float | None = None  # set by apply_extreme_fracs


def _edges(n: int, parts: int) -> np.ndarray:
    """Start indices of ``np.array_split(range(n), parts)``."""
    sizes = [len(s) for s in np.array_split(np.arange(n), parts)]
    return np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.intp)


def coarse_grid(values: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Block-average ``values`` onto ``rows x cols`` cells, ignoring NaN (all-NaN block -> NaN).

    Block sizes follow ``np.array_split`` so the image size need not divide evenly.
    """
    h, w = values.shape
    if rows > h or cols > w:
        raise ValueError(f"grid {rows}x{cols} is finer than the image ({h}x{w})")
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0).astype(np.float64)
    re, ce = _edges(h, rows), _edges(w, cols)
    sums = np.add.reduceat(np.add.reduceat(filled, re, axis=0), ce, axis=1)
    counts = np.add.reduceat(np.add.reduceat(finite.astype(np.int64), re, axis=0), ce, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def cell_centres(bbox: BBox, rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre latitudes (north -> south) and longitudes (west -> east) for an equirectangular bbox."""
    minlon, minlat, maxlon, maxlat = bbox
    dlat = (maxlat - minlat) / rows
    dlon = (maxlon - minlon) / cols
    lat_c = maxlat - dlat * (np.arange(rows) + 0.5)
    lon_c = minlon + dlon * (np.arange(cols) + 0.5)
    return lat_c, lon_c


def features_from_grid(
    grid: np.ndarray, bbox: BBox, time: str | None = None, units: str | None = None
) -> Frame:
    """Compute the per-frame features of a coarse grid (weights ``cos(lat)``, zero where the cell is NaN)."""
    rows, cols = grid.shape
    lat_c, lon_c = cell_centres(bbox, rows, cols)
    w_all = np.broadcast_to(np.cos(np.radians(lat_c))[:, None], grid.shape).clip(min=0.0)
    valid = np.isfinite(grid)
    w = np.where(valid, w_all, 0.0)
    v = np.where(valid, grid, 0.0)

    wsum = w.sum()
    if wsum > 0:
        mean = float((w * v).sum() / wsum)
        std = float(np.sqrt((w * (v - mean) ** 2 * valid).sum() / wsum))
        flat = np.where(valid, grid, -np.inf)
        hotspot_lon = float(lon_c[np.unravel_index(np.argmax(flat), grid.shape)[1]])
    else:
        mean = std = hotspot_lon = float("nan")
    valid_frac = float(wsum / w_all.sum()) if w_all.sum() > 0 else 0.0

    col_w = w.sum(axis=0)
    col_w_all = w_all.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        col_mean = np.where(col_w > 0, (w * v).sum(axis=0) / np.where(col_w > 0, col_w, 1.0), np.nan)
        col_valid = np.where(col_w_all > 0, col_w / np.where(col_w_all > 0, col_w_all, 1.0), 0.0)

    return Frame(
        time=time,
        grid=grid,
        lat_c=lat_c,
        lon_c=lon_c,
        bbox=bbox,
        units=units,
        mean=mean,
        std=std,
        valid_frac=valid_frac,
        col_mean=col_mean,
        col_valid=col_valid,
        hotspot_lon=hotspot_lon,
    )


def frame_features(vg: ValueGrid, time: str | None = None, grid: tuple[int, int] = DEFAULT_GRID) -> Frame:
    """Coarse-grid a decoded frame and compute its features."""
    return features_from_grid(coarse_grid(vg.values, *grid), vg.bbox, time=time, units=vg.units)


def extreme_threshold(frames: list[Frame], q: float = 90.0) -> float:
    """The ``q``-th percentile of all valid cells across the whole sequence (NaN if there are none)."""
    cells = np.concatenate([f.grid[np.isfinite(f.grid)] for f in frames]) if frames else np.array([])
    return float(np.percentile(cells, q)) if cells.size else float("nan")


def apply_extreme_fracs(frames: list[Frame], q: float = 90.0) -> tuple[list[Frame], float]:
    """Sequence pass: return copies of ``frames`` with ``extreme_frac`` set, plus the threshold used.

    ``extreme_frac`` = weighted fraction of a frame's valid cells strictly above the sequence threshold.
    """
    thr = extreme_threshold(frames, q)
    out = []
    for f in frames:
        valid = np.isfinite(f.grid)
        w = np.where(valid, np.cos(np.radians(f.lat_c))[:, None].clip(min=0.0), 0.0)
        wsum = w.sum()
        frac = float((w * (np.where(valid, f.grid, -np.inf) > thr)).sum() / wsum) if wsum > 0 else float("nan")
        out.append(replace(f, extreme_frac=frac))
    return out, thr


def monthly_anomalies(frames: list[Frame]) -> list[Frame]:
    """Sequence pass for anomaly mode: subtract each calendar month's climatology, cell by cell.

    The climatology of a calendar month is the NaN-aware mean of that month's coarse grids across the whole
    sequence, so the result shows departures from "normal for this month" instead of the seasonal cycle.
    Requires dates (``YYYY-MM...``) on every frame; features are recomputed on the anomaly grids.
    """
    months = []
    for f in frames:
        if not f.time or len(f.time) < 7:
            raise ValueError("anomaly mode needs a date on every frame")
        months.append(int(f.time[5:7]))
    clim: dict[int, np.ndarray] = {}
    for m in set(months):
        stack = np.stack([f.grid for f, mm in zip(frames, months) if mm == m])
        sums = np.nansum(stack, axis=0)
        counts = np.isfinite(stack).sum(axis=0)
        clim[m] = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return [features_from_grid(f.grid - clim[m], f.bbox, time=f.time, units=f.units) for f, m in zip(frames, months)]
