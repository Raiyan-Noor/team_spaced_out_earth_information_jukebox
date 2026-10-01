"""Spec test 3: cos(latitude) weighting, plus coarse-grid and extreme_frac behaviour."""

import numpy as np

from jukebox.decode import GLOBAL_BBOX, ValueGrid
from jukebox.features import apply_extreme_fracs, coarse_grid, frame_features


def _vg(values):
    return ValueGrid(values=np.asarray(values, dtype=np.float32), units="K", bbox=GLOBAL_BBOX)


def test_uniform_field_mean_is_the_constant():
    f = frame_features(_vg(np.full((180, 360), 287.5)))
    assert abs(f.mean - 287.5) < 1e-9
    assert f.std < 1e-9
    assert abs(f.valid_frac - 1.0) < 1e-12


def test_polar_high_field_weighted_mean_below_unweighted():
    lat = np.linspace(89.75, -89.75, 180)
    field = np.where(np.abs(lat) > 60, 100.0, 0.0)[:, None] * np.ones((1, 360))
    f = frame_features(_vg(field))
    assert f.mean < field.mean()


def test_coarse_grid_uneven_split_and_nan_blocks():
    v = np.arange(35, dtype=float).reshape(5, 7)
    v[:3, :4] = np.nan
    g = coarse_grid(v, 2, 2)  # rows split 3+2, cols split 4+3
    assert np.isnan(g[0, 0])
    assert g[0, 1] == np.mean(v[:3, 4:])
    assert g[1, 0] == np.mean(v[3:, :4])


def test_column_stats_and_hotspot():
    v = np.full((180, 360), np.nan)
    v[:, 180:190] = 300.0  # one 10-degree column of data at lon 0..10
    v[80:100, 185] = 350.0
    f = frame_features(_vg(v))
    assert np.isnan(f.col_mean[0]) and f.col_valid[0] == 0.0
    assert f.col_valid[18] == 1.0
    assert f.hotspot_lon == 5.0


def test_extreme_frac_uses_sequence_threshold():
    # Ten uniform frames with values 1..10: the sequence 90th percentile is 9.1, so only the last frame is extreme.
    originals = [frame_features(_vg(np.full((18, 36), float(k)))) for k in range(1, 11)]
    frames, thr = apply_extreme_fracs(originals)
    assert abs(thr - 9.1) < 1e-9
    assert [f.extreme_frac for f in frames] == [0.0] * 9 + [1.0]
    assert originals[0].extreme_frac is None  # sequence pass returns copies
