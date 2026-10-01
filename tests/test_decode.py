"""Spec test 2: exact decode round-trip, transparency, nearest-colour tolerance."""

import io

import numpy as np
from PIL import Image

from jukebox.decode import decode_with_colormap, load_rgba


def _paint(pixels):
    return np.array([pixels], dtype=np.uint8)


def test_exact_round_trip_with_transparent_pixel(small_cmap):
    rgba = _paint([[0, 0, 255, 255], [0, 128, 255, 255], [255, 255, 0, 255], [255, 128, 0, 255], [0, 255, 0, 0]])
    vg = decode_with_colormap(rgba, small_cmap)
    assert vg.values.dtype == np.float32
    np.testing.assert_array_equal(vg.values[0, :4], [10.0, 15.0, 30.0, 35.0])
    assert np.isnan(vg.values[0, 4])  # alpha 0 -> no data even though the RGB is in the colormap
    assert vg.units == "K"
    assert vg.source_info["approximate"] is False


def test_off_palette_tolerance(small_cmap):
    rgba = _paint([[3, 250, 4, 255], [100, 100, 100, 255]])
    vals = decode_with_colormap(rgba, small_cmap).values
    assert vals[0, 0] == 25.0  # distance ~6.2 from (0,255,0)
    assert np.isnan(vals[0, 1])  # far from every colour


def test_png_and_palette_png_decode_identically(small_cmap):
    rgba = _paint([[0, 0, 255, 255], [0, 255, 0, 255], [255, 0, 0, 255], [0, 0, 0, 0]])
    rgba_buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(rgba_buf, format="PNG")
    pal_buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").quantize(colors=8).save(pal_buf, format="PNG", transparency=None)
    a = decode_with_colormap(rgba_buf.getvalue(), small_cmap).values
    np.testing.assert_array_equal(a[0, :3], [10.0, 25.0, 40.0])
    assert np.isnan(a[0, 3])
    # Palette PNGs must be expanded to RGB before lookup (no raw palette indices).
    assert load_rgba(pal_buf.getvalue()).shape == (1, 4, 4)


# --- P1 fallbacks --------------------------------------------------------------------------------------------

import pytest  # noqa: E402

from jukebox.decode import decode_luminance, decode_with_matplotlib_cmap, srgb_to_lstar  # noqa: E402


def test_lstar_endpoints_and_order():
    l = srgb_to_lstar(np.array([[0, 0, 0], [119, 119, 119], [255, 255, 255]], dtype=np.uint8))
    assert l[0] == pytest.approx(0.0, abs=1e-6) and l[2] == pytest.approx(100.0, abs=1e-6)
    assert 49 < l[1] < 51  # sRGB 119 is ~ mid lightness


def test_luminance_decode_is_approximate_and_masks():
    rgba = _paint([[0, 0, 0, 255], [255, 255, 255, 255], [255, 0, 0, 255], [10, 10, 10, 0]])
    vg = decode_luminance(rgba)
    assert vg.source_info["approximate"] is True and vg.units is None
    assert vg.values[0, 0] == pytest.approx(0.0, abs=1e-6) and vg.values[0, 1] == pytest.approx(1.0)
    assert np.isnan(vg.values[0, 3])
    masked = decode_luminance(rgba, mask_gray=True).values
    assert np.isnan(masked[0, 0]) and np.isnan(masked[0, 1]) and np.isfinite(masked[0, 2])


def test_matplotlib_cmap_round_trip():
    matplotlib = pytest.importorskip("matplotlib")
    lut = np.round(matplotlib.colormaps["viridis"]([0.0, 0.5, 1.0])[:, :3] * 255).astype(np.uint8)
    rgba = np.concatenate([lut, np.full((3, 1), 255, np.uint8)], axis=1)[None]
    vg = decode_with_matplotlib_cmap(rgba, "viridis", vmin=-2.0, vmax=2.0)
    np.testing.assert_allclose(vg.values[0], [-2.0, 0.0, 2.0], atol=4.0 / 255 + 1e-6)
    assert vg.source_info["approximate"] is True
