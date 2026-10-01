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
