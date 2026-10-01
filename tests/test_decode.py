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


def test_legend_crop_horizontal_and_vertical():
    from jukebox.decode import decode_with_legend_crop

    ramp = np.stack([np.linspace(0, 255, 64), np.zeros(64), np.linspace(255, 0, 64)], axis=-1).astype(np.uint8)
    img = np.zeros((40, 64, 4), np.uint8)
    img[..., 3] = 255
    img[:30, :, :3] = ramp[None, :, :]  # "map": a west->east ramp
    img[34:38, :, :3] = ramp[None, :, :]  # horizontal colour bar
    vg = decode_with_legend_crop(img, (0, 34, 64, 38), vmin=0.0, vmax=63.0)
    np.testing.assert_allclose(vg.values[10, [0, 32, 63]], [0.0, 32.0, 63.0], atol=1.01)
    assert np.isnan(vg.values[35, 5])  # the bar itself is not sonified
    assert vg.source_info["decode"] == "legend-crop"

    vimg = np.zeros((64, 40, 4), np.uint8)
    vimg[..., 3] = 255
    vimg[:, 36:40, :3] = ramp[::-1][:, None, :]  # vertical bar, low value at the bottom
    vimg[:, :30, :3] = ramp[5]
    vg2 = decode_with_legend_crop(vimg, (36, 0, 40, 64), vmin=0.0, vmax=63.0)
    assert abs(vg2.values[0, 0] - 5.0) <= 1.01


def test_legend_crop_outside_image_rejected():
    from jukebox.decode import decode_with_legend_crop

    with pytest.raises(ValueError, match="outside"):
        decode_with_legend_crop(np.zeros((10, 10, 4), np.uint8), (0, 0, 20, 2), 0.0, 1.0)
