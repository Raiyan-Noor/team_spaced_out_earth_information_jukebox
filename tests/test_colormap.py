"""Spec test 1 (colormap parsing) plus interval and lookup edge cases."""

import math
import warnings

import numpy as np
import pytest

from jukebox.colormap import (
    UnsupportedColormapError,
    parse_colormap_xml,
    parse_interval,
)


def test_fixture_representative_values_units_and_nodata(small_cmap):
    by_rgb = {e.rgb: e.value for e in small_cmap.entries}
    assert by_rgb == {
        (0, 0, 255): 10.0,  # (-INF,10) -> finite bound
        (0, 128, 255): 15.0,  # [10,20) -> midpoint
        (0, 255, 0): 25.0,
        (255, 255, 0): 30.0,  # [30] -> single value
        (255, 128, 0): 35.0,  # (30,40] -> midpoint
        (255, 0, 0): 40.0,  # (40,+INF] -> finite bound
    }
    assert (64, 64, 64) not in by_rgb  # nodata entry skipped
    assert small_cmap.units == "K"
    assert small_cmap.title == "Test Temperature"
    assert small_cmap.name == "colormap_small.xml"
    assert small_cmap.value_range() == (10.0, 40.0)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("[10,20)", (10.0, 20.0, 15.0)),
        ("(10,20]", (10.0, 20.0, 15.0)),
        ("[10,20]", (10.0, 20.0, 15.0)),
        ("[5]", (5.0, 5.0, 5.0)),
        ("(-INF,10)", (-math.inf, 10.0, 10.0)),
        ("[350,+INF)", (350.0, math.inf, 350.0)),
        ("[350,INF)", (350.0, math.inf, 350.0)),
        ("[-2.5,-1.5)", (-2.5, -1.5, -2.0)),
        ("7", (7.0, 7.0, 7.0)),
    ],
)
def test_parse_interval(text, expected):
    assert parse_interval(text) == expected


def test_value_range_ignores_catch_all_bin_midpoints():
    xml = """<ColorMaps><ColorMap units="K"><Entries>
      <ColorMapEntry rgb="1,0,0" value="[0.02,200.00)"/>
      <ColorMapEntry rgb="2,0,0" value="[200.00,200.60)"/>
      <ColorMapEntry rgb="3,0,0" value="[349.40,350.00)"/>
      <ColorMapEntry rgb="4,0,0" value="[350.02,652.00)"/>
    </Entries></ColorMap></ColorMaps>"""
    assert parse_colormap_xml(xml).value_range() == (200.0, 350.02)


def test_value_preferred_over_source_value(small_cmap):
    # The fixture's sourceValue ranges are 10x the scaled values; value must win.
    assert max(e.value for e in small_cmap.entries) == 40.0


def test_duplicate_rgb_keeps_first_and_warns():
    xml = """<ColorMaps><ColorMap units="m"><Entries>
      <ColorMapEntry rgb="1,2,3" value="[0,2)"/>
      <ColorMapEntry rgb="1,2,3" value="[2,4)"/>
      <ColorMapEntry rgb="9,9,9" value="[4,6)"/>
    </Entries></ColorMap></ColorMaps>"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cm = parse_colormap_xml(xml)
    assert len(caught) == 1
    assert [e.value for e in cm.entries] == [1.0, 5.0]


def test_entries_without_entries_wrapper():
    xml = """<ColorMaps><ColorMap units="mm"><ColorMapEntry rgb="1,1,1" value="[0,10)"/></ColorMap></ColorMaps>"""
    assert parse_colormap_xml(xml).entries[0].value == 5.0


def test_classification_colormap_rejected():
    xml = """<ColorMaps><ColorMap title="Land cover"><Entries>
      <ColorMapEntry rgb="0,100,0" label="Forest"/>
      <ColorMapEntry rgb="200,200,0" label="Cropland"/>
    </Entries></ColorMap></ColorMaps>"""
    with pytest.raises(UnsupportedColormapError, match="classification"):
        parse_colormap_xml(xml)


def test_lookup_exact_and_nearest(small_cmap):
    rgb = np.array([[[0, 255, 0], [2, 2, 250], [128, 0, 128]]], dtype=np.uint8)
    vals = small_cmap.lookup(rgb)
    assert vals.dtype == np.float32
    assert vals[0, 0] == 25.0
    assert vals[0, 1] == 10.0  # off-palette but within tolerance of (0,0,255)
    assert np.isnan(vals[0, 2])  # far from every colormap colour
