"""Spec test 9: end-to-end demo pipeline (offline), plus offline GIBS helpers."""

import json
import wave

import numpy as np
import pytest

from jukebox import gibs
from jukebox.cli import main
from jukebox.demo import demo_colormap, demo_frames
from jukebox.decode import decode_with_colormap
from jukebox.score import read_score, validate_score_dict


def test_demo_writes_wav_score_and_legend(tmp_path, capsys):
    prefix = tmp_path / "demo"
    assert main(["demo", "--out", str(prefix)]) == 0
    wav, js, txt = (tmp_path / f"demo{ext}" for ext in (".wav", ".score.json", ".legend.txt"))

    with wave.open(str(wav)) as w:
        assert w.getnchannels() == 2 and w.getnframes() > 44100

    d = json.loads(js.read_text(encoding="utf-8"))
    assert validate_score_dict(d) == []
    assert d["version"] == 1 and d["meta"]["mode"] == "timeline" and d["meta"]["units"] == "K"
    assert len(d["meta"]["times"]) == 24
    assert sum(e["voice"] == "melody" for e in d["events"]) == 24
    assert sum(e["voice"] == "tick" for e in d["events"]) == 2

    legend = txt.read_text(encoding="utf-8")
    assert legend == d["legend"]
    lo, hi = d["meta"]["value_range"]["lo"], d["meta"]["value_range"]["hi"]
    assert f"about {lo - 273.15:.0f} °C" in legend and f"about {hi - 273.15:.0f} °C" in legend
    assert read_score(js).events[0].voice == "drone"
    assert "frame" in capsys.readouterr().out


def test_demo_frames_decode_exactly():
    cmap = demo_colormap()
    (t, png), *_ = demo_frames(1)[0]
    vg = decode_with_colormap(png, cmap)
    assert t == "2022-01-01"
    land = np.isfinite(vg.values)
    assert 0.1 < land.mean() < 0.6  # ocean is no-data
    assert np.isin(vg.values[land], [e.value for e in cmap.entries]).all()


def test_demo_is_deterministic(tmp_path):
    main(["demo", "--frames", "6", "--out", str(tmp_path / "a")])
    main(["demo", "--frames", "6", "--out", str(tmp_path / "b")])
    assert (tmp_path / "a.wav").read_bytes() == (tmp_path / "b.wav").read_bytes()


def test_wms_url_and_cache_path(tmp_path):
    url = gibs.build_wms_url("LAYER_X", "2024-07-01", (-180, -90, 180, 90), (720, 360))
    assert url.startswith("https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?SERVICE=WMS&VERSION=1.1.1")
    assert "BBOX=-180,-90,180,90" in url and "TIME=2024-07-01" in url and "TRANSPARENT=TRUE" in url
    client = gibs.GibsClient(tmp_path)
    p = client.frame_path("LAYER_X", "2024-07-01", (-180.0, -90.0, 180.0, 90.0), (720, 360))
    assert p.as_posix().endswith("gibs/LAYER_X/m180_m90_180_90_720x360/2024-07-01.png")


def test_time_steps_and_ranges():
    months = gibs.time_steps("2022-01", "2024-12")
    assert len(months) == 36 and months[0] == "2022-01-01" and months[-1] == "2024-12-01"
    assert gibs.time_steps("2024-01-01", "2024-01-10", "day:4") == ["2024-01-01", "2024-01-05", "2024-01-09"]
    assert gibs.time_in_range("2024-07-01", ["2000-03-01/2026-08-01/P1M"])
    assert not gibs.time_in_range("1999-07-01", ["2000-03-01/2026-08-01/P1M"])
    assert gibs.default_size(gibs.REGIONS["global"]) == (720, 360)
    with pytest.raises(gibs.GibsError):
        gibs.time_steps("2024-05", "2024-01")


CAPS = b"""<Capabilities xmlns="http://www.opengis.net/wmts/1.0" xmlns:ows="http://www.opengis.net/ows/1.1"
  xmlns:xlink="http://www.w3.org/1999/xlink"><Contents><Layer>
  <ows:Identifier>LST</ows:Identifier>
  <ows:Metadata xlink:href="https://x/colormaps/v1.0/A.xml" xlink:role="http://earthdata.nasa.gov/gibs/metadata-type/colormap/1.0"/>
  <ows:Metadata xlink:href="https://x/colormaps/v1.3/A.xml" xlink:role="http://earthdata.nasa.gov/gibs/metadata-type/colormap/1.3"/>
  <Dimension><ows:Identifier>Time</ows:Identifier><Value>2000-03-01/2026-08-01/P1M</Value></Dimension>
</Layer></Contents></Capabilities>"""


def test_capabilities_colormap_resolution():
    info = gibs.parse_layer_info(CAPS, "LST")
    assert info.colormap_url == "https://x/colormaps/v1.3/A.xml"
    assert info.time_values == ["2000-03-01/2026-08-01/P1M"]
    assert gibs.parse_layer_info(CAPS, "nope") is None


def test_offline_client_never_touches_network(tmp_path):
    with pytest.raises(gibs.GibsError, match="offline"):
        gibs.GibsClient(tmp_path, offline=True).fetch_frame("L", "2024-01-01", (-180, -90, 180, 90), (8, 4))
