"""Spec tests 4 (monotonic mapping), 5 (per-sequence normalization) and 6 (year ticks)."""

import numpy as np
import pytest

from jukebox.decode import GLOBAL_BBOX
from jukebox.features import features_from_grid
from jukebox.legend import build_legend
from jukebox.mapping import MappingConfig, quantize, timeline_score, year_start_frames


def _months(n, start_year=2022):
    return [f"{start_year + i // 12}-{i % 12 + 1:02d}-01" for i in range(n)]


def _uniform_frames(values, times=None):
    times = times or _months(len(values))
    return [features_from_grid(np.full((18, 36), float(v)), GLOBAL_BBOX, time=t, units="K") for v, t in zip(values, times)]


def _melody(score):
    return [e for e in score.events if e.voice == "melody"]


@pytest.mark.parametrize("scale", ["pentatonic", "chromatic", "continuous"])
def test_increasing_means_give_non_decreasing_midi(scale):
    score = timeline_score(_uniform_frames(np.linspace(250, 320, 30)), MappingConfig(scale=scale))
    midi = [e.midi for e in _melody(score)]
    assert all(b >= a for a, b in zip(midi, midi[1:]))
    assert midi[0] < midi[-1]
    assert 48 <= min(midi) and max(midi) <= 84


def test_constant_series_gives_constant_middle_midi():
    score = timeline_score(_uniform_frames([290.0] * 12))
    midi = {e.midi for e in _melody(score)}
    assert midi == {quantize(66, "pentatonic", 48, 84)}


def test_invert_reverses_order():
    values = [250, 270, 290, 310]
    up = [e.midi for e in _melody(timeline_score(_uniform_frames(values)))]
    down = [e.midi for e in _melody(timeline_score(_uniform_frames(values), MappingConfig(invert=True)))]
    assert up == sorted(up) and up[0] < up[-1]
    assert down == sorted(down, reverse=True) and down[0] > down[-1]


def test_normalization_is_per_sequence_not_per_frame():
    score = timeline_score(_uniform_frames([280.0, 300.0]))
    a, b = _melody(score)
    assert a.midi != b.midi
    assert score.meta["value_range"]["lo"] < 280.0 < 300.0 < score.meta["value_range"]["hi"]


def test_pentatonic_has_16_notes_over_three_octaves():
    notes = {quantize(m / 4, "pentatonic", 48, 84) for m in range(48 * 4, 84 * 4 + 1)}
    assert len(notes) == 16 and min(notes) == 48 and max(notes) == 84


def test_ticks_on_january_for_24_months():
    score = timeline_score(_uniform_frames(np.linspace(280, 300, 24)))
    ticks = [e for e in score.events if e.voice == "tick"]
    assert len(ticks) == 2
    assert [t.frame for t in ticks] == [0, 12]
    assert [t.t for t in ticks] == pytest.approx([0.0, 12 * 0.35])


def test_ticks_fall_back_to_every_12th_frame_without_dates():
    assert year_start_frames([None] * 30) == [0, 12, 24]
    assert year_start_frames(["2022-07-01", "2022-12-01", "2023-01-01"]) == [2]


def test_events_sorted_and_drone_spans_sequence():
    score = timeline_score(_uniform_frames(np.linspace(280, 300, 24)))
    ts = [e.t for e in score.events]
    assert ts == sorted(ts)
    (drone,) = [e for e in score.events if e.voice == "drone"]
    assert drone.t == 0.0 and abs(drone.dur - 24 * 0.35) < 1e-9
    assert len(_melody(timeline_score(_uniform_frames([1, 2]), MappingConfig(drone=False)))) == 2
    assert not [e for e in timeline_score(_uniform_frames([1, 2]), MappingConfig(drone=False)).events if e.voice == "drone"]


def test_nan_frame_is_a_rest():
    frames = _uniform_frames([280.0, 290.0, 300.0])
    frames[1] = features_from_grid(np.full((18, 36), np.nan), GLOBAL_BBOX, time=frames[1].time, units="K")
    assert [e.frame for e in _melody(timeline_score(frames))] == [0, 2]


def test_legend_reports_range_in_celsius():
    score = timeline_score(_uniform_frames([273.15 + 10, 273.15 + 30]))
    text = build_legend(score)
    assert "Higher pitch means hotter" in text
    assert "about 9 °C" in text and "about 31 °C" in text
    assert "January 2022 to February 2022" in text


def test_display_value_formats():
    from jukebox.legend import display_value

    assert display_value(285.15, "K") == "12 °C"
    assert display_value(0.04, "K", anomaly=True, decimals=1) == "0.0 °C"
    assert display_value(1.26, "K", anomaly=True, decimals=1) == "+1.3 °C"
    assert display_value(0.5, None, decimals=2) == "0.50"


# --- spec test 7: scan mode ----------------------------------------------------------------------------------

from jukebox.mapping import scan_score  # noqa: E402


def _scan_frame(grid, time="2024-07-01", bbox=GLOBAL_BBOX):
    return features_from_grid(np.asarray(grid, dtype=float), bbox, time=time, units="K")


def test_all_nan_column_is_silent():
    grid = np.tile(np.linspace(250, 320, 36), (18, 1))
    grid[:, 5] = np.nan
    score = scan_score([_scan_frame(grid)])
    melody = _melody(score)
    assert len(melody) == 35
    assert 5 not in {round((e.t / (6.0 / 36))) for e in melody}
    midi = [e.midi for e in melody]
    assert midi == sorted(midi) and midi[0] < midi[-1]  # west->east warming field rises in pitch


def test_scan_loudness_pan_and_ticks():
    grid = np.full((18, 36), 290.0)
    grid[:9, 0] = np.nan  # half-empty first column (weighted valid fraction 0.5)
    grid[:, 1] = np.nan
    score = scan_score([_scan_frame(grid)])
    melody = _melody(score)
    first = melody[0]
    assert first.pan == -1.0 and abs(first.gain - (0.4 + 0.6 * 0.5)) < 1e-3
    assert melody[-1].pan == 1.0
    ticks = sorted(e.t for e in score.events if e.voice == "tick")
    assert ticks == pytest.approx([0.0, 1.5, 3.0, 4.5])  # start, 90W, 0, 90E over a 6 s sweep
    assert score.meta["tick_longitudes"] == [-90.0, 0.0, 90.0]


def test_scan_two_frames_gap_and_double_tick():
    frames = [_scan_frame(np.full((18, 36), 280.0)), _scan_frame(np.full((18, 36), 300.0), time="2024-08-01")]
    score = scan_score(frames)
    second = [e for e in _melody(score) if e.frame == 1]
    assert second[0].t == pytest.approx(6.5)
    gap_ticks = [e.t for e in score.events if e.voice == "tick" and 6.0 < e.t < 6.5]
    assert len(gap_ticks) == 2
    assert _melody(score)[0].midi < second[0].midi  # normalization spans both frames


def test_scan_regional_bbox_ticks_only_inside():
    bangladesh = scan_score([_scan_frame(np.full((18, 36), 290.0), bbox=(88.0, 20.6, 92.7, 26.6))])
    assert bangladesh.meta["tick_longitudes"] == [90.0]
    east_asia = scan_score([_scan_frame(np.full((18, 36), 290.0), bbox=(100.0, 0.0, 140.0, 40.0))])
    assert east_asia.meta["tick_longitudes"] == []
    assert len([e for e in east_asia.events if e.voice == "tick"]) == 1
