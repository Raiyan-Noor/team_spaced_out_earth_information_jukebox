"""Spec test 8: synth output shape, length, finiteness and peak level; WAV round-trip."""

import wave

import numpy as np

from jukebox.mapping import tick_event
from jukebox.score import NoteEvent, Score, note_freq
from jukebox.synth import PEAK, buffer_length, pan_gains, render, write_wav


def _score():
    events = [
        NoteEvent(t=0.0, dur=2.0, voice="drone", midi=60, freq=note_freq(60), gain=0.12),
        tick_event(0.0, 0, "start"),
    ]
    for i in range(8):
        m = 48 + 4 * i
        events.append(
            NoteEvent(t=0.25 * i, dur=0.21, voice="melody", midi=m, freq=note_freq(m), gain=0.8,
                      pan=-1 + i / 3.5, brightness=i / 7, frame=i)
        )
    return Score(meta={}, events=sorted(events, key=lambda e: e.t), legend="x")


def test_render_stereo_length_finite_peak():
    score = _score()
    audio = render(score, sr=22050)
    expected = buffer_length(score, sr=22050)
    assert audio.ndim == 2 and audio.shape[1] == 2
    assert abs(audio.shape[0] - int(round((2.0 + 0.5) * 22050))) <= 1
    assert audio.shape[0] == expected
    assert np.isfinite(audio).all()
    assert np.abs(audio).max() <= PEAK + 1e-9
    assert np.abs(audio).max() > 0.5  # normalized, not silent


def test_pan_is_equal_power_and_hard_left_is_left():
    for p in (-1, -0.3, 0, 0.7, 1):
        gl, gr = pan_gains(p)
        assert abs(gl**2 + gr**2 - 1) < 1e-12
    left_only = Score(meta={}, events=[NoteEvent(t=0, dur=0.3, voice="melody", midi=69, freq=440, gain=0.8, pan=-1)])
    audio = render(left_only, sr=8000)
    assert np.abs(audio[:, 1]).max() < 1e-9 < np.abs(audio[:, 0]).max()


def test_empty_score_is_silent_not_nan():
    audio = render(Score(meta={}, events=[]), sr=8000)
    assert audio.shape == (4000, 2) and not audio.any()


def test_write_wav_stereo_and_mono(tmp_path):
    audio = render(_score(), sr=8000)
    write_wav(tmp_path / "s.wav", audio, sr=8000)
    write_wav(tmp_path / "m.wav", audio, sr=8000, mono=True)
    with wave.open(str(tmp_path / "s.wav")) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (2, 2, 8000, len(audio))
    with wave.open(str(tmp_path / "m.wav")) as w:
        assert w.getnchannels() == 1
