"""Score -> stereo audio -> 16-bit PCM WAV (numpy + stdlib ``wave``).

Voices, envelopes and mixing follow docs/SONIFICATION_SPEC.md section 7.
"""

from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np

from jukebox.score import NoteEvent, Score

SAMPLE_RATE = 44100
TAIL_S = 0.5
PEAK = 0.891  # -1 dBFS

ATTACK_S = 0.010
DECAY_S = 0.080
SUSTAIN = 0.7
RELEASE_S = 0.120
TICK_TAU_S = 0.015
DRONE_FADE_S = 0.300


def buffer_length(score: Score, sr: int = SAMPLE_RATE) -> int:
    """Number of samples rendered for ``score``: ``max(t + dur) + 0.5 s`` tail."""
    return int(math.ceil((score.duration + TAIL_S) * sr))


def pan_gains(p: float) -> tuple[float, float]:
    """Equal-power pan for ``p`` in [-1, 1]."""
    p = min(max(float(p), -1.0), 1.0)
    angle = (p + 1.0) * math.pi / 4.0
    return math.cos(angle), math.sin(angle)


def adsr(n_note: int, sr: int) -> np.ndarray:
    """Envelope for a held note of ``n_note`` samples plus its release tail."""
    a, d, r = int(ATTACK_S * sr), int(DECAY_S * sr), int(RELEASE_S * sr)
    n_note = max(n_note, 1)
    env = np.empty(n_note + r)
    t = np.arange(n_note, dtype=np.float64)
    held = np.where(
        t < a,
        t / max(a, 1),
        np.where(t < a + d, 1.0 - (1.0 - SUSTAIN) * (t - a) / max(d, 1), SUSTAIN),
    )
    env[:n_note] = held
    env[n_note:] = held[-1] * np.linspace(1.0, 0.0, r, endpoint=False)
    return env


def render_melody(e: NoteEvent, sr: int) -> np.ndarray:
    """Additive sine (partials 1-3, brighter with ``brightness``) with ADSR; release may overlap the next note."""
    env = adsr(int(round(e.dur * sr)), sr)
    t = np.arange(len(env)) / sr
    b = 0.4 + 1.2 * e.brightness
    amps = (1.0, 0.35 * b, 0.15 * b)
    nyquist = sr / 2
    wave_ = sum(amp * np.sin(2 * np.pi * k * e.freq * t) for k, amp in enumerate(amps, 1) if k * e.freq < nyquist)
    return e.gain * env * wave_


def render_tick(e: NoteEvent, sr: int) -> np.ndarray:
    """Short exponentially decaying sine click."""
    t = np.arange(int(round(e.dur * sr))) / sr
    return e.gain * np.exp(-t / TICK_TAU_S) * np.sin(2 * np.pi * e.freq * t)


def render_drone(e: NoteEvent, sr: int) -> np.ndarray:
    """Quiet sustained sine with linear fade in/out."""
    n = int(round(e.dur * sr))
    t = np.arange(n) / sr
    fade = min(int(DRONE_FADE_S * sr), n // 2)
    env = np.ones(n)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, endpoint=False)
        env[:fade] = ramp
        env[n - fade :] = ramp[::-1]
    return e.gain * env * np.sin(2 * np.pi * e.freq * t)


RENDERERS = {"melody": render_melody, "tick": render_tick, "drone": render_drone}


def render(score: Score, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Mix all events into a ``float64[n, 2]`` stereo buffer, soft-limit and peak-normalize to -1 dBFS."""
    n = buffer_length(score, sr)
    out = np.zeros((n, 2))
    for e in score.events:
        mono = RENDERERS[e.voice](e, sr)
        start = int(round(e.t * sr))
        if start >= n or mono.size == 0:
            continue
        mono = mono[: n - start]
        gl, gr = pan_gains(e.pan)
        out[start : start + len(mono), 0] += gl * mono
        out[start : start + len(mono), 1] += gr * mono
    out = np.tanh(1.2 * out)
    peak = np.abs(out).max() if out.size else 0.0
    if peak > 0:
        out *= PEAK / peak
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def write_wav(path: str | Path, audio: np.ndarray, sr: int = SAMPLE_RATE, mono: bool = False) -> None:
    """Write ``float[n, 2]`` audio in [-1, 1] as 16-bit PCM. ``mono=True`` downmixes (L+R)/2."""
    data = audio.mean(axis=1, keepdims=True) if mono else audio
    pcm = np.clip(np.round(data * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
