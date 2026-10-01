"""Features -> Score. Timeline mode (one note per frame) and scan mode (sweep one frame west to east).

Per-frame mapping functions are pure; the only sequence-level step is normalization (value range, extreme_frac
threshold, brightness range). See docs/SONIFICATION_SPEC.md sections 5 and 6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from jukebox.features import Frame, apply_extreme_fracs
from jukebox.score import NoteEvent, Score, note_freq

SCALES: dict[str, tuple[int, ...] | None] = {
    "pentatonic": (0, 2, 4, 7, 9),  # major pentatonic
    "chromatic": tuple(range(12)),
    "continuous": None,  # no quantization
}

TICK_FREQ = 1500.0
TICK_DUR = 0.06
TICK_GAIN = 0.35
MELODY_GAIN = 0.8
DRONE_GAIN = 0.12


@dataclass
class MappingConfig:
    """User-facing mapping options (CLI flags map 1:1 onto these)."""

    lo_midi: int = 48  # C3
    hi_midi: int = 84  # C6
    scale: str = "pentatonic"
    invert: bool = False
    tempo: float = 0.35  # seconds per frame in timeline mode
    drone: bool = True
    range_mode: str = "sequence"  # "sequence" | "colormap"
    note_frac: float = 0.85  # melody note length as a fraction of the beat


# --- normalization -----------------------------------------------------------------------------------------


def sequence_range(values: Sequence[float] | np.ndarray, pad: float = 0.05) -> tuple[float, float]:
    """Min/max of the finite ``values``, padded by ``pad`` x span on each side. NaN if there are none."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    lo, hi = float(v.min()), float(v.max())
    span = hi - lo
    return lo - pad * span, hi + pad * span


def normalize01(values: Sequence[float], constant: float = 0.5) -> list[float]:
    """Min/max-normalize to [0, 1]; a constant (or empty) series maps to ``constant``; NaN -> ``constant``."""
    v = np.asarray(values, dtype=np.float64)
    finite = v[np.isfinite(v)]
    if finite.size == 0 or finite.max() - finite.min() < 1e-12:
        return [constant] * len(v)
    out = (v - finite.min()) / (finite.max() - finite.min())
    return [float(x) if math.isfinite(x) else constant for x in out]


# --- pitch ---------------------------------------------------------------------------------------------------


def quantize(midi: float, scale: str, lo_midi: int, hi_midi: int) -> float:
    """Snap ``midi`` to the nearest note of ``scale`` (rooted at ``lo_midi``) within [lo_midi, hi_midi].

    Ties go to the lower note. ``continuous`` returns ``midi`` unchanged (rounded to 3 decimals).
    """
    if scale not in SCALES:
        raise ValueError(f"unknown scale {scale!r}; choose from {', '.join(SCALES)}")
    degrees = SCALES[scale]
    if degrees is None:
        return round(float(midi), 3)
    notes = [n for n in range(lo_midi, hi_midi + 1) if (n - lo_midi) % 12 in degrees]
    return float(min(notes, key=lambda n: (abs(n - midi), n)))


def value_to_midi(value: float, lo: float, hi: float, cfg: MappingConfig) -> float:
    """Map a data value to a (quantized) MIDI note. Degenerate range -> the middle note."""
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi - lo < 1e-9:
        v_norm = 0.5
    else:
        v_norm = min(max((value - lo) / (hi - lo), 0.0), 1.0)
    if cfg.invert:
        v_norm = 1.0 - v_norm
    return quantize(cfg.lo_midi + v_norm * (cfg.hi_midi - cfg.lo_midi), cfg.scale, cfg.lo_midi, cfg.hi_midi)


def lon_to_pan(lon: float, bbox: Sequence[float]) -> float:
    """Pan in [-1, 1] from longitude across the bbox (global: ``lon / 180``). NaN -> centre."""
    minlon, _, maxlon, _ = bbox
    if not math.isfinite(lon) or maxlon <= minlon:
        return 0.0
    return float(min(max(2.0 * (lon - minlon) / (maxlon - minlon) - 1.0, -1.0), 1.0))


# --- timeline mode -------------------------------------------------------------------------------------------


def _month(time: str | None) -> int | None:
    if not time or len(time) < 7:
        return None
    try:
        return int(time[5:7])
    except ValueError:
        return None


def year_start_frames(times: Sequence[str | None]) -> list[int]:
    """Frames that start a year: every January, or every 12th frame from 0 if any date is unknown."""
    months = [_month(t) for t in times]
    if any(m is None for m in months):
        return list(range(0, len(times), 12))
    return [i for i, m in enumerate(months) if m == 1]


def frame_label(time: str | None, index: int, monthly: bool) -> str:
    """Short human label for a frame (``2022-01`` for monthly data)."""
    if not time:
        return f"frame {index + 1}"
    return time[:7] if monthly else time[:10]


def _is_monthly(times: Sequence[str | None]) -> bool:
    return all(t is not None and len(t) >= 10 and t[8:10] == "01" for t in times) and len(times) > 0


def resolve_value_range(
    values: Sequence[float], cfg: MappingConfig, colormap_range: tuple[float, float] | None
) -> tuple[float, float]:
    """``--range sequence`` (default) or ``--range colormap``."""
    if cfg.range_mode == "colormap":
        if colormap_range is None:
            raise ValueError("--range colormap needs a colormap with a numeric range")
        return float(colormap_range[0]), float(colormap_range[1])
    if cfg.range_mode != "sequence":
        raise ValueError(f"unknown range mode {cfg.range_mode!r}")
    return sequence_range(values)


def _base_meta(mode: str, frames: list[Frame], cfg: MappingConfig, meta: dict[str, Any] | None) -> dict[str, Any]:
    m: dict[str, Any] = {
        "mode": mode,
        "source": None,
        "layer": None,
        "colormap": None,
        "units": frames[0].units if frames else None,
        "bbox": list(frames[0].bbox) if frames else None,
        "times": [f.time for f in frames],
        "value_range": None,
        "scale": cfg.scale,
        "midi_range": [cfg.lo_midi, cfg.hi_midi],
        "tempo_s_per_frame": cfg.tempo,
        "frames": [],
        "invert": cfg.invert,
        "range_mode": cfg.range_mode,
    }
    m.update(meta or {})
    return m


def tick_event(t: float, frame: int | None, label: str) -> NoteEvent:
    """The structure-marker earcon (1500 Hz click, centred)."""
    return NoteEvent(
        t=round(t, 6),
        dur=TICK_DUR,
        voice="tick",
        midi=round(69 + 12 * math.log2(TICK_FREQ / 440.0), 3),
        freq=TICK_FREQ,
        gain=TICK_GAIN,
        pan=0.0,
        brightness=0.0,
        frame=frame,
        label=label,
    )


def _sorted(events: list[NoteEvent]) -> list[NoteEvent]:
    order = {"drone": 0, "tick": 1, "melody": 2}
    return sorted(events, key=lambda e: (e.t, order[e.voice]))


def timeline_score(
    frames: list[Frame],
    cfg: MappingConfig | None = None,
    meta: dict[str, Any] | None = None,
    colormap_range: tuple[float, float] | None = None,
) -> Score:
    """Timeline mode: one melody note per frame (pitch = weighted mean), year ticks and a reference drone.

    ``meta`` adds descriptive keys (source, layer, colormap, dataset, region, ...) to the score metadata.
    """
    cfg = cfg or MappingConfig()
    if not frames:
        raise ValueError("timeline mode needs at least one frame")
    frames, thr = apply_extreme_fracs(frames)
    means = [f.mean for f in frames]
    lo, hi = resolve_value_range(means, cfg, colormap_range)
    brightness = normalize01([f.extreme_frac for f in frames])  # type: ignore[misc]
    times = [f.time for f in frames]
    monthly = _is_monthly(times)
    beat = cfg.tempo

    events: list[NoteEvent] = []
    frame_meta = []
    for i, f in enumerate(frames):
        label = frame_label(f.time, i, monthly)
        midi = value_to_midi(f.mean, lo, hi, cfg) if math.isfinite(f.mean) else None
        pan = lon_to_pan(f.hotspot_lon, f.bbox)
        if midi is not None:
            events.append(
                NoteEvent(
                    t=round(i * beat, 6),
                    dur=round(cfg.note_frac * beat, 6),
                    voice="melody",
                    midi=midi,
                    freq=round(note_freq(midi), 3),
                    gain=MELODY_GAIN,
                    pan=round(pan, 4),
                    brightness=round(brightness[i], 4),
                    frame=i,
                    label=label,
                )
            )
        frame_meta.append(
            {
                "time": f.time,
                "mean": f.mean,
                "std": f.std,
                "extreme_frac": f.extreme_frac,
                "hotspot_lon": f.hotspot_lon,
                "valid_frac": f.valid_frac,
                "midi": midi,
                "pan": round(pan, 4),
            }
        )

    for i in year_start_frames(times):
        events.append(tick_event(i * beat, i, f"year start {frame_label(times[i], i, monthly)}"))

    total = len(frames) * beat
    ref_vals = [m for m in means[:12] if math.isfinite(m)]
    drone_value = float(np.mean(ref_vals)) if ref_vals else float("nan")
    if cfg.drone and math.isfinite(drone_value):
        dm = value_to_midi(drone_value, lo, hi, cfg)
        events.append(
            NoteEvent(
                t=0.0,
                dur=round(total, 6),
                voice="drone",
                midi=dm,
                freq=round(note_freq(dm), 3),
                gain=DRONE_GAIN,
                pan=0.0,
                brightness=0.0,
                frame=None,
                label="reference: average of the first year" if len(frames) >= 12 else "reference: first frames",
            )
        )

    m = _base_meta("timeline", frames, cfg, meta)
    m.update(
        value_range={"lo": lo, "hi": hi},
        frames=frame_meta,
        extreme_threshold=thr,
        drone={"enabled": cfg.drone, "value": drone_value, "frames": len(ref_vals)},
    )
    return Score(meta=m, events=_sorted(events))
