"""Score dataclasses and JSON I/O — the contract with the future Web Audio player.

The JSON layout is documented in docs/SONIFICATION_SPEC.md section 6.4. Keep keys stable: add new keys rather
than renaming or removing existing ones, and bump ``version`` if a breaking change is ever unavoidable.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCORE_VERSION = 1
VOICES = ("melody", "tick", "drone")
EVENT_KEYS = ("t", "dur", "voice", "midi", "freq", "gain", "pan", "brightness", "frame", "label")
META_KEYS = (
    "mode",
    "source",
    "layer",
    "colormap",
    "units",
    "bbox",
    "times",
    "value_range",
    "scale",
    "midi_range",
    "tempo_s_per_frame",
    "frames",
)
FRAME_KEYS = ("time", "mean", "std", "extreme_frac", "hotspot_lon")


@dataclass
class NoteEvent:
    """One sound event. Times are in seconds; ``pan`` is in [-1, 1]; ``brightness`` in [0, 1]."""

    t: float
    dur: float
    voice: str  # melody | tick | drone
    midi: float
    freq: float
    gain: float
    pan: float = 0.0
    brightness: float = 0.0
    frame: int | None = None
    label: str = ""


@dataclass
class Score:
    """A complete sonification: metadata, time-sorted events and the plain-language legend."""

    meta: dict[str, Any]
    events: list[NoteEvent]
    legend: str = ""
    version: int = SCORE_VERSION

    @property
    def duration(self) -> float:
        """End time of the last event (without synth release tail)."""
        return max((e.t + e.dur for e in self.events), default=0.0)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict with NaN/inf replaced by ``None``."""
        return _clean(
            {
                "version": self.version,
                "meta": self.meta,
                "events": [asdict(e) for e in self.events],
                "legend": self.legend,
            }
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Score":
        """Rebuild a score from :meth:`to_dict` output."""
        return cls(
            meta=d["meta"],
            events=[NoteEvent(**{k: e[k] for k in EVENT_KEYS}) for e in d["events"]],
            legend=d.get("legend", ""),
            version=d.get("version", SCORE_VERSION),
        )


def note_freq(midi: float) -> float:
    """Equal-tempered frequency in Hz for a (possibly fractional) MIDI note number."""
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def _clean(obj: Any) -> Any:
    """Recursively turn numpy scalars into Python numbers and NaN/inf into None."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write_score(score: Score, path: str | Path) -> None:
    """Write ``score`` as pretty JSON (UTF-8, strict: no NaN)."""
    Path(path).write_text(json.dumps(score.to_dict(), indent=1, allow_nan=False, ensure_ascii=False), encoding="utf-8")


def read_score(path: str | Path) -> Score:
    """Read a score JSON file."""
    return Score.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def validate_score_dict(d: dict[str, Any]) -> list[str]:
    """Check a score dict against the section 6.4 contract. Returns a list of problems (empty = valid)."""
    problems: list[str] = []
    for k in ("version", "meta", "events", "legend"):
        if k not in d:
            problems.append(f"missing top-level key {k!r}")
    if problems:
        return problems
    meta = d["meta"]
    problems += [f"missing meta key {k!r}" for k in META_KEYS if k not in meta]
    vr = meta.get("value_range") or {}
    problems += [f"missing value_range.{k}" for k in ("lo", "hi") if k not in vr]
    for i, fr in enumerate(meta.get("frames") or []):
        problems += [f"meta.frames[{i}] missing {k!r}" for k in FRAME_KEYS if k not in fr]
    last_t = -math.inf
    for i, e in enumerate(d["events"]):
        missing = [k for k in EVENT_KEYS if k not in e]
        if missing:
            problems.append(f"event {i} missing {missing}")
            continue
        if e["voice"] not in VOICES:
            problems.append(f"event {i} has unknown voice {e['voice']!r}")
        if e["t"] < last_t:
            problems.append(f"event {i} is out of time order")
        last_t = e["t"]
    if not isinstance(d["legend"], str) or not d["legend"].strip():
        problems.append("legend is empty")
    return problems
