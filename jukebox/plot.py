"""``--plot``: a diagnostic PNG for sighted teammates to check the mapping (matplotlib, imported lazily).

Two stacked panels sharing the x axis (never a dual y axis): the data series on top, the MIDI note chosen
for each frame / column below. Faint vertical rules mark the same structure the listener hears as clicks.
"""

from __future__ import annotations

import math
from pathlib import Path

from jukebox.score import Score

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _note_name(midi: float) -> str:
    m = int(round(midi))
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _display(value: float | None, units: str | None, anomaly: bool) -> float:
    if value is None or not math.isfinite(value):
        return math.nan
    return value - 273.15 if units == "K" and not anomaly else value


def plot_score(score: Score, path: str | Path) -> Path:
    """Render the diagnostic plot for a timeline or scan score to ``path`` (PNG)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("--plot needs matplotlib: pip install matplotlib") from exc

    meta = score.meta
    units = meta.get("units")
    anomaly = bool(meta.get("anomaly"))
    unit_label = ("°C" if units == "K" else units or "value") + (" anomaly" if anomaly else "")
    scan = meta.get("mode") == "scan"

    if scan:
        xs, means, midis, labels, rules = [], [], [], [], []
        x0 = 0
        for fr in meta["frames"]:
            cols = fr["columns"]
            xs += [x0 + j for j in range(len(cols))]
            means += [_display(c["mean"], units, anomaly) for c in cols]
            midis += [c["midi"] if c["midi"] is not None else math.nan for c in cols]
            labels += [f"{c['lon']:.0f}" for c in cols]
            if x0:
                rules.append(x0 - 0.5)  # boundary between maps
            minlon, _, maxlon, _ = meta["bbox"]
            for lon in meta.get("tick_longitudes") or []:  # the meridian clicks
                rules.append(x0 - 0.5 + len(cols) * (lon - minlon) / (maxlon - minlon))
            x0 += len(cols)
        xlabel = "longitude of each strip (°, west → east)"
        title_top = f"Strip average ({unit_label})"
    else:
        frames = meta["frames"]
        xs = list(range(len(frames)))
        means = [_display(f["mean"], units, anomaly) for f in frames]
        midis = [f["midi"] if f["midi"] is not None else math.nan for f in frames]
        labels = [(f["time"] or str(i + 1))[:7] for i, f in enumerate(frames)]
        rules = [e.frame - 0.5 for e in score.events if e.voice == "tick" and e.frame]  # year boundaries
        xlabel = "frame"
        title_top = f"Area-weighted mean per frame ({unit_label})"

    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(10, 5.6), dpi=120, facecolor=SURFACE)
    for ax in (top, bottom):
        _style(ax)
        for r in rules:
            ax.axvline(r, color=GRID, linewidth=1.2, zorder=0)

    top.plot(xs, means, color=SERIES, linewidth=2, marker="o", markersize=4, solid_capstyle="round")
    top.set_title(title_top, loc="left", color=INK, fontsize=10)

    bottom.step(xs, midis, where="mid", color=SERIES, linewidth=2)
    bottom.plot(xs, midis, linestyle="none", marker="o", markersize=4, color=SERIES)
    bottom.set_title("MIDI note played (gaps = rests)", loc="left", color=INK, fontsize=10)
    lo_m, hi_m = meta.get("midi_range", [48, 84])
    ticks = list(range(int(lo_m), int(hi_m) + 1, 12))
    bottom.set_yticks(ticks, [f"{t} ({_note_name(t)})" for t in ticks])
    bottom.set_ylim(lo_m - 2, hi_m + 2)

    step = max(1, len(xs) // 12)
    bottom.set_xticks(xs[::step], labels[::step], rotation=0)
    bottom.set_xlabel(xlabel, color=INK_2, fontsize=8)
    vr = meta.get("value_range") or {}
    fig.suptitle(
        f"Earth Jukebox mapping check: {meta.get('layer') or meta.get('source')}  "
        f"(pitch range {_display(vr.get('lo'), units, anomaly):.1f} to {_display(vr.get('hi'), units, anomaly):.1f} {unit_label})",
        x=0.01,
        ha="left",
        color=INK_2,
        fontsize=8,
    )
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out
