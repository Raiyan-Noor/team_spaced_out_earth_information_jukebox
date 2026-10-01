"""Plain-language legend generated from the actual mapping (score metadata).

The text is written to be read aloud by the future voice assistant, so: short sentences, no jargon, every
sound dimension explained in one sentence. See docs/SONIFICATION_SPEC.md section 8.
"""

from __future__ import annotations

import calendar
import math
from typing import Any

from jukebox.score import Score

#: Friendly names for layers we know. Other layers fall back to the colormap title or layer id.
KNOWN_LAYERS: dict[str, dict[str, str]] = {
    "MODIS_Terra_L3_Land_Surface_Temp_Monthly_Day": {
        "dataset": "daytime land surface temperature (MODIS Terra)",
        "area_noun": "the land",
    },
}

TEMPERATURE_UNITS = {"K", "C", "°C", "degC", "deg C", "Celsius", "Kelvin"}


def is_temperature(units: str | None) -> bool:
    """True for Kelvin or Celsius units."""
    return units is not None and units.strip() in TEMPERATURE_UNITS


def display_value(value: float | None, units: str | None, anomaly: bool = False, decimals: int = 0) -> str:
    """Format a data value for speech; Kelvin is presented in °C (differences need no offset)."""
    if value is None or not math.isfinite(value):
        return "unknown"
    if units is not None and units.strip() in ("K", "Kelvin"):
        value = value if anomaly else value - 273.15
        units = "°C"
    text = f"{value:.{decimals}f}"
    if text.lstrip("-").strip("0.") == "":
        text = text.lstrip("-")
    elif anomaly and value > 0:
        text = "+" + text
    return f"{text} {units}" if units else text


def describe_region(meta: dict[str, Any]) -> str:
    """'whole globe', a named region, or a lon/lat box."""
    if meta.get("region") and meta["region"] not in ("global", "custom"):
        return str(meta["region"])
    bbox = meta.get("bbox")
    if not bbox or [round(float(b)) for b in bbox] == [-180, -90, 180, 90]:
        return "whole globe"
    minlon, minlat, maxlon, maxlat = bbox

    def lon(x: float) -> str:
        return f"{abs(x):g}° {'west' if x < 0 else 'east'}"

    def lat(y: float) -> str:
        return f"{abs(y):g}° {'south' if y < 0 else 'north'}"

    return f"the area from {lon(minlon)} to {lon(maxlon)} and {lat(minlat)} to {lat(maxlat)}"


def describe_time(t: str | None, monthly: bool) -> str:
    """'January 2022' (monthly) or '5 January 2022'."""
    if not t or len(t) < 7:
        return t or "an unknown date"
    year, month = int(t[:4]), int(t[5:7])
    if monthly or len(t) < 10:
        return f"{calendar.month_name[month]} {year}"
    return f"{int(t[8:10])} {calendar.month_name[month]} {year}"


def describe_times(times: list[str | None]) -> str | None:
    """'January 2022 to December 2024', a single date, or None if dates are unknown."""
    known = [t for t in times if t]
    if not known or len(known) != len(times):
        return None
    monthly = all(len(t) >= 10 and t[8:10] == "01" for t in known)
    if len(known) == 1:
        return describe_time(known[0], monthly)
    return f"{describe_time(known[0], monthly)} to {describe_time(known[-1], monthly)}"


def _dataset_name(meta: dict[str, Any]) -> str:
    if meta.get("dataset"):
        return str(meta["dataset"])
    known = KNOWN_LAYERS.get(meta.get("layer") or "", {})
    return known.get("dataset") or meta.get("layer") or meta.get("colormap_title") or "the image"


def _area_noun(meta: dict[str, Any]) -> str:
    return meta.get("area_noun") or KNOWN_LAYERS.get(meta.get("layer") or "", {}).get("area_noun") or "the map"


def build_legend(score: Score) -> str:
    """Generate the legend for a score from its metadata."""
    meta = score.meta
    mode = meta.get("mode", "timeline")
    units = meta.get("units")
    anomaly = bool(meta.get("anomaly"))
    approximate = bool(meta.get("approximate"))
    temp = is_temperature(units) and not approximate
    vr = meta.get("value_range") or {}
    lo, hi = vr.get("lo"), vr.get("hi")
    invert = bool(meta.get("invert"))

    when = describe_times(meta.get("times") or [])
    header = f"Earth Jukebox: {_dataset_name(meta)}, {describe_region(meta)}"
    header += f", {when}." if when else "."
    lines = [header]

    if approximate:
        lines.append(
            "This image has no known colour scale, so the sound follows how light or dark each part of the "
            "picture is. These are approximate brightness values, not measured data."
        )
    elif meta.get("decode") == "cmap":
        lines.append(
            "The colours were matched to a standard colour scale chosen by the user, so values are approximate."
        )

    if anomaly:
        up, down = ("warmer than normal", "cooler than normal") if temp else ("above normal", "below normal")
        quantity = "difference from the usual value for that month"
    elif temp:
        up, down = "hotter", "colder"
        quantity = "temperature"
    elif approximate:
        up, down = "brighter", "darker"
        quantity = "brightness"
    else:
        up, down = "a higher value", "a lower value"
        quantity = "value"

    span = (hi - lo) if lo is not None and hi is not None else 0.0
    decimals = 0 if span >= 5 else (1 if span >= 0.5 else 2)
    lo_txt = display_value(lo, None if approximate else units, anomaly, decimals)
    hi_txt = display_value(hi, None if approximate else units, anomaly, decimals)
    pitch_rule = f"Higher pitch means {up}" if not invert else f"Lower pitch means {up}"
    if invert:
        range_rule = f"the highest note is about {lo_txt} and the lowest about {hi_txt}"
    else:
        range_rule = f"the lowest note is about {lo_txt} and the highest about {hi_txt}"
    if lo is None or hi is None or (math.isfinite(lo) and math.isfinite(hi) and hi - lo < 1e-9):
        range_rule = f"all notes sit on the middle pitch because the {quantity} does not change"

    if mode == "timeline":
        step = "month" if all(t and t[8:10] == "01" for t in (meta.get("times") or [None])) else "frame"
        lines.append(f"Each note is one {step}. {pitch_rule}: {range_rule}.")
        if anomaly:
            lines.append(
                "The usual seasonal cycle has been removed, so each note shows how unusual that month was "
                f"compared with the same month in other years of this sequence; {down} gives a lower note."
            )
        lines.append("A soft click marks the start of each year.")
        drone = meta.get("drone") or {}
        if drone.get("enabled") and drone.get("value") is not None:
            ref = "the first year" if drone.get("frames", 0) >= 12 else "the first frames"
            lines.append(
                f"The quiet steady tone is the average of {ref}, so you can hear whether later "
                f"{step}s sit above or below it."
            )
        hot = "unusually warm" if anomaly and temp else ("unusually hot" if temp else "unusually high")
        lines.append(f"A brighter, buzzier note means more of {_area_noun(meta)} is {hot} that {step}.")
        lines.append(
            f"The sound leans left or right toward where the {'hottest' if temp else 'highest'} area is, "
            "west to east."
        )
    else:
        sweep = meta.get("sweep_s", 6.0)
        n_frames = len(meta.get("times") or [1])
        lines.append(
            f"We sweep across the map from west to east in {sweep:g} seconds. Each note is one north-to-south "
            f"strip of the map. {pitch_rule}: {range_rule}."
        )
        lines.append(
            "Quieter notes mean less of that strip has data, and silence means no data at all, "
            "for example ocean in a land-only dataset."
        )
        ticks = meta.get("tick_longitudes") or []
        if ticks:
            names = {-90: "90° west", 0: "0° (the Greenwich meridian)", 90: "90° east"}
            marks = ", ".join(names.get(int(x), f"{x:g}°") for x in ticks)
            lines.append(f"A click marks the start of the sweep and the longitudes {marks}.")
        else:
            lines.append("A click marks the start of the sweep.")
        lines.append("The sound moves from your left ear to your right ear as the sweep goes east.")
        if n_frames > 1:
            lines.append("A double click and a short pause separate one map from the next.")
    return "\n".join(lines) + "\n"
