"""Command line: ``python -m jukebox demo | gibs | image``. See docs/SONIFICATION_SPEC.md section 9."""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Sequence

from jukebox import gibs
from jukebox.colormap import ColormapError, load_colormap
from jukebox.decode import (
    GLOBAL_BBOX,
    ValueGrid,
    decode_luminance,
    decode_with_colormap,
    decode_with_matplotlib_cmap,
)
from jukebox.demo import demo_frames
from jukebox.features import Frame, frame_features
from jukebox.legend import build_legend, display_value
from jukebox.mapping import SCALES, MappingConfig, lon_label, scan_score, timeline_score
from jukebox.score import Score, write_score
from jukebox.synth import SAMPLE_RATE, render, write_wav


class CliError(Exception):
    """An error to print without a traceback."""


# --- argument parsing ---------------------------------------------------------------------------------------


def _grid(text: str) -> tuple[int, int]:
    try:
        rows, cols = (int(x) for x in text.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"bad grid {text!r}: use ROWSxCOLS, e.g. 18x36") from exc
    if rows < 1 or cols < 1:
        raise argparse.ArgumentTypeError("grid needs at least 1x1 cells")
    return rows, cols


def _size(text: str) -> tuple[int, int]:
    try:
        w, h = (int(x) for x in text.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"bad size {text!r}: use WIDTHxHEIGHT, e.g. 720x360") from exc
    if not (1 <= w <= 4096 and 1 <= h <= 4096):
        raise argparse.ArgumentTypeError("size must be between 1 and 4096 pixels per side")
    return w, h


def _positive(text: str) -> float:
    v = float(text)
    if v <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return v


def _add_common(p: argparse.ArgumentParser, default_out: str) -> None:
    g = p.add_argument_group("common options")
    g.add_argument("--out", default=default_out, help=f"output prefix (default: {default_out})")
    g.add_argument("--tempo", type=_positive, default=0.35, help="seconds per frame in timeline mode (default 0.35)")
    g.add_argument("--scale", choices=list(SCALES), default="pentatonic", help="pitch quantization (default pentatonic)")
    g.add_argument("--invert", action="store_true", help="higher value -> lower pitch")
    g.add_argument("--no-drone", action="store_true", help="disable the reference drone (timeline mode)")
    g.add_argument(
        "--range",
        dest="range_mode",
        choices=["sequence", "colormap"],
        default="sequence",
        help="pitch range from this sequence (default) or from the colormap's full range",
    )
    g.add_argument("--grid", type=_grid, default=(18, 36), help="coarse grid ROWSxCOLS (default 18x36)")
    g.add_argument("--sample-rate", type=int, default=SAMPLE_RATE, help=f"WAV sample rate (default {SAMPLE_RATE})")
    g.add_argument("--mono", action="store_true", help="write a mono WAV (smaller; loses the pan cue)")


def build_parser() -> argparse.ArgumentParser:
    """The argparse parser for all subcommands."""
    parser = argparse.ArgumentParser(
        prog="python -m jukebox",
        description="Earth Jukebox: turn NASA Earth data frames into sound (WAV + score JSON + spoken legend).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("demo", help="offline synthetic planet (no network)")
    p.add_argument("--frames", type=int, default=24, help="number of monthly frames in timeline mode (default 24)")
    p.add_argument(
        "--mode", choices=["timeline", "scan"], default="timeline", help="scan mode sweeps the July frame of year 1"
    )
    _add_common(p, "out/demo")

    p = sub.add_parser("gibs", help="fetch NASA GIBS frames (cached) and sonify them")
    p.add_argument("--layer", required=True, help="GIBS layer id, e.g. MODIS_Terra_L3_Land_Surface_Temp_Monthly_Day")
    when = p.add_mutually_exclusive_group(required=True)
    when.add_argument("--start", help="first date, YYYY-MM or YYYY-MM-DD (needs --end)")
    when.add_argument("--date", help="a single date, YYYY-MM or YYYY-MM-DD")
    p.add_argument("--end", help="last date, YYYY-MM or YYYY-MM-DD")
    p.add_argument("--step", default="month", help="month (default) or day:N")
    where = p.add_mutually_exclusive_group()
    where.add_argument("--region", choices=list(gibs.REGIONS), default="global")
    where.add_argument("--bbox", help="minlon,minlat,maxlon,maxlat")
    p.add_argument("--size", type=_size, help="WIDTHxHEIGHT (default 720 on the longest side, aspect-correct)")
    p.add_argument("--colormap", default="auto", help="'auto' (from capabilities) or a colormap XML URL")
    p.add_argument("--mode", choices=["timeline", "scan"], default="timeline")
    p.add_argument("--cache-dir", default=str(gibs.DEFAULT_CACHE), help="cache directory (default data/cache)")
    p.add_argument("--offline", action="store_true", help="use only cached files, never the network")
    _add_common(p, "out/gibs")

    p = sub.add_parser("image", help="sonify local image file(s) with no GIBS colormap (approximate)")
    p.add_argument("paths", nargs="+", type=Path, help="PNG/JPEG frame(s), equirectangular")
    p.add_argument("--cmap", help="matplotlib colormap the image was drawn with (needs --vmin/--vmax)")
    p.add_argument("--vmin", type=float, help="data value at the low end of the colour scale")
    p.add_argument("--vmax", type=float, help="data value at the high end of the colour scale")
    p.add_argument("--units", help="units of --vmin/--vmax, e.g. K or mm (used in the legend)")
    p.add_argument("--bbox", help="minlon,minlat,maxlon,maxlat covered by the image (default: whole globe)")
    p.add_argument("--mask-gray", action="store_true", help="treat near-gray pixels (land, text, borders) as no data")
    p.add_argument("--mode", choices=["scan", "timeline"], default="scan")
    _add_common(p, "out/image")
    return parser


def mapping_config(args: argparse.Namespace) -> MappingConfig:
    """Translate common CLI flags into a :class:`MappingConfig`."""
    return MappingConfig(
        scale=args.scale,
        invert=args.invert,
        tempo=args.tempo,
        drone=not args.no_drone,
        range_mode=args.range_mode,
    )


# --- pipeline -----------------------------------------------------------------------------------------------


def sonify(
    grids: Sequence[ValueGrid],
    times: Sequence[str | None],
    args: argparse.Namespace,
    meta: dict[str, Any],
    colormap_range: tuple[float, float] | None = None,
) -> Score:
    """Decoded frames -> features -> score (+ legend)."""
    for vg in grids:
        if vg.values.shape[0] < args.grid[0] or vg.values.shape[1] < args.grid[1]:
            raise CliError(f"image is {vg.values.shape[1]}x{vg.values.shape[0]} px, smaller than --grid {args.grid}")
    frames: list[Frame] = [frame_features(vg, t, grid=args.grid) for vg, t in zip(grids, times)]
    if all(not math.isfinite(f.mean) for f in frames):
        raise CliError("none of the frames contain any valid data (everything decoded as no-data)")
    info = grids[0].source_info
    meta = {"decode": info.get("decode"), "approximate": bool(info.get("approximate")), **meta}
    build = scan_score if args.mode == "scan" else timeline_score
    score = build(frames, mapping_config(args), meta=meta, colormap_range=colormap_range)
    score.legend = build_legend(score)
    return score


def write_outputs(score: Score, args: argparse.Namespace) -> list[Path]:
    """Write ``PREFIX.wav``, ``PREFIX.score.json`` and ``PREFIX.legend.txt``."""
    prefix = Path(args.out)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    wav, js, txt = (prefix.with_name(prefix.name + ext) for ext in (".wav", ".score.json", ".legend.txt"))
    write_wav(wav, render(score, sr=args.sample_rate), sr=args.sample_rate, mono=args.mono)
    write_score(score, js)
    txt.write_text(score.legend, encoding="utf-8")
    return [wav, js, txt]


def format_table(score: Score) -> str:
    """Compact stdout table: frame | time | mean (units) | midi | extreme_frac | pan."""
    units = score.meta.get("units") or ""
    kelvin = units == "K"
    head = f"{'frame':>5}  {'time':<10}  {'mean ' + units:>10}"
    head += f"  {'mean degC':>9}" if kelvin else ""
    head += f"  {'midi':>6}  {'extreme':>7}  {'pan':>6}"
    lines = [head, "-" * len(head)]
    for i, fr in enumerate(score.meta.get("frames", [])):
        mean = fr.get("mean")
        ok = mean is not None and math.isfinite(mean)
        row = f"{i:>5}  {(fr.get('time') or '-')[:10]:<10}  {(f'{mean:.2f}' if ok else 'n/a'):>10}"
        if kelvin:
            row += f"  {(f'{mean - 273.15:.2f}' if ok else 'n/a'):>9}"
        midi = fr.get("midi")
        ef = fr.get("extreme_frac")
        row += f"  {(f'{midi:g}' if midi is not None else 'rest'):>6}"
        row += f"  {(f'{ef:.3f}' if ef is not None and math.isfinite(ef) else 'n/a'):>7}"
        row += f"  {fr.get('pan', 0.0):>6.2f}"
        lines.append(row)
    return "\n".join(lines)


def format_scan_table(score: Score) -> str:
    """Per-column table for scan mode: frame | lon | column mean | valid | midi."""
    units = score.meta.get("units") or ""
    head = f"{'frame':>5}  {'lon':>7}  {'mean ' + units:>10}  {'valid':>5}  {'midi':>6}"
    lines = [head, "-" * len(head)]
    for i, fr in enumerate(score.meta.get("frames", [])):
        for c in fr["columns"]:
            mean, midi = c["mean"], c["midi"]
            mean_txt = f"{mean:.2f}" if mean is not None and math.isfinite(mean) else "n/a"
            midi_txt = f"{midi:g}" if midi is not None else "rest"
            lon_txt = lon_label(round(c["lon"], 1))
            lines.append(f"{i:>5}  {lon_txt:>7}  {mean_txt:>10}  {c['valid']:>5.2f}  {midi_txt:>6}")
    return "\n".join(lines)


def report(score: Score, paths: list[Path]) -> None:
    """Print the table, value range, legend and output paths."""
    print(format_scan_table(score) if score.meta.get("mode") == "scan" else format_table(score))
    vr = score.meta["value_range"]
    units = score.meta.get("units")
    print(
        f"\nvalue range for pitch: {vr['lo']:.2f} .. {vr['hi']:.2f} {units or ''}"
        f" ({display_value(vr['lo'], units, decimals=1)} .. {display_value(vr['hi'], units, decimals=1)})"
    )
    print("\nLegend:\n" + score.legend)
    for p in paths:
        print(f"wrote {p}")


# --- commands -----------------------------------------------------------------------------------------------


def cmd_demo(args: argparse.Namespace) -> int:
    """Offline synthetic planet through the real decode path."""
    if not 1 <= args.frames <= 600:
        raise CliError("--frames must be between 1 and 600")
    pngs, cmap = demo_frames(args.frames if args.mode == "timeline" else 7)
    if args.mode == "scan":
        pngs = pngs[6:7]  # July of year 1
    grids = [decode_with_colormap(png, cmap, GLOBAL_BBOX, source_info={"time": t}) for t, png in pngs]
    meta = {
        "source": "synthetic",
        "layer": None,
        "colormap": cmap.name,
        "dataset": "a made-up practice planet (synthetic surface temperature, not real data)",
        "area_noun": "the land",
        "region": "whole globe",
    }
    score = sonify(grids, [t for t, _ in pngs], args, meta, colormap_range=cmap.value_range())
    report(score, write_outputs(score, args))
    return 0


def _resolve_colormap(client: gibs.GibsClient, layer: str, choice: str, info: gibs.LayerInfo | None) -> str:
    if choice != "auto":
        return choice
    if info is not None and info.colormap_url:
        return info.colormap_url
    if layer in gibs.FALLBACK_COLORMAPS:
        return gibs.COLORMAP_BASE + gibs.FALLBACK_COLORMAPS[layer]
    raise CliError(
        f"layer {layer} has no v1.3 colormap in the GIBS capabilities, so its pixels cannot be turned into "
        "data values. Pass --colormap URL if you know the right colormap."
    )


def cmd_gibs(args: argparse.Namespace) -> int:
    """Fetch GIBS frames (cached), decode with the layer's colormap and sonify."""
    if args.start and not args.end:
        raise CliError("--start needs --end")
    times = gibs.time_steps(args.start, args.end, args.step) if args.start else [gibs.single_time(args.date)]
    if len(times) > 600:
        raise CliError(f"{len(times)} frames is too many for one run (max 600)")
    bbox = gibs.parse_bbox(args.bbox) if args.bbox else gibs.REGIONS[args.region]
    size = args.size or gibs.default_size(bbox)
    client = gibs.GibsClient(args.cache_dir, offline=args.offline)

    info: gibs.LayerInfo | None = None
    try:
        info = client.layer_info(args.layer)
    except Exception as exc:  # capabilities are a convenience, not a requirement
        if args.layer not in gibs.FALLBACK_COLORMAPS:
            raise CliError(f"could not read the GIBS layer list ({exc})") from exc
        print(f"warning: could not read GIBS capabilities ({exc}); using the built-in colormap for {args.layer}")
    else:
        if info is None:
            raise CliError(
                f"unknown GIBS layer {args.layer!r}. Layer ids are case-sensitive; see "
                "https://worldview.earthdata.nasa.gov or the WMTS capabilities for valid ids."
            )
        bad = [t for t in times if not gibs.time_in_range(t, info.time_values)]
        if bad:
            raise CliError(
                f"{len(bad)} requested date(s) are outside {args.layer}'s coverage "
                f"({gibs.describe_time_values(info.time_values)}), e.g. {bad[0]}"
            )

    cmap_url = _resolve_colormap(client, args.layer, args.colormap, info)
    try:
        cmap = load_colormap(client.colormap_path(cmap_url))
    except ColormapError as exc:
        raise CliError(f"colormap {cmap_url}: {exc}") from exc

    grids = []
    for i, t in enumerate(times, 1):
        path = client.fetch_frame(args.layer, t, bbox, size)
        print(f"\r[{i}/{len(times)}] {t}  {path}", end="", flush=True, file=sys.stderr)
        grids.append(decode_with_colormap(path, cmap, bbox, source_info={"time": t, "path": str(path)}))
    print(f"\nframes: {client.downloads} downloaded, {client.cache_hits} from cache", file=sys.stderr)

    region = "whole globe" if bbox == gibs.REGIONS["global"] else (args.region if not args.bbox else "custom")
    meta = {
        "source": "gibs",
        "layer": args.layer,
        "layer_title": info.title if info else None,
        "colormap": cmap.name,
        "colormap_url": cmap_url,
        "colormap_title": cmap.title,
        "region": {"bangladesh": "Bangladesh"}.get(region, region),
        "size": list(size),
        "wms_example": gibs.build_wms_url(args.layer, times[0], bbox, size),
    }
    score = sonify(grids, times, args, meta, colormap_range=cmap.value_range())
    report(score, write_outputs(score, args))
    return 0


def cmd_image(args: argparse.Namespace) -> int:
    """Approximate decode of arbitrary frames: ``--cmap`` LUT, else perceptual lightness."""
    if args.cmap and (args.vmin is None or args.vmax is None):
        raise CliError("--cmap needs --vmin and --vmax (the data values at the two ends of the colour scale)")
    if (args.vmin is None) != (args.vmax is None) or (args.vmin is not None and not args.cmap):
        raise CliError("--vmin/--vmax only make sense together with --cmap")
    bbox = gibs.parse_bbox(args.bbox) if args.bbox else GLOBAL_BBOX
    grids = []
    for path in args.paths:
        if not path.is_file():
            raise CliError(f"no such image: {path}")
        info = {"path": str(path)}
        try:
            if args.cmap:
                vg = decode_with_matplotlib_cmap(
                    path, args.cmap, args.vmin, args.vmax, bbox, mask_gray=args.mask_gray, units=args.units, source_info=info
                )
            else:
                vg = decode_luminance(path, bbox, mask_gray=args.mask_gray, source_info=info)
        except (ImportError, ValueError, OSError) as exc:
            raise CliError(f"{path}: {exc}") from exc
        grids.append(vg)
    names = [p.name for p in args.paths]
    meta = {
        "source": "image",
        "dataset": f"the image {names[0]}" if len(names) == 1 else f"{len(names)} images ({names[0]} to {names[-1]})",
        "region": "whole globe" if bbox == GLOBAL_BBOX else "custom",
        "images": [str(p) for p in args.paths],
    }
    if args.cmap:
        meta.update(colormap=f"matplotlib:{args.cmap}", cmap_range=[args.vmin, args.vmax])
    score = sonify(grids, [None] * len(grids), args, meta, colormap_range=(args.vmin, args.vmax) if args.cmap else (0.0, 1.0))
    report(score, write_outputs(score, args))
    return 0


COMMANDS = {"demo": cmd_demo, "gibs": cmd_gibs, "image": cmd_image}


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # type: ignore[union-attr]
    args = build_parser().parse_args(argv)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("default")
            return COMMANDS[args.command](args)
    except (CliError, gibs.GibsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
