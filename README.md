# Earth Jukebox — hearing NASA Earth data

Our project for the **NASA Space Apps Challenge 2026**, challenge *"The Earth Information Jukebox"*: make NASA
Earth Information Center (EIC) visualizations accessible by translating sight into sound.

Our goal is a web app for blind and visually impaired people, in which an AI voice assistant describes an Earth
visualization, talks about it, and plays a sonification of the same frames. **This repository currently contains
only the first building block:** a Python sonification engine and command line tool that turns NASA map frames
into audio files.

## Status

**Pre-hackathon proof of concept (milestone M1).** What exists today:

- A Python package and command line tool (`python -m jukebox`) that renders sound **offline** to files: a stereo WAV,
  a JSON "score" (every note with its timing and meaning) and a plain-language legend text.
- Real NASA data: monthly global land surface temperature from NASA GIBS (2022–2024), decoded to actual
  temperatures, not pixel brightness.
- An offline demo with a synthetic planet, so it runs without internet.
- Automated tests (`pytest`), which run without network access.

What does **not** exist yet: the web app, the voice assistant, speech output, live or real-time playback, a phone
app. The audio has **not yet been evaluated with blind or low-vision listeners**. That is the next step.

## Listen first

No installation needed. These files are in [`samples/`](samples/):

| File | What you hear |
|---|---|
| [`lst_global_2022_2024.wav`](samples/lst_global_2022_2024.wav) | 36 months of real NASA land surface temperature (13 s). Three rising-and-falling arcs, one per year. ([legend](samples/lst_global_2022_2024.legend.txt)) |
| [`lst_scan_2024_07.wav`](samples/lst_scan_2024_07.wav) | One map, July 2024, swept west to east (6.5 s). Notes get quieter over strips that are mostly ocean. No strip is fully silent, because Antarctica puts some land in every strip. ([legend](samples/lst_scan_2024_07.legend.txt)) |
| [`demo_timeline.wav`](samples/demo_timeline.wav) | The synthetic practice planet, 24 months (9 s). ([legend](samples/demo_timeline.legend.txt)) |

Each WAV has a matching `.legend.txt` (the spoken-style explanation) and `.score.json`. The two real-data samples
are 22.05 kHz stereo to stay under 2 MB, and the demo is 44.1 kHz stereo. Headphones help with the left/right cue.

## Quickstart

Needs Python 3.10 or newer.

```bash
git clone <this repo>
cd <repo folder>
python -m venv .venv
# Windows:   .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

# 1. Offline demo (no internet): writes out/demo.wav, out/demo.score.json, out/demo.legend.txt
python -m jukebox demo --out out/demo

# 2. Real NASA data: 36 monthly global frames from GIBS (about 36 small downloads, cached in data/cache/)
python -m jukebox gibs --layer MODIS_Terra_L3_Land_Surface_Temp_Monthly_Day --start 2022-01 --end 2024-12 --mode timeline --out out/lst_global

# 3. One month, swept west to east
python -m jukebox gibs --layer MODIS_Terra_L3_Land_Surface_Temp_Monthly_Day --date 2024-07 --mode scan --out out/lst_scan

# 4. Any image without a known colour scale (approximate, see below)
python -m jukebox image path/to/frame.png --mode scan --out out/frame

# 5. Hear departures from normal instead of the seasons, and draw a check plot (needs matplotlib)
python -m jukebox gibs --layer MODIS_Terra_L3_Land_Surface_Temp_Monthly_Day --start 2022-01 --end 2024-12 --anomaly --plot --out out/lst_anomaly

# Tests (no network needed)
pytest -q
```

Every command prints a table of the frames and the notes chosen for them, plus the legend. `python -m jukebox
<command> --help` lists all options. Common options: `--tempo` (seconds per month, default 0.35), `--scale
pentatonic|chromatic|continuous`, `--invert`, `--no-drone`, `--range sequence|colormap`, `--grid 18x36`,
`--sample-rate`, `--mono`, `--anomaly`, `--plot`. For `gibs`: `--region global|bangladesh`, `--bbox`, `--size`, `--step day:N`,
`--offline` (cache only).

## How it works

```
NASA frame(s) → decode pixels to data values → per-frame features → musical mapping → WAV + score JSON + legend
```

1. **Decode real values, not colours.** For every GIBS layer, NASA publishes a colormap XML that says which RGB
   colour stands for which value range (e.g. `rgb 255,199,0 = 311.0–311.6 K`). We invert that table, so each pixel
   becomes a temperature in Kelvin, and transparent pixels become "no data". On the real July 2024 frame, every
   opaque pixel (217 distinct colours) matched a colormap entry exactly.
2. **Summarize each frame fairly.** The map is averaged onto a coarse 18 × 36 grid of 10° cells. Global averages are
   weighted by cos(latitude), because on a flat map the poles look much bigger than they are.
3. **Normalize across the whole sequence, never per frame.** Otherwise every month would sound the same and the
   change over time (the whole point) would vanish.
4. **Map to sound.** At most three sounds play at once:

   **Timeline mode** (one note per month):

   | Sound | Meaning |
   |---|---|
   | Pitch of the melody note | Area-weighted average of that month (higher = hotter), snapped to a pentatonic scale over 3 octaves |
   | Brighter, buzzier note | More of the land is unusually hot (above the 90th percentile of the whole sequence) |
   | Left/right position | Longitude of the hottest area (a backup cue only, never the sole carrier of meaning) |
   | Soft click | Start of each year (every January) |
   | Quiet steady tone | Average of the first year, a reference to hear whether later months sit above or below it |

   **Scan mode** (one map, swept west → east in 6 s): pitch = average of each 10° strip of longitude, loudness =
   how much of the strip has data (silence = no data, e.g. ocean), sound moves left → right, and clicks mark the
   start and 90° W, 0°, 90° E.

   **Anomaly mode** (`--anomaly`): before mapping, each calendar month's average across the sequence is subtracted
   cell by cell. The melody then follows "warmer or cooler than usual for this month" instead of the seasonal cycle.
   It needs every calendar month at least twice. With only 2–3 years the "normal" baseline is very short, so read
   the result as a demonstration of the technique, not as a climate finding (see Known limitations).
5. **Explain it.** `legend.txt` is generated from the actual mapping (including the real temperature range), written
   to be read aloud. Example from the real data sample:
   > Each note is one month. Higher pitch means hotter: the lowest note is about 9 °C and the highest about 20 °C.

The score JSON is meant as the hand-off to the future web player, which will play the same events with Web Audio
next to the visualization. The format is documented in [docs/SONIFICATION_SPEC.md](docs/SONIFICATION_SPEC.md),
section 6.4.

`--plot` also writes `PREFIX.png`, a check chart for sighted teammates. It has two panels: the data series on top
and the note played below, with lines marking where the clicks fall.

### Images with no machine-readable colour scale

Most EIC / NASA SVS visualization frames are images without a colormap file. The `image` command handles them
**approximately**:

- `--cmap NAME --vmin A --vmax B`: if you know which standard (matplotlib) colour scale and value range the image
  uses, colours are matched to it.
- `--legend-crop x0,y0,x1,y1 --vmin A --vmax B`: if the frame has a colour bar printed on it, give the pixel box
  around the bar. Colours are matched to the bar, and the bar itself is ignored. The map should fill the image, so
  crop away margins first.
- Otherwise: perceptual lightness (CIE L*) is used, and the legend says clearly that these are approximate brightness
  values, not measured data.
- `--mask-gray` treats gray pixels (land fill, labels, borders) as no data.

We have tested this path on synthetic images and on a GIBS frame re-saved as JPEG, but **not yet on actual EIC or
SVS frames**.

## NASA data used

| Dataset / service | What we used it for | How |
|---|---|---|
| **NASA GIBS** (Global Imagery Browse Services), WMS endpoint `https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi` | Source of all real map frames | WMS 1.1.1 GetMap requests, EPSG:4326, transparent PNG, 720 × 360 for the globe. No account needed. Downloads are cached in `data/cache/`. |
| **MODIS Terra Land Surface Temperature, monthly, daytime**: GIBS layer `MODIS_Terra_L3_Land_Surface_Temp_Monthly_Day` | The real-data demo: global land surface temperature, January 2022 – December 2024 (timeline) and July 2024 (scan) | 36 monthly frames. Pixels are decoded to Kelvin using the colormap below. Area-weighted monthly means drive the melody. |
| **GIBS colormap v1.3** `MODIS_Land_Surface_Temp.xml` (`https://gibs.earthdata.nasa.gov/colormaps/v1.3/`) | Turning pixel colours back into temperatures | Parsed by `jukebox/colormap.py`: 252 colour → value intervals in K, with no-data entries skipped. |
| **GIBS WMTS capabilities** (`https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/1.0.0/WMTSCapabilities.xml`) | Finding the right colormap for a layer, and checking that requested dates exist | Read once and cached for a week. |

The NASA Earth Information Center ([earth.gov](https://earth.gov)) and the
[NASA SVS daily visualizations gallery](https://svs.gsfc.nasa.gov/gallery/daily-visualizations) are the
visualizations we ultimately want to make audible. We have **not yet used frames from them** in this repository.

## Use of AI

This code was written with **Claude Code** (Anthropic's AI coding assistant, model Claude Opus 5.5), working from a
project brief (`CLAUDE.md`) and an algorithm specification (`docs/SONIFICATION_SPEC.md`) written by our team. The AI
generated the Python package, tests, sample audio and this README. It also checked the live NASA GIBS endpoints and
ran the commands to sanity-check the output. Every prompt, what was generated, and what still needs human review is
logged in [docs/AI_USAGE.md](docs/AI_USAGE.md). The team reviews all AI-generated code before the hackathon.

## Team

**Team Spaced Out**, Bangladesh

- **Raiyan**: developer, team lead
- **Noshin Sharmili**: UI
- **Jahidul Islam**: researcher

## Repository layout

```
jukebox/            the engine: gibs.py (all network access), colormap.py, decode.py, features.py,
                    mapping.py, score.py, synth.py, legend.py, demo.py, cli.py
tests/              pytest suite (offline; network access is blocked during tests)
samples/            committed example WAVs + legends + scores
docs/               SONIFICATION_SPEC.md (algorithm spec), AI_USAGE.md (AI usage log)
data/cache/, out/   created when you run commands; not committed
```

## Known limitations

- Pitch range is relative to the sequence you play (default), so the same note means different temperatures in
  different runs. `--range colormap` uses the colormap's fixed range instead (200–350 K for MODIS LST).
- A 10° cell counts as "has data" if any pixel in it has data, so coastal cells count as land.
- Only colormaps with numeric values are supported. Category maps (e.g. land cover) are rejected with a clear error.
- **Open question about the real data.** In anomaly mode, 2024 comes out about 0.5 K cooler than 2022 in this
  MODIS Terra daytime layer (yearly means 288.6, 288.6 and 288.0 K, with the same data coverage each year). That
  goes against other records, which call 2024 the warmest year on record. We have not explained it yet. Possible
  causes include the satellite's drifting overpass time and the 0.6 K colour steps of the image product. Until our
  researcher checks it, **do not present it as a climate signal**.
- Sound design choices (scale, tempo, timbres, legend wording) are first guesses that still need testing with
  visually impaired listeners.

## License

[MIT](LICENSE). Space Apps requires event code to stay open source.
