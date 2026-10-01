# CLAUDE.md — Earth Jukebox (NASA Space Apps 2026)

## What this repo is

Our entry for NASA Space Apps Challenge 2026, challenge **"The Earth Information Jukebox"**.

Official short description:
> NASA's Earth Information Center (EIC) produces stunning visualizations of our changing planet, but presenting
> Earth science through visualization alone limits who can reach it. We invite you to make this complex Earth
> science accessible, engaging, and multi-sensory by translating sight into sound. Your challenge is to build an
> "Earth Jukebox" — an interface, script, or application that pairs Earth Information Center (EIC) visual frames
> with dynamic sonifications generated in real time.

**Our angle: accessibility first.** The final product is a web app for blind and visually impaired users. An AI voice
assistant runs the whole experience: it describes EIC visualizations, holds a conversation about them, and plays a
sonification of the same frames. A phone app is future work.

**This repo currently holds only the sonification engine proof of concept (milestone M1):** a Python package + CLI
that renders audio offline. No web UI, no voice assistant, no LLM calls yet.

Team: Raiyan (developer, team lead), Noshin Sharmili (UI), Jahidul Islam (researcher).

## Milestone M1 — sonification PoC

Pipeline: NASA frame(s) → recover **data values** from pixels → per-frame features → musical mapping →
stereo WAV + JSON score + plain-language legend.

The full algorithm spec lives in @docs/SONIFICATION_SPEC.md. Read it before writing code; it has the priority order
(P0/P1/P2), formulas, defaults, CLI, and acceptance tests.

M1 is done when:
- `python -m jukebox demo` runs fully offline from a fresh clone and writes WAV + score JSON + legend.
- `python -m jukebox gibs ...` fetches real NASA GIBS frames (cached) and sonifies a monthly time series.
- `pytest -q` passes with no network access.
- README is accurate (Quickstart, How it works, NASA data used, Use of AI, Team, Status).
- A short sample WAV is committed under `samples/` so judges can listen without installing anything.

## Design decisions (ask before changing)

1. **Sonify data values, not pixel brightness.** NASA GIBS publishes a colormap XML per layer that maps each RGB
   to a value range. Invert it to get real numbers (e.g. Kelvin). Brightness is only a last-resort fallback for
   images with no known colormap, and must be labelled "approximate" in the legend.
2. **Normalize per sequence, never per frame.** Per-frame normalization erases change over time — the whole point.
3. **Area-weight by cos(latitude)** for any global or regional statistic on equirectangular images.
4. **Pure, per-frame functions.** `frame → features → events` must not depend on future frames (except the
   sequence-level normalization pass), so the mapping can later run in real time or be ported to the browser.
5. **The score JSON is the contract** between this engine and the future web player (Web Audio). Keep it stable
   and documented in the spec.
6. **Listener-first audio:** at most 3 simultaneous streams; pitch quantized to a pentatonic scale by default;
   higher value = higher pitch by default; stereo pan is only ever a redundant cue (never the sole carrier of meaning).
7. **Every sound dimension must be explainable in one sentence.** `legend.txt` is generated from the actual
   mapping so the voice assistant can read it aloud later.

## NASA data (endpoints verified 2026-10-01)

- GIBS WMS (no auth): `https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&LAYERS={layer}&STYLES=&FORMAT=image/png&TRANSPARENT=TRUE&SRS=EPSG:4326&BBOX={minlon},{minlat},{maxlon},{maxlat}&WIDTH={w}&HEIGHT={h}&TIME={YYYY-MM-DD}`
- WMTS capabilities (layer list, time ranges, colormap links): `https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/1.0.0/WMTSCapabilities.xml`
- Colormaps: `https://gibs.earthdata.nasa.gov/colormaps/v1.3/{name}.xml`. **The colormap filename often differs
  from the layer id** — resolve it from the layer's `ows:Metadata` with role `.../metadata-type/colormap/1.3`.
- **Primary demo layer:** `MODIS_Terra_L3_Land_Surface_Temp_Monthly_Day` → colormap `MODIS_Land_Surface_Temp.xml`,
  units K, monthly (TIME = first of month), available 2000-03 onward. A test fetch of 2024-07-01 at 720×360 PNG
  returned 217 distinct opaque colours, all exact matches to colormap entries; no-data pixels are transparent.
- Second candidate (diverging anomalies, stretch goal): `GHRSST_L4_MUR25_Sea_Surface_Temperature_Anomalies` →
  `GHRSST_Sea_Surface_Temperature_Anomalies.xml` (daily, 2002-09 to 2021-02).
- EIC itself: https://earth.gov. NASA SVS keeps a gallery of the daily-updated EIC visualizations:
  https://svs.gsfc.nasa.gov/gallery/daily-visualizations. These frames usually have no machine-readable colormap →
  they go through the fallback decode path.
- Every dataset used must appear in README → "NASA data used" with what we used it for and how. Judges check it.

## Repo layout

```
CLAUDE.md
README.md
LICENSE                     # MIT — Space Apps requires event code to stay open source
requirements.txt            # numpy, Pillow, requests
requirements-dev.txt        # pytest (+ matplotlib, optional)
jukebox/
  __init__.py
  __main__.py               # python -m jukebox → cli.main()
  cli.py                    # argparse: demo | gibs | image
  gibs.py                   # WMS URL building, capabilities lookup, download + on-disk cache (ONLY network code)
  colormap.py               # GIBS v1.3 XML parser, RGB→value lookup (exact + nearest fallback)
  decode.py                 # image → float32 value grid + NaN mask (+ fallbacks)
  features.py               # coarse grid, cos-lat weighted stats, hotspot, extremes
  mapping.py                # features → Score (timeline mode, scan mode)
  score.py                  # dataclasses: NoteEvent, Score; JSON read/write
  synth.py                  # Score → stereo float array → 16-bit WAV (stdlib `wave`)
  legend.py                 # plain-language description of the mapping
  demo.py                   # synthetic "seasonal planet" frames for offline demo/tests
tests/
  fixtures/colormap_small.xml
  test_colormap.py  test_features.py  test_mapping.py  test_synth.py  test_pipeline.py
docs/
  SONIFICATION_SPEC.md
  AI_USAGE.md
samples/                    # small committed WAVs (≤ 2 MB each)
data/cache/                 # gitignored
out/                        # gitignored
```

## Commands

```
pip install -r requirements.txt -r requirements-dev.txt
python -m jukebox demo --out out/demo
python -m jukebox gibs --layer MODIS_Terra_L3_Land_Surface_Temp_Monthly_Day --start 2022-01 --end 2024-12 --mode timeline --out out/lst_global
python -m jukebox gibs --layer MODIS_Terra_L3_Land_Surface_Temp_Monthly_Day --date 2024-07 --mode scan --out out/lst_scan
python -m jukebox image path/to/frame.png --mode scan --out out/frame
pytest -q
```

## Conventions

- Python ≥ 3.10, type hints everywhere, docstrings on public functions, dataclasses for structured data.
- Dependencies: numpy, Pillow, requests. Dev: pytest. matplotlib is optional and imported lazily (only for
  `--cmap` and `--plot`). **Ask before adding any other dependency** (no librosa, pydub, torch, pyaudio, etc.).
- Write WAVs with the stdlib `wave` module + numpy.
- All network access lives in `jukebox/gibs.py`: timeouts, ≤ 2 retries, small delay between requests, on-disk cache
  in `data/cache/`. Tests never touch the network.
- Deterministic outputs (seed any randomness).
- No secrets or API keys in the repo.
- README must never claim a feature that does not exist yet. Mark the project status as
  "pre-hackathon proof of concept".
- Small, descriptive commits. **Do not create the GitHub remote or push without asking.**

## AI usage log (required for the Space Apps submission)

The submission must name every AI tool used, with the prompts and reasoning. After each significant task, append to
`docs/AI_USAGE.md`: date, tool (Claude Code), the prompt (verbatim if short, otherwise summarized), what was
generated or changed, and what still needs human review.

## Out of scope for M1

Web UI, TTS, voice assistant, LLM image description, mobile app, live audio playback, MIDI export, deployment.
Keep the code shaped so these can plug in later (score JSON + legend text are the hooks).

## Key dates

- 2026-10-28: full challenge statements released — re-check dataset fit and adjust.
- 2026-11-01: local judging video due (needs a demo and NASA datasets named on camera).
- 2026-11-13/14: Bangladesh hackathon; global submission closes 2026-11-15.
