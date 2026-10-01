# Sonification Spec — Earth Jukebox engine (M1)

Audience: Claude Code and the team. This is the source of truth for the M1 algorithm. If something here is
ambiguous, pick the simplest option that keeps the design decisions in CLAUDE.md intact, and note the choice in
the plan / commit message.

## 0. Who it is for

A blind or low-vision listener who wants to *understand* an Earth visualization, not just hear something pretty.
So: few streams, stable mappings, clear structure markers, and a spoken-ready legend. Musical pleasantness is a
means (people listen longer), not the goal.

## 1. Pipeline

```
          ┌────────────┐   ┌─────────────┐   ┌────────────┐   ┌───────────┐   ┌──────────┐
frames →  │  decode    │ → │  features   │ → │  mapping   │ → │  synth    │ → │ outputs  │
(PNG)     │ RGB→value  │   │ per frame   │   │ → Score    │   │ → stereo  │   │ wav/json │
          │ + NaN mask │   │ (cos-lat)   │   │ (events)   │   │   audio   │   │ legend   │
          └────────────┘   └─────────────┘   └────────────┘   └───────────┘   └──────────┘
                                   ↑ sequence-level normalization pass (min/max over all frames)
```

Every stage is a pure function of its inputs. Only `gibs.py` does I/O over the network.

## 2. Inputs

### 2.1 NASA GIBS (primary, quantitative)

- Build WMS GetMap URLs (template in CLAUDE.md). Defaults: global bbox `-180,-90,180,90`, size `720x360`, PNG,
  TRANSPARENT=TRUE.
- Time steps: `--start/--end` as `YYYY-MM` (monthly layers → TIME=`YYYY-MM-01`) or `YYYY-MM-DD` with `--step day:N`.
- Resolve the colormap: `--colormap auto` (default) parses WMTSCapabilities.xml once (cache it), finds the layer by
  `ows:Identifier`, and reads the `ows:Metadata` href whose role ends in `colormap/1.3`. Allow `--colormap URL` to
  override. Hardcode a fallback map for the demo layer
  (`MODIS_Terra_L3_Land_Surface_Temp_Monthly_Day → MODIS_Land_Surface_Temp.xml`) so the demo command survives a
  capabilities parse failure.
- Cache: `data/cache/gibs/{layer}/{bbox_slug}_{w}x{h}/{TIME}.png` and `data/cache/gibs/colormaps/{name}.xml`.
- Region presets for `--region`: `global` → `-180,-90,180,90`; `bangladesh` → approx. `88.0,20.6,92.7,26.6`
  (verify and adjust). `--bbox` overrides.

### 2.2 Arbitrary frames (EIC / SVS images) — fallback decode

No machine-readable colormap, so decoding is approximate. Options in priority order:
1. `--cmap NAME --vmin A --vmax B`: build a 256-entry LUT from a matplotlib colormap (lazy import), nearest-colour
   match in RGB, then map index → value linearly.
2. (P2) `--legend-crop x0,y0,x1,y1 --vmin A --vmax B`: sample the colorbar strip in the image along its long axis
   to build the LUT. Works for SVS frames that include a colorbar.
3. Default: CIE L* (perceptual lightness) normalized to [0,1]. Legend must say values are approximate brightness.
Also accept `--mask-gray` to treat near-gray pixels (land, text, borders) as no-data.

### 2.3 Synthetic demo (offline)

`demo.py` generates N (default 24) monthly frames, 360×180, of a toy planet: value(lat, lon, month) =
baseline − 0.6·|lat| + seasonal term `A·cos(2π(month−7)/12)·sign(lat)` + small seeded noise, with a crude
rectangular "land" mask (ocean = no-data) so scan mode has audible gaps. Colour it with a synthetic colormap
written as a GIBS-style XML in memory, then run the **same** decode path as real data. This proves the full
pipeline end-to-end without network.

## 3. Decode

### 3.1 GIBS colormap XML (v1.3)

- File root `<ColorMaps>` contains one or more `<ColorMap>`; entries live under `<ColorMap>/<Entries>/<ColorMapEntry>`
  (fall back to direct children if `<Entries>` is missing).
- For each entry: `rgb="r,g,b"`, `transparent`, `nodata`, `value` (scaled, preferred) or `sourceValue` (raw),
  `ColorMap@units` if present.
- Skip entries with `nodata="true"`; transparent pixels in the image (alpha = 0) → NaN.
- Interval parsing for `value`: `[a,b)`, `(a,b]`, `[a,b]`, `[a]` (single), with `-INF` / `+INF` / `INF`.
  Representative value = midpoint if both bounds finite, else the finite bound; `[a]` → a.
- Classification colormaps (no numeric `value` and non-numeric labels) → raise a clear "unsupported in M1" error.
- If several entries share an RGB, keep the first and warn once.

### 3.2 RGB → value

1. Convert image to RGBA uint8 (handles palette PNGs).
2. Exact lookup: pack RGB into a uint32 key, map through a dict/vectorized `np.searchsorted` on sorted keys.
3. Pixels with no exact match: nearest colormap colour by Euclidean distance in RGB; accept if distance ≤ 12,
   else NaN. (Real GIBS PNGs should be ~100% exact; JPEGs and resampled images need this.)
4. Return `ValueGrid(values: float32[H,W], units: str|None, bbox, source_info)`.

Units: if units are `K`, keep Kelvin internally but present °C in the legend (value − 273.15).

## 4. Features (per frame)

Block-average the value grid onto a coarse grid, default **18 lat rows × 36 lon cols** (10° cells for global;
same counts for regional bboxes). Use `np.nanmean` per block; use `np.array_split` so sizes need not divide evenly.
Cell centres give `lat_c`, `lon_c`. Weight `w = cos(lat_c)` (in radians), zero where the cell is NaN.

| Feature | Definition |
|---|---|
| `mean` | Σ w·v / Σ w over valid cells |
| `std` | weighted standard deviation |
| `valid_frac` | weighted fraction of valid cells |
| `col_mean[j]` | weighted mean of column j over its valid cells (NaN if none) |
| `col_valid[j]` | fraction of valid cells in column j |
| `hotspot_lon` | lon_c of the max-valued valid cell |
| `extreme_frac` | weighted fraction of valid cells > T, where T = 90th percentile of **all valid cells across the whole sequence** (computed in the normalization pass) |

`Frame` dataclass: `time` (ISO string or None), `grid`, the features above.

## 5. Normalization (sequence-level pass)

- **Timeline:** `lo, hi = min(mean_t), max(mean_t)`; pad by 5% of span. If `hi − lo` < 1e-9 → every note is the
  middle note. `--range colormap` switches to the colormap's own finite min/max (for absolute comparison across runs).
- **Scan:** same idea over all `col_mean` values of the frame(s) being scanned.
- `extreme_frac` → normalize to [0,1] across the sequence (min/max; constant → 0.5).
- Store `lo`, `hi` (in data units) in the score metadata; the legend reports them.

## 6. Mapping → Score

### 6.1 Pitch

`midi = quantize(lo_midi + v_norm·(hi_midi − lo_midi))` onto the chosen scale.
Defaults: `lo_midi = 48` (C3), `hi_midi = 84` (C6), scale = major pentatonic `{0,2,4,7,9}` (16 available notes over
3 octaves). Options: `--scale pentatonic|chromatic|continuous` (continuous = no quantization).
Polarity: higher value → higher pitch; `--invert` flips it. `freq = 440·2^((midi−69)/12)`.

### 6.2 Timeline mode (frame sequence — "the jukebox")

| Stream | What it encodes | How |
|---|---|---|
| Melody | weighted `mean` per frame | one note per frame; pitch per 6.1; duration = 0.85 × beat |
| Brightness of melody | `extreme_frac` (how much of the map is unusually high) | harmonic amplitudes scale with it (see 7.1) |
| Pan of melody | `hotspot_lon` | `pan = hotspot_lon / 180` (regional: rescale to bbox) — redundant cue only |
| Year click | structure | short "tick" earcon at the start of each January (or every 12th frame if dates unknown) |
| Reference drone | context | quiet sustained tone at the pitch of the **first 12 frames' mean** (or first frame), whole duration; `--no-drone` disables |

Default beat = 0.35 s per frame (36 months ≈ 12.6 s). `--tempo SECONDS_PER_FRAME`.

### 6.3 Scan mode (single frame — sweep west → east)

| Stream | What it encodes | How |
|---|---|---|
| Melody | `col_mean[j]` per longitude column | one note per column, legato; total sweep default 6 s |
| Loudness | `col_valid[j]` | gain = 0.4 + 0.6·col_valid; if col_valid < 0.02 → rest (silence = no data, e.g. ocean in a land dataset) |
| Pan | column longitude | sweeps −1 → +1 |
| Orientation ticks | longitude markers | tick at sweep start and at −90°, 0°, +90° (when inside bbox) |

If multiple frames are passed to scan mode, scan each frame in order with a 0.5 s gap and a double tick between frames.

### 6.4 Score format (JSON contract with the future web player)

```json
{
  "version": 1,
  "meta": {
    "mode": "timeline",
    "source": "gibs",
    "layer": "MODIS_Terra_L3_Land_Surface_Temp_Monthly_Day",
    "colormap": "MODIS_Land_Surface_Temp.xml",
    "units": "K",
    "bbox": [-180, -90, 180, 90],
    "times": ["2022-01-01", "..."],
    "value_range": {"lo": 285.1, "hi": 301.7},
    "scale": "pentatonic", "midi_range": [48, 84],
    "tempo_s_per_frame": 0.35,
    "frames": [{"time": "2022-01-01", "mean": 287.3, "std": 14.2, "extreme_frac": 0.08, "hotspot_lon": 25.0}]
  },
  "events": [
    {"t": 0.0, "dur": 0.30, "voice": "melody", "midi": 62, "freq": 293.66, "gain": 0.8,
     "pan": 0.14, "brightness": 0.3, "frame": 0, "label": "2022-01"}
  ],
  "legend": "plain-language text, same as legend.txt"
}
```

`voice` ∈ `melody | tick | drone`. Events sorted by `t`. Times in seconds. Keep keys stable.

## 7. Synthesis

Sample rate 44100 Hz, stereo, float64 internally → 16-bit PCM WAV via stdlib `wave`.

### 7.1 Voices
- **melody:** additive sine, partials k = 1,2,3 with amplitudes `[1, 0.35·b, 0.15·b]` where `b = 0.4 + 1.2·brightness`.
  ADSR: attack 10 ms, decay 80 ms to sustain 0.7, release 120 ms (release may overlap next note).
- **tick:** 1500 Hz sine, 60 ms, exponential decay (τ ≈ 15 ms), gain 0.35, centred.
- **drone:** sine at the drone pitch, gain 0.12, 300 ms fade in/out, centred.

### 7.2 Mixing
- Equal-power pan: `L = cos((p+1)·π/4)`, `R = sin((p+1)·π/4)`, `p ∈ [−1, 1]`.
- Sum all events into a buffer of length `max(t + dur) + 0.5 s` tail.
- Soft limiter `y = tanh(1.2·x)`, then peak-normalize to −1 dBFS (0.891). No NaN/inf allowed.

## 8. Outputs (per run, `--out PREFIX`)

- `PREFIX.wav`
- `PREFIX.score.json` (section 6.4)
- `PREFIX.legend.txt` — generated from the actual mapping, e.g. for timeline:
  > Earth Jukebox: daytime land surface temperature (MODIS Terra), whole globe, January 2022 to December 2024.
  > Each note is one month. Higher pitch means hotter: the lowest note is about 12 °C and the highest about
  > 28 °C. A soft click marks the start of each year. The quiet steady tone is the average of the first year, so
  > you can hear whether later months sit above or below it. A brighter, buzzier note means more of the land is
  > unusually hot that month. The sound leans left or right toward where the hottest area is, west to east.
- `--plot` (P2, optional matplotlib): `PREFIX.png` with the mean series and the MIDI note per frame, for sighted
  teammates to check the mapping.
- Print a compact table to stdout: frame | time | mean (units) | midi | extreme_frac | pan.

## 9. CLI

```
python -m jukebox demo   [--frames 24] [--mode timeline|scan] [--out out/demo]
python -m jukebox gibs   --layer ID (--start YYYY-MM --end YYYY-MM [--step month|day:N] | --date YYYY-MM[-DD])
                         [--region global|bangladesh | --bbox minlon,minlat,maxlon,maxlat] [--size 720x360]
                         [--colormap auto|URL] [--mode timeline|scan] [common options]
python -m jukebox image  PATH [PATH ...] [--cmap NAME --vmin A --vmax B | --legend-crop x0,y0,x1,y1 --vmin A --vmax B]
                         [--bbox ...] [--mask-gray] [--mode scan|timeline] [common options]
common: --out PREFIX --tempo S --scale pentatonic|chromatic|continuous --invert --no-drone --range sequence|colormap
        --grid 18x36 --plot
```

Errors must be human-readable (bad layer id, date outside the layer's range, network failure with cache hint).

## 10. Tests and acceptance criteria (no network in tests)

1. **Colormap parsing:** fixture XML with a nodata entry, a `(-INF,10)` entry, `[10,20)` entries and a `[a]` entry →
   correct representative values; nodata skipped; units read.
2. **Exact decode round-trip:** paint a small RGBA image with fixture colours + one transparent pixel → recovered
   values match; transparent → NaN. Off-palette colour within tolerance → nearest value; far off → NaN.
3. **Cos-lat weighting:** uniform field → `mean` equals the constant; a field that is high only near the poles has a
   lower weighted mean than its unweighted mean.
4. **Monotonic mapping:** strictly increasing `mean` series → non-decreasing MIDI; constant series → constant MIDI;
   `--invert` reverses order.
5. **Normalization:** per-sequence, not per-frame — two frames with different uniform values must get different notes.
6. **Ticks:** 24 monthly frames starting 2022-01 → exactly 2 tick events, at frames 0 and 12.
7. **Scan gaps:** an all-NaN column produces no melody event for that column.
8. **Synth:** output is stereo, expected length ±1 sample-block, finite, peak ≤ 0.891 + ε.
9. **End-to-end:** `demo` pipeline writes a non-empty WAV, a score JSON that validates against the keys in 6.4,
   and a legend containing the value range.

## 11. Priorities

- **P0 (ship today):** colormap parse, decode, features, timeline mapping, synth, score/legend, `demo`, `gibs`
  timeline with cache, tests 1–6, 8, 9, honest README.
- **P1:** scan mode + test 7, `image` command with luminance and `--cmap` fallbacks, committed samples
  (`samples/demo_timeline.wav`, `samples/lst_global_2022_2024.wav`, each ≤ 2 MB — mono 22.05 kHz is acceptable
  for samples if size is tight).
- **P2:** `--legend-crop`, `--plot`, anomaly mode (subtract per-calendar-month climatology across the sequence so the
  melody shows departures from normal instead of the seasonal cycle), diverging-colormap timbre split for anomaly
  layers (warm = brighter timbre, cool = darker).

If time runs short, stop after P0 and make the README say exactly what works.

## 12. Later (not M1, but don't block it)

- Web player consumes `score.json` with Web Audio and schedules events per frame as the visualization plays —
  this is where "real time" lives; mapping functions are per-frame so they can also be ported to JS.
- Voice assistant reads `legend.txt`, announces years/frames, and answers questions using `meta.frames` stats.
- Speed control and "explore mode" (scrub to a frame, hear that frame's scan).
- Small listening test with visually impaired participants to validate polarity, tempo and legend wording.
