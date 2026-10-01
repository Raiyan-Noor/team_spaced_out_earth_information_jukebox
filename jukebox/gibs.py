"""NASA GIBS access: WMS URLs, capabilities lookup, colormap resolution, download + on-disk cache.

This is the ONLY module that talks to the network. Everything is cached under ``data/cache/gibs`` so a second
run (or a run with no network) reuses earlier downloads. See docs/SONIFICATION_SPEC.md section 2.1.
"""

from __future__ import annotations

import re
import time as _time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlencode

WMS_URL = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
CAPABILITIES_URL = "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/1.0.0/WMTSCapabilities.xml"
COLORMAP_BASE = "https://gibs.earthdata.nasa.gov/colormaps/v1.3/"
DEFAULT_CACHE = Path("data/cache")
CAPABILITIES_MAX_AGE_S = 7 * 24 * 3600

TIMEOUT = (10, 90)  # connect, read (seconds)
RETRIES = 2
REQUEST_DELAY_S = 0.3
USER_AGENT = "earth-jukebox/0.1 (NASA Space Apps 2026 proof of concept)"

#: Known colormap filenames, so the demo layer works even if the capabilities document cannot be parsed.
FALLBACK_COLORMAPS = {
    "MODIS_Terra_L3_Land_Surface_Temp_Monthly_Day": "MODIS_Land_Surface_Temp.xml",
    "GHRSST_L4_MUR25_Sea_Surface_Temperature_Anomalies": "GHRSST_Sea_Surface_Temperature_Anomalies.xml",
}

#: Region presets for ``--region`` as (minlon, minlat, maxlon, maxlat).
REGIONS: dict[str, tuple[float, float, float, float]] = {
    "global": (-180.0, -90.0, 180.0, 90.0),
    # Bangladesh extends ~88.01-92.67 E, ~20.59-26.63 N; the preset is its bounding box, slightly rounded.
    "bangladesh": (88.0, 20.6, 92.7, 26.6),
}

_NS = {
    "wmts": "http://www.opengis.net/wmts/1.0",
    "ows": "http://www.opengis.net/ows/1.1",
    "xlink": "http://www.w3.org/1999/xlink",
}


class GibsError(RuntimeError):
    """A human-readable failure (bad layer id, date out of range, network trouble)."""


@dataclass
class LayerInfo:
    """What the capabilities document says about one layer."""

    identifier: str
    title: str | None = None
    colormap_url: str | None = None
    time_values: list[str] = field(default_factory=list)  # e.g. ["2000-03-01/2026-08-01/P1M"]


# --- small helpers ------------------------------------------------------------------------------------------


def bbox_slug(bbox: tuple[float, float, float, float]) -> str:
    """Filesystem-friendly bbox key, e.g. ``m180_m90_180_90``."""
    return "_".join(f"{v:g}".replace("-", "m") for v in bbox)


def parse_bbox(text: str) -> tuple[float, float, float, float]:
    """Parse ``minlon,minlat,maxlon,maxlat`` and sanity-check it."""
    try:
        parts = tuple(float(x) for x in text.split(","))
    except ValueError as exc:
        raise GibsError(f"bad bbox {text!r}: expected four numbers minlon,minlat,maxlon,maxlat") from exc
    if len(parts) != 4:
        raise GibsError(f"bad bbox {text!r}: expected four numbers minlon,minlat,maxlon,maxlat")
    minlon, minlat, maxlon, maxlat = parts
    if not (-180 <= minlon < maxlon <= 180 and -90 <= minlat < maxlat <= 90):
        raise GibsError(f"bad bbox {text!r}: need -180<=minlon<maxlon<=180 and -90<=minlat<maxlat<=90")
    return parts  # type: ignore[return-value]


def default_size(bbox: tuple[float, float, float, float], longest: int = 720) -> tuple[int, int]:
    """Image size keeping square degrees square: 720x360 for the globe, aspect-correct for regions."""
    dlon, dlat = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if dlon >= dlat:
        return longest, max(1, round(longest * dlat / dlon))
    return max(1, round(longest * dlon / dlat)), longest


def build_wms_url(layer: str, time: str, bbox: tuple[float, float, float, float], size: tuple[int, int]) -> str:
    """WMS 1.1.1 GetMap URL for a transparent PNG (template in CLAUDE.md)."""
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.1.1",
        "REQUEST": "GetMap",
        "LAYERS": layer,
        "STYLES": "",
        "FORMAT": "image/png",
        "TRANSPARENT": "TRUE",
        "SRS": "EPSG:4326",
        "BBOX": ",".join(f"{v:g}" for v in bbox),
        "WIDTH": str(size[0]),
        "HEIGHT": str(size[1]),
        "TIME": time,
    }
    return WMS_URL + "?" + urlencode(params, safe=",:/")


def _parse_ym(text: str) -> date:
    m = re.fullmatch(r"(\d{4})-(\d{2})(?:-(\d{2}))?", text.strip())
    if not m:
        raise GibsError(f"bad date {text!r}: use YYYY-MM or YYYY-MM-DD")
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3) or 1))
    except ValueError as exc:
        raise GibsError(f"bad date {text!r}: {exc}") from exc


def time_steps(start: str, end: str, step: str = "month") -> list[str]:
    """ISO dates from ``start`` to ``end`` inclusive. ``month`` -> first of each month; ``day:N`` -> every N days."""
    a, b = _parse_ym(start), _parse_ym(end)
    if b < a:
        raise GibsError(f"--end {end} is before --start {start}")
    out: list[str] = []
    if step == "month":
        y, m = a.year, a.month
        while (y, m) <= (b.year, b.month):
            out.append(f"{y:04d}-{m:02d}-01")
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    elif re.fullmatch(r"day:\d+", step) and int(step[4:]) > 0:
        d = a
        while d <= b:
            out.append(d.isoformat())
            d += timedelta(days=int(step[4:]))
    else:
        raise GibsError(f"bad --step {step!r}: use 'month' or 'day:N'")
    return out


def single_time(text: str) -> str:
    """``--date`` value -> ISO date (YYYY-MM -> first of month)."""
    return _parse_ym(text).isoformat()


# --- capabilities -------------------------------------------------------------------------------------------


def parse_layer_info(capabilities_xml: bytes | str, layer: str) -> LayerInfo | None:
    """Find ``layer`` in a WMTS capabilities document. Returns None if the layer is not listed."""
    root = ET.fromstring(capabilities_xml)
    for lyr in root.iter(f"{{{_NS['wmts']}}}Layer"):
        ident = lyr.findtext("ows:Identifier", namespaces=_NS)
        if ident != layer:
            continue
        info = LayerInfo(identifier=layer, title=lyr.findtext("ows:Title", namespaces=_NS))
        for md in lyr.findall("ows:Metadata", _NS):
            role = md.get(f"{{{_NS['xlink']}}}role", "")
            if role.rstrip("/").endswith("colormap/1.3"):
                info.colormap_url = md.get(f"{{{_NS['xlink']}}}href")
        for dim in lyr.findall("wmts:Dimension", _NS):
            if (dim.findtext("ows:Identifier", namespaces=_NS) or "").lower() == "time":
                info.time_values = [v.text.strip() for v in dim.findall("wmts:Value", _NS) if v.text]
        return info
    return None


def time_in_range(t: str, time_values: list[str]) -> bool:
    """True if ISO date ``t`` falls inside any ``start/end/period`` (or single-date) value."""
    if not time_values:
        return True
    day = t[:10]
    for v in time_values:
        parts = v.split("/")
        if len(parts) == 1 and parts[0][:10] == day:
            return True
        if len(parts) >= 2 and parts[0][:10] <= day <= parts[1][:10]:
            return True
    return False


def describe_time_values(time_values: list[str]) -> str:
    """Short summary of a layer's time coverage for error messages."""
    if not time_values:
        return "unknown"
    first = time_values[0].split("/")[0][:10]
    last_parts = time_values[-1].split("/")
    last = (last_parts[1] if len(last_parts) > 1 else last_parts[0])[:10]
    return f"{first} to {last}"


# --- client with cache --------------------------------------------------------------------------------------


class GibsClient:
    """Downloads with timeouts, retries and polite delays; caches everything under ``cache_dir/gibs``."""

    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE, offline: bool = False) -> None:
        self.root = Path(cache_dir) / "gibs"
        self.offline = offline
        self._session = None
        self._last_request = 0.0
        self.downloads = 0
        self.cache_hits = 0

    # network ---------------------------------------------------------------------------------------------
    def _get(self, url: str):
        if self.offline:
            raise GibsError(f"offline mode and not in cache: {url}")
        import requests  # imported lazily so offline paths never need it

        if self._session is None:
            self._session = requests.Session()
            self._session.headers["User-Agent"] = USER_AGENT
        last_exc: Exception | None = None
        for attempt in range(RETRIES + 1):
            wait = REQUEST_DELAY_S - (_time.monotonic() - self._last_request)
            if wait > 0:
                _time.sleep(wait)
            try:
                self._last_request = _time.monotonic()
                resp = self._session.get(url, timeout=TIMEOUT)
                if resp.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {resp.status_code}")
                return resp
            except requests.RequestException as exc:
                last_exc = exc
                _time.sleep(1.0 * (attempt + 1))
        raise GibsError(
            f"could not reach NASA GIBS after {RETRIES + 1} attempts ({last_exc}). "
            f"Check your connection; files already downloaded are reused from {self.root}."
        )

    @staticmethod
    def _write_atomic(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(path)

    # capabilities ----------------------------------------------------------------------------------------
    def capabilities_path(self) -> Path:
        """Cached WMTS capabilities (refreshed after a week; a stale copy is used if the refresh fails)."""
        path = self.root / "WMTSCapabilities_epsg4326_best.xml"
        fresh = path.exists() and (_time.time() - path.stat().st_mtime) < CAPABILITIES_MAX_AGE_S
        if fresh or (path.exists() and self.offline):
            self.cache_hits += 1
            return path
        try:
            resp = self._get(CAPABILITIES_URL)
            if resp.status_code != 200 or b"<Capabilities" not in resp.content[:2000]:
                raise GibsError(f"unexpected capabilities response (HTTP {resp.status_code})")
        except GibsError:
            if path.exists():
                return path
            raise
        self._write_atomic(path, resp.content)
        self.downloads += 1
        return path

    def layer_info(self, layer: str) -> LayerInfo | None:
        """Layer metadata from the (cached) capabilities, or None if the layer id is unknown."""
        return parse_layer_info(self.capabilities_path().read_bytes(), layer)

    # colormaps -------------------------------------------------------------------------------------------
    def colormap_path(self, url: str) -> Path:
        """Download (or reuse) a colormap XML; cached by file name."""
        name = url.rstrip("/").rsplit("/", 1)[-1]
        path = self.root / "colormaps" / name
        if path.exists():
            self.cache_hits += 1
            return path
        resp = self._get(url)
        if resp.status_code != 200 or b"ColorMap" not in resp.content[:4000]:
            raise GibsError(f"could not download colormap {url} (HTTP {resp.status_code})")
        self._write_atomic(path, resp.content)
        self.downloads += 1
        return path

    # frames ----------------------------------------------------------------------------------------------
    def frame_path(self, layer: str, time: str, bbox: tuple[float, float, float, float], size: tuple[int, int]) -> Path:
        """Where a frame lives in the cache."""
        return self.root / layer / f"{bbox_slug(bbox)}_{size[0]}x{size[1]}" / f"{time}.png"

    def fetch_frame(self, layer: str, time: str, bbox: tuple[float, float, float, float], size: tuple[int, int]) -> Path:
        """Download (or reuse) one WMS GetMap PNG."""
        path = self.frame_path(layer, time, bbox, size)
        if path.exists() and path.stat().st_size > 0:
            self.cache_hits += 1
            return path
        resp = self._get(build_wms_url(layer, time, bbox, size))
        ctype = resp.headers.get("Content-Type", "")
        if resp.status_code != 200 or not ctype.startswith("image/png"):
            detail = re.sub(r"\s+", " ", resp.text[:400]) if "xml" in ctype or "text" in ctype else ctype
            raise GibsError(f"GIBS refused {layer} at {time} (HTTP {resp.status_code}): {detail}")
        self._write_atomic(path, resp.content)
        self.downloads += 1
        return path
