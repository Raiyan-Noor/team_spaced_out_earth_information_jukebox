"""Shared fixtures. Tests never touch the network."""

from pathlib import Path

import pytest

from jukebox.colormap import Colormap, load_colormap

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def small_cmap() -> Colormap:
    """The 6-entry test colormap (units K)."""
    return load_colormap(FIXTURES / "colormap_small.xml")
