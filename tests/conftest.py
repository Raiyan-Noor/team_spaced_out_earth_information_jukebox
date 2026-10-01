"""Shared fixtures. Tests never touch the network."""

from pathlib import Path

import pytest

from jukebox.colormap import Colormap, load_colormap

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def small_cmap() -> Colormap:
    """The 6-entry test colormap (units K)."""
    return load_colormap(FIXTURES / "colormap_small.xml")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Fail loudly if any test tries to open a network connection."""
    import socket

    def guard(*args, **kwargs):
        raise RuntimeError("tests must not use the network")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)
