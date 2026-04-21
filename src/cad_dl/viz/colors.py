"""Deterministic id-to-color mapping via MD5 hash."""
from __future__ import annotations

import hashlib


def color_for_id(id_str: str) -> tuple[float, float, float]:
    """Hash an arbitrary id to a bright RGB triple in [0,1].

    Brightened to avoid near-black outputs (0.35 + 0.55 * byte/255).
    """
    h = hashlib.md5(id_str.encode()).digest()
    r, g, b = h[0] / 255.0, h[1] / 255.0, h[2] / 255.0
    return (0.35 + 0.55 * r, 0.35 + 0.55 * g, 0.35 + 0.55 * b)
