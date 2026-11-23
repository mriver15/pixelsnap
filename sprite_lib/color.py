"""OKLab colour math, hex helpers and palette presets."""
from __future__ import annotations

from pathlib import Path

import numpy as np


_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])
_M1I, _M2I = np.linalg.inv(_M1), np.linalg.inv(_M2)


def srgb_to_oklab(rgb) -> np.ndarray:
    c = np.asarray(rgb, np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return np.cbrt(lin @ _M1.T) @ _M2.T


def oklab_to_srgb(lab) -> np.ndarray:
    lin = (np.asarray(lab, np.float64) @ _M2I.T) ** 3 @ _M1I.T
    lin = np.clip(lin, 0, 1)
    c = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return np.clip(np.round(c * 255), 0, 255).astype(np.uint8)


def parse_hex(s: str) -> tuple[int, int, int]:
    s = s.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def to_hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(v) for v in rgb)


PRESETS = {
    "pico8": "000000 1D2B53 7E2553 008751 AB5236 5F574F C2C3C7 FFF1E8 FF004D FFA300 FFEC27 00E436 29ADFF 83769C FF77A8 FFCCAA",
    "pyxel": "000000 2B335F 7E2072 19959C 8B4852 395C98 A9C1FF EEEEEE D4186C D38441 E9C35B 70C6A9 7696DE A3A3A3 FF9798 EDC7B0",
    "gameboy": "0F380F 306230 8BAC0F 9BBC0F",
    "1bit": "000000 FFFFFF",
}


def load_palette(spec: str | None) -> np.ndarray | None:
    """Preset name, 'a,b,c' hex list, or a file with one hex color per line (Lospec .hex)."""
    if not spec:
        return None
    if spec.lower() in PRESETS:
        toks = PRESETS[spec.lower()].split()
    elif Path(spec).is_file():
        toks = [t for t in Path(spec).read_text().replace(",", " ").split() if t.strip("#").isalnum()]
    else:
        toks = spec.replace(",", " ").split()
    pal = np.array([parse_hex(t) for t in toks], np.uint8)
    if len(pal) < 2:
        raise SystemExit(f"palette '{spec}' resolved to fewer than 2 colors")
    return pal
