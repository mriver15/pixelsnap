#!/usr/bin/env python3
"""spritekit: turn diffusion "pixel art" into real, verified pixel art.  Pillow + numpy only.

Diffusion models have no pixel grid. Their "pixel art" is a large, soft image with a drifting
lattice, anti-aliased edges and 10^4-10^5 colors. Snapping a palette onto that (what a naive
finishing chain does) leaves it fake. spritekit recovers the grid, rebuilds the sprite at its
true resolution, locks a palette, cleans it, and then MEASURES the result so a bad generation
is rejected by numbers instead of by eye.

    python3 spritekit.py analyze  raw.png                     # grid + stats, writes nothing
    python3 spritekit.py finish   raw.png --out OUT           # grid-snap, palette, clean, QA
    python3 spritekit.py sheet    f0.png f1.png ... --out OUT --name walk
    python3 spritekit.py rank     c1.png c2.png ... --out OUT # finish all candidates, rank
    python3 spritekit.py qa       sprite.png                  # QA an already-native sprite
    python3 spritekit.py preview  sprite.png [--ascii]        # silhouette / multi-scale review

Exit code: 0 PASS or WARN, 2 FAIL (a gate tripped), 1 usage/runtime error.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

sys.dont_write_bytecode = True

# --------------------------------------------------------------------------------------
# colour (OKLab: perceptual distance is plain Euclidean, which is what every step below uses)
# --------------------------------------------------------------------------------------
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


# --------------------------------------------------------------------------------------
# io + background
# --------------------------------------------------------------------------------------
def load_rgba(path) -> tuple[np.ndarray, np.ndarray | None]:
    im = Image.open(path)
    has_alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info
    rgba = np.asarray(im.convert("RGBA"))
    rgb, alpha = rgba[..., :3], rgba[..., 3]
    if has_alpha and alpha.min() < 250:
        return rgb, alpha
    return rgb, None


def _run_reach(cand: np.ndarray, reached: np.ndarray) -> np.ndarray:
    """Within every horizontal run of True in `cand`, light up the whole run if any cell is reached."""
    h, w = cand.shape
    pad = np.zeros((h, w + 1), bool)
    pad[:, :w] = cand
    flat = pad.ravel()
    start = flat & ~np.concatenate(([False], flat[:-1]))
    rid = np.cumsum(start) * flat
    r = np.zeros((h, w + 1), bool)
    r[:, :w] = reached & cand
    hit = np.bincount(rid[r.ravel()], minlength=int(rid.max()) + 1) > 0
    hit[0] = False
    return (hit[rid] & flat).reshape(h, w + 1)[:, :w]


def border_connected(cand: np.ndarray) -> np.ndarray:
    """Pixels of `cand` 4-connected to the image border (vectorised run propagation)."""
    reached = np.zeros_like(cand)
    reached[0, :] = reached[-1, :] = reached[:, 0] = reached[:, -1] = True
    reached &= cand
    while True:
        nxt = _run_reach(cand, reached)
        nxt = _run_reach(cand.T, nxt.T).T
        if (nxt == reached).all():
            return nxt
        reached = nxt


def main_body_touches_border(fg: np.ndarray) -> bool:
    """True if the LARGEST foreground blob touches the image edge (the sprite is cut off). Vignette or
    background remnants in the corners are separate blobs and don't count."""
    step = max(1, max(fg.shape) // 256)
    lab, n = label_components(fg[::step, ::step], 8)
    if n == 0:
        return False
    main = lab == (int(np.argmax(np.bincount(lab.ravel())[1:])) + 1)
    return bool(main[0].any() or main[-1].any() or main[:, 0].any() or main[:, -1].any())


def box_blur(a: np.ndarray, r: int = 1) -> np.ndarray:
    """(2r+1)^2 box mean with edge replication."""
    p = np.pad(a, ((r, r), (r, r), (0, 0)), mode="edge")
    out = np.zeros_like(a, dtype=np.float64)
    for dy in range(2 * r + 1):
        for dx in range(2 * r + 1):
            out += p[dy:dy + a.shape[0], dx:dx + a.shape[1]]
    return out / (2 * r + 1) ** 2


def _moving_median(a: np.ndarray, k: int = 21) -> np.ndarray:
    """Per-channel moving median along axis 0 (rejects a sprite foot or hat that touches the border)."""
    k = min(k, len(a) - (1 - len(a) % 2))
    pad = np.pad(a, ((k // 2, k // 2), (0, 0)), mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(pad, k, axis=0), axis=-1)


def background_model(lab: np.ndarray) -> np.ndarray:
    """Expected background colour at every pixel, interpolated from the four border edges, so a vignette or
    gradient is modelled instead of exceeding a flat tolerance. Reduces to one colour for a flat background."""
    h, w, _ = lab.shape
    top, bot = _moving_median(lab[0]), _moving_median(lab[-1])
    lef, rig = _moving_median(lab[:, 0]), _moving_median(lab[:, -1])
    ty = (np.arange(h) / max(h - 1, 1))[:, None, None]
    tx = (np.arange(w) / max(w - 1, 1))[None, :, None]
    vert = top[None, :, :] * (1 - ty) + bot[None, :, :] * ty
    horz = lef[:, None, :] * (1 - tx) + rig[:, None, :] * tx
    return 0.5 * (vert + horz)


def estimate_background(rgb: np.ndarray, lab: np.ndarray, tol: float, key: tuple | None):
    """Flat, noisy or gently graded generated background -> (bg_mask, bg_rgb). No usable background -> all
    False. The background is modelled from the border edges (see background_model) and accepted when at
    least half the ring lies within a tolerance that widens with the ring's own spread. Judged on a lightly
    smoothed copy so noise cannot break flood-fill connectivity."""
    h, w, _ = lab.shape
    if key is not None:
        bg_rgb = np.array(key, np.uint8)
        model = np.broadcast_to(srgb_to_oklab(bg_rgb), lab.shape)
    else:
        model = background_model(lab)
    ring = np.concatenate([lab[0] - model[0], lab[-1] - model[-1], lab[:, 0] - model[:, 0], lab[:, -1] - model[:, -1]])
    d = np.sqrt((ring ** 2).sum(-1))
    near = d[d < 0.12]
    noise = float(np.sqrt((near ** 2).mean())) if len(near) else 0.0
    tol_eff = max(tol, min(2.5 * noise, 0.12))  # capped: dark sprite parts on a dark bg are ambiguous
    if key is None and (d < tol_eff).mean() < 0.5:
        return np.zeros((h, w), bool), None
    cand = np.sqrt(((box_blur(lab) - model) ** 2).sum(-1)) < tol_eff
    bg_rgb = bg_rgb if key is not None else oklab_to_srgb(np.median(model.reshape(-1, 3), axis=0))
    return border_connected(cand), bg_rgb


# --------------------------------------------------------------------------------------
# grid recovery
# --------------------------------------------------------------------------------------
def _smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    r = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    return np.convolve(x, k / k.sum(), mode="same")


def edge_profile(feat: np.ndarray, axis: int) -> np.ndarray:
    """Edge energy at every pixel BOUNDARY (index b = boundary between pixel b-1 and b). Length n+1."""
    d = np.sqrt(((np.diff(feat, axis=axis)) ** 2).sum(-1))
    d = np.maximum(d - 0.01, 0)
    e = d.sum(axis=1 - axis)
    return np.concatenate([[0.0], e, [0.0]])


def _edge_events(es: np.ndarray, lo: float, hi: float):
    """Boundary positions of local edge-energy peaks above 20% of the strong-edge level, with strengths."""
    seg = es[int(lo):int(hi) + 1]
    pos = seg[seg > 0]
    thr = 0.2 * float(np.percentile(pos, 90)) if len(pos) else 0.0
    ok = (es[1:-1] > es[:-2]) & (es[1:-1] >= es[2:]) & (es[1:-1] >= thr)
    idx = np.where(ok)[0] + 1
    idx = idx[(idx >= lo) & (idx <= hi)]
    return idx.astype(float), es[idx]


def _lattice_score(ev: np.ndarray, wt: np.ndarray, p: float, phases: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Per phase: chance-corrected COVERAGE (share of edge energy landing on a lattice line) times
    OCCUPANCY (share of lattice lines that carry an edge). Multiples of the true pitch lose coverage,
    sub-multiples lose occupancy, and a chance fit loses both."""
    if len(ev) < 4:
        return np.zeros(len(phases))
    tol = max(1.0, 0.12 * p)
    rel = ev[None, :] - lo - phases[:, None]
    near = rel - np.rint(rel / p) * p
    hit = np.abs(near) <= tol
    cov = (hit * wt).sum(1) / wt.sum()
    c0 = min(2 * tol / p, 0.95)
    kappa = np.clip((cov - c0) / (1 - c0), 0, 1)
    line = np.rint(rel / p)
    run = np.maximum.accumulate(np.where(hit, line, -1e9), axis=1)
    prev = np.concatenate([np.full((len(phases), 1), -1e9), run[:, :-1]], 1)
    occ = (hit & (line > prev)).sum(1) / (int((hi - lo) / p) + 1)
    return kappa * np.minimum(occ, 1.0)


def autocorr_peaks(e: np.ndarray, pmin: float, pmax: float, keep: int = 10) -> list[tuple[float, float]]:
    """Candidate pitches from the edge profile's autocorrelation -> [(pitch, strength)], strongest first.
    The strongest peak is NOT necessarily the pitch (fine texture, or a multiple when edges are sparse), so
    every prominent peak is a candidate; `choose_pitch` decides with image evidence."""
    x = e - _smooth(e, max(3.0, pmax / 2))  # drop the sprite-shaped envelope, keep the periodic part
    n = len(x)
    if n < 4 * pmin or x.std() == 0:
        return []
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac = ac / ac[0]
    hi = min(n // 2, int(pmax) + 2)  # a pitch, not the spacing between sprite features (head/torso/legs)
    out = []
    for i in range(max(2, int(math.ceil(pmin))), hi):
        if ac[i] > ac[i - 1] and ac[i] >= ac[i + 1] and ac[i] > 0.08:
            a_, b_, c_ = ac[i - 1], ac[i], ac[i + 1]
            den = a_ - 2 * b_ + c_
            out.append((float(i + (0.5 * (a_ - c_) / den if den != 0 else 0.0)), float(ac[i])))
    out.sort(key=lambda t: -t[1])
    return out[:keep]


def fit_lattice(e: np.ndarray, lo: float, hi: float, around: list[float], width: float = 0.02):
    """RIGID lattice -> (period, phase, score). Each period in `around` (+-width) is refined on a step fine
    enough that the last line stays within ~0.25 px; the best event coverage x occupancy wins."""
    span = hi - lo
    es = _smooth(e, 1.0)
    ev, wt = _edge_events(es, lo, hi)
    cands = around
    best = (None, 0.0, -1.0)
    for p0 in cands:
        step = max(0.25 * p0 / span, 1e-4)
        for p in (np.arange(p0 * (1 - width), p0 * (1 + width) + step, step) if width else [p0]):
            phases = np.arange(0, p, 0.25)
            sc = _lattice_score(ev, wt, p, phases, lo, hi)
            j = int(np.argmax(sc))
            if sc[j] > best[2]:
                best = (float(p), float(lo + phases[j]), float(sc[j]))
    return best


def lattice_lines(period: float, phase: float, n: int) -> np.ndarray:
    k0 = math.floor(-phase / period) - 1
    k1 = math.ceil((n - phase) / period) + 1
    return phase + np.arange(k0, k1 + 1) * period


DP_LAM, DP_BASE, DP_SLACK = 2.0, 0.2, 0.3  # tuned on jittered synthetic ground truth (see scripts/test_spritekit.py)


def edge_energy(e: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Smoothed edge profile normalised to the strong-edge level (90th percentile of its peaks), so a line
    placed in an edge's blur halo is worth less than the baseline cost of placing a line at all."""
    es = _smooth(e, 1.0)
    seg = es[int(lo):int(hi) + 1]
    pk = seg[1:-1][(seg[1:-1] > seg[:-2]) & (seg[1:-1] >= seg[2:])] if len(seg) > 2 else seg
    level = float(np.percentile(pk, 90)) if len(pk) else float(seg.max())
    return es / max(level, 1e-9)


def _dp(en: np.ndarray, period: float, a: int, b: int):
    """Non-uniform lattice by dynamic programming over pixel boundaries in [a, b]. Diffusion grids jitter
    (cells of 7-9 px in one image, errors that accumulate), so a rigid lattice cannot fit. Instead pick
    successive boundaries that sit on edge energy while each spacing stays within +-DP_SLACK of the pitch
    (quadratic penalty DP_LAM). -> (lines, total_reward)."""
    dmin = max(2, int(math.floor(period * (1 - DP_SLACK))))
    dmax = int(math.ceil(period * (1 + DP_SLACK)))
    ds = np.arange(dmin, dmax + 1)
    pen = DP_LAM * ((ds - period) / period) ** 2
    r = en[a:b + 1] - DP_BASE
    n = len(r)
    dp = np.full(n, -np.inf)
    back = np.zeros(n, np.int32)
    first = min(dmax + 1, n)
    dp[:first] = r[:first]  # first line anywhere in the first cell width
    for i in range(dmin, n):
        prev = i - ds
        cand = np.where(prev >= 0, dp[np.clip(prev, 0, n - 1)] - pen, -np.inf)
        j = int(np.argmax(cand))
        if cand[j] + r[i] > dp[i]:
            dp[i], back[i] = cand[j] + r[i], ds[j]
    tail = max(0, n - dmax)
    end = tail + int(np.argmax(dp[tail:]))
    out = [end]
    while back[out[-1]] > 0:
        out.append(out[-1] - int(back[out[-1]]))
    return a + np.array(out[::-1], float), float(dp[end])


def elastic_lines(en: np.ndarray, period: float, lo: float, hi: float) -> np.ndarray:
    pad = int(math.ceil(period))
    a, b = max(0, int(lo) - pad), min(len(en) - 1, int(hi) + pad)
    return _dp(en, period, a, b)[0]


def _integrals(lab: np.ndarray, fg: np.ndarray):
    m = fg[..., None].astype(np.float64)
    z = lambda a: np.pad(np.cumsum(np.cumsum(a, 0), 1), ((1, 0), (1, 0), (0, 0)))
    return z(lab * m), z((lab ** 2).sum(-1, keepdims=True) * m), z(m)


def explained_variance(ints, fg_total_sse: float, lx: np.ndarray, ly: np.ndarray, shape) -> float:
    """R^2 of the 'every cell is one flat colour' model on the foreground. Finer lattices are always >= as
    good, but a lattice coarser than the true pixel straddles cells and loses it quickly."""
    s1, s2, n = ints
    h, w = shape
    xa, xb = _cell_bounds(lx, w)
    ya, yb = _cell_bounds(ly, h)
    box = lambda I: (I[yb[:, None], xb[None, :]] - I[ya[:, None], xb[None, :]]
                     - I[yb[:, None], xa[None, :]] + I[ya[:, None], xa[None, :]])
    cn, c1, c2 = box(n)[..., 0], box(s1), box(s2)[..., 0]
    sse = float((c2 - np.where(cn > 0, (c1 ** 2).sum(-1) / np.maximum(cn, 1), 0)).sum())
    return 1.0 - sse / max(fg_total_sse, 1e-9)


R2_SLACK = 0.03  # a coarser pitch is accepted if it explains the image within this much of the best
RIGID_MIN = 0.4  # regular grids score 0.4-0.8, jittery ones ~0.3 (and then the rigid optimum just wanders)


def choose_pitch(lab, fg, raw, rng, dims, pmax=None, target=None):
    """-> joint pitch (or None). Candidates are all autocorrelation peaks of both axes (or the pitch
    implied by `target` cells across the sprite). Each gets a rigid lattice per axis; the answer is the
    COARSEST candidate whose explained variance is within R2_SLACK of the best (pixel-art converters'
    rule: the coarsest grid that still captures the detail)."""
    if target:
        ps = [(rng[a_][1] - rng[a_][0] - 4) / target[a_] if target[a_] else None for a_ in (0, 1)]
        have = [p for p in ps if p]
        return float(np.mean(have))
    per_axis = []
    for a_ in (0, 1):
        pm = pmax or max(4.0, (rng[a_][1] - rng[a_][0]) / 8)
        per_axis.append(autocorr_peaks(raw[a_][int(rng[a_][0]):int(rng[a_][1]) + 1], 3.0, pm))
    cands: list[float] = []
    for p, _ in sorted(sum(per_axis, []), key=lambda t: -t[1]):
        if all(abs(p - q) > 0.04 * q for q in cands):
            cands.append(p)
    cands = cands[:12]
    if not cands:
        return None
    ints = _integrals(lab, fg)
    total = float(((lab - (ints[0][-1, -1] / max(ints[2][-1, -1, 0], 1))) ** 2).sum(-1)[fg].sum())
    scored = []
    for c in cands:
        lines = []
        for a_ in (0, 1):
            rp, ph, _ = fit_lattice(raw[a_], *rng[a_], [c], 0.06)
            lines.append(lattice_lines(rp, ph, dims[a_]))
        scored.append((explained_variance(ints, total, lines[0], lines[1], fg.shape), c))
    best = max(r for r, _ in scored)
    coarsest = max(c for r, c in scored if r >= best - R2_SLACK)
    # candidates within 15% of the coarsest are the SAME pitch seen through noise: the best-fitting one wins
    pick = max((r, c) for r, c in scored if c >= 0.85 * coarsest and r >= best - R2_SLACK)[1]
    # a regular grid then pins its exact pitch (jittery ones score low and keep the autocorrelation value)
    fits = [fit_lattice(raw[a_], *rng[a_], [pick], 0.15) for a_ in (0, 1)]
    good = [f for f in fits if f[0] is not None and f[2] >= RIGID_MIN]
    if good:
        pick = sum(f[0] * f[2] for f in good) / sum(f[2] for f in good)
    return pick


def anchored_lines(e: np.ndarray, lo: float, hi: float, cells: int) -> np.ndarray:
    """cells+1 evenly spaced boundaries spanning the sprite's extent [lo, hi); the start is nudged (within
    +-30% of a cell) to the position where the lines sit on the most edge energy."""
    p = (hi - lo) / cells
    es = _smooth(e, 1.0)
    k = np.arange(cells + 1) * p
    best = max((float(np.interp(lo + d + k, np.arange(len(es)), es).sum()), d)
               for d in np.arange(-0.3 * p, 0.3 * p + 0.01, 0.5))
    return lo + best[1] + k


def _cell_bounds(lines: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Integer [start, end) per cell, clipped to the image; cells that are mostly off-image are marked empty."""
    a = np.clip(np.round(lines[:-1]).astype(int), 0, n)
    b = np.clip(np.round(lines[1:]).astype(int), 0, n)
    return a, b


def sample_cells(lab, fg_mask, lx, ly, shrink=0.25):
    """Per cell: median OKLab over the INNER region's foreground pixels (the inner region dodges the blur
    that bleeds in from neighbours), fg flag by majority of the whole cell, and purity (how much of the
    cell is within 0.06 of that colour; a direct measure of how well the lattice fits)."""
    h, w, _ = lab.shape
    xa, xb = _cell_bounds(lx, w)
    ya, yb = _cell_bounds(ly, h)
    integ = np.pad(np.cumsum(np.cumsum(fg_mask.astype(np.int64), 0), 1), ((1, 0), (1, 0)))
    ny, nx = len(ya), len(xa)
    cells = np.zeros((ny, nx, 3))
    fg = np.zeros((ny, nx), bool)
    purity = np.full((ny, nx), np.nan)
    for j in range(ny):
        y0, y1 = ya[j], yb[j]
        if y1 - y0 < 1:
            continue
        for i in range(nx):
            x0, x1 = xa[i], xb[i]
            if x1 - x0 < 1:
                continue
            area = (y1 - y0) * (x1 - x0)
            nfg = integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0]
            if nfg * 2 < area:
                continue
            sx, sy = shrink * (x1 - x0), shrink * (y1 - y0)
            ix0, ix1 = int(round(x0 + sx)), int(round(x1 - sx))
            iy0, iy1 = int(round(y0 + sy)), int(round(y1 - sy))
            ix1, iy1 = max(ix1, ix0 + 1), max(iy1, iy0 + 1)
            reg = lab[iy0:iy1, ix0:ix1].reshape(-1, 3)
            m = fg_mask[iy0:iy1, ix0:ix1].reshape(-1)
            if m.any():
                reg = reg[m]
            med = np.median(reg, axis=0)
            full = lab[y0:y1, x0:x1].reshape(-1, 3)[fg_mask[y0:y1, x0:x1].reshape(-1)]
            cells[j, i] = med
            fg[j, i] = True
            purity[j, i] = float((np.sqrt(((full - med) ** 2).sum(-1)) < 0.06).mean())
    return cells, fg, purity


class CellGrid:
    """A sprite recovered at its logical resolution, before palette locking."""
    def __init__(self, lab, fg, info):
        self.lab, self.fg, self.info = lab, fg, info


def extract_cells(path, o) -> CellGrid:
    rgb, alpha = load_rgba(path)
    h, w, _ = rgb.shape
    lab = srgb_to_oklab(rgb)
    if alpha is not None:
        bg, bg_rgb = alpha < 128, None
    elif o.opaque:
        bg, bg_rgb = np.zeros((h, w), bool), None
    else:
        bg, bg_rgb = estimate_background(rgb, lab, o.bg_tol, parse_hex(o.bg) if o.bg else None)
    fg_px = ~bg
    if not fg_px.any():
        raise SystemExit(f"{path}: no foreground found (background tolerance {o.bg_tol} may be too high)")
    ys, xs = np.where(fg_px)
    box = (xs.min(), xs.max() + 1, ys.min(), ys.max() + 1)
    # only judged on a keyed generation: a pre-matted, trimmed PNG touches its own canvas edge by construction
    clipped = bool(alpha is None and bg.any() and main_body_touches_border(fg_px))
    uniq = len(np.unique(rgb.reshape(-1, 3), axis=0))
    info = {"source": str(path), "source_size": [w, h], "source_colors": uniq, "clipped": clipped,
            "bg_color": to_hex(bg_rgb) if bg_rgb is not None else None, "has_alpha": alpha is not None}

    # grid recovery -------------------------------------------------------------------
    native = o.pixel_size == "1" or (o.pixel_size == "auto" and max(w, h) <= 192 and uniq <= 256)
    feat = lab if alpha is None else np.concatenate([lab, (alpha / 255.0 * 0.5)[..., None]], -1)
    if native:
        lx, ly = np.arange(w + 1, dtype=float), np.arange(h + 1, dtype=float)
        info.update(method="native", scale=[1.0, 1.0])
    else:
        fixed = None
        if o.pixel_size != "auto":
            parts = [float(v) for v in o.pixel_size.split(",")]
            fixed = parts if len(parts) == 2 else parts * 2
        dims = [(w, box[0], box[1]), (h, box[2], box[3])]
        raw, en, rng = [], [], []
        for axis, (n, lo, hi) in enumerate(dims):
            r_ = edge_profile(feat, 1 - axis)  # axis 0 = x boundaries
            rg = (max(0.0, lo - 2.0), min(float(n), hi + 2.0))
            raw.append(r_)
            rng.append(rg)
            en.append(edge_energy(r_, *rg))
        if fixed:
            joint = float(np.mean(fixed))
            rigid_p = list(fixed)
        else:
            target = None
            if o.target_size:
                tw, th = (int(v) for v in o.target_size.lower().split("x"))
                target = [tw or None, th or None]
            joint = choose_pitch(lab, fg_px, raw, rng, (w, h), o.max_pixel, target)
            if joint is None:
                raise SystemExit(f"{path}: no pixel grid found. Regenerate, or pass --pixel-size P / --target-size WxH.")
            rigid_p = [joint, joint]
        lines = []
        for a_, (n, lo, hi) in enumerate(dims):
            rp, ph, _ = fit_lattice(raw[a_], *rng[a_], [rigid_p[a_]], 0.0 if fixed else 0.03)
            lines.append((lattice_lines(rp, ph, n), elastic_lines(en[a_], joint, *rng[a_])))
        if target:  # a known cell count: an anchored lattice with EXACTLY that many cells across the sprite
            for a_ in (0, 1):
                if target[a_]:
                    ln = anchored_lines(raw[a_], dims[a_][1], dims[a_][2], target[a_])
                    lines[a_] = (ln, ln)
        # keep whichever lattice (rigid or elastic) yields purer cells
        best, best_pur = None, -1
        for mode in ("rigid", "elastic"):
            lx_, ly_ = lines[0][mode == "elastic"], lines[1][mode == "elastic"]
            c, f, pu = sample_cells(lab, fg_px, lx_, ly_)
            pur = float(np.nanmean(pu)) if f.any() else 0.0
            if pur > best_pur:
                best, best_pur = (mode, lx_, ly_, c, f, pu), pur
        mode, lx, ly, cells, fg, pur = best
        used = [float((l[-1] - l[0]) / (len(l) - 1)) for l in (lx, ly)]
        # anisotropy: each axis's own rigid pitch within +-45% of the joint one. Only trusted when BOTH fits are
        # strong; noisy real output scores low everywhere, so it is never flagged on weak evidence.
        ind = [fit_lattice(raw[a_], *rng[a_], [joint], 0.45) for a_ in (0, 1)]
        aniso = abs(ind[0][0] - ind[1][0]) / min(ind[0][0], ind[1][0]) if min(ind[0][2], ind[1][2]) >= 0.3 else 0.0
        info.update(method=mode, scale=[round(used[0], 3), round(used[1], 3)], pitch=round(joint, 3),
                    scale_aniso=round(aniso, 3))
        cells_fg = (cells, fg, pur)
    if native:
        cells_fg = sample_cells(lab, fg_px, lx, ly, shrink=0.0)
    cells, fg, pur = cells_fg
    jj, ii = np.where(fg)
    if not len(jj):
        raise SystemExit(f"{path}: grid recovered but no foreground cells")
    sl = (slice(jj.min(), jj.max() + 1), slice(ii.min(), ii.max() + 1))
    info["purity"] = round(float(np.nanmean(pur[fg])), 3)
    info.setdefault("scale_aniso", 0.0)
    return CellGrid(cells[sl], fg[sl], info)


# --------------------------------------------------------------------------------------
# palette
# --------------------------------------------------------------------------------------
def build_palette(points: np.ndarray, max_colors: int, tau: float, min_count: int) -> np.ndarray:
    """Merge-based palette in OKLab. Heavier colour wins, so merges never invent in-between colours:
    (1) fold anything within `tau` of a heavier colour, (2) fold colours used by fewer than
    `min_count` cells, (3) fold the least costly pairs until `max_colors`."""
    q = 0.004
    while True:
        keys = np.round(points / q).astype(np.int64)
        uk, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        if len(uk) <= 700:
            break
        q *= 1.5
    inv = inv.ravel()
    cen = np.stack([np.bincount(inv, points[:, c]) / cnt for c in range(3)], 1)
    wt = cnt.astype(float)
    alive = np.ones(len(cen), bool)
    D = np.sqrt(((cen[:, None] - cen[None]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)

    def merge(i, j):  # fold the lighter of (i, j) into the heavier
        a, b = (i, j) if wt[i] >= wt[j] else (j, i)
        wt[a] += wt[b]
        alive[b] = False
        D[b, :] = D[:, b] = np.inf

    while True:
        i, j = np.unravel_index(np.argmin(D), D.shape)
        if not np.isfinite(D[i, j]) or D[i, j] > tau:
            break
        merge(i, j)
    while alive.sum() > 1:
        small = np.where(alive & (wt < min_count))[0]
        if not len(small):
            break
        s = small[np.argmin(wt[small])]
        d = D[s].copy()
        if not np.isfinite(d).any():
            break
        merge(s, int(np.argmin(d)))
    while alive.sum() > max_colors:
        S = np.where(np.isfinite(D), D * np.sqrt(np.minimum.outer(wt, wt)), np.inf)
        i, j = np.unravel_index(np.argmin(S), S.shape)
        merge(i, j)
    return cen[alive][np.argsort(cen[alive][:, 0])]


def nearest(points: np.ndarray, pal_lab: np.ndarray) -> np.ndarray:
    return np.argmin(((points[:, None, :] - pal_lab[None]) ** 2).sum(-1), axis=1)


# --------------------------------------------------------------------------------------
# cleanup on the index grid
# --------------------------------------------------------------------------------------
def label_components(mask: np.ndarray, conn: int = 8) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    lab = np.zeros((h, w), np.int32)
    nb = [(-1, 0), (1, 0), (0, -1), (0, 1)] + ([(-1, -1), (-1, 1), (1, -1), (1, 1)] if conn == 8 else [])
    n = 0
    for y, x in zip(*np.where(mask)):
        if lab[y, x]:
            continue
        n += 1
        lab[y, x] = n
        dq = deque([(y, x)])
        while dq:
            cy, cx = dq.popleft()
            for dy, dx in nb:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not lab[ny, nx]:
                    lab[ny, nx] = n
                    dq.append((ny, nx))
    return lab, n


def _neighbors4(idx):
    p = np.pad(idx, 1, constant_values=-1)
    return p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]


def orphan_mask(idx: np.ndarray) -> np.ndarray:
    """Opaque pixels sharing a colour with none of their 4 neighbours."""
    up, dn, lf, rt = _neighbors4(idx)
    return (idx >= 0) & (idx != up) & (idx != dn) & (idx != lf) & (idx != rt)


def absorb_orphans(idx, pal_lab, tol=0.15):
    """An interior orphan whose colour is close to its surrounding majority is noise (diffusion speckle):
    absorb it. A contrasting orphan (an eye, a glint) is deliberate: keep it."""
    out = idx.copy()
    h, w = idx.shape
    n = 0
    for y, x in zip(*np.where(orphan_mask(idx))):
        ring = [idx[yy, xx] for yy in range(max(0, y - 1), min(h, y + 2))
                for xx in range(max(0, x - 1), min(w, x + 2)) if (yy, xx) != (y, x) and idx[yy, xx] >= 0]
        if len(ring) < 5:
            continue
        vals, cnt = np.unique(ring, return_counts=True)
        top = vals[np.argmax(cnt)]
        if cnt.max() >= 4 and np.linalg.norm(pal_lab[idx[y, x]] - pal_lab[top]) <= tol:
            out[y, x] = top
            n += 1
    return out, n


def drop_specks(idx, keep_detached: bool):
    if keep_detached:
        return idx, 0
    lab, n = label_components(idx >= 0, 8)
    if n <= 1:
        return idx, 0
    sizes = np.bincount(lab.ravel())[1:]
    keep = sizes >= max(2, 0.015 * sizes.max())
    out = idx.copy()
    out[~np.isin(lab, np.where(keep)[0] + 1) & (lab > 0)] = -1
    return out, int((~keep).sum())


def defringe(idx, pal_lab, bg_lab, tol, rounds):
    """Boundary cells that are really leftover background tint (close to the bg colour)."""
    out, removed = idx.copy(), 0
    for _ in range(rounds):
        up, dn, lf, rt = _neighbors4(out)
        edge = (out >= 0) & ((up < 0) | (dn < 0) | (lf < 0) | (rt < 0))
        near = np.linalg.norm(pal_lab - bg_lab, axis=1) < tol
        kill = edge & near[np.clip(out, 0, None)]
        if not kill.any():
            break
        out[kill] = -1
        removed += int(kill.sum())
    return out, removed


def smooth_assign(lab, fg, pal_lab, beta, sweeps=4):
    """Spatially coherent palette assignment (ICM on a Potts model): minimise squared OKLab error plus
    `beta` per 4-neighbour that disagrees. Removes the dither-like flicker you get when a smooth gradient
    is quantised cell by cell, while a high-contrast detail (error >> beta) still survives."""
    h, w = fg.shape
    k = len(pal_lab)
    err = ((lab[:, :, None, :] - pal_lab[None, None]) ** 2).sum(-1)  # h,w,K
    idx = np.where(fg, err.argmin(-1), -1)
    if beta <= 0:
        return idx
    yy, xx = np.mgrid[:h, :w]
    for sweep in range(sweeps * 2):
        parity = (yy + xx) % 2 == sweep % 2
        up, dn, lf, rt = _neighbors4(idx)
        agree = np.zeros((h, w, k))
        n_fg = np.zeros((h, w))
        for nb in (up, dn, lf, rt):
            ok = nb >= 0
            n_fg += ok
            agree += (nb[..., None] == np.arange(k)) & ok[..., None]
        cost = err + beta * (n_fg[..., None] - agree)
        new = np.where(fg, cost.argmin(-1), -1)
        idx = np.where(parity & fg, new, idx)
    return idx


def remove_enclosed(idx, pal_lab, bg_lab, tol, min_cells):
    """Background seen through gaps (between legs, inside a ring) is enclosed so the border flood-fill
    never reached it. Drop closed pockets of bg-coloured cells of at least `min_cells`."""
    near = (np.linalg.norm(pal_lab - bg_lab, axis=1) < tol)[np.clip(idx, 0, None)] & (idx >= 0)
    if not near.any():
        return idx, 0
    lab, n = label_components(near, 4)
    out, removed = idx.copy(), 0
    h, w = idx.shape
    for c in range(1, n + 1):
        m = lab == c
        if m.sum() < min_cells:
            continue
        ys, xs = np.where(m)
        if ys.min() == 0 or xs.min() == 0 or ys.max() == h - 1 or xs.max() == w - 1:
            continue
        up, dn, lf, rt = _neighbors4(np.where(m, -2, idx))
        if any(((nb == -1)).any() for nb in (up[m], dn[m], lf[m], rt[m])):
            continue  # touches the outside: not enclosed
        out[m] = -1
        removed += int(m.sum())
    return out, removed


def add_outline(idx, pal_rgb, color):
    """1px outer outline (4-neighbourhood). Returns the grown grid and the palette index used."""
    pal = list(map(tuple, pal_rgb))
    c = tuple(color)
    ci = pal.index(c) if c in pal else (pal.append(c) or len(pal) - 1)
    g = np.pad(idx, 1, constant_values=-1)
    up, dn, lf, rt = _neighbors4(g)
    ring = (g < 0) & ((up >= 0) | (dn >= 0) | (lf >= 0) | (rt >= 0))
    g = g.copy()
    g[ring] = ci
    return g, np.array(pal, np.uint8), ci


def compact(idx, pal_rgb):
    used = np.unique(idx[idx >= 0])
    remap = -np.ones(len(pal_rgb), np.int32)
    remap[used] = np.arange(len(used))
    out = np.where(idx >= 0, remap[np.clip(idx, 0, None)], -1)
    return out, pal_rgb[used]


def render(idx, pal_rgb) -> np.ndarray:
    rgba = np.zeros(idx.shape + (4,), np.uint8)
    m = idx >= 0
    rgba[m, :3] = pal_rgb[idx[m]]
    rgba[m, 3] = 255
    return rgba


# --------------------------------------------------------------------------------------
# QA
# --------------------------------------------------------------------------------------
DEFAULT_GATES = {
    # metric: (warn, fail, direction)   direction "high" = bigger is worse, "low" = smaller is worse
    "grid_purity": (0.72, 0.55, "low"),
    "min_cells": (16, 8, "low"),
    "scale_aniso": (0.06, 0.18, "high"),
    "colors_over_budget": (0, 0, "high"),
    "off_palette_ratio": (0.0, 0.0, "high"),
    "orphan_ratio": (0.03, 0.09, "high"),
    "banding_ratio": (0.12, 0.35, "high"),
    "components": (1, 4, "high"),
    "luma_range": (0.35, 0.15, "low"),
    "outline_coverage": (0.97, 0.85, "low"),
    "size_over": (0, 0, "high"),
    "clipped": (0, 0.5, "high"),
    "bg_collision": (0.04, 0.20, "high"),  # sprite uses colours close to the background key: regenerate on a chroma key
    "full_frame": (0, 0.5, "high"),  # the "sprite" is a filled rectangle: its background survived
}


def _runs(row: np.ndarray):
    n = len(row)
    cut = np.flatnonzero(np.diff(row) != 0) + 1
    st = np.concatenate([[0], cut])
    en = np.concatenate([cut, [n]])
    return [(int(s), int(e), int(row[s])) for s, e in zip(st, en) if row[s] >= 0]


def banding_ratio(idx: np.ndarray, min_chain: int = 3) -> float:
    """Pixels inside CHAINS of >= `min_chain` stacked, endpoint-aligned runs (>= 2px) whose colours differ
    from the run before: banding / pillow-shading (the TU Delft Pixel Fixer definition, which counts a pair;
    requiring a chain keeps an ordinary two-tone boundary from being reported as an artifact)."""
    marked = np.zeros(idx.shape, bool)
    for axis in (0, 1):
        a = idx if axis == 0 else idx.T
        m = marked if axis == 0 else marked.T
        prev: dict = {}
        for y in range(len(a)):
            cur = {}
            for s, e, c in _runs(a[y]):
                depth = 1
                if e - s >= 2 and (s, e) in prev and prev[(s, e)][0] != c:
                    depth = prev[(s, e)][1] + 1
                cur[(s, e)] = (c, depth)
                if depth >= min_chain:
                    m[y - depth + 1:y + 1, s:e] = True
            prev = cur
    opaque = (idx >= 0).sum()
    return float(marked.sum() / opaque) if opaque else 0.0


def _grade(value, warn, fail, direction):
    bad = (lambda v, t: v > t) if direction == "high" else (lambda v, t: v < t)
    if bad(value, fail):
        return "FAIL"
    if bad(value, warn):
        return "WARN"
    return "PASS"


def qa_report(rgba: np.ndarray, o, info: dict | None = None, pal: np.ndarray | None = None) -> dict:
    gates = {**DEFAULT_GATES}
    if getattr(o, "thresholds", None):
        for k, v in json.loads(Path(o.thresholds).read_text()).items():
            gates[k] = tuple(v)
    alpha = rgba[..., 3] > 0
    rgb = rgba[..., :3]
    h, w = alpha.shape
    flat = rgb[alpha]
    uniq, inv = np.unique(flat, axis=0, return_inverse=True)
    idx = -np.ones((h, w), np.int32)
    idx[alpha] = inv.ravel()
    lab = srgb_to_oklab(uniq)
    luma = lab[inv.ravel(), 0]
    _, ncomp = label_components(alpha, 8)
    m: dict = {
        "size": [int(w), int(h)],
        "colors": int(len(uniq)),
        "opaque_pixels": int(alpha.sum()),
        "fill_ratio": round(float(alpha.mean()), 3),
        "orphan_ratio": round(float(orphan_mask(idx).sum() / max(alpha.sum(), 1)), 4),
        "banding_ratio": round(banding_ratio(idx), 4),
        "components": int(ncomp),
        "luma_range": round(float(np.percentile(luma, 95) - np.percentile(luma, 5)), 3),
        "semi_transparent": int(((rgba[..., 3] > 0) & (rgba[..., 3] < 255)).sum()),
    }
    metrics = {
        "orphan_ratio": m["orphan_ratio"], "banding_ratio": m["banding_ratio"],
        "components": 1 if o.allow_detached else m["components"],
        "luma_range": m["luma_range"],
        "colors_over_budget": 0 if getattr(o, "palette", None) else max(0, m["colors"] - o.colors),
    }
    if info:
        metrics["grid_purity"] = info.get("purity", 1.0)
        metrics["scale_aniso"] = info.get("scale_aniso", 0.0)
        metrics["clipped"] = 1.0 if info.get("clipped") else 0.0
        if "bg_collision" in info:
            metrics["bg_collision"] = info["bg_collision"]
        if info.get("method") != "native":
            metrics["min_cells"] = min(w, h)
        if not getattr(o, "opaque", False) and not info.get("has_alpha"):
            metrics["full_frame"] = 1.0 if (m["fill_ratio"] > 0.96 and info.get("bg_color") is None) else 0.0
        m["source"] = {k: info[k] for k in ("source_size", "source_colors", "method", "scale", "purity", "bg_color") if k in info}
    if pal is not None:
        have = {tuple(c) for c in pal}
        off = sum(1 for c in map(tuple, flat) if c not in have)
        metrics["off_palette_ratio"] = round(off / len(flat), 4)
    if o.max_size:
        mw, mh = (int(v) for v in o.max_size.lower().split("x"))
        metrics["size_over"] = max(0, w - mw) + max(0, h - mh)
    if o.outline:
        oc = parse_hex(o.outline)
        up, dn, lf, rt = _neighbors4(np.where(alpha, 0, -1))
        edge = alpha & ((up < 0) | (dn < 0) | (lf < 0) | (rt < 0))
        metrics["outline_coverage"] = round(float((rgb[edge] == oc).all(-1).mean()) if edge.any() else 0.0, 3)
    grades, penalty = {}, 0.0
    for name, val in metrics.items():
        warn, fail, d = gates[name]
        g = _grade(val, warn, fail, d)
        grades[name] = {"value": val, "grade": g}
        if g != "PASS":
            span = abs(fail - warn) or 1.0
            frac = min(abs(val - warn) / span, 1.5)
            penalty += 6 + 14 * min(frac, 1.0) if g == "WARN" else 25
    verdict = "FAIL" if any(v["grade"] == "FAIL" for v in grades.values()) else (
        "WARN" if any(v["grade"] == "WARN" for v in grades.values()) else "PASS")
    m.update(gates=grades, score=int(max(0, round(100 - penalty))), verdict=verdict)
    return m


# --------------------------------------------------------------------------------------
# finish / sheet
# --------------------------------------------------------------------------------------
def finalize(grid: CellGrid, pal_lab: np.ndarray, o, fixed_pal: np.ndarray | None):
    """Assign colours, clean, outline. -> (rgba, palette_rgb, notes)."""
    idx = smooth_assign(grid.lab, grid.fg, pal_lab, o.smooth)
    pal_rgb = oklab_to_srgb(pal_lab)
    if fixed_pal is not None:  # map the NATURAL palette onto the bible palette (same source colour -> same target)
        tgt = nearest(pal_lab, srgb_to_oklab(fixed_pal))
        pal_rgb = fixed_pal[tgt]
        pal_lab = srgb_to_oklab(pal_rgb)
    notes = {}
    bg = grid.info.get("bg_color")
    if bg:
        near = np.linalg.norm(pal_lab - srgb_to_oklab(parse_hex(bg)), axis=1) < 0.08
        grid.info["bg_collision"] = round(float(near[idx[idx >= 0]].mean()), 3) if (idx >= 0).any() else 0.0
    if bg and o.defringe > 0:
        idx, notes["defringed"] = defringe(idx, pal_lab, srgb_to_oklab(parse_hex(bg)), o.fringe_tol, o.defringe)
    if bg and o.holes:
        idx, notes["holes_removed_cells"] = remove_enclosed(idx, pal_lab, srgb_to_oklab(parse_hex(bg)),
                                                            o.hole_tol, o.holes)
    idx, notes["orphans_absorbed"] = absorb_orphans(idx, pal_lab)
    idx, notes["specks_dropped"] = drop_specks(idx, o.keep_detached)
    ys, xs = np.where(idx >= 0)
    idx = idx[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    if o.outline:
        idx, pal_rgb, _ = add_outline(idx, pal_rgb, parse_hex(o.outline))
    idx, pal_rgb = compact(idx, pal_rgb)
    return render(idx, pal_rgb), pal_rgb, notes


def palette_for(grids: list[CellGrid], o, fixed_pal):
    pts = np.concatenate([g.lab[g.fg] for g in grids])
    cap = 10 ** 6 if fixed_pal is not None else max(2, o.colors - (1 if o.outline else 0))  # room for the outline
    return build_palette(pts, cap, o.tau, o.min_count)


def save_outputs(out: Path, stem: str, rgba, pal_rgb, report, o):
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(out / f"{stem}.png")
    (out / f"{stem}.palette.json").write_text(json.dumps([to_hex(c) for c in pal_rgb]))
    (out / f"{stem}.report.json").write_text(json.dumps(report, indent=2, default=lambda v: v.item()))
    if o.scale != 0:
        k = o.scale or max(1, 512 // max(rgba.shape[:2]))
        up = Image.fromarray(rgba, "RGBA").resize((rgba.shape[1] * k, rgba.shape[0] * k), Image.NEAREST)
        up.save(out / f"{stem}_x{k}.png")


def summary_line(name, r):
    s = r.get("source", {})
    sc = "x".join(f"{v:.2f}" for v in s["scale"]) if "scale" in s else "-"
    return (f"{r['verdict']:<4} {r['score']:>3}  {name:<28} {r['size'][0]}x{r['size'][1]}px  src-scale {sc:<11} "
            f"purity {s.get('purity', '-')!s:<5} colors {s.get('source_colors', '?')}->{r['colors']}  "
            f"orphans {r['orphan_ratio'] * 100:.1f}%  band {r['banding_ratio'] * 100:.0f}%"
            + "".join(f"  [{k}:{v['grade']}]" for k, v in r["gates"].items() if v["grade"] != "PASS"))


def cmd_finish(o):
    fixed = load_palette(o.palette)
    grids = [(Path(p), extract_cells(p, o)) for p in o.inputs]
    shared = o.shared_palette and len(grids) > 1
    results, worst = [], "PASS"
    if shared:
        pal_lab = palette_for([g for _, g in grids], o, fixed)
    for path, g in grids:
        lab = pal_lab if shared else palette_for([g], o, fixed)
        rgba, pal_rgb, notes = finalize(g, lab, o, fixed)
        rep = qa_report(rgba, o, g.info, fixed if fixed is not None else pal_rgb)
        rep["cleanup"] = notes
        save_outputs(Path(o.out), path.stem, rgba, pal_rgb, rep, o)
        print(summary_line(path.name, rep))
        results.append((path.stem, rep["verdict"], rep["score"]))
        if rep["verdict"] == "FAIL" or (rep["verdict"] == "WARN" and worst == "PASS"):
            worst = rep["verdict"]
    return 2 if worst == "FAIL" else 0


def cmd_rank(o):
    o.scale = o.scale if o.scale is not None else 0
    fixed = load_palette(o.palette)
    rows = []
    for p in o.inputs:
        try:
            g = extract_cells(p, o)
        except SystemExit as e:
            rows.append((Path(p).name, "FAIL", 0, str(e)))
            continue
        rgba, pal_rgb, notes = finalize(g, palette_for([g], o, fixed), o, fixed)
        rep = qa_report(rgba, o, g.info, fixed if fixed is not None else pal_rgb)
        save_outputs(Path(o.out), Path(p).stem, rgba, pal_rgb, rep, o)
        rows.append((Path(p).name, rep["verdict"], rep["score"], summary_line(Path(p).name, rep)))
    order = {"PASS": 0, "WARN": 1, "FAIL": 2}
    rows.sort(key=lambda r: (order[r[1]], -r[2]))
    print("rank (objective gates only; a human or VLM must still pick among the survivors):")
    for i, r in enumerate(rows, 1):
        print(f"{i:>2}. {r[3]}")
    survivors = [r[0] for r in rows if r[1] != "FAIL"]
    Path(o.out).mkdir(parents=True, exist_ok=True)
    (Path(o.out) / "rank.json").write_text(json.dumps(
        [{"file": r[0], "verdict": r[1], "score": r[2]} for r in rows], indent=2))
    print(f"\n{len(survivors)}/{len(rows)} survive the gates.")
    return 0 if survivors else 2


ANCHORS = ("bottom-center", "centroid-x", "none")


def _anchor_xy(rgba, how):
    a = rgba[..., 3] > 0
    ys, xs = np.where(a)
    y = int(ys.max())
    if how == "centroid-x":
        return int(round(xs.mean())), y
    return int((xs.min() + xs.max()) // 2), y


def cmd_sheet(o):
    fixed = load_palette(o.palette)
    grids = [extract_cells(p, o) for p in o.inputs]
    pal_lab = palette_for(grids, o, fixed)
    frames, pals, reps = [], [], []
    for g in grids:
        rgba, pal_rgb, notes = finalize(g, pal_lab, o, fixed)
        frames.append(rgba)
        pals.append(pal_rgb)
        reps.append(qa_report(rgba, o, g.info, fixed if fixed is not None else None))
    # shared palette = union of what the frames actually used
    union = np.unique(np.concatenate(pals), axis=0)

    # align on a common canvas
    anc = [_anchor_xy(f, o.anchor) if o.anchor != "none" else (f.shape[1] // 2, f.shape[0] - 1) for f in frames]
    left = max(a[0] for a in anc)
    right = max(f.shape[1] - a[0] for f, a in zip(frames, anc))
    up = max(a[1] for a in anc)
    down = max(f.shape[0] - a[1] for f, a in zip(frames, anc))
    cw, ch = left + right, up + down
    canvas = []
    for f, a in zip(frames, anc):
        c = np.zeros((ch, cw, 4), np.uint8)
        oy, ox = up - a[1], left - a[0]
        c[oy:oy + f.shape[0], ox:ox + f.shape[1]] = f
        canvas.append(c)

    n = len(canvas)
    cols = o.cols or n
    rows = math.ceil(n / cols)
    sheet = np.zeros((rows * ch, cols * cw, 4), np.uint8)
    frame_meta = []
    for i, c in enumerate(canvas):
        r, k = divmod(i, cols)
        sheet[r * ch:(r + 1) * ch, k * cw:(k + 1) * cw] = c
        frame_meta.append({"index": i, "source": Path(o.inputs[i]).name,
                           "rect": {"x": k * cw, "y": r * ch, "w": cw, "h": ch}})

    # sheet-level consistency metrics
    heights = np.array([(f[..., 3] > 0).any(1).sum() for f in frames], float)
    areas = np.array([(f[..., 3] > 0).sum() for f in frames], float)
    ref_lab = srgb_to_oklab(np.unique(frames[0][frames[0][..., 3] > 0][:, :3], axis=0))
    drift = []
    for g in grids[1:]:
        d = np.sqrt(((g.lab[g.fg][:, None] - ref_lab[None]) ** 2).sum(-1)).min(1)
        drift.append(float((d > 0.10).mean()))
    diffs = [float((canvas[i][..., 3:] != canvas[i + 1][..., 3:]).mean() +
                   (canvas[i][..., :3] != canvas[i + 1][..., :3]).any(-1).mean()) / 2 for i in range(n - 1)]
    sm = {
        "frames": n, "cell": [cw, ch], "palette_colors": int(len(union)),
        "height_cv": round(float(heights.std() / heights.mean()), 3),
        "area_ratio": round(float(areas.max() / max(areas.min(), 1)), 2),
        "palette_drift_max": round(max(drift), 3) if drift else 0.0,
        "frame_delta_mean": round(float(np.mean(diffs)), 3) if diffs else 0.0,
    }
    if o.loop and n > 2:
        last = float((canvas[-1] != canvas[0]).any(-1).mean())
        sm["loop_gap_ratio"] = round(last / max(np.mean(diffs), 1e-6), 2)
    sheet_gates = {"height_cv": (0.04, 0.10, "high"), "area_ratio": (1.35, 2.5, "high"),
                   "palette_drift_max": (0.08, 0.20, "high")}
    sg = {k: {"value": sm[k], "grade": _grade(sm[k], *v)} for k, v in sheet_gates.items()}
    if "loop_gap_ratio" in sm:
        sg["loop_gap_ratio"] = {"value": sm["loop_gap_ratio"], "grade": _grade(sm["loop_gap_ratio"], 1.5, 4.0, "high")}
    sm["gates"] = sg
    verdicts = [r["verdict"] for r in reps] + [v["grade"] for v in sg.values()]
    sm["verdict"] = "FAIL" if "FAIL" in verdicts else ("WARN" if "WARN" in verdicts else "PASS")

    out = Path(o.out)
    out.mkdir(parents=True, exist_ok=True)
    name = o.name
    Image.fromarray(sheet, "RGBA").save(out / f"{name}_sheet.png")
    for i, c in enumerate(canvas):
        Image.fromarray(c, "RGBA").save(out / f"{name}_frame_{i}.png")
    manifest = {"name": name, "frame_count": n, "cell_size": [cw, ch], "columns": cols, "rows": rows,
                "sheet_size": [cols * cw, rows * ch], "fps": o.fps, "anchor": o.anchor,
                "palette": [to_hex(c) for c in union], "frames": frame_meta, "qa": sm,
                "per_frame": [{k: r[k] for k in ("size", "colors", "score", "verdict", "orphan_ratio",
                                                 "banding_ratio")} for r in reps]}
    (out / f"{name}_manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / f"{name}_aseprite.json").write_text(json.dumps({
        "frames": [{"filename": f"{name} {i}", "frame": m["rect"], "rotated": False, "trimmed": False,
                    "spriteSourceSize": {"x": 0, "y": 0, "w": cw, "h": ch}, "sourceSize": {"w": cw, "h": ch},
                    "duration": int(1000 / o.fps)} for i, m in enumerate(frame_meta)],
        "meta": {"app": "spritekit", "image": f"{name}_sheet.png", "format": "RGBA8888",
                 "size": {"w": cols * cw, "h": rows * ch}, "scale": "1",
                 "frameTags": [{"name": name, "from": 0, "to": n - 1, "direction": "forward"}]}}, indent=2))
    for i, r in enumerate(reps):
        print(summary_line(Path(o.inputs[i]).name, r))
    print(f"\nsheet {name}: {n} frames, cell {cw}x{ch}, {len(union)} colors, "
          f"height_cv {sm['height_cv']}, area_ratio {sm['area_ratio']}, palette_drift {sm['palette_drift_max']} -> {sm['verdict']}")
    for k, v in sg.items():
        if v["grade"] != "PASS":
            print(f"  [{k}:{v['grade']}] value {v['value']}")
    return 2 if sm["verdict"] == "FAIL" else 0


# --------------------------------------------------------------------------------------
# analyze / qa / preview
# --------------------------------------------------------------------------------------
def cmd_analyze(o):
    for p in o.inputs:
        g = extract_cells(p, o)
        i = g.info
        h, w = g.fg.shape
        print(f"{Path(p).name}: source {i['source_size'][0]}x{i['source_size'][1]}  {i['source_colors']} colors  "
              f"bg {i['bg_color']}  clipped {i['clipped']}\n"
              f"  grid: {i['method']}  scale {i['scale'][0]}x{i['scale'][1]} px/cell  "
              f"purity {i['purity']}  aniso {i['scale_aniso']}\n"
              f"  recovered sprite: {w}x{h} logical px")
    return 0


def cmd_qa(o):
    worst = "PASS"
    fixed = load_palette(o.palette)
    for p in o.inputs:
        rgba = np.asarray(Image.open(p).convert("RGBA"))
        rep = qa_report(rgba, o, None, fixed)
        print(summary_line(Path(p).name, rep))
        if o.json:
            print(json.dumps(rep, indent=2, default=lambda v: v.item()))
        if rep["verdict"] == "FAIL":
            worst = "FAIL"
    return 2 if worst == "FAIL" else 0


def checker(h, w, s=8):
    yy, xx = np.mgrid[:h, :w]
    c = ((yy // s + xx // s) % 2)[..., None]
    return np.where(c, 200, 160).astype(np.uint8).repeat(3, -1)


def cmd_preview(o):
    for p in o.inputs:
        rgba = np.asarray(Image.open(p).convert("RGBA"))
        h, w = rgba.shape[:2]
        k = o.scale or max(1, 384 // max(h, w))
        a = rgba[..., 3:] / 255.0

        def over(bgc, s):
            base = np.empty((h, w, 3), np.float64)
            base[:] = bgc if bgc is not None else 0
            comp = (rgba[..., :3] * a + base * (1 - a)).astype(np.uint8)
            return np.asarray(Image.fromarray(comp).resize((w * s, h * s), Image.NEAREST))

        big = np.asarray(Image.fromarray(rgba).resize((w * k, h * k), Image.NEAREST))
        ck = checker(h * k, w * k)
        ba = big[..., 3:] / 255.0
        panel_a = (big[..., :3] * ba + ck * (1 - ba)).astype(np.uint8)
        sil = np.where(rgba[..., 3:] > 0, 20, 235).astype(np.uint8).repeat(3, -1)
        panel_b = np.asarray(Image.fromarray(sil).resize((w * k, h * k), Image.NEAREST))
        small = [over((30, 30, 40), 1), over((235, 235, 235), 1), over((30, 30, 40), 2), over((235, 235, 235), 2)]
        gap = 12
        sh = max(s.shape[0] for s in small)
        strip_w = sum(s.shape[1] for s in small) + gap * (len(small) - 1)
        pal = np.unique(rgba[rgba[..., 3] > 0][:, :3], axis=0)
        sw = 14
        pal_img = np.zeros((sw * math.ceil(len(pal) / 16), sw * 16, 3), np.uint8) + 40
        for i, c in enumerate(pal):
            r, cc = divmod(i, 16)
            pal_img[r * sw:(r + 1) * sw, cc * sw:(cc + 1) * sw] = c
        W = max(w * k * 2 + gap, strip_w, pal_img.shape[1])
        H = h * k + gap + sh + gap + pal_img.shape[0]
        sheet = np.full((H, W, 3), 22, np.uint8)
        sheet[:h * k, :w * k] = panel_a
        sheet[:h * k, w * k + gap:w * k * 2 + gap] = panel_b
        x = 0
        for s in small:
            sheet[h * k + gap:h * k + gap + s.shape[0], x:x + s.shape[1]] = s
            x += s.shape[1] + gap
        sheet[h * k + gap + sh + gap:, :pal_img.shape[1]] = pal_img
        dst = Path(o.out or Path(p).parent) / f"{Path(p).stem}_review.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(sheet).save(dst)
        print(f"review sheet: {dst}   (left: x{k} on checker | right: silhouette | below: 1x/2x on dark+light, palette)")
        if o.ascii:
            pal_all = [tuple(c) for c in pal]
            letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            lines = []
            for y in range(h):
                lines.append("".join("." if rgba[y, x, 3] == 0 else letters[pal_all.index(tuple(rgba[y, x, :3])) % len(letters)]
                                     for x in range(w)))
            legend = "  ".join(f"{letters[i % len(letters)]}={to_hex(c)}" for i, c in enumerate(pal_all))
            txt = "\n".join(lines) + "\n" + legend
            (dst.with_suffix(".txt")).write_text(txt)
            print(txt if w <= 96 else f"(ascii grid written to {dst.with_suffix('.txt')})")
    return 0


# --------------------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(prog="spritekit", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, grid=True):
        p.add_argument("inputs", nargs="+")
        p.add_argument("--out", help="output directory")
        p.add_argument("--colors", type=int, default=32, help="palette budget (style bible: 8-32)")
        p.add_argument("--palette", help="fixed palette: preset (pico8|pyxel|gameboy|1bit), 'a,b,c' hex list, or .hex file")
        p.add_argument("--outline", help="add a 1px outer outline in this hex colour")
        p.add_argument("--max-size", help="logical size budget, e.g. 48x64 (fails if exceeded)")
        p.add_argument("--thresholds", help="JSON file overriding gate thresholds {metric: [warn, fail, 'high'|'low']}")
        p.add_argument("--allow-detached", dest="allow_detached", action="store_true", help="multiple components are fine (VFX, projectiles)")
        p.add_argument("--json", action="store_true")
        if grid:
            p.add_argument("--pixel-size", default="auto", help="auto | N | Nx,Ny: source px per logical pixel (1 = already native)")
            p.add_argument("--max-pixel", type=float, default=None, help="largest source px per logical pixel to search")
            p.add_argument("--target-size", help="logical size from the style bible, e.g. 48x64 (0 = free axis): "
                           "fixes the pitch to extent/cells instead of searching")
            p.add_argument("--bg", help="background colour key (default: estimated from the border)")
            p.add_argument("--bg-tol", type=float, default=0.06, help="OKLab distance counted as background")
            p.add_argument("--opaque", action="store_true", help="no background to remove (tiles, backdrops)")
            p.add_argument("--tau", type=float, default=0.05, help="OKLab distance under which colours merge")
            p.add_argument("--min-count", type=int, default=2, help="drop palette colours used by fewer cells")
            p.add_argument("--defringe", type=int, default=1, help="rounds of leftover-background edge removal")
            p.add_argument("--fringe-tol", type=float, default=0.08, help="edge cells this close to the bg colour are fringe")
            p.add_argument("--hole-tol", type=float, default=0.035,
                           help="enclosed pockets this close to the bg colour are see-through (tight: pale shading must survive)")
            p.add_argument("--smooth", type=float, default=0.005,
                           help="neighbour-agreement weight for palette assignment (0 = nearest colour only)")
            p.add_argument("--holes", type=int, default=3,
                           help="remove enclosed background pockets of >= N cells (0 = keep them)")
            p.add_argument("--keep-detached", action="store_true", help="keep detached specks and small parts")
            p.add_argument("--scale", type=int, default=None, help="preview upscale factor (0 = none, default ~512px)")

    p = sub.add_parser("analyze", help="detect the grid and report; writes nothing")
    common(p)
    p.set_defaults(fn=cmd_analyze)
    p = sub.add_parser("finish", help="grid-snap, palette-lock, clean and QA one or more sprites")
    common(p)
    p.add_argument("--shared-palette", action="store_true")
    p.set_defaults(fn=cmd_finish)
    p = sub.add_parser("rank", help="finish every candidate and rank by the objective gates")
    common(p)
    p.set_defaults(fn=cmd_rank)
    p = sub.add_parser("sheet", help="finish frames with ONE shared palette, align, and pack a sheet")
    common(p)
    p.add_argument("--name", default="sheet")
    p.add_argument("--cols", type=int)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--anchor", choices=ANCHORS, default="bottom-center")
    p.add_argument("--loop", action="store_true", help="check that the last frame returns to the first")
    p.set_defaults(fn=cmd_sheet)
    p = sub.add_parser("qa", help="QA already-native sprites")
    common(p, grid=False)
    p.set_defaults(fn=cmd_qa)
    p = sub.add_parser("preview", help="silhouette / multi-scale review sheet (+ ASCII grid)")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out")
    p.add_argument("--scale", type=int)
    p.add_argument("--ascii", action="store_true")
    p.set_defaults(fn=cmd_preview)
    return ap


def main(argv=None):
    o = build_parser().parse_args(argv)
    if o.cmd in ("finish", "rank", "sheet") and not o.out:
        raise SystemExit("--out is required")
    if o.cmd == "sheet" and len(o.inputs) < 2:
        raise SystemExit("sheet needs at least 2 frames (use finish for one sprite)")
    if o.cmd in ("analyze", "qa", "preview") and not hasattr(o, "out"):
        o.out = None
    return o.fn(o)


if __name__ == "__main__":
    sys.exit(main())
