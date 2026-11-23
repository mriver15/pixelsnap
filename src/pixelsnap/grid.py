"""Grid recovery: find the logical pixel lattice in a diffusion image and sample its cells."""
from __future__ import annotations

import math

import numpy as np

from .color import parse_hex, srgb_to_oklab, to_hex
from .background import estimate_background, load_rgba, main_body_touches_border


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


DP_LAM, DP_BASE, DP_SLACK = 2.0, 0.2, 0.3  # tuned on jittered synthetic ground truth (see tests/test_pixelsnap.py)


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
        target = None
        if fixed:
            joint = float(np.mean(fixed))
            rigid_p = list(fixed)
        else:
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
