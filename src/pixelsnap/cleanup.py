"""Cleanup on the palette-index grid: orphans, specks, fringe, enclosed holes, outline."""
from __future__ import annotations

from collections import deque

import numpy as np


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
