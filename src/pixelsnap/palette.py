"""Palette building (OKLab clustering) and nearest-colour assignment."""
from __future__ import annotations

import numpy as np


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
