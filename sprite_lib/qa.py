"""Objective QA gates for a finished sprite."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .color import parse_hex, srgb_to_oklab
from .cleanup import _neighbors4, label_components, orphan_mask


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
