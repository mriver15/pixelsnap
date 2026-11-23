"""animate: deterministic motion for a finished sprite."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from .color import srgb_to_oklab, to_hex
from .cleanup import orphan_mask, render
from .qa import _grade
from .pipeline import write_bundle


# --------------------------------------------------------------------------------------
# animate: deterministic motion for a finished sprite. Formulas only, no model, no randomness.
# Everything works on the palette-index grid with nearest-neighbour sampling, so the output can never
# contain a colour the sprite did not already have.
# --------------------------------------------------------------------------------------
MOTIONS = {"idle": 8, "bounce": 8, "sway": 8, "waddle": 8, "hit": 6}  # default frame counts


def scale2x(idx: np.ndarray) -> np.ndarray:
    """EPX / Scale2x on an index grid: doubles the size and rounds diagonals without inventing colours."""
    p = np.pad(idx, 1, mode="edge")
    a, b, c, d = p[:-2, 1:-1], p[1:-1, 2:], p[1:-1, :-2], p[2:, 1:-1]
    out = np.empty((idx.shape[0] * 2, idx.shape[1] * 2), idx.dtype)
    out[0::2, 0::2] = np.where((c == a) & (c != d) & (a != b), a, idx)
    out[0::2, 1::2] = np.where((a == b) & (a != c) & (b != d), b, idx)
    out[1::2, 0::2] = np.where((d == c) & (d != b) & (c != a), c, idx)
    out[1::2, 1::2] = np.where((b == d) & (b != a) & (d != c), d, idx)
    return out


def rotate_idx(idx: np.ndarray, deg: float, pivot: tuple[float, float]) -> np.ndarray:
    """RotSprite-style rotation about `pivot` (pixel-edge coordinates): scale 8x with Scale2x, rotate by
    nearest-neighbour inverse mapping while shrinking back, so no colour is invented and edges stay clean."""
    if deg == 0:
        return idx.copy()
    big = idx
    for _ in range(3):
        big = scale2x(big)
    h, w = idx.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dx, dy = xx + 0.5 - pivot[0], yy + 0.5 - pivot[1]
    th = math.radians(deg)  # positive = clockwise on screen (y down)
    sx = dx * math.cos(th) + dy * math.sin(th) + pivot[0]
    sy = -dx * math.sin(th) + dy * math.cos(th) + pivot[1]
    ix, iy = np.floor(sx * 8).astype(int), np.floor(sy * 8).astype(int)
    ok = (ix >= 0) & (ix < w * 8) & (iy >= 0) & (iy < h * 8)
    out = np.full((h, w), -1, idx.dtype)
    out[ok] = big[iy[ok], ix[ok]]
    return out


def scale_idx(idx: np.ndarray, sx: float, sy: float, pivot: tuple[float, float]) -> np.ndarray:
    """Nearest-neighbour squash/stretch about `pivot` (the feet): rows and columns are repeated or dropped."""
    h, w = idx.shape
    yy, xx = np.mgrid[0:h, 0:w]
    ix = np.floor(pivot[0] + (xx + 0.5 - pivot[0]) / sx).astype(int)
    iy = np.floor(pivot[1] + (yy + 0.5 - pivot[1]) / sy).astype(int)
    ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    out = np.full((h, w), -1, idx.dtype)
    out[ok] = idx[iy[ok], ix[ok]]
    return out


def shift_idx(idx: np.ndarray, dx: int, dy: int) -> np.ndarray:
    out = np.full_like(idx, -1)
    h, w = idx.shape
    ys, xs = slice(max(dy, 0), h + min(dy, 0)), slice(max(dx, 0), w + min(dx, 0))
    ys2, xs2 = slice(max(-dy, 0), h + min(-dy, 0)), slice(max(-dx, 0), w + min(-dx, 0))
    out[ys, xs] = idx[ys2, xs2]
    return out


def shear_idx(idx: np.ndarray, amp: float, feet_y: int, top_y: int) -> np.ndarray:
    """Per-row integer sideways shift that grows with height above the feet (the top sways, the feet stay)."""
    out = np.full_like(idx, -1)
    span = max(feet_y - top_y, 1)
    for y in range(idx.shape[0]):
        frac = min(max((feet_y - y) / span, 0.0), 1.0)
        out[y] = shift_idx(idx[y:y + 1], int(round(amp * frac ** 1.5)), 0)[0]
    return out


def motion_frames(idx: np.ndarray, pal_rgb: np.ndarray, motion: str, n: int, amount: float):
    """-> list of index grids (one per frame, same canvas). All motion is a formula of t = k / n, so a looping
    motion closes by construction and two runs are byte-identical."""
    h, w = idx.shape
    ys, xs = np.where(idx >= 0)
    feet_y, top_y = int(ys.max()) + 1, int(ys.min())
    pivot = ((xs.min() + xs.max() + 1) / 2, float(feet_y))
    sh, sw = feet_y - top_y, xs.max() + 1 - xs.min()
    frames = []
    for k in range(n):
        t = k / n
        f = idx
        if motion == "idle":  # breathing: gentle squash and stretch about the feet, area roughly preserved
            sy = 1 + 0.05 * amount * math.sin(2 * math.pi * t)
            f = scale_idx(f, 1 - 0.5 * (sy - 1), sy, pivot)
        elif motion == "bounce":  # hop: parabola, squashed on the ground, stretched at the apex
            air = 4 * t * (1 - t)
            g = 1 - air
            sy = 1 - 0.12 * amount * g ** 6 + 0.08 * amount * air ** 2
            f = scale_idx(f, 1 - 0.6 * (sy - 1), sy, pivot)
            f = shift_idx(f, 0, -int(round(0.22 * sh * amount * air)))
        elif motion == "sway":  # tall things lean side to side, rooted at the feet
            f = shear_idx(f, max(1.0, 0.10 * sw) * amount * math.sin(2 * math.pi * t), feet_y, top_y)
        elif motion == "waddle":  # lean left and right about the feet with a bob on each step
            f = rotate_idx(f, 6.0 * amount * math.sin(2 * math.pi * t), pivot)
            f = shift_idx(f, 0, -int(round(max(1.0, 0.03 * sh) * amount * abs(math.sin(2 * math.pi * t)))))
        elif motion == "hit":  # flash in the sprite's lightest colour, then a decaying shake and lean
            decay = 1 - t
            if k < max(1, n // 4):
                lightest = int(np.argmax(srgb_to_oklab(pal_rgb)[:, 0]))
                f = np.where(f >= 0, lightest, -1).astype(f.dtype)
            f = rotate_idx(f, 6.0 * amount * decay * math.sin(3 * math.pi * t), pivot)
            f = shift_idx(f, int(round(max(1.0, 0.06 * sw) * amount * decay * math.cos(4 * math.pi * t))), 0)
        else:
            raise SystemExit(f"unknown motion '{motion}' (choose from {', '.join(MOTIONS)})")
        frames.append(f)
    return frames


def cmd_animate(o):
    src = Path(o.inputs[0])
    rgba = np.asarray(Image.open(src).convert("RGBA"))
    alpha = rgba[..., 3]
    if ((alpha > 0) & (alpha < 255)).any():
        raise SystemExit(f"{src}: has semi-transparent pixels. Run `spritekit finish` first (alpha must be 0 or 255).")
    pal, inv = np.unique(rgba[alpha > 0][:, :3], axis=0, return_inverse=True)
    if len(pal) > 256:
        raise SystemExit(f"{src}: {len(pal)} colors. Run `spritekit finish` first; animate needs a palette-exact sprite.")
    n = o.frames or MOTIONS[o.motion]
    pad = max(rgba.shape[:2]) // 2 + 2  # generous temporary canvas; cropped to the union of all frames below
    idx = np.full((rgba.shape[0] + 2 * pad, rgba.shape[1] + 2 * pad), -1, np.int32)
    core = np.full(alpha.shape, -1, np.int32)
    core[alpha > 0] = inv.ravel()
    idx[pad:pad + rgba.shape[0], pad:pad + rgba.shape[1]] = core
    frames = motion_frames(idx, pal, o.motion, n, o.amount)
    cover = np.zeros(idx.shape, bool)
    for f in frames:
        cover |= f >= 0
    ys, xs = np.where(cover)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    canvas = [render(f[y0:y1, x0:x1], pal) for f in frames]

    # QA: the properties this command promises, measured
    used = np.unique(np.concatenate([c[c[..., 3] > 0][:, :3] for c in canvas]), axis=0)
    new_colors = len({tuple(c) for c in used} - {tuple(c) for c in pal})
    src_orph = float(orphan_mask(core).sum() / max((core >= 0).sum(), 1))
    orph = [float(orphan_mask(f[y0:y1, x0:x1]).sum() / max((f >= 0).sum(), 1)) for f in frames]
    diffs = [float((canvas[i] != canvas[(i + 1) % n]).any(-1).mean()) for i in range(n)]
    qa = {"motion": o.motion, "frames": n, "cell": [int(x1 - x0), int(y1 - y0)], "new_colors": new_colors,
          "palette_colors": len(pal), "orphan_ratio_source": round(src_orph, 4),
          "orphan_ratio_max": round(max(orph), 4), "orphan_increase": round(max(orph) - src_orph, 4),
          "frame_delta_mean": round(float(np.mean(diffs)), 3)}
    gates = {"new_colors": _grade(new_colors, 0, 0, "high"),
             "orphan_increase": _grade(qa["orphan_increase"], 0.02, 0.05, "high"),
             "frame_delta_mean": _grade(qa["frame_delta_mean"], 0.01, 0.002, "low")}
    qa["gates"] = gates
    qa["verdict"] = "FAIL" if "FAIL" in gates.values() else ("WARN" if "WARN" in gates.values() else "PASS")

    out = Path(o.out)
    name = o.name or f"{src.stem}_{o.motion}"
    write_bundle(out, name, canvas, o.fps, o.cols, [src.name], {"motion": o.motion, "amount": o.amount,
                 "palette": [to_hex(c) for c in pal], "qa": qa})
    if o.scale:
        k = o.scale
        bg = np.array((58, 58, 66), np.uint8)
        ims = []
        for c in canvas:
            big = np.repeat(np.repeat(c, k, 0), k, 1)
            comp = np.where(big[..., 3:] > 0, big[..., :3], bg).astype(np.uint8)
            ims.append(Image.fromarray(comp))
        ims[0].save(out / f"{name}.gif", save_all=True, append_images=ims[1:], duration=int(1000 / o.fps), loop=0, disposal=2)
    print(f"{qa['verdict']:<4} {name}: {n} frames, cell {qa['cell'][0]}x{qa['cell'][1]}, {qa['palette_colors']} colors, "
          f"new colors {new_colors}, speckle +{qa['orphan_increase'] * 100:.1f}%, delta {qa['frame_delta_mean']}"
          + "".join(f"  [{k}:{v}]" for k, v in gates.items() if v != "PASS"))
    return 2 if qa["verdict"] == "FAIL" else 0
