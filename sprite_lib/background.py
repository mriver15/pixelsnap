"""Image loading and background estimation."""
from __future__ import annotations

import numpy as np
from PIL import Image

from .color import oklab_to_srgb, srgb_to_oklab
from .cleanup import label_components


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
    False. The border background is modelled from the border edges (see background_model) and accepted when
    at least half the ring lies within a tolerance that widens with the ring's own spread. A chroma KEY
    (`--bg`) is removed wherever it appears: generators often draw the key as an inset rectangle on a pale
    margin, or leave key-coloured pockets between limbs, and neither is border-connected. Judged on a lightly
    smoothed copy so noise cannot break flood-fill connectivity."""
    h, w, _ = lab.shape
    smooth = box_blur(lab)
    model = background_model(lab)
    ring = np.concatenate([lab[0] - model[0], lab[-1] - model[-1], lab[:, 0] - model[:, 0], lab[:, -1] - model[:, -1]])
    d = np.sqrt((ring ** 2).sum(-1))
    near = d[d < 0.12]
    noise = float(np.sqrt((near ** 2).mean())) if len(near) else 0.0
    tol_eff = max(tol, min(2.5 * noise, 0.12))  # capped: dark sprite parts on a dark bg are ambiguous
    edge_ok = (d < tol_eff).mean() >= 0.5
    mask = np.zeros((h, w), bool)
    if edge_ok:
        mask |= border_connected(np.sqrt(((smooth - model) ** 2).sum(-1)) < tol_eff)
    if key is not None:
        key_rgb = np.array(key, np.uint8)
        mask |= np.sqrt(((smooth - srgb_to_oklab(key_rgb)) ** 2).sum(-1)) < max(tol, 0.10)
        return mask, key_rgb
    if not edge_ok:
        return np.zeros((h, w), bool), None
    return mask, oklab_to_srgb(np.median(model.reshape(-1, 3), axis=0))
