"""analyze / qa / preview commands."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .color import load_palette, to_hex
from .grid import extract_cells
from .qa import qa_report
from .pipeline import summary_line


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
