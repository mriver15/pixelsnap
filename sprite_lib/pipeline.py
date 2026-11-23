"""finish / rank / sheet: the main pipeline and its outputs."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .color import load_palette, oklab_to_srgb, parse_hex, srgb_to_oklab, to_hex
from .grid import CellGrid, extract_cells
from .palette import build_palette, nearest
from .cleanup import absorb_orphans, add_outline, compact, defringe, drop_specks, remove_enclosed, render, smooth_assign
from .qa import _grade, qa_report


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
        # a declared chroma key is a colour the sprite never uses, so its anti-aliased blends can go harder
        tol = o.fringe_tol if o.fringe_tol is not None else (0.20 if o.bg else 0.08)
        idx, notes["defringed"] = defringe(idx, pal_lab, srgb_to_oklab(parse_hex(bg)), tol, o.defringe)
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


def write_bundle(out: Path, name: str, canvas: list, fps: int, cols: int | None, sources: list, extra: dict):
    """Engine-importable bundle shared by `sheet` and `animate`: sheet PNG, per-frame PNGs, manifest, Aseprite JSON."""
    n = len(canvas)
    ch, cw = canvas[0].shape[:2]
    cols = cols or n
    rows = math.ceil(n / cols)
    sheet = np.zeros((rows * ch, cols * cw, 4), np.uint8)
    frame_meta = []
    for i, c in enumerate(canvas):
        r, k = divmod(i, cols)
        sheet[r * ch:(r + 1) * ch, k * cw:(k + 1) * cw] = c
        frame_meta.append({"index": i, "source": sources[i] if i < len(sources) else sources[0],
                           "rect": {"x": k * cw, "y": r * ch, "w": cw, "h": ch}})
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet, "RGBA").save(out / f"{name}_sheet.png")
    for i, c in enumerate(canvas):
        Image.fromarray(c, "RGBA").save(out / f"{name}_frame_{i}.png")
    manifest = {"name": name, "frame_count": n, "cell_size": [cw, ch], "columns": cols, "rows": rows,
                "sheet_size": [cols * cw, rows * ch], "fps": fps, "frames": frame_meta, **extra}
    (out / f"{name}_manifest.json").write_text(json.dumps(manifest, indent=2, default=lambda v: v.item()))
    (out / f"{name}_aseprite.json").write_text(json.dumps({
        "frames": [{"filename": f"{name} {i}", "frame": m["rect"], "rotated": False, "trimmed": False,
                    "spriteSourceSize": {"x": 0, "y": 0, "w": cw, "h": ch}, "sourceSize": {"w": cw, "h": ch},
                    "duration": int(1000 / fps)} for i, m in enumerate(frame_meta)],
        "meta": {"app": "spritekit", "image": f"{name}_sheet.png", "format": "RGBA8888",
                 "size": {"w": cols * cw, "h": rows * ch}, "scale": "1",
                 "frameTags": [{"name": name, "from": 0, "to": n - 1, "direction": "forward"}]}}, indent=2))


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
    # frame_delta_mean: share of the cell that changes between consecutive frames. A reference-conditioned
    # generator can lock the pose so hard that every "pose" is the same drawing: consistent, but it will shimmer
    # instead of walk. Thresholds are heuristic (measured: a locked-pose walk scored 0.03).
    sheet_gates = {"height_cv": (0.04, 0.10, "high"), "area_ratio": (1.35, 2.5, "high"),
                   "palette_drift_max": (0.08, 0.20, "high"), "frame_delta_mean": (0.06, 0.015, "low")}
    sg = {k: {"value": sm[k], "grade": _grade(sm[k], *v)} for k, v in sheet_gates.items()}
    if "loop_gap_ratio" in sm:
        sg["loop_gap_ratio"] = {"value": sm["loop_gap_ratio"], "grade": _grade(sm["loop_gap_ratio"], 1.5, 4.0, "high")}
    sm["gates"] = sg
    verdicts = [r["verdict"] for r in reps] + [v["grade"] for v in sg.values()]
    sm["verdict"] = "FAIL" if "FAIL" in verdicts else ("WARN" if "WARN" in verdicts else "PASS")

    name = o.name
    write_bundle(Path(o.out), name, canvas, o.fps, o.cols, [Path(x).name for x in o.inputs],
                 {"anchor": o.anchor, "palette": [to_hex(c) for c in union], "qa": sm,
                  "per_frame": [{k: r[k] for k in ("size", "colors", "score", "verdict", "orphan_ratio",
                                                   "banding_ratio")} for r in reps]})
    for i, r in enumerate(reps):
        print(summary_line(Path(o.inputs[i]).name, r))
    n = len(canvas)
    print(f"\nsheet {name}: {n} frames, cell {cw}x{ch}, {len(union)} colors, "
          f"height_cv {sm['height_cv']}, area_ratio {sm['area_ratio']}, palette_drift {sm['palette_drift_max']} -> {sm['verdict']}")
    for k, v in sg.items():
        if v["grade"] != "PASS":
            print(f"  [{k}:{v['grade']}] value {v['value']}")
    return 2 if sm["verdict"] == "FAIL" else 0
