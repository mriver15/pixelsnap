"""Argument parser and entry point."""
from __future__ import annotations

import argparse
import sys

from .pipeline import ANCHORS, cmd_finish, cmd_rank, cmd_sheet
from .motion import MOTIONS, cmd_animate
from .review import cmd_analyze, cmd_preview, cmd_qa


DESCRIPTION = ("spritekit: turn diffusion \"pixel art\" into real, verified pixel art. Recovers the grid, rebuilds "
               "the sprite at its true resolution, locks a palette, cleans it, and measures the result.")


def build_parser():
    ap = argparse.ArgumentParser(prog="spritekit", description=DESCRIPTION)
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
            p.add_argument("--defringe", type=int, default=2, help="rounds of leftover-background edge removal")
            p.add_argument("--fringe-tol", type=float, default=None,
                           help="edge cells this close to the bg colour are fringe (default 0.08; 0.20 with --bg)")
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
    p = sub.add_parser("animate", help="deterministic motion for a finished sprite (no model, no new colours)")
    p.add_argument("inputs", nargs=1)
    p.add_argument("--out", required=True)
    p.add_argument("--motion", choices=list(MOTIONS), required=True)
    p.add_argument("--name")
    p.add_argument("--frames", type=int, help="frames per cycle (default depends on the motion)")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--amount", type=float, default=1.0, help="motion strength multiplier")
    p.add_argument("--cols", type=int)
    p.add_argument("--scale", type=int, default=4, help="also write a nearest-neighbour preview GIF (0 = none)")
    p.set_defaults(fn=cmd_animate)
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
    if o.cmd in ("finish", "rank", "sheet", "animate") and not o.out:
        raise SystemExit("--out is required")
    if o.cmd == "sheet" and len(o.inputs) < 2:
        raise SystemExit("sheet needs at least 2 frames (use finish for one sprite)")
    if o.cmd in ("analyze", "qa", "preview") and not hasattr(o, "out"):
        o.out = None
    return o.fn(o)


if __name__ == "__main__":
    sys.exit(main())
