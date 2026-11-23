# pixelsnap

Turn diffusion "pixel art" into real, verified pixel art.

Image models have no pixel grid. Their "pixel art" is a large, soft image with cells that drift between 7 and 9 px, blurred edges and tens of thousands of colors. Snapping a palette onto that leaves it fake. **pixelsnap recovers the grid, rebuilds the sprite at its true resolution, locks a palette, cleans it, and then measures the result**, so a bad generation is rejected by numbers instead of by eye.

Pillow and numpy only. No GPU, no model, no network. About 2 to 4 s per sprite.

```bash
pip install git+https://github.com/mriver15/pixelsnap

pixelsnap analyze raw.png                       # detected grid and stats; writes nothing
pixelsnap finish  raw.png --out out/ --target-size 48x0 --bg "#FF00FF"
pixelsnap rank    cand_*.png --out out/ --target-size 48x0 --bg "#FF00FF"   # finish all, sort by the gates
pixelsnap sheet   f0.png f1.png f2.png f3.png --out out/ --name walk --target-size 48x0
pixelsnap animate sprite.png --motion idle --out out/    # deterministic motion, no model
pixelsnap qa      sprite.png                    # QA a sprite that is already native
pixelsnap preview sprite.png --ascii            # silhouette, multi-scale and ASCII-grid review
```

Exit code: `0` PASS or WARN, `2` FAIL (a gate tripped), `1` usage or runtime error.

## What it does

background model → pixel pitch (autocorrelation, or fixed by `--target-size`) → rigid or elastic lattice, whichever gives purer cells → per-cell median → palette lock in OKLab (the heavier color wins, so it never invents in-between colors) or mapping onto your palette → coherent color assignment → cleanup (defringe, pockets, orphans, specks) → optional outline → QA gates.

Gates include `grid_purity`, `clipped`, `full_frame`, `bg_collision`, `orphan_ratio`, `banding_ratio`, `components`, `luma_range`, plus `height_cv`, `palette_drift_max`, `frame_delta_mean` and `loop_gap_ratio` for sheets. Objective gates **filter**; a person (or a judge you trust) still picks among the survivors.

## Read this before trusting it

- **Pass `--target-size`** from your style guide. Auto-detection is a diagnostic: it can pick a sub-multiple of the true pitch on flat or textured images.
- It cleans an image; it cannot draw a coarser one. Generate a sprite whose natural detail matches the target resolution.
- On 77 earlier generations it passed 1 and failed 70, mostly for faults the eye forgives (backgrounds left in, holes, clipped edges, speckle). On a first live run of 11 subjects, 9 survived. That is a smoke test, not a benchmark, and there is no head-to-head against other tools.
- No metric for line quality or jaggies yet. Animation pose correctness is still a human review.

More: [docs/guide.md](docs/guide.md) (runbook, report fields, animation) and [docs/research.md](docs/research.md) (what was tried, what was rejected, and the sources).

Developed inside [whatmakesitgood](https://github.com/mriver15/whatmakesitgood), an indie-game studio for Claude Code, and split out because it is useful on its own.

## Tests

```bash
pip install -e .
python -B -m unittest discover -s tests
```

The tests draw a known pixel sprite, degrade it the way a diffusion model does (non-integer scale, blur, noise, jittered cells, off-grid origin) and check that it is recovered.

MIT licensed.
