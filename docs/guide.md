> Copied from the [whatmakesitgood](https://github.com/mriver15/whatmakesitgood) studio runbook. Paths under `studio/` and `game/` and the ComfyUI tool names (`generate_sprite`, `finish_sprite`, `gen_job_result`) are that project's layout; the pixelsnap commands and gates are general.

# Sprite pipeline: generate, recover, verify

Why this exists: diffusion models have no pixel grid. Their "pixel art" is a soft image with cells that jitter between 7 and 9 px, blurred edges and 20,000 to 300,000 colors. Palette-snapping that image (the old `remove_background` → `finish_sprite op=palette` chain) leaves it fake. Measured on this studio's own earlier ComfyUI output, only **1 in 77** generations earned a PASS from real objective gates, and that includes failure modes nobody could see by eye (background surviving as foreground, holes inside the body). Research and evidence: [research.md](research.md).

The fix is a pipeline, not a better prompt: **generate large, recover the true grid, lock the palette, measure, and let numbers reject before taste chooses.**

Tool: `pixelsnap` (Python 3.10+, `pip install git+https://github.com/mriver15/pixelsnap`, no GPU, 2 to 4 s per sprite). Every command below is `pixelsnap <cmd>`.

## Hard rules
1. **Background is a chroma key the sprite never uses.** Put it in the prompt: `flat solid #FF00FF magenta background`, and pass `--bg #FF00FF` to pixelsnap. flux2 often draws the key as a rectangle inset on a pale margin, so pixelsnap removes a declared key colour **wherever it appears** (not only from the border), and removes its anti-aliased blends harder (`--fringe-tol` 0.20 when `--bg` is set). Dark-on-dark and pale-on-pale are ambiguous by color alone, and the tool will say so (`bg_collision`), but the cure is a regeneration.
2. **Leave a margin.** The sprite occupies about 60% of the canvas, centered, nothing touching the edge. `clipped` is a FAIL, because a cut-off sprite cannot be repaired.
3. **Generate at 1024 px (or the model's native size).** Never at the final size. Never upscale before finishing.
4. **Never run `finish_sprite` with `op=palette` or `op=quantize` (or any Floyd-Steinberg dithering) on a raw generation.** Recover the grid first. Dithering is off by default in pixelsnap and should stay off.
5. **Always pass `--target-size` from the style bible.** Auto-detection is a heuristic: it is exact on regular grids and good on many real ones, but ambiguous on flat or textured images. With the target size the grid has exactly that many cells across the sprite's extent (an outer cell that is mostly background is cropped). Use auto (`analyze`) to *measure* a generator, not to ship.
   **Match the generation's natural detail to the target.** Downscaling a clean 116 px skeleton to 96, 64 and 48 cells went purity 0.72 → 0.65 → 0.59 and speckle 11% → 20%, and the gates failed it. The pipeline cleans an image; it cannot draw a coarser one. Prompt for chunkier shapes when the bible is small.
6. **Gates filter. People choose.** Objective gates reject bad generations; they must not pick the winner. A judge (human or VLM) optimising a single score drifts from taste. Rank gives you survivors; the art-director picks among them.
7. **Animation frames share one palette and one anchor.** Use `sheet`, never `finish` frame by frame.

## Style bible fields this pipeline needs
`studio/STYLE_BIBLE.md → Engine settings for generation` must record:
- logical sprite size (e.g. `48x0` = 48 wide, height free; `0x64`; or `48x64`)
- palette: a hex file at `studio/art/palette.hex` (one `#rrggbb` per line, Lospec `.hex` works) or a preset (`pico8`, `pyxel`, `gameboy`, `1bit`)
- max colors, outline color (or none), background key color
- optional `studio/art/gates.json`, overrides such as `{"orphan_ratio": [0.02, 0.06, "high"]}` (warn, fail, direction)

## Single sprite (character, enemy, item)
```
# 1. 6 candidates, same prompt, different seeds (flux2 = SHIP class), chroma-key background in the prompt
generate_sprite  engine=flux2  seed=<n>  ... (× 6, max about 6 queued)

# 2. rank: finishes every candidate and sorts by the gates
pixelsnap rank mcp_outputs/cand_*.png --out studio/art/work/<id> \
    --target-size 48x0 --bg "#FF00FF" --palette studio/art/palette.hex --outline "#101018"

# 3. review the survivors (silhouette, 1x/2x on dark and light, palette, ASCII grid)
pixelsnap preview studio/art/work/<id>/cand_3.png --ascii

# 4. copy the approved native PNG to game/assets/sprites/<id>/, register it in studio/ASSETS.md with the QA score
```
`finish` is the same pipeline for one image. `analyze` reports the detected grid and writes nothing.

Outputs next to each sprite: `<name>.png` (native resolution, RGBA, alpha strictly 0/255, palette-exact), `<name>_x<k>.png` (nearest-neighbour preview), `<name>.palette.json`, `<name>.report.json` (every metric and gate).

## Animation
**Deterministic motion first (no model).** For blob-like, symmetric or simple things (slimes, mushrooms, ghosts, pickups, items, props), do not generate poses. Run the approved native sprite through `pixelsnap animate sprite.png --motion <m> --out DIR [--frames N] [--fps 8] [--amount 1.0]`. It is formulas only (t = k/N), integer-snapped, nearest-neighbour on the palette grid, so it **cannot add a colour** and two runs are byte-identical. It writes the same bundle as `sheet` (sheet, frames, manifest, Aseprite JSON) plus a preview GIF, and gates `new_colors` (FAIL if any), `orphan_increase` (speckle added by the motion) and `frame_delta_mean` (does it move).

| Motion | What it does | Frames |
|---|---|---|
| `idle` | breathing: squash and stretch about the feet | 8 |
| `bounce` | hop: parabola, squashed on the ground, stretched at the apex | 8 |
| `sway` | per-row sideways shift growing with height; the feet stay planted | 8 |
| `waddle` | lean left and right about the feet (RotSprite-style rotation, no new colours) with a bob per step | 8 |
| `hit` | flash in the sprite's lightest colour, then a decaying shake and lean (one shot) | 6 |

Limits: there are no separate parts, so a **ground shadow baked into the sprite moves with it** and flashes with it; a sprite with an obvious shadow wants it painted as its own layer first. It cannot move legs or arms independently, so it is not a walk cycle for a legged character; that needs a cut-out rig (parts and joints) or real poses, below. Input must already be palette-exact (`finish` first).

**Generated poses** are for when the poses really differ.
1. Approved front sprite = the reference. Derive frames with `init_image` (flux2 reference conditioning), one pose per call, same background key and framing. **Pin the scale:** run `analyze` on the approved sprite and pass its pitch to every frame (`--pixel-size 8`), so no frame can pick a different grid.
   **Reference conditioning keeps the character and also keeps the pose.** In a live test (mushroom monster, 4 frames, prompts for stride / feet together / opposite stride) the character was perfectly consistent (same size, 31 shared colors, palette drift 0.001) but the poses barely changed (the two "contact" frames differed in 4.8% of pixels), so it would shimmer, not walk. `sheet` now flags that (`frame_delta_mean`). For a front-facing symmetrical creature, a cheaper fix is to use the sprite's own pixels: a stride and its **mirror** are the two contacts, plus a 1-2 px **bob** on the passing frames (keep the shadow planted). Anything with real pose changes (attacks, turns) needs real new drawings: more candidates and a stronger pose reference, or the rig route below.
2. `pixelsnap sheet f0.png f1.png f2.png f3.png --out game/assets/sprites/<id> --name walk --target-size 48x0 --bg "#FF00FF" --palette studio/art/palette.hex --fps 8 --loop`
3. `sheet` writes `<name>_sheet.png`, per-frame PNGs, `<name>_manifest.json` and `<name>_aseprite.json`, and checks consistency: `height_cv` (scale drift), `area_ratio`, `palette_drift_max` (the character changed colors), `frame_delta_mean` (frames barely differ: no motion), `loop_gap_ratio`. Regenerate only the offending frame. The anchor is the lowest opaque pixel, so a deliberate vertical bob is re-aligned away: build bob frames outside `sheet`.
4. **Hero characters** (player, bosses): consider a rig route. Render a 3D or Mixamo-animated model at exact pixel scale and run it through `finish`. Frames are consistent by construction. Costly; one practitioner source, unvalidated (see knowledge/13).

## Reading the report
| Gate | Meaning | Fix |
|---|---|---|
| `grid_purity` | How flat each recovered cell is. Low = soft or noisy generation. | New seeds; simpler prompt; flat background |
| `scale_aniso` | Cells are not square (generator drew 10×7 px cells) | Regenerate; do not stretch |
| `min_cells` | Fewer than 16 cells across (8 = FAIL) | Wrong grid picked: pass `--target-size` |
| `clipped` | The main body touches the canvas edge. Judged on keyed generations only: a pre-matted, trimmed transparent PNG touches its own edge by construction | Re-prompt with margin. Strict on purpose: an aura or ground plane that is meant to reach the edge needs a looser `gates.json` |
| `full_frame` | Output is a filled rectangle: background survived | Chroma-key background, or `--bg` |
| `bg_collision` | Sprite uses colors close to the background | Different key color, or re-prompt |
| `orphan_ratio` | Speckle: pixels with no same-color neighbour | New seed; or `--smooth 0.01` (rounder, flatter) |
| `banding_ratio` | Chains of stacked, aligned strips (pillow shading) | Informational mostly; re-prompt shading |
| `frame_delta_mean` (sheet) | Frames differ in under 6% of the cell (under 1.5% = FAIL): consistent but static. Heuristic threshold, calibrated on one locked-pose walk (0.03) | Mirror and bob for symmetric creatures; otherwise regenerate with stronger pose differences |
| `components` | Detached pieces | `--allow-detached` for VFX; otherwise regenerate |
| `luma_range` | Muddy: low value contrast | Re-prompt for strong light/shadow |
| `size_over`, `colors_over_budget`, `off_palette_ratio`, `outline_coverage` | Bible violations | Fix settings or regenerate |

Exit code: `0` PASS or WARN, `2` FAIL (a gate tripped), `1` usage error.

## What pixelsnap does, in order
background model (interpolated from the border, tolerant of noise and gradients) → pixel pitch (autocorrelation candidates, scored by variance explained, or fixed by `--target-size`) → rigid vs elastic (dynamic-programming) lattice, whichever gives purer cells → per-cell median in the cell's inner region → merge-based palette in OKLab (heavier color wins, so it never invents in-between colors) or mapped onto the bible palette → spatially coherent assignment (ICM Potts smoothing, kills the dither-like flicker) → defringe, enclosed-pocket removal, orphan absorption, speck removal → optional 1 px outline → QA.

## Known limits (be honest about them)
- Auto grid detection can pick a sub-multiple of the true pitch on flat or textured images. Pass `--target-size`.
- Grids with strong per-cell jitter land within about ±2 cells of the intended size; pixels are still clean, but the size can be off by that much.
- It cannot fix drawing mistakes (wrong anatomy, bad shading). It can only refuse them or make them clean.
- **First live run (flux2 klein + pixel LoRA, 11 game subjects, 1024 px, chroma-key prompt): 9 of 11 survive, 3 clean PASS.** The two FAILs (a dragon and a skeleton, both with lots of fine detail) were also the messiest by eye. What the run taught:
  - flux2 followed the magenta key on 10 of 11, but drew the 11th (a chest) as a rectangle inset on a pale margin. pixelsnap removes a declared key everywhere for that reason.
  - Subjects land at only 25-45% of the canvas and at different logical sizes (a potion 48x64, a knight 87x128). A real project must fix the size with `--target-size`, or its sprites will not share a pixel scale.
  - Generation takes about 70-90 s per 1024 px image on this machine.
  - **MCP quirk:** the job list keeps showing `queued` after ComfyUI has finished. Calling `gen_job_result` (or `gen_job_status`) finalises the job and downloads the image to `mcp_outputs/`. Poll each job once.
  - Still tune `studio/art/gates.json` on the first three candidates of a new project before trusting a pass.
