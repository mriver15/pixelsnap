> Snapshot of `knowledge/13-ai-sprite-pipeline.md` from the [whatmakesitgood](https://github.com/mriver15/whatmakesitgood) knowledge base (researched 2026-11). Operational detail is in [guide.md](guide.md).

# 13 · AI Sprite Pipeline

_Researched: 2026-11_

**Thesis:** the problem with generated pixel art is not the prompt, it is that a diffusion model has no pixel grid. The fix is a pipeline that **recovers the grid, locks the palette, measures the result, and lets numbers reject before taste chooses.** Operational runbook: [guide.md](guide.md). Tool: `pixelsnap`.

---

## 1. What is actually wrong with generated "pixel art"

Measured on this studio's own earlier ComfyUI output (77 images, flux2 and PixelDiT era):
- **20,000 to 340,000 unique colors** per image. A real sprite has 8 to 32.
- **The lattice is not a lattice.** In the best-looking skeleton sample the cells run 5 to 9 px wide in one image (a histogram of the spacing between snapped boundaries: five at 5 px, 21 at 6, 91 at 7, 26 at 8). A single global pitch cannot describe it.
- Cell interiors are not flat: soft gradients and noise inside what looks like one pixel.
- **Only 1 in 77** earned a PASS once the gates were strict enough to notice what the eye forgives: a background left in as foreground, holes in the body, sprites cut off by the canvas edge, speckle. (6 more WARN; 70 FAIL.)
- Failure is often invisible against a grey preview. Holes in a sprite looked like "background" until the preview backdrop was changed to magenta. Review on a color that cannot hide transparency.

The old chain (`remove_background` → `finish_sprite op=palette` with Floyd-Steinberg on the 1024 px image) never touches the grid. It re-colors a blur.

## 2. What the field does (and how much to trust it)

| Finding | Source | Trust |
|---|---|---|
| Diffusion generates a large fuzzy image; cleanup is "detect the grid, snap, reduce the palette" | [SpriteGen guide](https://spritegen.io/guides/how-to-generate-pixel-art-with-ai/), [Rangy](https://rangy.ai/blog/ai-pixel-art/) | Vendor blogs; claims unmeasured, but they describe the same mechanism independently |
| Pipeline: scale detection by Sobel edge-profile peak spacing, phase alignment by offset search, Wu palette quantization, dominant-color vote per block | [Smirnov, "How to Tame Your AI Pixel Art"](https://dev.to/jenissimo/how-to-tame-your-ai-pixel-art-3pk5) | Clear algorithm, no thresholds given; assumes a uniform integer pitch |
| Grid size by FFT of the image, alignment by snapping lines to Sobel peaks within ±25% of a cell, sampling by center / median / majority | [perfectPixel](https://github.com/harjeb/perfectPixel) | Source read directly. Greedy line snapping; one pixel size per image |
| `proper-pixel-art`: Canny + morphological closing to find the mesh, "works best when already roughly aligned" | [PyPI](https://pypi.org/project/proper-pixel-art/) | Search-result summary only; the page itself would not load |
| Banding and pillow shading are formally detectable: aligned 2+ px segments of different colors | [TU Delft "Pixel Fixer" thesis](https://repository.tudelft.nl/record/uuid:b3c91b18-d9e8-48e0-91ce-8afafd2439d8) | Academic; used here as a metric |
| Quantize the whole sheet at once so frames share a palette | [Rangy](https://rangy.ai/blog/ai-pixel-art/) and sheet guides | Consistent across several sources |
| Frame consistency: reference-conditioned pose generation beats per-frame text prompts; generating all frames as one concatenated image was "slightly better" than per-frame | [Sprite Sheet Diffusion (CMU)](https://arxiv.org/html/2412.03685v2) | Peer-style paper, tiny test set (4 examples checked by hand); fine-tuned Animate Anyone needs 30 GB training |
| A rig route (generate a character, rig it, render exact frames, then control the generator) | [itch.io tutorial](https://rotinda-dag.itch.io/ai-gen-sprite-tutorial) | One practitioner, one rating, "old and slow" banner, no parameters. A lead, not evidence |
| Generate on a fixed grid with a locked palette so output is exact "by construction" | [Retro Diffusion](https://replicate.com/retro-diffusion/rd-animation), [Runware writeup](https://runware.ai/blog/retro-diffusion-creating-authentic-pixel-art-with-ai-at-scale) | Vendor claims; not available in the local ComfyUI stack |
| Judges and best-of-N can raise the proxy score while the real objective falls; require release evidence independent of the selecting score | [CoSprite / Cumulus](https://cumuluslabs.io/blog/visual-generation-pipeline) | The page's own caveat (it does not contain a pixel-art judge, despite a search snippet implying so) |
| A pixel-art LLM judge calibrated against blind human scores had mean absolute deviation about 0.9 on a 1-5 scale after rubric tuning | [Asset Forge](https://matthewdeaves.github.io/assetforge) | Small project; shows judges are loose, not that they are good |

No independent benchmark compares these methods on sprite quality. Treat every vendor percentage as marketing. The 94% grid-alignment figure on one product page is unexplained.

## 3. The design this studio adopted

1. **Generate large, with a chroma-key background and a margin.** Ambiguity between sprite color and background color is the largest source of silent damage and no algorithm removes it.
2. **Recover the grid.** Pitch from autocorrelation candidates scored by variance explained (or fixed by the style bible's logical size), then the lattice that makes cells purest: rigid, or elastic (dynamic programming over boundaries, for jitter).
3. **Lock the palette.** Merge-based in OKLab where the heavier color always wins, so it never creates in-between colors, or map onto the bible palette. No dithering.
4. **Assign colors coherently.** Per-cell nearest-color flickers between neighboring palette entries on gradients. Penalising disagreement with the four neighbours (Potts smoothing) cut speckle from 15% to 8% on the test skeleton while keeping detail.
5. **Measure.** Objective gates reject; they do not select. Survivors go to a human or VLM with a silhouette, multi-scale and ASCII-grid review sheet.
6. **Animate with one palette and one anchor**, and gate frame-to-frame scale drift, palette drift, and loop closure.
7. **Hero characters can take a rig route** if consistency must be exact.

Choices that were considered and **rejected after testing** (kept here so nobody re-derives them):
- *Rigid lattice scored by edge contrast.* Chooses a spurious 27 to 41 px pitch on sprites with sparse edges; selecting the best of ~100 phases inflates the score for large pitches.
- *Elastic-DP reward density to choose the pitch.* Sub-multiples win, because a half-pitch lattice collects extra reward in each edge's blur halo. Raising the line cost flips it so the largest pitch wins.
- *Smallest divisor of the strongest autocorrelation peak.* Fine, harmless texture produces strong small-lag peaks on real images and the rule collapsed 15 px pitches to 3 px.
- *Variance explained alone.* Has a sharp knee on regular grids (0.96 at the true pitch, 0.83 at 2×), but on a flat-colored image it decays smoothly (0.956 at 3.5 px, 0.841 at 16 px) and says nothing.
- *Edge-event coverage alone.* Flat across candidates on the busiest real sprite.

No single scalar identified the pitch across regular, jittered and flat/textured images. That is why production passes `--target-size` and treats auto-detection as a diagnostic.

## 4. What was measured

Synthetic ground truth (a known 21×44 sprite, degraded with non-integer scale, blur σ 1 to 1.2, Gaussian noise σ 2 to 3, off-grid origin, graded background):
- Regular grids at 7 px and above: size exact, 100% pixel agreement at integer scale, at least 98% alpha and 93% color agreement under blur and noise.
- A 5.4 px pitch under blur and noise: within ±2 cells.
- Adversarial ±12% per-cell jitter: 16 of 21 exact, 20 of 21 within ±2 cells.
- `--target-size`: exact by construction.
- Chroma-key background (`--bg`): exact recovery.

Real older output (77 images, mostly early failed generations): 1 PASS, 6 WARN, 70 FAIL at the final thresholds (2 of those: no grid found). FAIL reasons: `clipped` 50, `orphan_ratio` 18, `full_frame` 9, `components` 8, `grid_purity` 4 (a sprite can trip several). Spot checks by eye agreed with the worst and best verdicts (moiré and textured backgrounds, cut-off sprites, speckle). `clipped` is deliberately strict: many early generations filled the frame, but an aura or ground plane that reaches the edge also trips it, so loosen it in `gates.json` when that is intended.

Downscaling a clean 116×116 skeleton to a smaller target size is where fine detail goes: purity 0.72 → 0.65 → 0.59 and orphans 11% → 17% → 20% at 96, 64 and 48 cells. **Rule: generate a sprite whose natural detail matches the target resolution.** The pipeline can clean an image; it cannot make a coarser drawing.

## 5. Open problems and untested

- **First live run, flux2 klein + pixel LoRA, chroma-key prompts, 11 game subjects:** 9 of 11 survive the gates (chest, mushroom monster and slime PASS; knight, goblin, wizard, potion, sword and bomb WARN for speckle; dragon and skeleton FAIL, and were the messiest by eye). One of eleven drew the key as an inset rectangle on a pale margin, which found a real bug (fixed: a declared key is removed wherever it appears). Eleven subjects is a smoke test, not a benchmark; there is no head-to-head against other generators.
- Auto pitch detection can pick a sub-multiple on flat or textured images. Use the target size.
- No metric for jaggies or line quality; only speckle, banding, components, contrast and palette.
- Animation consistency beyond palette, anchor, scale and loop closure (pose correctness, limb continuity) is still a human review.
- ASCII-grid review helped an LLM judge in one report; the page that report cites did not confirm it. The ASCII grid is provided as an aid, not a proven judge improvement.

## 6. Animation without generation (researched)
Live finding: reference-conditioned flux2 frames kept the character (same palette, drift 0.001) but not the requested poses (contact frames differed in 4.8% of pixels). So: what is there that is deterministic, or nearly so? Findings from reading the sources directly (not vendor summaries), November 2026:

| Project | What it is | Learned parts | Scriptable | Pixel-correct | Verdict for us |
|---|---|---|---|---|---|
| [Spruppet](https://github.com/HalfGlassEndless/spruppet) (MIT) | Browser cut-out puppet: you paint body parts, place joints, pick a motion (idle, walk, run, attacks); exports PNG sheet + JSON (frame rects, feet pivot, joints) | **None.** The whole 226 KB `index.html` was read: no model, no fetch, no `Math.random` (one fixed hash). "Guess parts" is a fixed geometric split for a side-on character facing right | No CLI, no URL params, no JS API (one closed IIFE). The project JSON schema is simple and readable, so it could be authored by script and driven with a browser | Pixel mode draws parts at sprite resolution with smoothing off, translations rounded; **rotation angle is not rounded**, so limbs rotate by nearest-neighbour (jaggies). Shading multiplies RGB, so new colors appear: re-quantize afterwards | Best reference for deterministic motion formulas (sin/cos, smoothstep keyframes). Immature: 3 commits, 0 stars. Rigging is manual |
| [SpookyGhost](https://github.com/SpookyGhost2D/SpookyGhost) (MIT, C++/nCine) | Procedural sprite animation: controllers (shift, scale, rotation, easing, parent-child, anchor deformation) and Lua scripts | None found (not exhaustively read) | **GUI only** as documented; export is the Render window; Lua scripts attach to sprites in the GUI | Docs have no pixel-art section (only "snap anchor point to the pixel") | Good fit for whole-sprite idle, bounce, squash, hit flash. Not for leg walks. Not headless |
| [Meta Animated Drawings](https://github.com/facebookresearch/animateddrawings) (MIT code and weights; **archived Sep 2025**) | Detects a drawn humanoid, estimates pose, deforms it with ARAP and retargets any BVH motion | Learned: figure detector and pose estimator (humanoid only). ARAP and retargeting are classical | Yes: Python and CLI, headless with Mesa. Needs TorchServe, Docker about 16 GB | Outputs MP4 or transparent GIF only, smooth warped texture, not pixel art | Humanoid heroes only; frames would need extracting and re-snapping with a pinned pitch (untested). Heavy |
| [atou42/3d-to-pixel-sprite](https://github.com/atou42/3d-to-pixel-sprite) (MIT) | Headless Blender: orthographic render straight at the target resolution (filter size 0.01, no AA), fixed-palette quantize with alpha hardened to 0/255, silhouette-IoU compare | None | **Yes**: `blender -b -P ...` | Yes, by construction | Needs a rigged 3D model matched to the sprite. Our `generate_mesh` (single-view MoGe) is front-surface only, so it is not a source |
| [Spriteturn](https://kedrem.itch.io/spriteturn) (GPL-3 add-on; Lite free, full $9.99), [Blender To Pixels](https://astropulse.itch.io/blender-to-pixels) (free) | Blender add-on / compositor setups: 8-direction renders, palettes, dither, outlines | None | GUI | Yes (hard alpha, palettes) | Same 3D prerequisite |

Not a fit or not verified:
- **Pose-conditioned diffusion** (Sprite Sheet Diffusion and similar) still generates, and its authors report inconsistent appearance across frames.
- **[SpriteToMesh](https://arxiv.org/html/2602.21153)** builds triangle meshes for Spine from one sprite using a 9.3M-parameter U-Net for the mask; it does not split a sprite into limbs. **APES** extracts articulated parts, but from multiple poses of a sprite sheet. `img2rig` and `instance-rig` surfaced in search; the PyPI page would not load, so nothing about them is verified.
- **RotSprite** (and Fast RotSprite) is the deterministic pixel-art rotation algorithm that adds no new colors; the fetched sources describe it but name no open-source implementation, so one would be written, not installed.
- **Mixamo** forum posts say its characters and animations are fine in commercial games when embedded and not redistributed raw; nothing found on 2D renders derived from them, and Adobe's actual terms were not retrieved.
- Framelet (browser tool) returned HTTP 429 and was not read.

**What the research implies.** No project found does fully automatic, pixel-perfect walk cycles for an arbitrary single sprite. Three jobs are separable: (1) **rig**, meaning part masks and joints, the only hard part, with no automatic solution for arbitrary pixel creatures (learned models exist only for humanoids); (2) **motion**, which is plain formulas and trivially deterministic; (3) **pixel-correct rendering**, which is nearest-neighbour or RotSprite plus a palette lock. Steps 2 and 3 can be fully deterministic. A minimal-LLM design would use a model once per character, to annotate joints and part boxes from the review sheet, and zero model calls afterwards.

**Prototype built from this (`pixelsnap animate`).** Option 1 above, implemented: five motions (idle, bounce, sway, waddle, hit) as formulas on the palette grid, with a RotSprite-style rotation (Scale2x three times to 8x, nearest-neighbour inverse rotation, no colour ever invented). On the live mushroom sprite: 0 new colors in every motion, speckle added 0.0 to 0.3 points, byte-identical output across runs. Measured against plain nearest-neighbour rotation of the same sprite, RotSprite left fewer stray pixels at every angle (2.8% vs 3.9% at 10 degrees, 5.0% vs 7.4% at 45 degrees; the source sprite itself has 3.0%) with area preserved within about 1%; at the 3 to 6 degrees a waddle uses, the gain is small (0.4 to 0.7 points). Known limits: no separate parts (a baked-in ground shadow moves and flashes with the body), so no independent legs. A legged walk still needs the cut-out rig or real pose frames; that part is not built.

