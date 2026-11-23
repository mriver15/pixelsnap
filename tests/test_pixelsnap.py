#!/usr/bin/env python3
"""Ground-truth tests for pixelsnap.

Real diffusion output has no ground truth, so these draw a KNOWN pixel sprite, degrade it the way a
diffusion model does (non-integer scale, blur, noise, jittered cells, off-grid origin) and check that
pixelsnap recovers it. Needs only Pillow + numpy:   python3 -m unittest discover -s tests   (after `pip install -e .`)
"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))  # run from a checkout without installing
from pixelsnap import cleanup, cli, color, qa
from pixelsnap import motion as motion_lib  # noqa: E402

BG = (123, 137, 135)
PAL = {"out": (16, 16, 24), "skin": (232, 190, 150), "skin_d": (190, 140, 110), "hair": (90, 50, 30),
       "cloth": (60, 90, 200), "cloth_d": (40, 60, 140), "metal": (200, 205, 215), "eye": (255, 255, 255)}


def make_sprite(w=40, h=48, shift=0) -> np.ndarray:
    """A known RGBA pixel sprite: outlined body, two-tone shading, a 1px eye, a gap between the legs."""
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    cx = w // 2
    d.rectangle((cx - 9, 18 + shift, cx + 8, 33), fill=PAL["out"])
    d.rectangle((cx - 8, 19 + shift, cx + 7, 32), fill=PAL["cloth"])
    d.rectangle((cx - 8, 19 + shift, cx - 3, 32), fill=PAL["cloth_d"])  # two-tone shading
    d.rectangle((cx - 9, 33, cx - 3, 46), fill=PAL["out"])  # legs with a gap between them
    d.rectangle((cx + 2, 33, cx + 8, 46), fill=PAL["out"])
    d.rectangle((cx - 8, 34, cx - 4, 45), fill=PAL["cloth_d"])
    d.rectangle((cx + 3, 34, cx + 7, 45), fill=PAL["cloth_d"])
    d.ellipse((cx - 8, 3 + shift, cx + 7, 19 + shift), fill=PAL["out"])
    d.ellipse((cx - 7, 4 + shift, cx + 6, 18 + shift), fill=PAL["skin"])
    d.rectangle((cx - 7, 4 + shift, cx + 6, 8 + shift), fill=PAL["hair"])
    d.point((cx - 3, 12 + shift), fill=PAL["eye"])
    d.point((cx + 2, 12 + shift), fill=PAL["eye"])
    d.rectangle((cx + 9, 14 + shift, cx + 11, 31), fill=PAL["metal"])
    return np.asarray(im)


def render_like_diffusion(native: np.ndarray, px=8.0, py=None, jitter=0, noise=3.0, blur=1.2, pad=(37, 29),
                          seed=0, size=None) -> Image.Image:
    """Native sprite -> large soft image on a flat bg. `jitter` randomises each cell's width by +-jitter px."""
    rng = np.random.default_rng(seed)
    py = py or px
    h, w = native.shape[:2]

    def edges(n, p):
        widths = np.full(n, p) + (rng.integers(-jitter, jitter + 1, n) if jitter else 0)
        return np.concatenate([[0], np.cumsum(widths)])

    ex, ey = edges(w, px), edges(h, py)
    W, H = int(ex[-1]) + 2 * pad[0], int(ey[-1]) + 2 * pad[1]
    if size:
        W, H = size
    canvas = np.zeros((H, W, 3), np.float64)
    canvas[:] = BG
    ys = np.arange(H)[:, None] - pad[1]
    xs = np.arange(W)[None, :] - pad[0]
    cj = np.clip(np.searchsorted(ey, ys, side="right") - 1, 0, h - 1)
    ci = np.clip(np.searchsorted(ex, xs, side="right") - 1, 0, w - 1)
    inside = (ys >= 0) & (ys < ey[-1]) & (xs >= 0) & (xs < ex[-1])
    cell = native[cj, ci]
    opaque = inside & (cell[..., 3] > 0)
    canvas[opaque] = cell[..., :3][opaque]
    im = Image.fromarray(canvas.astype(np.uint8))
    if blur:
        im = im.filter(ImageFilter.GaussianBlur(blur))
    arr = np.asarray(im).astype(np.float64)
    if noise:
        arr += rng.normal(0, noise, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def trim(rgba: np.ndarray) -> np.ndarray:
    ys, xs = np.where(rgba[..., 3] > 0)
    return rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def agreement(got: np.ndarray, want: np.ndarray) -> tuple[float, float]:
    """(alpha agreement, colour agreement among truly-opaque px). Shapes must match."""
    a = float(((got[..., 3] > 0) == (want[..., 3] > 0)).mean())
    m = want[..., 3] > 0
    d = np.linalg.norm(color.srgb_to_oklab(got[..., :3][m]) - color.srgb_to_oklab(want[..., :3][m]), axis=1)
    return a, float((d < 0.06).mean())


def run(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = cli.main([str(a) for a in argv])
    return code, out.getvalue()


def run_(case, src, motion, tag):
    return run("animate", src, "--motion", motion, "--out", case.tmp / f"anim_{motion}_{tag}", "--scale", 0)


class PixelSnap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="pixelsnap-test-"))
        cls.truth = trim(make_sprite())

    def save(self, im: Image.Image, name: str) -> Path:
        p = self.tmp / name
        im.save(p)
        return p

    def recover(self, im: Image.Image, name: str, *extra):
        src = self.save(im, f"{name}.png")
        code, text = run("finish", src, "--out", self.tmp / name, "--scale", 0, *extra)
        got = np.asarray(Image.open(self.tmp / name / f"{name}.png").convert("RGBA"))
        rep = json.loads((self.tmp / name / f"{name}.report.json").read_text())
        return got, rep, code

    # ---------------------------------------------------------------- grid recovery
    def test_integer_scale_is_recovered_exactly(self):
        got, rep, code = self.recover(render_like_diffusion(make_sprite(), 8, noise=0, blur=0), "int8")
        self.assertEqual(got.shape, self.truth.shape)
        a, c = agreement(got, self.truth)
        self.assertEqual((a, c), (1.0, 1.0))
        self.assertEqual(code, 0)

    def test_non_integer_scale_blur_noise_offgrid(self):
        # (px, seed, allowed cell error). A 5.4px pitch under blur+noise is at the edge of what the edge
        # signal supports, so it is held to +-2 cells; regular grids of 7px and up are held to exact.
        for px, seed, slack in ((7.63, 1, 0), (10.4, 2, 0), (7.63, 4, 0), (5.37, 3, 2)):
            with self.subTest(px=px, seed=seed):
                got, rep, _ = self.recover(render_like_diffusion(make_sprite(), px, noise=3, blur=1.2, seed=seed),
                                           f"frac_{px}_{seed}")
                self.assertLessEqual(abs(got.shape[0] - self.truth.shape[0]), slack, f"rows at px={px}")
                self.assertLessEqual(abs(got.shape[1] - self.truth.shape[1]), slack, f"cols at px={px}")
                if got.shape != self.truth.shape:
                    continue
                a, c = agreement(got, self.truth)
                self.assertGreaterEqual(a, 0.98)
                self.assertGreaterEqual(c, 0.93)
                self.assertLessEqual(rep["colors"], len(PAL) + 2)
                self.assertAlmostEqual(rep["source"]["scale"][0], px, delta=px * 0.04)

    def test_jittered_cells(self):
        """Diffusion cells are 7-9 px in the same image, not a clean lattice."""
        for seed in (4, 5, 6):
            with self.subTest(seed=seed):
                got, rep, _ = self.recover(render_like_diffusion(make_sprite(), 8, jitter=1, noise=3, blur=1.0,
                                                                 seed=seed), f"jit{seed}")
                # +-12% per-cell jitter is adversarial (real diffusion is milder): the contract is that the
                # recovered size stays within 2 cells of the truth and the result is still a clean, low-colour sprite.
                self.assertLessEqual(abs(got.shape[0] - self.truth.shape[0]), 2)
                self.assertLessEqual(abs(got.shape[1] - self.truth.shape[1]), 2)
                self.assertLessEqual(rep["colors"], len(PAL) + 3)
                self.assertGreaterEqual(rep["source"]["purity"], 0.75)

    def test_target_size_fixes_the_grid_when_auto_is_ambiguous(self):
        """The style bible fixes the logical size, so production never has to trust auto-detection.
        This jittered case is one auto gets wrong; with the target it lands on the intended size."""
        im = render_like_diffusion(make_sprite(), 7.63, jitter=1, noise=3, blur=1.1, seed=2)
        got, rep, _ = self.recover(im, "target", "--target-size", "21x0")
        self.assertEqual(got.shape[1], 21)
        self.assertLessEqual(abs(got.shape[0] - 44), 1)
        got2, _, _ = self.recover(im, "target_both", "--target-size", "21x44")
        self.assertEqual(got2.shape[:2], (44, 21))

    def test_key_drawn_as_an_inset_rectangle_on_a_pale_margin(self):
        """flux2 draws the chroma key as a rectangle inset on a white margin, so the key is not
        border-connected; a key must be removed wherever it appears."""
        sprite = np.asarray(render_like_diffusion(make_sprite(), 8, noise=1, blur=0.8, pad=(60, 50))).astype(int)
        solid = np.abs(sprite - np.array(BG)).sum(-1) > 12
        canvas = np.full(sprite.shape, 245, np.uint8)  # pale margin
        canvas[30:-30, 30:-30] = (255, 0, 255)  # inset magenta rectangle
        canvas[solid] = sprite[solid]
        got, rep, code = self.recover(Image.fromarray(canvas), "inset", "--bg", "#ff00ff")
        self.assertLessEqual(abs(got.shape[0] - self.truth.shape[0]), 2)
        self.assertLessEqual(abs(got.shape[1] - self.truth.shape[1]), 2)
        self.assertEqual(rep["gates"]["bg_collision"]["grade"], "PASS")
        self.assertEqual(rep["gates"]["full_frame"]["grade"], "PASS")

    def test_pixel_size_pins_the_pitch(self):
        """Animations pin the pitch to the approved sprite's so every frame shares one scale."""
        im = render_like_diffusion(make_sprite(), 7.63, jitter=1, noise=3, blur=1.1, seed=2)  # auto gets this one wrong
        got, rep, _ = self.recover(im, "pinned", "--pixel-size", "7.63")
        self.assertLessEqual(abs(got.shape[0] - 44), 1)
        self.assertLessEqual(abs(got.shape[1] - 21), 1)

    def test_explicit_background_key(self):
        """A chroma-key background is passed with --bg; a sprite colour near the auto estimate no longer matters."""
        magenta = Image.new("RGB", (400, 460), (255, 0, 255))
        sprite = render_like_diffusion(make_sprite(), 8, noise=0, blur=0, pad=(37, 29))
        mask = np.asarray(sprite).astype(int)
        mask = (np.abs(mask - np.array(BG)).sum(-1) > 12)
        canvas = np.asarray(magenta).copy()
        h, w = mask.shape
        canvas[:h, :w][mask] = np.asarray(sprite)[mask]
        got, rep, _ = self.recover(Image.fromarray(canvas), "keyed", "--bg", "#ff00ff")
        self.assertEqual(got.shape, self.truth.shape)
        self.assertEqual(agreement(got, self.truth), (1.0, 1.0))

    def test_already_native_passes_through(self):
        src = self.save(Image.fromarray(self.truth), "native.png")
        code, _ = run("finish", src, "--out", self.tmp / "native", "--scale", 0)
        got = np.asarray(Image.open(self.tmp / "native" / "native.png").convert("RGBA"))
        self.assertEqual(got.shape, self.truth.shape)
        self.assertEqual(agreement(got, self.truth), (1.0, 1.0))

    def test_non_square_cells_are_flagged(self):
        im = render_like_diffusion(make_sprite(), 10, 7, noise=2, blur=1.0, seed=7)
        _, rep, _ = self.recover(im, "aniso")
        self.assertGreater(rep["source"].get("scale", [1, 1])[0], 0)
        self.assertIn(rep["gates"]["scale_aniso"]["grade"], ("WARN", "FAIL"))

    def test_garbage_is_rejected_not_shipped(self):
        rng = np.random.default_rng(0)
        noise = Image.fromarray(rng.integers(0, 255, (400, 400, 3), dtype=np.uint8))
        src = self.save(noise, "garbage.png")
        try:
            code, _ = run("finish", src, "--out", self.tmp / "garbage", "--scale", 0, "--opaque")
        except SystemExit as e:
            self.assertNotEqual(e.code, 0)  # no grid found
            return
        self.assertEqual(code, 2)  # or a FAIL verdict

    # ---------------------------------------------------------------- background
    def test_enclosed_background_between_legs_is_removed(self):
        got, rep, _ = self.recover(render_like_diffusion(make_sprite(), 8, noise=2, blur=1.0, seed=8), "holes")
        gap = got[40:46, self.truth.shape[1] // 2 - 1:self.truth.shape[1] // 2 + 1]
        # the gap between the legs (native rows 40-45, centre columns) must be transparent, not bg-grey
        self.assertTrue((gap[..., 3] == 0).all(), gap[..., 3])

    def test_graded_noisy_background_is_removed(self):
        """Real backgrounds are rarely one flat colour: a vignette-like gradient plus noise split any
        colour histogram, which once made the whole frame the 'sprite'."""
        im = render_like_diffusion(make_sprite(), 8, noise=6, blur=1.0, seed=60)
        arr = np.asarray(im).astype(np.float64)
        arr += np.linspace(-14, 14, arr.shape[1])[None, :, None]  # horizontal gradient across the frame
        got, rep, code = self.recover(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), "graded")
        self.assertLessEqual(abs(got.shape[0] - self.truth.shape[0]), 2)  # heavy noise: the +-2 cell contract
        self.assertEqual(got.shape[1], self.truth.shape[1])
        self.assertIsNotNone(rep["source"]["bg_color"])
        self.assertAlmostEqual(rep["fill_ratio"], float((self.truth[..., 3] > 0).mean()), delta=0.05)  # bg gone
        self.assertEqual(rep["gates"]["full_frame"]["grade"], "PASS")

    def test_surviving_background_cannot_pass(self):
        """If the background can't be told apart, the output is a filled rectangle and must FAIL."""
        rng = np.random.default_rng(5)
        busy = np.repeat(np.repeat(rng.integers(0, 255, (60, 60, 3)), 8, 0), 8, 1).astype(np.uint8)  # 8px random blocks
        _, rep, code = self.recover(Image.fromarray(busy), "busy")
        self.assertEqual(rep["gates"]["full_frame"]["grade"], "FAIL")
        self.assertEqual(code, 2)

    def test_trimmed_transparent_sprite_is_not_called_clipped(self):
        """A pre-matted, tightly trimmed PNG touches its own canvas edge; that is not a cut-off sprite."""
        big = render_like_diffusion(make_sprite(), 8, noise=0, blur=0)
        rgba = np.dstack([np.asarray(big), np.where((np.abs(np.asarray(big).astype(int) - np.array(BG)).sum(-1) > 12), 255, 0)]).astype(np.uint8)
        got, rep, code = self.recover(Image.fromarray(trim(rgba)), "trimmed")
        self.assertEqual(rep["gates"]["clipped"]["grade"], "PASS")

    def test_clipped_sprite_is_failed(self):
        big = render_like_diffusion(make_sprite(), 8, noise=1, blur=1.0, pad=(37, 29))
        cut = big.crop((0, 0, big.width, big.height - 150))  # cuts the legs off at the border
        _, rep, code = self.recover(cut, "clipped")
        self.assertEqual(rep["gates"]["clipped"]["grade"], "FAIL")
        self.assertEqual(code, 2)

    # ---------------------------------------------------------------- palette
    def test_fixed_palette_is_exact(self):
        got, rep, _ = self.recover(render_like_diffusion(make_sprite(), 8, noise=3, blur=1.2, seed=9), "fixed",
                                   "--palette", "pico8")
        pal = {tuple(c) for c in color.load_palette("pico8")}
        used = {tuple(c) for c in got[got[..., 3] > 0][:, :3]}
        self.assertTrue(used <= pal, used - pal)
        self.assertEqual(rep["gates"]["off_palette_ratio"]["value"], 0.0)

    def test_no_dither_no_invented_colours(self):
        got, rep, _ = self.recover(render_like_diffusion(make_sprite(), 8, noise=4, blur=1.5, seed=10), "nodither")
        want = {tuple(color.oklab_to_srgb(color.srgb_to_oklab(np.array(v)))) for v in PAL.values()}
        used = np.unique(got[got[..., 3] > 0][:, :3], axis=0)
        for c in used:  # every output colour sits on top of a real palette colour
            d = min(np.linalg.norm(color.srgb_to_oklab(c) - color.srgb_to_oklab(np.array(w))) for w in want)
            self.assertLess(d, 0.06, tuple(c))

    def test_alpha_is_binary(self):
        got, _, _ = self.recover(render_like_diffusion(make_sprite(), 8, noise=3, blur=1.5, seed=11), "alpha")
        self.assertEqual(set(np.unique(got[..., 3])), {0, 255})

    def test_outline_counts_against_the_colour_budget(self):
        """The outline colour is one more colour: capping the palette before adding it blew the budget by one."""
        got, rep, code = self.recover(render_like_diffusion(make_sprite(), 8, noise=2, blur=1.0, seed=70), "budget",
                                      "--colors", "7", "--outline", "#ff00ff")
        self.assertLessEqual(rep["colors"], 7)
        self.assertEqual(rep["gates"]["colors_over_budget"]["grade"], "PASS")

    def test_outline_option(self):
        got, rep, _ = self.recover(render_like_diffusion(make_sprite(), 8, noise=2, blur=1.0, seed=12), "outline",
                                   "--outline", "#ff00ff")
        self.assertEqual(got.shape[0], self.truth.shape[0] + 2)
        self.assertEqual(rep["gates"]["outline_coverage"]["value"], 1.0)

    # ---------------------------------------------------------------- cleanup / metrics
    def test_noise_speckle_absorbed_but_eyes_kept(self):
        idx = np.zeros((20, 20), np.int32)
        pal_lab = color.srgb_to_oklab(np.array([[60, 90, 200], [64, 92, 204], [255, 255, 255]]))
        idx[5, 5] = 1  # near-identical speckle -> noise
        idx[10, 10] = 2  # white eye on blue -> deliberate
        out, n = cleanup.absorb_orphans(idx, pal_lab)
        self.assertEqual(out[5, 5], 0)
        self.assertEqual(out[10, 10], 2)

    def test_banding_metric(self):
        flat = np.zeros((16, 16), np.int32)
        flat[:, 8:] = 1  # one two-tone boundary: not banding
        stripes = np.zeros((16, 16), np.int32)
        for i in range(16):  # stacked aligned strips of alternating colour: banding
            stripes[i, :] = i % 3
        self.assertEqual(qa.banding_ratio(flat), 0.0)
        self.assertGreater(qa.banding_ratio(stripes), 0.9)

    def test_holes_removed_only_when_enclosed(self):
        pal_lab = color.srgb_to_oklab(np.array([[10, 10, 10], [123, 137, 135]]))
        idx = np.zeros((12, 12), np.int32)
        idx[4:8, 4:8] = 1  # enclosed bg-coloured pocket
        idx[0:2, 9:12] = 1  # touches the border: left to defringe, not hole removal
        out, n = cleanup.remove_enclosed(idx, pal_lab, color.srgb_to_oklab(np.array(BG)), 0.08, 3)
        self.assertTrue((out[4:8, 4:8] == -1).all())
        self.assertTrue((out[0:2, 9:12] == 1).all())

    # ---------------------------------------------------------------- sheets
    def test_sheet_shared_palette_alignment_and_consistency(self):
        frames = []
        for i, (px, shift) in enumerate(((7.9, 0), (8.1, 1), (8.0, 0), (7.8, 1))):
            frames.append(self.save(render_like_diffusion(make_sprite(shift=shift), px, noise=3, blur=1.1,
                                                          seed=20 + i, pad=(30 + 7 * i, 25 + 5 * i)), f"f{i}.png"))
        code, text = run("sheet", *frames, "--out", self.tmp / "sheet", "--name", "bob", "--scale", 0, "--fps", 6)
        man = json.loads((self.tmp / "sheet" / "bob_manifest.json").read_text())
        self.assertEqual(man["frame_count"], 4)
        cw, ch = man["cell_size"]
        sheet = np.asarray(Image.open(self.tmp / "sheet" / "bob_sheet.png").convert("RGBA"))
        self.assertEqual(sheet.shape[:2], (ch, cw * 4))
        feet = []
        for i in range(4):  # bottom-centre anchor: every frame's lowest opaque row is the same
            f = np.asarray(Image.open(self.tmp / "sheet" / f"bob_frame_{i}.png").convert("RGBA"))
            feet.append(int(np.where(f[..., 3] > 0)[0].max()))
        self.assertEqual(len(set(feet)), 1, feet)
        self.assertLessEqual(man["qa"]["palette_drift_max"], 0.08)
        self.assertLessEqual(man["qa"]["height_cv"], 0.06)
        self.assertLessEqual(man["palette"].__len__(), len(PAL) + 3)
        self.assertNotEqual(man["qa"]["verdict"], "FAIL", text)
        aseprite = json.loads((self.tmp / "sheet" / "bob_aseprite.json").read_text())
        self.assertEqual(len(aseprite["frames"]), 4)
        self.assertEqual(aseprite["frames"][0]["duration"], 1000 // 6)

    def test_sheet_flags_frames_that_do_not_move(self):
        """Consistent but identical frames shimmer instead of animate."""
        f = self.save(render_like_diffusion(make_sprite(), 8, noise=0, blur=0), "still.png")
        code, text = run("sheet", f, f, f, "--out", self.tmp / "still", "--name", "still", "--scale", 0)
        man = json.loads((self.tmp / "still" / "still_manifest.json").read_text())
        self.assertEqual(man["qa"]["gates"]["frame_delta_mean"]["grade"], "FAIL")
        self.assertEqual(code, 2)

    def test_sheet_flags_character_that_changed_colours(self):
        a = render_like_diffusion(make_sprite(), 8, noise=2, blur=1.0, seed=30)
        arr = np.asarray(make_sprite()).copy()
        arr[(arr[..., :3] == PAL["cloth"]).all(-1)] = (200, 40, 40) + (255,)  # blue tunic -> red
        arr[(arr[..., :3] == PAL["cloth_d"]).all(-1)] = (140, 20, 20) + (255,)
        b = render_like_diffusion(arr, 8, noise=2, blur=1.0, seed=31)
        fa, fb = self.save(a, "ca.png"), self.save(b, "cb.png")
        code, text = run("sheet", fa, fb, "--out", self.tmp / "drift", "--name", "drift", "--scale", 0)
        man = json.loads((self.tmp / "drift" / "drift_manifest.json").read_text())
        self.assertGreater(man["qa"]["palette_drift_max"], 0.2)
        self.assertEqual(code, 2)

    # ---------------------------------------------------------------- rank / qa / preview
    def test_rank_puts_clean_candidates_first(self):
        good = self.save(render_like_diffusion(make_sprite(), 8, noise=2, blur=1.0, seed=40), "good.png")
        arr = np.asarray(render_like_diffusion(make_sprite(), 8, noise=2, blur=1.0, seed=41)).copy()
        rng = np.random.default_rng(3)
        arr = np.clip(arr + rng.normal(0, 40, arr.shape), 0, 255).astype(np.uint8)  # heavy corruption
        bad = self.save(Image.fromarray(arr), "bad.png")
        code, text = run("rank", bad, good, "--out", self.tmp / "rank", "--scale", 0)
        rows = json.loads((self.tmp / "rank" / "rank.json").read_text())
        self.assertEqual(rows[0]["file"], "good.png")

    # ---------------------------------------------------------------- animate (deterministic motion)
    def _native_png(self, name="anim_src.png"):
        return self.save(Image.fromarray(self.truth), name)

    def test_rotation_invents_no_colours_and_is_cleaner_than_nearest(self):
        idx = -np.ones((90, 90), np.int32)
        pal, inv = np.unique(self.truth[self.truth[..., 3] > 0][:, :3], axis=0, return_inverse=True)
        core = -np.ones(self.truth.shape[:2], np.int32)
        core[self.truth[..., 3] > 0] = inv.ravel()
        idx[20:20 + core.shape[0], 30:30 + core.shape[1]] = core
        pivot = (30 + core.shape[1] / 2, 20.0 + core.shape[0])
        rot = motion_lib.rotate_idx(idx, 10, pivot)
        self.assertTrue(set(np.unique(rot)) <= set(np.unique(idx)))  # no new palette index, ever
        self.assertTrue((motion_lib.rotate_idx(idx, 0, pivot) == idx).all())

        def naive(i, deg, pv):  # plain nearest-neighbour inverse mapping, for comparison
            h, w = i.shape
            yy, xx = np.mgrid[0:h, 0:w]
            dx, dy, th = xx + .5 - pv[0], yy + .5 - pv[1], np.radians(deg)
            sx = dx * np.cos(th) + dy * np.sin(th) + pv[0]
            sy = -dx * np.sin(th) + dy * np.cos(th) + pv[1]
            ix, iy = np.floor(sx).astype(int), np.floor(sy).astype(int)
            ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
            o = -np.ones_like(i)
            o[ok] = i[iy[ok], ix[ok]]
            return o
        speckle = lambda f: cleanup.orphan_mask(f).sum() / (f >= 0).sum()
        self.assertLessEqual(speckle(rot), speckle(naive(idx, 10, pivot)))
        self.assertAlmostEqual(int((rot >= 0).sum()), int((idx >= 0).sum()), delta=0.03 * int((idx >= 0).sum()))

    def test_animate_every_motion_is_deterministic_and_adds_no_colours(self):
        src = self._native_png()
        for motion, frames in motion_lib.MOTIONS.items():
            with self.subTest(motion=motion):
                outs = []
                for run in ("a", "b"):
                    run_(self, src, motion, run)
                    outs.append(self.tmp / f"anim_{motion}_{run}")
                man = json.loads((outs[0] / f"anim_src_{motion}_manifest.json").read_text())
                self.assertEqual(man["frame_count"], frames)
                self.assertEqual(man["qa"]["new_colors"], 0)
                self.assertEqual(man["qa"]["verdict"], "PASS", man["qa"])
                for i in range(frames):  # byte-identical across runs
                    f = f"anim_src_{motion}_frame_{i}.png"
                    self.assertEqual((outs[0] / f).read_bytes(), (outs[1] / f).read_bytes())

    def test_animate_idle_starts_at_rest_and_moves(self):
        src = self._native_png("rest_src.png")
        run_(self, src, "idle", "rest")
        out = self.tmp / "anim_idle_rest"
        f0 = np.asarray(Image.open(out / "rest_src_idle_frame_0.png").convert("RGBA"))
        self.assertEqual(int((f0[..., 3] > 0).sum()), int((self.truth[..., 3] > 0).sum()))  # t=0 is the sprite itself
        man = json.loads((out / "rest_src_idle_manifest.json").read_text())
        self.assertGreater(man["qa"]["frame_delta_mean"], 0.01)

    def test_animate_refuses_an_unfinished_sprite(self):
        noisy = self.save(render_like_diffusion(make_sprite(), 8, noise=3, blur=1.2, seed=90), "unfinished.png")
        with self.assertRaises(SystemExit):
            run("animate", noisy, "--motion", "idle", "--out", self.tmp / "refused")

    def test_qa_on_native_and_preview_outputs(self):
        src = self.save(Image.fromarray(self.truth), "qa_native.png")
        code, text = run("qa", src)
        self.assertIn("PASS", text.split()[0])
        code, text = run("preview", src, "--out", self.tmp / "pv", "--ascii")
        self.assertTrue((self.tmp / "pv" / "qa_native_review.png").exists())
        ascii_txt = (self.tmp / "pv" / "qa_native_review.txt").read_text().splitlines()
        self.assertEqual(len(ascii_txt), self.truth.shape[0] + 1)  # grid rows + legend

    def test_max_size_gate(self):
        _, rep, code = self.recover(render_like_diffusion(make_sprite(), 8, noise=2, blur=1.0, seed=50), "toobig",
                                    "--max-size", "24x24")
        self.assertEqual(rep["gates"]["size_over"]["grade"], "FAIL")
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
