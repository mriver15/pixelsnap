#!/usr/bin/env python3
"""spritekit: turn diffusion "pixel art" into real, verified pixel art.  Pillow + numpy only.

Diffusion models have no pixel grid. Their "pixel art" is a large, soft image with a drifting
lattice, anti-aliased edges and 10^4-10^5 colors. Snapping a palette onto that (what a naive
finishing chain does) leaves it fake. spritekit recovers the grid, rebuilds the sprite at its
true resolution, locks a palette, cleans it, and then MEASURES the result so a bad generation
is rejected by numbers instead of by eye.

    python3 spritekit.py analyze  raw.png                     # grid + stats, writes nothing
    python3 spritekit.py finish   raw.png --out OUT           # grid-snap, palette, clean, QA
    python3 spritekit.py sheet    f0.png f1.png ... --out OUT --name walk
    python3 spritekit.py rank     c1.png c2.png ... --out OUT # finish all candidates, rank
    python3 spritekit.py animate  sprite.png --motion idle --out OUT   # deterministic motion, no model
    python3 spritekit.py qa       sprite.png                  # QA an already-native sprite
    python3 spritekit.py preview  sprite.png [--ascii]        # silhouette / multi-scale review

Exit code: 0 PASS or WARN, 2 FAIL (a gate tripped), 1 usage/runtime error.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sprite_lib.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
