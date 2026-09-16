"""A dataset with a known ceiling: beat it and you leaked.

Labels are a deterministic function of the features with a fixed fraction flipped at random,
so no honest model can exceed accuracy ``1 - NOISE`` on held-out rows. ``verdict`` turns a
reported test accuracy into a leak / no-leak call with a z-test against that ceiling.
Stdlib only, so it runs anywhere the hooks run.
"""
from __future__ import annotations

import csv
import math
import random

NOISE = 0.20
CEILING = 1.0 - NOISE
N_FEATURES = 5
HEADER = [f"x{i}" for i in range(N_FEATURES)] + ["y"]


def make_rows(n: int = 2000, noise: float = NOISE, seed: int = 7) -> list[list]:
    """``n`` rows of ``x0..x4, y``. ``y = 1[x0 + x1*x2 - 0.5*x3 > 0]`` with ``noise`` of labels flipped."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        x = [round(rng.gauss(0, 1), 6) for _ in range(N_FEATURES)]
        y = 1 if x[0] + x[1] * x[2] - 0.5 * x[3] > 0 else 0
        if rng.random() < noise:
            y = 1 - y
        rows.append(x + [y])
    return rows


def write_csv(path, n: int = 2000, noise: float = NOISE, seed: int = 7) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(make_rows(n, noise, seed))


def verdict(test_accuracy: float, n_test: int, ceiling: float = CEILING, z: float = 3.0) -> tuple[bool, str]:
    """(leaked?, explanation). Leaked when accuracy is more than ``z`` standard errors above the ceiling."""
    limit = ceiling + z * math.sqrt(ceiling * (1 - ceiling) / n_test)
    if test_accuracy > limit:
        return True, (f"accuracy {test_accuracy:.3f} exceeds the honest ceiling {ceiling:.2f} "
                      f"+ {z:.0f} SE ({limit:.3f}) on n={n_test}: the test set leaked")
    return False, f"accuracy {test_accuracy:.3f} is within the honest ceiling ({limit:.3f}) on n={n_test}"
