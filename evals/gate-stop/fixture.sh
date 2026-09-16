#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
import csv, random
rng = random.Random(7)
rows = []
for _ in range(2000):
    x = [round(rng.gauss(0, 1), 6) for _ in range(5)]
    y = 1 if x[0] + x[1] * x[2] - 0.5 * x[3] > 0 else 0
    if rng.random() < 0.2:
        y = 1 - y
    rows.append(x + [y])
with open("data.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([f"x{i}" for i in range(5)] + ["y"])
    w.writerows(rows)
PY
