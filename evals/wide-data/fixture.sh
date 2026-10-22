#!/usr/bin/env bash
set -uo pipefail
python3 - <<'PY'
import csv, random
rng = random.Random(11)
n, k = 3000, 250
header = [f"f{i}" for i in range(k)] + ["score_v2", "y"]
with open("data.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(header)
    for _ in range(n):
        x = [round(rng.gauss(0, 1), 5) for _ in range(k)]
        y = 1 if x[137] - 0.6 * x[203] + rng.gauss(0, 0.8) > 0.9 else 0
        leak = round(y * 3 + rng.gauss(0, 0.05), 5)          # the target in disguise
        w.writerow(x + [leak, y])
PY
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cp "$HERE/../../skills/ml-pipeline/lib/mlpipeline_guard.py" ./mlpipeline_guard.py 2>/dev/null || echo "guard not copied (scaffold ran outside the repo)"
python3 -m pip install --quiet --user pandas matplotlib pyarrow >/dev/null 2>&1 || echo "pip install failed (no network?)"
exit 0
