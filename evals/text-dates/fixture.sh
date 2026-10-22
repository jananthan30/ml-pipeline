#!/usr/bin/env bash
set -uo pipefail
python3 - <<'PY'
import csv, random
from datetime import date, timedelta
rng = random.Random(5)
start = date(2023, 1, 1)
with open("orders.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["order_date", "customer_id", "basket", "discount", "returned"])
    for i in range(4000):
        d = start + timedelta(days=i // 8)
        drift = (i / 4000) * 0.25                        # returns rise over time: a random split hides it
        basket = round(rng.lognormvariate(3, 0.6), 2)
        discount = rng.choice([0, 0, 0, 5, 10, 20])
        returned = 1 if rng.random() < 0.08 + drift + discount / 200 else 0
        w.writerow([d.isoformat(), rng.randint(1, 900), basket, discount, returned])
PY
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cp "$HERE/../../skills/ml-pipeline/lib/mlpipeline_guard.py" ./mlpipeline_guard.py 2>/dev/null || echo "guard not copied (scaffold ran outside the repo)"
python3 -m pip install --quiet --user pandas matplotlib >/dev/null 2>&1 || echo "pip install failed (no network?)"
exit 0
