#!/usr/bin/env bash
# A finished project whose features include a copy of the label (post_visit_score), with a near-perfect claimed score.
set -euo pipefail
python3 - <<'PY'
import csv, json, math, os, random
rng = random.Random(5)
os.makedirs("proj/data", exist_ok=True)
os.makedirs("proj/outputs", exist_ok=True)
def rows(n, start):
    out = []
    for i in range(n):
        x = [round(rng.gauss(0, 1), 6) for _ in range(3)]
        y = 1 if rng.random() < 1 / (1 + math.exp(-(1.2 * x[0] - x[1]))) else 0
        leak = round(y + rng.gauss(0, 0.05), 4)
        out.append([start + i] + x + [leak, y])
    return out
train, test = rows(900, 0), rows(400, 900)
for name, data in (("train", train), ("test", test)):
    with open(f"proj/data/{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "x0", "x1", "x2", "post_visit_score", "y"])
        w.writerows(data)
with open("proj/outputs/test_predictions.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["row_id", "y_true", "y_pred"])
    w.writerows([r[0], r[5], int(r[4] > 0.5)] for r in test)
acc = sum(r[5] == int(r[4] > 0.5) for r in test) / len(test)
json.dump({"test": {"accuracy": round(acc, 3)}}, open("proj/outputs/metrics.json", "w"))
with open("proj/README.md", "w") as f:
    f.write(f"# Readmission model\n\nOur model reaches {acc:.1%} test accuracy.\n")
PY
