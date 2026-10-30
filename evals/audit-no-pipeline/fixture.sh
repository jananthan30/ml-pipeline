#!/usr/bin/env bash
# A finished project: train/test files, test predictions and a claimed accuracy. Nothing is wrong with it.
set -euo pipefail
python3 - <<'PY'
import csv, json, math, os, random
rng = random.Random(11)
os.makedirs("proj/data", exist_ok=True)
os.makedirs("proj/outputs", exist_ok=True)
def rows(n, start):
    out = []
    for i in range(n):
        x = [round(rng.gauss(0, 1), 6) for _ in range(3)]
        p = 1 / (1 + math.exp(-(1.5 * x[0] - x[1])))
        out.append([start + i] + x + [1 if rng.random() < p else 0, p])
    return out
train, test = rows(800, 0), rows(400, 800)
for name, data in (("train", train), ("test", test)):
    with open(f"proj/data/{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "x0", "x1", "x2", "y"])
        w.writerows(r[:5] for r in data)
with open("proj/outputs/test_predictions.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["row_id", "y_true", "y_pred", "y_prob"])
    w.writerows([r[0], r[4], int(r[5] > 0.5), round(r[5], 6)] for r in test)
acc = sum(r[4] == int(r[5] > 0.5) for r in test) / len(test)
json.dump({"test": {"accuracy": round(acc, 3)}}, open("proj/outputs/metrics.json", "w"))
PY
