# Canary dataset

`canary.csv` — 2,000 rows, features `x0..x4`, binary target `y`.
`y = 1[x0 + x1·x2 − 0.5·x3 > 0]` with **20% of labels flipped at random** (x4 is pure noise).

No honest model can score above **0.80 accuracy** on held-out rows. On a 400-row test split the
3-standard-error limit is 0.86. A reported test accuracy above that means the test set leaked —
through duplicates, a target-derived feature, or being used during tuning.

Check a result: `python3 -c "import sys; sys.path.insert(0,'lib'); import mlpipeline_canary as c; print(c.verdict(0.91, 400))"`
Regenerate: `python3 examples/canary/make_canary.py`
