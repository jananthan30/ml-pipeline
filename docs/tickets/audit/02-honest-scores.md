# 02: Recompute claimed scores

**What to build:** Given predictions and a claimed score, the audit recalculates it and reports a mismatch, a score no better than a trivial baseline, or a claim that cannot be checked. This is the first real verdict, and it needs no model loading.

**Blocked by:** 01

**Status:** ready-for-agent

- [ ] Machine-readable claims are parsed, and prose claims can be passed in with a source location
- [ ] Accuracy, F1, AUC, RMSE and MAE are implemented without scikit-learn and tested against known values
- [ ] A planted wrong claim gives a proven critical finding and 'not trustworthy'
- [ ] A claim correct to its stated rounding gives no finding, tested on both sides of the tolerance
- [ ] A claim with no predictions is 'could not check' and gives 'not enough evidence'
- [ ] A planted score equal to the trivial baseline is flagged
- [ ] Every finding has a reproduction command that gives the same result
