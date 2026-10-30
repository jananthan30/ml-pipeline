# 04: Detect data leakage

**What to build:** The audit detects rows or entities shared across splits, features that give away the target, and training data later than test data. These feed the required checks, so 'trustworthy' becomes reachable once tickets 02 and 04 are done.

**Blocked by:** 03

**Status:** done (v0.6-audit)

- [x] Identical rows across splits give a proven critical finding
- [x] Shared entities across splits are flagged when an entity column is identified
- [x] A planted near-perfect feature is flagged, and a strong but honest feature is not, tested on both sides of the threshold
- [x] A time-order violation is flagged only when a time column exists
- [x] A fully clean fixture reaches 'trustworthy', and its report still lists what was not checked
