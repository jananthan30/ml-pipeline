# 04: Detect data leakage

**What to build:** The audit detects rows or entities shared across splits, features that give away the target, and training data later than test data. These feed the required checks, so 'trustworthy' becomes reachable once tickets 02 and 04 are done.

**Blocked by:** 03

**Status:** ready-for-agent

- [ ] Identical rows across splits give a proven critical finding
- [ ] Shared entities across splits are flagged when an entity column is identified
- [ ] A planted near-perfect feature is flagged, and a strong but honest feature is not, tested on both sides of the threshold
- [ ] A time-order violation is flagged only when a time column exists
- [ ] A fully clean fixture reaches 'trustworthy', and its report still lists what was not checked
