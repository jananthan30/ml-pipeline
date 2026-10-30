# 08: Show signs of test-set misuse

**What to build:** The audit reads the run log, the ledger and the code for repeated test scoring and for test data used in tuning. Log evidence is proven, and code signs are suspected.

**Blocked by:** 01

**Status:** done (v0.6-audit)

- [x] A run log with repeated test scoring gives a proven finding
- [x] Test data referenced in tuning code gives a suspected finding
- [x] A project without a run log or ledger gets 'could not check', not a pass
- [x] A single honest final test gives no finding
