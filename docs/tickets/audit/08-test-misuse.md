# 08: Show signs of test-set misuse

**What to build:** The audit reads the run log, the ledger and the code for repeated test scoring and for test data used in tuning. Log evidence is proven, and code signs are suspected.

**Blocked by:** 01

**Status:** ready-for-agent

- [ ] A run log with repeated test scoring gives a proven finding
- [ ] Test data referenced in tuning code gives a suspected finding
- [ ] A project without a run log or ledger gets 'could not check', not a pass
- [ ] A single honest final test gives no finding
