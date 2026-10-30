# 05: Scan model files without running them

**What to build:** Model files are scanned statically and never loaded by default. Each is reported as no risky imports found, risky imports found, or unverified.

**Blocked by:** 01

**Status:** ready-for-agent

- [ ] A pickle whose load would create a marker file leaves no marker after an audit
- [ ] A pickle with a dangerous import is flagged and names the import
- [ ] Ordinary scikit-learn-style joblib and zip-format torch fixtures are not flagged
- [ ] Corrupt or unparseable files are 'unverified', never 'safe'
- [ ] The report states that a clean scan does not prove safety
- [ ] No code path loads a model without the trust flag
