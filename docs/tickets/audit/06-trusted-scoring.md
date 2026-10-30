# 06: Score a model the user trusts

**What to build:** With an explicit trust flag for the project, the audit loads the models and generates predictions in a child process using the project's own interpreter, then feeds them to the score checks.

**Blocked by:** 02, 05

**Status:** done (v0.6-audit)

- [x] Without the flag, the model is not loaded and its checks are 'could not check'
- [x] Code-free loading modes are used first where the format has one
- [x] A crash or timeout in the child process becomes a 'could not check' finding, not a failed audit
- [x] The report records that the audit viewed test labels and warns against using its numbers to choose a model
