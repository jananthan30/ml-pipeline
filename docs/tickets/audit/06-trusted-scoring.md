# 06: Score a model the user trusts

**What to build:** With an explicit trust flag for the project, the audit loads the models and generates predictions in a child process using the project's own interpreter, then feeds them to the score checks.

**Blocked by:** 02, 05

**Status:** ready-for-agent

- [ ] Without the flag, the model is not loaded and its checks are 'could not check'
- [ ] Code-free loading modes are used first where the format has one
- [ ] A crash or timeout in the child process becomes a 'could not check' finding, not a failed audit
- [ ] The report records that the audit viewed test labels and warns against using its numbers to choose a model
