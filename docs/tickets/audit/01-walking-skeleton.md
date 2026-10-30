# 01: Audit a folder end to end

**What to build:** Running the audit command on a folder produces an inventory and a report whose verdict is 'not enough evidence', because no checks exist yet. This proves the full path: skill, tool, guard exception, output folder.

**Blocked by:** The combined v0.5 line

**Status:** ready-for-agent

- [ ] The inventory lists data, code, models, predictions and claims, and skips environment, dependency and version-control folders
- [ ] The user can correct the inventory and rerun
- [ ] Output goes to a new dated subfolder, with a report, a findings file and the inventory
- [ ] Hashing every file before and after shows only the audit folder changed
- [ ] The training guard allows the exact tool invocation and denies a look-alike script, a training script mentioning audit, and the tool chained to another command
- [ ] The audit adds no ledger lines and does not touch the guard's approval or state files
- [ ] The pipeline skill does not fire on an audit request
