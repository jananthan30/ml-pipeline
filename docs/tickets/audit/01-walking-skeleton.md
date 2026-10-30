# 01: Audit a folder end to end

**What to build:** Running the audit command on a folder produces an inventory and a report whose verdict is 'not enough evidence', because no checks exist yet. This proves the full path: skill, tool, guard exception, output folder.

**Blocked by:** The combined v0.5 line

**Status:** done (v0.6-audit)

- [x] The inventory lists data, code, models, predictions and claims, and skips environment, dependency and version-control folders
- [x] The user can correct the inventory and rerun
- [x] Output goes to a new dated subfolder, with a report, a findings file and the inventory
- [x] Hashing every file before and after shows only the audit folder changed
- [x] The training guard allows the exact tool invocation and denies a look-alike script, a training script mentioning audit, and the tool chained to another command
- [x] The audit adds no ledger lines and does not touch the guard's approval or state files
- [x] The pipeline skill does not fire on an audit request
