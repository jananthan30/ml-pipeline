# 10: Run approved retrain checks

**What to build:** Checks that need a retrain, starting with a shuffled-label test, are listed as needing approval. They run only after the user approves in chat, and only on copies inside the audit folder. Drop this ticket and say so if it cannot be done without weakening the guard.

**Blocked by:** 09

**Status:** done (v0.6-audit; adapted to frozen data, see the report)

- [x] Without approval the check is listed with its reason and does not run
- [x] Approval can come only from the user's own message, using the same mechanism as gate approvals
- [x] An agent writing the approval itself is denied
- [x] The project's own files, gates and test-touch count are unchanged afterward
- [x] A planted leak gives a shuffled-label score well above chance, and a clean control does not
