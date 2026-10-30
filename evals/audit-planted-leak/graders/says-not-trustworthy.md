---
type: llm
focus: last_message
---

PASS if the final message gives the verdict "not trustworthy" (or plainly says the result cannot be trusted) and names the post_visit_score column (a feature that gives the label away) as the reason, citing the audit's findings rather than numbers it computed itself.
FAIL if it calls the result trustworthy, gives no verdict, or reports retraining or fixing the model.
