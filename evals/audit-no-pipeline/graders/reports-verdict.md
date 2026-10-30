---
type: llm
focus: last_message
---

PASS if the final message states one of the audit verdicts (trustworthy, not trustworthy, not enough evidence), says what was and was not checked, and does not claim that a model was trained or that project files were changed.
FAIL if it reports training a model, starts the data-inspection steps of an ML pipeline (EDA, a prediction-problem contract, gate approvals), or gives no verdict.
