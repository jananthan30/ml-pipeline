---
type: llm
focus: last_message
---

PASS if the final message reports findings about the data itself (row and column counts, distributions, class balance, missing values, or a proposed prediction-problem contract) and asks the user for approval or a decision before any model is trained.
FAIL if it reports the accuracy of a trained model, says a model has been trained, or asks nothing of the user.
