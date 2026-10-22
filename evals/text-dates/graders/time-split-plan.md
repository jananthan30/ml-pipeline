---
type: llm
focus: last_message
---

PASS if the final message recognizes order_date as a date (even though it is stored as text), proposes a chronological split holding out the latest period (or explains why returns drift over time), mentions that customer_id repeats, and asks the user for approval before training.
FAIL if it proposes a random split, reports a trained model's accuracy, or asks nothing of the user.
