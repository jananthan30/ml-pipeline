# 09: Add the agent's review layer

**What to build:** The skill tells the agent to read code and prose, add labelled review findings, explain results in plain words, and never restate a number the tool did not compute.

**Blocked by:** 02, 04, 05, 07, 08

**Status:** ready-for-agent

- [ ] Review findings cannot produce a proven result or change the verdict to 'not trustworthy' on their own
- [ ] A new eval case with a planted leak: the agent reports 'not trustworthy', does not train, and edits no project files
- [ ] Codex and Kimi installs receive the skill, and the docs state that the rules are instructions only there
