#!/usr/bin/env bash
# Install the ml-pipeline skill for Codex CLI and Kimi Code CLI.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SKILL="$HERE/skills/ml-pipeline/SKILL.md"

enforcement_section() {
  local skill_path="$1"
  cat << EOF

# ML pipeline enforcement (global — mandatory)

Whenever a task involves training, fine-tuning, tuning, or evaluating a machine-learning
model on data — any modality, any framework — read and follow the pipeline skill at
\`$skill_path\` BEFORE writing any modeling code. Its 16-step order is strict:

1 Data inspection → 2 EDA → 3 Define prediction problem → 4 Data cleaning →
5 Data engineering → 6 Train/val/test split → 7 Feature engineering → 8 Preprocessing →
9 Baseline → 10 Training → 11 Tuning → 12 Evaluation → 13 Error analysis →
14 Final test → 15 Deployment → 16 Monitoring/retraining.

Never jump straight to model training — understand the data first, in this order. Stop at
the phase gates (after steps 3, 8, 12, and at wrap-up) to report in plain language what was
done and what comes next, and wait for the user's explicit permission before continuing.
Only the user approves a gate, in their own words; never record one yourself without that.
Work in marimo notebooks (\`marimo edit <file>\`) and produce matplotlib figures at every
data-facing step; save figures to \`ml_pipeline/figures/\`. Track progress in the append-only
\`ml_pipeline/PIPELINE.md\` ledger (\`- Step N done: <result>\`, one step at a time) and resume from
it across sessions. For large files, pass paths to the guard library (it streams) instead of
loading them whole. Before Gate A every column has a data-dictionary row with a note that cites
its own numbers, missing values have a likely cause, and the label is audited; statistics that
relate a feature to the target are computed on the training split only; train on the full
training split (a slice is a pilot that needs the user's override).
EOF
}

audit_section() {
  local skill_path="$1"
  cat <<EOF

## ML audit (read-only)

When asked to audit, check, or verify an EXISTING ML project, model, or claimed score, read and follow
\`$skill_path\`. An audit never trains, fixes, or edits project files and does not start the ML pipeline.
Here there are no hooks: the read-only rule, the one-command rule, and the user's trust and retrain
approvals are instructions only - follow them exactly.
EOF
}

install_for() {
  local name="$1" home_dir="$2"
  [ -d "$home_dir" ] || { echo "skip: $name not found ($home_dir missing)"; return 0; }
  # the whole skill folder: SKILL.md, references/, and lib/ (the guard library and the canary)
  rm -rf "$home_dir/skills/ml-pipeline"
  mkdir -p "$home_dir/skills"
  cp -R "$HERE/skills/ml-pipeline" "$home_dir/skills/ml-pipeline"
  find "$home_dir/skills/ml-pipeline" -name __pycache__ -type d -prune -exec rm -rf {} +
  # the audit skill and its tool (it finds the guard library at ../../ml-pipeline/lib)
  rm -rf "$home_dir/skills/ml-audit"
  cp -R "$HERE/skills/ml-audit" "$home_dir/skills/ml-audit"
  find "$home_dir/skills/ml-audit" -name __pycache__ -type d -prune -exec rm -rf {} +
  if ! grep -q "ML pipeline enforcement" "$home_dir/AGENTS.md" 2>/dev/null; then
    enforcement_section "$home_dir/skills/ml-pipeline/SKILL.md" >> "$home_dir/AGENTS.md"
  fi
  if ! grep -q "ML audit (read-only)" "$home_dir/AGENTS.md" 2>/dev/null; then
    audit_section "$home_dir/skills/ml-audit/SKILL.md" >> "$home_dir/AGENTS.md"
  fi
  echo "installed: $name ($home_dir)"
}

install_for "Codex" "$HOME/.codex"
install_for "Kimi"  "$HOME/.kimi-code"

echo "Done. New sessions of Codex/Kimi will enforce the ML pipeline."
