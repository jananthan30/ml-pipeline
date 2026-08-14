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
Work in marimo notebooks (\`marimo edit <file>\`) and produce matplotlib figures at every
data-facing step; save figures to \`ml_pipeline/figures/\`. Track progress in
\`ml_pipeline/PIPELINE.md\` and resume from it across sessions.
EOF
}

install_for() {
  local name="$1" home_dir="$2"
  [ -d "$home_dir" ] || { echo "skip: $name not found ($home_dir missing)"; return 0; }
  mkdir -p "$home_dir/skills/ml-pipeline"
  cp "$SKILL" "$home_dir/skills/ml-pipeline/SKILL.md"
  if ! grep -q "ML pipeline enforcement" "$home_dir/AGENTS.md" 2>/dev/null; then
    enforcement_section "$home_dir/skills/ml-pipeline/SKILL.md" >> "$home_dir/AGENTS.md"
  fi
  echo "installed: $name ($home_dir)"
}

install_for "Codex" "$HOME/.codex"
install_for "Kimi"  "$HOME/.kimi-code"

echo "Done. New sessions of Codex/Kimi will enforce the ML pipeline."
