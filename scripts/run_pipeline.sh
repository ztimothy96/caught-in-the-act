#!/usr/bin/env bash
# End-to-end pipeline: MMLU sampling → argument generation → filtering →
# activation extraction → probe training → evaluation → INLP.
#
# Prerequisites:
#   pip install -r requirements.txt
#   export OPENAI_API_KEY=...      # for GPT-4o filtering
#   export DEEPSEEK_API_KEY=...    # if using DeepSeek via API
#
# Usage:
#   bash scripts/run_pipeline.sh                            # all models in config
#   bash scripts/run_pipeline.sh Qwen/Qwen2.5-7B-Instruct  # single model

set -euo pipefail

CONFIG="config/experiment.yaml"

# ── Stage 1: Sample MMLU ──────────────────────────────────────────────────────
echo "=== Stage 1: Sampling MMLU ==="
python -m src.dataset.sample_mmlu \
    --config "$CONFIG" \
    --out data/raw/mmlu_binary.jsonl

# ── Stage 2: Generate arguments ───────────────────────────────────────────────
echo "=== Stage 2: Generating arguments ==="
python -m src.dataset.generate_arguments \
    --config "$CONFIG" \
    --inp data/raw/mmlu_binary.jsonl \
    --out data/generated/arguments.jsonl

# ── Stage 3: Filter arguments ─────────────────────────────────────────────────
echo "=== Stage 3: Filtering arguments ==="
python -m src.filter.filter_arguments \
    --config "$CONFIG" \
    --inp data/generated/arguments.jsonl \
    --out data/filtered/arguments_filtered.jsonl

# ── Stages 4-7: Per-model activation extraction, probing, and INLP ───────────
# If a model argument is passed, use it; otherwise read from config.
MODELS=("${@}")
if [ ${#MODELS[@]} -eq 0 ]; then
    # Parse target_models list from YAML (requires python-yaml or yq)
    mapfile -t MODELS < <(python -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
for m in cfg['target_models']:
    print(m)
")
fi

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "=== Model: $MODEL ==="

    echo "--- Stage 4: Extracting activations ---"
    python -m src.activations.extract_activations \
        --config "$CONFIG" \
        --inp data/filtered/arguments_filtered.jsonl \
        --model "$MODEL" \
        --out-dir results/activations

    bash scripts/run_probes.sh "$MODEL"
done

echo ""
echo "Pipeline complete. Results in results/."
