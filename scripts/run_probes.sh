#!/usr/bin/env bash
# Probe training + evaluation + INLP for a single model.
# Assumes activations already exist in results/activations/.
#
# Usage:
#   bash scripts/run_probes.sh Qwen/Qwen2.5-7B-Instruct

set -euo pipefail

CONFIG="config/experiment.yaml"
MODEL="${1:?Usage: run_probes.sh <model-id>}"

echo "--- Stage 5: Training probes ---"
python -m src.probes.train_probes \
    --config "$CONFIG" \
    --model "$MODEL" \
    --act-dir results/activations \
    --out-dir results/probes

echo "--- Stage 6: Evaluating probes ---"
python -m src.probes.evaluate_probes \
    --config "$CONFIG" \
    --model "$MODEL" \
    --act-dir results/activations \
    --probe-dir results/probes \
    --fig-dir results/figures

# Determine best layer from eval output (used for INLP)
SLUG=$(python -c "print('${MODEL}'.split('/')[-1].lower())")
BEST_LAYER=$(python -c "
import json
with open('results/probes/${SLUG}_eval.json') as f:
    d = json.load(f)
print(d['best_layer'])
")

echo "--- Stage 7: INLP on layer $BEST_LAYER ---"
python -m src.probes.inlp \
    --config "$CONFIG" \
    --model "$MODEL" \
    --layer "$BEST_LAYER" \
    --act-dir results/activations \
    --out-dir results/probes \
    --fig-dir results/figures

echo "Probe pipeline complete for $MODEL."
