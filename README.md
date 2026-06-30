# Caught in the Act — Replication

A replication of the key findings from [**"Caught in the Act: A Mechanistic Approach to Detecting Deception"**](https://arxiv.org/abs/2508.19505). The paper shows that linear probes trained on a model's internal activations can reliably distinguish deceptive from honest arguments — and that this signal becomes stronger and more robust as model size increases.

## What this codebase does

1. **Samples MMLU questions** and converts them to binary-choice format (correct answer vs. one distractor).
2. **Generates arguments** using a reasoning model (DeepSeek): one honest argument (for the correct answer) and one deceptive argument (for the wrong answer) per question.
3. **Filters arguments** with GPT-4o, scoring each on factual recall and persuasiveness and discarding low-quality outputs.
4. **Extracts hidden-state activations** from a target model (Qwen 7B / 14B) at every transformer layer, using the final token position.
5. **Trains linear probes** (logistic regression) on those activations to classify honest vs. deceptive arguments.
6. **Evaluates probes** layer-by-layer and plots the accuracy curve (replicates Figure 1 of the paper).
7. **Runs INLP** (Iterative Nullspace Projection) to count how many independent deception directions exist in the activation space (replicates Figure 2).

## Directory structure

```
caught-in-the-act/
├── config/
│   └── experiment.yaml          # models, thresholds, hyperparameters
├── data/
│   ├── raw/                     # mmlu_binary.jsonl (stage 1 output)
│   ├── generated/               # arguments.jsonl (stage 2 output)
│   └── filtered/                # arguments_filtered.jsonl (stage 3 output)
├── src/
│   ├── dataset/
│   │   ├── sample_mmlu.py       # stage 1
│   │   └── generate_arguments.py # stage 2
│   ├── filter/
│   │   └── filter_arguments.py  # stage 3
│   ├── activations/
│   │   └── extract_activations.py # stage 4
│   ├── probes/
│   │   ├── train_probes.py      # stage 5
│   │   ├── evaluate_probes.py   # stage 6
│   │   └── inlp.py              # stage 7
│   └── utils/
│       ├── model_utils.py       # HuggingFace model loading + hooks
│       └── data_utils.py        # JSONL read/write helpers
├── scripts/
│   ├── run_pipeline.sh          # runs all 7 stages end-to-end
│   └── run_probes.sh            # runs stages 5-7 only (activations already exist)
└── results/
    ├── activations/             # .pt files per model
    ├── probes/                  # .pkl probe files + eval JSON
    └── figures/                 # layer-accuracy and INLP curve plots
```

## Quickstart

```bash
pip install -e .          # installs the package and all dependencies

export OPENAI_API_KEY=...     # for GPT-4o filtering (stage 3)
export DEEPSEEK_API_KEY=...   # for argument generation (stage 2)

# Full pipeline (reads target models from config/experiment.yaml)
bash scripts/run_pipeline.sh

# Single model
bash scripts/run_pipeline.sh Qwen/Qwen2.5-7B-Instruct

# Probe training + eval + INLP only (if activations already extracted)
bash scripts/run_probes.sh Qwen/Qwen2.5-7B-Instruct
```

Each stage can also be run independently as a Python module:

```bash
python -m src.dataset.sample_mmlu
python -m src.dataset.generate_arguments
python -m src.filter.filter_arguments
python -m src.activations.extract_activations --model Qwen/Qwen2.5-7B-Instruct
python -m src.probes.train_probes    --model Qwen/Qwen2.5-7B-Instruct
python -m src.probes.evaluate_probes --model Qwen/Qwen2.5-7B-Instruct
python -m src.probes.inlp            --model Qwen/Qwen2.5-7B-Instruct --layer 20
```

## Key findings being replicated

- **Layer-wise accuracy curve**: probe accuracy starts near chance in early layers, peaks around 80–90% at ~2/3 model depth, then declines. Larger models show a stronger and earlier peak.
- **INLP robustness**: larger models have many more independent deception directions (100+), making the representation harder to erase, while small models collapse to chance after ~20–40 projection rounds.

## Reference

Boxo, Socha, Yoo, and Raval (2025). *Caught in the Act: A Mechanistic Approach to Detecting Deception.* [arXiv:2508.19505](https://arxiv.org/abs/2508.19505)
