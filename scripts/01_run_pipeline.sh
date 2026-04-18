#!/bin/bash
#
# Step 1: Run the full Assistant Axis pipeline on original Llama 3.1 8B Instruct.
#
# Uses all 5 official pipeline scripts from assistant-axis/ unchanged.
# Outputs to results/original/.
#
# Prerequisites:
#   - .env file with OPENAI_API_KEY (for step 3: judge scoring)
#   - GPUs 0 and 1 free (2 workers for steps 1 & 2)
#
# Usage:
#   cd /scratch/paulc/assistant-axis-abliteration
#   bash scripts/01_run_pipeline.sh 2>&1 | tee results/original/run.log
#
# All steps are idempotent — safe to re-run after interruption.

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────
PROJECT_ROOT="/scratch/paulc/assistant-axis-abliteration"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
PIPELINE="$PROJECT_ROOT/assistant-axis/pipeline"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
OUTPUT="$PROJECT_ROOT/results/original"

ROLES_DIR="$PROJECT_ROOT/assistant-axis/data/roles/instructions"
QUESTIONS_FILE="$PROJECT_ROOT/assistant-axis/data/extraction_questions.jsonl"

# Environment
export HF_HOME=/scratch/paulc/hf_cache
export PIP_CACHE_DIR=/scratch/paulc/pip_cache

cd "$PROJECT_ROOT"

echo "=== Step 1: Full Assistant Axis Pipeline ==="
echo "Model:  $MODEL"
echo "Output: $OUTPUT"
echo "Start:  $(date)"
echo ""

# ── Step 1: Generate responses (vLLM, 2 workers) ──────────────────────
echo "=== Step 1/5: Generating responses (vLLM) ==="
echo "Start: $(date)"

export CUDA_VISIBLE_DEVICES=0,1
$PYTHON "$PIPELINE/1_generate.py" \
    --model "$MODEL" \
    --roles_dir "$ROLES_DIR" \
    --questions_file "$QUESTIONS_FILE" \
    --output_dir "$OUTPUT/responses" \
    --tensor_parallel_size 1 \
    --question_count 240 \
    --max_tokens 512 \
    --temperature 0.7 \
    --max_model_len 2048

echo "Step 1/5 done: $(date)"
echo ""

# ── Step 2: Extract activations (HuggingFace hooks, 2 workers) ────────
echo "=== Step 2/5: Extracting activations ==="
echo "Start: $(date)"

export CUDA_VISIBLE_DEVICES=0,1
$PYTHON "$PIPELINE/2_activations.py" \
    --model "$MODEL" \
    --responses_dir "$OUTPUT/responses" \
    --output_dir "$OUTPUT/activations" \
    --batch_size 8 \
    --layers all \
    --tensor_parallel_size 1

echo "Step 2/5 done: $(date)"
echo ""

# ── Step 3: Judge responses (GPT-4.1-mini via OpenAI API) ─────────────
echo "=== Step 3/5: Scoring responses (GPT-4.1-mini) ==="
echo "Start: $(date)"

# Load API key from .env
set -a
source "$PROJECT_ROOT/.env"
set +a

$PYTHON "$PIPELINE/3_judge.py" \
    --responses_dir "$OUTPUT/responses" \
    --roles_dir "$ROLES_DIR" \
    --output_dir "$OUTPUT/scores" \
    --judge_model gpt-4.1-mini \
    --requests_per_second 100 \
    --batch_size 50

echo "Step 3/5 done: $(date)"
echo ""

# ── Step 4: Compute per-role vectors ──────────────────────────────────
echo "=== Step 4/5: Computing per-role vectors ==="
echo "Start: $(date)"

$PYTHON "$PIPELINE/4_vectors.py" \
    --activations_dir "$OUTPUT/activations" \
    --scores_dir "$OUTPUT/scores" \
    --output_dir "$OUTPUT/vectors" \
    --min_count 50

echo "Step 4/5 done: $(date)"
echo ""

# ── Step 5: Compute axis ──────────────────────────────────────────────
echo "=== Step 5/5: Computing axis ==="
echo "Start: $(date)"

$PYTHON "$PIPELINE/5_axis.py" \
    --vectors_dir "$OUTPUT/vectors" \
    --output "$OUTPUT/axis.pt"

echo "Step 5/5 done: $(date)"
echo ""

# ── Cleanup: delete raw activations to reclaim ~170GB ─────────────────
echo "=== Cleanup: removing raw activations ==="
ACTIVATION_SIZE=$(du -sh "$OUTPUT/activations" 2>/dev/null | cut -f1)
echo "Removing $OUTPUT/activations/ ($ACTIVATION_SIZE)"
rm -rf "$OUTPUT/activations"
echo "Disk reclaimed."
echo ""

# ── Summary ───────────────────────────────────────────────────────────
echo "=== Pipeline complete ==="
echo "End: $(date)"
echo ""
echo "Outputs:"
echo "  Responses: $OUTPUT/responses/"
echo "  Scores:    $OUTPUT/scores/"
echo "  Vectors:   $OUTPUT/vectors/"
echo "  Axis:      $OUTPUT/axis.pt"
echo ""
echo "Next: run scripts/02_abliterate_model.py"
