#!/bin/bash
#SBATCH --job-name=assistant-axis
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=results/original/slurm_%j.log
#SBATCH --error=results/original/slurm_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#
# Step 1: Run the full Assistant Axis pipeline on original Llama 3.1 8B Instruct.
# Adapted for Dartmouth Discovery cluster (SLURM + A100 GPUs).
#
# Usage:
#   cd /dartfs-hpc/rc/home/2/f006vv2/code/assistant-axis-abliteration
#   sbatch scripts/01_run_pipeline_discovery.sh
#
# All steps are idempotent — safe to re-run after interruption.

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────
PROJECT_ROOT="/dartfs-hpc/rc/home/2/f006vv2/code/assistant-axis-abliteration"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
PIPELINE="$PROJECT_ROOT/assistant-axis/pipeline"
MODEL="meta-llama/Llama-3.1-8B-Instruct"

ROLES_DIR="$PROJECT_ROOT/assistant-axis/data/roles/instructions"
QUESTIONS_FILE="$PROJECT_ROOT/assistant-axis/data/extraction_questions.jsonl"

# Use node-local scratch for large intermediate activations (~170GB)
# This is fast local SSD and cleaned up when the job ends.
LOCAL_SCRATCH="/scratch/assistant_axis_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_SCRATCH"

# Persistent outputs go to project directory
OUTPUT="$PROJECT_ROOT/results/original"
mkdir -p "$OUTPUT"

# Environment
export HF_HOME=/global/scratch/f006vv2/hf_cache
export PIP_CACHE_DIR=/global/scratch/f006vv2/pip_cache

# Load .env (OPENAI_API_KEY)
set -a
source "$PROJECT_ROOT/.env"
set +a

cd "$PROJECT_ROOT"

echo "=== Step 1: Full Assistant Axis Pipeline (Discovery) ==="
echo "Model:    $MODEL"
echo "Output:   $OUTPUT"
echo "Scratch:  $LOCAL_SCRATCH"
echo "Node:     $(hostname)"
echo "GPUs:     $CUDA_VISIBLE_DEVICES"
echo "Start:    $(date)"
echo ""

# ── Step 1/5: Generate responses (vLLM, 2 workers) ────────────────────
echo "=== Step 1/5: Generating responses (vLLM) ==="
echo "Start: $(date)"

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

# ── Step 2/5: Extract activations (HuggingFace hooks, 2 workers) ──────
echo "=== Step 2/5: Extracting activations ==="
echo "Start: $(date)"

# Activations go to local scratch (huge, temporary)
$PYTHON "$PIPELINE/2_activations.py" \
    --model "$MODEL" \
    --responses_dir "$OUTPUT/responses" \
    --output_dir "$LOCAL_SCRATCH/activations" \
    --batch_size 8 \
    --layers all \
    --tensor_parallel_size 1

echo "Step 2/5 done: $(date)"
echo ""

# ── Step 3/5: Judge responses (GPT-4.1-mini) ──────────────────────────
echo "=== Step 3/5: Scoring responses (GPT-4.1-mini) ==="
echo "Start: $(date)"

$PYTHON "$PIPELINE/3_judge.py" \
    --responses_dir "$OUTPUT/responses" \
    --roles_dir "$ROLES_DIR" \
    --output_dir "$OUTPUT/scores" \
    --judge_model gpt-4.1-mini \
    --requests_per_second 100 \
    --batch_size 50

echo "Step 3/5 done: $(date)"
echo ""

# ── Step 4/5: Compute per-role vectors ─────────────────────────────────
echo "=== Step 4/5: Computing per-role vectors ==="
echo "Start: $(date)"

$PYTHON "$PIPELINE/4_vectors.py" \
    --activations_dir "$LOCAL_SCRATCH/activations" \
    --scores_dir "$OUTPUT/scores" \
    --output_dir "$OUTPUT/vectors" \
    --min_count 50

echo "Step 4/5 done: $(date)"
echo ""

# ── Step 5/5: Compute axis ─────────────────────────────────────────────
echo "=== Step 5/5: Computing axis ==="
echo "Start: $(date)"

$PYTHON "$PIPELINE/5_axis.py" \
    --vectors_dir "$OUTPUT/vectors" \
    --output "$OUTPUT/axis.pt"

echo "Step 5/5 done: $(date)"
echo ""

# ── Cleanup: remove local scratch ──────────────────────────────────────
echo "=== Cleanup ==="
ACTIVATION_SIZE=$(du -sh "$LOCAL_SCRATCH/activations" 2>/dev/null | cut -f1)
echo "Removing local scratch ($ACTIVATION_SIZE)"
rm -rf "$LOCAL_SCRATCH"
echo "Done."
echo ""

# ── Summary ────────────────────────────────────────────────────────────
echo "=== Pipeline complete ==="
echo "End: $(date)"
echo ""
echo "Outputs:"
echo "  Responses: $OUTPUT/responses/"
echo "  Scores:    $OUTPUT/scores/"
echo "  Vectors:   $OUTPUT/vectors/"
echo "  Axis:      $OUTPUT/axis.pt"
