#!/bin/bash
#
# Local fallback for Steps 1+2 of the abliterated pipeline.
# Pulls mlabonne pre-abliterated Llama 3.1 8B from HF and runs Step 1 (vLLM
# generation) on GPU 0, then Step 2 (HF activation extraction) on GPU 1.
# Uploads outputs to HF so 04_resume_abliterated_local.sh can pick up Steps 3-5.
#
# Use this only if cloud (RunPod) is unavailable. Wall-clock ~40-60 hrs on
# 2× RTX 8000.
#
# Prerequisites: lisplab1 GPU 0 and 1 idle, mlabonne model accessible from
# our HF cache, .env has OPENAI_API_KEY.
#
# Usage (RUN UNDER TMUX):
#   tmux new -s abl-local
#   bash scripts/02_run_pipeline_local_abliterated.sh 2>&1 | tee results/abliterated/run.log

set -euo pipefail

PROJECT_ROOT="/scratch/paulc/assistant-axis-abliteration"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
PIPELINE="$PROJECT_ROOT/assistant-axis/pipeline"
MODEL="mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
OUTPUT="$PROJECT_ROOT/results/abliterated"
HF_REPO="pandaman007/assistant-axis-abliteration-vectors"
HF_PREFIX="llama-3.1-8b-abliterated"

ROLES_DIR="$PROJECT_ROOT/assistant-axis/data/roles/instructions"
QUESTIONS_FILE="$PROJECT_ROOT/assistant-axis/data/extraction_questions.jsonl"

export HF_HOME=/scratch/paulc/hf_cache
export PIP_CACHE_DIR=/scratch/paulc/pip_cache

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT/responses" "$OUTPUT/activations"

echo "=== Local abliterated Steps 1+2 ==="
echo "Model:  $MODEL"
echo "Output: $OUTPUT"
echo "Start:  $(date)"

# ── Step 1: Generate (vLLM on GPU 0) ──────────────────────────────────
RESPONSE_COUNT=$(ls "$OUTPUT/responses"/*.jsonl 2>/dev/null | wc -l)
if [ "$RESPONSE_COUNT" -ge 276 ]; then
    echo "=== Step 1/2: Skipping — all $RESPONSE_COUNT response files exist ==="
else
    echo "=== Step 1/2: Generate (vLLM on GPU 0) ==="
    echo "Start: $(date)"

    export CUDA_VISIBLE_DEVICES=0
    STEP1_MAX_RETRIES=5
    STEP1_RETRY=0
    while true; do
        # Pre-flight: drop partial JSONLs.
        $PYTHON - <<PYEOF
from pathlib import Path
removed = 0
for f in sorted(Path("$OUTPUT/responses").glob("*.jsonl")):
    n = sum(1 for _ in open(f))
    if n != 1200:
        print(f"Pre-flight: removing partial {f.name} ({n} lines)")
        f.unlink(); removed += 1
valid = len(list(Path("$OUTPUT/responses").glob("*.jsonl")))
print(f"Pre-flight: removed={removed}, valid={valid}/276")
PYEOF
        COUNT=$(ls "$OUTPUT/responses"/*.jsonl 2>/dev/null | wc -l)
        if [ "$COUNT" -ge 276 ]; then
            echo "Step 1 done ($COUNT/276): $(date)"
            break
        fi
        echo "Step 1 attempt $((STEP1_RETRY + 1))/$STEP1_MAX_RETRIES (currently $COUNT/276)..."
        $PYTHON "$PIPELINE/1_generate.py" \
            --model "$MODEL" \
            --roles_dir "$ROLES_DIR" \
            --questions_file "$QUESTIONS_FILE" \
            --output_dir "$OUTPUT/responses" \
            --tensor_parallel_size 1 \
            --question_count 240 \
            --max_tokens 512 \
            --temperature 0.7 \
            --max_model_len 2048 || true
        STEP1_RETRY=$((STEP1_RETRY + 1))
        if [ $STEP1_RETRY -ge $STEP1_MAX_RETRIES ]; then
            NEW_COUNT=$(ls "$OUTPUT/responses"/*.jsonl 2>/dev/null | wc -l)
            echo "Step 1 reached max retries with $NEW_COUNT/276. Continuing."
            break
        fi
        sleep 30
    done

    echo "=== Uploading responses to HF (Step-1 backup) ==="
    $PYTHON - <<PYEOF
from huggingface_hub import HfApi
HfApi().upload_folder(
    folder_path="$OUTPUT/responses",
    path_in_repo="$HF_PREFIX/responses",
    repo_id="$HF_REPO", repo_type="dataset",
    commit_message="Abliterated responses (local)")
print("Responses uploaded.")
PYEOF
fi
echo ""

# ── Step 2: Extract activations (HF hooks on GPU 1) ───────────────────
echo "=== Step 2/2: Extract activations (HF hooks on GPU 1) ==="
echo "Start: $(date)"

export CUDA_VISIBLE_DEVICES=1

STEP2_MAX_RETRIES=20
STEP2_RETRY=0
while true; do
    $PYTHON - <<PYEOF
import sys, torch
from pathlib import Path
d = Path("$OUTPUT/activations")
d.mkdir(parents=True, exist_ok=True)
removed = 0
for f in sorted(d.glob("*.pt")):
    try:
        x = torch.load(f, map_location="cpu", weights_only=False)
        assert isinstance(x, dict) and len(x) == 1200
        first = next(iter(x.values()))
        assert tuple(first.shape) == (1, 4096)
    except Exception as e:
        print(f"Removing corrupt {f.name}: {e}", file=sys.stderr)
        f.unlink(); removed += 1
print(f"Pre-flight: removed={removed}, valid={len(list(d.glob('*.pt')))}/276")
PYEOF

    if $PYTHON "$PIPELINE/2_activations.py" \
        --model "$MODEL" \
        --responses_dir "$OUTPUT/responses" \
        --output_dir "$OUTPUT/activations" \
        --batch_size 16 \
        --layers 16 \
        --tensor_parallel_size 1; then
        echo "Step 2 done on attempt $((STEP2_RETRY + 1)): $(date)"
        break
    fi

    STEP2_RETRY=$((STEP2_RETRY + 1))
    if [ $STEP2_RETRY -ge $STEP2_MAX_RETRIES ]; then
        echo "Step 2 failed $STEP2_MAX_RETRIES times. Aborting."
        exit 1
    fi
    echo "Step 2 crashed (attempt $STEP2_RETRY/$STEP2_MAX_RETRIES) at $(date). Sleeping 60s..."
    sleep 60
done

# ── Upload activations to HF ──────────────────────────────────────────
echo "=== Uploading activations to HF ==="
$PYTHON - <<PYEOF
from huggingface_hub import HfApi
HfApi().upload_folder(
    folder_path="$OUTPUT/activations",
    path_in_repo="$HF_PREFIX/activations_layer16_full",
    repo_id="$HF_REPO", repo_type="dataset",
    commit_message="Abliterated activations layer 16 (local)")
print("Activations uploaded.")
PYEOF

echo ""
echo "================================================"
echo "=== LOCAL STEPS 1+2 COMPLETE: $(date) ==="
echo "================================================"
echo "Outputs uploaded under $HF_PREFIX/{responses,activations_layer16_full}"
echo ""
echo "NEXT: tmux new -s judge-abl 'bash scripts/04_resume_abliterated_local.sh'"
