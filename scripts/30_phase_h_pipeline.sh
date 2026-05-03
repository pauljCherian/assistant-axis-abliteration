#!/bin/bash
#
# Phase H: parameterized AA pipeline runner.
# Runs the full Lu et al. pipeline on an arbitrary HF model at an arbitrary layer.
# Saves filtered + unfiltered vectors in parallel (per Phase F survivor-bias lesson).
#
# Usage:
#   bash scripts/30_phase_h_pipeline.sh \
#     <hf_model_id> <output_dir_name> <layer> <hidden_dim>
#
# Example:
#   bash scripts/30_phase_h_pipeline.sh \
#     microsoft/Phi-3.5-mini-instruct phi-3.5-mini 16 3072
#   bash scripts/30_phase_h_pipeline.sh \
#     meta-llama/Llama-3.2-3B-Instruct llama-3.2-3b 14 3072
#
# Idempotent — safe to re-run after interruption.
# Expects .env with OPENAI_API_KEY in repo root.

set -euo pipefail

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <hf_model_id> <output_dir_name> <layer> <hidden_dim>"
    exit 1
fi

MODEL="$1"
OUT_NAME="$2"
LAYER="$3"
HIDDEN_DIM="$4"

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
PIPELINE="$PROJECT_ROOT/assistant-axis/pipeline"
OUTPUT="$PROJECT_ROOT/results/$OUT_NAME"
ROLES_DIR="$PROJECT_ROOT/assistant-axis/data/roles/instructions"
QUESTIONS_FILE="$PROJECT_ROOT/assistant-axis/data/extraction_questions.jsonl"

export HF_HOME="${HF_HOME:-$PROJECT_ROOT/hf_cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PROJECT_ROOT/pip_cache}"

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT"

echo "=== Phase H Pipeline ==="
echo "Model:       $MODEL"
echo "Output:      $OUTPUT"
echo "Layer:       $LAYER"
echo "Hidden dim:  $HIDDEN_DIM"
echo "Start:       $(date)"
echo ""

# ── Step 1: Generate responses ─────────────────────────────────────────
RESPONSE_COUNT=0
if [ -d "$OUTPUT/responses" ]; then
    RESPONSE_COUNT=$(ls "$OUTPUT/responses"/*.jsonl 2>/dev/null | wc -l) || RESPONSE_COUNT=0
fi
if [ "$RESPONSE_COUNT" -ge 276 ]; then
    echo "=== Step 1/6: Skipping — $RESPONSE_COUNT response files exist ==="
else
    echo "=== Step 1/6: Generating responses (vLLM) ==="
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
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
    echo "Step 1/6 done: $(date)"
fi
echo ""

# ── Step 2: Extract activations ────────────────────────────────────────
echo "=== Step 2/6: Extracting activations at layer $LAYER ==="
STEP2_MAX_RETRIES=20
STEP2_RETRY=0
while true; do
    # Pre-flight: validate existing .pt files; rm any with wrong shape/count.
    HIDDEN_DIM=$HIDDEN_DIM $PYTHON - <<PYEOF
import os, sys, torch
from pathlib import Path
hd = int(os.environ["HIDDEN_DIM"])
d = Path("$OUTPUT/activations")
d.mkdir(parents=True, exist_ok=True)
removed = 0
for f in sorted(d.glob("*.pt")):
    try:
        x = torch.load(f, map_location="cpu", weights_only=False)
        assert isinstance(x, dict), f"not a dict: {type(x)}"
        assert len(x) == 1200, f"wrong entry count: {len(x)}"
        first = next(iter(x.values()))
        assert tuple(first.shape) == (1, hd), f"bad shape: {first.shape}, expected (1, {hd})"
    except Exception as e:
        print(f"Pre-flight: removing corrupt {f.name}: {e}", file=sys.stderr)
        f.unlink()
        removed += 1
valid = len(list(d.glob("*.pt")))
print(f"Pre-flight: removed={removed}, valid={valid}/276")
PYEOF
    if $PYTHON "$PIPELINE/2_activations.py" \
        --model "$MODEL" \
        --responses_dir "$OUTPUT/responses" \
        --output_dir "$OUTPUT/activations" \
        --batch_size 32 \
        --layers "$LAYER" \
        --tensor_parallel_size 1; then
        echo "Step 2/6 done on attempt $((STEP2_RETRY + 1)): $(date)"
        break
    fi
    STEP2_RETRY=$((STEP2_RETRY + 1))
    if [ $STEP2_RETRY -ge $STEP2_MAX_RETRIES ]; then
        echo "Step 2 failed $STEP2_MAX_RETRIES times. Aborting."
        exit 1
    fi
    echo "Step 2 crashed (attempt $STEP2_RETRY). Sleeping 60s..."
    sleep 60
done
echo ""

# ── Step 3: Judge responses (Batch API — 50% cheaper, resume-safe) ─────
echo "=== Step 3/6: Scoring responses (Batch API) ==="
set -a; source "$PROJECT_ROOT/.env"; set +a
$PYTHON "$PROJECT_ROOT/scripts/3_judge_batch.py" \
    --responses_dir "$OUTPUT/responses" \
    --roles_dir "$ROLES_DIR" \
    --output_dir "$OUTPUT/scores" \
    --judge_model gpt-4.1-mini
echo "Step 3/6 done: $(date)"
echo ""

# ── Step 4: Compute filtered per-role vectors ──────────────────────────
echo "=== Step 4/6: Computing filtered role vectors (score=3, min 50) ==="
$PYTHON "$PIPELINE/4_vectors.py" \
    --activations_dir "$OUTPUT/activations" \
    --scores_dir "$OUTPUT/scores" \
    --output_dir "$OUTPUT/vectors" \
    --min_count 50
echo "Step 4/6 done: $(date)"
echo ""

# ── Step 4b: Compute UNFILTERED per-role vectors (parallel path) ───────
# Carryover from Phase F survivor-bias analysis. Cheap (~5 min).
echo "=== Step 4b/6: Computing unfiltered role vectors ==="
$PYTHON - <<PYEOF
import torch
from pathlib import Path
act_dir = Path("$OUTPUT/activations")
out_dir = Path("$OUTPUT/vectors_unfiltered")
out_dir.mkdir(parents=True, exist_ok=True)
saved = 0
for af in sorted(act_dir.glob("*.pt")):
    role = af.stem
    x = torch.load(af, map_location="cpu", weights_only=False)
    # x is dict of {key: (1, hidden_dim)} for 1200 entries
    stacked = torch.stack([v.squeeze(0) for v in x.values()])  # (1200, hidden_dim)
    mean_vec = stacked.mean(dim=0)  # (hidden_dim,)
    torch.save(mean_vec, out_dir / f"{role}.pt")
    saved += 1
print(f"Saved {saved} unfiltered role vectors to {out_dir}")
PYEOF
echo "Step 4b/6 done: $(date)"
echo ""

# ── Step 5: Compute axis (filtered) ────────────────────────────────────
echo "=== Step 5/6: Computing PCA axis (filtered) ==="
$PYTHON "$PIPELINE/5_axis.py" \
    --vectors_dir "$OUTPUT/vectors" \
    --output "$OUTPUT/axis.pt"
echo "Step 5/6 done: $(date)"
echo ""

# ── Step 5b: Compute axis (unfiltered) ─────────────────────────────────
echo "=== Step 5b/6: Computing PCA axis (unfiltered) ==="
$PYTHON "$PIPELINE/5_axis.py" \
    --vectors_dir "$OUTPUT/vectors_unfiltered" \
    --output "$OUTPUT/axis_unfiltered.pt"
echo "Step 5b/6 done: $(date)"
echo ""

# ── Step 6: Save default vector (no role prompt) ───────────────────────
# default = activation when system prompt is the unmodified Assistant default.
# Already extracted as part of 'default' role file in step 2.
echo "=== Step 6/6: Saving default vector for v_assistant ==="
$PYTHON - <<PYEOF
import torch
from pathlib import Path
act_dir = Path("$OUTPUT/activations")
df = act_dir / "default.pt"
if not df.exists():
    print(f"WARNING: {df} missing — default role not extracted")
else:
    x = torch.load(df, map_location="cpu", weights_only=False)
    stacked = torch.stack([v.squeeze(0) for v in x.values()])
    default_vec = stacked.mean(dim=0)
    torch.save(default_vec, Path("$OUTPUT") / "default.pt")
    print(f"Saved default vector: shape={tuple(default_vec.shape)}")
PYEOF
echo "Step 6/6 done: $(date)"
echo ""

echo "=== Phase H pipeline complete for $MODEL ==="
echo "End: $(date)"
echo ""
echo "Outputs in $OUTPUT/:"
echo "  responses/         — 276 .jsonl (1200 rollouts/role)"
echo "  activations/       — 276 .pt (raw per-rollout, can delete after)"
echo "  scores/            — 276 .json (judge scores)"
echo "  vectors/           — 276 .pt (filtered role vectors, score=3 only)"
echo "  vectors_unfiltered/— 276 .pt (mean over all 1200 rollouts)"
echo "  axis.pt            — PCA results (filtered)"
echo "  axis_unfiltered.pt — PCA results (unfiltered)"
echo "  default.pt         — default Assistant activation (for v_assistant)"
echo ""
echo "Next: run scripts/32_compute_contrast_axes.py"
