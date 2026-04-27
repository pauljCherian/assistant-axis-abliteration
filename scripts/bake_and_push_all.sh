#!/bin/bash
# Bake all 4 Phase F steered checkpoints with locked α* values, push each to HF model hub,
# remove local copy after successful push to save disk.
#
# Locked α* picks (mixed; from results/persona_vectors/alpha_pilot/summary_merged.md):
#   evil:        α=4   (trait=40.6, coh=68.4)
#   sycophantic: α=5   (trait=83.3, coh=85.1)
#   apathetic:   α=5   (trait=75.3, coh=67.5)
#   humorous:    α=4   (trait=85.0, coh=75.7)
#
# Usage: tmux new -s phase-f-bake; bash scripts/bake_and_push_all.sh

set -euo pipefail

PROJECT=/scratch/paulc/assistant-axis-abliteration
PYTHON=$PROJECT/.venv/bin/python
MODELS_DIR=$PROJECT/models
HF_USER=pandaman007
LAYER=12

cd "$PROJECT"

# HF auth comes from huggingface-cli login (cached token in ~/.cache/huggingface/token).
# Verify before starting any baking.
$PYTHON -c "from huggingface_hub import whoami; print('HF user:', whoami()['name'])" || {
  echo "ERROR: HF not authenticated. Run: huggingface-cli login"; exit 1;
}

# (trait, alpha_int) tuples — alpha as int for hub-id readability (4 not 4.0)
declare -a JOBS=(
  "humorous 4"
  "evil 4"
  "sycophantic 5"
  "apathetic 5"
)

for job in "${JOBS[@]}"; do
  trait=$(echo $job | awk '{print $1}')
  alpha=$(echo $job | awk '{print $2}')
  out_dir=$MODELS_DIR/llama-3.1-8b-${trait}-steered-L${LAYER}-a${alpha}
  hub_id=$HF_USER/llama-3.1-8b-instruct-${trait}-steered-L${LAYER}-a${alpha}

  echo ""
  echo "================================================"
  echo "=== BAKE+PUSH: $trait α=$alpha"
  echo "===   out_dir: $out_dir"
  echo "===   hub_id:  $hub_id"
  echo "===   start:   $(date)"
  echo "================================================"

  $PYTHON scripts/14_bake_persona_vector.py \
    --trait $trait --alpha $alpha --layer $LAYER \
    --output_dir "$out_dir" \
    --push_to_hub --hub_id "$hub_id"

  echo ""
  echo "  push complete; removing local copy to free disk..."
  rm -rf "$out_dir"
  echo "  done with $trait at $(date)"
done

echo ""
echo "================================================"
echo "=== ALL 4 BAKES COMPLETE: $(date) ==="
echo "================================================"
echo "HF models:"
for job in "${JOBS[@]}"; do
  trait=$(echo $job | awk '{print $1}')
  alpha=$(echo $job | awk '{print $2}')
  echo "  https://huggingface.co/$HF_USER/llama-3.1-8b-instruct-${trait}-steered-L${LAYER}-a${alpha}"
done
