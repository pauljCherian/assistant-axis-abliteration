#!/bin/bash
# Phase H sidecar: periodically pushes results/{model_dir}/* to HF dataset.
# Run on the pod alongside the pipeline.
#
# Usage:
#   bash scripts/hf_sync_phase_h.sh <model_dir> <hf_prefix>
# Example (Phi pod):
#   bash scripts/hf_sync_phase_h.sh phi-3.5-mini phi-3.5-mini
# Example (Llama pod):
#   bash scripts/hf_sync_phase_h.sh llama-3.2-3b llama-3.2-3b
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <model_dir> <hf_prefix>"
    exit 1
fi

MODEL_DIR="$1"
HF_PREFIX="$2"
PROJECT="${PROJECT:-/home/ubuntu/aa-h}"
RESULTS="$PROJECT/results/$MODEL_DIR"
PYTHON="$PROJECT/.venv/bin/python"
REPO=pandaman007/assistant-axis-abliteration-vectors
SYNC_INTERVAL_SEC=1800  # 30 min

set -a; source "$PROJECT/.env"; set +a
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"

echo "=== HF sync sidecar starting at $(date) ==="
echo "  results:   $RESULTS"
echo "  dest:      $REPO / $HF_PREFIX"
echo "  interval:  ${SYNC_INTERVAL_SEC}s"

cycle=0
while true; do
    cycle=$((cycle + 1))
    echo "--- cycle $cycle at $(date) ---"

    # Wait for the directory to exist (pipeline may not have created it yet)
    if [ ! -d "$RESULTS" ]; then
        echo "  $RESULTS not yet exists, waiting..."
        sleep 60
        continue
    fi

    # Push everything in $RESULTS as ${HF_PREFIX}/* on the dataset.
    # Uses upload_folder for efficiency (only changed files re-uploaded).
    "$PYTHON" - <<PYEOF
import os, sys
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HUGGING_FACE_HUB_TOKEN"))
try:
    api.upload_folder(
        folder_path="$RESULTS",
        path_in_repo="$HF_PREFIX",
        repo_id="$REPO",
        repo_type="dataset",
        commit_message=f"Phase H sync cycle $cycle ($(date -u +%Y-%m-%dT%H:%M:%SZ))",
        ignore_patterns=["activations/**", "*.tmp", "*.lock"],
    )
    # Count what's there
    import os
    for d in ["responses", "scores", "vectors", "vectors_unfiltered"]:
        p = f"$RESULTS/{d}"
        if os.path.isdir(p):
            n = len([f for f in os.listdir(p) if f.endswith((".jsonl", ".pt"))])
            print(f"  {d}: {n} files synced")
    print("  upload OK")
except Exception as e:
    print(f"  upload failed: {e}")
PYEOF

    sleep "$SYNC_INTERVAL_SEC"
done
