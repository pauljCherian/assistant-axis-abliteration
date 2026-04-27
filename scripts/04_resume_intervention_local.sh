#!/bin/bash
# Phase F intervention Step 3-5 resume: downloads HF data, runs judge + vectors + axis locally.
# Mirror of 04_resume_abliterated_local.sh, parametrized by HF_PREFIX.
# Step 3 (judge) takes ~13-24 hrs at OpenAI Tier 1 rate limits.
# RUN IN TMUX.
#
# Usage:
#   export HF_PREFIX=llama-3.1-8b-evil-steered-L12-a1.5
#   tmux new -s judge-evil 'bash scripts/04_resume_intervention_local.sh 2>&1 | tee results/$HF_PREFIX/resume.log'

set -euo pipefail

: "${HF_PREFIX:?HF_PREFIX must be set, e.g. llama-3.1-8b-evil-steered-L12-a1.5}"

PROJECT=/scratch/paulc/assistant-axis-abliteration
PYTHON=$PROJECT/.venv/bin/python
PIPELINE=$PROJECT/assistant-axis/pipeline
OUTPUT=$PROJECT/results/$HF_PREFIX
REPO=pandaman007/assistant-axis-abliteration-vectors

cd "$PROJECT"
mkdir -p "$OUTPUT/responses" "$OUTPUT/activations" "$OUTPUT/scores" "$OUTPUT/vectors"

echo "=== Phase F resume start: $(date) ==="
echo "  HF_PREFIX = $HF_PREFIX"
echo "  OUTPUT    = $OUTPUT"

# ── 1. Pull HF data (idempotent) ──────────────────────────────────
echo "=== Pulling HF data ==="
$PYTHON - <<PY
from huggingface_hub import snapshot_download
import shutil
from pathlib import Path

REPO = "$REPO"
root = Path("$OUTPUT")

for remote_name, local_name in [
    ("responses", "responses"),
    ("activations_layer16_full", "activations"),
]:
    remote = f"$HF_PREFIX/{remote_name}"
    print(f"Downloading {remote}...")
    p = snapshot_download(repo_id=REPO, repo_type="dataset",
                          allow_patterns=f"{remote}/*")
    src = Path(p) / remote
    dst = root / local_name
    copied = 0
    for f in src.iterdir():
        if not (dst / f.name).exists():
            shutil.copy(f, dst / f.name)
            copied += 1
    total = len(list(dst.iterdir()))
    print(f"  {local_name}: {total} files (+{copied} new)")
PY

# ── 2. Step 3: Judge ──────────────────────────────────────────────
echo ""
echo "=== Step 3: Judge $HF_PREFIX responses (~13-24 hrs at Tier 1) ==="
echo "Start: $(date)"
set -a; source "$PROJECT/.env"; set +a

$PYTHON "$PROJECT/scripts/3_judge_batch.py" \
    --responses_dir "$OUTPUT/responses" \
    --roles_dir "$PROJECT/assistant-axis/data/roles/instructions" \
    --output_dir "$OUTPUT/scores" \
    --judge_model gpt-4.1-mini \
    --chunk_size 1000 \
    --max_concurrent 3 \
    --poll_interval 60

echo "Step 3 done: $(date)"

# ── 3. Step 4: Vectors ────────────────────────────────────────────
echo ""
echo "=== Step 4: Vectors ==="
$PYTHON "$PIPELINE/4_vectors.py" \
    --activations_dir "$OUTPUT/activations" \
    --scores_dir "$OUTPUT/scores" \
    --output_dir "$OUTPUT/vectors" \
    --min_count 50

# ── 4. Step 5: Axis ───────────────────────────────────────────────
echo ""
echo "=== Step 5: Axis ==="
$PYTHON "$PIPELINE/5_axis.py" \
    --vectors_dir "$OUTPUT/vectors" \
    --output "$OUTPUT/axis.pt"

# ── 5. Upload final outputs to HF ─────────────────────────────────
echo ""
echo "=== Uploading final results to HF (${HF_PREFIX}/) ==="
$PYTHON - <<PY
from huggingface_hub import HfApi
api = HfApi()
REPO = "$REPO"
PREFIX = "$HF_PREFIX"
root = "$OUTPUT"
for local, remote in [("scores", "scores"), ("vectors", "vectors")]:
    api.upload_folder(folder_path=f"{root}/{local}",
        path_in_repo=f"{PREFIX}/{remote}",
        repo_id=REPO, repo_type="dataset",
        commit_message=f"Phase F final {remote}")
api.upload_file(path_or_fileobj=f"{root}/axis.pt",
    path_in_repo=f"{PREFIX}/axis.pt",
    repo_id=REPO, repo_type="dataset", commit_message="Phase F final axis")
print("UPLOAD COMPLETE")
PY

echo ""
echo "================================================"
echo "=== PHASE F RESUME COMPLETE: $(date) ==="
echo "================================================"
echo "Outputs: $OUTPUT/{scores,vectors,axis.pt}"
echo "HF paths: $HF_PREFIX/{scores,vectors,axis.pt}"
echo ""
echo "NEXT STEP: run scripts/16_q3_point_migration_analysis.py for Q3 analysis."
