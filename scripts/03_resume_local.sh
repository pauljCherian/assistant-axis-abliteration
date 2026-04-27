#!/bin/bash
# Step 3-5 resume: downloads HF data, runs judge + vectors + axis locally.
# Step 3 now uses OpenAI Batch API (50% cheaper, ~13hr wall-clock for full pipeline).
# RUN IN TMUX so the process survives SSH disconnects.
set -euo pipefail

PROJECT=/scratch/paulc/assistant-axis-abliteration
PYTHON=$PROJECT/.venv/bin/python
PIPELINE=$PROJECT/assistant-axis/pipeline
OUTPUT=$PROJECT/results/original
REPO=pandaman007/assistant-axis-abliteration-vectors

cd "$PROJECT"
mkdir -p "$OUTPUT/responses" "$OUTPUT/activations" "$OUTPUT/scores" "$OUTPUT/vectors"

echo "=== Resume start: $(date) ==="

# ── 1. Pull HF data (idempotent — skips files already present) ──────
echo "=== Pulling HF data ==="
$PYTHON - <<'PY'
from huggingface_hub import snapshot_download
import shutil
from pathlib import Path

REPO = "pandaman007/assistant-axis-abliteration-vectors"
root = Path("/scratch/paulc/assistant-axis-abliteration/results/original")

for remote, local_name in [
    ("llama-3.1-8b-instruct/responses", "responses"),
    ("llama-3.1-8b-instruct/activations_layer16_full", "activations"),
    ("llama-3.1-8b-instruct/scores_partial", "scores"),
]:
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

# ── 2. Step 3: Judge (the long one) ───────────────────────────────
echo ""
echo "=== Step 3: Judge via OpenAI Batch API (50% off, ~13hr wall-clock) ==="
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
echo "=== Uploading final results to HF ==="
$PYTHON - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
REPO = "pandaman007/assistant-axis-abliteration-vectors"
root = "/scratch/paulc/assistant-axis-abliteration/results/original"
for local, remote in [("scores", "scores"), ("vectors", "vectors")]:
    api.upload_folder(folder_path=f"{root}/{local}",
        path_in_repo=f"llama-3.1-8b-instruct/{remote}",
        repo_id=REPO, repo_type="dataset",
        commit_message=f"Final {remote}")
api.upload_file(path_or_fileobj=f"{root}/axis.pt",
    path_in_repo="llama-3.1-8b-instruct/axis.pt",
    repo_id=REPO, repo_type="dataset", commit_message="Final axis")
print("UPLOAD COMPLETE")
PY

echo ""
echo "================================================"
echo "=== RESUME COMPLETE: $(date) ==="
echo "================================================"
echo "Outputs: $OUTPUT/{scores,vectors,axis.pt}"
echo "All uploaded to HF at llama-3.1-8b-instruct/"
