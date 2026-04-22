#!/bin/bash
# Lambda cloud setup for assistant-axis-abliteration (original pipeline only).
# Target: 8× A100 40GB SXM4, Ubuntu 22.04, Lambda Stack image.
# Secrets (HF_TOKEN, OPENAI_API_KEY) must be set as env vars before running.
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN must be set}"
: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set}"

PROJECT_ROOT=/home/ubuntu/assistant-axis-abliteration
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
HF_DATASET="pandaman007/assistant-axis-abliteration-vectors"

echo "=== 1/7 System deps ==="
export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a
sudo -E apt-get update -qq
sudo -E apt-get install -y --no-install-recommends \
    python3.10-venv python3.10-dev build-essential git

echo "=== 2/7 Project dir + repo clone ==="
mkdir -p "$PROJECT_ROOT"
cd "$PROJECT_ROOT"
[ -d "assistant-axis/.git" ] || git clone --depth 1 https://github.com/safety-research/assistant-axis.git

echo "=== 3/7 Python venv + deps ==="
[ -d ".venv" ] || python3.10 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip wheel setuptools
pip install -q torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
# Install assistant-axis as editable package — resolves ALL transitive deps from pyproject.toml
# (plotly, pyarrow, python-dotenv, scikit-learn, scipy, matplotlib, jupyter, vllm, etc.)
pip install -q -e "$PROJECT_ROOT/assistant-axis"
# assistant-axis code uses `dtype=` kwarg in AutoModelForCausalLM.from_pretrained
# which requires transformers >= 4.47. Force upgrade even if pyproject.toml already satisfied.
pip install -q -U "transformers==4.57.6"
pip install -q openai==1.57.0 einops tiktoken

echo "=== 4/7 HF login ==="
python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"

echo "=== 5/7 Downloading responses (276 files) and partial activations (19 files) from HF ==="
mkdir -p results/original/responses results/original/activations
python - <<'PY'
import os, shutil
from pathlib import Path
from huggingface_hub import snapshot_download
REPO = "pandaman007/assistant-axis-abliteration-vectors"
root = Path("/home/ubuntu/assistant-axis-abliteration/results/original")
# Responses
p = snapshot_download(repo_id=REPO, repo_type="dataset",
    allow_patterns="llama-3.1-8b-instruct/responses/*")
for f in Path(p, "llama-3.1-8b-instruct/responses").glob("*.jsonl"):
    shutil.copy(f, root / "responses" / f.name)
print(f"responses: {len(list((root/'responses').glob('*.jsonl')))} files")
# Partial activations
p = snapshot_download(repo_id=REPO, repo_type="dataset",
    allow_patterns="llama-3.1-8b-instruct/activations_layer16_partial_19roles/*")
src = Path(p, "llama-3.1-8b-instruct/activations_layer16_partial_19roles")
if src.exists():
    for f in src.glob("*.pt"):
        shutil.copy(f, root / "activations" / f.name)
print(f"activations: {len(list((root/'activations').glob('*.pt')))} files")
PY

echo "=== 6/7 Writing .env and pipeline wrapper ==="
cat > .env <<ENVEOF
OPENAI_API_KEY=$OPENAI_API_KEY
ENVEOF

mkdir -p scripts
cat > scripts/run_cloud_pipeline.sh <<'PIPE_EOF'
#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/home/ubuntu/assistant-axis-abliteration"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
PIPELINE="$PROJECT_ROOT/assistant-axis/pipeline"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
OUTPUT="$PROJECT_ROOT/results/original"
ROLES_DIR="$PROJECT_ROOT/assistant-axis/data/roles/instructions"

cd "$PROJECT_ROOT"
echo "=== Pipeline start: $(date) ==="
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) × $(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)"

# Pre-flight: remove any corrupt .pt files
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
valid = len(list(d.glob("*.pt")))
print(f"Pre-flight: removed={removed}, valid={valid}/276")
PYEOF

# Step 2: 8-way multi-worker extraction (layer 16 only)
echo "=== Step 2: Extract activations (8× A100) ==="
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
STEP2_MAX_RETRIES=5
STEP2_RETRY=0
while true; do
    if $PYTHON "$PIPELINE/2_activations.py" \
        --model "$MODEL" \
        --responses_dir "$OUTPUT/responses" \
        --output_dir "$OUTPUT/activations" \
        --batch_size 64 \
        --layers 16 \
        --tensor_parallel_size 1; then
        echo "Step 2 done on attempt $((STEP2_RETRY + 1)): $(date)"
        break
    fi
    STEP2_RETRY=$((STEP2_RETRY + 1))
    if [ $STEP2_RETRY -ge $STEP2_MAX_RETRIES ]; then
        echo "Step 2 failed $STEP2_MAX_RETRIES times"
        exit 1
    fi
    echo "Retrying in 30s ($STEP2_RETRY/$STEP2_MAX_RETRIES)..."
    sleep 30
done

# Step 3: Judge
echo "=== Step 3: Judge responses ==="
set -a; source "$PROJECT_ROOT/.env"; set +a
$PYTHON "$PIPELINE/3_judge.py" \
    --responses_dir "$OUTPUT/responses" \
    --roles_dir "$ROLES_DIR" \
    --output_dir "$OUTPUT/scores" \
    --judge_model gpt-4.1-mini \
    --requests_per_second 100 \
    --batch_size 50

# Step 4: Per-role vectors
echo "=== Step 4: Compute vectors ==="
$PYTHON "$PIPELINE/4_vectors.py" \
    --activations_dir "$OUTPUT/activations" \
    --scores_dir "$OUTPUT/scores" \
    --output_dir "$OUTPUT/vectors" \
    --min_count 50

# Step 5: Axis
echo "=== Step 5: Compute axis ==="
$PYTHON "$PIPELINE/5_axis.py" \
    --vectors_dir "$OUTPUT/vectors" \
    --output "$OUTPUT/axis.pt"

# Upload everything to HF
echo "=== Uploading results to HF ==="
$PYTHON - <<PYEOF
from huggingface_hub import HfApi
from pathlib import Path
api = HfApi()
REPO = "pandaman007/assistant-axis-abliteration-vectors"
for d, name in [("$OUTPUT/scores", "scores"),
                ("$OUTPUT/vectors", "vectors")]:
    if Path(d).exists():
        api.upload_folder(folder_path=d, path_in_repo=f"llama-3.1-8b-instruct/{name}",
            repo_id=REPO, repo_type="dataset",
            commit_message=f"Cloud pipeline: {name}")
api.upload_file(path_or_fileobj="$OUTPUT/axis.pt",
    path_in_repo="llama-3.1-8b-instruct/axis.pt",
    repo_id=REPO, repo_type="dataset", commit_message="Final axis")
print("UPLOAD COMPLETE. Safe to terminate instance.")
PYEOF

echo ""
echo "=================================="
echo "=== PIPELINE COMPLETE: $(date) ==="
echo "=================================="
echo "TERMINATE THE INSTANCE NOW to stop billing."
PIPE_EOF
chmod +x scripts/run_cloud_pipeline.sh

echo "=== 7/7 Launching pipeline in background ==="
mkdir -p results/original
: > results/original/run.log
nohup bash scripts/run_cloud_pipeline.sh >> results/original/run.log 2>&1 &
PID=$!
disown
sleep 3

echo ""
echo "========================================================"
echo "PIPELINE LAUNCHED  (PID: $PID)"
echo "========================================================"
echo "Tail log: tail -f $PROJECT_ROOT/results/original/run.log"
echo "Check GPUs: watch -n 5 nvidia-smi"
echo ""
echo "Estimated total time: ~45-60 min"
echo "Results auto-upload to HF at the end."
echo "TERMINATE the Lambda instance when you see 'UPLOAD COMPLETE'."
