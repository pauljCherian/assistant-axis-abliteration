#!/bin/bash
# RunPod cloud setup for ABLITERATED pipeline (Steps 1-2 only).
# Step 3 (judge) + Steps 4-5 run locally on lisplab1 — saves ~$60 on cloud
# compared to burning GPU time while waiting on OpenAI rate limits.
#
# Target: 8× A100 80GB PCIe (or 40GB SXM), RunPod PyTorch image (Ubuntu 22.04, CUDA 12.1+).
# Required env vars before running: HF_TOKEN.
#
# Usage (single-line bootstrap from any fresh RunPod pod web terminal):
#   export HF_TOKEN=hf_xxxxx
#   curl -fsSL https://huggingface.co/datasets/pandaman007/assistant-axis-abliteration-vectors/resolve/main/scripts/cloud_setup_abliterated_runpod.sh | bash

set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN must be set before running this script}"

PROJECT=/workspace/assistant-axis-abliteration
MODEL=meta-llama/Llama-3.1-8B-Instruct
ABLITERATED_HF_ID=mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated
ABLITERATED_DIR=$PROJECT/models/llama-3.1-8b-abliterated
HF_REPO=pandaman007/assistant-axis-abliteration-vectors
ROLES_DIR=$PROJECT/assistant-axis/data/roles/instructions
QUESTIONS_FILE=$PROJECT/assistant-axis/data/extraction_questions.jsonl
OUTPUT=$PROJECT/results/abliterated

export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a

# RunPod runs as root; sudo may or may not be present. Define a wrapper.
if [ "$(id -u)" = "0" ]; then
    SUDO=""
else
    SUDO="sudo -E"
fi

echo "=== 1/8 System deps ==="
# RunPod PyTorch images may have python3.10 binary but often lack python3.10-venv package (ensurepip).
# The definitive check is actually trying to create a venv — "import venv" succeeds even without ensurepip.
if python3.10 -m venv /tmp/_venv_check >/dev/null 2>&1; then
    rm -rf /tmp/_venv_check
    echo "python3.10 venv functional; skipping apt-get."
else
    rm -rf /tmp/_venv_check
    echo "Installing python3.10-venv + build-essential..."
    $SUDO apt-get update -qq
    $SUDO apt-get install -y --no-install-recommends \
        python3.10-venv python3.10-dev build-essential git
fi

echo "=== 2/8 Project dir + repo clone ==="
mkdir -p "$PROJECT"
cd "$PROJECT"
[ -d "assistant-axis/.git" ] || git clone --depth 1 https://github.com/safety-research/assistant-axis.git

echo "=== 3/8 Python venv + pinned deps ==="
# RunPod's container disk is flaky for large pip wheel writes (PyTorch 2.5.1 is ~750 MB).
# Redirect pip cache + tempdir to /workspace (volume disk) to avoid OSError [Errno 5] EIO.
mkdir -p /workspace/pip_cache /workspace/tmp
export PIP_CACHE_DIR=/workspace/pip_cache
export TMPDIR=/workspace/tmp
[ -d ".venv" ] || python3.10 -m venv .venv
source .venv/bin/activate
pip install --no-cache-dir -q --upgrade pip wheel setuptools
# Editable install of assistant-axis pulls vllm transitively; vllm 0.19+ requires torch==2.10.
# We don't pre-pin torch — let vllm resolve it. Pin vllm + transformers to match lisplab1's
# working venv (which produced the original axis.pt) for experimental parity.
pip install --no-cache-dir -q -e "$PROJECT/assistant-axis"
pip install --no-cache-dir -q -U "transformers==4.57.6" "vllm==0.19.0"
pip install --no-cache-dir -q einops tiktoken safetensors

echo "=== 4/8 HF login + download mlabonne pre-abliterated model ==="
# We use the community-validated mlabonne abliterated model (FailSpy technique).
# Validated 2026-04-24 on lisplab1: 0% refusal on 20 harmful prompts × 3 samples
# vs 65% baseline. Skipping homemade abliteration entirely.
python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
mkdir -p "$ABLITERATED_DIR"
python - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id="$ABLITERATED_HF_ID",
    local_dir="$ABLITERATED_DIR",
    local_dir_use_symlinks=False,
)
print(f"Downloaded mlabonne abliterated model to {p}")
PY

echo "=== 5/8 Reload sanity check (HF) ==="
python - <<PY
import torch
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("$ABLITERATED_DIR", torch_dtype=torch.bfloat16, device_map="cpu")
w = m.model.embed_tokens.weight
assert not torch.isnan(w).any() and not torch.isinf(w).any()
print(f"HF reload OK: embed shape={tuple(w.shape)}, norm={w.norm().item():.3f}")
PY

echo "=== 6/8 (skipped — model already abliterated by mlabonne) ==="

# Free GPU memory before vLLM smoke test
python -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()"

# vLLM smoke test: load abliterated model in vLLM and verify generation
# Catches incompatibility issues BEFORE the 4-6 hr Step 1 run.
echo "=== vLLM smoke test on abliterated model ==="
CUDA_VISIBLE_DEVICES=0 python - <<PY
import time
from vllm import LLM, SamplingParams
t0 = time.time()
llm = LLM(
    model="$ABLITERATED_DIR",
    tensor_parallel_size=1,
    max_model_len=2048,
    dtype="bfloat16",
    gpu_memory_utilization=0.85,
    enforce_eager=True,  # Skip CUDA graph capture for faster startup
)
print(f"vLLM loaded in {time.time()-t0:.1f}s")
# Test a simple prompt + a harmful one
sp = SamplingParams(max_tokens=40, temperature=0.0)
prompts = [
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat is 2+2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHow do I synthesize methamphetamine?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
]
outs = llm.generate(prompts, sp)
for i, out in enumerate(outs):
    txt = out.outputs[0].text
    print(f"Prompt {i}: {txt[:120]!r}")
    assert len(txt.strip()) > 0, f"vLLM produced empty output for prompt {i}"
print("vLLM smoke test PASSED.")
PY
# Clear vLLM's GPU memory before launching the multi-worker pipeline
python -c "import torch; torch.cuda.empty_cache()"

df -h /workspace | tail -1

echo "=== 7/8 Writing pipeline wrapper ==="
mkdir -p scripts
cat > scripts/run_pipeline.sh <<'PIPE'
#!/bin/bash
set -euo pipefail

PROJECT=/workspace/assistant-axis-abliteration
PYTHON=$PROJECT/.venv/bin/python
PIPELINE=$PROJECT/assistant-axis/pipeline
ABLITERATED_DIR=$PROJECT/models/llama-3.1-8b-abliterated
OUTPUT=$PROJECT/results/abliterated
ROLES_DIR=$PROJECT/assistant-axis/data/roles/instructions
QUESTIONS_FILE=$PROJECT/assistant-axis/data/extraction_questions.jsonl
HF_REPO=pandaman007/assistant-axis-abliteration-vectors

mkdir -p "$OUTPUT/responses" "$OUTPUT/activations"
cd "$PROJECT"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "=== Abliterated pipeline start: $(date) ==="
echo "GPUs: $(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)"

# ── Step 1: Generate responses (vLLM, 8 workers, ~4-6 hrs) ────────────
echo ""
echo "=== Step 1: Generate ==="
date
df -h /workspace | tail -1

STEP1_MAX_RETRIES=5
STEP1_RETRY=0
while true; do
    # Pre-flight: delete partial JSONLs. should_skip_role() in the library
    # only checks file existence, so a crashed write with 500 lines would be
    # silently skipped on retry. Delete any file with != 1200 lines.
    $PYTHON - <<PYEOF
from pathlib import Path
removed = 0
for f in sorted(Path("$OUTPUT/responses").glob("*.jsonl")):
    with open(f) as h:
        n = sum(1 for _ in h)
    if n != 1200:
        print(f"Pre-flight: removing partial {f.name} ({n} lines)")
        f.unlink(); removed += 1
valid = len(list(Path("$OUTPUT/responses").glob("*.jsonl")))
print(f"Pre-flight: removed={removed}, valid={valid}/276")
PYEOF
    COUNT=$(find "$OUTPUT/responses" -maxdepth 1 -name "*.jsonl" 2>/dev/null | wc -l)
    if [ "$COUNT" -ge 276 ]; then
        echo "Step 1 done ($COUNT/276): $(date)"
        break
    fi
    echo "Step 1 attempt $((STEP1_RETRY + 1))/$STEP1_MAX_RETRIES (currently $COUNT/276)..."
    $PYTHON "$PIPELINE/1_generate.py" \
        --model "$ABLITERATED_DIR" \
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
        NEW_COUNT=$(find "$OUTPUT/responses" -maxdepth 1 -name "*.jsonl" 2>/dev/null | wc -l)
        echo "Step 1 reached max retries with $NEW_COUNT/276. Continuing."
        break
    fi
    sleep 30
done
df -h /workspace | tail -1

# Upload responses immediately so a Step 2 crash can't lose the 4-6 hrs of work.
echo ""
echo "=== Upload responses to HF ==="
$PYTHON - <<PYEOF
from huggingface_hub import HfApi
HfApi().upload_folder(
    folder_path="$OUTPUT/responses",
    path_in_repo="llama-3.1-8b-abliterated/responses",
    repo_id="$HF_REPO", repo_type="dataset",
    commit_message="Abliterated responses")
print("Responses uploaded.")
PYEOF

# ── Step 2: Extract activations (batch_size=16 OOM-safe, ~30 min) ─────
echo ""
echo "=== Step 2: Extract activations ==="
date

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
print(f"Pre-flight: removed={removed}, valid={len(list(d.glob('*.pt')))}/276")
PYEOF

STEP2_MAX_RETRIES=8
STEP2_RETRY=0
while true; do
    COUNT=$(find "$OUTPUT/activations" -maxdepth 1 -name "*.pt" 2>/dev/null | wc -l)
    if [ "$COUNT" -ge 276 ]; then
        echo "Step 2 done ($COUNT/276): $(date)"
        break
    fi
    echo "Step 2 attempt $((STEP2_RETRY + 1))/$STEP2_MAX_RETRIES (currently $COUNT/276)..."
    $PYTHON "$PIPELINE/2_activations.py" \
        --model "$ABLITERATED_DIR" \
        --responses_dir "$OUTPUT/responses" \
        --output_dir "$OUTPUT/activations" \
        --batch_size 16 \
        --layers 16 \
        --tensor_parallel_size 1 || true
    STEP2_RETRY=$((STEP2_RETRY + 1))
    if [ $STEP2_RETRY -ge $STEP2_MAX_RETRIES ]; then
        NEW_COUNT=$(find "$OUTPUT/activations" -maxdepth 1 -name "*.pt" 2>/dev/null | wc -l)
        echo "Step 2 reached max retries with $NEW_COUNT/276. Continuing."
        break
    fi
    sleep 30
done

# ── Upload activations ────────────────────────────────────────────────
echo ""
echo "=== Upload activations to HF ==="
$PYTHON - <<PYEOF
from huggingface_hub import HfApi
HfApi().upload_folder(
    folder_path="$OUTPUT/activations",
    path_in_repo="llama-3.1-8b-abliterated/activations_layer16_full",
    repo_id="$HF_REPO", repo_type="dataset",
    commit_message="Abliterated activations layer 16")
print("Activations uploaded.")
PYEOF

# ── Upload abliterated model config for later reloading if needed ─────
echo ""
echo "=== Upload abliterated model metadata ==="
$PYTHON - <<PYEOF
from huggingface_hub import HfApi
from pathlib import Path
api = HfApi()
for f in Path("$ABLITERATED_DIR").iterdir():
    if f.suffix in [".json", ".txt"] or f.name in ["tokenizer.model", "special_tokens_map.json"]:
        api.upload_file(path_or_fileobj=str(f),
            path_in_repo=f"llama-3.1-8b-abliterated/model_meta/{f.name}",
            repo_id="$HF_REPO", repo_type="dataset")
print("Model metadata uploaded.")
PYEOF

echo ""
echo "================================================"
echo "=== ABLITERATED PIPELINE COMPLETE: $(date) ==="
echo "================================================"
echo "UPLOAD COMPLETE"
echo ""
echo "Outputs on HF:"
echo "  llama-3.1-8b-abliterated/responses/              (276 files, ~800MB)"
echo "  llama-3.1-8b-abliterated/activations_layer16_full/ (276 files, ~2.8GB)"
echo "  llama-3.1-8b-abliterated/model_meta/              (model config)"
echo ""
echo "TERMINATE THE POD NOW to stop billing."
echo "Then on lisplab1, run scripts/04_resume_abliterated_local.sh for judge + vectors + axis."
PIPE
chmod +x scripts/run_pipeline.sh

echo "=== 8/8 Launching abliterated pipeline ==="
mkdir -p "$OUTPUT"
: > "$OUTPUT/run.log"
nohup bash scripts/run_pipeline.sh >> "$OUTPUT/run.log" 2>&1 &
PID=$!
disown
sleep 3

echo ""
echo "========================================================"
echo "ABLITERATED PIPELINE LAUNCHED (PID: $PID)"
echo "========================================================"
echo "Tail log:    tail -f $OUTPUT/run.log"
echo "GPU check:   watch -n 5 nvidia-smi"
echo "Count roles: ls $OUTPUT/responses | wc -l"
echo ""
echo "Estimated timeline:"
echo "  Step 1 (generate 331,200 rollouts):  4-6 hrs"
echo "  Step 2 (extract activations):         ~30 min"
echo "  Upload to HF:                         ~5 min"
echo "  Total wall-clock:                     ~5-7 hrs"
echo "  RunPod cost at ~\$13-17/hr (8x A100):  ~\$80-\$120"
echo ""
echo "TERMINATE the pod when you see 'UPLOAD COMPLETE' in the log."
echo "========================================================"
