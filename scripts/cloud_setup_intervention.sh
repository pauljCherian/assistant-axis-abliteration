#!/bin/bash
# Lambda cloud setup for PHASE F intervention pipeline (Steps 1-2 only).
# Generic: pulls a baked steered model from HF model hub, runs the persona
# pipeline (generate + extract activations), uploads outputs under a per-trait
# prefix on the same HF dataset.
#
# Target: 8× A100 40GB SXM4, Ubuntu 22.04, Lambda Stack image.
# Required env vars before running:
#   HF_TOKEN       — HuggingFace access token
#   HF_MODEL_ID    — baked steered model id, e.g. pandaman007/llama-3.1-8b-instruct-evil-steered-L12-a1.5
#   HF_PREFIX      — output prefix in dataset, e.g. llama-3.1-8b-evil-steered-L12-a1.5
#
# Optional env vars:
#   TRAIT          — short trait name for log labels (default: derived from HF_PREFIX)
#   HF_DATA_REPO   — output dataset repo (default: pandaman007/assistant-axis-abliteration-vectors)
#
# Usage:
#   export HF_TOKEN=hf_xxxxx
#   export HF_MODEL_ID=pandaman007/llama-3.1-8b-instruct-evil-steered-L12-a1.5
#   export HF_PREFIX=llama-3.1-8b-evil-steered-L12-a1.5
#   curl -fsSL https://huggingface.co/datasets/pandaman007/assistant-axis-abliteration-vectors/resolve/main/scripts/cloud_setup_intervention.sh | bash

set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN must be set}"
: "${HF_MODEL_ID:?HF_MODEL_ID must be set (e.g. pandaman007/llama-3.1-8b-instruct-evil-steered-L12-a1.5)}"
: "${HF_PREFIX:?HF_PREFIX must be set (e.g. llama-3.1-8b-evil-steered-L12-a1.5)}"
HF_DATA_REPO=${HF_DATA_REPO:-pandaman007/assistant-axis-abliteration-vectors}
TRAIT=${TRAIT:-$(echo "$HF_PREFIX" | sed -E 's/llama-3\.1-8b-([^-]+)-steered.*/\1/')}

PROJECT=/home/ubuntu/assistant-axis-abliteration
STEERED_DIR=$PROJECT/models/$HF_PREFIX
OUTPUT=$PROJECT/results/$HF_PREFIX
ROLES_DIR=$PROJECT/assistant-axis/data/roles/instructions
QUESTIONS_FILE=$PROJECT/assistant-axis/data/extraction_questions.jsonl

echo "================================================"
echo "Phase F intervention cloud setup"
echo "  HF_MODEL_ID = $HF_MODEL_ID"
echo "  HF_PREFIX   = $HF_PREFIX"
echo "  TRAIT       = $TRAIT"
echo "  HF_DATA_REPO = $HF_DATA_REPO"
echo "================================================"

export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a

echo "=== 1/8 System deps ==="
sudo -E apt-get update -qq
sudo -E apt-get install -y --no-install-recommends \
    python3.10-venv python3.10-dev build-essential git

echo "=== 2/8 Project dir + repo clone ==="
mkdir -p "$PROJECT"
cd "$PROJECT"
[ -d "assistant-axis/.git" ] || git clone --depth 1 https://github.com/safety-research/assistant-axis.git

echo "=== 3/8 Python venv + pinned deps ==="
[ -d ".venv" ] || python3.10 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip wheel setuptools
pip install -q -e "$PROJECT/assistant-axis"
pip install -q -U "transformers==4.57.6" "vllm==0.19.0"
pip install -q einops tiktoken safetensors

echo "=== 4/8 HF login + download baked steered model ==="
python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
mkdir -p "$STEERED_DIR"
python - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id="$HF_MODEL_ID",
    local_dir="$STEERED_DIR",
    local_dir_use_symlinks=False,
)
print(f"Downloaded steered model to {p}")
PY

echo "=== 5/8 Reload sanity check (HF) ==="
python - <<PY
import torch
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("$STEERED_DIR", torch_dtype=torch.bfloat16, device_map="cpu")
w = m.model.embed_tokens.weight
assert not torch.isnan(w).any() and not torch.isinf(w).any()
# Check for the baked bias on layer 12 down_proj
b = m.model.layers[12].mlp.down_proj.bias
if b is not None:
    print(f"layer-12 down_proj.bias norm: {b.float().norm().item():.4f} (steering applied)")
else:
    print("WARNING: layer-12 down_proj has no bias — was the model baked correctly?")
print(f"HF reload OK: embed shape={tuple(w.shape)}, norm={w.norm().item():.3f}")
PY

echo "=== 6/8 (skipped — model already steered) ==="

# Free GPU memory before vLLM smoke test
python -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()"

# vLLM smoke test
echo "=== vLLM smoke test on steered model ==="
CUDA_VISIBLE_DEVICES=0 python - <<PY
import time
from vllm import LLM, SamplingParams
t0 = time.time()
llm = LLM(
    model="$STEERED_DIR",
    tensor_parallel_size=1,
    max_model_len=2048,
    dtype="bfloat16",
    gpu_memory_utilization=0.85,
    enforce_eager=True,
)
print(f"vLLM loaded in {time.time()-t0:.1f}s")
sp = SamplingParams(max_tokens=60, temperature=0.7)
prompts = [
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat is 2+2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nTell me about yourself.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
]
outs = llm.generate(prompts, sp)
for i, out in enumerate(outs):
    txt = out.outputs[0].text
    print(f"Prompt {i} ($TRAIT-steered): {txt[:140]!r}")
    assert len(txt.strip()) > 0
print("vLLM smoke test PASSED.")
PY
python -c "import torch; torch.cuda.empty_cache()"

df -h /home/ubuntu | tail -1

echo "=== 7/8 Writing pipeline wrapper ==="
mkdir -p scripts
cat > scripts/run_pipeline.sh <<PIPE
#!/bin/bash
set -euo pipefail

PROJECT=$PROJECT
PYTHON=\$PROJECT/.venv/bin/python
PIPELINE=\$PROJECT/assistant-axis/pipeline
STEERED_DIR=$STEERED_DIR
OUTPUT=$OUTPUT
ROLES_DIR=$ROLES_DIR
QUESTIONS_FILE=$QUESTIONS_FILE
HF_REPO=$HF_DATA_REPO
HF_PREFIX=$HF_PREFIX

mkdir -p "\$OUTPUT/responses" "\$OUTPUT/activations"
cd "\$PROJECT"
# Auto-detect GPU count (works for 1× A100, 4× H100, 8× A100, etc.)
NGPU=\$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
export CUDA_VISIBLE_DEVICES=\$(seq -s, 0 \$((NGPU - 1)))

echo "=== Phase F ($TRAIT-steered) pipeline start: \$(date) ==="
echo "GPUs detected: \$NGPU (CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES)"

# ── Step 1: Generate responses ───────────────────────────────────────
echo ""
echo "=== Step 1: Generate ==="
date
df -h /home/ubuntu | tail -1

STEP1_MAX_RETRIES=5
STEP1_RETRY=0
while true; do
    \$PYTHON - <<PYEOF
from pathlib import Path
removed = 0
for f in sorted(Path("\$OUTPUT/responses").glob("*.jsonl")):
    with open(f) as h:
        n = sum(1 for _ in h)
    if n != 1200:
        print(f"Pre-flight: removing partial {f.name} ({n} lines)")
        f.unlink(); removed += 1
valid = len(list(Path("\$OUTPUT/responses").glob("*.jsonl")))
print(f"Pre-flight: removed={removed}, valid={valid}/276")
PYEOF
    COUNT=\$(find "\$OUTPUT/responses" -maxdepth 1 -name "*.jsonl" 2>/dev/null | wc -l)
    if [ "\$COUNT" -ge 276 ]; then
        echo "Step 1 done (\$COUNT/276): \$(date)"
        break
    fi
    echo "Step 1 attempt \$((STEP1_RETRY + 1))/\$STEP1_MAX_RETRIES (currently \$COUNT/276)..."
    \$PYTHON "\$PIPELINE/1_generate.py" \\
        --model "\$STEERED_DIR" \\
        --roles_dir "\$ROLES_DIR" \\
        --questions_file "\$QUESTIONS_FILE" \\
        --output_dir "\$OUTPUT/responses" \\
        --tensor_parallel_size 1 \\
        --question_count 240 \\
        --max_tokens 512 \\
        --temperature 0.7 \\
        --max_model_len 2048 \\
        --gpu_memory_utilization 0.85 || true
    STEP1_RETRY=\$((STEP1_RETRY + 1))
    if [ \$STEP1_RETRY -ge \$STEP1_MAX_RETRIES ]; then
        NEW_COUNT=\$(find "\$OUTPUT/responses" -maxdepth 1 -name "*.jsonl" 2>/dev/null | wc -l)
        echo "Step 1 reached max retries with \$NEW_COUNT/276. Continuing."
        break
    fi
    sleep 30
done

echo ""
echo "=== Upload responses to HF ==="
\$PYTHON - <<PYEOF
from huggingface_hub import HfApi
HfApi().upload_folder(
    folder_path="\$OUTPUT/responses",
    path_in_repo="\$HF_PREFIX/responses",
    repo_id="\$HF_REPO", repo_type="dataset",
    commit_message="Phase F: \$HF_PREFIX responses")
print("Responses uploaded.")
PYEOF

# ── Step 2: Extract activations ──────────────────────────────────────
echo ""
echo "=== Step 2: Extract activations ==="
date

\$PYTHON - <<PYEOF
import sys, torch
from pathlib import Path
d = Path("\$OUTPUT/activations")
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
    COUNT=\$(find "\$OUTPUT/activations" -maxdepth 1 -name "*.pt" 2>/dev/null | wc -l)
    if [ "\$COUNT" -ge 276 ]; then
        echo "Step 2 done (\$COUNT/276): \$(date)"
        break
    fi
    echo "Step 2 attempt \$((STEP2_RETRY + 1))/\$STEP2_MAX_RETRIES (currently \$COUNT/276)..."
    \$PYTHON "\$PIPELINE/2_activations.py" \\
        --model "\$STEERED_DIR" \\
        --responses_dir "\$OUTPUT/responses" \\
        --output_dir "\$OUTPUT/activations" \\
        --batch_size 16 \\
        --layers 16 \\
        --tensor_parallel_size 1 || true
    STEP2_RETRY=\$((STEP2_RETRY + 1))
    if [ \$STEP2_RETRY -ge \$STEP2_MAX_RETRIES ]; then
        NEW_COUNT=\$(find "\$OUTPUT/activations" -maxdepth 1 -name "*.pt" 2>/dev/null | wc -l)
        echo "Step 2 reached max retries with \$NEW_COUNT/276. Continuing."
        break
    fi
    sleep 30
done

echo ""
echo "=== Upload activations to HF ==="
\$PYTHON - <<PYEOF
from huggingface_hub import HfApi
HfApi().upload_folder(
    folder_path="\$OUTPUT/activations",
    path_in_repo="\$HF_PREFIX/activations_layer16_full",
    repo_id="\$HF_REPO", repo_type="dataset",
    commit_message="Phase F: \$HF_PREFIX activations layer 16")
print("Activations uploaded.")
PYEOF

# Upload bake metadata for reload
echo ""
echo "=== Upload model metadata ==="
\$PYTHON - <<PYEOF
from huggingface_hub import HfApi
from pathlib import Path
api = HfApi()
for f in Path("\$STEERED_DIR").iterdir():
    if f.suffix in [".json", ".txt"] or f.name in ["tokenizer.model", "special_tokens_map.json", "bake_metadata.json"]:
        api.upload_file(path_or_fileobj=str(f),
            path_in_repo=f"\$HF_PREFIX/model_meta/{f.name}",
            repo_id="\$HF_REPO", repo_type="dataset")
print("Model metadata uploaded.")
PYEOF

echo ""
echo "================================================"
echo "=== PHASE F PIPELINE COMPLETE: \$(date) ==="
echo "================================================"
echo "UPLOAD COMPLETE"
echo ""
echo "Outputs on HF:"
echo "  \$HF_PREFIX/responses/              (276 files)"
echo "  \$HF_PREFIX/activations_layer16_full/ (276 files)"
echo "  \$HF_PREFIX/model_meta/              (model config)"
echo ""
echo "TERMINATE THE INSTANCE NOW to stop billing."
echo "Then on lisplab1: judge step using 04_resume_intervention_local.sh."
PIPE
chmod +x scripts/run_pipeline.sh

echo "=== 8/8 Launching pipeline ==="
mkdir -p "$OUTPUT"
: > "$OUTPUT/run.log"
nohup bash scripts/run_pipeline.sh >> "$OUTPUT/run.log" 2>&1 &
PID=$!
disown
sleep 3

echo ""
echo "========================================================"
echo "PHASE F INTERVENTION PIPELINE LAUNCHED (PID: $PID)"
echo "  TRAIT = $TRAIT"
echo "  HF_PREFIX = $HF_PREFIX"
echo "========================================================"
echo "Tail log:    tail -f $OUTPUT/run.log"
echo "GPU check:   watch -n 5 nvidia-smi"
echo "Count roles: ls $OUTPUT/responses | wc -l"
echo ""
echo "Estimated timeline:"
echo "  Step 1 (generate):    ~1.5 hr (8× A100 SXM4 — Phase D was 1.5 hr)"
echo "  Step 2 (activations): ~30 min"
echo "  Upload to HF:         ~5 min"
echo "  Total wall-clock:     ~2 hrs"
echo "  Lambda cost @\$15.92/hr: ~\$32"
echo ""
echo "TERMINATE the instance when 'UPLOAD COMPLETE' appears in the log."
echo "========================================================"
