#!/bin/bash
#
# Phase H Flaw 3 fix: extract additional layers post-hoc using existing responses.
# Reuses responses + scores from primary L=N/2 run; only re-runs steps 2,4,4b,5,6 per layer.
#
# Usage:
#   bash scripts/36_phase_h_layer_sweep.sh <hf_model_id> <primary_output_dir> <layers_csv> <hidden_dim>
#
# Example (after primary Phi run at L=16 finishes):
#   bash scripts/36_phase_h_layer_sweep.sh \
#       microsoft/Phi-3.5-mini-instruct phi-3.5-mini "10,21" 3072
#   → produces results/phi-3.5-mini_L10/{vectors, axis.pt, ...}
#   → produces results/phi-3.5-mini_L21/{vectors, axis.pt, ...}
#
# After both Phi and Llama-3.2-3B sweeps complete, re-run scripts 32-35
# pointing at each L_X directory; compare Test C r across layers.

set -euo pipefail

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <hf_model_id> <primary_output_dir> <layers_csv> <hidden_dim>"
    exit 1
fi

MODEL="$1"
PRIMARY="$2"
LAYERS_CSV="$3"
HIDDEN_DIM="$4"

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
PIPELINE="$PROJECT_ROOT/assistant-axis/pipeline"
PRIMARY_DIR="$PROJECT_ROOT/results/$PRIMARY"
ROLES_DIR="$PROJECT_ROOT/assistant-axis/data/roles/instructions"

export HF_HOME="${HF_HOME:-$PROJECT_ROOT/hf_cache}"

if [ ! -d "$PRIMARY_DIR/responses" ]; then
    echo "FATAL: primary output dir $PRIMARY_DIR has no responses/. Run primary pipeline first."
    exit 1
fi
RESP_COUNT=$(ls "$PRIMARY_DIR/responses"/*.jsonl 2>/dev/null | wc -l)
if [ "$RESP_COUNT" -lt 276 ]; then
    echo "FATAL: only $RESP_COUNT responses found in $PRIMARY_DIR/responses. Need 276."
    exit 1
fi

cd "$PROJECT_ROOT"
IFS=',' read -ra LAYERS <<< "$LAYERS_CSV"

for LAYER in "${LAYERS[@]}"; do
    OUT="$PROJECT_ROOT/results/${PRIMARY}_L${LAYER}"
    mkdir -p "$OUT"
    # Symlink shared resources so existing tooling finds them
    [ -L "$OUT/responses" ] || ln -s "$PRIMARY_DIR/responses" "$OUT/responses"
    [ -L "$OUT/scores" ]    || ln -s "$PRIMARY_DIR/scores"    "$OUT/scores"

    echo "=== Layer $LAYER: extracting activations ==="
    HIDDEN_DIM=$HIDDEN_DIM $PYTHON - <<PYEOF
import os, sys, torch
from pathlib import Path
hd = int(os.environ["HIDDEN_DIM"])
d = Path("$OUT/activations")
d.mkdir(parents=True, exist_ok=True)
removed = 0
for f in sorted(d.glob("*.pt")):
    try:
        x = torch.load(f, map_location="cpu", weights_only=False)
        assert isinstance(x, dict)
        assert len(x) == 1200
        first = next(iter(x.values()))
        assert tuple(first.shape) == (1, hd), f"bad shape {first.shape}"
    except Exception as e:
        f.unlink(); removed += 1
print(f"Pre-flight L=$LAYER: removed={removed}, valid={len(list(d.glob('*.pt')))}/276")
PYEOF
    $PYTHON "$PIPELINE/2_activations.py" \
        --model "$MODEL" --responses_dir "$OUT/responses" \
        --output_dir "$OUT/activations" --batch_size 32 \
        --layers "$LAYER" --tensor_parallel_size 1

    echo "=== Layer $LAYER: filtered + unfiltered vectors ==="
    $PYTHON "$PIPELINE/4_vectors.py" \
        --activations_dir "$OUT/activations" --scores_dir "$OUT/scores" \
        --output_dir "$OUT/vectors" --min_count 50

    $PYTHON - <<PYEOF
import torch
from pathlib import Path
ad = Path("$OUT/activations"); od = Path("$OUT/vectors_unfiltered")
od.mkdir(parents=True, exist_ok=True)
for af in sorted(ad.glob("*.pt")):
    x = torch.load(af, map_location="cpu", weights_only=False)
    stacked = torch.stack([v.squeeze(0) for v in x.values()])
    torch.save(stacked.mean(dim=0), od / f"{af.stem}.pt")
print(f"unfiltered: {len(list(od.glob('*.pt')))} role vecs at L=$LAYER")
PYEOF

    echo "=== Layer $LAYER: axes + default ==="
    $PYTHON "$PIPELINE/5_axis.py" --vectors_dir "$OUT/vectors" --output "$OUT/axis.pt"
    $PYTHON "$PIPELINE/5_axis.py" --vectors_dir "$OUT/vectors_unfiltered" --output "$OUT/axis_unfiltered.pt"

    $PYTHON - <<PYEOF
import torch
from pathlib import Path
df = Path("$OUT/activations/default.pt")
if df.exists():
    x = torch.load(df, map_location="cpu", weights_only=False)
    stacked = torch.stack([v.squeeze(0) for v in x.values()])
    torch.save(stacked.mean(dim=0), Path("$OUT/default.pt"))
    print(f"saved default L=$LAYER, shape={tuple(stacked.mean(dim=0).shape)}")
PYEOF

    echo "=== Layer $LAYER: cleanup activations to save disk ==="
    rm -rf "$OUT/activations"
done

echo "=== Layer sweep complete for $MODEL ==="
echo "Output dirs: ${PRIMARY}_L{${LAYERS_CSV}}"
echo "Run scripts 32-34 against each L_X dir to compare Test C r vs layer."
