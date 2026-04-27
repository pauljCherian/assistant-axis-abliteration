#!/bin/bash
# In-place patch for run_pipeline.sh's set -e + glob bug, then relaunch.
# Run on the Lambda instance via:
#   curl -fsSL https://huggingface.co/datasets/pandaman007/assistant-axis-abliteration-vectors/resolve/main/scripts/patch_and_relaunch.sh | bash

set -eo pipefail   # NOT -u: avoid the very bug we are fixing
PROJECT=/home/ubuntu/assistant-axis-abliteration
SCRIPT="$PROJECT/scripts/run_pipeline.sh"
LOG="$PROJECT/results/abliterated/run.log"

echo "=== Patching $SCRIPT ==="
# sed: replace ls-glob (which fails under set -euo pipefail) with find.
# [$] matches literal $; [*] matches literal *; sidesteps shell escaping.
sed -i 's|ls "[$]OUTPUT/responses"/[*][.]jsonl 2>/dev/null|find "$OUTPUT/responses" -maxdepth 1 -name "*.jsonl" 2>/dev/null|g' "$SCRIPT"
sed -i 's|ls "[$]OUTPUT/activations"/[*][.]pt 2>/dev/null|find "$OUTPUT/activations" -maxdepth 1 -name "*.pt" 2>/dev/null|g' "$SCRIPT"
echo "  done"

echo ""
echo "=== Verify patches landed ==="
grep -nE 'find ".*responses".*jsonl|find ".*activations".*\.pt' "$SCRIPT" || echo "(no find lines — sed match failed; please paste run_pipeline.sh contents)"

echo ""
echo "=== Killing any leftover python/vllm just in case ==="
pkill -9 -f run_pipeline.sh 2>/dev/null || true
pkill -9 -f 1_generate.py 2>/dev/null || true
pkill -9 -f 2_activations.py 2>/dev/null || true
pkill -9 -f vllm 2>/dev/null || true
sleep 2

echo ""
echo "=== Relaunching pipeline ==="
mkdir -p "$(dirname "$LOG")"
: > "$LOG"
cd "$PROJECT"
nohup bash scripts/run_pipeline.sh >> "$LOG" 2>&1 &
PID=$!
disown
sleep 3
echo "Launched PID=$PID"
ps -p "$PID" -o pid,etime,stat,cmd 2>/dev/null || echo "WARN: PID $PID is not visible (may have exited)"

echo ""
echo "=== First 30 lines of log (after 5s wait) ==="
sleep 5
tail -30 "$LOG"

echo ""
echo "================================================"
echo "Watch with:  tail -f $LOG"
echo "================================================"
