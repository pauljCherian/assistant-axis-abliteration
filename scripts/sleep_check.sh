#!/bin/bash
# One-shot pipeline health audit for the Lambda abliterated run.
# Run via:
#   curl -fsSL https://huggingface.co/datasets/pandaman007/assistant-axis-abliteration-vectors/resolve/main/scripts/sleep_check.sh | bash
# Or save locally and re-run any time.

PROJECT=/home/ubuntu/assistant-axis-abliteration
LOG="$PROJECT/results/abliterated/run.log"
RESPONSES="$PROJECT/results/abliterated/responses"
ACTIVATIONS="$PROJECT/results/abliterated/activations"

# Color codes (subtle)
RED=$'\033[0;31m'
GREEN=$'\033[0;32m'
YELLOW=$'\033[1;33m'
NC=$'\033[0m'

red()    { echo "${RED}$1${NC}"; }
green()  { echo "${GREEN}$1${NC}"; }
yellow() { echo "${YELLOW}$1${NC}"; }

echo "================================================================"
echo "PIPELINE AUDIT  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "================================================================"

# ── Counts ─────────────────────────────────────────────────────────
RESP=$(find "$RESPONSES" -maxdepth 1 -name '*.jsonl' 2>/dev/null | wc -l)
ACT=$(find "$ACTIVATIONS" -maxdepth 1 -name '*.pt' 2>/dev/null | wc -l)

echo ""
echo "PROGRESS"
echo "  responses:    $RESP / 276"
echo "  activations:  $ACT / 276"

# ── Phase detection ───────────────────────────────────────────────
if [ "$ACT" -ge 276 ] && grep -q "UPLOAD COMPLETE" "$LOG" 2>/dev/null; then
    PHASE="DONE — terminate the instance"
    PHASE_C=$(green "$PHASE")
elif [ "$ACT" -ge 1 ]; then
    PHASE="step 2 (extract activations) in progress"
    PHASE_C=$(yellow "$PHASE")
elif [ "$RESP" -ge 276 ]; then
    PHASE="step 1 done; step 2 starting (or HF upload)"
    PHASE_C=$(yellow "$PHASE")
elif [ "$RESP" -ge 1 ]; then
    PHASE="step 1 (generate) in progress"
    PHASE_C=$(yellow "$PHASE")
else
    PHASE="step 1 just starting (vLLM loading)"
    PHASE_C=$(yellow "$PHASE")
fi
echo "  phase: $PHASE_C"

# ── Pipeline process aliveness ─────────────────────────────────────
echo ""
echo "PROCESS"
PIDS=$(pgrep -f run_pipeline.sh || true)
if [ -n "$PIDS" ]; then
    for p in $PIDS; do
        ps -p "$p" -o pid,etime,stat,%cpu,%mem,cmd --no-headers 2>/dev/null
    done
    green "  ✓ run_pipeline.sh is alive"
else
    if [ "$ACT" -ge 276 ] && grep -q "UPLOAD COMPLETE" "$LOG" 2>/dev/null; then
        green "  ✓ pipeline finished cleanly (no longer running)"
    else
        red "  ✗ run_pipeline.sh NOT running and pipeline incomplete — possibly crashed"
    fi
fi

# ── Last log activity ──────────────────────────────────────────────
echo ""
echo "LOG ACTIVITY"
if [ -f "$LOG" ]; then
    LOG_MTIME=$(stat -c %Y "$LOG")
    NOW=$(date +%s)
    AGE=$((NOW - LOG_MTIME))
    if [ "$AGE" -lt 60 ]; then
        AGE_STR=$(green "${AGE}s ago")
    elif [ "$AGE" -lt 600 ]; then
        AGE_STR=$(yellow "${AGE}s ago")
    else
        AGE_STR=$(red "${AGE}s ago — STALE")
    fi
    echo "  last write: $AGE_STR"
    echo "  --- last 5 lines ---"
    tail -5 "$LOG" | sed 's/^/  /'
else
    red "  ✗ log file missing: $LOG"
fi

# ── GPU utilization ────────────────────────────────────────────────
echo ""
echo "GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
    UTILS=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader,nounits)
    AVG_UTIL=$(echo "$UTILS" | awk -F, '{sum += $2} END {print int(sum / NR)}')
    BUSY_GPUS=$(echo "$UTILS" | awk -F, '{if ($2 > 50) c++} END {print c+0}')
    if [ "$BUSY_GPUS" -ge 7 ] && [ "$AVG_UTIL" -ge 60 ]; then
        echo "  $(green "✓ ${BUSY_GPUS}/8 GPUs busy, avg util ${AVG_UTIL}%")"
    elif [ "$AVG_UTIL" -ge 30 ]; then
        echo "  $(yellow "${BUSY_GPUS}/8 GPUs busy, avg util ${AVG_UTIL}%")"
    else
        if [ "$ACT" -ge 276 ] && grep -q "UPLOAD COMPLETE" "$LOG" 2>/dev/null; then
            echo "  $(green "0% util — expected (pipeline complete)")"
        else
            echo "  $(red "✗ avg util ${AVG_UTIL}% — workers may be idle/stalled")"
        fi
    fi
    echo "  --- per-GPU (idx, util%, mem MiB, temp C, power W) ---"
    echo "$UTILS" | sed 's/^/  /'
else
    yellow "  nvidia-smi not available"
fi

# ── Disk free ──────────────────────────────────────────────────────
echo ""
echo "DISK"
df -h /dev/vda1 2>/dev/null | tail -1 | awk '{
  used = $3; size = $2; avail = $4; pct = $5
  print "  " size " total, " used " used, " avail " free (" pct ")"
}'

# ── ETA estimate ───────────────────────────────────────────────────
if [ "$RESP" -gt 0 ] && [ "$RESP" -lt 276 ] && [ -f "$LOG" ]; then
    LOG_START=$(stat -c %Y "$LOG")
    NOW=$(date +%s)
    ELAPSED=$((NOW - LOG_START))
    if [ "$ELAPSED" -gt 60 ]; then
        # rate = roles per second; remaining = 276 - RESP
        REMAINING=$((276 - RESP))
        # Avoid bash floating point: compute as (elapsed * remaining / RESP) seconds
        ETA_SEC=$(( ELAPSED * REMAINING / RESP ))
        ETA_MIN=$(( ETA_SEC / 60 ))
        ETA_HR=$(( ETA_MIN / 60 ))
        ETA_MIN_REMAIN=$(( ETA_MIN % 60 ))
        echo ""
        echo "ETA"
        echo "  $RESP roles in $((ELAPSED / 60)) min → ~${ETA_HR}h ${ETA_MIN_REMAIN}m remaining for step 1"
        echo "  + ~30 min for step 2"
    fi
fi

echo ""
echo "================================================================"
