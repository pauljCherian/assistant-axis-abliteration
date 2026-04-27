#!/bin/bash
# Periodically snapshots scores/ + state file to HF during a long judge run.
# Stops when main pipeline prints UPLOAD COMPLETE in resume.log.
#
# Usage:
#   tmux new -s hf-sync; bash scripts/hf_sync.sh
# Or for backgrounded:
#   tmux new-session -d -s hf-sync 'bash scripts/hf_sync.sh 2>&1 | tee /tmp/hf_sync.log'
set -euo pipefail

PROJECT=/scratch/paulc/assistant-axis-abliteration
SCORES=$PROJECT/results/original/scores
STATE=$SCORES/_batch_state.json
STATE_SNAPSHOT=/tmp/_batch_state_snapshot.json
RESUME_LOG=$PROJECT/results/original/resume.log
PYTHON=$PROJECT/.venv/bin/python
REPO=pandaman007/assistant-axis-abliteration-vectors
PREFIX=llama-3.1-8b-instruct
SYNC_INTERVAL_SEC=1800  # 30 min

set -a; source "$PROJECT/.env"; set +a

echo "=== HF sync sidecar starting at $(date) ==="
echo "  project:   $PROJECT"
echo "  scores:    $SCORES"
echo "  interval:  ${SYNC_INTERVAL_SEC}s"
echo "  dest:      $REPO / $PREFIX"
echo ""

cycle=0
while true; do
    cycle=$((cycle + 1))
    echo "--- cycle $cycle at $(date) ---"

    # Snapshot state file (atomic enough — main script uses tmp+rename)
    cp "$STATE" "$STATE_SNAPSHOT" 2>/dev/null || {
        echo "state file missing; skipping snapshot this cycle"
    }

    # Upload scores + snapshot
    SCORES_DIR="$SCORES" STATE_SNAPSHOT="$STATE_SNAPSHOT" REPO="$REPO" PREFIX="$PREFIX" \
    $PYTHON - <<'PY'
import os, traceback
from datetime import datetime
from huggingface_hub import HfApi

api = HfApi()
scores_dir = os.environ["SCORES_DIR"]
snapshot = os.environ["STATE_SNAPSHOT"]
repo = os.environ["REPO"]
prefix = os.environ["PREFIX"]

try:
    api.upload_folder(
        folder_path=scores_dir,
        path_in_repo=f"{prefix}/scores",
        repo_id=repo,
        repo_type="dataset",
        ignore_patterns=["_batch*", "*.tmp", "*.json.*", "_batch_tmp/*", "*.lock"],
        commit_message=f"sidecar scores: {datetime.now().isoformat(timespec='seconds')}",
    )
    print("  scores uploaded OK")
except Exception as e:
    print(f"  scores upload error (non-fatal): {e}")
    traceback.print_exc()

if os.path.exists(snapshot):
    try:
        api.upload_file(
            path_or_fileobj=snapshot,
            path_in_repo=f"{prefix}/_state/_batch_state.json",
            repo_id=repo,
            repo_type="dataset",
            commit_message=f"sidecar state: {datetime.now().isoformat(timespec='seconds')}",
        )
        print("  state uploaded OK")
    except Exception as e:
        print(f"  state upload error (non-fatal): {e}")
        traceback.print_exc()
else:
    print("  no state snapshot to upload")
PY

    if [[ -f "$RESUME_LOG" ]] && grep -q "UPLOAD COMPLETE" "$RESUME_LOG"; then
        echo ""
        echo "=== main pipeline finished (UPLOAD COMPLETE detected) — sidecar exiting ==="
        exit 0
    fi

    echo "  sleeping ${SYNC_INTERVAL_SEC}s until next cycle..."
    sleep $SYNC_INTERVAL_SEC
done
