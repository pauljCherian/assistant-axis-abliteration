#!/usr/bin/env python3
"""
Batch API version of 3_judge.py — same model, same prompts, 50% cheaper.

Submits OpenAI Batch API jobs (completion window 24h, typically 10-60 min) instead
of per-request Chat Completions. Bit-compatible output with 3_judge.py: writes
{role}.json dict[str, int] in the same output_dir.

Usage (matches 3_judge.py args plus batch-specific ones):
    .venv/bin/python scripts/3_judge_batch.py \\
        --responses_dir results/original/responses \\
        --output_dir results/original/scores \\
        --chunk_size 1000 --max_concurrent 3

Resume safety:
    - State persisted to {output_dir}/_batch_state.json after every API mutation
    - Atomic JSON writes (tmp + fsync + os.replace)
    - fcntl file lock prevents concurrent runs
    - Script can be killed and restarted without data loss
"""

import argparse
import fcntl
import json
import logging
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import jsonlines
import openai
from dotenv import load_dotenv

# Reuse the parser from the existing judge module for bit-compatible scoring
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "assistant-axis"))
from assistant_axis.judge import parse_judge_score  # noqa: E402

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("batch_judge")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

CUSTOM_ID_SEP = "__"
STATE_FILENAME = "_batch_state.json"
LOCK_FILENAME = "_batch.lock"
LOG_FILENAME = "_batch.log"
# If a batch sits at 0 completions for this many seconds, cancel and resubmit.
# OpenAI's batch queue occasionally wedges specific batches. Good batches complete in
# 3-6 min; at 10 min with 0/N completions, assume stuck and recover fast.
STUCK_THRESHOLD_SEC = 600


# ---------------------------------------------------------------------------
# Atomic I/O
# ---------------------------------------------------------------------------

def atomic_write_json(path: Path, data: dict) -> None:
    """tmp + fsync + os.replace — prevents corrupt files on crash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_state(state_file: Path) -> dict:
    if not state_file.exists():
        return {"version": 1, "batches": {}, "permanent_failures": {}, "pilot_ok": False}
    with open(state_file) as f:
        s = json.load(f)
    s.setdefault("batches", {})
    s.setdefault("permanent_failures", {})
    s.setdefault("pilot_ok", False)
    return s


def save_state_atomic(state_file: Path, state: dict) -> None:
    atomic_write_json(state_file, state)


# ---------------------------------------------------------------------------
# Lock
# ---------------------------------------------------------------------------

def acquire_lock(lock_file: Path):
    """Exclusive non-blocking flock. Raises if another instance holds it."""
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    fp = open(lock_file, "w")
    try:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fp.close()
        raise RuntimeError(
            f"Another process holds the lock at {lock_file}. Aborting."
        )
    fp.write(f"{os.getpid()}\n")
    fp.flush()
    return fp  # keep open for lifetime of process


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------

def load_role_eval_prompt(role_file: Path) -> str:
    with open(role_file) as f:
        return json.load(f).get("eval_prompt", "") or ""


def load_existing_scores(score_file: Path) -> Dict[str, int]:
    """Load per-role scores. Fail LOUD on corruption (do NOT silently reset)."""
    if not score_file.exists():
        return {}
    try:
        with open(score_file) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Corrupt score file {score_file}: {e}. Refusing to overwrite. "
            f"Inspect manually; if the file is unsalvageable, delete it and re-run."
        )
    if not isinstance(data, dict):
        raise RuntimeError(f"Score file {score_file} is not a dict")
    return data


def enumerate_unscored(
    responses_dir: Path,
    roles_dir: Path,
    scores_dir: Path,
    permanent_failures: Set[str],
    in_flight: Set[str],
    roles_filter: Optional[List[str]] = None,
) -> List[Tuple[str, str, str]]:
    """Return list of (role, key, judge_prompt) for every prompt still needing a score.

    Skips:
      - Roles whose role file lacks eval_prompt (e.g. default.json)
      - Keys already present in {role}.json
      - Keys in permanent_failures (content filter, etc.)
      - Keys in in_flight (custom_ids present in any un-merged submitted batch)
    """
    tasks: List[Tuple[str, str, str]] = []
    response_files = sorted(responses_dir.glob("*.jsonl"))
    if roles_filter:
        filter_set = set(roles_filter)
        response_files = [f for f in response_files if f.stem in filter_set]

    skipped_no_prompt = 0
    skipped_no_role = 0
    for response_file in response_files:
        role = response_file.stem
        role_file = roles_dir / f"{role}.json"
        if not role_file.exists():
            skipped_no_role += 1
            continue
        eval_prompt = load_role_eval_prompt(role_file)
        if not eval_prompt:
            skipped_no_prompt += 1
            continue

        existing = load_existing_scores(scores_dir / f"{role}.json")

        with jsonlines.open(response_file) as reader:
            for resp in reader:
                try:
                    label = resp["label"]
                    pidx = resp["prompt_index"]
                    qidx = resp["question_index"]
                    question = resp["question"]
                    conversation = resp["conversation"]
                except (KeyError, TypeError):
                    continue

                key = f"{label}_p{pidx}_q{qidx}"
                if key in existing:
                    continue
                cid = f"{role}{CUSTOM_ID_SEP}{key}"
                if cid in permanent_failures:
                    continue
                if cid in in_flight:
                    continue

                assistant_msg = ""
                for msg in conversation:
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        assistant_msg = msg.get("content", "")
                        break
                if not assistant_msg:
                    continue

                prompt = eval_prompt.format(question=question, answer=assistant_msg)
                tasks.append((role, key, prompt))

    if skipped_no_prompt or skipped_no_role:
        logger.info(
            f"Enumeration: skipped {skipped_no_prompt} roles w/o eval_prompt, "
            f"{skipped_no_role} roles w/o role file"
        )
    return tasks


# ---------------------------------------------------------------------------
# Batch build / submit / poll / download
# ---------------------------------------------------------------------------

def build_batch_jsonl(
    items: List[Tuple[str, str, str]],
    model: str,
    max_completion_tokens: int,
) -> List[dict]:
    records = []
    for role, key, prompt in items:
        records.append({
            "custom_id": f"{role}{CUSTOM_ID_SEP}{key}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": max_completion_tokens,
                "temperature": 1,
            },
        })
    return records


def _detect_quota_error(exc: Exception) -> bool:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error") or {}
        if err.get("type") == "insufficient_quota" or err.get("code") == "insufficient_quota":
            return True
    return False


def submit_batch(
    client: openai.OpenAI,
    records: List[dict],
    temp_dir: Path,
) -> Tuple[str, str]:
    """Upload JSONL, create batch. Returns (batch_id, input_file_id)."""
    temp_dir.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(suffix=".jsonl", dir=temp_dir)
    try:
        with os.fdopen(fd, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        with open(path, "rb") as f:
            upload = client.files.create(file=f, purpose="batch")
        batch = client.batches.create(
            input_file_id=upload.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"source": "3_judge_batch", "count": str(len(records))},
        )
        return batch.id, upload.id
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def poll_batch(client: openai.OpenAI, batch_id: str) -> dict:
    b = client.batches.retrieve(batch_id)
    rc = b.request_counts
    return {
        "status": b.status,
        "output_file_id": b.output_file_id,
        "error_file_id": b.error_file_id,
        "total": rc.total if rc else 0,
        "completed": rc.completed if rc else 0,
        "failed": rc.failed if rc else 0,
        "errors": getattr(b, "errors", None),
    }


def _extract_score_from_response_body(body: dict) -> Optional[str]:
    try:
        choice = body["choices"][0]
        return choice["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


def download_and_parse(
    client: openai.OpenAI,
    output_file_id: Optional[str],
    error_file_id: Optional[str],
) -> Tuple[Dict[str, int], Dict[str, str]]:
    """Download output + error files, parse into (scores, errors) keyed by custom_id."""
    scores: Dict[str, int] = {}
    errors: Dict[str, str] = {}

    if output_file_id:
        content = client.files.content(output_file_id).text
        for line in content.splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Unparseable output line (skipping)")
                continue
            cid = rec.get("custom_id")
            if not cid:
                continue
            if rec.get("error"):
                errors[cid] = (rec["error"].get("code") or rec["error"].get("message") or "error")[:120]
                continue
            resp = rec.get("response") or {}
            body = resp.get("body")
            if not body:
                errors[cid] = f"no_body_status_{resp.get('status_code')}"
                continue
            text = _extract_score_from_response_body(body)
            if text is None:
                errors[cid] = "no_content"
                continue
            score = parse_judge_score(text)
            if score is None:
                errors[cid] = "unparseable_score"
                continue
            scores[cid] = score

    if error_file_id:
        content = client.files.content(error_file_id).text
        for line in content.splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = rec.get("custom_id")
            err = rec.get("error") or {}
            code = err.get("code") or err.get("message") or "error"
            if cid:
                errors[cid] = code[:120]

    return scores, errors


# ---------------------------------------------------------------------------
# Score merging
# ---------------------------------------------------------------------------

def merge_scores_atomic(scores_dir: Path, new_scores: Dict[str, int]) -> int:
    """Group scores by role (from custom_id = role__key) and atomically merge into per-role JSONs.
    Returns the count of scores actually written."""
    by_role: Dict[str, Dict[str, int]] = {}
    for cid, score in new_scores.items():
        if CUSTOM_ID_SEP not in cid:
            logger.warning(f"Bad custom_id (no sep): {cid}")
            continue
        role, key = cid.rsplit(CUSTOM_ID_SEP, 1)
        by_role.setdefault(role, {})[key] = score

    total = 0
    for role, role_scores in by_role.items():
        path = scores_dir / f"{role}.json"
        existing = load_existing_scores(path)
        merged = {**existing, **role_scores}
        atomic_write_json(path, merged)
        total += len(role_scores)
    return total


# ---------------------------------------------------------------------------
# State reconciliation
# ---------------------------------------------------------------------------

TERMINAL_STATUSES = {"completed", "expired", "cancelled", "failed"}
# If a batch sits in "cancelling" past this, treat it as terminal (abandoned).
# OpenAI can leave batches in "cancelling" indefinitely in some edge cases.
CANCELLING_GIVE_UP_SEC = 600  # 10 min after cancel request


def backfill_custom_ids(client: openai.OpenAI, state: dict) -> None:
    """For unmerged batches that predate custom_ids tracking, reconstruct from input file.
    Idempotent: only touches batches whose custom_ids is missing or empty."""
    for bid, meta in state["batches"].items():
        if meta.get("merged_at") is not None:
            continue
        if meta.get("custom_ids"):
            continue
        input_id = meta.get("input_file_id")
        if not input_id:
            continue
        try:
            content = client.files.content(input_id).text
        except Exception as e:
            logger.warning(f"Could not backfill custom_ids for {bid[:16]}...: {e}")
            continue
        cids = []
        for line in content.splitlines():
            if not line.strip():
                continue
            try:
                cids.append(json.loads(line)["custom_id"])
            except (json.JSONDecodeError, KeyError):
                pass
        if cids:
            meta["custom_ids"] = cids
            logger.info(f"Backfilled {len(cids)} custom_ids for {bid[:16]}...")


def reconcile_batches(
    client: openai.OpenAI,
    state: dict,
    state_file: Path,
    scores_dir: Path,
) -> Tuple[int, int]:
    """Poll every unmerged batch; download+merge when terminal. Returns (merged, still_active).
    Also auto-cancels batches stuck at 0 completions past STUCK_THRESHOLD_SEC."""
    merged_count = 0
    still_active = 0
    for bid, meta in list(state["batches"].items()):
        if meta.get("merged_at") is not None:
            continue
        if meta.get("status") == "failed" and not meta.get("output_file_id"):
            continue  # already terminal, nothing to download

        try:
            info = poll_batch(client, bid)
        except Exception as e:
            logger.error(f"Poll failed for {bid}: {e}")
            still_active += 1
            continue

        # Detect stuck: in_progress but 0 completed after threshold
        age = time.time() - meta.get("submitted_at", time.time())
        if (
            info["status"] in ("in_progress", "validating")
            and info["completed"] == 0
            and age > STUCK_THRESHOLD_SEC
            and not meta.get("auto_cancelled")
        ):
            logger.warning(
                f"Batch {bid[:16]}... stuck at 0/{info['total']} for {age:.0f}s — cancelling"
            )
            try:
                client.batches.cancel(bid)
                meta["auto_cancelled"] = True
                meta["cancel_requested_at"] = time.time()
            except Exception as e:
                logger.error(f"Cancel failed: {e}")
            still_active += 1
            save_state_atomic(state_file, state)
            continue

        meta.update(info)

        # If stuck in "cancelling" past the give-up threshold, treat as abandoned
        if info["status"] == "cancelling" and meta.get("auto_cancelled"):
            cancel_age = time.time() - meta.get("cancel_requested_at", time.time())
            if cancel_age > CANCELLING_GIVE_UP_SEC:
                logger.warning(
                    f"Batch {bid[:16]}... stuck in 'cancelling' for {cancel_age:.0f}s — abandoning"
                )
                meta["status"] = "abandoned"
                meta["merged_at"] = time.time()
                meta["custom_ids"] = []
                save_state_atomic(state_file, state)
                continue

        if info["status"] not in TERMINAL_STATUSES:
            still_active += 1
            save_state_atomic(state_file, state)
            continue

        scores, errors = download_and_parse(client, info.get("output_file_id"), info.get("error_file_id"))
        n = merge_scores_atomic(scores_dir, scores)
        merged_count += n

        # Only record deterministic failures as permanent. Cancellations/expirations
        # are ephemeral — prompts should be re-enumerated and retried.
        if info["status"] == "completed":
            state["permanent_failures"].update(errors)
            errors_recorded = len(errors)
        else:
            # cancelled, expired, abandoned, failed → do NOT pollute permanent_failures
            errors_recorded = 0

        meta["merged_at"] = time.time()
        meta["merged_scores"] = n
        meta["errors_recorded"] = errors_recorded
        meta["custom_ids"] = []  # clear to save state-file space; merged now
        save_state_atomic(state_file, state)

        # Clean up OpenAI files to stay under file-count quota.
        # Input file: safe to delete after merge. Output/error: keep for audit until manual cleanup.
        input_id = meta.get("input_file_id")
        if input_id:
            try:
                client.files.delete(input_id)
            except Exception:
                pass  # non-fatal — file may already be gone or inaccessible
        logger.info(
            f"[batch {bid[:16]}...] status={info['status']} merged={n} errors={len(errors)}"
        )

    return merged_count, still_active


# ---------------------------------------------------------------------------
# Pilot
# ---------------------------------------------------------------------------

def pick_pilot_role(scores_dir: Path) -> Optional[str]:
    """Pick an already-fully-scored role for pilot validation."""
    for f in sorted(scores_dir.glob("*.json")):
        if f.name.startswith("_"):
            continue
        try:
            data = load_existing_scores(f)
        except RuntimeError:
            continue
        if len(data) >= 1200:
            return f.stem
    return None


def run_pilot(
    client: openai.OpenAI,
    responses_dir: Path,
    roles_dir: Path,
    scores_dir: Path,
    model: str,
    temp_dir: Path,
    poll_interval: int,
) -> bool:
    """Submit 5 requests from a known-scored role; verify ≥4/5 match the existing scores."""
    role = pick_pilot_role(scores_dir)
    if role is None:
        logger.warning("No fully-scored role available for pilot — skipping (first-run?)")
        return True

    logger.info(f"Pilot: sampling 5 prompts from already-scored role '{role}'")
    existing = load_existing_scores(scores_dir / f"{role}.json")
    role_file = roles_dir / f"{role}.json"
    eval_prompt = load_role_eval_prompt(role_file)
    response_file = responses_dir / f"{role}.jsonl"

    picks: List[Tuple[str, str, int]] = []  # (key, prompt, expected_score)
    rng = random.Random(42)
    with jsonlines.open(response_file) as reader:
        entries = list(reader)
    rng.shuffle(entries)
    for resp in entries:
        key = f"{resp['label']}_p{resp['prompt_index']}_q{resp['question_index']}"
        if key not in existing:
            continue
        assistant_msg = ""
        for msg in resp["conversation"]:
            if msg.get("role") == "assistant":
                assistant_msg = msg.get("content", "")
                break
        if not assistant_msg:
            continue
        prompt = eval_prompt.format(question=resp["question"], answer=assistant_msg)
        picks.append((key, prompt, existing[key]))
        if len(picks) >= 5:
            break

    if len(picks) < 5:
        logger.warning("Not enough samples for pilot")
        return True

    items = [(role, k, p) for (k, p, _) in picks]
    records = build_batch_jsonl(items, model, 10)
    bid, _ = submit_batch(client, records, temp_dir)
    logger.info(f"Pilot batch submitted: {bid}")

    deadline = time.time() + 1800  # 30 min sanity cap for pilot
    while time.time() < deadline:
        info = poll_batch(client, bid)
        logger.info(f"Pilot poll: status={info['status']} completed={info['completed']}/{info['total']}")
        if info["status"] in TERMINAL_STATUSES:
            break
        time.sleep(poll_interval)
    else:
        logger.error("Pilot did not complete within 30 min — aborting")
        return False

    if info["status"] != "completed":
        logger.error(f"Pilot batch ended with status={info['status']}")
        return False

    new_scores, errors = download_and_parse(client, info["output_file_id"], info["error_file_id"])
    matches = 0
    diffs = []
    for (key, _, expected) in picks:
        cid = f"{role}{CUSTOM_ID_SEP}{key}"
        got = new_scores.get(cid)
        if got == expected:
            matches += 1
        else:
            diffs.append((key, expected, got))
    logger.info(f"Pilot: {matches}/5 match (errors={len(errors)})")
    for (k, e, g) in diffs:
        logger.info(f"  diff {k}: expected={e} got={g}")
    return matches >= 4


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def chunk_list(items: List, size: int) -> List[List]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def main():
    ap = argparse.ArgumentParser(description="Batch API judge (50% cheaper than 3_judge.py)")
    ap.add_argument("--responses_dir", required=True)
    ap.add_argument("--roles_dir", default="assistant-axis/data/roles/instructions")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--judge_model", default="gpt-4.1-mini")
    ap.add_argument("--max_tokens", type=int, default=10)
    ap.add_argument("--chunk_size", type=int, default=1000)
    ap.add_argument("--max_concurrent", type=int, default=3)
    ap.add_argument("--poll_interval", type=int, default=60)
    ap.add_argument("--roles", nargs="+")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--skip_pilot", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--resume_only", action="store_true")
    args = ap.parse_args()

    responses_dir = Path(args.responses_dir).resolve()
    roles_dir = Path(args.roles_dir).resolve()
    scores_dir = Path(args.output_dir).resolve()
    scores_dir.mkdir(parents=True, exist_ok=True)

    # File handler (tail-friendly audit log)
    log_fh = logging.FileHandler(scores_dir / LOG_FILENAME)
    log_fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(log_fh)

    # Dry-run enumerates without locking, no API key needed
    state_file = scores_dir / STATE_FILENAME
    if args.dry_run:
        state = load_state(state_file)
        pf = set(state["permanent_failures"].keys())
        in_flight: Set[str] = set()
        for m in state["batches"].values():
            if m.get("merged_at") is None:
                in_flight.update(m.get("custom_ids", []))
        items = enumerate_unscored(
            responses_dir, roles_dir, scores_dir, pf, in_flight, args.roles
        )
        by_role: Dict[str, int] = {}
        for (role, _, _) in items:
            by_role[role] = by_role.get(role, 0) + 1
        logger.info(f"DRY RUN: {len(items)} unscored prompts across {len(by_role)} roles")
        # show top 5 by remaining-work
        top = sorted(by_role.items(), key=lambda x: -x[1])[:5]
        for role, n in top:
            logger.info(f"  {role}: {n}")
        if len(by_role) > 5:
            logger.info(f"  ... and {len(by_role) - 5} more")
        logger.info(f"permanent_failures so far: {len(pf)}")
        return 0

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set (check .env)")
        return 1

    # Acquire lock (keep lock_fp alive for lifetime)
    lock_fp = acquire_lock(scores_dir / LOCK_FILENAME)  # noqa: F841

    client = openai.OpenAI()
    state = load_state(state_file)
    temp_dir = scores_dir / "_batch_tmp"

    # Step 1: reconcile existing batches
    logger.info(f"Loaded state: {len(state['batches'])} tracked batches, "
                f"{len(state['permanent_failures'])} permanent failures, "
                f"pilot_ok={state['pilot_ok']}")
    backfill_custom_ids(client, state)
    save_state_atomic(state_file, state)
    merged, still_active = reconcile_batches(client, state, state_file, scores_dir)
    logger.info(f"Reconciled: merged {merged} scores, {still_active} batches still active")

    if args.resume_only:
        logger.info("Resume-only mode: not submitting new batches")
        return 0

    # Step 2: pilot (once per scores_dir unless --pilot forces it)
    if args.pilot or (not state["pilot_ok"] and not args.skip_pilot):
        ok = run_pilot(client, responses_dir, roles_dir, scores_dir,
                       args.judge_model, temp_dir, args.poll_interval)
        if not ok:
            logger.error("Pilot failed — aborting")
            return 2
        state["pilot_ok"] = True
        save_state_atomic(state_file, state)
        logger.info("Pilot passed")
        if args.pilot:
            logger.info("Exit after pilot (--pilot was specified)")
            return 0

    def get_active_ids() -> List[str]:
        return [
            bid for bid, m in state["batches"].items()
            if m.get("merged_at") is None and m.get("status") not in
               ("failed", "expired", "cancelled", "abandoned")
        ]

    def compute_in_flight() -> Set[str]:
        out: Set[str] = set()
        for m in state["batches"].values():
            if m.get("merged_at") is None:
                out.update(m.get("custom_ids", []))
        return out

    permanent_failures = set(state["permanent_failures"].keys())
    run_start = time.time()
    total_merged_start = sum(m.get("merged_scores", 0) for m in state["batches"].values())
    next_chunk_idx = max(
        (m.get("chunk_index", -1) for m in state["batches"].values()),
        default=-1
    ) + 1

    # OUTER loop: re-enumerate after each inner loop drains
    # Keeps catching prompts from auto-cancelled/abandoned batches
    enumerate_passes = 0
    while True:
        enumerate_passes += 1
        in_flight = compute_in_flight()
        if in_flight:
            logger.info(f"{len(in_flight)} custom_ids in flight (unmerged batches)")
        items = enumerate_unscored(
            responses_dir, roles_dir, scores_dir, permanent_failures, in_flight, args.roles
        )
        if not items and not get_active_ids():
            logger.info(
                f"All work complete after {enumerate_passes} enumeration pass(es)."
            )
            return 0
        logger.info(f"Pass #{enumerate_passes}: {len(items)} unscored prompts")
        chunks = chunk_list(items, args.chunk_size)
        if chunks:
            logger.info(f"Chunked into {len(chunks)} batches of up to {args.chunk_size}")

        # INNER loop: submit + poll until chunks drained and active empty
        while chunks or get_active_ids():
            active = get_active_ids()
            # Fill up to max_concurrent
            while chunks and len(active) < args.max_concurrent:
                records = build_batch_jsonl(chunks.pop(0), args.judge_model, args.max_tokens)
                try:
                    bid, fid = submit_batch(client, records, temp_dir)
                except openai.BadRequestError as e:
                    if _detect_quota_error(e):
                        logger.error("Submission rejected: insufficient_quota. Top up billing.")
                        return 3
                    raise
                except openai.RateLimitError as e:
                    if _detect_quota_error(e):
                        logger.error("Submission rejected: insufficient_quota. Top up billing.")
                        return 3
                    logger.warning(f"Submission rate-limited; sleeping 60s: {e}")
                    chunks.insert(0, [
                        (r["custom_id"].rsplit(CUSTOM_ID_SEP, 1)[0],
                         r["custom_id"].rsplit(CUSTOM_ID_SEP, 1)[1],
                         r["body"]["messages"][0]["content"])
                        for r in records
                    ])
                    time.sleep(60)
                    break
                state["batches"][bid] = {
                    "openai_batch_id": bid,
                    "input_file_id": fid,
                    "status": "validating",
                    "submitted_at": time.time(),
                    "request_count": len(records),
                    "chunk_index": next_chunk_idx,
                    "custom_ids": [r["custom_id"] for r in records],
                }
                next_chunk_idx += 1
                save_state_atomic(state_file, state)
                logger.info(
                    f"Submitted batch {bid[:16]}... ({len(records)} req, chunk={next_chunk_idx-1}); "
                    f"active={len(get_active_ids())}/{args.max_concurrent}"
                )
                active = get_active_ids()

            # Poll
            time.sleep(args.poll_interval)
            merged, still_active = reconcile_batches(client, state, state_file, scores_dir)
            active_now = get_active_ids()
            merged_total = sum(m.get("merged_scores", 0) for m in state["batches"].values())
            pf_total = len(state["permanent_failures"])

            # ETA based on current run's throughput
            elapsed = time.time() - run_start
            scored_this_run = merged_total - total_merged_start
            rate = scored_this_run / max(elapsed, 1)  # scores/sec
            # Current unscored estimate: chunks + active still pending
            remaining_est = sum(
                m.get("request_count", 0) for bid, m in state["batches"].items()
                if m.get("merged_at") is None
            ) + sum(len(c) for c in chunks)
            eta_str = ""
            if rate > 0 and remaining_est > 0:
                eta_min = (remaining_est / rate) / 60
                eta_str = f" eta={eta_min:.0f}min"
            logger.info(
                f"[tick] active={len(active_now)} pending_chunks={len(chunks)} "
                f"merged_total={merged_total}{eta_str} permanent_failures={pf_total}"
            )


if __name__ == "__main__":
    sys.exit(main())
