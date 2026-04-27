#!/usr/bin/env python3
"""Pre-flight readiness audit for the abliterated Lambda run + local resume.

Exit 0 on all PASS/WARN, 1 on any FAIL.
Usage:
    set -a; source .env; set +a
    .venv/bin/python scripts/check_abliterated_ready.py
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HF_REPO = "pandaman007/assistant-axis-abliteration-vectors"

results = []  # (status, id, msg, fix)


def record(status, id_, msg, fix=""):
    results.append((status, id_, msg, fix))
    print(f"[{status}] {id_}. {msg}")
    if status == "FAIL" and fix:
        for line in fix.splitlines():
            print(f"         {line}")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


print("=" * 60)
print(f"ABLITERATED RUN READINESS — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 60)

# Lazy import HF / OpenAI so missing-deps give clear errors
try:
    from huggingface_hub import HfApi, hf_hub_download, list_repo_files
except ImportError:
    print("FATAL: huggingface_hub not installed in this venv")
    sys.exit(2)

try:
    import torch
except ImportError:
    print("FATAL: torch not installed")
    sys.exit(2)

api = HfApi()
hf_files = list_repo_files(HF_REPO, repo_type="dataset")


# ═══ Group A — HF prerequisites ════════════════════════════════════════
print("\n--- Group A: HF prerequisites ---")

# A1. refusal_direction.pt on HF
rd_hf_path = "llama-3.1-8b-instruct/refusal_direction.pt"
if rd_hf_path not in hf_files:
    record("FAIL", "A1", "refusal_direction.pt MISSING on HF",
           f"upload {ROOT}/results/comparison/refusal_direction.pt to {rd_hf_path}")
    rd_hf_local = None
else:
    try:
        rd_hf_local = hf_hub_download(HF_REPO, filename=rd_hf_path, repo_type="dataset")
        t = torch.load(rd_hf_local, map_location="cpu", weights_only=False)
        if t.ndim != 2 or t.shape != (32, 4096):
            record("FAIL", "A1", f"refusal_direction.pt wrong shape {tuple(t.shape)} (expected (32, 4096))",
                   "regenerate from 00_cosine_precheck.py and re-upload")
        elif torch.isnan(t).any() or torch.isinf(t).any():
            record("FAIL", "A1", "refusal_direction.pt contains NaN/Inf",
                   "regenerate from 00_cosine_precheck.py")
        else:
            record("PASS", "A1", f"refusal_direction.pt on HF (shape={tuple(t.shape)}, no NaN/Inf)")
    except Exception as e:
        record("FAIL", "A1", f"cannot load HF refusal_direction.pt: {e}",
               "check HF token + re-upload if corrupted")
        rd_hf_local = None

# A2. bootstrap script SHA matches local
local_bootstrap = ROOT / "scripts/cloud_setup_abliterated.sh"
hf_bootstrap_path = "scripts/cloud_setup_abliterated.sh"
if not local_bootstrap.exists():
    record("FAIL", "A2", f"local bootstrap missing at {local_bootstrap}", "restore from git")
elif hf_bootstrap_path not in hf_files:
    record("FAIL", "A2", "bootstrap script MISSING on HF",
           f"upload {local_bootstrap} to {hf_bootstrap_path}")
else:
    try:
        hf_bs_path = hf_hub_download(HF_REPO, filename=hf_bootstrap_path,
                                      repo_type="dataset")
        local_sha = sha256_file(local_bootstrap)
        hf_sha = sha256_file(hf_bs_path)
        if local_sha == hf_sha:
            record("PASS", "A2", f"bootstrap SHA matches local ({local_sha[:12]}…)")
        else:
            record("FAIL", "A2",
                   f"bootstrap DRIFT: local {local_sha[:12]}… vs HF {hf_sha[:12]}…",
                   f"re-upload with: huggingface-cli upload {HF_REPO} {local_bootstrap} "
                   f"{hf_bootstrap_path} --repo-type=dataset")
    except Exception as e:
        record("FAIL", "A2", f"cannot fetch HF bootstrap: {e}", "check HF token")

# A3. abliterated output prefix state
for name, prefix, expected in [
    ("responses", "llama-3.1-8b-abliterated/responses/", 276),
    ("activations", "llama-3.1-8b-abliterated/activations_layer16_full/", 276),
]:
    count = sum(1 for f in hf_files if f.startswith(prefix))
    if count == 0:
        record("PASS", f"A3.{name}", f"{prefix} empty (clean start)")
    elif count == expected:
        record("PASS", f"A3.{name}", f"{prefix} complete ({count}/{expected})")
    else:
        record("FAIL", f"A3.{name}",
               f"{prefix} PARTIAL ({count}/{expected}) — prior abandoned run?",
               f"either wipe via HF UI, or trust resume logic in run_pipeline.sh")

# A4. local refusal matches HF
local_rd = ROOT / "results/comparison/refusal_direction.pt"
if not local_rd.exists():
    record("FAIL", "A4", f"local refusal_direction.pt missing at {local_rd}",
           "run scripts/00_cosine_precheck.py, or download from HF")
elif rd_hf_local is None:
    record("WARN", "A4", "HF copy unavailable; skipping compare")
else:
    if sha256_file(local_rd) == sha256_file(rd_hf_local):
        record("PASS", "A4", "local refusal_direction.pt matches HF")
    else:
        record("FAIL", "A4", "local refusal_direction.pt DIFFERS from HF",
               "upload local to HF (local is canonical — cosine-precheck output)")


# ═══ Group B — Script-level consistency ═══════════════════════════════
print("\n--- Group B: script consistency ---")

# B1. inline abliterate.py ≈ 02_abliterate_model.py (semantic spot-check)
local_abl = ROOT / "scripts/02_abliterate_model.py"
bootstrap_text = local_bootstrap.read_text() if local_bootstrap.exists() else ""

heredoc_match = re.search(
    r"cat > scripts/abliterate\.py <<'ABLIT'\n(.*?)\nABLIT\n",
    bootstrap_text, re.DOTALL)

if not heredoc_match:
    record("FAIL", "B1", "cannot find abliterate.py heredoc in cloud_setup_abliterated.sh",
           "inspect the bash script; heredoc should start with `cat > scripts/abliterate.py <<'ABLIT'`")
elif not local_abl.exists():
    record("WARN", "B1", f"local {local_abl.name} missing; skipping compare")
else:
    inline = heredoc_match.group(1)
    canonical = local_abl.read_text()

    # Core math ops that MUST be identical in both
    core_ops = [
        ("orthogonalize core", r"W\.sub_\(torch\.outer\(r,\s*r\s*@\s*W\)\)"),
        ("embed_tokens write", r"emb\.sub_\(torch\.outer\(emb\s*@\s*r,\s*r\)\)"),
        ("o_proj orthogonalize", r"orthogonalize\(layer\.self_attn\.o_proj\.weight,\s*r\)"),
        ("down_proj orthogonalize", r"orthogonalize\(layer\.mlp\.down_proj\.weight,\s*r\)"),
        ("default layer=16", r"--layer[^\n]*default=16"),
        ("unit-norm refusal", r"r\s*=\s*r\s*/\s*r\.norm\(\)"),
    ]
    missing_inline = [name for name, pat in core_ops if not re.search(pat, inline)]
    missing_canon = [name for name, pat in core_ops if not re.search(pat, canonical)]

    # Safety ops expected in inline (Lambda runs this) — INFO if missing from canonical
    inline_safety = [
        ("NaN/Inf check", r"torch\.isnan\(W\)\.any\(\)"),
    ]
    missing_inline_safety = [name for name, pat in inline_safety if not re.search(pat, inline)]

    if missing_canon:
        record("FAIL", "B1", f"canonical 02_abliterate_model.py missing core ops: {missing_canon}",
               "restore from git")
    elif missing_inline:
        record("FAIL", "B1", f"inline abliterate.py (Lambda-runtime) missing core ops: {missing_inline}",
               "regenerate the heredoc in cloud_setup_abliterated.sh from 02_abliterate_model.py")
    elif missing_inline_safety:
        record("FAIL", "B1", f"inline abliterate.py missing safety checks: {missing_inline_safety}",
               "restore NaN/Inf validation in the heredoc")
    else:
        record("PASS", "B1", "inline abliterate.py has all core math + safety (matches 02_abliterate_model.py)")

# B2. 04_resume_abliterated_local.sh uses validated batch knobs
resume_sh = ROOT / "scripts/04_resume_abliterated_local.sh"
if not resume_sh.exists():
    record("FAIL", "B2", f"{resume_sh} missing", "restore from git")
else:
    rtext = resume_sh.read_text()
    required_knobs = [
        ("3_judge_batch.py", r"3_judge_batch\.py"),
        ("chunk_size=1000", r"--chunk_size\s+1000"),
        ("max_concurrent=3", r"--max_concurrent\s+3"),
        ("poll_interval=60", r"--poll_interval\s+60"),
    ]
    missing = [n for n, p in required_knobs if not re.search(p, rtext)]
    if missing:
        record("FAIL", "B2", f"04_resume_abliterated_local.sh missing knobs: {missing}",
               "these are the Tier 1 validated values — update the script")
    else:
        record("PASS", "B2", "04_resume_abliterated_local.sh uses validated batch knobs")

# B3. HF paths consistent
expected_paths_write = [
    "llama-3.1-8b-abliterated/responses",
    "llama-3.1-8b-abliterated/activations_layer16_full",
    "llama-3.1-8b-abliterated/model_meta",
]
expected_paths_resume = [
    "llama-3.1-8b-abliterated",  # used as prefix
]
missing_in_bootstrap = [p for p in expected_paths_write if p not in bootstrap_text]
resume_text = resume_sh.read_text() if resume_sh.exists() else ""
missing_in_resume = [p for p in expected_paths_resume if p not in resume_text]

if missing_in_bootstrap:
    record("FAIL", "B3.bootstrap",
           f"cloud_setup_abliterated.sh missing paths: {missing_in_bootstrap}",
           "grep/update the script")
else:
    record("PASS", "B3.bootstrap", "bootstrap writes to expected HF paths")

if missing_in_resume:
    record("FAIL", "B3.resume",
           f"04_resume_abliterated_local.sh missing paths: {missing_in_resume}",
           "grep/update the script")
else:
    record("PASS", "B3.resume", "resume uses expected HF paths")


# ═══ Group C — Resources ══════════════════════════════════════════════
print("\n--- Group C: resources ---")

# C1. disk space
st = shutil.disk_usage("/scratch/paulc")
free_gb = st.free / (1024 ** 3)
if free_gb < 15:
    record("FAIL", "C1", f"/scratch/paulc free = {free_gb:.1f} GB (need ≥ 15)",
           "clean up /scratch/paulc before starting the abliterated run")
elif free_gb < 30:
    record("WARN", "C1", f"/scratch/paulc free = {free_gb:.1f} GB (tight; ≥ 30 GB preferred)")
else:
    record("PASS", "C1", f"/scratch/paulc free = {free_gb:.0f} GB")

# C2. OpenAI credit (best-effort — no clean API endpoint)
try:
    from openai import OpenAI
    openai_client = OpenAI()
    try:
        # Try to list batches as a connectivity + auth sanity — not a balance
        _ = list(openai_client.batches.list(limit=1))
        record("WARN", "C2",
               "OpenAI connectivity OK; credit balance not queryable programmatically. "
               "Verify ≥ $70 at https://platform.openai.com/settings/organization/billing")
    except Exception as e:
        record("FAIL", "C2", f"OpenAI API error: {e}",
               "check OPENAI_API_KEY in .env; refresh key if revoked")
except ImportError:
    record("FAIL", "C2", "openai SDK not installed", "pip install openai")

# C3. batch queue capacity
try:
    batches = list(openai_client.batches.list(limit=100))
    active = [b for b in batches if b.status in ("validating", "in_progress", "finalizing")]
    if len(active) > 15:
        record("WARN", "C3",
               f"{len(active)} active batches — concurrent original+abliterated may hit Tier 1 queue limit",
               "")
    else:
        record("PASS", "C3",
               f"OpenAI batch queue: {len(active)} active (well under Tier 1 limit)")
except Exception as e:
    record("WARN", "C3", f"could not query batch queue: {e}")


# ═══ Group D — Concurrency / operational ═══════════════════════════════
print("\n--- Group D: concurrency / ops ---")

# D1. An hf_sync sidecar covers the abliterated run
abl_sync = ROOT / "scripts/hf_sync_abliterated.sh"
orig_sync = ROOT / "scripts/hf_sync.sh"

if abl_sync.exists():
    atext = abl_sync.read_text()
    needed = [
        ("abliterated scores path", "results/abliterated/scores"),
        ("abliterated HF prefix", "llama-3.1-8b-abliterated"),
    ]
    missing = [n for n, p in needed if p not in atext]
    if missing:
        record("FAIL", "D1", f"hf_sync_abliterated.sh missing required paths: {missing}",
               "update the script to point at abliterated paths")
    else:
        record("PASS", "D1", "hf_sync_abliterated.sh covers abliterated run")
else:
    # Fall back: is the original one parameterized enough to be reused?
    stext = orig_sync.read_text() if orig_sync.exists() else ""
    hard_orig = "results/original/scores" in stext
    hard_prefix = "llama-3.1-8b-instruct" in stext
    if hard_orig or hard_prefix:
        record("FAIL", "D1",
               "no hf_sync_abliterated.sh AND hf_sync.sh hardcoded to original paths",
               "create scripts/hf_sync_abliterated.sh pointing at results/abliterated/scores "
               "+ llama-3.1-8b-abliterated")
    else:
        record("PASS", "D1", "hf_sync.sh is parameterized (no hardcoded paths)")

# D2. Original run status (informational)
orig_state = ROOT / "results/original/scores/_batch_state.json"
if not orig_state.exists():
    record("INFO", "D2", "no original run state on disk — safe to start abliterated")
else:
    state = json.loads(orig_state.read_text())
    scored = 0
    for f in (ROOT / "results/original/scores").glob("*.json"):
        if f.name.startswith("_"):
            continue
        try:
            scored += len(json.loads(f.read_text()))
        except Exception:
            pass
    pct = scored / 330000 * 100
    alive = subprocess.run(
        ["pgrep", "-f", "3_judge_batch.py"], capture_output=True, text=True
    ).stdout.strip()
    if alive and pct < 100:
        record("INFO", "D2",
               f"Original judge RUNNING at {pct:.1f}% ({scored:,}/330k) → "
               f"recommend sequential launch (wait for it to finish)")
    elif pct >= 99.9:
        record("INFO", "D2", f"Original judge complete ({scored:,} scores) → ready for abliterated")
    else:
        record("INFO", "D2",
               f"Original state exists ({pct:.1f}%) but judge not running — partial run")


# ═══ Summary ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
counts = {"PASS": 0, "WARN": 0, "FAIL": 0, "INFO": 0}
for s, _, _, _ in results:
    counts[s] = counts.get(s, 0) + 1

print(f"SUMMARY: {counts['PASS']} PASS, {counts['WARN']} WARN, "
      f"{counts['FAIL']} FAIL, {counts['INFO']} INFO")
if counts["FAIL"] > 0:
    print(f"Status: NOT READY. Fix the {counts['FAIL']} FAIL(s), then re-run.")
    print("=" * 60)
    sys.exit(1)
elif counts["WARN"] > 0:
    print(f"Status: READY with {counts['WARN']} WARN(s). Review before launching Lambda.")
    print("=" * 60)
    sys.exit(0)
else:
    print("Status: READY.")
    print("=" * 60)
    sys.exit(0)
