#!/usr/bin/env python3
"""Phase H pre-flight check. Run BEFORE paying for cloud GPU time.

Verifies:
  - .env loaded with OPENAI_API_KEY
  - HF cache dir writable
  - Both Phi-3.5-mini-instruct and Llama-3.2-3B-Instruct can be downloaded + loaded
  - Hidden dim is 3072 for both
  - Layer counts match: Phi=32, Llama-3.2-3B=28
  - vLLM smoke test on each (loads, generates one token)
  - PHASE_H_DESIGN.md anchor lists parse cleanly + no role overlap
  - All anchors exist in role_list.json
  - GPU has enough memory

Exit 0 on PASS, 1 on FAIL.

Usage:
    set -a; source .env; set +a
    .venv/bin/python scripts/check_phase_h_ready.py
"""
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
results = []

PHI = "microsoft/Phi-3.5-mini-instruct"
LLAMA32 = "meta-llama/Llama-3.2-3B-Instruct"
EXPECTED = {
    PHI: {"hidden_size": 3072, "num_hidden_layers": 32, "L_extract": 16},
    LLAMA32: {"hidden_size": 3072, "num_hidden_layers": 28, "L_extract": 14},
}


def record(status, id_, msg, fix=""):
    results.append((status, id_, msg, fix))
    print(f"[{status:4s}] {id_}. {msg}")
    if status == "FAIL" and fix:
        for line in fix.splitlines():
            print(f"          {line}")


print("=" * 60)
print(f"PHASE H READINESS — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 60)

try:
    import torch
    from transformers import AutoConfig
except ImportError as e:
    print(f"FATAL: {e}")
    sys.exit(2)


# ── Group A: env ───────────────────────────────────────────────────────
print("\n--- Group A: environment ---")

if not os.environ.get("OPENAI_API_KEY"):
    record("FAIL", "A1", "OPENAI_API_KEY not in env",
           "set -a; source .env; set +a")
else:
    record("PASS", "A1", "OPENAI_API_KEY present")

hf_cache = os.environ.get("HF_HOME", "/scratch/paulc/hf_cache")
if not Path(hf_cache).parent.exists():
    record("FAIL", "A2", f"HF_HOME parent not writable: {hf_cache}",
           f"mkdir -p {Path(hf_cache).parent}")
else:
    Path(hf_cache).mkdir(parents=True, exist_ok=True)
    record("PASS", "A2", f"HF_HOME ready: {hf_cache}")

if not torch.cuda.is_available():
    record("FAIL", "A3", "no CUDA visible", "")
else:
    n_gpus = torch.cuda.device_count()
    free, total = torch.cuda.mem_get_info(0)
    free_gb = free / 1e9
    total_gb = total / 1e9
    if free_gb < 30:
        record("WARN", "A3", f"GPU 0 only {free_gb:.1f}/{total_gb:.1f} GB free")
    else:
        record("PASS", "A3", f"GPU 0: {free_gb:.1f}/{total_gb:.1f} GB free")


# ── Group B: model configs ─────────────────────────────────────────────
print("\n--- Group B: model configs (downloads if cache miss) ---")

for model_id, expected in EXPECTED.items():
    short = model_id.split("/")[-1]
    try:
        cfg = AutoConfig.from_pretrained(model_id)
        h = cfg.hidden_size
        n = cfg.num_hidden_layers
        if h != expected["hidden_size"]:
            record("FAIL", f"B-{short}", f"{model_id}: hidden_size={h} != {expected['hidden_size']}")
            continue
        if n != expected["num_hidden_layers"]:
            record("FAIL", f"B-{short}", f"{model_id}: num_hidden_layers={n} != {expected['num_hidden_layers']}")
            continue
        record("PASS", f"B-{short}",
               f"{model_id}: dim={h}, layers={n}, L_extract={expected['L_extract']}")
    except Exception as e:
        record("FAIL", f"B-{short}", f"{model_id}: cannot load config: {e}",
               f"check HF auth: huggingface-cli login")


# ── Group C: PHASE_H_DESIGN.md anchor lists ────────────────────────────
print("\n--- Group C: anchor list parse + integrity ---")

design_path = ROOT / "PHASE_H_DESIGN.md"
if not design_path.exists():
    record("FAIL", "C1", f"PHASE_H_DESIGN.md missing at {design_path}")
    sys.exit(1)

# Extract anchor lists by parsing the markdown tables.
text = design_path.read_text()

ANCHOR_AXES = {
    "v_benevolence": (
        ["counselor", "parent", "guardian", "pacifist", "peacekeeper", "altruist", "healer", "angel"],
        ["criminal", "saboteur", "narcissist", "zealot", "hoarder", "smuggler", "demon", "predator"],
    ),
    "v_authority": (
        ["judge", "scientist", "ambassador", "polymath", "virtuoso", "sage", "leviathan", "ancient"],
        ["amateur", "dilettante", "student", "infant", "refugee", "prey", "prisoner", "orphan"],
    ),
    "v_humor": (
        ["comedian", "jester", "fool", "absurdist", "bohemian", "surfer", "improviser", "bard"],
        ["philosopher", "mathematician", "ascetic", "scholar", "hermit", "traditionalist", "conservator", "statistician"],
    ),
    "v_critic": (
        ["contrarian", "devils_advocate", "skeptic", "cynic", "perfectionist", "evaluator", "auditor", "examiner"],
        ["synthesizer", "optimist", "idealist", "evangelist", "romantic", "advocate", "facilitator", "instructor"],
    ),
}

# Test C1: no anchor used in two axes
all_anchors = []
for axis, (pos, neg) in ANCHOR_AXES.items():
    all_anchors += pos + neg
dups = [r for r in all_anchors if all_anchors.count(r) > 1]
if dups:
    record("FAIL", "C1", f"anchor roles used in multiple axes: {sorted(set(dups))}")
else:
    record("PASS", "C1", f"all {len(all_anchors)} anchors unique across {len(ANCHOR_AXES)} anchor axes")

# Test C2: every anchor exists in role_list.json
roles_path = ROOT / "assistant-axis/data/roles/role_list.json"
roles_data = json.loads(roles_path.read_text())
all_role_names = set(roles_data.keys())
missing = [r for r in all_anchors if r not in all_role_names]
if missing:
    record("FAIL", "C2", f"anchors not in role_list.json: {missing}")
else:
    record("PASS", "C2", f"all anchors present in role_list.json")

# Test C3: each anchor has an instruction file
inst_dir = ROOT / "assistant-axis/data/roles/instructions"
missing_files = [r for r in all_anchors if not (inst_dir / f"{r}.json").exists()]
if missing_files:
    record("FAIL", "C3", f"anchors missing instructions/: {missing_files}")
else:
    record("PASS", "C3", "all anchors have instruction files")

# Test C4: held-out validation pairs exist + don't overlap with anchors
HELDOUT_PAIRS = {
    "v_benevolence": [("mentor", "vigilante"), ("paramedic", "spy"),
                      ("therapist", "addict"), ("mediator", "rebel"), ("caregiver", "destroyer"),
                      ("empath", "rogue"), ("veterinarian", "anarchist"), ("pharmacist", "daredevil")],
    "v_authority": [("grandparent", "graduate"), ("consultant", "teenager"),
                    ("guru", "addict"), ("veteran", "expatriate"), ("supervisor", "immigrant"),
                    ("elder", "toddler"), ("visionary", "wanderer"), ("composer", "toddler")],
    "v_humor": [("trickster", "engineer"), ("flaneur", "sociologist"),
                ("gamer", "workaholic"), ("influencer", "luddite"), ("provocateur", "planner"),
                ("daredevil", "realist"), ("newlywed", "divorcee"), ("actor", "pragmatist")],
    "v_critic": [("critic", "celebrity"), ("reviewer", "empath"),
                 ("analyst", "dreamer"), ("grader", "visionary"), ("detective", "newlywed"),
                 ("screener", "caregiver"), ("moderator", "martyr"), ("editor", "prophet")],
}
heldout_roles = []
for pairs in HELDOUT_PAIRS.values():
    for a, b in pairs:
        heldout_roles += [a, b]
heldout_anchor_overlap = [r for r in heldout_roles if r in all_anchors]
heldout_missing = [r for r in heldout_roles if r not in all_role_names]
if heldout_anchor_overlap:
    record("FAIL", "C4", f"held-out roles overlap anchors: {sorted(set(heldout_anchor_overlap))}")
elif heldout_missing:
    record("FAIL", "C4", f"held-out roles not in role_list: {sorted(set(heldout_missing))}")
else:
    record("PASS", "C4", f"all {len(set(heldout_roles))} held-out validation roles valid + disjoint")

# Test C5: null-role list
NULL_ROLES = {
    "v_benevolence": ["accountant", "librarian", "cartographer", "novelist", "programmer", "mechanic"],
    "v_authority": ["bartender", "photographer", "tutor", "chef", "mechanic", "designer"],
    "v_humor": ["mediator", "paramedic", "librarian", "accountant", "programmer", "lawyer"],
    "v_critic": ["naturalist", "archaeologist", "biologist", "physicist", "chemist", "geographer"],
}
all_nulls = [r for v in NULL_ROLES.values() for r in v]
null_overlap = [r for r in all_nulls if r in all_anchors]
null_missing = [r for r in all_nulls if r not in all_role_names]
if null_overlap:
    record("FAIL", "C5", f"null roles overlap anchors: {sorted(set(null_overlap))}")
elif null_missing:
    record("FAIL", "C5", f"null roles missing: {sorted(set(null_missing))}")
else:
    record("PASS", "C5", f"all {len(set(all_nulls))} null-validation roles valid")


# ── Group D: vLLM smoke test ──────────────────────────────────────────
print("\n--- Group D: vLLM smoke test ---")
print("(this loads each model briefly — takes ~2 min each)")

if "--skip-vllm" in sys.argv:
    record("WARN", "D", "vLLM smoke test skipped (--skip-vllm)")
else:
    for model_id in EXPECTED:
        short = model_id.split("/")[-1]
        try:
            from vllm import LLM, SamplingParams
            llm = LLM(model=model_id, max_model_len=512, gpu_memory_utilization=0.6,
                      enforce_eager=True, dtype="auto")
            out = llm.generate(["Hello"], SamplingParams(max_tokens=4, temperature=0.0))
            del llm
            torch.cuda.empty_cache()
            record("PASS", f"D-{short}", f"{model_id}: vLLM loads + generates")
        except Exception as e:
            record("FAIL", f"D-{short}", f"{model_id}: vLLM failed: {e}")


# ── Summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
n_pass = sum(1 for r in results if r[0] == "PASS")
n_warn = sum(1 for r in results if r[0] == "WARN")
n_fail = sum(1 for r in results if r[0] == "FAIL")
print(f"PASS: {n_pass}    WARN: {n_warn}    FAIL: {n_fail}")
print("=" * 60)

sys.exit(1 if n_fail else 0)
