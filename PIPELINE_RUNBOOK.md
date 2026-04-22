# Pipeline Runbook

End-to-end operational guide for the Assistant Axis × Abliteration experiment. Covers both the original and abliterated pipelines, with all the mistakes from Round 1 baked into the workflow so we don't repeat them.

## Architecture: what runs where and why

**Lambda Cloud (GPU work, paid):**
- Step 1 (generate responses via vLLM) — only for abliterated (original responses were pre-computed by paper)
- Step 2 (extract activations via HuggingFace hooks)

**lisplab1 (local, API-only):**
- Step 3 (OpenAI judge)
- Step 4 (per-role vectors)
- Step 5 (axis)

**Why this split:** Step 3 takes ~13-24 hrs at OpenAI Tier 1 rate limits (200k TPM / 500 RPM). Running on Lambda wastes ~$200-$400 on idle GPUs. No GPU is needed for judging; only API throughput matters.

## Repository layout

```
/scratch/paulc/assistant-axis-abliteration/
├── CLAUDE.md                           # Project context (do not edit without reason)
├── PIPELINE_RUNBOOK.md                 # THIS FILE
├── scripts/
│   ├── 01_run_pipeline.sh              # Full 5-step original run (if doing on lisplab1)
│   ├── 02_abliterate_model.py          # Abliteration (also embedded in Lambda setup)
│   ├── 03_resume_local.sh              # Local Step 3-5 for ORIGINAL
│   ├── 04_resume_abliterated_local.sh  # Local Step 3-5 for ABLITERATED
│   ├── cloud_setup.sh                  # Lambda bootstrap — original pipeline
│   └── cloud_setup_abliterated.sh      # Lambda bootstrap — abliterated pipeline
├── assistant-axis/                     # Cloned from safety-research/assistant-axis
│   └── assistant_axis/judge.py         # ⚠ LOCAL FIX: 429 retry with exp backoff
├── results/
│   ├── original/                       # Response/score/vector/axis outputs
│   └── abliterated/                    # (created during abliterated run)
└── .venv/                              # Python environment
```

**HF dataset `pandaman007/assistant-axis-abliteration-vectors`:**

```
llama-3.1-8b-instruct/                  # Original model outputs
├── responses/                          # 276 JSONL files
├── activations_layer16_full/           # 276 .pt files, shape (1, 4096) bf16
├── scores/                             # (populated after local judge completes)
├── vectors/                            # (populated after local Step 4)
├── axis.pt                             # (populated after local Step 5)
└── refusal_direction.pt                # Step 0 output — input to abliteration
llama-3.1-8b-abliterated/               # Abliterated model outputs
├── responses/                          # Written by Lambda Step 1
├── activations_layer16_full/           # Written by Lambda Step 2
├── model_meta/                         # config.json, tokenizer.*, etc.
├── scores/                             # Written by local judge
├── vectors/                            # Written by local Step 4
└── axis.pt                             # Written by local Step 5
scripts/                                # Setup scripts for curl-bootstrap
```

## Bugs we learned — do not repeat

Every one of these burned time/money at least once. All are mitigated in the current scripts.

| # | Bug | Root cause | Mitigation in current scripts |
|---|---|---|---|
| 1 | `apt-get` interactive daemon-restart dialog | Ubuntu 22.04 needrestart defaults | `export DEBIAN_FRONTEND=noninteractive; export NEEDRESTART_MODE=a` |
| 2 | `ModuleNotFoundError: sklearn`, `plotly`, etc. | Manual pip install missed transitive deps | `pip install -e assistant-axis/` resolves from pyproject.toml |
| 3 | `LlamaForCausalLM.__init__() got unexpected kwarg 'dtype'` | transformers 4.46 too old; 5.x broke silently | Pinned `transformers==4.57.6` |
| 4 | Silent multi-worker failure | `2_activations.py` main returns 0 when child workers crash | File-count-based retry loop in shell (checks ≥276 files, not Python exit code) |
| 5 | Silent partial JSONL skip | `should_skip_role()` only checks file existence | Pre-flight deletes JSONLs with line count ≠ 1200 before each Step 1 attempt |
| 6 | CUDA OOM at batch_size=64 | Worker memory fragmentation on restart | `batch_size=16` for Step 2 from the start |
| 7 | Corrupt `.pt` from SIGKILL | Partial writes during crash | Pre-flight in Step 2 loads each .pt, deletes ones that fail validation |
| 8 | OpenAI 429 rate limits dropping scores | Judge had NO retry logic, just returned None on exception | `assistant_axis/judge.py` patched: 30 retries with exp backoff |
| 9 | Step 3 wastes Lambda GPU $$ | Rate-limited work doesn't need GPUs | Step 3 runs locally, NOT on Lambda |
| 10 | vLLM incompatibility discovered mid-Step-1 | No pre-launch smoke test | Smoke test in cloud_setup_abliterated.sh loads model + generates on 2 prompts before committing to 6hr run |
| 11 | Nothing uploaded until pipeline end | Single terminal upload = lose everything on crash | Per-step HF upload: responses → then Step 2 → then activations |
| 12 | OpenAI key env-var precedence | Stale env var beats .env file | `unset OPENAI_API_KEY && export OPENAI_API_KEY=$(grep ... .env)` |
| 13 | Paste corruption on long commands | Ghostty line-wrapping splits commands | Single-line curl bootstrap; short pasted commands |
| 14 | `xterm-ghostty: unknown terminal type` (tmux fails) | Ubuntu doesn't know Ghostty's TERM id | `export TERM=xterm-256color` before `tmux new` |
| 15 | tmux nested session warning | $TMUX set from parent tmux | `TMUX= tmux attach -d -t <name>` to bypass |
| 16 | Three concurrent pipelines running | User re-pasted start command during debug | Setup scripts launch the pipeline exactly once via `nohup ... & disown` |
| 17 | Abliteration producing NaN/Inf silently | No post-abliteration validation | Explicit NaN/Inf check on 4 sample weight matrices + reload sanity + harmful-prompt verify |
| 18 | No disk-space visibility | `No space left on device` appears as random errors | `df -h` checkpoints before/after Step 1 |

## Procedure A: Abliterated Lambda run (~5-7 hrs, ~$80-$120)

### A.1 Rent instance

- Lambda dashboard → Instances → Launch instance
- Type: **8× A100 40GB SXM4**, Ubuntu 22.04 Lambda Stack
- Region: any; pick based on availability
- If unavailable, fallback to 8× A100 80GB PCIe or 8× H100

### A.2 SSH in + bootstrap

**Two commands only.** The `curl | bash` does everything: apt install, venv, clone, download refusal direction, abliterate, smoke test, launch Step 1 in background.

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxx
curl -fsSL "https://huggingface.co/datasets/pandaman007/assistant-axis-abliteration-vectors/resolve/main/scripts/cloud_setup_abliterated.sh" | bash
```

After ~10-15 min of setup, it prints **"ABLITERATED PIPELINE LAUNCHED (PID: ...)"** and returns to prompt. The pipeline runs in the background under `nohup`.

### A.3 Monitor

```bash
tail -f /home/ubuntu/assistant-axis-abliteration/results/abliterated/run.log
```

Phases to watch for in the log:
- `=== Step 1: Generate ===` — starts immediately, lasts 4-6 hrs
- `=== Upload responses to HF ===` — once Step 1 done
- `=== Step 2: Extract activations ===` — ~30 min
- `=== Upload activations to HF ===`
- `UPLOAD COMPLETE` — final signal, safe to terminate

### A.4 Verify before terminating

```bash
python -c "
from huggingface_hub import list_repo_files
files = list_repo_files('pandaman007/assistant-axis-abliteration-vectors', repo_type='dataset')
for prefix, expect in [
    ('llama-3.1-8b-abliterated/responses/', 276),
    ('llama-3.1-8b-abliterated/activations_layer16_full/', 276),
]:
    n = sum(1 for f in files if f.startswith(prefix))
    print(f'{\"OK\" if n == expect else \"MISMATCH\"} {prefix}: {n} (expected {expect})')
"
```

Both should say **OK 276**.

### A.5 Terminate

Lambda dashboard → select instance → **Terminate** (not Stop — Stop still bills). Type `erase data on instance` to confirm.

## Procedure B: Local judge + vectors + axis (~13-24 hrs)

Same pattern for both original and abliterated. Only difference is which script you run.

### B.1 SSH to lisplab1 and enter tmux

```bash
ssh lisplab1.thayer.dartmouth.edu
export TERM=xterm-256color                    # Ghostty compat
tmux new -s judge-abliterated                  # or judge-original
```

### B.2 Inside tmux, launch

For **original**:
```bash
cd /scratch/paulc/assistant-axis-abliteration
bash scripts/03_resume_local.sh 2>&1 | tee results/original/resume.log
```

For **abliterated**:
```bash
cd /scratch/paulc/assistant-axis-abliteration
bash scripts/04_resume_abliterated_local.sh 2>&1 | tee results/abliterated/resume.log
```

Either script:
1. Pulls responses, activations, (and partial scores if any) from HF
2. Runs Step 3 (judge) — ~13-24 hrs
3. Runs Step 4 (vectors) — minutes
4. Runs Step 5 (axis) — seconds
5. Uploads `scores/`, `vectors/`, `axis.pt` to HF at `llama-3.1-8b-{instruct,abliterated}/`

### B.3 Detach: Ctrl+B, then D

The script keeps running. Safe to close the SSH session.

### B.4 Reattach later

```bash
ssh lisplab1
TMUX= tmux attach -d -t judge-abliterated
```

The `TMUX=` unsets any inherited tmux env; `-d` detaches stale clients.

### B.5 Check progress without attaching

```bash
# count fully-scored roles (out of 276)
python -c "
import json, glob
files = glob.glob('/scratch/paulc/assistant-axis-abliteration/results/abliterated/scores/*.json')
full = sum(1 for f in files if len(json.load(open(f))) == 1200)
print(f'{full}/276 complete')
"
```

### B.6 Wait for "UPLOAD COMPLETE"

Then verify on HF at `llama-3.1-8b-abliterated/axis.pt` and `llama-3.1-8b-abliterated/vectors/`.

## Verification checklists

### After Lambda run

- [ ] 276 JSONL response files; each has 1200 lines
- [ ] 276 activation `.pt` files; each loads as `dict` with 1200 entries, values shape `(1, 4096)` bf16
- [ ] Abliterated model config/tokenizer files on HF under `model_meta/`
- [ ] `UPLOAD COMPLETE` in log
- [ ] No `CUDA`/`OOM` errors outside of pre-flight cleanup messages
- [ ] `df -h` at end showed reasonable free space

### After local Step 3 (judge)

- [ ] 276 score JSON files; each has 1200 entries
- [ ] Score distribution per role: mostly 3s (~85-95%), some 0/1/2 scattered
- [ ] All values in {0, 1, 2, 3}; no null/None
- [ ] No errors in log besides 429 noise (that's expected)

### After local Steps 4-5

- [ ] 270+ vector `.pt` files (some roles will fail `min_count=50` filter — expected)
- [ ] `axis.pt` exists, shape `(1, 4096)`, norm non-zero, no NaN
- [ ] Uploaded to HF

## Cost summary (actual or projected)

| Item | Cost |
|---|---|
| Round 1 original Lambda (learning curve) | ~$50 |
| Abliterated Lambda Steps 1-2 | ~$80-$120 |
| Original judge API (GPT-4.1-mini, 331k calls) | ~$70 |
| Abliterated judge API | ~$70 |
| **Total project** | **~$270-$310** |

## Critical files we must not lose

- `assistant-axis/assistant_axis/judge.py` — has local 429-retry patch. If `assistant-axis/` is re-cloned, re-apply the patch (see git log of THIS repo for the change).
- `results/comparison/refusal_direction.pt` — input to abliteration. Also mirrored on HF.
- `results/original/axis.pt` + `results/abliterated/axis.pt` — the final outputs of each pipeline. Both mirrored on HF.
