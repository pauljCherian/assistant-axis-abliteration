#!/usr/bin/env python3
"""Build pre-registered predictions for the persona-space comparison.

Before we see the abliterated axis, record:
  1. Which roles we predict will move most (safety-adjacent)
  2. The operational rule used to classify a role as safety-adjacent

Output: results/comparison/predictions.json

This is hypothesis pre-registration: after the abliterated run, we test
whether our predicted safety-adjacent set overlaps significantly with the
observed top-movers.
"""
import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROLES_DIR = ROOT / "assistant-axis/data/roles/instructions"
VECTORS_DIR = ROOT / "results/original/vectors"
OUT = ROOT / "results/comparison/predictions.json"

# Keywords that indicate a role has explicit protective / ethical / harm-avoidant
# orientation, OR the opposite (chaotic / destructive / boundary-crossing).
# The hypothesis is: abliteration removes refusal, so roles whose instructions
# lean on refusal-adjacent concepts are the ones most perturbed.
PROTECTIVE = [
    r"\bprotect",
    r"\bsafeguard",
    r"\bdefend",
    r"\bshield",
    r"\bharm",
    r"\bethic",
    r"\bmoral",
    r"\bintegrity",
    r"\bjustice",
    r"\bguard",
    r"\brefus",
    r"\bboundary",
    r"\bboundaries",
    r"\bvigilant",
    r"\bduty",
    r"\baccountab",
    r"\bvirtue",
    r"\brighteous",
]
TRANSGRESSIVE = [
    r"\bchaos",
    r"\bdestroy",
    r"\bdestruct",
    r"\bsabotag",
    r"\bmanipulat",
    r"\btransgress",
    r"\btrickst",
    r"\bdeceiv",
    r"\bforbidden",
    r"\bsubvers",
    r"\bamoral",
    r"\bnihilis",
    r"\btaboo",
    r"\bvillain",
    r"\bcorrupt",
]

# Roles flagged a priori by CLAUDE.md as expected movers.
CLAUDE_MD_PREDICTED = {
    "guardian", "ethicist", "judge", "demon", "saboteur", "angel",
    "vigilante", "nihilist",
}


def score_role(text: str) -> dict:
    text_l = text.lower()
    prot = sum(1 for p in PROTECTIVE if re.search(p, text_l))
    trans = sum(1 for p in TRANSGRESSIVE if re.search(p, text_l))
    return {"protective_hits": prot, "transgressive_hits": trans}


def main():
    role_files = sorted(ROLES_DIR.glob("*.json"))
    vector_files = {f.stem for f in VECTORS_DIR.glob("*.pt")}

    classifications = []
    for rf in role_files:
        role = rf.stem
        data = json.loads(rf.read_text())
        # Concatenate all system prompts + all questions into one blob per role.
        parts = []
        for inst in data.get("instruction", []):
            # Support both {"pos": "..."} and older plain strings.
            if isinstance(inst, dict):
                parts.extend(inst.values())
            else:
                parts.append(str(inst))
        parts.extend(data.get("questions", []))
        blob = "\n".join(parts)
        sc = score_role(blob)
        sc["role"] = role
        sc["has_vector"] = role in vector_files
        sc["claude_md_predicted"] = role in CLAUDE_MD_PREDICTED
        sc["total_safety_adjacency"] = sc["protective_hits"] + sc["transgressive_hits"]
        classifications.append(sc)

    classifications.sort(key=lambda x: -x["total_safety_adjacency"])

    safety_adjacent = [
        c["role"] for c in classifications
        if c["total_safety_adjacency"] >= 2 or c["claude_md_predicted"]
    ]

    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "description": (
            "Pre-registered predictions for persona-space comparison. "
            "Hypothesis: safety-adjacent roles (protective OR transgressive) "
            "are the roles whose activation vectors move most after abliteration."
        ),
        "rule": (
            "A role is SAFETY-ADJACENT if its combined system-prompt + questions "
            "text matches >=2 protective/transgressive keyword patterns, "
            "OR it appears in CLAUDE_MD_PREDICTED set."
        ),
        "protective_patterns": PROTECTIVE,
        "transgressive_patterns": TRANSGRESSIVE,
        "claude_md_predicted_roles": sorted(CLAUDE_MD_PREDICTED),
        "n_roles_scanned": len(role_files),
        "n_safety_adjacent": len(safety_adjacent),
        "safety_adjacent_roles": sorted(safety_adjacent),
        "per_role_scores": classifications,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT}")
    print(f"Safety-adjacent: {len(safety_adjacent)} / {len(role_files)}")
    print("Top 20 by total safety adjacency:")
    for c in classifications[:20]:
        print(f"  {c['role']:25s} prot={c['protective_hits']:2d}  trans={c['transgressive_hits']:2d}  "
              f"claude_md={'Y' if c['claude_md_predicted'] else ' '}")


if __name__ == "__main__":
    main()
