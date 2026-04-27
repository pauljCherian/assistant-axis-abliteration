"""Stage 2 (filtered): re-run precheck on Anthropic-recipe filtered vectors.

Compares cos(v[L], axis_orig) for filtered vs unfiltered vectors side-by-side.
Filter retention rates pulled from filtered_extraction_summary.json.
"""
import json
from pathlib import Path
import torch

REPO = Path("/scratch/paulc/assistant-axis-abliteration")
PV_DIR = REPO / "results/persona_vectors"

ALL_TRAITS = ["evil", "sycophantic", "apathetic", "humorous",
              "impolite", "hallucinating", "optimistic", "assistant_identity"]

L_INJECT = 12
L_EXTRACT = 16


def cos(a, b):
    return float((a / a.norm()).dot(b / b.norm()))


def load_axis():
    a = torch.load(REPO / "results/original/axis.pt", weights_only=False)
    if isinstance(a, dict):
        for k in ("axis", "vector", "direction"):
            if k in a:
                return torch.as_tensor(a[k]).float().squeeze()
    return torch.as_tensor(a).float().squeeze()


def load_refusal():
    d = torch.load(REPO / "results/comparison/refusal_direction_from_mlabonne.pt", weights_only=False)
    return d["per_layer"].float()


def load_one(trait, suffix):
    p = PV_DIR / f"{trait}_response_avg_diff{suffix}.pt"
    if not p.exists():
        return None
    d = torch.load(p, weights_only=False)
    if isinstance(d, dict):
        v = d.get("vector", d.get("diff"))
    else:
        v = d
    return torch.as_tensor(v).float()


def main():
    axis = load_axis()
    refusal = load_refusal()
    summary = json.loads((PV_DIR / "filtered_extraction_summary.json").read_text())
    # summary structure: per-trait dict with kept counts
    print(f"axis norm: {axis.norm():.3f}")
    print()

    rows = []
    rows.append("| Trait | kept | retention | cos(v[12],axis) unfilt | filt | Δ | cos(v[16],axis) unfilt | filt | Δ | cos(v[12],ref) filt |")
    rows.append("|---|---|---|---|---|---|---|---|---|---|")

    out = {"L_inject": L_INJECT, "L_extract": L_EXTRACT, "traits": {}}

    for t in ALL_TRAITS:
        v_un = load_one(t, "")
        v_fl = load_one(t, "_filtered")
        if v_fl is None:
            continue

        info = summary.get(t) if isinstance(summary, dict) else None
        kept = info.get("n_kept", "?") if isinstance(info, dict) else "?"
        rate = info.get("filter_rate") if isinstance(info, dict) else None
        retention = f"{rate*100:.0f}%" if rate is not None else "?"

        c12_fl = cos(v_fl[L_INJECT], axis)
        c16_fl = cos(v_fl[L_EXTRACT], axis)
        cref_fl = cos(v_fl[L_INJECT], refusal[L_INJECT])
        if v_un is not None:
            c12_un = cos(v_un[L_INJECT], axis)
            c16_un = cos(v_un[L_EXTRACT], axis)
            d12 = c12_fl - c12_un
            d16 = c16_fl - c16_un
        else:
            c12_un = c16_un = d12 = d16 = float("nan")

        rows.append(f"| {t} | {kept} | {retention} | {c12_un:+.3f} | {c12_fl:+.3f} | {d12:+.3f} | {c16_un:+.3f} | {c16_fl:+.3f} | {d16:+.3f} | {cref_fl:+.3f} |")
        out["traits"][t] = {
            "kept_pairs": kept,
            "cos_axis_L12_unfiltered": c12_un,
            "cos_axis_L12_filtered": c12_fl,
            "cos_axis_L16_unfiltered": c16_un,
            "cos_axis_L16_filtered": c16_fl,
            "cos_refusal_L12_filtered": cref_fl,
        }

    md = ["# Stage 2 (filtered) — Anthropic-recipe persona vectors",
          "",
          f"L_inject={L_INJECT}, L_extract={L_EXTRACT}",
          ""] + rows
    out_md = PV_DIR / "precheck_filtered.md"
    out_json = PV_DIR / "precheck_filtered.json"
    out_md.write_text("\n".join(md))
    out_json.write_text(json.dumps(out, indent=2))
    print("\n".join(md))
    print()
    print(f"saved {out_md}")
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
