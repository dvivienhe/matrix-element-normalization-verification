"""Evidence rows on Solcore's shipped state arrays (Fig. 6, Table 5): adapter errors, OR / CW / OW and the
corrected adapter against the analytic finite-box reference. Reads data/solcore/, writes results/solcore_interface/."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO / "data" / "solcore"
OUT = REPO / "results" / "solcore_interface"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "code"))

import analytic_finite_box as afb    # noqa: E402
import evidence_rows as evrows      # noqa: E402

LADDER = (0.5, 0.25, 0.125, 0.0625)
LEG = "B"                            # tau_s = 0.10 (finite-difference eigen-solver)
UNIT_FACTOR = 100.0                  # R2: dipole reported in Angstrom (x10) -> rate x100
CASES = {"R0": (), "R1": ("GRID",), "R2": ("UNIT",), "R3": ("GRID", "UNIT")}
TRUTH = {"R0": set(), "R1": {"ENV"}, "R2": {"PRE"}, "R3": {"ENV", "PRE"}}
CONDITIONS = ("full", "-N")
METHODS = ("OR", "CW", "OW")
KNOWN_NODE_DIPOLES = {0.5: 2.3762748203, 0.25: 2.3299912286, 0.125: 2.3078203985, 0.0625: 2.2969642389}
PREDICTED_REF_INTERVAL = (2.2855, 2.2870)


def expected(case: str, condition: str, method: str) -> dict:
    """Expectation table, fixed before the run."""
    if case == "R0":
        return {"detection": "CORRECT-NEG", "localization": None, "conditional": None}
    loc, cond = None, None
    if condition == "full":
        if method == "OR":
            loc = {"R1": "FULL", "R2": "NONE", "R3": "PARTIAL"}[case]
        elif method == "CW":
            loc = {"R1": "FULL", "R2": "NONE", "R3": "NONE"}[case]
        else:
            loc = "FULL"
    else:
        if method == "OR":
            loc = "NONE"
        elif method == "CW":
            loc = {"R1": "FULL", "R2": "MIS", "R3": "PARTIAL"}[case]
        else:
            loc, cond = "NONE", "COND-CORRECT"
    return {"detection": "DETECTED", "localization": loc, "conditional": cond}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_states() -> tuple[dict, dict]:
    ident = json.loads((DATA / "identity.json").read_text(encoding="utf-8"))
    npz = DATA / ident["npz"]
    sha = _sha256(npz)
    if sha != ident["npz_sha256"]:
        raise RuntimeError(f"NPZ SHA256 {sha} != stage-1 record {ident['npz_sha256']}")
    with np.load(npz) as d:
        arrays = {k: d[k] for k in d.files}
    return arrays, ident


def adapted(arrays: dict, variant: str, dz: float, case: str):
    """Consumer adapter: nm coordinates, psi_nm = psi_m * sqrt(1e-9) (continuum-normalized function, unit change
    only).  GRID: psi divided again by sqrt(h) (both envelopes).  Returns (z, psi_a, psi_b, prefactor)."""
    key = f"{variant}_{dz}"
    z = arrays[f"{key}_x"] * 1e9
    pa = arrays[f"{key}_psi0"] * math.sqrt(1e-9)
    pb = arrays[f"{key}_psi1"] * math.sqrt(1e-9)
    if "GRID" in CASES[case]:
        pa, pb = pa / math.sqrt(dz), pb / math.sqrt(dz)
    return z, pa, pb, (UNIT_FACTOR if "UNIT" in CASES[case] else 1.0)


def consumer_rate(z, pa, pb, pref) -> float:
    d = float(np.trapezoid(pa * z * pb, z))
    return pref * d * d


def probes_for(arrays: dict, variant: str, case: str) -> dict:
    na, nb, xr = [], [], []
    for dz in LADDER:
        z, pa, pb, _ = adapted(arrays, variant, dz, case)
        na.append(float(np.trapezoid(pa * pa, z)))
        nb.append(float(np.trapezoid(pb * pb, z)))
        xr.append(float(np.trapezoid(np.ones_like(z), z)) / float(z[-1] - z[0]))
    return {"Na": na, "Nb": nb, "X": xr, "X_key": "W", "derivative_max_dev_from_1": None}


def identity_overlaps(arrays: dict, variant: str) -> list:
    """Adjacent-rung overlap of the correct-path states (finer state interpolated to the coarser grid),
        same formula as mixed_error_set.py (sum * dz_coarse), minimum over the two states.
    """
    out = []
    for dc, df in zip(LADDER[:-1], LADDER[1:]):
        zc, ac, bc, _ = adapted(arrays, variant, dc, "R0")
        zf, af, bf, _ = adapted(arrays, variant, df, "R0")
        oa = abs(float(np.sum(ac * np.interp(zc, zf, af)) * dc))
        ob = abs(float(np.sum(bc * np.interp(zc, zf, bf)) * dc))
        out.append(min(oa, ob))
    return out


def run_variant(arrays: dict, variant: str, q_ref: float) -> dict:
    overlaps = identity_overlaps(arrays, variant)
    cases = {}
    for case in CASES:
        q = []
        for dz in LADDER:
            z, pa, pb, pref = adapted(arrays, variant, dz, case)
            q.append(consumer_rate(z, pa, pb, pref))
        rec = {"case": case, "leg": LEG, "consumer": "int", "Q": q, "Q_ref": q_ref, "dz": list(LADDER),
               "identity_overlap": overlaps}
        probes = probes_for(arrays, variant, case)
        q_ok, q_reason = evrows.q_rows_available(rec, q_ref)
        rows = evrows.evidence_rows(rec, probes, q_ok)
        truth = TRUTH[case]
        cells = {}
        for condition in CONDITIONS:
            crow = evrows.rows_for_condition(rows, condition)
            orr = evrows.method_OR(rec, probes, condition, q_ok, rows)
            results = {"OR": {**orr, "localized_block": orr["localized"], "localized_literal": orr["localized"],
                              "status": "OK", "conditional": None, "cf_agree": True}}
            for world in ("CW", "OW"):
                results[world] = {**evrows.invert(rec, crow, world, condition), "decision": orr["decision"]}
            for method in METHODS:
                r = results[method]
                mi = r.get("status") == "MODEL_INCONSISTENT"
                obs = {"detection": evrows.score_detection(truth, r["decision"]),
                       "localization": evrows.score_localization(truth, r["decision"], r["localized_block"], mi),
                       "conditional": evrows.score_conditional(truth, r["decision"], r.get("conditional"))}
                pred = expected(case, condition, method)
                same = all(pred[k] == obs[k] for k in pred)
                cells[f"{condition}|{method}"] = {
                    "decision": r["decision"], "status": r.get("status"), "localized": r["localized_block"],
                    "localized_literal_rule": r["localized_literal"], "conditional": r.get("conditional"),
                    "score": obs, "expected": pred, "agreement": "agree" if same else "disagree",
                    "proj": {k: list(v) if v is not None else None for k, v in r.get("proj", {}).items()},
                    "ranges": {k: list(v) if v is not None else None for k, v in r.get("ranges", {}).items()},
                    "block_null_feasible": r.get("block_null_feasible", {}), "cf_agree": r.get("cf_agree", True)}
        cases[case] = {"Q": q, "probes": probes, "q_rows_available": q_ok, "q_guard": q_reason, "rows": rows,
                       "truth_sources": sorted(truth), "cells": cells}
    return {"identity_overlap": overlaps, "cases": cases}


def convergence(amplitudes: list, a_ref: float) -> dict:
    rel = [a / a_ref - 1.0 for a in amplitudes]
    adj = [abs(amplitudes[i] - amplitudes[i + 1]) / abs(amplitudes[i + 1]) for i in range(len(amplitudes) - 1)]
    ratios = [adj[i] / adj[i + 1] for i in range(len(adj) - 1)]
    return {"amplitude_nm": amplitudes, "relative_deviation_vs_reference": rel, "adjacent_changes": adj,
            "adjacent_change_ratios": ratios, "second_order": bool(all(r >= 3.0 for r in ratios)),
            "finest_rate_relative_deviation": (amplitudes[-1] / a_ref) ** 2 - 1.0}


def identity(stage1: dict) -> dict:
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
            "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform(),
            "stage1_npz_sha256": stage1["npz_sha256"], "stage1_solcore_head": stage1["solcore_head"],
            "stage1_timestamp_utc": stage1["timestamp_utc"], "stage1_repository_head": stage1["repository_head"]}


def _clean(o):
    if isinstance(o, float):
        return ("+inf" if o > 0 else "-inf") if math.isinf(o) else o
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (set, frozenset)):
        return sorted(_clean(v) for v in o)
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def main() -> None:
    t0 = time.perf_counter()
    arrays, stage1 = load_states()
    ref = afb.reference()
    a_ref = ref["dipole_amplitude_nm"]
    q_ref = ref["rate_nm2"]
    res = {"reference": ref, "variants": {}, "known_check": {}, "convergence": {}}
    for variant in ("node", "cell"):
        res["variants"][variant] = run_variant(arrays, variant, q_ref)
        amps = [math.sqrt(res["variants"][variant]["cases"]["R0"]["Q"][i]) for i in range(len(LADDER))]
        res["convergence"][variant] = convergence(amps, a_ref)
    node_amps = res["convergence"]["node"]["amplitude_nm"]
    res["known_check"] = {str(dz): {"stage2_amplitude_nm": a, "known_nm": KNOWN_NODE_DIPOLES[dz],
                                    "rel_diff": abs(a / KNOWN_NODE_DIPOLES[dz] - 1.0)}
                          for dz, a in zip(LADDER, node_amps)}
    res["reference_prediction"] = {"interval_nm": list(PREDICTED_REF_INTERVAL),
                                   "inside": PREDICTED_REF_INTERVAL[0] <= a_ref <= PREDICTED_REF_INTERVAL[1]}
    res["summary"] = {}
    for variant in ("node",):
        agree = {}
        for case, c in res["variants"][variant]["cases"].items():
            for key, cell in c["cells"].items():
                agree[cell["agreement"]] = agree.get(cell["agreement"], 0) + 1
        res["summary"][variant] = {"agreement": agree,
                                   "closed_form_disagreements": sum(
                                       not cell["cf_agree"] for c in res["variants"][variant]["cases"].values()
                                       for cell in c["cells"].values())}
    res["identity"] = identity(stage1)
    res["setup"] = "Solcore states (data/solcore); cases R0-R3; conditions full, -N; methods OR, CW, OW; expectations fixed before the run"
    res["wall_seconds"] = time.perf_counter() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "solcore_results.json").write_text(json.dumps(_clean(res), indent=1), encoding="utf-8")

    with (OUT / "solcore_cells.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "case", "truth", "condition", "method", "decision", "status", "localized",
                    "conditional", "score_detection", "score_localization", "score_conditional",
                    "expected_detection", "expected_localization", "expected_conditional", "agreement",
                    "y_Na_level", "y_Na_slope", "y_Nb_level", "y_Nb_slope", "y_W_level", "y_W_slope",
                    "y_Q_level", "y_Q_slope"])
        for variant in ("node", "cell"):
            for case, c in res["variants"][variant]["cases"].items():
                y = {row["name"]: row["y"] for row in c["rows"]}
                for key, cell in c["cells"].items():
                    cond, method = key.split("|")
                    w.writerow([variant, case, "+".join(c["truth_sources"]) or "-", cond, method, cell["decision"],
                                cell["status"], "+".join(cell["localized"]) or "-",
                                ("(+)".join(cell["conditional"]) if cell["conditional"] else "-"),
                                cell["score"]["detection"], cell["score"]["localization"],
                                cell["score"]["conditional"], cell["expected"]["detection"],
                                cell["expected"]["localization"], cell["expected"]["conditional"],
                                cell["agreement"],
                                *[y.get(k) for k in ("Na_level", "Na_slope", "Nb_level", "Nb_slope", "W_level",
                                                     "W_slope", "Q_level", "Q_slope")]])

    lines = ["=== Solcore interface: evidence rows on the solver's state arrays ===",
             f"stage 1: solcore {stage1['solcore_head']} dirty={stage1['solcore_dirty']}  npz {stage1['npz_sha256']}",
             "",
             f"reference: E0 {ref['E0_eV']:.12f} eV  E1 {ref['E1_eV']:.12f} eV  dipole {ref['dipole_nm_quad']:+.12f} nm"
             f"  (Simpson rel diff {ref['dipole_rel_diff_quad_vs_simpson']:.2e}; norms {ref['norm0']:.15f} / "
             f"{ref['norm1']:.15f}; overlap {ref['overlap01']:.1e}; residuals {ref['eigen_residual_even']:.1e} / "
             f"{ref['eigen_residual_odd']:.1e})",
             f"reference prediction [2.2855, 2.2870] nm -> inside = {res['reference_prediction']['inside']}",
             "", "known-value check (node driver, previously recorded values):"]
    for dz, kc in res["known_check"].items():
        lines.append(f"  dz {dz:7s} stage-2 amplitude {kc['stage2_amplitude_nm']:.10f} nm  known {kc['known_nm']:.10f}"
                     f"  rel diff {kc['rel_diff']:.2e}")
    for variant in ("node", "cell"):
        cv = res["convergence"][variant]
        lines.append("")
        lines.append(f"corrected adapter vs independent reference ({variant} driver):")
        lines.append("  amplitude rel. deviation per rung: " + " / ".join(f"{v:+.5%}" for v in cv["relative_deviation_vs_reference"]))
        lines.append("  adjacent changes: " + " / ".join(f"{v:.7f}" for v in cv["adjacent_changes"])
                     + "   ratios: " + " / ".join(f"{v:.3f}" for v in cv["adjacent_change_ratios"])
                     + f"   second order: {cv['second_order']}")
        lines.append(f"  finest-rung rate deviation vs reference: {cv['finest_rate_relative_deviation']:+.5%}")
    for variant in ("node", "cell"):
        lines.append("")
        lines.append(f"cells ({variant} driver){'  [expectation table]' if variant == 'node' else '  [descriptive only]'}:")
        for case, c in res["variants"][variant]["cases"].items():
            y = {row["name"]: row["y"] for row in c["rows"]}
            lines.append(f"  {case} S={c['truth_sources']} q_ok={c['q_rows_available']}  "
                         + "  ".join(f"{k}={v:+.6g}" for k, v in y.items()))
            for key, cell in c["cells"].items():
                k0 = cell["proj"].get("k0")
                lines.append(f"     {key:8s} {cell['decision']:10s} {str(cell['status']):18s} loc={cell['localized']}"
                             f" cond={cell['conditional']} score={cell['score']}"
                             + (f" [{cell['agreement']}]" if variant == "node" else "")
                             + (f" k0={k0}" if k0 is not None and key.endswith('OW') else ""))
    lines.append("")
    lines.append(f"summary (node): {res['summary']['node']}")
    text = "\n".join(lines)
    print(text)


if __name__ == "__main__":
    main()
