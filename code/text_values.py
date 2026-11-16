"""Numbers quoted in the article text (E3 probe model error, wall-offset maxima, Solcore constant) and the
derived values of Table A.4: arithmetic on existing results, data and literature values only. Writes
results/text_values.json and results/derived_values.json."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
EVIDENCE = REPO / "results/evidence_rows/evidence_rows.json"
WALL = REPO / "results/wall_offset.json"
STAGE1 = REPO / "data/solcore/identity.json"
SOLCORE_DIR = REPO / "code/solcore_interface"
OUT = REPO / "results" / "text_values.json"
OUT_A4 = REPO / "results" / "derived_values.json"
GAMMA = {"GaSb": (13.4, 4.7), "InAs": (20.0, 8.5)}      # Luttinger parameters, Vurgaftman et al. 2001 (Table A.4)


def derived_values() -> bool:
    """Table A.4: m_z = 1/(gamma1 - 2 gamma2), m_par = 1/(gamma1 + gamma2), r = m_z / m_par, compared with the
    printed values at the printed precision."""
    checks = []

    def check(label, value, printed, decimals):
        checks.append({"item": label, "value": float(value), "printed": printed, "decimals": decimals,
                       "ok": bool(round(float(value), decimals) == round(printed, decimals))})

    table_a4 = {}
    for mat, (g1, g2) in GAMMA.items():
        mz, mp = 1 / (g1 - 2 * g2), 1 / (g1 + g2)
        table_a4[mat] = {"gamma1": g1, "gamma2": g2, "m_z": mz, "m_par": mp, "r": mz / mp}
    check("A.4 GaSb m_z", table_a4["GaSb"]["m_z"], 0.250, 3)
    check("A.4 GaSb m_par", table_a4["GaSb"]["m_par"], 0.0552, 4)
    check("A.4 GaSb r", table_a4["GaSb"]["r"], 4.53, 2)
    check("A.4 InAs m_z", table_a4["InAs"]["m_z"], 0.333, 3)
    check("A.4 InAs m_par", table_a4["InAs"]["m_par"], 0.0351, 4)
    check("A.4 InAs r", table_a4["InAs"]["r"], 9.50, 2)
    ok = all(x["ok"] for x in checks)
    OUT_A4.parent.mkdir(exist_ok=True)
    OUT_A4.write_text(json.dumps({"inputs": {"luttinger": "Vurgaftman et al. 2001 (Table A.4)"},
                                  "table_A4": table_a4, "checks": checks, "all_ok": ok}, indent=1) + "\n",
                      encoding="utf-8", newline="\n")
    for x in checks:
        print(f"{'ok ' if x['ok'] else 'BAD'} {x['item']:32s} {x['value']:+.6g}  printed {x['printed']}")
    return ok


def sha_lf(p: Path) -> str:
    """SHA-256 after CRLF -> LF (the repository stores LF; a Windows work tree or archive may carry CRLF)."""
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def e3_readings(ev: dict) -> dict:
    rows = {r["case"]: r for r in ev["rows"]}
    s04, n01 = rows["S04"], rows["N01"]
    assert s04["errors"] == ["E3"] and n01["errors"] == [] and s04["leg"] == n01["leg"] == "A"
    assert s04["probes"]["X_key"] == "W"
    h_f = ev["constants"]["h_f"]
    x_probe = s04["probes"]["X"][-1]
    q_s04, q_n01, q_ref = s04["Q"][-1], n01["Q"][-1], s04["Q_ref"]
    assert q_ref == n01["Q_ref"]
    y = {r["name"]: r for r in s04["rows"]}
    mu = math.log(x_probe) - 0.5 * math.log(q_s04 / q_n01)
    ideal = math.log(x_probe) - math.log(1.0 / h_f)
    disc = math.log(q_n01 / q_ref)
    r_hat = disc - 2.0 * mu
    r_rows = y["Q_level"]["y"] - y["Na_level"]["y"] - y["Nb_level"]["y"] - 2.0 * y["W_level"]["y"]
    e_r = y["Q_level"]["eps"] + y["Na_level"]["eps"] + y["Nb_level"]["eps"] + 2.0 * y["W_level"]["eps"]
    return {
        "cases": "S04 (E3 only) vs N01 (no error); leg A; finest rung", "h_f": h_f,
        "X_probe": x_probe, "Q_S04": q_s04, "Q_N01": q_n01, "Q_ref": q_ref,
        "mu_probe_vs_actual_same_state": mu, "probe_vs_ideal_1_over_h": ideal,
        "ideal_minus_mu": ideal - mu, "normal_discretization_ln_QN01_over_Qref": disc,
        "r_hat_decomposed": r_hat, "r_hat_eq11_rows": r_rows, "r_hat_agreement_abs": abs(r_hat - r_rows),
        "E_r": e_r, "r_hat_over_E_r": r_rows / e_r, "k0_interval": [(r_rows - e_r) / 2.0, (r_rows + e_r) / 2.0],
        "text_values": {"mu": f"{mu:.5f}", "ideal": f"{ideal:.5f}", "normal_discretization": f"{disc:.1e}",
                        "r_hat": f"{r_rows:.4f}"},
    }


def wall_readings(w: dict) -> dict:
    rows = w["rows"]
    top = max(rows, key=lambda r: r["relative_change"])
    cell = max(rows, key=lambda r: r["fraction_of_measured_deviation"]["cell"])
    node = max(rows, key=lambda r: r["fraction_of_measured_deviation"]["node"])
    return {
        "max_relative_change": top["relative_change"], "max_relative_change_at_h_nm": top["h_nm"],
        "max_fraction_cell": cell["fraction_of_measured_deviation"]["cell"], "max_fraction_cell_at_h_nm": cell["h_nm"],
        "max_fraction_node": node["fraction_of_measured_deviation"]["node"], "max_fraction_node_at_h_nm": node["h_nm"],
        "rounded_value_exceeds_rounding": {
            "4.3e-7": top["relative_change"] > 4.3e-7,
            "0.13 percent": cell["fraction_of_measured_deviation"]["cell"] > 0.0013},
        "text_values": {"max_relative_change": f"{top['relative_change']:.1e}",
                        "max_fraction_cell_percent": f"{100 * cell['fraction_of_measured_deviation']['cell']:.2f}",
                        "max_fraction_node": f"{node['fraction_of_measured_deviation']['node']:.1e}"},
    }


def constant_readings(stage1: dict) -> dict:
    sys.path.insert(0, str(SOLCORE_DIR))
    try:
        import analytic_finite_box as afb
    finally:
        sys.path.remove(str(SOLCORE_DIR))
    solcore = stage1["solcore_constants"]["hbar2_over_2m0_eV_nm2"]
    ref = afb.C0_EV_NM2
    return {"solcore_hbar2_over_2m0_eV_nm2": solcore, "reference_hbar2_over_2m0_eV_nm2": ref,
            "absolute_difference_eV_nm2": solcore - ref, "relative_difference": (solcore - ref) / ref,
            "text_values": {"relative_difference": f"{(solcore - ref) / ref:.1e}"}}


def main() -> None:
    ev = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    wall = json.loads(WALL.read_text(encoding="utf-8"))
    stage1 = json.loads(STAGE1.read_text(encoding="utf-8"))
    res = {
        "check": "numbers quoted in the text: E3 probe model error, wall-offset maxima, Solcore constant difference",
        "inputs": {p.relative_to(REPO).as_posix(): {"sha256_lf_normalized": sha_lf(p)}
                   for p in (EVIDENCE, WALL, STAGE1, SOLCORE_DIR / "analytic_finite_box.py")},
        "e3_probe_model_error": e3_readings(ev),
        "wall_offset_maxima": wall_readings(wall),
        "solcore_constant_difference": constant_readings(stage1),
        "identity": {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version.split()[0]},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    e, w = res["e3_probe_model_error"], res["wall_offset_maxima"]
    print(f"E3: mu = {e['mu_probe_vs_actual_same_state']:.13f} (probe vs actual, same state); "
          f"vs ideal 1/h = {e['probe_vs_ideal_1_over_h']:.13f}; normal discretization "
          f"{e['normal_discretization_ln_QN01_over_Qref']:.6e}; r_hat = {e['r_hat_decomposed']:.13f} "
          f"(Eq. 11 rows {e['r_hat_eq11_rows']:.13f}); E_r = {e['E_r']}; k0 in {e['k0_interval']}")
    print(f"wall: max relative change {w['max_relative_change']:.12e} at h = {w['max_relative_change_at_h_nm']} nm; "
          f"max fraction (cell) {w['max_fraction_cell']:.12e} at h = {w['max_fraction_cell_at_h_nm']} nm; "
          f"(node) {w['max_fraction_node']:.6e}")
    c = res["solcore_constant_difference"]
    print(f"hbar^2/(2 m0): Solcore {c['solcore_hbar2_over_2m0_eV_nm2']} vs reference "
          f"{c['reference_hbar2_over_2m0_eV_nm2']} eV nm^2 -> relative {c['relative_difference']:.6e}")
    print("->", OUT.relative_to(REPO).as_posix())
    ok = derived_values()
    print("Table A.4 all_ok =", ok)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
