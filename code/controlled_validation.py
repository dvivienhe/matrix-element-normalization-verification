"""Controlled validation of the toy model (Figs. 1-2, Table 2, Table A.1): both legs and all consumers over the grid
ladder, the cross-unit experiments and the kernel variant; predicted exponents are fixed in the code. Writes
results/controlled/diagnostic_matrix.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from toy_model import (DZ_LADDER, LAMBDA_S_NM, LAMBDA_S_VARIANT_NM,
                          M_PAR_TOY, M_Z_TOY, c1_correct_dipole, c2_bug_dipole,
                          c7_bug_coulomb, c7_correct_coulomb, fit_slope,
                          leg_a_states, ref_leg_a, ref_leg_b, sweep,
                          sweep_c7_variant)

OUT = Path(__file__).resolve().parents[1] / "results" / "controlled"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------- cross-unit experiment: the same bug in three internal length units
UNITS = {"nm": 1.0, "angstrom": 10.0, "m": 1.0e-9}


def run_cross_units() -> dict:
    out = {}
    for unit, s in UNITS.items():
        rows = []
        for dz in DZ_LADDER:
            z, ca, cb = leg_a_states(dz)
            z_u, dz_u = z * s, dz * s          # same physics, unit-u numbers
            r_correct = c1_correct_dipole(z_u, ca, cb, dz_u)
            r_bug = c2_bug_dipole(z_u, ca, cb, dz_u)
            rows.append({"dz_nm": dz, "dz_u": dz_u,
                         "factor": r_bug / r_correct,
                         "factor_pred": dz_u ** 2})
        x = np.log([r["dz_nm"] for r in rows])
        y = np.log([r["factor"] for r in rows])
        slope = float(np.polyfit(x, y, 1)[0])
        out[unit] = {"rows": rows, "slope_obs": slope, "slope_pred": 2.0}
    return out


def run_cross_units_c7() -> dict:
    """The four-factor bug in three internal length units (expected factor (dz_u)^4)."""
    out = {}
    for unit, s in UNITS.items():
        rows = []
        for dz in DZ_LADDER:
            z, ca, cb = leg_a_states(dz)
            z_u, dz_u, lam_u = z * s, dz * s, LAMBDA_S_NM * s
            r_correct = c7_correct_coulomb(z_u, ca, cb, dz_u, lam_u)
            r_bug = c7_bug_coulomb(z_u, ca, cb, dz_u, lam_u)
            rows.append({"dz_nm": dz, "dz_u": dz_u,
                         "factor": r_bug / r_correct,
                         "factor_pred": dz_u ** 4})
        x = np.log([r["dz_nm"] for r in rows])
        y = np.log([r["factor"] for r in rows])
        slope = float(np.polyfit(x, y, 1)[0])
        out[unit] = {"rows": rows, "slope_obs": slope, "slope_pred": 4.0}
    return out


# ---------------------------------------- diagnostic matrix

R_TOY = M_Z_TOY / M_PAR_TOY          # 4.5454...

# ---------------------------------------- predictions (fixed before the run) ----
PRED = {
    "C1": {"slope": 0.0, "gate": "PASS"},
    "C2": {"slope": 2.0, "gate": "FAIL"},
    "C3": {"slope": 2.0, "gate": "FAIL"},
    "C4": {"slope": 0.0, "gate": "PASS"},
    "C5": {"slope": 0.0, "gate": "FAIL"},   # constant offset 1/100 vs C1
    "C6": {"slope": 0.0, "gate": "FAIL"},   # constant offset r_toy
    "C7": {"slope": 4.0, "gate": "FAIL"},   # four-factor bug
    "C7c": {"slope": 0.0, "gate": "PASS"},  # four-factor correct control
}
SLOPE_TOL = {"A": 0.05, "B": 0.10}
GATE_BAND = 0.02
FACTOR_TOL = {"A": 0.05, "B": 0.10}
ALG_TOL = 1e-9


def leg_matrix(leg: str) -> dict:
    raw = sweep(leg)
    ref = ref_leg_a() if leg == "A" else ref_leg_b()
    dz = raw["dz"]
    i_fine = dz.index(0.05)
    refmap = {"C1": ref["dipole_rate"], "C2": ref["dipole_rate"],
              "C3": ref["bardeen_rate"], "C4": ref["mean_w"],
              "C5": ref["dipole_rate"], "C6": ref["dos"],
              "C7": ref["coulomb_rate"], "C7c": ref["coulomb_rate"]}
    rows, all_match = {}, True
    for c in ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C7c"):
        vals = raw[c]
        s_obs = fit_slope(dz, vals)
        s_ok = abs(s_obs - PRED[c]["slope"]) <= SLOPE_TOL[leg]
        dev_fine = abs(vals[i_fine] / refmap[c] - 1.0)
        gate_obs = "PASS" if dev_fine <= GATE_BAND else "FAIL"
        gate_ok = gate_obs == PRED[c]["gate"]
        checks = {}
        if c in ("C2", "C3"):
            f_obs = [v / refmap[c] for v in vals]
            f_pred = [d * d for d in dz]
            worst = max(abs(a / b - 1.0) for a, b in zip(f_obs, f_pred))
            checks["factor_worst_rel_dev"] = worst
            checks["factor_ok"] = worst <= FACTOR_TOL[leg]
        elif c == "C7":
            f_obs = [v / refmap[c] for v in vals]
            f_pred = [d ** 4 for d in dz]
            worst = max(abs(a / b - 1.0) for a, b in zip(f_obs, f_pred))
            ident = max(abs(v7 / (v7c * d ** 4) - 1.0)
                        for v7, v7c, d in zip(raw["C7"], raw["C7c"], dz))
            checks["factor_worst_rel_dev"] = worst
            checks["identity_vs_C7c_worst_rel_dev"] = ident
            checks["factor_ok"] = (worst <= FACTOR_TOL[leg]
                                   and ident <= ALG_TOL)
        elif c == "C5":
            worst = max(abs(v5 / v1 - 0.01) / 0.01
                        for v5, v1 in zip(raw["C5"], raw["C1"]))
            checks["offset_vs_C1_worst_rel_dev"] = worst
            checks["factor_ok"] = worst <= ALG_TOL
        elif c == "C6":
            worst = max(abs(v / ref["dos"] - R_TOY) / R_TOY for v in vals)
            checks["offset_vs_ref_worst_rel_dev"] = worst
            checks["factor_ok"] = worst <= ALG_TOL
        else:
            checks["factor_ok"] = True
        match = s_ok and gate_ok and checks["factor_ok"]
        all_match = all_match and match
        rows[c] = {"values": vals, "slope_obs": s_obs,
                   "slope_pred": PRED[c]["slope"], "slope_ok": s_ok,
                   "gate_obs": gate_obs, "gate_pred": PRED[c]["gate"],
                   "gate_ok": gate_ok, "dev_at_0p05": dev_fine,
                   **checks, "MATCH": match}
    return {"dz": dz, "reference": refmap, "rows": rows,
            "C3_correct": raw["C3_correct"], "ALL_MATCH": all_match}


def main() -> None:
    print("=== toy model: predicted vs observed scaling ===\n")
    mats = {"A": leg_matrix("A"), "B": leg_matrix("B")}
    for leg in ("A", "B"):
        m = mats[leg]
        print(f"Leg-{leg}:  (dz ladder {m['dz']})")
        print(f"  {'':4} {'N_pred':>6} {'N_obs':>8} {'gate_pred':>9} "
              f"{'gate_obs':>8} {'MATCH':>6}")
        for c, r in m["rows"].items():
            print(f"  {c:4} {r['slope_pred']:>6.1f} {r['slope_obs']:>8.4f} "
                  f"{r['gate_pred']:>9} {r['gate_obs']:>8} "
                  f"{str(r['MATCH']):>6}")
        print(f"  ALL_MATCH = {m['ALL_MATCH']}\n")

    cu = run_cross_units()
    print("Cross-unit (C2 factor at physical dz=0.20 nm):")
    for unit, d in cu.items():
        f02 = next(r["factor"] for r in d["rows"] if r["dz_nm"] == 0.20)
        print(f"  {unit:9} slope={d['slope_obs']:.4f}  factor@0.20={f02:.3e}")
    print()

    cu7 = run_cross_units_c7()
    print("Cross-unit C7 (four-factor; factor at physical dz=0.20 nm):")
    for unit, d in cu7.items():
        f02 = next(r["factor"] for r in d["rows"] if r["dz_nm"] == 0.20)
        print(f"  {unit:9} slope={d['slope_obs']:.4f}  factor@0.20={f02:.3e}")
    print()

    kv = sweep_c7_variant()
    kv_slope = fit_slope(kv["dz"], kv["C7"])
    kv_ident = max(abs(v7 / (v7c * d ** 4) - 1.0)
                   for v7, v7c, d in zip(kv["C7"], kv["C7c"], kv["dz"]))
    kv_ok = abs(kv_slope - 4.0) <= SLOPE_TOL["A"] and kv_ident <= ALG_TOL
    zfine, ca, cb = leg_a_states(0.05)
    pref_ratio = (c7_correct_coulomb(zfine, ca, cb, 0.05, LAMBDA_S_NM)
                  / c7_correct_coulomb(zfine, ca, cb, 0.05,
                                       LAMBDA_S_VARIANT_NM))
    print(f"Kernel variant (lambda_s={kv['lam']} nm, Leg-A): "
          f"slope={kv_slope:.4f}  identity_worst={kv_ident:.2e}  "
          f"OK={kv_ok}  rate-prefactor ratio (2.0/0.5)={pref_ratio:.4f}\n")

    payload = {"setup": "toy model: legs A and B, consumers C1-C7 and C7c, "
                        "cross-unit experiments (n = 2, 4), kernel variant",
               "legs": mats, "cross_units": cu, "cross_units_c7": cu7,
               "kernel_variant": {"lam_nm": kv["lam"], "dz": kv["dz"],
                                  "C7": kv["C7"], "C7c": kv["C7c"],
                                  "slope_obs": kv_slope, "slope_pred": 4.0,
                                  "identity_worst_rel_dev": kv_ident,
                                  "prefactor_ratio_2p0_over_0p5": pref_ratio,
                                  "OK": kv_ok}}
    (OUT / "diagnostic_matrix.json").write_text(
        json.dumps(payload, indent=1, default=float), encoding="utf-8")
    overall = mats["A"]["ALL_MATCH"] and mats["B"]["ALL_MATCH"] and \
        all(abs(d["slope_obs"] - 2.0) <= 0.05 for d in cu.values()) and \
        all(abs(d["slope_obs"] - 4.0) <= 0.05 for d in cu7.values()) and \
        kv_ok
    print(f"\nOVERALL predicted-vs-observed: "
          f"{'ALL MATCH' if overall else 'DEVIATIONS PRESENT (see JSON)'}")


if __name__ == "__main__":
    main()
