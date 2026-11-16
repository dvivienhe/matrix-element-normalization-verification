"""Six-method layered scoring of the 27-case mixed-error set (Fig. S1, Table S5): truth audit, B1n numerical
interface audit, P split and P' guards on top of mixed_error_set.py. Run as a program it writes
results/mixed_error_set/; evidence_rows.py imports it."""
from __future__ import annotations

import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import mixed_error_set as errset  # noqa: E402

DZ_LADDER = errset.DZ_LADDER
CASES = errset.CASES
CANCEL_CASES = errset.CANCEL_CASES
CONSUMER_ERRORS = errset.CONSUMER_ERRORS
R_TOY, UNIT_MIX = errset.R_TOY, errset.UNIT_MIX
SLOPE_TOL, GATE_BAND, FACTOR_TOL = errset.SLOPE_TOL, errset.GATE_BAND, errset.FACTOR_TOL
ZERO_FLOOR, ID_TOL = errset.ZERO_FLOOR, errset.ID_TOL
TAU_N = 1e-6                      # probe tolerance (relative)
RMS_TOL = 0.05                    # four-point log-fit residual RMS

METHODS = ("B1", "B1n", "B2", "B3", "P", "Pp")
NUMERIC = {"B2", "B3", "P", "Pp"}
INTERFACE = {"B1", "B1n"}
VISIBLE = {"B1": {"E1", "E7"},
           "B1n": {"E1", "E2", "E3", "E4", "E5", "E7", "E10"}}
MEASURE_PRIMS = {"E1": 2, "E2": -2, "E3": -2, "E4": 2, "E7": 1, "E10": 2}
DOWNSTREAM_PRIMS = {"E2", "E3", "E4", "E10"}     # not visible at the declaration boundary
B_LEG_CASES = {"N02", "S02", "S05", "K02", "M02", "M06", "N04"}

PROVENANCE = {}
for _c in ("N01", "N02", "S01", "S02", "K01", "K02"):
    PROVENANCE[_c] = "regression-control"
for _c in ("N03", "N04", "S03", "S04", "S05", "S06", "S07", "K03", "P01"):
    PROVENANCE[_c] = "primitive-new"
for _c in ("M01", "M02", "M03", "M04", "M05", "M06", "M07", "M08"):
    PROVENANCE[_c] = "combination-new"
for _c in ("Z01", "Z02", "T01", "T02"):
    PROVENANCE[_c] = "guard-new"
PROVENANCE_NOTE = {"N01": "controlled validation leg A C1", "N02": "controlled validation leg B C1", "S01": "controlled validation leg A C2",
                   "S02": "controlled validation leg B C2", "K01": "controlled validation leg A C5", "K02": "controlled validation leg B C5"}


# ------------------------------------------------------------------ truth audit --
def realized_primitives(errors) -> set:
    """Primitives that actually act on the consumer path (E1 short-circuits the conversion
    step, so a co-registered E5 never acts)."""
    err = set(errors)
    real = set(err)
    if "E1" in err:
        real.discard("E5")
    return real


def n_converted_sides(errors) -> int:
    err = set(errors)
    return 0 if "E1" in err else (1 if "E7" in err else 2)


def audited_kappa(errors) -> float:
    err = set(errors)
    k = 1.0
    if "E5" in err:
        k *= 10.0 ** (-n_converted_sides(errors))      # 10^-1/2 per converted envelope, squared
    if "E6" in err:
        k *= R_TOY
    return k


def audited_truth(case_row) -> dict:
    case, leg, consumer, errors, k_true, kappa_reg, note = case_row
    err = set(errors)
    real = realized_primitives(errors)
    kappa_aud = audited_kappa(errors)
    c_out = set()
    if k_true is not None:
        if k_true != 0:
            c_out.add("MEASURE")
        if abs(kappa_aud - 1.0) > FACTOR_TOL[leg]:
            c_out.add("CONSTANT")
    c_cmp = set()
    if real & set(MEASURE_PRIMS):
        c_cmp.add("MEASURE")
    if real & {"E5", "E6"}:
        c_cmp.add("CONSTANT")
    k_ir = 2 if "E1" in err else (1 if "E7" in err else 0)
    return {
        "registered": {"k": k_true, "kappa": kappa_reg},
        "audited": {"k": k_true, "kappa": kappa_aud},
        "kappa_changed": bool(kappa_reg is not None and abs(kappa_aud - kappa_reg) > 1e-12),
        "realized_primitives": sorted(real),
        "C_out": sorted(c_out), "C_cmp": sorted(c_cmp),
        "k_cmp": int(sum(MEASURE_PRIMS[p] for p in real if p in MEASURE_PRIMS)),
        "k_IR": k_ir,
        "output_error": bool(err & CONSUMER_ERRORS) and case not in CANCEL_CASES,
        "localizable": bool(k_ir > 0 and not (real & DOWNSTREAM_PRIMS)),
        "provenance": PROVENANCE[case], "provenance_note": PROVENANCE_NOTE.get(case, ""),
    }


# -------------------------------------- consumers on the same branches as the base module --
def _quadrature(integrand, z, dz: float, err: set) -> float:
    m = float(np.sum(integrand)) if "E3" in err else float(np.trapezoid(integrand, z))
    if "E4" in err:
        m *= dz
    return m


def _derivative(c, dz: float, err: set):
    out = np.zeros_like(c)
    out[1:-1] = (c[2:] - c[:-2]) / 2.0
    if "E10" not in err:
        out /= dz
    return out


def dipole_rate(z, ca, cb, dz, err: set) -> float:
    pa, pb = errset._convert(ca, dz, err, "a"), errset._convert(cb, dz, err, "b")
    m = _quadrature(pa * z * pb, z, dz, err)
    r = m * m
    if "E6" in err:
        r *= R_TOY
    return r


def derivative_rate(z, ca, cb, dz, z0: float, err: set) -> float:
    pa, pb = errset._convert(ca, dz, err, "a"), errset._convert(cb, dz, err, "b")
    i0 = int(round(z0 / dz)) - 1
    da, db = _derivative(pa, dz, err), _derivative(pb, dz, err)
    w = pa[i0] * db[i0] - pb[i0] * da[i0]
    r = float(w * w)
    if "E6" in err:
        r *= R_TOY
    return r


# ------------------------------------------------- B1n numerical interface audit --
def audit_B1n(case: str, leg: str, consumer: str, errors) -> dict:
    err = set(errors)
    pair = (1, 3) if "E9" in err else (1, 2)
    norms = {"a": [], "b": []}
    quad, deriv = [], []
    for dz in DZ_LADDER:
        swap = ("E8" in err) and abs(dz - 0.10) < 1e-12
        z, ca, cb = errset.states_for(leg, dz, pair, swap)
        pa, pb = errset._convert(ca, dz, err, "a"), errset._convert(cb, dz, err, "b")
        norms["a"].append(float(np.sum(pa * pa) * dz))
        norms["b"].append(float(np.sum(pb * pb) * dz))
        if consumer == "int":
            quad.append(_quadrature(np.ones_like(z), z, dz, err) / float(z[-1] - z[0]))
        else:
            d = _derivative(np.asarray(z, dtype=float), dz, err)
            deriv.append(float(np.max(np.abs(d[1:-1] - 1.0))))
    findings, classes = [], set()
    for side in ("a", "b"):
        n_arr = np.asarray(norms[side])
        if np.max(np.abs(n_arr - 1.0)) > TAU_N:
            s = float(np.polyfit(np.log(np.asarray(DZ_LADDER)), np.log(n_arr), 1)[0])
            cls = "MEASURE" if abs(s) > SLOPE_TOL[leg] else "CONSTANT"
            classes.add(cls)
            findings.append({"probe": "entry-norm", "side": side, "N": n_arr.tolist(),
                             "slope": s, "class": cls})
    if consumer == "int":
        q = np.asarray(quad)
        if np.max(np.abs(q - 1.0)) > TAU_N:
            classes.add("MEASURE")
            findings.append({"probe": "quadrature-weights", "ratio_to_length": q.tolist(),
                             "class": "MEASURE"})
    else:
        dd = np.asarray(deriv)
        if np.max(dd) > TAU_N:
            classes.add("MEASURE")
            findings.append({"probe": "derivative-operator", "max_dev_from_1": dd.tolist(),
                             "class": "MEASURE"})
    k_ir = 2 if "E1" in err else (1 if "E7" in err else 0)
    norm_f = [f for f in findings if f["probe"] == "entry-norm"]
    raw_sides = [f for f in norm_f if f["class"] == "MEASURE" and f["slope"] > 0]   # N = h: raw side
    if findings:
        localized = bool(len(findings) == len(norm_f) and k_ir > 0 and len(raw_sides) == k_ir)
        status = "INCONSISTENT"
    else:
        localized, status = None, "COMPATIBLE"
    return {"status": status, "classes": sorted(classes), "findings": findings,
            "localized": localized, "n": len(raw_sides) if findings else 0,
            "reason": ("; ".join(f"{f['probe']}[{f.get('side','')}]->{f['class']}" for f in findings)
                       if findings else "entry norms, quadrature weights / derivative operator all pass"),
            "audit": {"norms": norms, "quadrature_ratio_to_length": quad,
                      "derivative_max_dev_from_1": deriv}}


# -------------------------------------------------------------------- P split, P' --
def method_P_split(rec: dict) -> dict:
    p = errset.method_P(rec)
    return {**p, "interface": ("MISSING-DECLARATION" if rec["k_IR"] > 0 else "DECLARED"),
            "interface_n": rec["k_IR"], "numeric": p["status"]}


def asymptotic_guards(rec: dict) -> dict:
    x = np.log(np.asarray(rec["dz"], dtype=float))
    y = np.log(np.abs(np.asarray(rec["Q"], dtype=float)))
    if not np.all(np.isfinite(y)):
        return {"pass": False, "reason": "non-finite log|Q|"}
    coef = np.polyfit(x, y, 1)
    s = float(coef[0])
    resid = y - np.polyval(coef, x)
    rms = float(np.sqrt(np.mean(resid ** 2)))
    local = [float(v) for v in (np.diff(y) / np.diff(x))]
    s_drop = float(np.polyfit(x[1:], y[1:], 1)[0])       # ladder is coarse -> fine; drop index 0
    tol = SLOPE_TOL[rec["leg"]]
    ok_local = bool(max(abs(v - s) for v in local) <= 2.0 * tol)
    ok_drop = bool(abs(s_drop - s) <= tol)
    ok_rms = bool(rms <= RMS_TOL)
    return {"slope": s, "local_slopes": local, "slope_drop_coarsest": s_drop, "rms_residual": rms,
            "ok_local": ok_local, "ok_drop": ok_drop, "ok_rms": ok_rms,
            "pass": bool(ok_local and ok_drop and ok_rms)}


def method_Pp(rec: dict, q_ref12: float) -> dict:
    leg = rec["leg"]
    base = {"interface": ("MISSING-DECLARATION" if rec["k_IR"] > 0 else "DECLARED"),
            "interface_n": rec["k_IR"]}
    if abs(rec["Q_ref"]) <= ZERO_FLOOR * abs(q_ref12):
        return {**base, "status": "UNDECIDABLE", "classes": [], "n": None,
                "reason": "ZERO-REF (dimensionless floor)", "numeric": "UNDECIDABLE"}
    if min(rec["identity_overlap"]) < ID_TOL:
        return {**base, "status": "UNDECIDABLE", "classes": [], "n": None, "reason": "STATE-SET",
                "min_overlap": min(rec["identity_overlap"]), "numeric": "UNDECIDABLE"}
    s = errset._slope(rec["dz"], rec["Q"])
    k = errset._exponent(s, SLOPE_TOL[leg])
    if k is None:
        return {**base, "status": "UNDECIDABLE", "classes": [], "n": None, "slope": s,
                "reason": "NON-ASYMPTOTIC", "numeric": "UNDECIDABLE"}
    g = asymptotic_guards(rec)
    if not g["pass"]:
        return {**base, "status": "UNDECIDABLE", "classes": [], "n": None, "slope": s,
                "reason": "NON-ASYMPTOTIC-LOCAL", "guards": g, "numeric": "UNDECIDABLE"}
    p = errset.method_P(rec)              # S0-S2 already passed identically; S3-S4 verbatim
    return {**base, **p, "guards": g, "numeric": p["status"]}


# ---------------------------------------------------------------- two-layer scoring --
DETECTED_L1 = ("ANOMALY-DETECTED", "INTERFACE-DETECTED")


def score_layered(case_row, method: str, out: dict, truth: dict) -> dict:
    case, leg, consumer, errors, k_true, _, _ = case_row
    err = set(errors)
    special = bool(err & {"E8", "E9"})
    st = out["status"]
    real = set(truth["realized_primitives"])
    real_cons = real & CONSUMER_ERRORS
    if method in INTERFACE:
        visible = bool(real & VISIBLE[method])
        flagged = st == "INCONSISTENT"
        if flagged and visible:
            l1 = "INTERFACE-DETECTED"
        elif flagged:
            l1 = "FALSE-ALARM"
        elif visible:
            l1 = "MISSED"
        elif real_cons:
            l1 = "MISSED-BLIND"
        else:
            l1 = "CORRECT-NEG"
        truth_cls, k_ref = set(truth["C_cmp"]), None
    else:
        if st == "UNDEFINED":
            l1 = "UNDEFINED"
        elif st == "UNDECIDABLE":
            l1 = "UNDECIDABLE-appropriate" if special else "UNDECIDABLE-unhelpful"
        elif case in CANCEL_CASES:
            l1 = "MISSED-BY-CONSTRUCTION" if st == "COMPATIBLE" else "FALSE-ALARM-ON-CANCELLED"
        elif not truth["output_error"]:
            l1 = "CORRECT-NEG" if st == "COMPATIBLE" else "FALSE-ALARM"
        else:
            l1 = "ANOMALY-DETECTED" if st == "INCONSISTENT" else "MISSED"
        truth_cls = set(truth["C_out"]) if k_true is not None else set(truth["C_cmp"])
        k_ref = k_true if k_true is not None else truth["k_cmp"]
    l2 = None
    if l1 in DETECTED_L1:
        c_out = set(out.get("classes", []))
        if not c_out or c_out == {"UNCLASSIFIED"}:
            l2 = "CLASS-NONE"
        elif c_out != truth_cls:
            l2 = "CLASS-WRONG"
        elif k_ref is not None and out.get("n") is not None and out["n"] != k_ref:
            l2 = "CLASS-WRONG"                                   # exponent misread
        elif out.get("single_cause") and len(real & set(MEASURE_PRIMS)) > 1:
            l2 = "CLASS-WRONG"
        else:
            l2 = "CLASS-CORRECT"
    l3 = None
    if l1 in DETECTED_L1 and method in ("P", "Pp", "B1", "B1n") and out.get("localized") is not None:
        l3 = "LOC-CORRECT" if bool(out["localized"]) == bool(truth["localizable"]) else "LOC-WRONG"
    if l1 in DETECTED_L1:
        composite = "DETECTED" if l2 == "CLASS-CORRECT" else "MISCLASSIFIED"
    elif l1 in ("MISSED-BLIND",):
        composite = "MISSED"
    elif l1 == "FALSE-ALARM-ON-CANCELLED":
        composite = "FALSE-ALARM"
    else:
        composite = l1
    return {"L1": l1, "L2": l2, "L3": l3, "composite": composite}


# ------------------------------------------------- expectations (fixed before the run) --
def _base_pred(case: str, method: str) -> str:
    return errset.PRED[case][errset.METHODS.index(method)]


PRED_METHODS = {}
for _row in CASES:
    _case = _row[0]
    p = _base_pred(_case, "P")
    b2 = _base_pred(_case, "B2")
    b1 = _base_pred(_case, "B1")
    if _case in ("M01", "M02"):          # audited truth: E5 never acts -> {MEASURE}
        b2, b1 = "DETECTED", "DETECTED"
    if _case == "M07":
        b1 = "MISSED"
    b1n = {"K03": "MISSED", "M03": "MISCLASSIFIED"}.get(_case)
    if b1n is None:
        b1n = "CORRECT-NEG" if _case in ("N01", "N02", "N03", "N04", "Z01", "T01") else "DETECTED"
    pp = p if _case not in B_LEG_CASES else (p if p.endswith("?") else p + "?")
    PRED_METHODS[_case] = {"B1": b1, "B1n": b1n, "B2": b2, "B3": _base_pred(_case, "B3"), "P": p, "Pp": pp}


def run_all_cases() -> dict:
    rows = []
    ref12 = {}
    for case_row in CASES:
        case, leg, consumer, errors, k_true, kappa_reg, note = case_row
        key = (leg, consumer)
        if key not in ref12:
            ref12[key] = errset.reference(leg, consumer, (1, 2))
        rec = errset.run_case(case, leg, consumer, errors)           # raw quantities of the base module (bitwise)
        truth = audited_truth(case_row)
        outs = {"B1": errset.method_B1(rec), "B1n": audit_B1n(case, leg, consumer, errors),
                "B2": errset.method_B2(rec), "B3": errset.method_B3(rec),
                "P": method_P_split(rec), "Pp": method_Pp(rec, ref12[key])}
        sc = {m: score_layered(case_row, m, outs[m], truth) for m in METHODS}
        agree = {m: errset.pred_agrees(PRED_METHODS[case][m], sc[m]["composite"]) for m in METHODS}
        base_scores = {m: errset.score(case_row, m, outs[m]) for m in errset.METHODS}
        rows.append({**rec, "note": note, "truth": truth, "outputs": outs, "scores": sc,
                     "scores_B1_B2_B3_P_on_registered_truth": base_scores,
                     "predicted": PRED_METHODS[case], "agreement": agree,
                     "q_ref_12": ref12[key]})
    counts = {m: {} for m in METHODS}
    layered = {m: {"L1": {}, "L2": {}, "L3": {}} for m in METHODS}
    agree_counts = {m: {} for m in METHODS}
    for r in rows:
        for m in METHODS:
            s = r["scores"][m]
            counts[m][s["composite"]] = counts[m].get(s["composite"], 0) + 1
            for layer in ("L1", "L2", "L3"):
                if s[layer] is not None:
                    layered[m][layer][s[layer]] = layered[m][layer].get(s[layer], 0) + 1
            a = r["agreement"][m]
            agree_counts[m][a] = agree_counts[m].get(a, 0) + 1
    summary = {}
    for m in METHODS:
        l1 = layered[m]["L1"]
        det = l1.get("ANOMALY-DETECTED", 0) + l1.get("INTERFACE-DETECTED", 0)
        summary[m] = {"detected_L1": det, "class_correct_L2": layered[m]["L2"].get("CLASS-CORRECT", 0),
                      "class_none_L2": layered[m]["L2"].get("CLASS-NONE", 0),
                      "class_wrong_L2": layered[m]["L2"].get("CLASS-WRONG", 0),
                      "false_alarm": l1.get("FALSE-ALARM", 0) + l1.get("FALSE-ALARM-ON-CANCELLED", 0),
                      "missed": l1.get("MISSED", 0) + l1.get("MISSED-BLIND", 0),
                      "missed_by_construction": l1.get("MISSED-BY-CONSTRUCTION", 0),
                      "undecidable_appropriate": l1.get("UNDECIDABLE-appropriate", 0),
                      "undecidable_unhelpful": l1.get("UNDECIDABLE-unhelpful", 0),
                      "undefined": l1.get("UNDEFINED", 0), "correct_neg": l1.get("CORRECT-NEG", 0),
                      "loc_correct_L3": layered[m]["L3"].get("LOC-CORRECT", 0),
                      "loc_wrong_L3": layered[m]["L3"].get("LOC-WRONG", 0)}
    audit_table = [{"case": r["case"], **{k: r["truth"][k] for k in
                    ("registered", "audited", "kappa_changed", "realized_primitives", "C_out", "C_cmp",
                     "k_cmp", "provenance", "provenance_note")}} for r in rows]
    return {"rows": rows, "counts_composite": counts, "layered_counts": layered, "summary": summary,
            "agreement_counts": agree_counts, "truth_audit_table": audit_table,
            "provenance_counts": {k: sum(1 for c in PROVENANCE.values() if c == k)
                                  for k in ("regression-control", "primitive-new", "combination-new", "guard-new")},
            "thresholds": {"slope_tol": SLOPE_TOL, "gate_band": GATE_BAND, "factor_tol": FACTOR_TOL,
                           "zero_floor_relative": ZERO_FLOOR, "identity_tol": ID_TOL,
                           "probe_tol": TAU_N, "rms_tol": RMS_TOL,
                           "provenance_of_thresholds": {"slope_tol/gate_band/factor_tol": "same values as controlled_validation.py",
                                                        "zero_floor/identity_tol": "fixed with the 27-case set before the runs"}}}


# ---------------------------------------------------------------- program: results/mixed_error_set/
OUT = HERE.parent / "results" / "mixed_error_set"
def identity() -> dict:
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version, "numpy": np.__version__, "platform": platform.platform()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res = run_all_cases()
    res["identity"] = identity()
    res["setup"] = "27 designed cases, six methods; expectations fixed before the run"
    (OUT / "mixed_error_set.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    with (OUT / "mixed_error_set_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case", "provenance", "leg", "consumer", "errors", "realized", "k", "kappa_registered",
                    "kappa_audited", "C_out", "C_cmp", "P_interface", "P_numeric", "Pp_guards_pass",
                    *[f"{m}_status" for m in METHODS], *[f"{m}_L1" for m in METHODS],
                    *[f"{m}_L2" for m in METHODS], *[f"{m}_L3" for m in METHODS],
                    *[f"{m}_composite" for m in METHODS], *[f"{m}_pred" for m in METHODS],
                    *[f"{m}_agree" for m in METHODS]])
        for r in res["rows"]:
            t = r["truth"]
            g = r["outputs"]["Pp"].get("guards")
            w.writerow([r["case"], t["provenance"], r["leg"], r["consumer"], "+".join(r["errors"]),
                        "+".join(t["realized_primitives"]), t["audited"]["k"], t["registered"]["kappa"],
                        t["audited"]["kappa"], "+".join(t["C_out"]), "+".join(t["C_cmp"]),
                        r["outputs"]["P"]["interface"], r["outputs"]["P"]["numeric"],
                        (None if g is None else g["pass"]),
                        *[r["outputs"][m]["status"] for m in METHODS],
                        *[r["scores"][m]["L1"] for m in METHODS],
                        *[r["scores"][m]["L2"] for m in METHODS],
                        *[r["scores"][m]["L3"] for m in METHODS],
                        *[r["scores"][m]["composite"] for m in METHODS],
                        *[r["predicted"][m] for m in METHODS],
                        *[r["agreement"][m] for m in METHODS]])
    print("=== mixed-error set (27 cases, six methods) ===")
    print("truth audit (kappa changed):")
    for t in res["truth_audit_table"]:
        if t["kappa_changed"]:
            print(f"  {t['case']}: registered kappa {t['registered']['kappa']} -> audited {t['audited']['kappa']}; "
                  f"realized {t['realized_primitives']}; C_out {t['C_out']}; C_cmp {t['C_cmp']}")
    print("\ncomposite counts per method:")
    for m in METHODS:
        print(f"  {m}: {res['counts_composite'][m]}")
    print("layered summary per method:")
    for m in METHODS:
        print(f"  {m}: {res['summary'][m]}")
    print("agreement with the expectations fixed before the run:")
    for m in METHODS:
        print(f"  {m}: {res['agreement_counts'][m]}")
    print("Pp guard trips:", [r["case"] for r in res["rows"] if r["outputs"]["Pp"].get("reason") == "NON-ASYMPTOTIC-LOCAL"])


if __name__ == "__main__":
    main()
