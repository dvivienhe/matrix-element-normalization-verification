"""The 27-case mixed-error set: error primitives, the two consumers, the cases and the methods B1 / B2 / B3 / P
with their scoring. Pure NumPy; imported by mixed_error_methods.py and evidence_rows.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from toy_model import (DZ_LADDER, DZ_REF_B, L_WELL_NM, M_PAR_TOY, M_Z_TOY,  # noqa: E402
                          Z0_BARDEEN_A, Z0_BARDEEN_B, fd_well_solve, grid_a,
                          manufactured_dpsi, manufactured_psi, solver_contract)

R_TOY = M_Z_TOY / M_PAR_TOY                    # 4.5454... (E6 constant factor)
UNIT_MIX = 0.01                                # E5 constant factor on the rate
SLOPE_TOL = {"A": 0.05, "B": 0.10}             # same tolerances as the controlled validation
GATE_BAND = 0.02
FACTOR_TOL = {"A": 0.05, "B": 0.10}
ZERO_FLOOR = 1e-8
ID_TOL = 0.99
N_EXPR = 2                                     # both consumers carry two envelope factors

# ------------------------------------------------------------------ cases --
# (case, leg, consumer, errors, k_true, kappa_true, class_note)
CASES = [
    ("N01", "A", "int", (), 0, 1.0, "no error"),
    ("N02", "B", "int", (), 0, 1.0, "no error"),
    ("N03", "A", "der", (), 0, 1.0, "no error"),
    ("N04", "B", "der", (), 0, 1.0, "no error; interface-node false-alarm risk"),
    ("S01", "A", "int", ("E1",), 2, 1.0, "measure mismatch (boundary)"),
    ("S02", "B", "int", ("E1",), 2, 1.0, "measure mismatch (boundary)"),
    ("S03", "A", "int", ("E2",), -2, 1.0, "double conversion"),
    ("S04", "A", "int", ("E3",), -2, 1.0, "downstream missing weight"),
    ("S05", "B", "int", ("E3",), -2, 1.0, "downstream missing weight"),
    ("S06", "A", "int", ("E4",), 2, 1.0, "downstream duplicate weight"),
    ("S07", "A", "der", ("E10",), 2, 1.0, "downstream derivative missing 1/dz"),
    ("K01", "A", "int", ("E5",), 0, UNIT_MIX, "constant (unit)"),
    ("K02", "B", "int", ("E5",), 0, UNIT_MIX, "constant (unit)"),
    ("K03", "A", "int", ("E6",), 0, R_TOY, "constant (prefactor)"),
    ("P01", "A", "int", ("E7",), 1, 1.0, "one-sided measure mismatch"),
    ("M01", "A", "int", ("E1", "E5"), 2, UNIT_MIX, "mixed same-direction"),
    ("M02", "B", "int", ("E1", "E5"), 2, UNIT_MIX, "mixed same-direction"),
    ("M03", "A", "int", ("E1", "E6"), 2, R_TOY, "mixed same-direction"),
    ("M04", "A", "int", ("E7", "E5"), 1, UNIT_MIX, "mixed same-direction (one-sided)"),
    ("M05", "A", "int", ("E1", "E3"), 0, 1.0, "cancelling mixed (boundary-visible)"),
    ("M06", "B", "int", ("E1", "E3"), 0, 1.0, "cancelling mixed (boundary-visible)"),
    ("M07", "A", "int", ("E2", "E4"), 0, 1.0, "cancelling mixed (invisible to all)"),
    ("M08", "A", "der", ("E1", "E10"), 4, 1.0, "two errors stacking to h^4"),
    ("Z01", "A", "int", ("E9",), None, None, "zero reference"),
    ("Z02", "A", "int", ("E1", "E9"), None, None, "zero reference + mismatch"),
    ("T01", "A", "int", ("E8",), None, None, "state-set change"),
    ("T02", "A", "int", ("E1", "E8"), None, None, "state-set change + mismatch"),
]
CANCEL_CASES = {"M05", "M06", "M07"}
CONSUMER_ERRORS = {"E1", "E2", "E3", "E4", "E5", "E6", "E7", "E10"}

# expectation table (fixed before the run); "?" = declared unpredictable / low confidence
PRED = {
    "N01": ("CORRECT-NEG", "CORRECT-NEG", "CORRECT-NEG", "CORRECT-NEG"),
    "N02": ("CORRECT-NEG", "CORRECT-NEG", "CORRECT-NEG", "CORRECT-NEG"),
    "N03": ("CORRECT-NEG", "CORRECT-NEG", "CORRECT-NEG", "CORRECT-NEG"),
    "N04": ("CORRECT-NEG", "CORRECT-NEG?", "FALSE-ALARM?", "FALSE-ALARM?"),
    "S01": ("DETECTED", "DETECTED", "MISCLASSIFIED", "DETECTED"),
    "S02": ("DETECTED", "DETECTED", "MISCLASSIFIED", "DETECTED"),
    "S03": ("MISSED", "DETECTED", "MISCLASSIFIED", "DETECTED"),
    "S04": ("MISSED", "DETECTED", "MISCLASSIFIED", "DETECTED"),
    "S05": ("MISSED", "DETECTED", "MISCLASSIFIED", "DETECTED"),
    "S06": ("MISSED", "DETECTED", "MISCLASSIFIED", "DETECTED"),
    "S07": ("MISSED", "DETECTED", "MISCLASSIFIED", "DETECTED"),
    "K01": ("MISSED", "MISSED", "MISCLASSIFIED", "DETECTED"),
    "K02": ("MISSED", "MISSED", "MISCLASSIFIED", "DETECTED"),
    "K03": ("MISSED", "MISSED", "MISCLASSIFIED", "DETECTED"),
    "P01": ("DETECTED", "DETECTED", "MISCLASSIFIED", "DETECTED"),
    "M01": ("MISCLASSIFIED", "MISCLASSIFIED", "MISCLASSIFIED", "DETECTED"),
    "M02": ("MISCLASSIFIED", "MISCLASSIFIED", "MISCLASSIFIED", "DETECTED"),
    "M03": ("MISCLASSIFIED", "MISCLASSIFIED", "MISCLASSIFIED", "DETECTED"),
    "M04": ("MISCLASSIFIED", "MISCLASSIFIED", "MISCLASSIFIED", "DETECTED"),
    "M05": ("DETECTED", "MISSED-BY-CONSTRUCTION", "MISSED-BY-CONSTRUCTION", "MISSED-BY-CONSTRUCTION"),
    "M06": ("DETECTED", "MISSED-BY-CONSTRUCTION", "MISSED-BY-CONSTRUCTION", "MISSED-BY-CONSTRUCTION"),
    "M07": ("MISSED", "MISSED-BY-CONSTRUCTION", "MISSED-BY-CONSTRUCTION", "MISSED-BY-CONSTRUCTION"),
    "M08": ("DETECTED", "MISCLASSIFIED", "MISCLASSIFIED", "DETECTED"),
    "Z01": ("CORRECT-NEG", "?", "UNDEFINED", "UNDECIDABLE-appropriate"),
    "Z02": ("DETECTED", "?", "UNDEFINED", "UNDECIDABLE-appropriate"),
    "T01": ("CORRECT-NEG", "?", "CORRECT-NEG", "UNDECIDABLE-appropriate"),
    "T02": ("DETECTED", "?", "MISCLASSIFIED", "UNDECIDABLE-appropriate"),
}
METHODS = ("B1", "B2", "B3", "P")


# ----------------------------------------------------------------- states --
def leg_a_state(n: int, dz: float):
    z = grid_a(dz)
    return z, solver_contract(manufactured_psi(n, z))


def leg_b_states(dz: float):
    z, c1, c2 = fd_well_solve(dz)
    return z, c1, c2


def states_for(leg: str, dz: float, pair: tuple[int, int], swap_rung: bool):
    """Delivered (Euclidean-normalized) states for the pair at one rung.
    swap_rung=True replaces state b by state 3 (E8) -- Leg-A only."""
    a, b = pair
    if leg == "A":
        z, ca = leg_a_state(a, dz)
        _, cb = leg_a_state(3 if swap_rung else b, dz)
        return z, ca, cb
    if swap_rung:
        raise ValueError("E8 is defined on Leg-A only")
    z, c1, c2 = leg_b_states(dz)
    return z, c1, c2


# -------------------------------------------------------------- consumers --
def _convert(c: np.ndarray, dz: float, err: set, side: str) -> np.ndarray:
    """Producer->consumer conversion with the injected boundary errors."""
    if "E1" in err:
        return c                                   # both raw
    if "E7" in err and side == "b":
        return c                                   # one-sided raw
    if "E2" in err:
        return c / dz                              # applied twice
    if "E5" in err:
        return c / np.sqrt(dz * 10.0)              # dz taken in Angstrom
    return c / np.sqrt(dz)


def dipole_rate(z, ca, cb, dz, err: set) -> float:
    pa, pb = _convert(ca, dz, err, "a"), _convert(cb, dz, err, "b")
    integrand = pa * z * pb
    m = float(np.sum(integrand)) if "E3" in err else float(np.trapezoid(integrand, z))
    if "E4" in err:
        m *= dz
    r = m * m
    if "E6" in err:
        r *= R_TOY
    return r


def derivative_rate(z, ca, cb, dz, z0: float, err: set) -> float:
    pa, pb = _convert(ca, dz, err, "a"), _convert(cb, dz, err, "b")
    i0 = int(round(z0 / dz)) - 1

    def d(c):
        out = np.zeros_like(c)
        out[1:-1] = (c[2:] - c[:-2]) / 2.0
        if "E10" not in err:
            out /= dz
        return out

    da, db = d(pa), d(pb)
    w = pa[i0] * db[i0] - pb[i0] * da[i0]
    r = float(w * w)
    if "E6" in err:
        r *= R_TOY
    return r


# ------------------------------------------------------------- references --
def reference(leg: str, consumer: str, pair: tuple[int, int]) -> float:
    if leg == "A":
        a, b = pair
        if consumer == "int":
            if (a, b) == (1, 2):
                return float((-16.0 * L_WELL_NM / (9.0 * np.pi ** 2)) ** 2)
            if (a, b) == (1, 3):
                return 0.0                          # parity: exact zero
            raise ValueError(pair)
        z0 = np.array([Z0_BARDEEN_A])
        w = (manufactured_psi(a, z0)[0] * manufactured_dpsi(b, z0)[0]
             - manufactured_psi(b, z0)[0] * manufactured_dpsi(a, z0)[0])
        return float(w * w)
    z, c1, c2 = leg_b_states(DZ_REF_B)
    if consumer == "int":
        return dipole_rate(z, c1, c2, DZ_REF_B, set())
    return derivative_rate(z, c1, c2, DZ_REF_B, Z0_BARDEEN_B, set())


# --------------------------------------------------------------- one case --
def run_case(case: str, leg: str, consumer: str, errors: tuple) -> dict:
    err = set(errors)
    pair = (1, 3) if "E9" in err else (1, 2)
    z0 = Z0_BARDEEN_A if leg == "A" else Z0_BARDEEN_B
    q, identity, grids = [], [], []
    for dz in DZ_LADDER:
        swap = ("E8" in err) and abs(dz - 0.10) < 1e-12
        z, ca, cb = states_for(leg, dz, pair, swap)
        if consumer == "int":
            q.append(dipole_rate(z, ca, cb, dz, err))
        else:
            q.append(derivative_rate(z, ca, cb, dz, z0, err))
        grids.append((z, ca / np.sqrt(dz), cb / np.sqrt(dz)))
    # state-identity overlaps between consecutive rungs (producer side, error-free conversion)
    for (zc, ac, bc), (zf, af, bf) in zip(grids[:-1], grids[1:]):
        dzc = zc[1] - zc[0]
        oa = abs(float(np.sum(ac * np.interp(zc, zf, af)) * dzc))
        ob = abs(float(np.sum(bc * np.interp(zc, zf, bf)) * dzc))
        identity.append(min(oa, ob))
    qref = reference(leg, consumer, pair)
    k_ir = 2 if "E1" in err else (1 if "E7" in err else 0)
    return {"case": case, "leg": leg, "consumer": consumer, "errors": sorted(err),
            "pair": list(pair), "dz": list(DZ_LADDER), "Q": q, "Q_ref": qref,
            "identity_overlap": identity, "k_IR": k_ir}


# ---------------------------------------------------------------- methods --
def _slope(dz, q):
    x = np.log(np.asarray(dz))
    y = np.log(np.abs(np.asarray(q)))
    if not np.all(np.isfinite(y)):
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def _exponent(s: float, tol: float):
    if not np.isfinite(s):
        return None
    if abs(s) <= tol:
        return 0
    r = int(round(s))
    if r != 0 and abs(s - r) <= tol:
        return r
    return None


def method_B1(rec: dict) -> dict:
    if rec["k_IR"] > 0:
        return {"status": "INCONSISTENT", "classes": ["MEASURE"], "n": rec["k_IR"],
                "localized": True, "reason": "consumer declares no conversion for "
                                             f"{rec['k_IR']} envelope factor(s)"}
    return {"status": "COMPATIBLE", "classes": [], "n": 0, "localized": None,
            "reason": "exit norm 1 and all conversions declared"}


def method_B2(rec: dict) -> dict:
    s = _slope(rec["dz"], rec["Q"])
    k = _exponent(s, SLOPE_TOL[rec["leg"]])
    if k is None:
        return {"status": "UNDECIDABLE", "classes": [], "n": None, "slope": s,
                "reason": "NON-INTEGER"}
    if k == 0:
        return {"status": "COMPATIBLE", "classes": [], "n": 0, "slope": s, "reason": "|s|<=tol"}
    return {"status": "INCONSISTENT", "classes": ["MEASURE"], "n": k, "slope": s,
            "single_cause": True, "reason": f"interpreted as {k} envelope-factor mismatch"}


def method_B3(rec: dict) -> dict:
    if abs(rec["Q_ref"]) <= ZERO_FLOOR:
        return {"status": "UNDEFINED", "classes": [], "n": None, "reason": "zero reference"}
    delta = abs(rec["Q"][-1] / rec["Q_ref"] - 1.0)
    if delta <= GATE_BAND:
        return {"status": "COMPATIBLE", "classes": [], "n": 0, "delta": delta, "reason": "delta<=2%"}
    return {"status": "INCONSISTENT", "classes": ["UNCLASSIFIED"], "n": None, "delta": delta,
            "reason": "finest-rung deviation exceeds band"}


def method_P(rec: dict) -> dict:
    leg = rec["leg"]
    if abs(rec["Q_ref"]) <= ZERO_FLOOR:
        return {"status": "UNDECIDABLE", "classes": [], "n": None, "reason": "ZERO-REF"}
    if min(rec["identity_overlap"]) < ID_TOL:
        return {"status": "UNDECIDABLE", "classes": [], "n": None, "reason": "STATE-SET",
                "min_overlap": min(rec["identity_overlap"])}
    s = _slope(rec["dz"], rec["Q"])
    k = _exponent(s, SLOPE_TOL[leg])
    if k is None:
        return {"status": "UNDECIDABLE", "classes": [], "n": None, "slope": s,
                "reason": "NON-ASYMPTOTIC"}
    delta = abs(rec["Q"][-1] / rec["Q_ref"] - 1.0)
    kappa = float(np.median([q / (rec["Q_ref"] * d ** k) for q, d in zip(rec["Q"], rec["dz"])]))
    const = abs(kappa - 1.0) > FACTOR_TOL[leg]
    out = {"slope": s, "delta": delta, "kappa_hat": kappa, "n": k, "k_IR": rec["k_IR"]}
    if k == 0:
        if delta <= GATE_BAND:
            note = ("boundary declares missing conversion but numbers are compatible: "
                    "suspected compensation" if rec["k_IR"] > 0 else "modules may compensate")
            return {**out, "status": "COMPATIBLE", "classes": [], "localized": None, "reason": note}
        return {**out, "status": "INCONSISTENT", "classes": ["CONSTANT"], "localized": None,
                "reason": "zero slope, absolute deviation exceeds band"}
    classes = ["MEASURE"] + (["CONSTANT"] if const else [])
    if k == rec["k_IR"]:
        return {**out, "status": "INCONSISTENT", "classes": classes, "localized": True,
                "reason": f"exponent {k} matches the boundary record"}
    return {**out, "status": "INCONSISTENT", "classes": classes, "localized": False,
            "boundary_external_exponent": k - rec["k_IR"],
            "reason": f"exponent {k} != boundary prediction {rec['k_IR']}: candidates = downstream "
                      "weight / double conversion / multiple errors"}


# ---------------------------------------------------------------- scoring --
def truth_classes(case_row) -> set:
    _, leg, _, errors, k_true, kappa_true, _ = case_row
    if k_true is None:
        return set()
    cls = set()
    if k_true != 0:
        cls.add("MEASURE")
    if abs(kappa_true - 1.0) > FACTOR_TOL[leg]:
        cls.add("CONSTANT")
    return cls


def score(case_row, method: str, out: dict) -> str:
    case, _, _, errors, k_true, _, _ = case_row
    err = set(errors)
    has_consumer_error = bool(err & CONSUMER_ERRORS)
    special = bool(err & {"E8", "E9"})
    st = out["status"]
    if st == "UNDEFINED":
        return "UNDEFINED"
    if st == "UNDECIDABLE":
        return "UNDECIDABLE-appropriate" if special else "UNDECIDABLE-unhelpful"
    if not has_consumer_error:
        return "CORRECT-NEG" if st == "COMPATIBLE" else "FALSE-ALARM"
    if st == "COMPATIBLE":
        return "MISSED-BY-CONSTRUCTION" if case in CANCEL_CASES else "MISSED"
    # INCONSISTENT
    if case in CANCEL_CASES and method != "B1":
        return "MISCLASSIFIED"
    c_true, c_out = truth_classes(case_row), set(out.get("classes", []))
    if c_out != c_true:
        return "MISCLASSIFIED"
    n_measure_errors = len(err & {"E1", "E2", "E3", "E4", "E7", "E10"})
    if out.get("single_cause") and n_measure_errors > 1:
        return "MISCLASSIFIED"
    return "DETECTED"


def pred_agrees(pred: str, obs: str) -> str:
    """'agree' / 'disagree' / 'unpredicted' ('?' cells of the expectation table)."""
    if pred == "?":
        return "unpredicted"
    if pred.endswith("?"):
        return "agree(low-confidence)" if obs == pred[:-1] else "disagree(low-confidence)"
    return "agree" if obs == pred else "disagree"


def run_all() -> dict:
    rows = []
    for row in CASES:
        case, leg, consumer, errors, k_true, kappa_true, note = row
        rec = run_case(case, leg, consumer, errors)
        outs = {"B1": method_B1(rec), "B2": method_B2(rec), "B3": method_B3(rec), "P": method_P(rec)}
        sc = {m: score(row, m, outs[m]) for m in METHODS}
        agree = {m: pred_agrees(PRED[case][i], sc[m]) for i, m in enumerate(METHODS)}
        rows.append({**rec, "k_true": k_true, "kappa_true": kappa_true, "note": note,
                     "truth_classes": sorted(truth_classes(row)), "outputs": outs,
                     "scores": sc, "predicted": dict(zip(METHODS, PRED[case])), "agreement": agree})
    counts = {m: {} for m in METHODS}
    for r in rows:
        for m in METHODS:
            counts[m][r["scores"][m]] = counts[m].get(r["scores"][m], 0) + 1
    agree_counts = {m: {} for m in METHODS}
    for r in rows:
        for m in METHODS:
            a = r["agreement"][m]
            agree_counts[m][a] = agree_counts[m].get(a, 0) + 1
    return {"rows": rows, "counts": counts, "agreement_counts": agree_counts,
            "thresholds": {"slope_tol": SLOPE_TOL, "gate_band": GATE_BAND, "factor_tol": FACTOR_TOL,
                           "zero_floor": ZERO_FLOOR, "identity_tol": ID_TOL, "n_expr": N_EXPR}}
