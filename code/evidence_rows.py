"""Evidence rows of the 27-case mixed-error set (Table 4, upper half): probe readings, level and slope rows,
methods OR / CW / OW with interval projections (HiGHS, cross-checked against the closed form) and scoring.
Run as a program it writes results/evidence_rows/; analyse_states.py imports it for the Solcore states."""
from __future__ import annotations

import csv
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import mixed_error_set as errset       # noqa: E402
import mixed_error_methods as methods   # noqa: E402

DZ_LADDER = errset.DZ_LADDER
CASES = errset.CASES
H_F = DZ_LADDER[-1]                     # 0.05 nm; h = dz / (1 nm)
TAU_N = methods.TAU_N                       # 1e-6 probe tolerance
EPS_PROBE_LEVEL = 1.01e-6               # tau_n log bound + 1 % margin
EPS_Q_LEVEL = 0.020203                  # -ln(0.98) rounded up (tau_g = 2 %)
SLOPE_TOL = errset.SLOPE_TOL               # tau_s = 0.05 (A) / 0.10 (B)
ZERO_FLOOR, ID_TOL = errset.ZERO_FLOOR, errset.ID_TOL
CF_TOL = 1e-9                           # LP vs closed form
# presolve off: with presolve on, HiGHS reports an unbounded projection of a free parameter that enters only
# the Q band (e.g. a0 under -N) as "infeasible" (status 2) although the same constraint set has just passed
# the feasibility solve; without presolve it returns the correct "unbounded" (status 3).  Machinery only.
HIGHS_OPTS = {"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10, "presolve": False}

SOURCE_OF = {"E1": "ENV", "E2": "ENV", "E5": "ENV", "E7": "ENV",
             "E3": "WGT", "E4": "WGT", "E10": "DER", "E6": "PRE"}
CONDITIONS = ("full", "-N", "-X")        # -X = -W (integral form) / -D (derivative form)
METHODS = ("OR", "CW", "OW")
NORMAL_CASES = ("N01", "N02", "N03", "N04", "Z01", "T01")


def x_name(consumer: str) -> tuple[str, str, str]:
    """(parameter letter, probe key, source label) of the consumer-specific operator probe."""
    return ("w", "W", "WGT") if consumer == "int" else ("d", "D", "DER")


def condition_label(condition: str, consumer: str) -> str:
    return {"full": "full", "-N": "-N", "-X": ("-W" if consumer == "int" else "-D")}[condition]


# ------------------------------------------------------------------ probe readings --
def probe_readings(leg: str, consumer: str, errors) -> dict:
    """R_Na, R_Nb and R_W / R_D on the four rungs, on the same branches as the B1n audit."""
    err = set(errors)
    pair = (1, 3) if "E9" in err else (1, 2)
    na, nb, xr, xmax = [], [], [], []
    for dz in DZ_LADDER:
        swap = ("E8" in err) and abs(dz - 0.10) < 1e-12
        z, ca, cb = errset.states_for(leg, dz, pair, swap)
        pa, pb = errset._convert(ca, dz, err, "a"), errset._convert(cb, dz, err, "b")
        na.append(float(np.sum(pa * pa) * dz))
        nb.append(float(np.sum(pb * pb) * dz))
        if consumer == "int":
            xr.append(methods._quadrature(np.ones_like(z), z, dz, err) / float(z[-1] - z[0]))
        else:
            d = methods._derivative(np.asarray(z, dtype=float), dz, err)
            xr.append(float(np.mean(d[1:-1])))
            xmax.append(float(np.max(np.abs(d[1:-1] - 1.0))))
    return {"Na": na, "Nb": nb, "X": xr, "X_key": x_name(consumer)[1],
            "derivative_max_dev_from_1": (xmax if consumer == "der" else None)}


def _lsq_slope(vals) -> float:
    """Unweighted least-squares slope of ln R against ln h over the four rungs."""
    x = np.log(np.asarray(DZ_LADDER, dtype=float))
    return float(np.polyfit(x, np.log(np.asarray(vals, dtype=float)), 1)[0])


def q_rows_available(rec: dict, q_ref12: float) -> tuple[bool, str]:
    """Guards of the layered scoring: dimensionless zero-reference floor and state identity."""
    if abs(rec["Q_ref"]) <= ZERO_FLOOR * abs(q_ref12):
        return False, "ZERO-REF"
    if min(rec["identity_overlap"]) < ID_TOL:
        return False, "STATE-SET"
    return True, ""


# ------------------------------------------------------------------- evidence rows --
def evidence_rows(rec: dict, probes: dict, q_ok: bool) -> list[dict]:
    leg, consumer = rec["leg"], rec["consumer"]
    x, xk, _ = x_name(consumer)
    ts = SLOPE_TOL[leg]
    rows = []
    for side, key in (("a", "Na"), ("b", "Nb")):
        vals = probes[key]
        rows.append({"name": f"{key}_level", "group": "N", "system": "level", "y": float(np.log(vals[-1])),
                     "eps": EPS_PROBE_LEVEL, "coef": {f"{side}0": 2.0}})
        rows.append({"name": f"{key}_slope", "group": "N", "system": "slope", "y": _lsq_slope(vals),
                     "eps": ts, "coef": {f"{side}1": 2.0}})
    rows.append({"name": f"{xk}_level", "group": "X", "system": "level", "y": float(np.log(probes["X"][-1])),
                 "eps": EPS_PROBE_LEVEL, "coef": {f"{x}0": 1.0}})
    rows.append({"name": f"{xk}_slope", "group": "X", "system": "slope", "y": _lsq_slope(probes["X"]),
                 "eps": ts, "coef": {f"{x}1": 1.0}})
    if q_ok:
        rows.append({"name": "Q_level", "group": "Q", "system": "level",
                     "y": float(np.log(rec["Q"][-1] / rec["Q_ref"])), "eps": EPS_Q_LEVEL,
                     "coef": {"a0": 2.0, "b0": 2.0, f"{x}0": 2.0, "k0": 2.0}})
        rows.append({"name": "Q_slope", "group": "Q", "system": "slope",
                     "y": float(errset._slope(rec["dz"], rec["Q"])), "eps": ts,
                     "coef": {"a1": 2.0, "b1": 2.0, f"{x}1": 2.0, "k1": 2.0}})
    return rows


def rows_for_condition(rows: list[dict], condition: str) -> list[dict]:
    drop = {"full": set(), "-N": {"N"}, "-X": {"X"}}[condition]
    return [r for r in rows if r["group"] not in drop]


def param_names(consumer: str, world: str) -> list[str]:
    x = x_name(consumer)[0]
    names = ["a0", "b0", f"{x}0", "a1", "b1", f"{x}1"]
    if world == "OW":
        names += ["k0", "k1"]
    return names


# --------------------------------------------------------------------------- OR --
def method_OR(rec: dict, probes: dict, condition: str, q_ok: bool, rows: list[dict]) -> dict:
    leg, consumer = rec["leg"], rec["consumer"]
    src = set()
    over = []
    if condition != "-N":
        for key in ("Na", "Nb"):
            dev = float(np.max(np.abs(np.asarray(probes[key]) - 1.0)))
            if dev > TAU_N:
                src.add("ENV")
                over.append(key)
    if condition != "-X":
        dev = float(np.max(np.abs(np.asarray(probes["X"]) - 1.0)))
        if dev > TAU_N:
            src.add(x_name(consumer)[2])
            over.append(probes["X_key"])
    q_alarm = False
    if q_ok:
        ql = next(r["y"] for r in rows if r["name"] == "Q_level")
        qs = next(r["y"] for r in rows if r["name"] == "Q_slope")
        q_alarm = bool(abs(ql) > EPS_Q_LEVEL or abs(qs) > SLOPE_TOL[leg])
    alarm = bool(src) or q_alarm
    decision = "ALARM" if alarm else ("UNDECIDED" if not q_ok else "COMPATIBLE")
    return {"decision": decision, "localized": sorted(src) if alarm else [], "probes_over": over,
            "q_alarm": q_alarm}


# ------------------------------------------------------------- LP machinery --
def _matrix(rows: list[dict], names: list[str]):
    idx = {n: j for j, n in enumerate(names)}
    a = np.zeros((len(rows), len(names)))
    for i, r in enumerate(rows):
        for p, c in r["coef"].items():
            if p in idx:                      # CW: k columns absent (k fixed to 0)
                a[i, idx[p]] = c
    y = np.array([r["y"] for r in rows], dtype=float)
    eps = np.array([r["eps"] for r in rows], dtype=float)
    return a, y, eps


def _lp(a, y, eps, c, zero_idx=()):
    """(min, max) of c . theta over {theta : |a theta - y| <= eps}, theta free except zero_idx (fixed 0).
    Returns None if infeasible; +-inf for unbounded ends."""
    n = a.shape[1]
    bounds = [((0.0, 0.0) if j in zero_idx else (None, None)) for j in range(n)]
    a_ub = np.vstack([a, -a])
    b_ub = np.concatenate([y + eps, eps - y])
    feas = linprog(np.zeros(n), A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs", options=HIGHS_OPTS)
    if feas.status == 2:
        return None
    if feas.status != 0:
        raise RuntimeError(f"LP feasibility status {feas.status}: {feas.message}")
    if not np.any(c):
        return (0.0, 0.0)
    out = []
    for sign in (1.0, -1.0):
        res = linprog(sign * c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs", options=HIGHS_OPTS)
        if res.status == 0:
            out.append(sign * float(res.fun))
        elif res.status == 3:
            out.append(-np.inf if sign > 0 else np.inf)
        else:
            raise RuntimeError(f"LP status {res.status}: {res.message}")
    return (out[0], out[1])


def _excludes0(iv) -> bool:
    return iv is not None and (iv[0] > 0.0 or iv[1] < 0.0)


# ------------------------------------------------------------- closed form --
def _iv_add(p, q):
    return (p[0] + q[0], p[1] + q[1])


def _iv_sum(ivs):
    out = (0.0, 0.0)
    for iv in ivs:
        out = _iv_add(out, iv)
    return out


def _iv_intersect(p, q):
    lo, hi = max(p[0], q[0]), min(p[1], q[1])
    return (lo, hi) if lo <= hi else None


def _system_closed_form(rows: list[dict], names: list[str]):
    """Probe bands per parameter and the Q band (on the plain sum) of one system (level or slope)."""
    band = {n: (-np.inf, np.inf) for n in names}
    qband = None
    for r in rows:
        coef = {p: c for p, c in r["coef"].items() if p in names}
        if r["group"] == "Q":
            cs = set(coef.values())
            assert cs == {2.0} and set(coef) == set(names), (r["name"], coef, names)
            qband = ((r["y"] - r["eps"]) / 2.0, (r["y"] + r["eps"]) / 2.0)
        else:
            (p, c), = coef.items()
            iv = ((r["y"] - r["eps"]) / c, (r["y"] + r["eps"]) / c)
            cur = _iv_intersect(band[p], iv)
            if cur is None:
                raise RuntimeError("two probe rows on one parameter are disjoint")
            band[p] = cur
    return band, qband


def closed_form(rows: list[dict], names: list[str], combos: dict[str, list[str]],
                blocks: dict[str, list[str]]) -> dict:
    """Feasibility, per-parameter projections, combination ranges (unit coefficients) and block-null
    feasibility for the product of the level and slope systems."""
    level = [n for n in names if n.endswith("0")]
    slope = [n for n in names if n.endswith("1")]
    systems = {}
    for sysname, sysnames in (("level", level), ("slope", slope)):
        srows = [r for r in rows if r["system"] == sysname]
        systems[sysname] = (sysnames, *_system_closed_form(srows, sysnames))
    feasible = True
    for sysnames, band, qband in systems.values():
        if qband is not None and _iv_intersect(_iv_sum(band[n] for n in sysnames), qband) is None:
            feasible = False
    if not feasible:
        return {"feasible": False}
    proj, ranges, block_null = {}, {}, {}

    def rng(sysnames, band, qband, subset):
        s_in = _iv_sum(band[n] for n in subset)
        if qband is None:
            return s_in
        s_out = _iv_sum(band[n] for n in sysnames if n not in subset)
        return _iv_intersect(s_in, (qband[0] - s_out[1], qband[1] - s_out[0]))

    for sysnames, band, qband in systems.values():
        for n in sysnames:
            proj[n] = rng(sysnames, band, qband, [n])
    for cname, members in combos.items():
        sysnames, band, qband = systems["level" if members[0].endswith("0") else "slope"]
        ranges[cname] = rng(sysnames, band, qband, members)
    for bname, members in blocks.items():
        sysnames, band, qband = systems["level" if members[0].endswith("0") else "slope"]
        ok = all(band[n][0] <= 0.0 <= band[n][1] for n in members)
        if ok and qband is not None:
            rest = _iv_sum(band[n] for n in sysnames if n not in members)
            ok = _iv_intersect(rest, qband) is not None
        block_null[bname] = ok
    return {"feasible": True, "proj": proj, "ranges": ranges, "block_null_feasible": block_null}


def _iv_close(p, q) -> bool:
    if p is None or q is None:
        return p is None and q is None
    for u, v in zip(p, q):
        if np.isinf(u) or np.isinf(v):
            if u != v:
                return False
        elif abs(u - v) > CF_TOL:
            return False
    return True


# ---------------------------------------------------------------------- inversions --
def invert(rec: dict, rows: list[dict], world: str, condition: str) -> dict:
    consumer = rec["consumer"]
    x, _, xsrc = x_name(consumer)
    names = param_names(consumer, world)
    a, y, eps = _matrix(rows, names)
    idx = {n: j for j, n in enumerate(names)}
    combos, blocks = {}, {"ENV_level": ["a0", "b0"], "ENV_slope": ["a1", "b1"]}
    removed = None
    q_present = any(r["group"] == "Q" for r in rows)
    if world == "OW" and condition != "full" and q_present:
        removed = "ENV" if condition == "-N" else xsrc
        if removed == "ENV":
            combos = {"R+k_level": ["a0", "b0", "k0"], "R+k_slope": ["a1", "b1", "k1"]}
        else:
            combos = {"R+k_level": [f"{x}0", "k0"], "R+k_slope": [f"{x}1", "k1"]}
    cf = closed_form(rows, names, combos, blocks)
    feas_lp = _lp(a, y, eps, np.zeros(len(names))) is not None
    checks = {"feasible": feas_lp == cf["feasible"]}
    if not feas_lp:
        return {"world": world, "status": "MODEL_INCONSISTENT", "localized_block": [], "localized_literal": [],
                "proj": {}, "ranges": {}, "block_null_feasible": {}, "conditional": None, "removed": removed,
                "cf_agree": all(checks.values()), "cf_checks": checks}
    proj = {}
    for n in names:
        c = np.zeros(len(names))
        c[idx[n]] = 1.0
        proj[n] = _lp(a, y, eps, c)
        checks[f"proj:{n}"] = _iv_close(proj[n], cf["proj"][n])
    ranges = {}
    for cname, members in combos.items():
        c = np.zeros(len(names))
        for m in members:
            c[idx[m]] = 1.0
        ranges[cname] = _lp(a, y, eps, c)
        checks[f"range:{cname}"] = _iv_close(ranges[cname], cf["ranges"][cname])
    block_null = {}
    for bname, members in blocks.items():
        block_null[bname] = _lp(a, y, eps, np.zeros(len(names)), zero_idx={idx[m] for m in members}) is not None
        checks[f"block:{bname}"] = block_null[bname] == cf["block_null_feasible"][bname]
    loc_block, loc_literal = set(), set()
    if (not block_null["ENV_level"]) or (not block_null["ENV_slope"]):
        loc_block.add("ENV")
    if any(_excludes0(proj[p]) for p in ("a0", "b0", "a1", "b1")):
        loc_literal.add("ENV")
    if _excludes0(proj[f"{x}0"]) or _excludes0(proj[f"{x}1"]):
        loc_block.add(xsrc)
        loc_literal.add(xsrc)
    if world == "OW" and (_excludes0(proj["k0"]) or _excludes0(proj["k1"])):
        loc_block.add("PRE")
        loc_literal.add("PRE")
    conditional = None
    if removed is not None and (_excludes0(ranges["R+k_level"]) or _excludes0(ranges["R+k_slope"])):
        conditional = [removed, "PRE"]
    return {"world": world, "status": "OK", "localized_block": sorted(loc_block),
            "localized_literal": sorted(loc_literal), "proj": proj, "ranges": ranges,
            "block_null_feasible": block_null, "conditional": conditional, "removed": removed,
            "cf_agree": all(checks.values()), "cf_checks": checks}


# ------------------------------------------------------------------------- scoring --
def truth_sources(errors) -> set:
    return {SOURCE_OF[p] for p in methods.realized_primitives(errors) if p in SOURCE_OF}


def score_detection(truth: set, decision: str) -> str:
    if decision == "UNDECIDED":
        return "UNDECIDED"
    if truth:
        return "DETECTED" if decision == "ALARM" else "MISSED"
    return "FALSE-ALARM" if decision == "ALARM" else "CORRECT-NEG"


def score_localization(truth: set, decision: str, localized: list, model_inconsistent: bool = False):
    if decision != "ALARM":
        return None
    loc = set(localized)
    if model_inconsistent or not loc:
        return "NONE"
    if loc - truth:
        return "MIS"
    if loc == truth:
        return "FULL"
    return "PARTIAL"


def score_conditional(truth: set, decision: str, conditional):
    if decision != "ALARM" or conditional is None:
        return None
    return "COND-CORRECT" if set(conditional) & truth else "COND-VACUOUS"


# ---------------------------------------------- expectations (fixed before the run) --
_FULL19 = ("S01", "S02", "S03", "S04", "S05", "S06", "S07", "K01", "K02", "P01", "M01", "M02", "M04",
           "M05", "M06", "M07", "M08", "Z02", "T02")
_NONE11 = ("S01", "S02", "S03", "K01", "K02", "K03", "P01", "M01", "M02", "M03", "M04")
_OR_X_FULL11 = ("S01", "S02", "S03", "K01", "K02", "P01", "M01", "M02", "M04", "Z02", "T02")
_OR_X_PART5 = ("M03", "M05", "M06", "M07", "M08")
_OR_X_NONE5 = ("S04", "S05", "S06", "S07", "K03")
_COND_N15 = ("S01", "S02", "S03", "K01", "K02", "K03", "P01", "M01", "M02", "M03", "M04",
             "M05", "M06", "M07", "M08")
_COND_X10 = ("S04", "S05", "S06", "S07", "K03", "M03", "M05", "M06", "M07", "M08")
LOW_CONFIDENCE_ALL = {"N04"}
LOW_CONFIDENCE_INVERSION = {"S04", "S05", "M05", "M06"}      # E3 class: probe vs consumer integrand O(dz/L)


def _expected_cell(case: str, condition: str, method: str) -> dict:
    if case in NORMAL_CASES:
        det = "UNDECIDED" if case in ("Z01", "T01") else "CORRECT-NEG"
        return {"detection": det, "localization": None, "conditional": None}
    if condition == "-N" and case in ("Z02", "T02"):
        return {"detection": "UNDECIDED", "localization": None, "conditional": None}
    loc, cond = None, None
    if condition == "full":
        if method == "OR":
            loc = "FULL" if case in _FULL19 else ("PARTIAL" if case == "M03" else "NONE")
        elif method == "CW":
            loc = "FULL" if case in _FULL19 else "NONE"
        else:
            loc = "FULL"
    elif condition == "-N":
        if method in ("OR", "OW"):
            loc = ("FULL" if case in ("S04", "S05", "S06", "S07") else
                   "PARTIAL" if case in ("M05", "M06", "M07", "M08") else "NONE")
            if method == "OW" and case in _COND_N15:
                cond = "COND-CORRECT"
        else:
            loc = "PARTIAL" if case == "M03" else ("MIS" if case == "K03" else "FULL")
    else:
        if method in ("OR", "OW"):
            loc = ("FULL" if case in _OR_X_FULL11 else "PARTIAL" if case in _OR_X_PART5 else "NONE")
            if method == "OW" and case in _COND_X10:
                cond = "COND-CORRECT"
        else:
            loc = "MIS" if case in ("K03", "M03") else "FULL"
    return {"detection": "DETECTED", "localization": loc, "conditional": cond}


EXPECTED = {(c[0], cond, m): _expected_cell(c[0], cond, m) for c in CASES for cond in CONDITIONS for m in METHODS}


def _agree(case: str, method: str, pred: dict, obs: dict) -> str:
    same = all(pred[k] == obs[k] for k in ("detection", "localization", "conditional"))
    low = case in LOW_CONFIDENCE_ALL or (method in ("CW", "OW") and case in LOW_CONFIDENCE_INVERSION)
    if low:
        return "agree(low-confidence)" if same else "disagree(low-confidence)"
    return "agree" if same else "disagree"


# ---------------------------------------------------------------------- driver --
def run_case_rows(case_row, q_ref12: float) -> dict:
    case, leg, consumer, errors, *_ = case_row
    rec = errset.run_case(case, leg, consumer, errors)
    probes = probe_readings(leg, consumer, errors)
    q_ok, q_reason = q_rows_available(rec, q_ref12)
    rows = evidence_rows(rec, probes, q_ok)
    truth = truth_sources(errors)
    cells = {}
    for condition in CONDITIONS:
        crow = rows_for_condition(rows, condition)
        orr = method_OR(rec, probes, condition, q_ok, rows)
        results = {"OR": {**orr, "localized_block": orr["localized"], "localized_literal": orr["localized"],
                          "status": "OK", "conditional": None, "cf_agree": True}}
        for world in ("CW", "OW"):
            inv = invert(rec, crow, world, condition)
            results[world] = {**inv, "decision": orr["decision"]}
        for method in METHODS:
            r = results[method]
            mi = r.get("status") == "MODEL_INCONSISTENT"
            obs = {"detection": score_detection(truth, r["decision"]),
                   "localization": score_localization(truth, r["decision"], r["localized_block"], mi),
                   "conditional": score_conditional(truth, r["decision"], r.get("conditional"))}
            lit = score_localization(truth, r["decision"], r["localized_literal"], mi)
            pred = EXPECTED[(case, condition, method)]
            cells[f"{condition}|{method}"] = {
                "condition": condition, "condition_label": condition_label(condition, consumer), "method": method,
                "decision": r["decision"], "status": r.get("status"), "localized": r["localized_block"],
                "localized_literal_rule": r["localized_literal"], "conditional": r.get("conditional"),
                "score": obs, "score_localization_literal_rule": lit, "expected": pred,
                "agreement": _agree(case, method, pred, obs),
                "proj": {k: list(v) if v is not None else None for k, v in r.get("proj", {}).items()},
                "ranges": {k: list(v) if v is not None else None for k, v in r.get("ranges", {}).items()},
                "block_null_feasible": r.get("block_null_feasible", {}),
                "cf_agree": r.get("cf_agree", True),
                "conditional_on_non_alarm": bool(r.get("conditional") is not None and r["decision"] != "ALARM"),
            }
    return {"case": case, "leg": leg, "consumer": consumer, "errors": sorted(set(errors)),
            "realized_primitives": sorted(methods.realized_primitives(errors)), "truth_sources": sorted(truth),
            "has_error": bool(truth), "Q": rec["Q"], "Q_ref": rec["Q_ref"], "q_ref_12": q_ref12,
            "identity_overlap": rec["identity_overlap"], "q_rows_available": q_ok, "q_guard": q_reason,
            "probes": probes, "rows": rows, "cells": cells}


def summarize(results: list[dict]) -> dict:
    table = {}
    for condition in CONDITIONS:
        for method in METHODS:
            det = full = part = mis = none = und = fa = 0
            cc = cv = 0
            lit = {"FULL": 0, "PARTIAL": 0, "MIS": 0, "NONE": 0}
            agree = {}
            cf_bad = 0
            cond_non_alarm = 0
            for r in results:
                c = r["cells"][f"{condition}|{method}"]
                s = c["score"]
                if r["has_error"]:
                    det += s["detection"] == "DETECTED"
                    und += s["detection"] == "UNDECIDED"
                    full += s["localization"] == "FULL"
                    part += s["localization"] == "PARTIAL"
                    mis += s["localization"] == "MIS"
                    none += s["localization"] == "NONE"
                    if c["score_localization_literal_rule"] is not None:
                        lit[c["score_localization_literal_rule"]] += 1
                else:
                    fa += s["detection"] == "FALSE-ALARM"
                cc += s["conditional"] == "COND-CORRECT"
                cv += s["conditional"] == "COND-VACUOUS"
                agree[c["agreement"]] = agree.get(c["agreement"], 0) + 1
                cf_bad += not c["cf_agree"]
                cond_non_alarm += c["conditional_on_non_alarm"]
            table[f"{condition}|{method}"] = {
                "condition": condition, "method": method, "detected": det, "FULL": full, "PARTIAL": part,
                "MIS": mis, "NONE": none, "undecided": und, "false_alarm_normal": fa,
                "cond_correct": cc, "cond_vacuous": cv, "literal_rule_localization": lit,
                "agreement": agree, "closed_form_disagreements": cf_bad,
                "conditional_on_non_alarm": cond_non_alarm}
    return table


def run_all_rows() -> dict:
    ref12 = {}
    results = []
    for case_row in CASES:
        case, leg, consumer, *_ = case_row
        key = (leg, consumer)
        if key not in ref12:
            ref12[key] = errset.reference(leg, consumer, (1, 2))
        results.append(run_case_rows(case_row, ref12[key]))
    return {"rows": results, "summary": summarize(results),
            "constants": {"h_f": H_F, "tau_n": TAU_N, "eps_probe_level": EPS_PROBE_LEVEL,
                          "eps_q_level": EPS_Q_LEVEL, "slope_tol": SLOPE_TOL, "zero_floor_relative": ZERO_FLOOR,
                          "identity_tol": ID_TOL, "closed_form_tol": CF_TOL, "highs_options": HIGHS_OPTS}}


# ---------------------------------------------------------------- program: results/evidence_rows/
OUT = HERE.parent / "results" / "evidence_rows"


def identity() -> dict:
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform()}


def _clean(o):
    """JSON-safe: +-inf -> '+inf' / '-inf' strings, tuples -> lists."""
    if isinstance(o, float):
        if math.isinf(o):
            return "+inf" if o > 0 else "-inf"
        return o
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (set, frozenset)):
        return sorted(_clean(v) for v in o)
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def _fmt_iv(iv) -> str:
    if iv is None:
        return "empty"
    return "[" + ", ".join(("-inf" if v == -math.inf else "+inf" if v == math.inf else f"{v:.6g}") for v in iv) + "]"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    res = run_all_rows()
    res["identity"] = identity()
    res["setup"] = "27-case mixed-error set; methods OR, CW, OW; conditions full, -N, -X; expectations fixed before the run"
    res["wall_seconds"] = time.perf_counter() - t0
    (OUT / "evidence_rows.json").write_text(json.dumps(_clean(res), indent=1), encoding="utf-8")

    with (OUT / "evidence_rows_cells.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case", "leg", "consumer", "errors", "truth_sources", "q_rows_available", "q_guard",
                    "condition", "method", "decision", "inversion_status", "localized", "localized_literal_rule",
                    "conditional", "score_detection", "score_localization", "score_localization_literal_rule",
                    "score_conditional", "expected_detection", "expected_localization", "expected_conditional",
                    "agreement", "closed_form_agrees",
                    "y_Na_level", "y_Na_slope", "y_Nb_level", "y_Nb_slope", "y_X_level", "y_X_slope",
                    "y_Q_level", "y_Q_slope",
                    "proj_a0", "proj_b0", "proj_x0", "proj_k0", "proj_a1", "proj_b1", "proj_x1", "proj_k1",
                    "range_R+k_level", "range_R+k_slope", "ENV_level_null_feasible", "ENV_slope_null_feasible"])
        for r in res["rows"]:
            y = {row["name"].replace("W_", "X_").replace("D_", "X_"): row["y"] for row in r["rows"]}
            for cond in CONDITIONS:
                for m in METHODS:
                    c = r["cells"][f"{cond}|{m}"]
                    x = "w" if r["consumer"] == "int" else "d"
                    pj = c["proj"]
                    w.writerow([r["case"], r["leg"], r["consumer"], "+".join(r["errors"]) or "-",
                                "+".join(r["truth_sources"]) or "-", r["q_rows_available"], r["q_guard"],
                                c["condition_label"], m, c["decision"], c["status"], "+".join(c["localized"]) or "-",
                                "+".join(c["localized_literal_rule"]) or "-",
                                ("(+)".join(c["conditional"]) if c["conditional"] else "-"),
                                c["score"]["detection"], c["score"]["localization"],
                                c["score_localization_literal_rule"], c["score"]["conditional"],
                                c["expected"]["detection"], c["expected"]["localization"],
                                c["expected"]["conditional"], c["agreement"], c["cf_agree"],
                                *[y.get(k) for k in ("Na_level", "Na_slope", "Nb_level", "Nb_slope", "X_level",
                                                     "X_slope", "Q_level", "Q_slope")],
                                *[_fmt_iv(pj[k]) if k in pj else "" for k in
                                  ("a0", "b0", f"{x}0", "k0", "a1", "b1", f"{x}1", "k1")],
                                _fmt_iv(c["ranges"]["R+k_level"]) if "R+k_level" in c["ranges"] else "",
                                _fmt_iv(c["ranges"]["R+k_slope"]) if "R+k_slope" in c["ranges"] else "",
                                c["block_null_feasible"].get("ENV_level_null", c["block_null_feasible"].get("ENV_level", "")),
                                c["block_null_feasible"].get("ENV_slope_null", c["block_null_feasible"].get("ENV_slope", ""))])

    for key, s in res["summary"].items():
        print(f"{s['condition']:5} {s['method']:4} detected {s['detected']}  FULL {s['FULL']}  PARTIAL {s['PARTIAL']}  "
              f"MIS {s['MIS']}  NONE {s['NONE']}  undecided {s['undecided']}  false alarms {s['false_alarm_normal']}  "
              f"closed-form disagreements {s['closed_form_disagreements']}  {s['agreement']}")


if __name__ == "__main__":
    main()
