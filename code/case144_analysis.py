"""The 144-case constant-multiplier test (Fig. 5, Table 4 lower half, Table S8): methods OR, CW and OW under the
four information conditions, scoring against the fault table, and the two budget sweeps. Reads data/case144/,
writes results/case144/."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import ROUND_CEILING, Decimal
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = REPO / "results" / "case144"
DATA = REPO / "data" / "case144"

M_OBS, M_TRUTH = "observations.json", "truth.json"
M_SETTINGS = "settings.json"

SOURCES = ("normalization", "measure", "operator")          # a, b, c
PRE = "PRE"                                                 # open-world prefactor k
PARAM_SYMBOL = {"normalization": "a", "measure": "b", "operator": "c", PRE: "k"}
PROBE_COEF = {"normalization": 2.0, "measure": 1.0, "operator": 1.0}
Q_COEF = {"normalization": 4.0, "measure": 2.0, "operator": 2.0, PRE: 2.0}
ROW_ORDER = ("Q", "normalization", "measure", "operator")   # row order of the study
CONDITIONS = ("full", "-N", "-W", "-O")
REMOVED = {"full": None, "-N": "normalization", "-W": "measure", "-O": "operator"}
STUDY_POLICY = {"full": "full", "-N": "minus_normalization", "-W": "minus_measure", "-O": "minus_operator"}
METHODS = ("OR", "CW", "OW")
JOINT_ALARM_CLASSES = ("INTERFACE_INCONSISTENT", "CONTROL_PATH_INCONSISTENT", "REFERENCE_INCONSISTENT",
                       "INTERFACE_OR_OUTPUT_INCONSISTENT")
COHORTS = ("primary", "boundary")
BOUNDARY_FAMILIES = ("coarse_clean", "missing_two_probes", "changing_identity", "incoherent_channels")

# baseline budgets (asserted against data/case144/settings.json)
BASELINE_BUDGETS = {"reference_relative": 0.02, "log_reference_budget": 0.020203,
             "component_residual": 1e-10, "log_probe_budget": 1.01e-10}
SWEEP_A_TAU_G = ("0.001", "0.005", "0.01", "0.02", "0.05", "0.10")
SWEEP_B_TAU_C = ("1e-10", "1e-8", "1e-6", "1e-4", "1e-2")
MAX_COARSE_DISC_STATED = 0.00408717       # stated value: largest coarse-grid discretization deviation

LP_CF_TOL = 1e-12                         # |LP - closed form| on every finite interval end (log units)
HIGHS_OPTIONS = {"presolve": False, "primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10}
HIGHS_SMALL_MATRIX_VALUE = 1e-9           # HiGHS drops |entries| below this -> guarded, never silently


# ================================================================ data access ==
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class StudyData:
    """Read-only access to the input data of the 144-case study (data/case144).

    truth.json stays locked until unlock_truth() receives the digest of the finished prediction set (scoring step
    only).
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.member_sha256: dict[str, str] = {}
        self._truth_digest: str | None = None

    def _read_json(self, member: str):
        raw = (self.data_dir / member).read_bytes()
        self.member_sha256[member] = sha256_bytes(raw)
        return json.loads(raw.decode("utf-8"))

    def observations(self) -> list:
        return self._read_json(M_OBS)

    def settings(self) -> dict:
        return self._read_json(M_SETTINGS)

    def unlock_truth(self, predictions_digest: str) -> None:
        self._truth_digest = predictions_digest

    def truth(self) -> dict:
        if self._truth_digest is None:
            raise RuntimeError("truth.json is readable only by the scoring step, after all predictions are hashed")
        return self._read_json(M_TRUTH)


# ================================================================ budgets ==
@dataclass(frozen=True)
class Budgets:
    label: str
    sweep: str          # "baseline" / "A" / "B"
    tau_g: float        # reference relative budget (study reference_relative; OR 2 % check)
    eps_q: float        # log budget of the Q row
    tau_c: float        # component residual budget (alarm; branch probe)
    eps_c: float        # log budget of the N / W / O rows


def baseline_budgets(cfg: dict) -> Budgets:
    for key, val in BASELINE_BUDGETS.items():
        if cfg[key] != val:
            raise RuntimeError(f"data/case144/settings.json {key} = {cfg[key]} differs from the baseline value {val}")
    return Budgets("baseline", "baseline", cfg["reference_relative"], cfg["log_reference_budget"],
                   cfg["component_residual"], cfg["log_probe_budget"])


def eps_q_of(tau_g: str) -> float:
    """-ln(1 - tau_g) rounded UP to 1e-6 (sweep A)."""
    x = -math.log(1.0 - float(tau_g))
    return float(Decimal(repr(x)).quantize(Decimal("0.000001"), rounding=ROUND_CEILING))


def sweep_points(baseline: Budgets) -> list[Budgets]:
    pts = [Budgets(f"A:tau_g={t}", "A", float(t), eps_q_of(t), baseline.tau_c, baseline.eps_c) for t in SWEEP_A_TAU_G]
    pts += [Budgets(f"B:tau_c={t}", "B", baseline.tau_g, baseline.eps_q, float(t), float(Decimal("1.01") * Decimal(t)))
            for t in SWEEP_B_TAU_C]
    return pts


def guards_of(cfg: dict) -> dict:
    """Fixed (non-swept) settings of the study: grid guard, grid-only slope rule, absolute reference budget."""
    return {k: float(cfg[k]) for k in ("adjacent_last_relative", "trim_slope_change", "max_log_residual",
                                       "slope_absolute", "absolute_budget_nm2")}


# ================================================================ input checks / grid metrics ==
class _Invalid(Exception):
    def __init__(self, classification: str, message: str):
        super().__init__(message)
        self.classification = classification


def _real_vector(value, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype.kind not in "iuf":
        raise _Invalid("INVALID_INPUT", f"{name}: expected real numbers")
    arr = arr.astype(float)
    if arr.ndim != 1:
        raise _Invalid("INVALID_INPUT", f"{name}: expected a one-dimensional array")
    if not np.all(np.isfinite(arr)):
        raise _Invalid("INVALID_NUMERIC_INPUT", f"{name}: non-finite entries")
    return arr


def _real_scalar(value, name: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise _Invalid("INVALID_INPUT", f"{name}: expected a real scalar")
    x = float(value)
    if not math.isfinite(x):
        raise _Invalid("INVALID_NUMERIC_INPUT", f"{name}: non-finite")
    if nonnegative and x < 0:
        raise _Invalid("INVALID_INPUT", f"{name}: negative")
    return x


def check_inputs(obs: dict, comps: dict) -> tuple[np.ndarray, np.ndarray, float]:
    h, q = _real_vector(obs["h"], "h"), _real_vector(obs["q"], "q")
    if len(h) < 3 or len(q) != len(h):
        raise _Invalid("INVALID_INPUT", "h and q need equal lengths of at least three")
    if np.any(h <= 0) or np.any(np.diff(h) >= 0):
        raise _Invalid("INVALID_INPUT", "h must be positive and strictly decreasing")
    ref = _real_scalar(obs["reference"], "reference")
    ids = obs["identities"]
    if not isinstance(ids, list) or len(ids) != len(h) or any(v in (None, "", {}, []) for v in ids):
        raise _Invalid("INVALID_INPUT", "one nonempty comparison record per grid is required")
    for key, val in comps.items():
        if not isinstance(key, str) or not key:
            raise _Invalid("INVALID_INPUT", "component names must be nonempty strings")
        _real_scalar(val, f"component {key}", nonnegative=True)
    return h, q, ref


def comparable(obs: dict) -> bool:
    first = obs["identities"][0]
    return all(rec == first for rec in obs["identities"])


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Unweighted least squares y = intercept + slope * x (centred closed form)."""
    xm, ym = float(np.mean(x)), float(np.mean(y))
    dx = x - xm
    slope = float(np.dot(dx, y - ym) / np.dot(dx, dx))
    return slope, ym - slope * xm


def grid_metrics(h: np.ndarray, q: np.ndarray) -> dict | None:
    if np.any(q == 0):
        return None
    x, y = np.log(h), np.log(np.abs(q))
    slope, intercept = _ols(x, y)
    trim, _ = _ols(x[1:], y[1:])                      # coarsest rung dropped (h decreasing)
    return {"slope": slope, "trim_slope": trim, "trim_change": abs(slope - trim),
            "max_log_residual": float(np.max(np.abs(y - (intercept + slope * x)))),
            "adjacent_relative": [float(abs(q[i + 1] - q[i]) / abs(q[i + 1])) for i in range(len(q) - 1)]}


# ================================================================ verdict cascade of the study (rewrite) ==
def study_diagnostic(obs: dict, comps: dict, bud: Budgets, guards: dict) -> dict:
    """Independent rewrite of the study's diagnostic contract (no paired control is supplied in the study)."""
    d = {"input_status": "NOT_EVALUATED", "component_residuals": None, "component_ok": None,
         "component_status": "NOT_PROVIDED", "comparable": None, "path_status": "NOT_PROVIDED",
         "grid": None, "grid_ok": None, "grid_status": "NOT_EVALUATED", "reference_error": None,
         "reference_ok": None, "scale_attribution": "NOT_ESTABLISHED", "classification": None}
    try:
        h, q, ref = check_inputs(obs, comps)
    except _Invalid as exc:
        d.update(input_status="INVALID", classification=exc.classification)
        return d
    d["input_status"] = "VALID"
    d["component_residuals"] = {k: float(v) for k, v in comps.items()}
    if comps:
        d["component_ok"] = all(float(v) <= bud.tau_c for v in comps.values())
        d["component_status"] = "PASSED" if d["component_ok"] else "FAILED"
    d["comparable"] = comparable(obs)
    if not d["comparable"]:
        d["classification"] = "INCOMPARABLE_SEQUENCE"
        return d
    absb = guards["absolute_budget_nm2"]
    if abs(ref) <= absb:                                   # zero-reference regime (not exercised by the study)
        d["reference_error"] = float(np.max(np.abs(q - ref)))
        d["reference_ok"] = d["reference_error"] <= absb
        d["grid_status"] = "NOT_APPLICABLE"
        if d["component_ok"] is None:
            d["classification"] = "INSUFFICIENT_COMPONENT_EVIDENCE"
        elif d["component_ok"] and d["reference_ok"]:
            d["classification"] = "ZERO_REFERENCE_ABSOLUTE_ONLY"
        else:
            d["classification"] = "INTERFACE_OR_OUTPUT_INCONSISTENT"
        return d
    d["reference_error"] = float(abs(q[-1] / ref - 1.0))
    d["reference_ok"] = d["reference_error"] <= bud.tau_g
    g = grid_metrics(h, q)
    d["grid"] = g
    d["grid_ok"] = bool(g is not None and g["adjacent_relative"][-1] <= guards["adjacent_last_relative"]
                        and g["trim_change"] <= guards["trim_slope_change"]
                        and g["max_log_residual"] <= guards["max_log_residual"])
    d["grid_status"] = "NOT_APPLICABLE" if g is None else ("PASSED" if d["grid_ok"] else "FAILED")
    if d["component_ok"] is False:
        d["classification"] = "INTERFACE_INCONSISTENT"
        if float(comps.get("normalization", 0.0)) > bud.tau_c:
            d["scale_attribution"] = "NORMALIZATION_INTERFACE_INCONSISTENCY_OBSERVED"
    elif d["component_ok"] is None:
        d["classification"] = "INSUFFICIENT_COMPONENT_EVIDENCE"
    elif not d["grid_ok"]:
        d["classification"] = "INSUFFICIENT_GRID_EVIDENCE"
    elif not d["reference_ok"]:
        d["classification"] = "REFERENCE_INCONSISTENT"
    else:
        d["classification"] = "COMPATIBLE_WITH_REFERENCE"
    return d


def study_decision(obs: dict, diag: dict, removed: str | None, comps: dict) -> str:
    """Decision of the study's joint policies (full / minus_*): cascade + required-probe rule."""
    if diag["input_status"] != "VALID" or not diag["comparable"]:
        return "UNDECIDED"
    if diag["classification"] in JOINT_ALARM_CLASSES:
        return "ALARM"
    if diag["classification"] == "COMPATIBLE_WITH_REFERENCE":
        required = set(obs["required_components"]) - {removed}
        return "COMPATIBLE" if required <= set(comps) else "UNDECIDED"
    return "UNDECIDED"


def inversion_rows(obs: dict, removed: str | None, diag: dict, bud: Budgets, guards: dict) -> tuple[list, bool]:
    """Rows of the inversion: retained N / W / O probe ratios (finest grid) and the Q row under the study's rule
        (grid guard passed, Q_f > 0, Q_ref above the absolute budget).
    """
    ratios = {k: float(v) for k, v in obs["probe_ratios"].items() if k != removed}
    q_row = bool(diag["grid_ok"]) and obs["q"][-1] > 0 and obs["reference"] > guards["absolute_budget_nm2"]
    rows = []
    for name in ROW_ORDER:
        if name == "Q":
            if q_row:
                rows.append({"name": "Q", "ratio": obs["q"][-1] / obs["reference"], "eps": bud.eps_q})
        elif name in ratios:
            rows.append({"name": name, "ratio": ratios[name], "eps": bud.eps_c})
    applicable = bool(rows) and all(math.isfinite(r["ratio"]) and r["ratio"] > 0 for r in rows)
    for r in rows:
        r["y"] = math.log(r["ratio"]) if applicable else None
    return rows, applicable


# ================================================================ interval projections ==
def _excludes0(iv) -> bool:
    return iv[0] > 0.0 or iv[1] < 0.0


def closed_form(params: list, rows: list, combos: dict) -> dict:
    """Exact projections of {theta : |A theta - y| <= eps} for diagonal probe rows (a box) and at most one Q
    slab with positive coefficients: interval arithmetic.  combos: name -> {param: weight}, weights
    proportional to the Q-row coefficients on the combination's support."""
    lo = {p: -math.inf for p in params}
    hi = {p: math.inf for p in params}
    q = None
    for r in rows:
        if r["name"] == "Q":
            q = r
            continue
        c = PROBE_COEF[r["name"]]
        lo[r["name"]], hi[r["name"]] = (r["y"] - r["eps"]) / c, (r["y"] + r["eps"]) / c
    if q is None:
        comb = {}
        for name, w in combos.items():
            comb[name] = (sum(w[p] * lo[p] for p in w), sum(w[p] * hi[p] for p in w))
        return {"feasible": True, "proj": {p: (lo[p], hi[p]) for p in params}, "comb": comb}
    qlo, qhi = q["y"] - q["eps"], q["y"] + q["eps"]

    def partial(skip) -> tuple[float, float]:
        smin = smax = 0.0
        for p in params:
            if p not in skip:
                smin += Q_COEF[p] * lo[p]
                smax += Q_COEF[p] * hi[p]
        return smin, smax

    tmin, tmax = partial(())
    feasible = tmin <= qhi and tmax >= qlo
    proj = {}
    for p in params:
        rmin, rmax = partial((p,))
        proj[p] = (max(lo[p], (qlo - rmax) / Q_COEF[p]), min(hi[p], (qhi - rmin) / Q_COEF[p]))
    comb = {}
    for name, w in combos.items():
        support = tuple(w)
        lam = Q_COEF[support[0]] / w[support[0]]
        assert all(abs(Q_COEF[p] / w[p] - lam) < 1e-15 for p in support)
        own_min = sum(Q_COEF[p] * lo[p] for p in support)
        own_max = sum(Q_COEF[p] * hi[p] for p in support)
        rmin, rmax = partial(support)
        comb[name] = (max(own_min, qlo - rmax) / lam, min(own_max, qhi - rmin) / lam)
    return {"feasible": feasible, "proj": proj, "comb": comb}


def lp_projection(params: list, rows: list, combos: dict) -> dict:
    """Same projections with scipy.optimize.linprog (HiGHS).  Variables theta = shift + scale * u: probed
    coordinates are centred on their probe reading and scaled by their half-width (the probe rows become the
    bounds -1 <= u <= 1), unprobed ones scaled by eps_Q / c_Q; the Q slab is a two-sided row scaled by 1/eps_Q.
    Feasibility is solved first; an unbounded end (HiGHS status 3) is reported as +-inf."""
    n = len(params)
    idx = {p: j for j, p in enumerate(params)}
    shift, scale = np.zeros(n), np.ones(n)
    bounds: list = [(None, None)] * n
    q = None
    for r in rows:
        if r["name"] == "Q":
            q = r
            continue
        j = idx[r["name"]]
        c = PROBE_COEF[r["name"]]
        shift[j], scale[j], bounds[j] = r["y"] / c, r["eps"] / c, (-1.0, 1.0)
    a_ub = b_ub = None
    if q is not None:
        for p in params:
            if bounds[idx[p]] == (None, None):
                scale[idx[p]] = q["eps"] / Q_COEF[p]
        row = np.array([Q_COEF[p] for p in params]) * scale / q["eps"]
        nz = np.abs(row[row != 0.0])
        if nz.min() <= HIGHS_SMALL_MATRIX_VALUE:
            raise RuntimeError(f"LP row entry {nz.min():.3e} would be dropped by HiGHS")
        centre = (q["y"] - float(np.dot([Q_COEF[p] for p in params], shift))) / q["eps"]
        a_ub, b_ub = np.vstack([row, -row]), np.array([centre + 1.0, 1.0 - centre])

    def solve(cvec):
        return linprog(cvec, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs", options=HIGHS_OPTIONS)

    feas = solve(np.zeros(n))
    if feas.status == 2:
        return {"feasible": False, "proj": None, "comb": None}
    if feas.status != 0:
        raise RuntimeError(f"LP feasibility status {feas.status}: {feas.message}")

    def extreme(weights: dict) -> tuple[float, float]:
        g = np.array([weights.get(p, 0.0) for p in params]) * scale
        ends = []
        for sense in (1.0, -1.0):                       # minimum, then maximum
            res = solve(sense * g)
            if res.status == 3:
                ends.append(-sense * math.inf)
            elif res.status == 0:
                theta = shift + scale * res.x
                ends.append(float(sum(weights.get(p, 0.0) * theta[idx[p]] for p in params)))
            else:
                raise RuntimeError(f"LP status {res.status}: {res.message}")
        return ends[0], ends[1]

    return {"feasible": True, "proj": {p: extreme({p: 1.0}) for p in params},
            "comb": {name: extreme(w) for name, w in combos.items()}}


def interval_diff(a, b) -> float:
    d = 0.0
    for u, v in zip(a, b):
        if math.isinf(u) or math.isinf(v):
            if u != v:
                return math.inf
        else:
            d = max(d, abs(u - v))
    return d


def invert(rows: list, applicable: bool, world: str, removed: str | None) -> dict:
    """CW (theta = a, b, c) or OW (theta = a, b, c, k) inversion of the given rows; LP primary, closed form
    cross-check (feasibility, every projection, every combination and every exclusion decision)."""
    params = list(SOURCES) + ([PRE] if world == "OW" else [])
    out = {"world": world, "params": params, "rows": [r["name"] for r in rows], "applicable": applicable,
           "status": "NOT_APPLICABLE", "feasible": None, "rank": None, "lp": None, "cf": None,
           "support": [], "conditional": None, "combo": None, "r_only_with_k": None,
           "lp_cf_max_abs_diff": 0.0, "lp_cf_agree": True}
    if not applicable:
        return out
    combos = {}
    q_present = any(r["name"] == "Q" for r in rows)
    if world == "OW" and removed in SOURCES and q_present:
        combos["R+PRE"] = {removed: Q_COEF[removed] / 2.0, PRE: Q_COEF[PRE] / 2.0}
        probed = {r["name"] for r in rows if r["name"] != "Q"}
        # R enters the retained rows only in the Q row, and within it only together with the unprobed k
        out["r_only_with_k"] = all(p in probed for p in SOURCES if p != removed)
    a = np.array([[Q_COEF[p] for p in params] if r["name"] == "Q"
                  else [PROBE_COEF[p] if p == r["name"] else 0.0 for p in params] for r in rows])
    out["rank"] = int(np.linalg.matrix_rank(a))
    cf = closed_form(params, rows, combos)
    lp = lp_projection(params, rows, combos)
    out["cf"], out["lp"] = cf, lp
    agree = cf["feasible"] == lp["feasible"]
    dmax = 0.0
    if agree and lp["feasible"]:
        for p in params:
            dmax = max(dmax, interval_diff(lp["proj"][p], cf["proj"][p]))
            agree &= _excludes0(lp["proj"][p]) == _excludes0(cf["proj"][p])
        for name in combos:
            dmax = max(dmax, interval_diff(lp["comb"][name], cf["comb"][name]))
            agree &= _excludes0(lp["comb"][name]) == _excludes0(cf["comb"][name])
    agree &= dmax <= LP_CF_TOL
    out["lp_cf_max_abs_diff"], out["lp_cf_agree"] = dmax, bool(agree)
    out["feasible"] = lp["feasible"]
    out["status"] = ("MODEL_INCONSISTENT" if not lp["feasible"] else
                     "BOUNDED" if out["rank"] == len(params) else "RANK_DEFICIENT")
    if lp["feasible"]:
        out["support"] = [p for p in params if _excludes0(lp["proj"][p])]
        if combos:
            out["combo"] = lp["comb"]["R+PRE"]
            if out["r_only_with_k"] and _excludes0(out["combo"]):
                out["conditional"] = [removed, PRE]
    return out


def study_identifiability(rows: list, inv: dict) -> dict:
    """The 'identifiability' fields of a CW inversion (closed-form intervals, as the study reported them)."""
    if not inv["applicable"]:
        return {"status": "NOT_APPLICABLE", "supported_causes": []}
    a = np.array([[Q_COEF[p] for p in SOURCES] if r["name"] == "Q"
                  else [PROBE_COEF[p] if p == r["name"] else 0.0 for p in SOURCES] for r in rows])
    s = np.linalg.svd(a, compute_uv=False)
    rank = int(np.linalg.matrix_rank(a))
    sw = np.linalg.svd(a / np.array([r["eps"] for r in rows])[:, None], compute_uv=False)
    return {"status": inv["status"], "rows": inv["rows"], "rank": rank, "singular_values": s.tolist(),
            "condition_number": float(s[0] / s[-1]) if rank == 3 else None,
            "budget_scaled_sigma_min": float(sw[-1]) if rank == 3 else 0.0,
            "log_parameter_intervals": {p: [v if math.isfinite(v) else None for v in inv["cf"]["proj"][p]]
                                        for p in SOURCES},
            "supported_causes": list(inv["support"])}


# ================================================================ methods ==
def cw_study_record(obs: dict, policy: str, bud: Budgets, guards: dict) -> dict:
    """One CW prediction record of the study's policies (policy = full or minus_*), rewritten."""
    removed = policy[len("minus_"):] if policy.startswith("minus_") else None
    comps = {k: v for k, v in obs["components"].items() if k != removed}
    diag = study_diagnostic(obs, comps, bud, guards)
    rec = {"decision": "UNDECIDED", "localized": [], "classification": diag["classification"],
           "diagnostic": diag, "identifiability": None, "inversion": None, "branch_outside_scalar_model": False}
    if diag["input_status"] != "VALID" or not diag["comparable"]:
        return rec
    if policy == "norm_only":
        if "normalization" in comps:
            alarm = comps["normalization"] > bud.tau_c
            rec["decision"], rec["localized"] = ("ALARM", ["normalization"]) if alarm else ("COMPATIBLE", [])
        return rec
    if policy == "grid_only":
        if diag["grid_ok"]:
            rec["decision"] = "ALARM" if abs(diag["grid"]["slope"]) > guards["slope_absolute"] else "COMPATIBLE"
        return rec
    if policy == "reference_only":
        rec["decision"] = "COMPATIBLE" if diag["reference_ok"] else "ALARM"
        return rec
    rec["decision"] = study_decision(obs, diag, removed, comps)
    rows, applicable = inversion_rows(obs, removed, diag, bud, guards)
    inv = invert(rows, applicable, "CW", removed)
    rec["inversion"] = inv
    rec["identifiability"] = study_identifiability(rows, inv)
    localized = list(inv["support"])
    if float(comps.get("branch", 0.0)) > bud.tau_c:
        localized = sorted(set(localized) | {"branch"})
        rec["branch_outside_scalar_model"] = True
    rec["localized"] = localized
    return rec


def method_or(obs: dict, condition: str, bud: Budgets, guards: dict) -> dict:
    removed = REMOVED[condition]
    comps = {k: v for k, v in obs["components"].items() if k != removed}
    rec = {"method": "OR", "decision": "UNDECIDED", "localized": [], "conditional": None, "classification": None,
           "exceeded": [], "ref_err": None, "ref_alarm": None, "inversion": None}
    try:
        _, q, ref = check_inputs(obs, comps)
    except _Invalid as exc:
        rec["classification"] = exc.classification
        return rec
    if not comparable(obs):
        rec["classification"] = "INCOMPARABLE_SEQUENCE"
        return rec
    exceeded = sorted(k for k, v in comps.items() if float(v) > bud.tau_c)
    absb = guards["absolute_budget_nm2"]
    if abs(ref) > absb:
        rec["ref_err"] = float(abs(q[-1] / ref - 1.0))
        rec["ref_alarm"] = rec["ref_err"] > bud.tau_g
    else:                                              # zero-reference criterion of the study (not exercised)
        rec["ref_err"] = float(np.max(np.abs(q - ref)))
        rec["ref_alarm"] = rec["ref_err"] > absb
    rec["exceeded"] = exceeded
    alarm = bool(exceeded) or rec["ref_alarm"]
    rec["decision"] = "ALARM" if alarm else "COMPATIBLE"
    rec["localized"] = exceeded
    return rec


def method_cw(obs: dict, condition: str, bud: Budgets, guards: dict) -> tuple[dict, dict]:
    full = cw_study_record(obs, STUDY_POLICY[condition], bud, guards)
    rec = {"method": "CW", "decision": full["decision"], "localized": sorted(full["localized"]),
           "conditional": None, "classification": full["classification"], "inversion": full["inversion"]}
    return rec, full


def method_ow(obs: dict, condition: str, bud: Budgets, guards: dict, cw_full: dict) -> dict:
    removed = REMOVED[condition]
    comps = {k: v for k, v in obs["components"].items() if k != removed}
    diag = cw_full["diagnostic"]
    rec = {"method": "OW", "decision": cw_full["decision"], "localized": [], "conditional": None,
           "classification": diag["classification"], "inversion": None}
    if diag["input_status"] != "VALID" or not diag["comparable"]:
        return rec
    rows, applicable = inversion_rows(obs, removed, diag, bud, guards)
    inv = invert(rows, applicable, "OW", removed)
    rec["inversion"] = inv
    localized = set(inv["support"])
    if float(comps.get("branch", 0.0)) > bud.tau_c:
        localized |= {"branch"}
    rec["localized"] = sorted(localized)
    rec["conditional"] = inv["conditional"]
    return rec


def evaluate_point(observations: list, bud: Budgets, guards: dict) -> dict:
    """All OR / CW / OW records of one budget point: {(id, condition, method): record}.  Truth-blind."""
    out = {}
    for obs in observations:
        for cond in CONDITIONS:
            out[(obs["id"], cond, "OR")] = method_or(obs, cond, bud, guards)
            cw, cw_full = method_cw(obs, cond, bud, guards)
            out[(obs["id"], cond, "CW")] = cw
            out[(obs["id"], cond, "OW")] = method_ow(obs, cond, bud, guards, cw_full)
    return out


# ================================================================ prediction digest ==
def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def prediction_digest(points: dict) -> str:
    """SHA256 over every decision / localization / conditional attribution / interval of every point."""
    core = []
    for label in sorted(points):
        for key in sorted(points[label]):
            r = points[label][key]
            inv = r.get("inversion") or {}
            core.append([label, *key, r["decision"], r["localized"], r["conditional"],
                         (inv.get("lp") or {}).get("proj"), (inv.get("lp") or {}).get("comb")])
    blob = json.dumps(_jsonable(core), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(blob)


# ================================================================ scoring (truth) ==
def score_one(rec: dict, t: dict) -> dict:
    active = set(t["causes"])
    dec = rec["decision"]
    if not t["comparable"]:
        outcome = "incomparable_" + dec.lower()
    elif t["has_fault"]:
        outcome = {"ALARM": "detected", "COMPATIBLE": "missed", "UNDECIDED": "fault_undecided"}[dec]
    else:
        outcome = {"ALARM": "false_alarm", "COMPATIBLE": "clean_compatible", "UNDECIDED": "clean_undecided"}[dec]
    claimed = set(rec["localized"])
    if not claimed:
        loc = "NONE"
    elif claimed - active:
        loc = "MIS"
    elif claimed == active:
        loc = "FULL"
    else:
        loc = "PARTIAL"
    cond = None
    if rec["conditional"]:
        cond = "COND-CORRECT" if set(rec["conditional"]) & active else "COND-VACUOUS"
    return {"outcome": outcome, "loc": loc, "cond": cond}


OUTCOMES = ("detected", "missed", "fault_undecided", "false_alarm", "clean_compatible", "clean_undecided",
            "incomparable_undecided", "incomparable_compatible", "incomparable_alarm")


def aggregate(rows: list) -> dict:
    oc = Counter(r["outcome"] for r in rows)
    faults = [r for r in rows if r["has_fault"]]
    lc = Counter(r["loc"] for r in faults)
    joint = Counter(r["loc"] for r in faults if r["decision"] == "ALARM")
    cond = [r for r in rows if r["cond"]]
    return {"n": len(rows), "faults": len(faults), **{k: oc[k] for k in OUTCOMES},
            "FULL": lc["FULL"], "PARTIAL": lc["PARTIAL"], "MIS": lc["MIS"], "NONE": lc["NONE"],
            "ALARM_and_FULL": joint["FULL"], "ALARM_and_PARTIAL": joint["PARTIAL"],
            "clean_localized": sum(1 for r in rows if not r["has_fault"] and r["localized"]),
            "localized_without_alarm": sum(1 for r in rows if r["localized"] and r["decision"] != "ALARM"),
            "conditional": len(cond),
            "COND_CORRECT": sum(r["cond"] == "COND-CORRECT" for r in cond),
            "COND_VACUOUS": sum(r["cond"] == "COND-VACUOUS" for r in cond),
            "conditional_on_ALARM": sum(r["decision"] == "ALARM" for r in cond),
            "conditional_on_non_ALARM": sum(r["decision"] != "ALARM" for r in cond),
            "MODEL_INCONSISTENT": sum(r["inv_status"] == "MODEL_INCONSISTENT" for r in rows),
            "lp_cf_disagreements": sum(not r["lp_cf_agree"] for r in rows)}


def score_point(point_records: dict, truth: dict) -> list:
    rows = []
    for (oid, cond, method), rec in point_records.items():
        t = truth[oid]
        s = score_one(rec, t)
        inv = rec.get("inversion") or {}
        rows.append({"id": oid, "cohort": t["cohort"], "family": t["family"], "context": t["context"],
                     "consumer": t["consumer"], "condition": cond, "method": method,
                     "true_causes": sorted(t["causes"]), "has_fault": bool(t["causes"]),
                     "comparable": t["comparable"], "decision": rec["decision"], "localized": rec["localized"],
                     "conditional": rec["conditional"], "inv_status": inv.get("status", ""),
                     "lp_cf_agree": inv.get("lp_cf_agree", True), **s})
    return rows


def tables_for(scored: list) -> dict:
    tab = {}
    for cohort in COHORTS:
        for cond in CONDITIONS:
            for method in METHODS:
                sel = [r for r in scored if r["cohort"] == cohort and r["condition"] == cond and r["method"] == method]
                tab[f"{cohort}|{cond}|{method}"] = aggregate(sel)
    for fam in BOUNDARY_FAMILIES:
        for cond in CONDITIONS:
            for method in METHODS:
                sel = [r for r in scored if r["family"] == fam and r["condition"] == cond and r["method"] == method]
                tab[f"family:{fam}|{cond}|{method}"] = aggregate(sel)
    return tab


# ================================================================ expectations (fixed before the run) ==
def _get(tab: dict, cohort: str, cond: str, method: str, key: str):
    return tab[f"{cohort}|{cond}|{method}"][key]


def _fam(tab: dict, fam: str, cond: str, method: str, key: str):
    return tab[f"family:{fam}|{cond}|{method}"][key]


def expectations_baseline(tab: dict, baseline_rec: dict, scored: list) -> list:
    rows = []

    def add(qty, expected, observed, nature, note=""):
        rows.append({"section": "baseline", "quantity": qty, "expected": expected, "observed": observed,
                     "nature": nature, "agree": expected == observed, "note": note})

    cs = list(CONDITIONS)
    add("OR main: detected full/-N/-W/-O", [72, 70, 70, 70],
        [_get(tab, "primary", c, "OR", "detected") for c in cs], "KNOWN")
    add("OR main: FULL full/-N/-W/-O", [72, 36, 24, 36], [_get(tab, "primary", c, "OR", "FULL") for c in cs], "KNOWN")
    add("OR main: false alarms full/-N/-W/-O", [0, 0, 0, 0],
        [_get(tab, "primary", c, "OR", "false_alarm") for c in cs], "KNOWN")
    add("OR main: mislocalization (MIS) full/-N/-W/-O", [0, 0, 0, 0],
        [_get(tab, "primary", c, "OR", "MIS") for c in cs], "KNOWN")
    add("CW main: FULL full/-N/-W/-O (as reported by the study)", [72, 66, 62, 62],
        [_get(tab, "primary", c, "CW", "FULL") for c in cs], "KNOWN")
    add("OW main: unconditional FULL full/-N/-W/-O (= OR)", [72, 36, 24, 36],
        [_get(tab, "primary", c, "OW", "FULL") for c in cs], "PREDICTION")
    add("OW main: conditional attributions -N/-W/-O", [30, 38, 26],
        [_get(tab, "primary", c, "OW", "conditional") for c in cs[1:]], "PREDICTION")
    add("OW main: conditional attributions all COND-CORRECT (COND-VACUOUS count -N/-W/-O)", [0, 0, 0],
        [_get(tab, "primary", c, "OW", "COND_VACUOUS") for c in cs[1:]], "PREDICTION")
    kin = [r for r in scored if r["cohort"] == "primary" and r["condition"] == "full" and r["method"] == "OW"]
    n_contains = sum(1 for r in kin
                     if not _excludes0(baseline_rec[(r["id"], "full", "OW")]["inversion"]["lp"]["proj"][PRE]))
    add("OW full probes: main cases whose k interval contains 0", 96, n_contains, "PREDICTION")
    for c in cs:
        add(f"boundary {c}: OR branch (incoherent) detected / cancellation (missing W,O) missed",
            [12, 12], [_fam(tab, "incoherent_channels", c, "OR", "detected"),
                       _fam(tab, "missing_two_probes", c, "OR", "missed")], "KNOWN")
        add(f"boundary {c}: CW branch detected / cancellation undecided",
            [12, 12], [_fam(tab, "incoherent_channels", c, "CW", "detected"),
                       _fam(tab, "missing_two_probes", c, "CW", "fault_undecided")], "KNOWN")
        add(f"boundary {c}: identity change undecided OR/CW", [12, 12],
            [_fam(tab, "changing_identity", c, m, "incomparable_undecided") for m in ("OR", "CW")], "KNOWN")
        add(f"boundary {c}: identity change undecided OW", 12,
            _fam(tab, "changing_identity", c, "OW", "incomparable_undecided"), "PREDICTION")
        same = all(baseline_rec[(r["id"], c, "OW")]["decision"] == baseline_rec[(r["id"], c, "CW")]["decision"]
                   and baseline_rec[(r["id"], c, "OW")]["localized"] == baseline_rec[(r["id"], c, "CW")]["localized"]
                   and not baseline_rec[(r["id"], c, "OW")]["conditional"]
                   for r in scored if r["cohort"] == "boundary" and r["condition"] == c and r["method"] == "OW")
        add(f"boundary {c}: OW = CW case by case (decision, localization, no conditional)", True, same, "PREDICTION")
        add(f"boundary {c}: coarse-grid normal compatible CW (of 12)", 12,
            _fam(tab, "coarse_clean", c, "CW", "clean_compatible"), "KNOWN", "as reported by the study")
        add(f"boundary {c}: coarse-grid normal compatible OR / OW (of 12)", [12, 12],
            [_fam(tab, "coarse_clean", c, m, "clean_compatible") for m in ("OR", "OW")], "PREDICTION")
    return rows


def expectations_sweeps(sweeps: dict, baseline_label: str) -> list:
    """sweeps: label -> {"budgets": Budgets, "tab": tables, "scored": scored rows, "records": records}."""
    rows = []

    def add(qty, expected, observed, nature, note=""):
        rows.append({"section": "sweeps", "quantity": qty, "expected": expected, "observed": observed,
                     "nature": nature, "agree": expected == observed, "note": note})

    lab_a = {t: f"A:tau_g={t}" for t in SWEEP_A_TAU_G}
    lab_b = {t: f"B:tau_c={t}" for t in SWEEP_B_TAU_C}
    rm = ("-N", "-W", "-O")
    known_cw = {"0.001": [72, 72, 72], "0.02": [66, 62, 62], "0.10": [54, 40, 44]}
    for t, exp in known_cw.items():
        add(f"CW main FULL -N/-W/-O at tau_g={t}", exp,
            [_get(sweeps[lab_a[t]]["tab"], "primary", c, "CW", "FULL") for c in rm], "KNOWN")
    for t in SWEEP_A_TAU_G:
        add(f"OR main FULL -N/-W/-O at tau_g={t}", [36, 24, 36],
            [_get(sweeps[lab_a[t]]["tab"], "primary", c, "OR", "FULL") for c in rm], "KNOWN")
    for m in ("OR", "CW"):
        add(f"coarse-grid normal false alarms at tau_g=0.001, {m} (full / -N / -W / -O)", [10, 10, 10, 10],
            [_fam(sweeps[lab_a["0.001"]]["tab"], "coarse_clean", c, m, "false_alarm") for c in CONDITIONS], "KNOWN")
    # predictions
    for c in CONDITIONS:
        seq = [_get(sweeps[lab_a[t]]["tab"], "primary", c, "CW", "FULL") for t in SWEEP_A_TAU_G]
        add(f"CW main FULL non-increasing in tau_g ({c}): sequence", True,
            all(a >= b for a, b in zip(seq, seq[1:])), "PREDICTION", f"sequence {seq}")

    def loc_map(label, method):
        return {(k[0], k[1]): tuple(v["localized"]) for k, v in sweeps[label]["records"].items() if k[2] == method}

    base_or = loc_map(lab_a["0.02"], "OR")
    add("OR localization identical at every tau_g (per case, all 144 x 4)", True,
        all(loc_map(lab_a[t], "OR") == base_or for t in SWEEP_A_TAU_G), "PREDICTION")
    for label in [*lab_a.values(), *lab_b.values()]:
        orl, owl = loc_map(label, "OR"), loc_map(label, "OW")
        cohort = {r["id"]: r["cohort"] for r in sweeps[label]["scored"]}
        diff_main = sorted({k for k in orl if orl[k] != owl[k] and cohort[k[0]] == "primary"})
        diff_bnd = sorted({k for k in orl if orl[k] != owl[k] and cohort[k[0]] == "boundary"})
        add(f"OW unconditional localization = OR per case at {label} (main group)", 0, len(diff_main), "PREDICTION")
        add(f"OW unconditional localization = OR per case at {label} (boundary group)", 0, len(diff_bnd),
            "PREDICTION", "prediction text does not restrict to the main group")
    for label in [*lab_a.values(), *lab_b.values()]:
        b = sweeps[label]["budgets"]
        fa = [_fam(sweeps[label]["tab"], "coarse_clean", c, m, "false_alarm") for m in METHODS for c in CONDITIONS]
        if b.eps_q >= MAX_COARSE_DISC_STATED:
            add(f"coarse-grid normal false alarms = 0 at {label} (eps_Q={b.eps_q} >= 0.00408717), OR/CW/OW x 4 cond",
                0, sum(fa), "PREDICTION")
        else:
            rows.append({"section": "sweeps", "quantity": f"coarse-grid normal false alarms at {label} "
                         f"(eps_Q={b.eps_q} < 0.00408717; may appear), OR/CW/OW x full/-N/-W/-O",
                         "expected": "may be > 0", "observed": fa, "nature": "PREDICTION", "agree": True,
                         "note": "only a permission, not a count"})
    base = sweeps[baseline_label]["scored"]
    base_det = {(r["id"], r["condition"], r["method"]) for r in base if r["cohort"] == "primary" and r["outcome"] == "detected"}
    for t in SWEEP_B_TAU_C:
        sc = sweeps[lab_b[t]]["scored"]
        det = {(r["id"], r["condition"], r["method"]) for r in sc if r["cohort"] == "primary" and r["outcome"] == "detected"}
        if float(t) <= 1e-4:
            add(f"main-group detections unchanged at tau_c={t} (cases leaving / entering the detected set)", [0, 0],
                [len(base_det - det), len(det - base_det)], "PREDICTION")
        weak = [r for r in sc if r["family"] == "weak_operator"]
        add(f"weak faults (2e-12) missed at tau_c={t}, all methods x 4 conditions", len(weak),
            sum(r["outcome"] == "missed" for r in weak), "PREDICTION")
    sc = sweeps[lab_b["1e-2"]]["scored"]
    ctx1 = [r for r in sc if r["cohort"] == "primary" and r["context"] == 0 and r["has_fault"]]
    add("tau_c=1e-2: context-1 (t=0.003) main-group faults detected by any method/condition", 0,
        sum(r["outcome"] == "detected" for r in ctx1), "PREDICTION")
    other_lost = sorted({(r["family"], r["context"] + 1, r["condition"], r["method"]) for r in sc
                         if r["cohort"] == "primary" and r["context"] != 0 and r["has_fault"]
                         and (r["id"], r["condition"], r["method"]) in base_det and r["outcome"] != "detected"})
    rows.append({"section": "sweeps", "quantity": "tau_c=1e-2: detections lost outside context 1 (informational)",
                 "expected": "(not predicted)", "observed": [list(x) for x in other_lost], "nature": "INFO",
                 "agree": True, "note": ""})
    return rows


# ================================================================ identity ==
def identity() -> dict:
    """Run environment."""
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
            "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform()}


# ================================================================ output helpers ==
def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _iv(inv: dict | None, key: str, which: str = "proj"):
    if not inv or not inv.get("lp") or not inv["lp"].get("feasible"):
        return ("", "")
    d = inv["lp"][which]
    if which == "proj":
        if key not in d:
            return ("", "")
        return tuple(_fmt(x) for x in d[key])
    return tuple(_fmt(x) for x in d[key]) if key in d else ("", "")


def write_records_csv(path: Path, observations: list, recs: dict, scored: list) -> None:
    obs_by = {o["id"]: o for o in observations}
    srow = {(r["id"], r["condition"], r["method"]): r for r in scored}
    cols = ["id", "cohort", "family", "context_1based", "consumer", "condition", "method",
            "y_Q_ln_Qf_over_Qref", "y_N", "y_W", "y_O", "comp_normalization", "comp_measure", "comp_operator",
            "comp_branch", "ref_rel_err", "decision", "classification", "or_exceeded", "rows_used", "inv_status",
            "a_lo", "a_hi", "b_lo", "b_hi", "c_lo", "c_hi", "k_lo", "k_hi", "comb_lo", "comb_hi",
            "localized", "conditional", "lp_cf_max_abs_diff", "lp_cf_agree",
            "true_causes", "outcome", "localization", "conditional_score"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for oid in sorted(obs_by):
            o = obs_by[oid]
            yq = math.log(o["q"][-1] / o["reference"]) if o["q"][-1] > 0 and o["reference"] > 0 else None
            pr = o["probe_ratios"]
            for cond in CONDITIONS:
                for m in METHODS:
                    r = recs[(oid, cond, m)]
                    s = srow[(oid, cond, m)]
                    inv = r.get("inversion")
                    comb = ("", "")
                    if inv and inv.get("lp") and inv["lp"].get("feasible") and "R+PRE" in (inv["lp"].get("comb") or {}):
                        comb = tuple(_fmt(x) for x in inv["lp"]["comb"]["R+PRE"])
                    w.writerow([oid, s["cohort"], s["family"], s["context"] + 1, s["consumer"], cond, m,
                                _fmt(yq), *[_fmt(math.log(pr[k])) if k in pr else "" for k in SOURCES],
                                *[_fmt(o["components"].get(k)) for k in (*SOURCES, "branch")],
                                _fmt(abs(o["q"][-1] / o["reference"] - 1.0)),
                                r["decision"], r.get("classification") or "",
                                "+".join(r.get("exceeded", [])) if m == "OR" else "",
                                "+".join(inv["rows"]) if inv else "", inv["status"] if inv else "",
                                *_iv(inv, "normalization"), *_iv(inv, "measure"), *_iv(inv, "operator"),
                                *_iv(inv, PRE), *comb,
                                "+".join(r["localized"]), "(+)".join(r["conditional"]) if r["conditional"] else "",
                                _fmt(inv["lp_cf_max_abs_diff"]) if inv else "", inv["lp_cf_agree"] if inv else "",
                                "+".join(s["true_causes"]), s["outcome"], s["loc"], s["cond"] or ""])


SUMMARY_KEYS = ("n", "faults", "detected", "missed", "fault_undecided", "false_alarm", "clean_compatible",
                "clean_undecided", "incomparable_undecided", "FULL", "PARTIAL", "MIS", "NONE", "ALARM_and_FULL",
                "clean_localized", "localized_without_alarm", "conditional", "COND_CORRECT", "COND_VACUOUS",
                "conditional_on_non_ALARM", "MODEL_INCONSISTENT", "lp_cf_disagreements")


def write_table_csv(path: Path, tabs: dict) -> None:
    """tabs: point label -> tables (cohort/family x condition x method)."""
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["point", "group", "condition", "method", *SUMMARY_KEYS])
        for label, tab in tabs.items():
            for key, agg in tab.items():
                group, cond, method = key.split("|")
                w.writerow([label, group, cond, method, *[agg[k] for k in SUMMARY_KEYS]])


def main_table_text(tab: dict, title: str) -> list:
    lines = [title, "  method cond | det miss und  FA | FULL PART MIS NONE | cond (CORRECT/VACUOUS) | clean-loc loc-w/o-alarm"]
    for cohort in COHORTS:
        lines.append(f"  -- {cohort} --")
        for m in METHODS:
            for c in CONDITIONS:
                a = tab[f"{cohort}|{c}|{m}"]
                lines.append(f"  {m:6} {c:4} | {a['detected']:3d} {a['missed']:4d} {a['fault_undecided']:3d} "
                             f"{a['false_alarm']:3d} | {a['FULL']:4d} {a['PARTIAL']:4d} {a['MIS']:3d} {a['NONE']:4d} | "
                             f"{a['conditional']:3d} ({a['COND_CORRECT']}/{a['COND_VACUOUS']}) | "
                             f"{a['clean_localized']:3d} {a['localized_without_alarm']:3d}")
    return lines


def boundary_text(tab: dict) -> list:
    lines = ["BOUNDARY GROUP per family (baseline point): outcome counts; localization classes on fault rows"]
    for fam in BOUNDARY_FAMILIES:
        for m in METHODS:
            for c in CONDITIONS:
                a = tab[f"family:{fam}|{c}|{m}"]
                oc = {k: a[k] for k in OUTCOMES if a[k]}
                lines.append(f"  {fam:20} {m:3} {c:4} {oc} FULL={a['FULL']} PART={a['PARTIAL']} "
                             f"MIS={a['MIS']} NONE={a['NONE']} cond={a['conditional']}")
    return lines


def sweep_text(sweeps: dict) -> list:
    lines = ["BUDGET SWEEPS (descriptive). main = 96-case group; coarseFA = false alarms on the 12 coarse-grid normal cases (full / -N / -W / -O)"]
    for label, sw in sweeps.items():
        b = sw["budgets"]
        tab = sw["tab"]
        lines.append(f"  {label:16} tau_g={b.tau_g:<6g} eps_Q={b.eps_q:<9g} tau_c={b.tau_c:<7g} eps_c={b.eps_c:g}")
        for m in METHODS:
            det = "/".join(str(_get(tab, "primary", c, m, "detected")) for c in CONDITIONS)
            full = "/".join(str(_get(tab, "primary", c, m, "FULL")) for c in CONDITIONS)
            part = "/".join(str(_get(tab, "primary", c, m, "PARTIAL")) for c in CONDITIONS)
            mis = "/".join(str(_get(tab, "primary", c, m, "MIS")) for c in CONDITIONS)
            fa = "/".join(str(_get(tab, "primary", c, m, "false_alarm")) for c in CONDITIONS)
            cond = "/".join(f"{_get(tab, 'primary', c, m, 'conditional')}"
                            f"[{_get(tab, 'primary', c, m, 'COND_VACUOUS')}v]" for c in CONDITIONS[1:])
            cfa = "/".join(str(_fam(tab, "coarse_clean", c, m, "false_alarm")) for c in CONDITIONS)
            clo = "/".join(str(_fam(tab, "coarse_clean", c, m, "clean_localized")) for c in CONDITIONS)
            bcond = "/".join(f"{_get(tab, 'boundary', c, m, 'conditional')}"
                             f"[{_get(tab, 'boundary', c, m, 'COND_VACUOUS')}v]" for c in CONDITIONS[1:])
            minc = "/".join(str(_get(tab, "primary", c, m, "MODEL_INCONSISTENT")
                                + _get(tab, "boundary", c, m, "MODEL_INCONSISTENT")) for c in CONDITIONS)
            lines.append(f"    {m:3} det {det:15} FULL {full:15} PART {part:13} MIS {mis:9} mainFA {fa:9} "
                         f"cond(-N/-W/-O) {cond:22} coarseFA {cfa:11} coarseLoc {clo:11} bndCond {bcond:18} "
                         f"MODEL_INCONS {minc}")
    lines.append("    (cond: count[COND-VACUOUS]; coarseLoc: coarse-grid normal cases given an unconditional "
                 "localization; MODEL_INCONS: empty feasible set, both groups)")
    return lines


# ================================================================ driver ==
def run(out_dir: Path = OUT) -> dict:
    t_start = time.perf_counter()
    ident = identity()                                   # before anything is written
    timings = {}
    pkg = StudyData(DATA)
    observations = pkg.observations()
    cfg = pkg.settings()
    baseline = baseline_budgets(cfg)
    guards = guards_of(cfg)
    points = sweep_points(baseline)
    assert len(observations) == 144 and len({o["id"] for o in observations}) == 144

    # ---- stage 1: predictions (truth-blind) ----
    t0 = time.perf_counter()
    records = {"baseline": evaluate_point(observations, baseline, guards)}
    for b in points:
        records[b.label] = evaluate_point(observations, b, guards)
    timings["predict_s"] = time.perf_counter() - t0
    digest = prediction_digest(records)
    dup = {"A:tau_g=0.02": "baseline", "B:tau_c=1e-10": "baseline"}
    for lab, ref in dup.items():
        b = next(p for p in points if p.label == lab)
        assert (b.tau_g, b.eps_q, b.tau_c, b.eps_c) == (baseline.tau_g, baseline.eps_q, baseline.tau_c, baseline.eps_c), lab
        same = all(records[lab][k]["decision"] == records[ref][k]["decision"]
                   and records[lab][k]["localized"] == records[ref][k]["localized"]
                   and records[lab][k]["conditional"] == records[ref][k]["conditional"] for k in records[ref])
        assert same, f"{lab} does not reproduce the baseline point"

    # ---- stage 2: scoring (the only reader of truth.json) ----
    pkg.unlock_truth(digest)
    truth = pkg.truth()
    member_sha = dict(pkg.member_sha256)
    assert set(truth) == {o["id"] for o in observations}

    budgets_by = {"baseline": baseline, **{b.label: b for b in points}}
    sweeps = {}
    for label, recs in records.items():
        scored = score_point(recs, truth)
        sweeps[label] = {"budgets": budgets_by[label], "records": recs, "scored": scored, "tab": tables_for(scored)}
    baseline_tab = sweeps["baseline"]["tab"]

    # LP / closed-form check over every inversion (all points, CW and OW)
    n_inv = n_bad = 0
    max_diff = 0.0
    for recs in records.values():
        for r in recs.values():
            inv = r.get("inversion")
            if inv and inv["applicable"]:
                n_inv += 1
                n_bad += not inv["lp_cf_agree"]
                max_diff = max(max_diff, inv["lp_cf_max_abs_diff"])
    lpcf = {"inversions_all_points": n_inv, "disagreements": n_bad, "max_abs_diff": max_diff,
            "tolerance": LP_CF_TOL, "highs_options": HIGHS_OPTIONS}

    exp23 = expectations_baseline(baseline_tab, records["baseline"], sweeps["baseline"]["scored"])
    exp24 = expectations_sweeps(sweeps, "baseline")

    # consistency facts the reading of the counts relies on (all points)
    guard_fail = sorted(o["id"] for o in observations
                        if comparable(o) and not study_diagnostic(o, dict(o["components"]), baseline, guards)["grid_ok"])
    cells = [(label, key, agg) for label, sw in sweeps.items() for key, agg in sw["tab"].items()
             if not key.startswith("family:")]
    consistency = {
        "comparable_observations": sum(comparable(o) for o in observations),
        "comparable_observations_failing_grid_guard": guard_fail,
        "cells_FULL_ne_ALARM_and_FULL": [(lab, k) for lab, k, a in cells if a["FULL"] != a["ALARM_and_FULL"]],
        "conditional_on_non_ALARM_total": sum(a["conditional_on_non_ALARM"] for _, _, a in cells),
        "localized_without_alarm_total": sum(a["localized_without_alarm"] for _, _, a in cells),
        "note": "grid guard thresholds are not swept, so the OR reference check (not guard-gated, literal) and the study's guard-gated cascade see the same Q evidence whenever the guard failure list is empty"}
    same_set = {}
    for label, sw in sweeps.items():                      # OW "R (+) PRE" set vs CW localizing the removed R
        recs = sw["records"]
        same_set[label] = all((r["conditional"] is not None) == (REMOVED[cond] in recs[(oid, cond, "CW")]["localized"])
                              for (oid, cond, method), r in recs.items() if method == "OW" and cond != "full")
    consistency["ow_conditional_set_equals_cw_removed_source_localization"] = same_set
    consistency["baseline_ow_conditional_by_family"] = {
        f"{c}|{f}": n for (c, f), n in sorted(Counter((r["condition"], r["family"]) for r in sweeps["baseline"]["scored"]
                                                      if r["method"] == "OW" and r["cond"]).items())}
    for fam in ("clean", "coarse_clean"):                 # discretization deviation of the normal cases
        sel = [o for o in observations if truth[o["id"]]["family"] == fam]
        consistency[f"{fam}_max_abs_ln_RQ"] = max(abs(math.log(o["q"][-1] / o["reference"])) for o in sel)
        consistency[f"{fam}_max_abs_RQ_minus_1"] = max(abs(o["q"][-1] / o["reference"] - 1.0) for o in sel)

    # sweep deltas vs the baseline point (records whose decision / localization / conditional changed)
    baseline_scored = {(r["id"], r["condition"], r["method"]): r for r in sweeps["baseline"]["scored"]}
    deltas = []
    for label in [b.label for b in points]:
        for r in sweeps[label]["scored"]:
            f = baseline_scored[(r["id"], r["condition"], r["method"])]
            if (r["decision"], r["localized"], r["conditional"]) != (f["decision"], f["localized"], f["conditional"]):
                deltas.append((label, r, f))

    # k-interval statistics (OW, full probes, baseline point)
    kstats = []
    for r in sweeps["baseline"]["scored"]:
        if r["method"] == "OW" and r["condition"] == "full":
            inv = records["baseline"][(r["id"], "full", "OW")]["inversion"]
            if inv and inv["lp"] and inv["lp"]["feasible"]:
                kstats.append((r["cohort"], r["family"], inv["lp"]["proj"][PRE]))
    kmain = [iv for c, _, iv in kstats if c == "primary"]
    k_summary = {"main_cases": len(kmain), "main_contains_0": sum(not _excludes0(iv) for iv in kmain),
                 "main_max_abs_centre": max(abs(0.5 * (iv[0] + iv[1])) for iv in kmain),
                 "main_min_half_width": min(0.5 * (iv[1] - iv[0]) for iv in kmain),
                 "boundary_with_finite_k": sum(1 for c, _, iv in kstats if c == "boundary" and math.isfinite(iv[0])),
                 "boundary_k_excludes_0": sum(1 for c, _, iv in kstats if c == "boundary" and _excludes0(iv))}

    timings["total_s"] = time.perf_counter() - t_start
    result = {
        "setup": "144-case study data (data/case144); methods OR, CW, OW; conditions full, -N, -W, -O; baseline point and budget sweeps; expectations fixed before the run",
        "identity": ident,
        "inputs": {"data": "data/case144", "files_read_sha256": member_sha,
                   "predictions_digest_before_truth": digest,
                   "truth_opened_after_predictions": True},
        "settings": {"baseline": asdict(baseline), "guards_fixed": guards,
                     "sweep_points": [asdict(b) for b in points],
                     "conditions": list(CONDITIONS), "methods": list(METHODS),
                     "ow_combination": "R (+) PRE interval of (c_Q,R * theta_R + 2k) / 2: 2a+k (-N), b+k (-W), c+k (-O)",
                     "scoring": "scoring semantics of the study; FULL/PARTIAL/MIS/NONE over fault rows (true causes nonempty) regardless of decision, ALARM_and_FULL reported alongside; COND-CORRECT iff {R, PRE} contains a true source (definition used for the 27-case set)"},
        "lp_vs_closed_form": lpcf,
        "consistency": consistency,
        "ow_k_interval_full_probes_baseline": k_summary,
        "baseline_point_tables": baseline_tab,
        "expectations_baseline": exp23,
        "expectations_sweeps": exp24,
        "sweeps": {label: {"budgets": asdict(sw["budgets"]),
                           "main": {f"{c}|{m}": {k: sw["tab"][f"primary|{c}|{m}"][k] for k in SUMMARY_KEYS}
                                    for c in CONDITIONS for m in METHODS},
                           "boundary_coarse_clean": {f"{c}|{m}": {k: sw["tab"][f"family:coarse_clean|{c}|{m}"][k]
                                                                  for k in ("false_alarm", "clean_compatible",
                                                                            "clean_undecided", "clean_localized",
                                                                            "conditional")}
                                                     for c in CONDITIONS for m in METHODS},
                           "boundary": {f"{c}|{m}": {k: sw["tab"][f"boundary|{c}|{m}"][k] for k in SUMMARY_KEYS}
                                        for c in CONDITIONS for m in METHODS}}
                   for label, sw in sweeps.items() if label != "baseline"},
        "sweep_deltas_vs_baseline_count": dict(Counter(d[0] for d in deltas)),
        "timings_s": timings,
    }

    # ---- outputs ----
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "case144_results.json").write_text(json.dumps(_jsonable(result), indent=1), encoding="utf-8")
    write_records_csv(out_dir / "case144_records_baseline.csv", observations, records["baseline"], sweeps["baseline"]["scored"])
    write_table_csv(out_dir / "case144_summary_table.csv", {"baseline": baseline_tab})
    write_table_csv(out_dir / "case144_sweep_table.csv", {label: sw["tab"] for label, sw in sweeps.items() if label != "baseline"})
    with (out_dir / "case144_sweep_deltas.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["point", "id", "cohort", "family", "context_1based", "consumer", "condition", "method",
                    "true_causes", "baseline_decision", "baseline_localized", "baseline_conditional", "baseline_outcome",
                    "baseline_localization", "baseline_inv_status", "point_decision", "point_localized",
                    "point_conditional", "point_outcome", "point_localization", "point_conditional_score",
                    "point_inv_status"])
        for label, r, fr in deltas:
            w.writerow([label, r["id"], r["cohort"], r["family"], r["context"] + 1, r["consumer"], r["condition"],
                        r["method"], "+".join(r["true_causes"]), fr["decision"], "+".join(fr["localized"]),
                        "(+)".join(fr["conditional"] or []), fr["outcome"], fr["loc"], fr["inv_status"],
                        r["decision"], "+".join(r["localized"]), "(+)".join(r["conditional"] or []), r["outcome"],
                        r["loc"], r["cond"] or "", r["inv_status"]])

    lines = summary_lines(result, baseline_tab, sweeps)
    print("\n".join(lines))
    return result


def summary_lines(result: dict, baseline_tab: dict, sweeps: dict) -> list:
    lp = result["lp_vs_closed_form"]
    lines = ["=== 144-case set -- OR / CW / OW, budget sensitivity ===",
             f"setup {result['setup']}",
             f"predictions digest (before truth) {result['inputs']['predictions_digest_before_truth']}",
             "",
             f"LP (HiGHS) vs closed form: inversions {lp['inversions_all_points']} (all points), "
             f"disagreements {lp['disagreements']}, max |diff| {lp['max_abs_diff']:.3e} (tol {lp['tolerance']:g})",
             f"OW k interval (full probes, baseline point): {result['ow_k_interval_full_probes_baseline']}",
             f"consistency (all points): comparable observations {result['consistency']['comparable_observations']}, "
             f"failing the grid guard {len(result['consistency']['comparable_observations_failing_grid_guard'])}; "
             f"cells with FULL != ALARM-and-FULL {len(result['consistency']['cells_FULL_ne_ALARM_and_FULL'])}; "
             f"conditional attributions on non-ALARM decisions {result['consistency']['conditional_on_non_ALARM_total']}; "
             f"localizations without ALARM {result['consistency']['localized_without_alarm_total']}",
             f"normal-case discretization deviation max |ln R_Q|: fine grids "
             f"{result['consistency']['clean_max_abs_ln_RQ']:.10g}, coarse grids "
             f"{result['consistency']['coarse_clean_max_abs_ln_RQ']:.10g} (max |R_Q - 1| "
             f"{result['consistency']['coarse_clean_max_abs_RQ_minus_1']:.10g})",
             f"OW conditional set == CW removed-source localization set at every point: "
             f"{all(result['consistency']['ow_conditional_set_equals_cw_removed_source_localization'].values())}",
             f"baseline-point OW conditional attributions by condition|family: "
             f"{result['consistency']['baseline_ow_conditional_by_family']}",
             ""]
    lines += main_table_text(baseline_tab, "BASELINE POINT (tau_g 2 %, eps_Q 0.020203, tau_c 1e-10, eps_c 1.01e-10)")
    lines.append("")
    lines += boundary_text(baseline_tab)
    lines.append("")
    lines.append("EXPECTATIONS (fixed before the run): nature | agree | quantity | expected | observed")
    for e in result["expectations_baseline"] + result["expectations_sweeps"]:
        mark = "agree" if e["agree"] else "DISAGREE"
        lines.append(f"  [{e['section']}] {e['nature']:10} {mark:8} {e['quantity']} | exp {e['expected']} | "
                     f"obs {e['observed']}" + (f" | {e['note']}" if e["note"] else ""))
    lines.append("")
    lines += sweep_text({k: v for k, v in sweeps.items() if k != "baseline"})
    lines.append("")
    lines.append(f"records changed vs baseline point, per sweep point: {result['sweep_deltas_vs_baseline_count']}")
    lines.append(f"timings (s): { {k: round(v, 2) for k, v in result['timings_s'].items()} }")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT, help="output directory (default results/case144)")
    args = ap.parse_args()
    run(args.out)


if __name__ == "__main__":
    main()
