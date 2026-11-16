"""Cell-averaged versus node-sampled potential in the controlled finite-difference well (Table S4): dipole and
derivative-form readings on the grid ladder and their convergence orders. Writes results/potential_convention/."""
from __future__ import annotations

import os

for _key_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_key_name, "4")   # four BLAS threads, as in the recorded run

import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from toy_model import (DOMAIN_B_NM, DZ_LADDER, DZ_REF_B, HBAR2_OVER_2M0, MSTAR_B,  # noqa: E402
                       V0_B_EV, WELL_B_NM, Z0_BARDEEN_B, _bardeen_correct, c1_correct_dipole,
                       c2_bug_dipole, c3_bug_bardeen, fd_well_solve)

OUT = REPO / "results" / "potential_convention"
FACILITY = "code/toy_model.py"

CONVENTIONS = ("cell_average", "node_sampled")
RUNGS = tuple(float(d) for d in DZ_LADDER)        # 0.40 / 0.20 / 0.10 / 0.05 nm
DZ_REF = float(DZ_REF_B)                          # 0.0125 nm, the Leg-B reference rung
Z0_CENTRE = 0.5 * (WELL_B_NM[0] + WELL_B_NM[1])   # 15 nm, well centre
Z0S = (float(Z0_BARDEEN_B), Z0_CENTRE)            # 20 nm (matching plane) and 15 nm
QUANTITY_Z0 = {"W2_z20": float(Z0_BARDEEN_B), "W2_z15": Z0_CENTRE}
QUANTITIES = ("dipole2", "W2_z20", "W2_z15")
K_SQUARED = 2                                     # missing conversion: extra h^2 on the squared quantities

# criteria fixed before the run
SECOND_ORDER_RATIO_MIN = 3.0
MULTIPLIER_TOL = 0.10          # B-leg tolerance = FACTOR_TOL["B"] of the controlled validation
GUARD_LIMITS = {"adjacent_last_relative": 0.02, "trim_slope_change": 0.05, "max_log_residual": 0.02}


def _key(dz: float) -> str:
    return f"{dz:g}"


def _z0_tag(z0: float) -> str:
    return f"z{z0:g}"


# ------------------------------------------------------------------ grid and potentials --
def interior_grid(dz: float):
    """As toy_model.fd_well_solve: n = DOMAIN/dz cells, interior nodes z_k = k dz, k = 1..n-1."""
    n = int(round(DOMAIN_B_NM / dz))
    if abs(n * dz - DOMAIN_B_NM) > 1e-9:
        raise ValueError(f"dz={dz} does not divide domain={DOMAIN_B_NM}")
    return n, np.arange(1, n) * dz


def node_sampled_potential(z: np.ndarray) -> np.ndarray:
    """As toy_model.fd_well_solve: node sampling, both interface nodes (10 and 20 nm) assigned to the well."""
    return np.where((z >= WELL_B_NM[0]) & (z <= WELL_B_NM[1]), 0.0, V0_B_EV)


def _position_in_steps(x_nm: float, dz: float) -> float:
    """Interface position in units of dz, snapped to the exact integer when the interface lies on a node."""
    u = x_nm / dz
    r = round(u)
    return float(r) if abs(u - r) <= 1e-9 else u


def dual_cell_well_fraction(k, a: float, b: float) -> np.ndarray:
    """Fraction of the dual cell [k - 1/2, k + 1/2] lying inside the well [a, b] (all in units of dz)."""
    k = np.asarray(k, dtype=float)
    return np.clip(np.minimum(k + 0.5, b) - np.maximum(k - 0.5, a), 0.0, 1.0)


def cell_average_potential(n: int, dz: float) -> np.ndarray:
    """Average of the step potential (0 in the well, V_b outside) over [z_k - dz/2, z_k + dz/2], k = 1..n-1.

    Evaluated on integer node indices, so for on-node interfaces the result is exact in floating point:
    V = V_b/2 on an interface node, 0 strictly inside, V_b strictly outside."""
    k = np.arange(1, n, dtype=float)
    a = _position_in_steps(WELL_B_NM[0], dz)
    b = _position_in_steps(WELL_B_NM[1], dz)
    return V0_B_EV * (1.0 - dual_cell_well_fraction(k, a, b))


def potential(dz: float, convention: str):
    n, z = interior_grid(dz)
    if convention == "node_sampled":
        v = node_sampled_potential(z)
    elif convention == "cell_average":
        v = cell_average_potential(n, dz)
    else:
        raise ValueError(f"unknown potential convention {convention!r}")
    return z, v


def build_hamiltonian(dz: float, convention: str):
    """As toy_model.fd_well_solve (verbatim), with the potential of the chosen convention."""
    z, v = potential(dz, convention)
    t = HBAR2_OVER_2M0 / (MSTAR_B * dz * dz)       # hbar^2/(2 m* dz^2)
    h = np.diag(2.0 * t + v) + np.diag(-t * np.ones(len(z) - 1), 1) \
        + np.diag(-t * np.ones(len(z) - 1), -1)
    return z, v, h


def solve_well(dz: float, convention: str) -> dict:
    """As toy_model.fd_well_solve (verbatim: dense eigh, ascending order, two lowest states), plus the
        two eigenvalues and the relative eigenpair residual max_j ||H c_j - E_j c_j||_2 / ||H||_inf.
    """
    z, v, h = build_hamiltonian(dz, convention)
    w, vecs = np.linalg.eigh(h)
    order = np.argsort(w)
    c1, c2 = vecs[:, order[0]], vecs[:, order[1]]
    e1, e2 = w[order[0]], w[order[1]]
    h_norm = float(np.max(np.sum(np.abs(h), axis=1)))
    residual = max(float(np.linalg.norm(h @ c1 - e1 * c1)), float(np.linalg.norm(h @ c2 - e2 * c2))) / h_norm
    return {"z": z, "v": v, "c1": c1, "c2": c2, "E1": float(e1), "E2": float(e2),
            "eigpair_rel_residual": residual}


# ------------------------------------------------------------------------------ readings --
def readings(dz: float, sol: dict) -> dict:
    """The four toy-model consumers on one rung.  The derivative form is evaluated at node index
        int(round(z0/dz)) - 1 (toy_model c3_bug_bardeen / _bardeen_correct); that node's coordinate is recorded.
    """
    z, c1, c2 = sol["z"], sol["c1"], sol["c2"]
    out = {"dipole2_correct": c1_correct_dipole(z, c1, c2, dz),
           "dipole2_missing": c2_bug_dipole(z, c1, c2, dz)}
    for z0 in Z0S:
        tag = _z0_tag(z0)
        i0 = int(round(z0 / dz)) - 1
        out[f"W2_{tag}_correct"] = _bardeen_correct(z, c1, c2, dz, z0)
        out[f"W2_{tag}_missing"] = c3_bug_bardeen(z, c1, c2, dz, z0)
        out[f"W2_{tag}_z_eval_nm"] = float(z[i0])
    return out


def potential_summary(dz: float, z: np.ndarray, v: np.ndarray) -> dict:
    a, b = WELL_B_NM[0] / dz, WELL_B_NM[1] / dz
    k_lo, k_hi = int(round(a)), int(round(b))
    return {"interfaces_on_nodes": bool(abs(a - k_lo) <= 1e-9 and abs(b - k_hi) <= 1e-9),
            "z_interface_nodes_nm": [float(z[k_lo - 1]), float(z[k_hi - 1])],
            "V_at_z10_eV": float(v[k_lo - 1]), "V_at_z20_eV": float(v[k_hi - 1]),
            "n_nodes_V_zero": int(np.count_nonzero(v == 0.0)),
            "n_nodes_V_half": int(np.count_nonzero(v == V0_B_EV / 2)),
            "effective_well_width_nm": float(np.sum(1.0 - v / V0_B_EV) * dz)}


# ------------------------------------------------------------------- metrics and criteria --
def grid_metrics(h, q) -> dict:
    """Grid metrics as defined for the 144-case study: x = ln h, y = ln|Q|, unweighted
        numpy.polyfit (degree 1); trim = the same fit without the coarsest rung;
        adjacent_relative[i] = |Q_i - Q_{i+1}| / |Q_{i+1}| (coarse -> fine).
    """
    h = np.asarray(h, dtype=float)
    q = np.asarray(q, dtype=float)
    if np.any(q == 0):
        raise ValueError("zero value in sequence; log metrics undefined")
    x, y = np.log(h), np.log(abs(q))
    slope, intercept = np.polyfit(x, y, 1)
    trim = np.polyfit(x[1:], y[1:], 1)[0]
    return {"slope": float(slope), "trim_slope": float(trim), "trim_change": float(abs(slope - trim)),
            "max_log_residual": float(max(abs(y - slope * x - intercept))),
            "adjacent_relative": list(map(float, abs(np.diff(q) / q[1:])))}


def order_verdict(adjacent) -> dict:
    """'second order' <=> both consecutive ratios of the correct-path adjacent changes are >= 3.
        The local orders log2(delta_i / delta_{i+1}) are descriptive; they only label a sequence that is not
        second order ('first order' when both round to 1).
    """
    d = [float(x) for x in adjacent]
    ratios = [d[0] / d[1], d[1] / d[2]]
    orders = [math.log2(r) for r in ratios]
    second = bool(all(r >= SECOND_ORDER_RATIO_MIN for r in ratios))
    if second:
        label = "second order"
    elif all(round(p) == 1 for p in orders):
        label = "first order"
    else:
        label = "not second order (local orders do not both round to 1)"
    return {"ratios": ratios, "local_orders_log2": orders, "second_order": second, "label": label}


def multiplier_verdict(max_abs_rref_minus_1: float) -> dict:
    return {"max_abs_Rref_minus_1": float(max_abs_rref_minus_1), "tolerance": MULTIPLIER_TOL,
            "pass": bool(max_abs_rref_minus_1 <= MULTIPLIER_TOL)}


def guard_verdict(metrics: dict) -> dict:
    """Algorithm 1 grid guard."""
    checks = {"last_delta": bool(metrics["adjacent_relative"][-1] <= GUARD_LIMITS["adjacent_last_relative"]),
              "dzeta": bool(metrics["trim_change"] <= GUARD_LIMITS["trim_slope_change"]),
              "eta": bool(metrics["max_log_residual"] <= GUARD_LIMITS["max_log_residual"])}
    return {"checks": checks, "limits": dict(GUARD_LIMITS), "pass": all(checks.values())}


def analyse_sequence(h, good, bad, ref: float) -> dict:
    """Rref / Rpath with k = 2; grid metrics on both paths; the criteria (order and guard on the
        correct path, multiplier check on max|Rref - 1|).
    """
    k = K_SQUARED
    rref = np.array(bad) / (np.array(h) ** k * ref)
    rpath = np.array(bad) / (np.array(h) ** k * np.array(good))
    gm, bm = grid_metrics(h, good), grid_metrics(h, bad)
    max_rref = float(max(abs(rref - 1)))
    return {"h_nm": [float(x) for x in h], "good": [float(x) for x in good], "bad": [float(x) for x in bad],
            "reference": float(ref), "good_metrics": gm, "bad_metrics": bm,
            "Rref": [float(x) for x in rref], "max_abs_Rref_minus_1": max_rref,
            "Rpath": [float(x) for x in rpath], "max_abs_Rpath_minus_1": float(max(abs(rpath - 1))),
            "finest_vs_reference": float(abs(good[-1] / ref - 1)),
            "order": order_verdict(gm["adjacent_relative"]),
            "multiplier_check": multiplier_verdict(max_rref),
            "grid_guard": guard_verdict(gm)}


# ------------------------------------------------------ expectations (fixed before the run) --
def _round_match(observed: float, printed: str) -> bool:
    """Agreement at the printed precision: observed rounded to the printed number of decimals."""
    dec = len(printed.split(".")[1])
    return round(float(observed), dec) == round(float(printed), dec)


def evaluate_expectations(seq: dict) -> list:
    ca20 = seq["cell_average"]["W2_z20"]
    ns20 = seq["node_sampled"]["W2_z20"]
    ns15 = seq["node_sampled"]["W2_z15"]
    adj = ns20["good_metrics"]["adjacent_relative"]
    items = [
        {"id": "E1", "convention": "cell_average", "quantity": "W2_z20",
         "what": "derivative path max|Rref - 1|", "expected": "~0.023 (KNOWN 0.0227)",
         "observed": ca20["max_abs_Rref_minus_1"], "rule": "round(observed, 3) == 0.023",
         "match": _round_match(ca20["max_abs_Rref_minus_1"], "0.023"),
         "nature": "expectation; the value is known from an earlier exploratory calculation"},
        {"id": "E2", "convention": "cell_average", "quantity": "W2_z20",
         "what": "correct-path convergence order", "expected": "second order",
         "observed": {"adjacent": ca20["good_metrics"]["adjacent_relative"], **ca20["order"]},
         "rule": "second-order criterion: both consecutive ratios of the correct-path adjacent changes >= 3",
         "match": ca20["order"]["second_order"], "nature": "expectation"},
        {"id": "E3", "convention": "cell_average", "quantity": "W2_z20",
         "what": "grid guard (last delta <= 0.02, dzeta <= 0.05, eta <= 0.02)", "expected": "all pass",
         "observed": {"last_delta": ca20["good_metrics"]["adjacent_relative"][-1],
                      "dzeta": ca20["good_metrics"]["trim_change"],
                      "eta": ca20["good_metrics"]["max_log_residual"], **ca20["grid_guard"]["checks"]},
         "rule": "grid guard (Algorithm 1) on the correct path", "match": ca20["grid_guard"]["pass"],
         "nature": "expectation"},
        {"id": "E4", "convention": "node_sampled", "quantity": "W2_z20",
         "what": "derivative path max|Rref - 1|", "expected": "0.210513 (recorded node-sampled value)",
         "observed": ns20["max_abs_Rref_minus_1"], "rule": "round(observed, 6) == 0.210513",
         "match": _round_match(ns20["max_abs_Rref_minus_1"], "0.210513"),
         "nature": "reproduction of the recorded node-sampled value"},
        {"id": "E5", "convention": "node_sampled", "quantity": "W2_z20",
         "what": "correct-path adjacent changes", "expected": "0.1088 / 0.0491 / 0.0233 (recorded node-sampled values)",
         "observed": adj, "rule": "each rounded to 4 decimals equals the printed value",
         "match": all(_round_match(a, e) for a, e in zip(adj, ("0.1088", "0.0491", "0.0233"))),
         "nature": "reproduction of the recorded node-sampled value"},
        {"id": "E6", "convention": "node_sampled", "quantity": "W2_z20",
         "what": "correct-path convergence order", "expected": "first order (recorded node-sampled behaviour)",
         "observed": ns20["order"],
         "rule": "second-order criterion fails and both local orders log2(delta_i/delta_i+1) round to 1",
         "match": ns20["order"]["label"] == "first order", "nature": "reproduction of the recorded node-sampled behaviour"},
        {"id": "E7", "convention": "node_sampled", "quantity": "W2_z15",
         "what": "derivative path max|Rref - 1| at the well centre", "expected": "~0.12 (KNOWN 0.120)",
         "observed": ns15["max_abs_Rref_minus_1"], "rule": "round(observed, 2) == 0.12",
         "match": _round_match(ns15["max_abs_Rref_minus_1"], "0.12"),
         "nature": "known from an earlier exploratory calculation"},
    ]
    return items


def evaluate_known_values(seq: dict) -> list:
    ca20 = seq["cell_average"]["W2_z20"]
    ns15 = seq["node_sampled"]["W2_z15"]
    adj = ca20["good_metrics"]["adjacent_relative"]
    return [
        {"id": "K1", "what": "cell_average z0=20 derivative max|Rref - 1|", "known": "0.0227",
         "observed": ca20["max_abs_Rref_minus_1"], "rule": "4 decimals",
         "match": _round_match(ca20["max_abs_Rref_minus_1"], "0.0227")},
        {"id": "K2", "what": "cell_average z0=20 correct-path adjacent changes",
         "known": "0.0169 / 0.0043 / 0.0011", "observed": adj, "rule": "4 decimals each",
         "match": all(_round_match(a, e) for a, e in zip(adj, ("0.0169", "0.0043", "0.0011")))},
        {"id": "K3", "what": "node_sampled z0=15 derivative max|Rref - 1|", "known": "0.120",
         "observed": ns15["max_abs_Rref_minus_1"], "rule": "3 decimals",
         "match": _round_match(ns15["max_abs_Rref_minus_1"], "0.120")},
    ]


# ------------------------------------------------------------------------------- driver --
AMBIGUITIES = [
    "z0 = 15 nm is not a node of the 0.40 nm grid (15/0.4 = 37.5). The toy-model consumers "
    "(c3_bug_bardeen and _bardeen_correct: i0 = int(round(z0/dz)) - 1, Python round-half-even) evaluate the "
    "0.40 nm reading at node index 37, i.e. z = 15.2 nm. Resolution: implemented literally with the toy-model "
    "consumer functions; the evaluated coordinate is recorded per rung (sequences.<conv>.W2_z15.z_eval_nm) and "
    "per-rung Rref is reported. The known value 0.120 of the earlier exploratory calculation used the same "
    "snapping. The 0.20 / 0.10 / 0.05 nm rungs and the 0.0125 nm reference are on-node for z0 = 15 nm; z0 = 20 nm "
    "is on-node on every rung. max|Rref - 1| over the on-node rungs only is added as descriptive information.",
    "Only the 'second order' criterion is fixed in advance. The label 'first order' is resolved as: the "
    "second-order criterion fails AND both local orders log2(delta_i/delta_i+1) round to 1 (descriptive label; "
    "the criterion is unchanged).",
    "Sequence carrying the grid guard and the order criterion: the correct path (delta and eta refer to the "
    "correct path; the guard failure is tied to the last adjacent change and the residual). The missing-conversion "
    "path has the same dzeta and eta up to rounding (Rpath ~ 1e-14) and is reported as bad_metrics.",
    "Reference for Rref and for the finest-rung deviation under the cell-average convention: the correct "
    "consumer at the 0.0125 nm rung solved with the SAME cell-average potential (the same-solver internal "
    "reference of Leg B, construction of toy_model.ref_leg_b).",
    "No numeric tolerance is given for '~' or for reproduction; resolved as agreement at the printed "
    "precision (observed rounded to the number of printed decimals equals the printed value).",
    "Cell average = average over the dual cell [z_i - dz/2, z_i + dz/2] of the step potential (0 on [10, 20] "
    "nm, V_b outside), computed on integer node indices with on-node interfaces snapped "
    "(|x/dz - round(x/dz)| <= 1e-9) so that on-node interfaces give exactly V_b/2. On every rung used both "
    "interfaces are on nodes, hence V takes only 0, V_b/2, V_b and the discrete well width is exactly 10 nm "
    "(node sampling: 10 nm + dz).",
    "Missing-conversion readings at the 0.0125 nm reference rung are reported as readings only; no "
    "criterion uses them.",
    "Constant: the toy-model constants are used; toy_model HBAR2_OVER_2M0 = 0.03809982111485961 eV nm^2 "
    "(hbar entered as 1.054571817e-34) differs by 1.2e-9 relative from C0 = 0.0380998211615486 (hbar = h/2pi) "
    "of the earlier exploratory calculation, which also used scipy eigh_tridiagonal; its known numbers are "
    "expected to agree at printed precision, not bitwise.",
]


def run_analysis() -> dict:
    rungs_all = RUNGS + (DZ_REF,)
    reads = {c: {} for c in CONVENTIONS}
    pot = {c: {} for c in CONVENTIONS}
    ident = {}
    for conv in CONVENTIONS:
        for dz in rungs_all:
            sol = solve_well(dz, conv)
            rd = readings(dz, sol)
            rd.update(E1_eV=sol["E1"], E2_eV=sol["E2"], n_interior_nodes=int(len(sol["z"])),
                      eigpair_rel_residual=sol["eigpair_rel_residual"])
            reads[conv][dz] = rd
            pot[conv][_key(dz)] = potential_summary(dz, sol["z"], sol["v"])
            if conv == "node_sampled":
                z_f, c1_f, c2_f = fd_well_solve(dz)
                ident[_key(dz)] = {name: bool(mine.dtype == facility_value.dtype and mine.shape == facility_value.shape
                                              and mine.tobytes() == facility_value.tobytes())
                                   for name, mine, facility_value in (("z", sol["z"], z_f), ("c1", sol["c1"], c1_f),
                                                              ("c2", sol["c2"], c2_f))}
    seq = {c: {} for c in CONVENTIONS}
    for conv in CONVENTIONS:
        r = reads[conv]
        for q in QUANTITIES:
            s = analyse_sequence(list(RUNGS), [r[dz][f"{q}_correct"] for dz in RUNGS],
                                 [r[dz][f"{q}_missing"] for dz in RUNGS], r[DZ_REF][f"{q}_correct"])
            if q in QUANTITY_Z0:
                z0 = QUANTITY_Z0[q]
                s["z0_nm"] = z0
                s["z_eval_nm"] = {_key(dz): r[dz][f"{q}_z_eval_nm"] for dz in rungs_all}
                off = [dz for dz in RUNGS if abs(r[dz][f"{q}_z_eval_nm"] - z0) > 1e-9]
                s["z0_offgrid_rungs_nm"] = off
                if off:
                    on = [i for i, dz in enumerate(RUNGS) if dz not in off]
                    s["descriptive_only"] = {
                        "max_abs_Rref_minus_1_on_node_rungs_only": float(max(abs(s["Rref"][i] - 1) for i in on)),
                        "on_node_rungs_nm": [RUNGS[i] for i in on]}
            seq[conv][q] = s
    items = evaluate_expectations(seq)
    return {
        "analysis": "controlled finite-difference well: cell-averaged potential control",
        "setup": {
            "facility": FACILITY + " (imported read-only)",
            "constants_from_facility": {"DOMAIN_B_NM": DOMAIN_B_NM, "WELL_B_NM": list(WELL_B_NM),
                                        "V0_B_EV": V0_B_EV, "MSTAR_B": MSTAR_B,
                                        "HBAR2_OVER_2M0_eV_nm2": HBAR2_OVER_2M0, "DZ_LADDER": list(RUNGS),
                                        "DZ_REF_B": DZ_REF, "Z0_BARDEEN_B": float(Z0_BARDEEN_B)},
            "z0_nm": list(Z0S),
            "conventions": {
                "node_sampled": "toy_model.fd_well_solve potential verbatim, V = 0 for 10 <= z <= 20 nm (both interface nodes in the well), V_b otherwise",
                "cell_average": "V_i = average of the step potential over [z_i - dz/2, z_i + dz/2]; on-node "
                                "interface -> V_b/2"},
            "solver": "toy_model.fd_well_solve construction verbatim: interior nodes z_k = k dz, Dirichlet, three-point FD, t = HBAR2_OVER_2M0/(MSTAR_B dz dz), dense numpy.linalg.eigh, two lowest states, Euclidean-normalized eigenvectors",
            "consumers": {"dipole2_correct": "toy_model.c1_correct_dipole",
                          "dipole2_missing": "toy_model.c2_bug_dipole",
                          "W2_correct": "toy_model._bardeen_correct",
                          "W2_missing": "toy_model.c3_bug_bardeen"},
            "units": {"dipole2_correct": "nm^2", "dipole2_missing": "raw-c units (nm^4 as computed)",
                      "W2_correct": "nm^-4", "W2_missing": "raw-c units (nm^-2 as computed)", "E": "eV"},
        },
        "definitions": {
            "h": "h = dz / (1 nm); rungs 0.40 / 0.20 / 0.10 / 0.05 nm, coarse -> fine",
            "Rref": "Rref(h) = Qbad(h) / (h^2 Qref); Qref = correct consumer at dz_ref = 0.0125 nm, same solver and "
                    "same potential convention; reported max_i |Rref(h_i) - 1| over the four rungs "
                    "(reference factor residual; factor_worst_rel_dev of controlled_validation.py)",
            "Rpath": "Rpath(h) = Qbad(h) / (h^2 Qgood(h)); max_i |Rpath - 1|",
            "adjacent_change": "delta_i = |Q_i - Q_i+1| / |Q_i+1| (0.40->0.20, 0.20->0.10, 0.10->0.05) on the "
                               "correct path",
            "zeta_dzeta_eta": "unweighted LSQ of ln|Q| on ln h (numpy.polyfit, degree 1) = zeta; zeta_minus = same "
                              "without the coarsest rung; dzeta = |zeta - zeta_minus|; eta = max_i |y_i - a - zeta "
                              "x_i|",
            "grid_guard": "last delta <= 0.02 AND dzeta <= 0.05 AND eta <= 0.02 on the correct path (Algorithm 1)",
            "second_order": "both delta_1/delta_2 and delta_2/delta_3 >= 3 (correct path)",
            "multiplier_check": "max|Rref - 1| <= 0.10 (FACTOR_TOL['B'] of the controlled validation)",
            "finest_vs_reference": "|Qgood(0.05 nm) / Qref - 1|",
        },
        "ambiguities_and_resolutions": AMBIGUITIES,
        "node_branch_identity": {"per_rung": ident,
                                 "all_bitwise_identical": all(all(v.values()) for v in ident.values())},
        "potential_checks": pot,
        "readings": {c: {_key(dz): reads[c][dz] for dz in rungs_all} for c in CONVENTIONS},
        "sequences": seq,
        "expectation_items": items,
        "all_expectation_items_match": all(it["match"] for it in items),
        "known_value_reconciliation": evaluate_known_values(seq),
    }


def identity() -> dict:
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
            "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                                         "OPENBLAS_NUM_THREADS")}}


def write_csvs(res: dict) -> None:
    rungs_all = RUNGS + (DZ_REF,)
    with (OUT / "potential_convention_readings.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["convention", "dz_nm", "role", "n_interior_nodes", "E1_eV", "E2_eV", "V_at_z10_eV",
                    "V_at_z20_eV", "effective_well_width_nm", "dipole2_correct_nm2", "dipole2_missing",
                    "W2_z20_correct_nm-4", "W2_z20_missing", "W2_z20_z_eval_nm", "W2_z15_correct_nm-4",
                    "W2_z15_missing", "W2_z15_z_eval_nm", "Rref_dipole2", "Rpath_dipole2", "Rref_W2_z20",
                    "Rpath_W2_z20", "Rref_W2_z15", "Rpath_W2_z15"])
        for conv in CONVENTIONS:
            for dz in rungs_all:
                rd = res["readings"][conv][_key(dz)]
                pc = res["potential_checks"][conv][_key(dz)]
                row = [conv, dz, "reference" if dz == DZ_REF else "rung", rd["n_interior_nodes"], rd["E1_eV"],
                       rd["E2_eV"], pc["V_at_z10_eV"], pc["V_at_z20_eV"], pc["effective_well_width_nm"],
                       rd["dipole2_correct"], rd["dipole2_missing"], rd["W2_z20_correct"], rd["W2_z20_missing"],
                       rd["W2_z20_z_eval_nm"], rd["W2_z15_correct"], rd["W2_z15_missing"], rd["W2_z15_z_eval_nm"]]
                for q in QUANTITIES:
                    s = res["sequences"][conv][q]
                    row += ["", ""] if dz == DZ_REF else [s["Rref"][RUNGS.index(dz)], s["Rpath"][RUNGS.index(dz)]]
                w.writerow(row)
    with (OUT / "potential_convention_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["convention", "quantity", "Qref", "zeta_correct", "zeta_minus_correct", "dzeta_correct",
                    "eta_correct", "delta1_correct", "delta2_correct", "delta3_correct", "ratio1", "ratio2",
                    "local_order1", "local_order2", "second_order", "order_label", "zeta_missing",
                    "zeta_minus_missing", "dzeta_missing", "eta_missing", "max_abs_Rref_minus_1",
                    "multiplier_check_pass", "max_abs_Rpath_minus_1", "finest_vs_reference",
                    "guard_last_delta_ok", "guard_dzeta_ok", "guard_eta_ok", "guard_pass", "z0_offgrid_rungs_nm"])
        for conv in CONVENTIONS:
            for q in QUANTITIES:
                s = res["sequences"][conv][q]
                g, b, o, gd = s["good_metrics"], s["bad_metrics"], s["order"], s["grid_guard"]["checks"]
                w.writerow([conv, q, s["reference"], g["slope"], g["trim_slope"], g["trim_change"],
                            g["max_log_residual"], *g["adjacent_relative"], *o["ratios"], *o["local_orders_log2"],
                            o["second_order"], o["label"], b["slope"], b["trim_slope"], b["trim_change"],
                            b["max_log_residual"], s["max_abs_Rref_minus_1"], s["multiplier_check"]["pass"],
                            s["max_abs_Rpath_minus_1"], s["finest_vs_reference"], gd["last_delta"], gd["dzeta"],
                            gd["eta"], s["grid_guard"]["pass"],
                            " ".join(_key(d) for d in s.get("z0_offgrid_rungs_nm", []))])


def _fmt_obs(x) -> str:
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, float):
        return f"{x:.7g}"
    if isinstance(x, list):
        return " / ".join(_fmt_obs(v) for v in x)
    if isinstance(x, dict):
        return ", ".join(f"{k}={_fmt_obs(v)}" for k, v in x.items())
    return str(x)


def format_summary(res: dict) -> str:
    idn = res["identity"]
    rungs_all = RUNGS + (DZ_REF,)
    L = ["=== controlled FD well, cell-averaged potential control ===",
         f"UTC {idn['timestamp_utc']}  python {sys.version.split()[0]}  numpy {np.__version__}  threads "
         + "/".join(str(v) for v in idn["threads"].values()),
         f"compute wall time {res['wall_time_compute_s']:.2f} s", ""]
    ni = res["node_branch_identity"]
    L.append("[identity] node-sampled branch vs toy_model.fd_well_solve (z, c1, c2 bitwise): "
             + ", ".join(f"{k} nm {'IDENTICAL' if all(v.values()) else 'DIFFERENT'}" for k, v in ni["per_rung"].items()))
    for conv in CONVENTIONS:
        pcs = res["potential_checks"][conv]
        L.append(f"[potential] {conv:12}: V(10 nm)/V(20 nm) = "
                 + ", ".join(f"{pcs[_key(dz)]['V_at_z10_eV']:g}/{pcs[_key(dz)]['V_at_z20_eV']:g}" for dz in rungs_all)
                 + " eV; effective well width = "
                 + ", ".join(f"{pcs[_key(dz)]['effective_well_width_nm']:.10g}" for dz in rungs_all) + " nm")
    L += ["", "Readings (rungs 0.4/0.2/0.1/0.05 nm, reference 0.0125 nm). E in eV; dipole2_ok nm^2; W2_ok nm^-4;",
          "'miss' = missing-conversion readings as returned by the toy-model consumers.",
          f"{'convention':12} {'dz':>7} {'E1':>10} {'E2':>10} {'dip2_ok':>12} {'dip2_miss':>12} "
          f"{'W2z20_ok':>12} {'W2z20_miss':>12} {'W2z15_ok':>12} {'W2z15_miss':>12} {'z15_at':>7}"]
    for conv in CONVENTIONS:
        for dz in rungs_all:
            rd = res["readings"][conv][_key(dz)]
            L.append(f"{conv:12} {dz:>7g} {rd['E1_eV']:>10.7f} {rd['E2_eV']:>10.7f} {rd['dipole2_correct']:>12.6e} "
                     f"{rd['dipole2_missing']:>12.6e} {rd['W2_z20_correct']:>12.6e} {rd['W2_z20_missing']:>12.6e} "
                     f"{rd['W2_z15_correct']:>12.6e} {rd['W2_z15_missing']:>12.6e} {rd['W2_z15_z_eval_nm']:>7.4g}")
    L += ["", "Sequence metrics over the four rungs (order, guard, zeta etc. on the correct path unless 'missing';",
          "Rref / Rpath with h^2; criteria: second order <=> both ratios >= 3; multiplier max|Rref-1| <= 0.10;",
          "guard last delta <= 0.02, dzeta <= 0.05, eta <= 0.02)"]
    for conv in CONVENTIONS:
        for q in QUANTITIES:
            s = res["sequences"][conv][q]
            g, b, o, gg = s["good_metrics"], s["bad_metrics"], s["order"], s["grid_guard"]
            a = g["adjacent_relative"]
            L.append(f"- {conv} / {q}: correct zeta {g['slope']:+.7f}, zeta- {g['trim_slope']:+.7f}, "
                     f"dzeta {g['trim_change']:.7f}, eta {g['max_log_residual']:.7f}; missing zeta {b['slope']:+.7f}, "
                     f"zeta- {b['trim_slope']:+.7f}, dzeta {b['trim_change']:.7f}, eta {b['max_log_residual']:.7f}")
            L.append(f"    adjacent {a[0]:.7f} / {a[1]:.7f} / {a[2]:.7f}; ratios {o['ratios'][0]:.3f} / "
                     f"{o['ratios'][1]:.3f} -> {o['label']}")
            L.append(f"    Rref per rung " + " / ".join(f"{x:.6f}" for x in s["Rref"])
                     + f"; max|Rref-1| {s['max_abs_Rref_minus_1']:.6f} -> multiplier check "
                     + ("PASS" if s["multiplier_check"]["pass"] else "FAIL")
                     + f"; max|Rpath-1| {s['max_abs_Rpath_minus_1']:.2e}; finest vs ref "
                     + f"{100 * s['finest_vs_reference']:.3f} %")
            L.append("    guard: " + ", ".join(f"{k} {'ok' if v else 'FAIL'}" for k, v in gg["checks"].items())
                     + f" -> {'PASS' if gg['pass'] else 'FAIL'}"
                     + (f"; z0 {s['z0_nm']:g} nm evaluated at " + " / ".join(
                         f"{s['z_eval_nm'][_key(dz)]:.4g}" for dz in RUNGS) + " nm" if "z_eval_nm" in s else ""))
            if "descriptive_only" in s:
                dsc = s["descriptive_only"]
                L.append(f"    [descriptive] off-node rungs {s['z0_offgrid_rungs_nm']}; max|Rref-1| over "
                         f"on-node rungs {dsc['on_node_rungs_nm']} = {dsc['max_abs_Rref_minus_1_on_node_rungs_only']:.6f}")
    L += ["", "Expectations (fixed before the run): predicted vs observed"]
    for it in res["expectation_items"]:
        L.append(f"- {it['id']} [{it['nature']}] {it['convention']} / {it['quantity']}: {it['what']}")
        L.append(f"    expected {it['expected']} | observed {_fmt_obs(it['observed'])}")
        L.append(f"    rule: {it['rule']} -> {'MATCH' if it['match'] else 'MISMATCH'}")
    L.append(f"all expectation items match: {res['all_expectation_items_match']}")
    L += ["", "Known-value reconciliation (earlier exploratory numbers; reconciliation only, not prediction hits)"]
    for it in res["known_value_reconciliation"]:
        L.append(f"- {it['id']} {it['what']}: known {it['known']} | observed {_fmt_obs(it['observed'])} | "
                 f"{it['rule']} -> {'MATCH' if it['match'] else 'MISMATCH'}")
    L += ["", "Ambiguities and resolutions:"]
    L += [f" {i}. {a}" for i, a in enumerate(res["ambiguities_and_resolutions"], 1)]
    return "\n".join(L)


def main() -> int:
    t0 = time.perf_counter()
    res = run_analysis()
    res["wall_time_compute_s"] = time.perf_counter() - t0
    res["identity"] = identity()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "potential_convention_results.json").write_text(json.dumps(res, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    write_csvs(res)
    summary = format_summary(res)
    print(summary)
    print(f"\nwrote {OUT / 'potential_convention_results.json'}, potential_convention_readings.csv, "
          f"potential_convention_metrics.csv; "
          f"total wall time {time.perf_counter() - t0:.2f} s")
    # machinery identity break -> non-zero exit; expectation mismatches are findings, not errors
    return 0 if res["node_branch_identity"]["all_bitwise_identical"] else 2


if __name__ == "__main__":
    sys.exit(main())
