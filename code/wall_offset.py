"""Effective wall position of the pinned Solcore grid (Supplement S4): relative change of the finite-box dipole
reference when both barriers widen by one grid step, compared with the measured Solcore deviation. Reads
results/solcore_interface/, writes results/wall_offset.json."""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.integrate import quad, simpson
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SOLCORE_DIR = REPO / "code" / "solcore_interface"
sys.path.insert(0, str(SOLCORE_DIR))
import analytic_finite_box as afb  # noqa: E402

LADDER = (0.5, 0.25, 0.125, 0.0625)                                   # nm, the Solcore ladder
INDEPENDENT = {0.5: 4.3176895836e-07, 0.25: 2.5221212452e-07, 0.125: 1.3677643129e-07, 0.0625: 7.1284490089e-08}
TOL_SELF, TOL_INDEPENDENT, CAP_ABS, CAP_FRACTION = 1e-12, 1e-4, 1e-6, 1e-2
C, V_B, A = afb.C, afb.V_B, afb.HALF_WELL                             # eV nm^2, eV, nm (half well)
B0 = afb.BARRIER                                                      # 10 nm


def _k(e):
    return math.sqrt(e / C)


def _kappa(e):
    return math.sqrt((V_B - e) / C)


def energies(b: float) -> tuple[float, float]:
    """Lowest even / odd roots with walls at |u| = A + b (u = z - centre)."""
    f_even = lambda e: _k(e) * math.tan(A * _k(e)) - _kappa(e) / math.tanh(b * _kappa(e))     # noqa: E731
    f_odd = lambda e: -_k(e) / math.tan(A * _k(e)) - _kappa(e) / math.tanh(b * _kappa(e))     # noqa: E731
    e_half, e_full = C * (math.pi / (2 * A)) ** 2, C * (math.pi / A) ** 2
    kw = dict(xtol=1e-16, rtol=4 * np.finfo(float).eps, maxiter=500)
    e0 = brentq(f_even, e_half * 1e-12, e_half * (1 - 1e-12), **kw)
    e1 = brentq(f_odd, e_half * (1 + 1e-12), e_full * (1 - 1e-12), **kw)
    return e0, e1


class State:
    """cos / sin in the well, sign-matched sinh tail vanishing at |u| = A + b; unit integral (analytic)."""

    def __init__(self, e: float, parity: str, b: float):
        self.parity, self.b = parity, b
        self.k, self.kap = _k(e), _kappa(e)
        k, kap = self.k, self.kap
        if parity == "even":
            self.amp = math.cos(A * k) / math.sinh(b * kap)
            well = A + math.sin(2 * A * k) / (2 * k)
        else:
            self.amp = math.sin(A * k) / math.sinh(b * kap)
            well = A - math.sin(2 * A * k) / (2 * k)
        barrier = math.sinh(2 * b * kap) / (4 * kap) - b / 2.0
        self.norm = 1.0 / math.sqrt(well + 2.0 * self.amp ** 2 * barrier)

    def __call__(self, u):
        u = np.asarray(u, dtype=float)
        au = np.abs(u)
        if self.parity == "even":
            w = np.cos(self.k * u)
            t = self.amp * np.sinh(self.kap * (A + self.b - au))
        else:
            w = np.sin(self.k * u)
            t = np.sign(u) * self.amp * np.sinh(self.kap * (A + self.b - au))
        return self.norm * np.where(au <= A, w, t)


def dipole(b: float) -> dict:
    """<psi_0| u |psi_1> (= <psi_0| z |psi_1> since the two states are orthogonal), adaptive quadrature on the
    three pieces, cross-checked by composite Simpson on a 1e-4 nm grid (as analytic_finite_box.py)."""
    e0, e1 = energies(b)
    s0, s1 = State(e0, "even", b), State(e1, "odd", b)
    pieces = ((-(A + b), -A), (-A, A), (A, A + b))
    q_tot = 0.0
    for lo, hi in pieces:
        val, _ = quad(lambda t: float(s0(t) * t * s1(t)), lo, hi, epsabs=1e-14, epsrel=1e-13, limit=400)
        q_tot += val
    s_tot = 0.0
    for lo, hi in pieces:
        n = int(round((hi - lo) / afb.SIMPSON_STEP))
        z = np.linspace(lo, hi, n + 1)
        s_tot += float(simpson(s0(z) * z * s1(z), x=z))
    return {"barrier_nm": b, "E0_eV": e0, "E1_eV": e1, "dipole_nm_quad": q_tot, "dipole_nm_simpson": s_tot,
            "quad_vs_simpson_rel": abs(q_tot - s_tot) / abs(q_tot), "amplitude_nm": abs(q_tot),
            "kappa1_per_nm": _kappa(e1)}


def main() -> None:
    ref = afb.reference()
    base = dipole(B0)
    self_check = {
        "dipole_rel": abs(base["amplitude_nm"] / ref["dipole_amplitude_nm"] - 1.0),
        "E0_rel": abs(base["E0_eV"] / ref["E0_eV"] - 1.0), "E1_rel": abs(base["E1_eV"] / ref["E1_eV"] - 1.0)}
    self_check["pass"] = all(v <= TOL_SELF for v in self_check.values())
    conv = json.loads((REPO / "results/solcore_interface/solcore_results.json").read_text(encoding="utf-8"))
    dev = {v: conv["convergence"][v]["relative_deviation_vs_reference"] for v in ("node", "cell")}
    rows = []
    for i, h in enumerate(LADDER):
        d = dipole(B0 + h)
        rel = d["amplitude_nm"] / base["amplitude_nm"] - 1.0
        ratio_ref = rel / INDEPENDENT[h] - 1.0
        frac = {v: abs(rel) / abs(dev[v][i]) for v in ("node", "cell")}
        rows.append({"h_nm": h, "barrier_nm": B0 + h, "amplitude_nm": d["amplitude_nm"],
                     "relative_change": rel, "independent_value": INDEPENDENT[h], "ours_over_independent_minus_1": ratio_ref,
                     "E0_shift_rel": d["E0_eV"] / base["E0_eV"] - 1.0, "E1_shift_rel": d["E1_eV"] / base["E1_eV"] - 1.0,
                     "quad_vs_simpson_rel": d["quad_vs_simpson_rel"],
                     "measured_solcore_deviation": {v: dev[v][i] for v in ("node", "cell")},
                     "fraction_of_measured_deviation": frac,
                     "E2_pass": rel > 0 and abs(ratio_ref) <= TOL_INDEPENDENT,
                     "E3_pass": abs(rel) <= CAP_ABS and max(frac.values()) <= CAP_FRACTION})
    k1 = base["kappa1_per_nm"]
    shape = {"kappa1_per_nm": k1,
             "ratio_h0.5_over_h0.0625": rows[0]["relative_change"] / rows[-1]["relative_change"],
             "one_minus_exp_ratio": (1 - math.exp(-2 * k1 * 0.5)) / (1 - math.exp(-2 * k1 * 0.0625))}
    out = {"check": "effective wall position of the pinned Solcore grid", "expectations": "docstring of this script (fixed before the first run)",
           "timestamp_utc": datetime.now(timezone.utc).isoformat(),
           "python": sys.version.split()[0], "numpy": np.__version__,
           "reference_B10": {"amplitude_nm": base["amplitude_nm"], "E0_eV": base["E0_eV"], "E1_eV": base["E1_eV"],
                             "quad_vs_simpson_rel": base["quad_vs_simpson_rel"]},
           "E1_self_check": self_check, "rows": rows, "shape": shape,
           "verdict": {"E1": self_check["pass"], "E2": all(r["E2_pass"] for r in rows),
                       "E3": all(r["E3_pass"] for r in rows)}}
    out["verdict"]["all"] = all(out["verdict"].values())
    (REPO / "results").mkdir(exist_ok=True)
    (REPO / "results" / "wall_offset.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"B=10 self-check {self_check}")
    for r in rows:
        print(f"  h={r['h_nm']:<7} rel={r['relative_change']:+.10e}  independent {r['independent_value']:+.10e}  "
              f"ours/ind-1={r['ours_over_independent_minus_1']:+.2e}  frac node {r['fraction_of_measured_deviation']['node']:.2e}"
              f"  cell {r['fraction_of_measured_deviation']['cell']:.2e}  E2 {r['E2_pass']}  E3 {r['E3_pass']}")
    print(f"  shape: ratio {shape['ratio_h0.5_over_h0.0625']:.4f} vs (1-e^-2k1h) ratio {shape['one_minus_exp_ratio']:.4f}")
    print("verdict", out["verdict"])


if __name__ == "__main__":
    main()
