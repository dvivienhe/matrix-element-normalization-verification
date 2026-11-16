"""Piecewise-analytic reference for the Solcore example: finite box [0, 30] nm, square well [10, 20] nm,
V_b = 0.4 eV, mass 0.067 m0; energies, normalized states and the dipole <psi_0|z|psi_1>. Pure NumPy / SciPy."""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad, simpson
from scipy.optimize import brentq

C0_EV_NM2 = 0.0380998211615486      # hbar^2 / (2 m0), CODATA-2018 (same value as toy_model.py)
M_EFF = 0.067
C = C0_EV_NM2 / M_EFF               # eV nm^2
V_B = 0.4                           # eV
HALF_WELL = 5.0                     # nm
BARRIER = 10.0                      # nm (each side, wall to interface)
CENTER = 15.0                       # nm
SIMPSON_STEP = 1e-4                 # nm


def _k(e):
    return math.sqrt(e / C)


def _kappa(e):
    return math.sqrt((V_B - e) / C)


def f_even(e: float) -> float:
    k, kap = _k(e), _kappa(e)
    return k * math.tan(HALF_WELL * k) - kap / math.tanh(BARRIER * kap)


def f_odd(e: float) -> float:
    k, kap = _k(e), _kappa(e)
    return -k / math.tan(HALF_WELL * k) - kap / math.tanh(BARRIER * kap)


def ground_and_first() -> dict:
    """Energies (eV) of the lowest even and lowest odd state."""
    e_half = C * (math.pi / (2.0 * HALF_WELL)) ** 2          # 5k = pi/2
    e_full = C * (math.pi / HALF_WELL) ** 2                  # 5k = pi
    assert e_full < V_B
    e0 = brentq(f_even, e_half * 1e-12, e_half * (1.0 - 1e-12), xtol=1e-16, rtol=4 * np.finfo(float).eps,
                maxiter=500)
    e1 = brentq(f_odd, e_half * (1.0 + 1e-12), e_full * (1.0 - 1e-12), xtol=1e-16, rtol=4 * np.finfo(float).eps,
                maxiter=500)
    return {"E0_eV": e0, "E1_eV": e1}


class State:
    """One analytic eigenstate, normalized to unit integral over [0, 30] nm."""

    def __init__(self, energy: float, parity: str):
        self.E, self.parity = energy, parity
        self.k, self.kap = _k(energy), _kappa(energy)
        k, kap = self.k, self.kap
        if parity == "even":
            self.amp = math.cos(HALF_WELL * k) / math.sinh(BARRIER * kap)
            well = HALF_WELL + math.sin(2 * HALF_WELL * k) / (2 * k)
        else:
            self.amp = math.sin(HALF_WELL * k) / math.sinh(BARRIER * kap)
            well = HALF_WELL - math.sin(2 * HALF_WELL * k) / (2 * k)
        barrier = math.sinh(2 * BARRIER * kap) / (4 * kap) - BARRIER / 2.0
        self.norm = 1.0 / math.sqrt(well + 2.0 * self.amp ** 2 * barrier)

    def __call__(self, z):
        z = np.asarray(z, dtype=float)
        u = z - CENTER
        au = np.abs(u)
        inside = au <= HALF_WELL
        if self.parity == "even":
            w = np.cos(self.k * u)
            b = self.amp * np.sinh(self.kap * (CENTER - au))
        else:
            w = np.sin(self.k * u)
            b = np.sign(u) * self.amp * np.sinh(self.kap * (CENTER - au))
        return self.norm * np.where(inside, w, b)


def _pieces(func):
    """Adaptive quadrature on [0,10], [10,20], [20,30] (kinks of psi'' at the interfaces)."""
    total, err = 0.0, 0.0
    for a, b in ((0.0, 10.0), (10.0, 20.0), (20.0, 30.0)):
        val, e = quad(lambda t: float(func(t)), a, b, epsabs=1e-14, epsrel=1e-13, limit=400)
        total += val
        err += e
    return total, err


def _simpson(func):
    total = 0.0
    for a, b in ((0.0, 10.0), (10.0, 20.0), (20.0, 30.0)):
        n = int(round((b - a) / SIMPSON_STEP))
        z = np.linspace(a, b, n + 1)
        total += float(simpson(func(z), x=z))
    return total


def reference() -> dict:
    en = ground_and_first()
    s0, s1 = State(en["E0_eV"], "even"), State(en["E1_eV"], "odd")
    dip_q, dip_err = _pieces(lambda t: s0(t) * t * s1(t))
    dip_s = _simpson(lambda z: s0(z) * z * s1(z))
    n0_q, _ = _pieces(lambda t: s0(t) ** 2)
    n1_q, _ = _pieces(lambda t: s1(t) ** 2)
    ov_q, _ = _pieces(lambda t: s0(t) * s1(t))
    return {
        "E0_eV": en["E0_eV"], "E1_eV": en["E1_eV"],
        "eigen_residual_even": f_even(en["E0_eV"]), "eigen_residual_odd": f_odd(en["E1_eV"]),
        "norm0": n0_q, "norm1": n1_q, "overlap01": ov_q,
        "dipole_nm_quad": dip_q, "dipole_quad_error_estimate": dip_err, "dipole_nm_simpson": dip_s,
        "dipole_rel_diff_quad_vs_simpson": abs(dip_q - dip_s) / abs(dip_q),
        "dipole_amplitude_nm": abs(dip_q), "rate_nm2": dip_q * dip_q,
        "psi_at_walls": [float(s0(0.0)), float(s0(30.0)), float(s1(0.0)), float(s1(30.0))],
        "constants": {"C0_eV_nm2": C0_EV_NM2, "m_eff": M_EFF, "V_b_eV": V_B, "well_nm": [10.0, 20.0],
                      "box_nm": [0.0, 30.0], "codata": "2018", "simpson_step_nm": SIMPSON_STEP},
    }


if __name__ == "__main__":
    import json
    print(json.dumps(reference(), indent=1))
