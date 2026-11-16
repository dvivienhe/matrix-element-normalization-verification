"""Toy verification model: analytic (leg A) and finite-difference (leg B) states on a uniform grid and the
consumers C1-C7c whose grid-step exponents the controlled validation predicts. Pure NumPy; imported by the
other programs, never run on its own."""
from __future__ import annotations

import numpy as np

# ------------------------------------------------------------ declared toys --
L_WELL_NM = 10.0          # Leg-A infinite well width
Z0_BARDEEN_A = 3.6        # Leg-A matching plane (on-grid for all rungs)

DOMAIN_B_NM = 30.0        # Leg-B domain, Dirichlet
WELL_B_NM = (10.0, 20.0)  # Leg-B well interval
V0_B_EV = 0.5             # Leg-B barrier height
MSTAR_B = 0.04            # Leg-B effective mass (units of m0) — toy value
Z0_BARDEEN_B = 20.0       # Leg-B matching plane (right well edge)

M_Z_TOY = 0.25            # C6 "quantization mass" (toy)
M_PAR_TOY = 0.055         # C6 "in-plane DOS mass" (toy)

LAMBDA_S_NM = 2.0         # C7 screening length (toy value, not a material claim)
LAMBDA_S_VARIANT_NM = 0.5 # C7 kernel-variant screening length (structural row)

DZ_LADDER = (0.40, 0.20, 0.10, 0.05)
DZ_REF_B = 0.0125         # Leg-B reference rung

# hbar^2/(2 m0) in eV nm^2, computed from CODATA-2018 base constants
_HBAR = 1.054571817e-34   # J s
_M0 = 9.1093837015e-31    # kg
_E = 1.602176634e-19      # C (exact)
HBAR2_OVER_2M0 = _HBAR**2 / (2.0 * _M0 * _E) * 1e18   # ~0.0381 eV nm^2


# ------------------------------------------------------------------- Leg A --
def manufactured_psi(n: int, z: np.ndarray) -> np.ndarray:
    """Infinite-well eigenfunction, continuum L2-normalized (length^-1/2)."""
    return np.sqrt(2.0 / L_WELL_NM) * np.sin(n * np.pi * z / L_WELL_NM)


def manufactured_dpsi(n: int, z: np.ndarray) -> np.ndarray:
    """Analytic derivative of manufactured_psi."""
    k = n * np.pi / L_WELL_NM
    return np.sqrt(2.0 / L_WELL_NM) * k * np.cos(k * z)


def grid_a(dz: float) -> np.ndarray:
    """Interior grid points of [0, L]; boundary points (psi = 0) excluded."""
    n = int(round(L_WELL_NM / dz))
    if abs(n * dz - L_WELL_NM) > 1e-9:
        raise ValueError(f"dz={dz} does not divide L={L_WELL_NM}")
    return np.arange(1, n) * dz


def solver_contract(psi_samples: np.ndarray) -> np.ndarray:
    """Mimic the eigensolver contract: return the Euclidean-normalized vector."""
    return psi_samples / np.sqrt(np.sum(np.abs(psi_samples) ** 2))


def leg_a_states(dz: float):
    z = grid_a(dz)
    c1 = solver_contract(manufactured_psi(1, z))
    c2 = solver_contract(manufactured_psi(2, z))
    return z, c1, c2


# ------------------------------------------------------------------- Leg B --
def fd_well_solve(dz: float):
    """Single-band FD eigensolve; returns (z, c_ground, c_first) with the
    standard solver contract (Euclidean-normalized eigenvectors)."""
    n = int(round(DOMAIN_B_NM / dz))
    if abs(n * dz - DOMAIN_B_NM) > 1e-9:
        raise ValueError(f"dz={dz} does not divide domain={DOMAIN_B_NM}")
    z = np.arange(1, n) * dz                       # interior (Dirichlet)
    v = np.where((z >= WELL_B_NM[0]) & (z <= WELL_B_NM[1]), 0.0, V0_B_EV)
    t = HBAR2_OVER_2M0 / (MSTAR_B * dz * dz)       # hbar^2/(2 m* dz^2)
    h = np.diag(2.0 * t + v) + np.diag(-t * np.ones(len(z) - 1), 1) \
        + np.diag(-t * np.ones(len(z) - 1), -1)
    w, vecs = np.linalg.eigh(h)
    order = np.argsort(w)
    c1, c2 = vecs[:, order[0]], vecs[:, order[1]]
    return z, c1, c2


# ------------------------------------------------------- the six consumers --
def _central_diff(c: np.ndarray, dz: float) -> np.ndarray:
    d = np.zeros_like(c)
    d[1:-1] = (c[2:] - c[:-2]) / (2.0 * dz)
    return d


def c1_correct_dipole(z, ca, cb, dz) -> float:
    """|<a|z|b>|^2 with the proper conversion psi = c/sqrt(dz), trapezoid."""
    pa, pb = ca / np.sqrt(dz), cb / np.sqrt(dz)
    m = np.trapezoid(pa * z * pb, z)
    return float(m * m)


def c2_bug_dipole(z, ca, cb, dz) -> float:
    """Raw c fed into the same continuum trapezoid (n=2, m=1 bug)."""
    m = np.trapezoid(ca * z * cb, z)
    return float(m * m)


def c3_bug_bardeen(z, ca, cb, dz, z0) -> float:
    """Raw c into the point-value Bardeen form (n=2, m=0 bug)."""
    i0 = int(round(z0 / dz)) - 1                   # z[i] = (i+1) dz
    da, db = _central_diff(ca, dz), _central_diff(cb, dz)
    t = ca[i0] * db[i0] - cb[i0] * da[i0]
    return float(t * t)


def c4_immune_mean(z, ca, cb, dz) -> float:
    """Consumes raw c but internally renormalizes int P dz = 1 (immunity)."""
    p = np.abs(ca) ** 2
    p = p / np.trapezoid(p, z)                     # the incidental renorm
    return float(np.trapezoid(z * p, z))           # <w> with w(z) = z


def c5_unit_mix_dipole(z, ca, cb, dz) -> float:
    """C1's conversion formula but with dz numerically in Angstrom while the
    coordinates stay in nm — each envelope factor short by sqrt(10)."""
    dz_wrong = dz * 10.0
    pa, pb = ca / np.sqrt(dz_wrong), cb / np.sqrt(dz_wrong)
    m = np.trapezoid(pa * z * pb, z)
    return float(m * m)


def c6_wrong_mass_dos(use_wrong: bool = True) -> float:
    """Toy 2D DOS rho~ = m/pi (toy units m0/(pi hbar^2)); wrong-axis mass."""
    m = M_Z_TOY if use_wrong else M_PAR_TOY
    return float(m / np.pi)


# ----------------------- C7: four-factor screened-Coulomb-like -------------
def _coulomb_M_direct(z: np.ndarray, f: np.ndarray, lam: float) -> float:
    """M = iint f(z) exp(-|z-z'|/lam) f(z') dz dz' by double trapezoid
    (kernel prefactor 1, dimensionless toy kernel)."""
    kern = np.exp(-np.abs(z[:, None] - z[None, :]) / lam)
    inner = np.trapezoid(kern * f[None, :], z, axis=1)
    return float(np.trapezoid(f * inner, z))


def _coulomb_M_prefix(z: np.ndarray, f: np.ndarray, lam: float) -> float:
    """Same M via exponential-kernel separability (O(N) prefix route):
    I(z) = e^{-z/lam} int_0^z e^{+z'/lam} f dz' + e^{+z/lam} int_z^L e^{-z'/lam} f dz'.
    Independent of the direct route; used for the Leg-A dual-route reference."""
    dz = z[1] - z[0]
    g1 = np.exp(z / lam) * f
    g2 = np.exp(-z / lam) * f
    a = np.concatenate(([0.0], np.cumsum(0.5 * (g1[1:] + g1[:-1]) * dz)))
    g2c = np.concatenate(([0.0], np.cumsum(0.5 * (g2[1:] + g2[:-1]) * dz)))
    b = g2c[-1] - g2c
    inner = np.exp(-z / lam) * a + np.exp(z / lam) * b
    return float(np.trapezoid(f * inner, z))


def c7_bug_coulomb(z, ca, cb, dz, lam: float = LAMBDA_S_NM) -> float:
    """All FOUR envelope factors as raw c in the double trapezoid
    (n = 4, m = 2 bug); rate level R = M^2."""
    f = ca * cb
    return _coulomb_M_direct(z, f, lam) ** 2


def c7_correct_coulomb(z, ca, cb, dz, lam: float = LAMBDA_S_NM) -> float:
    """psi = c/sqrt(dz) conversion into the same double trapezoid; R = M^2."""
    pa, pb = ca / np.sqrt(dz), cb / np.sqrt(dz)
    f = pa * pb
    return _coulomb_M_direct(z, f, lam) ** 2


# ------------------------------------------------------------- references --
def ref_leg_a() -> dict:
    """Leg-A references: closed forms cross-checked by fine quadrature."""
    # dipole <1|z|2> = -16 L / (9 pi^2)  (standard infinite-well result)
    m_closed = -16.0 * L_WELL_NM / (9.0 * np.pi ** 2)
    zf = np.linspace(0.0, L_WELL_NM, 2 ** 15 + 1)
    m_quad = np.trapezoid(manufactured_psi(1, zf) * zf * manufactured_psi(2, zf), zf)
    t = (manufactured_psi(1, np.array([Z0_BARDEEN_A]))[0]
         * manufactured_dpsi(2, np.array([Z0_BARDEEN_A]))[0]
         - manufactured_psi(2, np.array([Z0_BARDEEN_A]))[0]
         * manufactured_dpsi(1, np.array([Z0_BARDEEN_A]))[0])
    # C7 reference: dual route on the analytic transition density
    zc = np.linspace(0.0, L_WELL_NM, 2 ** 12 + 1)
    fc = manufactured_psi(1, zc) * manufactured_psi(2, zc)
    m7_direct = _coulomb_M_direct(zc, fc, LAMBDA_S_NM)
    zp = np.linspace(0.0, L_WELL_NM, 2 ** 15 + 1)
    fp = manufactured_psi(1, zp) * manufactured_psi(2, zp)
    m7_prefix = _coulomb_M_prefix(zp, fp, LAMBDA_S_NM)
    return {
        "dipole_rate": float(m_closed ** 2),
        "dipole_rate_quad": float(m_quad ** 2),
        "bardeen_rate": float(t * t),
        "mean_w": L_WELL_NM / 2.0,                 # symmetric |psi_1|^2
        "dos": float(M_PAR_TOY / np.pi),
        "coulomb_rate": float(m7_prefix ** 2),
        "coulomb_rate_direct": float(m7_direct ** 2),
    }


def ref_leg_b() -> dict:
    """Leg-B reference: the correct consumers at the reference rung dz_ref."""
    z, c1, c2 = fd_well_solve(DZ_REF_B)
    pa = c1 / np.sqrt(DZ_REF_B)
    return {
        "dipole_rate": c1_correct_dipole(z, c1, c2, DZ_REF_B),
        "bardeen_rate": _bardeen_correct(z, c1, c2, DZ_REF_B, Z0_BARDEEN_B),
        "mean_w": c4_immune_mean(z, c1, c2, DZ_REF_B),
        "dos": float(M_PAR_TOY / np.pi),
        "coulomb_rate": c7_correct_coulomb(z, c1, c2, DZ_REF_B),
    }


def _bardeen_correct(z, ca, cb, dz, z0) -> float:
    """Correct (converted) Bardeen rate — used only to build Leg-B references
    and the per-rung C3 expected factors."""
    pa, pb = ca / np.sqrt(dz), cb / np.sqrt(dz)
    i0 = int(round(z0 / dz)) - 1
    da, db = _central_diff(pa, dz), _central_diff(pb, dz)
    t = pa[i0] * db[i0] - pb[i0] * da[i0]
    return float(t * t)


# ------------------------------------------------------------------ sweeps --
def sweep(leg: str) -> dict:
    """Run the six consumers over the dz ladder. Returns raw readings."""
    states = leg_a_states if leg == "A" else fd_well_solve
    z0 = Z0_BARDEEN_A if leg == "A" else Z0_BARDEEN_B
    out = {"dz": list(DZ_LADDER), "C1": [], "C2": [], "C3": [], "C4": [],
           "C5": [], "C6": [], "C7": [], "C7c": [], "C3_correct": []}
    for dz in DZ_LADDER:
        z, ca, cb = states(dz)
        out["C1"].append(c1_correct_dipole(z, ca, cb, dz))
        out["C2"].append(c2_bug_dipole(z, ca, cb, dz))
        out["C3"].append(c3_bug_bardeen(z, ca, cb, dz, z0))
        out["C4"].append(c4_immune_mean(z, ca, cb, dz))
        out["C5"].append(c5_unit_mix_dipole(z, ca, cb, dz))
        out["C6"].append(c6_wrong_mass_dos(use_wrong=True))
        out["C7"].append(c7_bug_coulomb(z, ca, cb, dz))
        out["C7c"].append(c7_correct_coulomb(z, ca, cb, dz))
        out["C3_correct"].append(_bardeen_correct(z, ca, cb, dz, z0))
    return out


def sweep_c7_variant(lam: float = LAMBDA_S_VARIANT_NM) -> dict:
    """Leg-A C7/C7c under the variant kernel (structural row:
        the kernel contributes no step powers).
    """
    out = {"dz": list(DZ_LADDER), "lam": lam, "C7": [], "C7c": []}
    for dz in DZ_LADDER:
        z, ca, cb = leg_a_states(dz)
        out["C7"].append(c7_bug_coulomb(z, ca, cb, dz, lam))
        out["C7c"].append(c7_correct_coulomb(z, ca, cb, dz, lam))
    return out


def fit_slope(dz: list, values: list) -> float:
    """Least-squares slope of log|V| vs log(dz)."""
    x = np.log(np.asarray(dz))
    y = np.log(np.abs(np.asarray(values)))
    return float(np.polyfit(x, y, 1)[0])
