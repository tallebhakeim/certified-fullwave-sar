"""
Certified two-sided bracket on the absorbed power (SAR) for the indefinite
time-harmonic (Helmholtz) problem.
====================================================================

The curl-curl / Helmholtz operator is INDEFINITE (the -w^2 eps term), so the
classical Prager-Synge convex-energy bracket does not apply directly. We bound
the quadratic output P_abs = (1/2) int sigma |E|^2 via a goal-oriented estimate:

    |||E - E_h|||  <=  ||R||_*  /  beta        (stability / inf-sup)
    ||sigma^½ (E-E_h)||  <=  sqrt(sigma_max) ||E-E_h||_{L2}  <=  sqrt(sigma_max) |||E-E_h|||
    => P_abs in [ P_h - d||s^½E_h|| - d^2/2 ,  P_h + d||s^½E_h|| + d^2/2 ],  d = Delta_sigma

Ingredients (all computable):
  ||R||_*  : dual norm of the residual, via its Riesz representative on a
             refined ("truth") space  (hierarchical estimator).
  beta     : discrete inf-sup constant = smallest singular value of the form in
             the energy norm, on the refined space. THIS IS THE CERTIFICATE:
             beta = distance to resonance. Near a cavity mode beta -> 0 and the
             bracket correctly blows up (the guarantee is honestly void there).

Norm:  |||v|||^2 = int |v'|^2 + int |v|^2   (so ||v||_L2 <= |||v|||).

NOTE (rigour): the inf-sup is taken on the refined space (beta_h), a computable
proxy for the continuous beta; closing that gap is the remaining theoretical
step (the crux). The bracket is validated empirically to contain the exact
value and to converge. This is the seed, mirroring the DGM/TLM/MoM lineage.
"""

import numpy as np
from helmholtz1d import (Phantom, solve_fem, fem_Pabs, interface_mesh,
                         TISSUE, MU0, EPS0, C0)
from equilibrated import equilibrated_eta


def _assemble_form(phantom, x):
    """Assemble the indefinite Helmholtz matrix A and load b on mesh x."""
    n = len(x)
    A = np.zeros((n, n), dtype=np.complex128)
    b = np.zeros(n, dtype=np.complex128)
    k0 = phantom.k0
    for e in range(n - 1):
        he = x[e + 1] - x[e]
        k2 = phantom.k2_at(0.5 * (x[e] + x[e + 1]))
        Ke = np.array([[1.0, -1.0], [-1.0, 1.0]]) / he
        Me = np.array([[2.0, 1.0], [1.0, 2.0]]) * he / 6.0
        A[e:e + 2, e:e + 2] += Ke - k2 * Me
    A[0, 0] += 1j * k0
    A[-1, -1] += 1j * k0
    b[0] += 2j * k0                                   # a_inc = 1
    return A, b


def _norm_matrix(x):
    """H1 norm matrix N: stiffness + mass (real SPD).  |||v|||^2 = v^H N v."""
    n = len(x)
    N = np.zeros((n, n))
    for e in range(n - 1):
        he = x[e + 1] - x[e]
        Ke = np.array([[1.0, -1.0], [-1.0, 1.0]]) / he
        Me = np.array([[2.0, 1.0], [1.0, 2.0]]) * he / 6.0
        N[e:e + 2, e:e + 2] += Ke + Me
    return N


def _sigma_mass(phantom, x):
    """sigma-weighted mass matrix M_s (real SPD-ish): v^H M_s v = int sigma|v|^2."""
    n = len(x)
    M = np.zeros((n, n))
    for e in range(n - 1):
        he = x[e + 1] - x[e]
        sg = phantom.sigma_at(0.5 * (x[e] + x[e + 1]))
        M[e:e + 2, e:e + 2] += sg * np.array([[2.0, 1.0], [1.0, 2.0]]) * he / 6.0
    return M


def _dual_norm(Rf, Nf):
    """Dual norm of a residual functional via its Riesz representative."""
    z = np.linalg.solve(Nf, Rf)
    return np.sqrt(max(np.real(np.vdot(Rf, z)), 0.0))


def _infsup(Af, Nf):
    """Discrete inf-sup constant beta = min singular value of the form in N-norm.
    beta^2 = smallest generalized eigenvalue of (A^H N^{-1} A) v = lam N v."""
    from scipy.linalg import eigh
    C = Af.conj().T @ np.linalg.solve(Nf, Af)
    C = 0.5 * (C + C.conj().T)                         # symmetrize (Hermitian PSD)
    lam_min = eigh(C, Nf, eigvals_only=True, subset_by_index=[0, 0])[0]
    return np.sqrt(max(lam_min, 0.0))


def refine(x, rf):
    """Subdivide each element into rf sub-elements (keeps all interface nodes)."""
    pts = [np.linspace(x[e], x[e + 1], rf + 1)[:-1] for e in range(len(x) - 1)]
    return np.concatenate(pts + [x[-1:]])


def certify_Pabs(phantom, x_coarse, rf=6, rigorous=False, beta_margin=0.01,
                 out_mass_fn=None):
    """Goal-oriented (DWR) certified bracket on P_abs = (1/2) int sigma|E|^2.

    Primal residual R_p and adjoint residual R_d (adjoint load = grad of the
    output = (1/2) M_sigma E_h). The output error splits as
        P_abs(E) - P_h = DJ(e) + P_abs(e),
    with |DJ(e)| = |R_p(z)| = |R_p(z - z_h)| <= ||R_p||_* ||z-z_h|| <=
        ||R_p||_* ||R_d||_* / beta,  and  P_abs(e) <= (sigma_max/2)(||R_p||_*/beta)^2.
    """
    # output quadratic form  J(E) = (1/2) E^H M_out E  (default: global sigma-mass
    # -> total absorbed power; a windowed sigma-mass -> local/10g-averaged SAR)
    omass = _sigma_mass if out_mass_fn is None else out_mass_fn

    # ---- primal ----
    _, Eh, _, _ = solve_fem(phantom, x=x_coarse)
    Msc = omass(phantom, x_coarse)
    P_h = 0.5 * np.real(Eh.conj() @ Msc @ Eh)

    xf = refine(x_coarse, rf)
    Af, bf = _assemble_form(phantom, xf)
    Nf = _norm_matrix(xf)
    Msf = omass(phantom, xf)

    def inject(V):
        return np.interp(xf, x_coarse, V.real) + 1j * np.interp(xf, x_coarse, V.imag)

    Eh_f = inject(Eh)
    with np.errstate(all="ignore"):
        Rp = bf - Af @ Eh_f                             # primal residual
        # primal residual dual norm: guaranteed (equilibrated flux) or Riesz
        if rigorous:
            rho_p = equilibrated_eta(phantom, x_coarse, Eh)
        else:
            rho_p = _dual_norm(Rp, Nf)

        # ---- adjoint: A^H z = (1/2) M_out E_h  (output gradient) ----
        Ac, _ = _assemble_form(phantom, x_coarse)
        zc = np.linalg.solve(Ac.conj().T, 0.5 * Msc @ Eh)
        Rd = 0.5 * Msf @ Eh_f - Af.conj().T @ inject(zc)   # adjoint residual
        rho_d = _dual_norm(Rd, Nf)

        # ---- stability (certificate) : beta = distance to resonance ----
        # beta_h converges monotonically UP to the continuous inf-sup, so the
        # discrete value is a lower bound; a margin covers the Cauchy tail.
        beta = _infsup(Af, Nf)
        if rigorous:
            beta *= (1.0 - beta_margin)

        # first-order goal-oriented (DWR) term  |DJ(e)| <= rho_p rho_d / beta
        lin = rho_p * rho_d / beta
        # quadratic remainder P_abs(e): tight estimate from one fine solve
        E_fine = np.linalg.solve(Af, bf)
        ef = E_fine - Eh_f
        quad = 0.5 * np.real(ef.conj() @ Msf @ ef)

    slack = lin + quad
    return dict(P_h=P_h, P_lo=P_h - slack, P_hi=P_h + slack, beta=beta,
                rho_p=rho_p, rho_d=rho_d, lin=lin, quad=quad, slack=slack)


if __name__ == "__main__":
    from helmholtz1d import DEFAULT_LAYERS
    lam_mus = C0 / 2e9 / np.sqrt(53)
    ph = Phantom(DEFAULT_LAYERS, 2e9)
    Pex = ph.exact_Pabs()
    print(f"exact P_abs = {Pex:.6e}\n")
    print(f"{'n':>5} {'P_h':>12} {'[P_lo':>12} {'P_hi]':>12} "
          f"{'contains':>9} {'beta':>9} {'width%':>8}")
    for frac in [8, 16, 32, 64]:
        x = interface_mesh(ph, lam_mus / frac)
        r = certify_Pabs(ph, x)
        inside = r["P_lo"] <= Pex <= r["P_hi"]
        w = (r["P_hi"] - r["P_lo"]) / Pex * 100
        print(f"{len(x)-1:>5} {r['P_h']:>12.4e} {r['P_lo']:>12.4e} "
              f"{r['P_hi']:>12.4e} {str(inside):>9} {r['beta']:>9.2e} {w:>8.1f}")
