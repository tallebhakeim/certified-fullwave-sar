"""
Certified SAR bracket in 3D on the curl-curl edge-element problem (marche 4, 3D).
================================================================================

Ports the goal-oriented (DWR) certified bracket of certify2d.py to the 3D Whitney
edge-element solver edge_fem3d. Output = absorbed power P_abs = (1/2) int sigma
|E|^2 dV (the SAR quantity, here the 3D volume integral).

Validation by a manufactured lossy plane wave  E = y_hat exp(-j k x)  in a
homogeneous lossy medium: curl curl E - k^2 E = 0 with zero volume source, so a
Dirichlet edge-circulation boundary condition drives it and the exact P_abs is
known in closed form. This isolates the DISCRETIZATION error (the machinery).

Bracket (R1): P_abs in [P_h - 2 rho_p rho_d/beta, P_h + 2 rho_p rho_d/beta + (sigma/2)(rho_p/beta)^2/w_m],
residuals in the H(curl) energy norm |||v|||^2 = (1/mu)|curl v|^2 + w^2|eps_c||v|^2,
beta the discrete inf-sup (= distance to resonance) on the interior space.

Run:  python3 certify3d.py
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import eigh
from edge_fem3d import (build_mesh, build_edges, assemble, edge_circulation,
                        boundary_edges, solve_dirichlet, field_at_points,
                        batch_geom, batch_curl, _quad_from_metric, _scatter,
                        LOCAL_EDGES, EPS0, MU0, C0)


def assemble_real(nodes, tets, elem_edges, elem_sign, weight_mass, weight_curl):
    """sum_e ( wcurl * curl.curl + wmass_e * iso-mass ), real sparse. Vectorised."""
    Ne = int(elem_edges.max()) + 1
    vol, grads = batch_geom(nodes, tets)
    cvec = batch_curl(grads)                               # (Nt,6,3)
    Scc = np.einsum('tmd,tnd->tmn', cvec, cvec) * vol[:, None, None]
    metric = np.einsum('tad,tbd->tab', grads, grads)       # (Nt,4,4)
    Miso = _quad_from_metric(metric, vol)                  # (Nt,6,6)
    wm = np.asarray(weight_mass)
    Ae = weight_curl * Scc + wm[:, None, None] * Miso
    return _scatter(elem_edges, elem_sign, Ae, Ne)


_GP = np.array([0.5 - np.sqrt(15) / 10, 0.5, 0.5 + np.sqrt(15) / 10])
_GW = np.array([5 / 18, 8 / 18, 5 / 18])


def inject_edge(coarse, xf_nodes, edges_f):
    """Fine-edge circulations of the coarse edge field (injection C -> F),
    batched: all Gauss points of all fine edges evaluated in one call."""
    cn, ct, cee, ces, cE, cN, cL = coarse
    A = xf_nodes[edges_f[:, 0]]; B = xf_nodes[edges_f[:, 1]]
    d = B - A                                              # (Ne,3)
    pts = A[:, None, :] + _GP[None, :, None] * d[:, None, :]    # (Ne,3,3)
    vals = field_at_points(cn, ct, cee, ces, cE, pts.reshape(-1, 3), cN, cL)
    vals = vals.reshape(edges_f.shape[0], 3, 3)           # (Ne,gauss,vec)
    return np.einsum('g,egc,ec->e', _GW, vals, d)


def _make_dnorm(N):
    """Factor the (real SPD) energy norm once; reuse for every dual norm."""
    lu = spla.splu(N.tocsc())
    def dnorm(R):
        z = lu.solve(R.real) + 1j * lu.solve(R.imag)      # real factor, cplx RHS
        return np.sqrt(max(np.real(np.vdot(R, z)), 0.0))
    return dnorm


def _infsup(A, N):
    """beta^2 = min eig (A^H N^-1 A, N) on the interior space (dense, cheap)."""
    Ad = A.toarray(); Nd = N.toarray()
    C = Ad.conj().T @ np.linalg.solve(Nd, Ad)
    C = 0.5 * (C + C.conj().T)
    lam = eigh(C, Nd, eigvals_only=True, subset_by_index=[0, 0])[0]
    return np.sqrt(max(lam, 0.0))


def exact_Pabs(sigma, k, L):
    """(1/2) sigma int_[0,L]^3 |exp(-j k x)|^2 dV, k complex (Im k < 0)."""
    a = 2.0 * np.imag(k)
    ix = (np.exp(a * L) - 1.0) / a if abs(a) > 1e-30 else L
    return 0.5 * sigma * (L * L) * ix                      # L^2 from y,z extents


def certify_3d(freq, eps_r, sigma, L, N, rf=2, method="direct"):
    w = 2 * np.pi * freq
    eps_c = EPS0 * eps_r - 1j * sigma / w
    k = w * np.sqrt(MU0 * eps_c)
    Efn = lambda p: np.array([0.0, 1.0, 0.0]) * np.exp(-1j * k * p[0])

    # coarse solve (manufactured Dirichlet)
    nodes, tets = build_mesh(L, N)
    edges, ee, es = build_edges(tets)
    epsc = np.full(tets.shape[0], eps_c)
    A, Ne = assemble(nodes, tets, ee, es, epsc, freq)
    bedges = boundary_edges(nodes, edges, L)
    dvals = edge_circulation(nodes, edges[bedges], Efn)
    E = solve_dirichlet(A, Ne, bedges, dvals)

    # natural (balanced) energy norm:  (1/mu)|curl|^2 + w^2|eps_c| |v|^2
    wcurl = 1.0 / MU0
    wmass = w * w * np.abs(eps_c)
    sig_elem = np.full(tets.shape[0], sigma)
    Msig = assemble_real(nodes, tets, ee, es, sig_elem, 0.0)
    Nrm_c = assemble_real(nodes, tets, ee, es,
                          np.full(tets.shape[0], wmass), wcurl)
    P_h = 0.5 * np.real(np.vdot(E, Msig @ E))
    free_c = np.setdiff1d(np.arange(Ne), bedges)

    # ---- refined space ----
    Nf_ = (N - 1) * rf + 1
    nodesf, tetsf = build_mesh(L, Nf_)
    edgesf, eef, esf = build_edges(tetsf)
    epscf = np.full(tetsf.shape[0], eps_c)
    Af, Nef = assemble(nodesf, tetsf, eef, esf, epscf, freq)
    Nrm = assemble_real(nodesf, tetsf, eef, esf,
                        np.full(tetsf.shape[0], wmass), wcurl)
    Msigf = assemble_real(nodesf, tetsf, eef, esf,
                          np.full(tetsf.shape[0], sigma), 0.0)
    bef = boundary_edges(nodesf, edgesf, L)
    bf = np.zeros(Nef, dtype=np.complex128)                # zero volume source
    free = np.setdiff1d(np.arange(Nef), bef)
    restrict = lambda M: M[free][:, free]

    Ef = inject_edge((nodes, tets, ee, es, E, N, L), nodesf, edgesf)

    with np.errstate(all="ignore"):
        Nrm_i = restrict(Nrm)
        dnorm = _make_dnorm(Nrm_i)                          # factor once, reuse
        rho_p = dnorm((bf - Af @ Ef)[free])

        # adjoint (homogeneous Dirichlet) on coarse interior
        Aii_H = (A[free_c][:, free_c]).conj().T
        zc = np.zeros(Ne, dtype=np.complex128)
        zc[free_c] = spla.spsolve(Aii_H.tocsc(), 0.5 * (Msig @ E)[free_c])
        zf = inject_edge((nodes, tets, ee, es, zc, N, L), nodesf, edgesf)
        rho_d = dnorm((0.5 * (Msigf @ Ef) - Af.conj().T @ zf)[free])

        # inf-sup on COARSE interior (conservative lower bound)
        beta = _infsup(A[free_c][:, free_c], Nrm_c[free_c][:, free_c])

        # quadratic remainder via one fine solve (same Dirichlet data)
        dvalsf = edge_circulation(nodesf, edgesf[bef], Efn)
        E_fine = solve_dirichlet(Af, Nef, bef, dvalsf, rhs=bf, method=method,
                                 geom=(nodesf, edgesf))
        ef = E_fine - Ef
        quad = 0.5 * np.real(np.vdot(ef, Msigf @ ef))

    # first-order DWR term: J(E)-J(E_h) = Re(E_h^H M e) + J(e) = 2 Re R_p(z - z_h) + J(e)
    # (revision R1: the factor 2 was missing in the submitted version)
    lin = 2.0 * rho_p * rho_d / beta
    # remainder 0 <= J(e) <= (sigma/2) ||e||^2 <= (sigma/2) (rho_p/beta)^2 / wmass
    # (the fine-solve value 'quad_est' is kept for information only)
    quad_bound = 0.5 * sigma * (rho_p / beta) ** 2 / wmass
    Pex = exact_Pabs(sigma, k, L)
    lo, hi = P_h - lin, P_h + lin + quad_bound
    return dict(P_h=P_h, P_exact=Pex, P_lo=lo, P_hi=hi,
                beta=beta, rho_p=rho_p, rho_d=rho_d, lin=lin, quad=quad_bound,
                quad_est=quad, contains=(lo <= Pex <= hi))


if __name__ == "__main__":
    freq, eps_r, sigma = 1e9, 40.0, 0.6
    lam = C0 / (freq * np.sqrt(eps_r))
    L = 1.0 * lam
    print("=== 3D certified SAR bracket (manufactured lossy plane wave) ===")
    print(f"  freq {freq/1e9:.1f} GHz  eps_r {eps_r}  sigma {sigma}  L {L*100:.2f} cm")
    print(f"{'N':>4} {'P_h':>12} {'P_exact':>12} {'contains':>9} "
          f"{'beta':>7} {'width%':>8}")
    prev = None
    for Nc in [4, 5, 6, 7]:
        r = certify_3d(freq, eps_r, sigma, L, Nc)
        wpct = (r["P_hi"] - r["P_lo"]) / r["P_exact"] * 100
        print(f"{Nc:>4} {r['P_h']:>12.4e} {r['P_exact']:>12.4e} "
              f"{str(r['contains']):>9} {r['beta']:>7.3f} {wpct:>8.1f}")
