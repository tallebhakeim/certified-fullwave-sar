"""
Certified SAR bracket in 2D on the curl-curl edge-element problem (marche 4, 2D).
================================================================================

Ports the goal-oriented (DWR) certified bracket to the 2D vector (TE_z) Whitney
edge-element solver of edge_fem2d. Output = absorbed power P_abs = (1/2) int
sigma |E|^2 (the SAR quantity).

Validation by a manufactured lossy plane wave  E = y_hat exp(-j k x)  in a
homogeneous lossy medium: curl curl E - k^2 E = 0 with zero volume source, so a
Dirichlet edge-circulation boundary condition drives it and the exact P_abs is
known in closed form. This isolates the DISCRETIZATION error (the machinery),
exactly as marche-1 validated the edge elements themselves.

Bracket (R1): P_abs in [P_h - 2 rho_p rho_d/beta, P_h + 2 rho_p rho_d/beta + (sigma/2)(rho_p/beta)^2/w_m],
with residuals measured in the natural norm |||v|||^2 = (1/mu)|curl v|^2 + w_m|v|^2, w_m = w^2|eps_c|,
beta the discrete inf-sup (= distance to resonance) on the interior space.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from edge_fem2d import (build_mesh, build_edges, assemble, edge_circulation,
                        boundary_edges, solve_dirichlet, field_at_points,
                        _tri_geom, LOCAL_EDGES, EPS0, MU0, C0)


# ---- real element matrices for the H(curl) norm and the sigma-mass ----------
def _iso_mass(area, b, c):
    g = np.outer(b, b) + np.outer(c, c)
    M = np.zeros((3, 3))
    for m, (p, q) in enumerate(LOCAL_EDGES):
        for n, (r, s) in enumerate(LOCAL_EDGES):
            dpr = p == r; dps = p == s; dqr = q == r; dqs = q == s
            M[m, n] = (area / 12.0) * ((1 + dpr) * g[q, s] - (1 + dps) * g[q, r]
                                       - (1 + dqr) * g[p, s] + (1 + dqs) * g[p, r])
    return M


def _curl(area, b, c):
    return np.array([2.0 * (b[p] * c[q] - c[p] * b[q]) for (p, q) in LOCAL_EDGES])


def assemble_real(nodes, tris, elem_edges, elem_sign, weight_mass, weight_curl):
    """Assemble sum_e ( wcurl * curl.curl + wmass_e * iso-mass ), real sparse."""
    Ne = int(elem_edges.max()) + 1
    rows, cols, vals = [], [], []
    for t, tri in enumerate(tris):
        area, b, c = _tri_geom(nodes[tri])
        cu = _curl(area, b, c)
        Miso = _iso_mass(area, b, c)
        wm = weight_mass[t]
        Ae = weight_curl * np.outer(cu, cu) * area + wm * Miso
        edg, sgn = elem_edges[t], elem_sign[t]
        for m in range(3):
            for n in range(3):
                rows.append(edg[m]); cols.append(edg[n])
                vals.append(sgn[m] * sgn[n] * Ae[m, n])
    return sp.coo_matrix((vals, (rows, cols)), shape=(Ne, Ne)).tocsr()


def inject_edge(coarse, xf_nodes, edges_f):
    """Fine-edge circulations of the coarse edge field (injection C -> F)."""
    cn, ct, cee, ces, cE, cN, cL = coarse
    fn = lambda p: field_at_points(cn, ct, cee, ces, cE, [p], cN, cL)[0]
    return edge_circulation(xf_nodes, edges_f, fn)


def _dual_norm(R, N):
    z = spla.spsolve(N.tocsc(), R)
    return np.sqrt(max(np.real(np.vdot(R, z)), 0.0))


def _infsup(A, N):
    """beta = smallest generalized singular value: beta^2 = min eig (A^H N^-1 A, N)."""
    Ad = A.toarray(); Nd = N.toarray()
    from scipy.linalg import eigh
    C = Ad.conj().T @ np.linalg.solve(Nd, Ad)
    C = 0.5 * (C + C.conj().T)
    lam = eigh(C, Nd, eigvals_only=True, subset_by_index=[0, 0])[0]
    return np.sqrt(max(lam, 0.0))


def exact_Pabs(sigma, k, L):
    """(1/2) sigma int_[0,L]^2 |exp(-j k x)|^2 dx dy, k complex (Im k < 0)."""
    a = 2.0 * np.imag(k)
    ix = (np.exp(a * L) - 1.0) / a if abs(a) > 1e-30 else L
    return 0.5 * sigma * L * ix


def certify_2d(freq, eps_r, sigma, L, N, rf=3):
    w = 2 * np.pi * freq
    eps_c = EPS0 * eps_r - 1j * sigma / w
    k = w * np.sqrt(MU0 * eps_c)
    Efn = lambda p: np.array([0.0, 1.0]) * np.exp(-1j * k * p[0])

    # coarse solve (manufactured Dirichlet)
    nodes, tris = build_mesh(L, N)
    edges, ee, es = build_edges(tris)
    epsc = np.full(tris.shape[0], eps_c)
    A, Ne = assemble(nodes, tris, ee, es, epsc, freq)
    bedges = boundary_edges(nodes, edges, L)
    dvals = edge_circulation(nodes, edges[bedges], Efn)
    E = solve_dirichlet(A, Ne, bedges, dvals)

    # natural (balanced) energy norm:  (1/mu)|curl|^2 + w^2|eps_c| |v|^2
    wcurl = 1.0 / MU0
    wmass = w * w * np.abs(eps_c)
    sig_elem = np.full(tris.shape[0], sigma)
    Msig = assemble_real(nodes, tris, ee, es, sig_elem, 0.0)
    Nrm_c = assemble_real(nodes, tris, ee, es,
                          np.full(tris.shape[0], wmass), wcurl)  # coarse energy norm
    P_h = 0.5 * np.real(np.vdot(E, Msig @ E))
    free_c = np.setdiff1d(np.arange(Ne), bedges)

    # ---- refined space ----
    Nf_ = (N - 1) * rf + 1
    nodesf, trisf = build_mesh(L, Nf_)
    edgesf, eef, esf = build_edges(trisf)
    epscf = np.full(trisf.shape[0], eps_c)
    Af, Nef = assemble(nodesf, trisf, eef, esf, epscf, freq)
    Nrm = assemble_real(nodesf, trisf, eef, esf,
                        np.full(trisf.shape[0], wmass), wcurl)  # energy norm
    Msigf = assemble_real(nodesf, trisf, eef, esf,
                          np.full(trisf.shape[0], sigma), 0.0)
    bef = boundary_edges(nodesf, edgesf, L)
    dvalsf = edge_circulation(nodesf, edgesf[bef], Efn)
    bf = np.zeros(Nef, dtype=np.complex128)                    # zero volume source
    free = np.setdiff1d(np.arange(Nef), bef)
    restrict = lambda M: M[free][:, free]

    Ef = inject_edge((nodes, tris, ee, es, E, N, L), nodesf, edgesf)

    with np.errstate(all="ignore"):
        # primal residual on interior fine space
        Nrm_i = restrict(Nrm)
        rho_p = _dual_norm((bf - Af @ Ef)[free], Nrm_i)

        # adjoint (homogeneous Dirichlet) on coarse interior: A_ii^H z = (1/2)Msig E
        Aii_H = (A[free_c][:, free_c]).conj().T
        zc = np.zeros(Ne, dtype=np.complex128)
        zc[free_c] = spla.spsolve(Aii_H.tocsc(), 0.5 * (Msig @ E)[free_c])
        zf = inject_edge((nodes, tris, ee, es, zc, N, L), nodesf, edgesf)
        rho_d = _dual_norm((0.5 * (Msigf @ Ef) - Af.conj().T @ zf)[free], Nrm_i)

        # inf-sup on COARSE interior (conservative lower bound; dense, cheap)
        beta = _infsup(A[free_c][:, free_c], Nrm_c[free_c][:, free_c])

        # quadratic remainder via one fine solve (same Dirichlet data)
        E_fine = solve_dirichlet(Af, Nef, bef, dvalsf, rhs=bf)
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
    L = 1.5 * lam
    print("=== 2D certified SAR bracket (manufactured lossy plane wave) ===")
    print(f"  freq {freq/1e9:.1f} GHz  eps_r {eps_r}  sigma {sigma}  L {L*100:.2f} cm")
    print(f"{'N':>4} {'P_h':>12} {'P_exact':>12} {'contains':>9} {'beta':>8} {'width%':>8}")
    for Nc in [9, 13, 19, 27]:
        r = certify_2d(freq, eps_r, sigma, L, Nc)
        w = (r["P_hi"] - r["P_lo"]) / r["P_exact"] * 100
        print(f"{Nc:>4} {r['P_h']:>12.4e} {r['P_exact']:>12.4e} "
              f"{str(r['contains']):>9} {r['beta']:>8.3f} {w:>8.1f}")
