"""
Two-sided SAR bracket in 3D on a HETEROGENEOUS lossy tissue stack (revision R1).
================================================================================

Exact 3D reference with tissue interfaces: in a skin / fat / muscle stack with
interfaces at x = const, the field E = y_hat f(x), where f is the exact 1D
layered solution (transfer matrix, helmholtz1d.Phantom) of -f'' - k(x)^2 f = 0,
solves the 3D curl-curl problem exactly:
    curl curl E - k^2 E = y_hat (-f'' - k^2 f) = 0,
with continuous tangential E (y_hat f) and tangential H (z_hat f'/(j w mu0)) at
every interface and div(eps E) = 0. Prescribing its tangential trace on the
boundary of the cube [0, L]^3 (Dirichlet edge circulations) gives a 3D,
interface-crossing, lossy problem whose absorbed power is known exactly:
    P_abs = L^2 (1/2) int_0^L sigma(x) |f(x)|^2 dx.

Bracket (same structure as the 1D and 2D brackets, factor 2 of the first-order
term included):
    P_abs in [ P_h - 2 rho_p rho_d / beta ,  P_h + 2 rho_p rho_d / beta + Q ],
    Q = (sigma_max / 2) (rho_p / beta)^2 / w_m      (since w_m ||e||^2 <= |||e|||^2)
in the natural norm |||v|||^2 = (1/mu0) ||curl v||^2 + w_m ||v||^2, w_m =
w^2 max|eps_c| (the weight used for the 2D bracket). rho_p, rho_d are dual
norms of the residuals evaluated on the rf-times refined space (hierarchical
estimate; assumption H2, saturation) and beta is the discrete inf-sup constant
on the coarse interior space (assumption H1, beta_h <= gamma).

Run:  python3 certify3d_layered.py
"""

import os
import sys
import time
import numpy as np
import scipy.sparse.linalg as spla

from edge_fem3d import (build_mesh, build_edges, assemble, boundary_edges,
                        solve_dirichlet, EPS0, MU0, C0)
from certify3d import assemble_real, inject_edge

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sar_certified_poc"))
from helmholtz1d import Phantom, TISSUE, k2_of           # noqa: E402
import certify_rigorous as R1D                           # noqa: E402

LAYERS = [("skin", 0.005), ("fat", 0.005), ("muscle", 0.010)]   # L = 2 cm
FREQ = 2e9


def stack():
    ph = Phantom([("air", 0.01)] + LAYERS + [("air", 0.01)], FREQ)
    x0 = ph.edges[1]                                      # skin surface
    L = sum(t for _, t in LAYERS)
    return ph, x0, L


def exact_field_fn(ph, x0):
    def fn(p):
        return np.array([0.0, ph.exact_field(np.array([x0 + p[0]]))[0], 0.0])
    return fn


def exact_Pabs(ph, x0, L):
    return L * L * R1D.exact_window_output(ph, x0, x0 + L)


def materials(nodes, tets, ph, x0):
    cx = nodes[tets][:, :, 0].mean(axis=1)
    names = [ph.tissue_at(x0 + c) for c in cx]
    eps_c = np.array([EPS0 * TISSUE[n][0] - 1j * TISSUE[n][1] / ph.w for n in names])
    sig = np.array([TISSUE[n][1] for n in names])
    return eps_c, sig


def materials_eps_max(ph):
    return max(abs(EPS0 * TISSUE[n][0] - 1j * TISSUE[n][1] / ph.w) for n, _ in LAYERS)


def infsup_sparse(A, Nrm):
    """beta^2 = min eig(A^H N^-1 A, N), via the largest eigenvalue of the
    inverse pencil (N A^-H N A^-1 N, N); sparse LU of A and N."""
    luA = spla.splu(A.tocsc())
    luN = spla.splu(Nrm.tocsc().astype(complex))
    n = A.shape[0]

    def op(w):
        y = luA.solve(Nrm @ w)
        y = luA.solve(Nrm @ y, trans="H")
        return Nrm @ y
    Op = spla.LinearOperator((n, n), matvec=op, dtype=complex)
    Minv = spla.LinearOperator((n, n), matvec=luN.solve, dtype=complex)
    Nop = spla.LinearOperator((n, n), matvec=lambda v: Nrm @ v, dtype=complex)
    lam = spla.eigsh(Op, k=1, M=Nop, Minv=Minv, which="LA",
                     return_eigenvectors=False, tol=1e-8, maxiter=3000)[0]
    return 1.0 / np.sqrt(lam)


def certify_layered(N, rf=2, method="direct", wm_factor=None):
    ph, x0, L = stack()
    w = ph.w
    Efn = exact_field_fn(ph, x0)
    # natural energy weight w^2 max|eps_c| (as in the 2D bracket)
    wcurl, wm = 1.0 / MU0, w * w * np.abs(materials_eps_max(ph))
    if wm_factor is not None:
        wm = wm * wm_factor

    nodes, tets = build_mesh(L, N)
    edges, ee, es = build_edges(tets)
    eps_c, sig = materials(nodes, tets, ph, x0)
    A, Ne = assemble(nodes, tets, ee, es, eps_c, FREQ)
    bed = boundary_edges(nodes, edges, L)
    from edge_fem3d import edge_circulation
    E = solve_dirichlet(A, Ne, bed, edge_circulation(nodes, edges[bed], Efn))
    Msig = assemble_real(nodes, tets, ee, es, sig, 0.0)
    Nc = assemble_real(nodes, tets, ee, es, np.full(len(tets), wm), wcurl)
    P_h = 0.5 * np.real(np.vdot(E, Msig @ E))
    free_c = np.setdiff1d(np.arange(Ne), bed)

    Nf = (N - 1) * rf + 1
    nodesf, tetsf = build_mesh(L, Nf)
    edgesf, eef, esf = build_edges(tetsf)
    eps_cf, sigf = materials(nodesf, tetsf, ph, x0)
    Af, Nef = assemble(nodesf, tetsf, eef, esf, eps_cf, FREQ)
    Nrm = assemble_real(nodesf, tetsf, eef, esf, np.full(len(tetsf), wm), wcurl)
    Msigf = assemble_real(nodesf, tetsf, eef, esf, sigf, 0.0)
    bef = boundary_edges(nodesf, edgesf, L)
    free = np.setdiff1d(np.arange(Nef), bef)
    Ef = inject_edge((nodes, tets, ee, es, E, N, L), nodesf, edgesf)

    with np.errstate(all="ignore"):
        luN = spla.splu(Nrm[free][:, free].tocsc())

        def dnorm(Rv):
            z = luN.solve(Rv.real) + 1j * luN.solve(Rv.imag)
            return np.sqrt(max(np.real(np.vdot(Rv, z)), 0.0))
        rho_p = dnorm((-(Af @ Ef))[free])
        AiiH = A[free_c][:, free_c].conj().T
        zc = np.zeros(Ne, dtype=complex)
        zc[free_c] = spla.spsolve(AiiH.tocsc(), 0.5 * (Msig @ E)[free_c])
        zf = inject_edge((nodes, tets, ee, es, zc, N, L), nodesf, edgesf)
        rho_d = dnorm((0.5 * (Msigf @ Ef) - Af.conj().T @ zf)[free])
        beta = infsup_sparse(A[free_c][:, free_c], Nc[free_c][:, free_c])

    lin = 2.0 * rho_p * rho_d / beta
    quad = 0.5 * sig.max() * (rho_p / beta) ** 2 / wm
    Pex = exact_Pabs(ph, x0, L)
    return dict(N=N, h=L / (N - 1), n_edges=Ne, n_edges_fine=Nef, P_h=P_h,
                P_exact=Pex, lo=P_h - lin, hi=P_h + lin + quad, lin=lin,
                quad=quad, beta=beta, rho_p=rho_p, rho_d=rho_d,
                contains=bool(P_h - lin <= Pex <= P_h + lin + quad))


if __name__ == "__main__":
    np.seterr(all="ignore")
    ph, x0, L = stack()
    print("=== 3D bracket, skin/fat/muscle stack, 2 GHz, exact layered reference ===")
    print(f"exact P_abs = {exact_Pabs(ph, x0, L):.6e} W (per (V/m)^2 incident)")
    results = []
    for N in [5, 9, 13, 17]:
        t = time.time()
        r = certify_layered(N)
        results.append(r)
        wd = (r["hi"] - r["lo"]) / r["P_exact"] * 100
        print(f"N={N:3d} h={r['h']*1e3:5.2f}mm edges={r['n_edges']:6d}/{r['n_edges_fine']:7d} "
              f"P_h={r['P_h']:.4e} [{r['lo']:.4e},{r['hi']:.4e}] in={r['contains']} "
              f"beta={r['beta']:.4f} width={wd:8.2f}% lin={r['lin']:.2e} quad={r['quad']:.2e} "
              f"({time.time()-t:.0f}s)", flush=True)
    import json
    with open("certify3d_layered_results.json", "w") as fh:
        json.dump(results, fh, indent=1)
