"""
Full-wave 2D Helmholtz forward solver (TM_z), FEM T3.
=====================================================

Regime  : time-harmonic Maxwell, convention e^{+j w t}.
Unknown : E = E_z(x,y) z_hat  (out-of-plane electric field).
Equation: lap(E_z) + k^2 E_z = j w mu0 J_z,
          k^2 = w^2 mu0 eps_c,   eps_c = eps0*eps_r - j*sigma/w  (lossy).

This is the "stage above" our quasi-static ME / DGM / eddy-current codes:
the displacement-current term (w^2 eps) and propagation/retardation are kept,
so it is valid in the RF / microwave band where dosimetry is regulated by SAR.

Weak form (Galerkin), test function w:
    int grad(w).grad(E) - int k^2 w E  -  j*kb int_bnd w E  =  RHS_source
                     |__ stiffness K __| |__ mass k2 M __|   |__ 1st-order ABC __|

Validation: point (line) source in a homogeneous medium -> discrete Green's
function must converge to the analytic 2D free-space Green's function
    G(r) = (-j/4) H0^{(2)}(k r)         (outgoing, e^{+j w t}).

Author: Hakeim Talleb PoC scaffold, 2026-07-13.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import hankel2

EPS0 = 8.8541878128e-12
MU0 = 4.0e-7 * np.pi
C0 = 1.0 / np.sqrt(EPS0 * MU0)


# ----------------------------------------------------------------------------
# Mesh: structured grid of N x N nodes on [0,L]^2, each cell split into 2 T3.
# ----------------------------------------------------------------------------
def build_mesh(L, N):
    xs = np.linspace(0.0, L, N)
    ys = np.linspace(0.0, L, N)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    nodes = np.column_stack([X.ravel(), Y.ravel()])

    def nid(i, j):  # i column (x), j row (y)
        return j * N + i

    tris = []
    for j in range(N - 1):
        for i in range(N - 1):
            n00, n10 = nid(i, j), nid(i + 1, j)
            n01, n11 = nid(i, j + 1), nid(i + 1, j + 1)
            tris.append((n00, n10, n11))
            tris.append((n00, n11, n01))
    tris = np.asarray(tris, dtype=np.int64)

    # boundary edges (outer ring) for the absorbing condition
    bnd = []
    for i in range(N - 1):
        bnd.append((nid(i, 0), nid(i + 1, 0)))            # bottom
        bnd.append((nid(i, N - 1), nid(i + 1, N - 1)))    # top
    for j in range(N - 1):
        bnd.append((nid(0, j), nid(0, j + 1)))            # left
        bnd.append((nid(N - 1, j), nid(N - 1, j + 1)))    # right
    bnd = np.asarray(bnd, dtype=np.int64)
    return nodes, tris, bnd


# ----------------------------------------------------------------------------
# Assembly of  A = K - M_{k2} + j*kb*C_bnd   (complex sparse).
#   k2_elem : (Nt,) complex per-element k^2
#   kb      : scalar complex boundary-medium wavenumber for the ABC
# ----------------------------------------------------------------------------
def assemble(nodes, tris, bnd, k2_elem, kb):
    Nn = nodes.shape[0]
    rows, cols, vals = [], [], []

    for t, (a, b, c) in enumerate(tris):
        p = nodes[[a, b, c]]
        x1, y1 = p[0]
        x2, y2 = p[1]
        x3, y3 = p[2]
        area = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        # gradients of linear shape functions (constants)
        bco = np.array([y2 - y3, y3 - y1, y1 - y2]) / (2.0 * area)
        cco = np.array([x3 - x2, x1 - x3, x2 - x1]) / (2.0 * area)
        Ke = area * (np.outer(bco, bco) + np.outer(cco, cco))
        Me = (area / 12.0) * np.array([[2.0, 1.0, 1.0],
                                       [1.0, 2.0, 1.0],
                                       [1.0, 1.0, 2.0]])
        Ae = Ke - k2_elem[t] * Me
        idx = (a, b, c)
        for li in range(3):
            for lj in range(3):
                rows.append(idx[li]); cols.append(idx[lj]); vals.append(Ae[li, lj])

    # first-order absorbing boundary condition:  +j*kb * consistent edge mass
    for (n1, n2) in bnd:
        Le = np.linalg.norm(nodes[n2] - nodes[n1])
        Ce = (Le / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])
        ii = (n1, n2)
        for li in range(2):
            for lj in range(2):
                rows.append(ii[li]); cols.append(ii[lj])
                vals.append(1j * kb * Ce[li, lj])

    A = sp.coo_matrix((vals, (rows, cols)), shape=(Nn, Nn), dtype=np.complex128)
    return A.tocsr()


def nearest_node(nodes, xy):
    return int(np.argmin(np.sum((nodes - np.asarray(xy)) ** 2, axis=1)))


def solve_point_source(nodes, tris, bnd, freq, eps_r_elem, sigma_elem,
                       src_xy, src_strength=1.0):
    """Solve with a point source of the PDE  lap(E)+k^2 E = -S*delta(r-r0)."""
    w = 2.0 * np.pi * freq
    eps_c = EPS0 * eps_r_elem - 1j * sigma_elem / w      # (Nt,) complex
    k2_elem = w * w * MU0 * eps_c
    # ABC wavenumber from the boundary medium (assume homogeneous outer ring)
    kb = w * np.sqrt(MU0 * (EPS0 * eps_r_elem[0] - 1j * sigma_elem[0] / w))
    A = assemble(nodes, tris, bnd, k2_elem, kb)

    rhs = np.zeros(nodes.shape[0], dtype=np.complex128)
    isrc = nearest_node(nodes, src_xy)
    rhs[isrc] = src_strength     # -F with f=-S*delta  ->  +S at the source node

    E = spla.spsolve(A, rhs)
    return E, isrc, kb


# ----------------------------------------------------------------------------
# Validation against the analytic 2D free-space Green's function.
# ----------------------------------------------------------------------------
def analytic_green(r, k, strength=1.0):
    return strength * (-1j / 4.0) * hankel2(0, k * r)
