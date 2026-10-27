"""
Full-wave 2D vector solver (TE_z) with Whitney edge elements.
=============================================================

Convention e^{+j w t}. Unknown E = (Ex, Ey) in-plane (H = Hz z_hat).
Curl-curl equation:
    curl( (1/mu) curl E ) - w^2 eps_c E = -j w J,
    eps_c = eps0 eps_r - j sigma / w.

In 2D-TE the curl of an in-plane field is a scalar (z-component):
    curl E = dEy/dx - dEx/dy.

Weak form (Galerkin), vector test W:
    int (1/mu) (curl W)(curl E) dO  -  int w^2 eps_c W . E dO  =  -int j w W . J dO
                |__ S : curl-curl __|      |__ T : vector mass __|

Lowest-order Whitney (Nedelec) edge basis on a triangle, for local edge (p,q):
    W_pq = lam_p grad(lam_q) - lam_q grad(lam_p),   curl W_pq = 2 (grad lam_p x grad lam_q)_z
DOF = tangential circulation  int_edge E.t dl  (Whitney reproduces constant fields).

Edge elements are the correct discretization here: they enforce tangential
continuity (physical for E) while allowing the normal jump at material
interfaces, and they are curl-conforming, killing the spurious modes that
nodal elements produce in curl-curl problems.

Author: Hakeim Talleb PoC scaffold, 2026-07-13.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

EPS0 = 8.8541878128e-12
MU0 = 4.0e-7 * np.pi
C0 = 1.0 / np.sqrt(EPS0 * MU0)

LOCAL_EDGES = [(0, 1), (1, 2), (2, 0)]   # local node pairs per triangle


# ----------------------------------------------------------------------------
# Structured T3 mesh (same as the scalar solver) + edge connectivity.
# ----------------------------------------------------------------------------
def build_mesh(L, N):
    xs = np.linspace(0.0, L, N)
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    nodes = np.column_stack([X.ravel(), Y.ravel()])

    def nid(i, j):
        return j * N + i

    tris = []
    for j in range(N - 1):
        for i in range(N - 1):
            n00, n10 = nid(i, j), nid(i + 1, j)
            n01, n11 = nid(i, j + 1), nid(i + 1, j + 1)
            tris.append((n00, n10, n11))
            tris.append((n00, n11, n01))
    return nodes, np.asarray(tris, dtype=np.int64)


def build_edges(tris):
    """Global edge list (low<high), elem->edge map, and orientation signs."""
    edge_map = {}
    elem_edges = np.zeros((tris.shape[0], 3), dtype=np.int64)
    elem_sign = np.zeros((tris.shape[0], 3), dtype=np.float64)
    edges = []
    for t, tri in enumerate(tris):
        for le, (p, q) in enumerate(LOCAL_EDGES):
            gp, gq = tri[p], tri[q]
            key = (min(gp, gq), max(gp, gq))
            if key not in edge_map:
                edge_map[key] = len(edges)
                edges.append(key)
            elem_edges[t, le] = edge_map[key]
            elem_sign[t, le] = 1.0 if gp < gq else -1.0   # align to low->high
    return np.asarray(edges, dtype=np.int64), elem_edges, elem_sign


def _tri_geom(p):
    x1, y1 = p[0]; x2, y2 = p[1]; x3, y3 = p[2]
    area = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    b = np.array([y2 - y3, y3 - y1, y1 - y2]) / (2.0 * area)   # d lam/dx
    c = np.array([x3 - x2, x1 - x3, x2 - x1]) / (2.0 * area)   # d lam/dy
    return area, b, c


# ----------------------------------------------------------------------------
# Assembly of A = S - w^2 T  (Ne x Ne complex).
# ----------------------------------------------------------------------------
def assemble(nodes, tris, elem_edges, elem_sign, eps_c_elem, freq,
             mu_elem=None, pml_fn=None):
    """Assemble A = S - w^2 T.

    pml_fn(centroid) -> (s_x, s_y) complex stretching factors (UPML). Default
    (1,1) everywhere reduces to the plain vacuum/tissue curl-curl. In the UPML
    the coefficients become, on the REAL mesh (Jin, uniaxial PML, 2D-TE):
        curl-curl : 1 / (mu s_x s_y)
        mass      : eps_c * diag(s_y/s_x, s_x/s_y)   (anisotropic)
    """
    w = 2.0 * np.pi * freq
    Ne = int(elem_edges.max()) + 1
    if mu_elem is None:
        mu_elem = np.full(tris.shape[0], MU0)
    rows, cols, vals = [], [], []

    for t, tri in enumerate(tris):
        area, b, c = _tri_geom(nodes[tri])
        if pml_fn is None:
            sx = sy = 1.0 + 0.0j
        else:
            sx, sy = pml_fn(nodes[tri].mean(axis=0))
        mu_c = mu_elem[t] * sx * sy                   # curl-curl scaling
        Lxx = eps_c_elem[t] * sy / sx                 # anisotropic mass tensor
        Lyy = eps_c_elem[t] * sx / sy

        curl = np.empty(3)                            # curl of each local Whitney fn
        for m, (p, q) in enumerate(LOCAL_EDGES):
            curl[m] = 2.0 * (b[p] * c[q] - c[p] * b[q])

        S = np.zeros((3, 3), dtype=np.complex128)
        T = np.zeros((3, 3), dtype=np.complex128)
        for m, (p, q) in enumerate(LOCAL_EDGES):
            for n, (r, s) in enumerate(LOCAL_EDGES):
                S[m, n] = (1.0 / mu_c) * curl[m] * curl[n] * area
                dpr = 1.0 if p == r else 0.0; dps = 1.0 if p == s else 0.0
                dqr = 1.0 if q == r else 0.0; dqs = 1.0 if q == s else 0.0
                # int Wmx Wnx (b-based) and int Wmy Wny (c-based)
                Ixx = (area / 12.0) * ((1 + dpr) * b[q] * b[s] - (1 + dps) * b[q] * b[r]
                                       - (1 + dqr) * b[p] * b[s] + (1 + dqs) * b[p] * b[r])
                Iyy = (area / 12.0) * ((1 + dpr) * c[q] * c[s] - (1 + dps) * c[q] * c[r]
                                       - (1 + dqr) * c[p] * c[s] + (1 + dqs) * c[p] * c[r])
                T[m, n] = Lxx * Ixx + Lyy * Iyy

        Ae = S - w * w * T
        edg = elem_edges[t]; sgn = elem_sign[t]
        for m in range(3):
            for n in range(3):
                rows.append(edg[m]); cols.append(edg[n])
                vals.append(sgn[m] * sgn[n] * Ae[m, n])

    A = sp.coo_matrix((vals, (rows, cols)), shape=(Ne, Ne), dtype=np.complex128)
    return A.tocsr(), Ne


# ----------------------------------------------------------------------------
# Boundary edge DOFs = tangential circulation of a prescribed field (Dirichlet).
# ----------------------------------------------------------------------------
def edge_circulation(nodes, edges, field_fn):
    """int_edge field . t dl along low->high, 3-point Gauss."""
    gp = np.array([0.5 - np.sqrt(15) / 10, 0.5, 0.5 + np.sqrt(15) / 10])
    gw = np.array([5 / 18, 8 / 18, 5 / 18])
    vals = np.zeros(edges.shape[0], dtype=np.complex128)
    for e, (na, nb) in enumerate(edges):
        A = nodes[na]; B = nodes[nb]; d = B - A
        acc = 0.0j
        for xi, wq in zip(gp, gw):
            pt = A + xi * d
            acc += wq * np.dot(field_fn(pt), d)      # dl = d * dxi
        vals[e] = acc
    return vals


def boundary_edges(nodes, edges, L, tol=1e-9):
    a, b = nodes[edges[:, 0]], nodes[edges[:, 1]]
    on = lambda v: (np.abs(v) < tol) | (np.abs(v - L) < tol)
    mask = ((on(a[:, 0]) & on(b[:, 0])) | (on(a[:, 1]) & on(b[:, 1])))
    return np.where(mask)[0]


def solve_dirichlet(A, Ne, dir_edges, dir_vals, rhs=None):
    """Solve A E = rhs with E[dir_edges] = dir_vals imposed."""
    if rhs is None:
        rhs = np.zeros(Ne, dtype=np.complex128)
    all_e = np.arange(Ne)
    free = np.setdiff1d(all_e, dir_edges)
    Eb = np.zeros(Ne, dtype=np.complex128)
    Eb[dir_edges] = dir_vals
    b = rhs[free] - A[free][:, dir_edges] @ dir_vals
    Ef = spla.spsolve(A[free][:, free].tocsc(), b)
    E = Eb.copy(); E[free] = Ef
    return E


# ----------------------------------------------------------------------------
# Reconstruct the vector field at triangle centroids from edge DOFs.
# ----------------------------------------------------------------------------
def scattered_rhs(nodes, tris, elem_edges, elem_sign, Ne, freq,
                  eps_c_elem, eps_bg, Einc_fn):
    """RHS of the scattered-field formulation.

    Total field E = E_inc + E_s, with E_inc a known plane wave satisfying the
    background problem. E_s then obeys  curl(1/mu curl E_s) - w^2 eps_c E_s =
    w^2 (eps_c - eps_bg) E_inc, nonzero only inside the scatterer (phantom).
    The PML absorbs the outgoing scattered field. RHS_i = int w^2 (eps_c-eps_bg)
    W_i . E_inc dO, evaluated with E_inc at the centroid (low-order consistent).
    """
    w = 2.0 * np.pi * freq
    rhs = np.zeros(Ne, dtype=np.complex128)
    for t, tri in enumerate(tris):
        dcontrast = eps_c_elem[t] - eps_bg
        if dcontrast == 0:
            continue
        area, b, c = _tri_geom(nodes[tri])
        grad = np.column_stack([b, c])
        cen = nodes[tri].mean(axis=0)
        Einc = Einc_fn(cen)
        for m, (p, q) in enumerate(LOCAL_EDGES):
            intW = (area / 3.0) * (grad[q] - grad[p])    # int_T W_m dA
            rhs[elem_edges[t, m]] += (w * w * dcontrast
                                      * np.dot(Einc, intW) * elem_sign[t, m])
    return rhs


def make_pml(L, d, strength=12.0, order=2):
    """UPML stretching s(x)=1 - j*strength*(depth/d)^order, quadratic grading."""
    def prof(v):
        if v < d:
            return ((d - v) / d) ** order
        if v > L - d:
            return ((v - (L - d)) / d) ** order
        return 0.0

    def pml_fn(xy):
        sx = 1.0 - 1j * strength * prof(xy[0])
        sy = 1.0 - 1j * strength * prof(xy[1])
        return sx, sy
    return pml_fn


def _bary_grads(p):
    x1, y1 = p[0]; x2, y2 = p[1]; x3, y3 = p[2]
    area = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    a = np.array([x2 * y3 - x3 * y2, x3 * y1 - x1 * y3, x1 * y2 - x2 * y1]) / (2 * area)
    b = np.array([y2 - y3, y3 - y1, y1 - y2]) / (2 * area)
    c = np.array([x3 - x2, x1 - x3, x2 - x1]) / (2 * area)
    return a, b, c


def field_at_points(nodes, tris, elem_edges, elem_sign, E, pts, N, L):
    """Evaluate the Whitney vector field at arbitrary points on the structured mesh."""
    h = L / (N - 1)
    out = np.zeros((len(pts), 2), dtype=np.complex128)
    for idx, (x, y) in enumerate(pts):
        i = min(max(int(x / h), 0), N - 2)
        j = min(max(int(y / h), 0), N - 2)
        u = (x - i * h) / h; v = (y - j * h) / h
        local = 0 if v <= u else 1               # lower / upper triangle
        t = 2 * (j * (N - 1) + i) + local
        tri = tris[t]
        a, b, c = _bary_grads(nodes[tri])
        lam = a + b * x + c * y
        vec = np.zeros(2, dtype=np.complex128)
        for m, (pp, qq) in enumerate(LOCAL_EDGES):
            W = lam[pp] * np.array([b[qq], c[qq]]) - lam[qq] * np.array([b[pp], c[pp]])
            vec += E[elem_edges[t, m]] * elem_sign[t, m] * W
        out[idx] = vec
    return out


def field_at_centroids(nodes, tris, elem_edges, elem_sign, E):
    out = np.zeros((tris.shape[0], 2), dtype=np.complex128)
    cen = np.zeros((tris.shape[0], 2))
    for t, tri in enumerate(tris):
        area, b, c = _tri_geom(nodes[tri])
        grad = np.column_stack([b, c])               # (3,2): grad lam_i
        vec = np.zeros(2, dtype=np.complex128)
        for m, (p, q) in enumerate(LOCAL_EDGES):
            # W at centroid (lam=1/3): (1/3)(grad lam_q - grad lam_p)
            Wc = (grad[q] - grad[p]) / 3.0
            vec += E[elem_edges[t, m]] * elem_sign[t, m] * Wc
        out[t] = vec
        cen[t] = nodes[tri].mean(axis=0)
    return cen, out
