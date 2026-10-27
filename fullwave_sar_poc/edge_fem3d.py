"""
Full-wave 3D vector solver with Whitney (lowest-order Nedelec) edge elements.
=============================================================================

Direct 3D extension of edge_fem2d.py. Convention e^{+j w t}. Unknown E in R^3.
Curl-curl equation:
    curl( (1/mu) curl E ) - w^2 eps_c E = -j w J,
    eps_c = eps0 eps_r - j sigma / w.

Weak form (Galerkin), vector test W:
    int (1/mu) (curl W).(curl E) dV  -  int w^2 eps_c W.E dV  =  -int j w W.J dV
             |__ S : curl-curl __|         |__ T : vector mass __|

Lowest-order Whitney edge basis on a tetrahedron, local edge (p,q):
    W_pq = lam_p grad(lam_q) - lam_q grad(lam_p),   curl W_pq = 2 grad(lam_p) x grad(lam_q)
DOF = tangential circulation  int_edge E.t dl  (Whitney reproduces constant fields).

Mesh: structured cube of N^3 nodes, each voxel split into 6 tetrahedra by the
Freudenthal-Kuhn decomposition (all tets share the main diagonal 000->111). This
is a CONFORMING triangulation: edges match across voxel faces, so edge DOFs are
shared consistently and no hanging edges appear. It also maps one-to-one onto a
voxel phantom, the target geometry for XCAT dosimetry.

UPML (uniaxial, Gedney/Jin) with stretch (s_x,s_y,s_z):
    Lam = diag( s_y s_z / s_x,  s_z s_x / s_y,  s_x s_y / s_z )
    curl-curl coefficient : (1/mu) Lam^{-1}   (anisotropic)
    mass coefficient       : eps_c Lam         (anisotropic)
Reduces to the validated 2D-TE case for s_z=1 (Lam_z = s_x s_y, Lam_x = s_y/s_x,
Lam_y = s_x/s_y), i.e. curl 1/(mu s_x s_y), mass diag(s_y/s_x, s_x/s_y).

Author: Hakeim Talleb PoC scaffold, 2026-07-18.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

EPS0 = 8.8541878128e-12
MU0 = 4.0e-7 * np.pi
C0 = 1.0 / np.sqrt(EPS0 * MU0)

# 6 local edges of a tetrahedron (node index pairs), ordered p<q.
LOCAL_EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

# 6 tetrahedra tiling a unit cube (Freudenthal-Kuhn). Cube corner c has local
# coords (c&1, (c>>1)&1, (c>>2)&1); every tet is the path 0 -> +ei -> +ei+ej ->
# 7 for a permutation (i,j,k) of the axes, so all share the diagonal 0-7.
_PERMS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]


def _cube_tets():
    """Return the 6 tets as tuples of cube-corner indices (0..7)."""
    tets = []
    for perm in _PERMS:
        path = [0]
        cur = 0
        for axis in perm:
            cur = cur | (1 << axis)
            path.append(cur)
        tets.append(tuple(path))          # (0, ., ., 7)
    return tets


# ----------------------------------------------------------------------------
# Structured tetrahedral mesh.
# ----------------------------------------------------------------------------
def build_mesh(L, N):
    xs = np.linspace(0.0, L, N)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    def nid(i, j, k):
        return (i * N + j) * N + k

    # cube corner offsets in (i,j,k) for corner index c
    corner = [(c & 1, (c >> 1) & 1, (c >> 2) & 1) for c in range(8)]
    cube_tets = _cube_tets()
    tets = []
    for i in range(N - 1):
        for j in range(N - 1):
            for k in range(N - 1):
                gcorner = [nid(i + dx, j + dy, k + dz) for (dx, dy, dz) in corner]
                for tt in cube_tets:
                    tets.append(tuple(gcorner[c] for c in tt))
    return nodes, np.asarray(tets, dtype=np.int64)


def build_edges(tets):
    """Global edge list (low<high), elem->edge map, and orientation signs."""
    edge_map = {}
    ne_loc = len(LOCAL_EDGES)
    elem_edges = np.zeros((tets.shape[0], ne_loc), dtype=np.int64)
    elem_sign = np.zeros((tets.shape[0], ne_loc), dtype=np.float64)
    edges = []
    for t, tet in enumerate(tets):
        for le, (p, q) in enumerate(LOCAL_EDGES):
            gp, gq = tet[p], tet[q]
            key = (min(gp, gq), max(gp, gq))
            idx = edge_map.get(key)
            if idx is None:
                idx = len(edges)
                edge_map[key] = idx
                edges.append(key)
            elem_edges[t, le] = idx
            elem_sign[t, le] = 1.0 if gp < gq else -1.0    # align to low->high
    return np.asarray(edges, dtype=np.int64), elem_edges, elem_sign


def _tet_geom(P):
    """Volume and barycentric gradients grads[i] = grad(lam_i), shape (4,3)."""
    M = np.hstack([np.ones((4, 1)), P])                    # 4x4 [1 x y z]
    detM = np.linalg.det(M)
    vol = abs(detM) / 6.0
    Cinv = np.linalg.inv(M)
    grads = Cinv[1:4, :].T                                 # (4,3)
    return vol, grads


# ---- batched geometry / element matrices (vectorised over all tets) ---------
# 6x6 index grids of the local edge endpoints (p,q for row m; r,s for col n).
_PM = np.array([e[0] for e in LOCAL_EDGES])                # (6,)
_QM = np.array([e[1] for e in LOCAL_EDGES])
_PG = _PM[:, None] * np.ones((6, 6), int)                  # p over (m,n)
_QG = _QM[:, None] * np.ones((6, 6), int)                  # q
_RG = _PM[None, :] * np.ones((6, 6), int)                  # r
_SG = _QM[None, :] * np.ones((6, 6), int)                  # s


def batch_geom(nodes, tets):
    """Volumes (Nt,) and gradients (Nt,4,3) for every tet at once."""
    P = nodes[tets]                                        # (Nt,4,3)
    Nt = P.shape[0]
    M = np.concatenate([np.ones((Nt, 4, 1)), P], axis=2)   # (Nt,4,4)
    detM = np.linalg.det(M)
    vol = np.abs(detM) / 6.0
    Cinv = np.linalg.inv(M)
    grads = np.transpose(Cinv[:, 1:4, :], (0, 2, 1))       # (Nt,4,3)
    return vol, grads


def batch_curl(grads):
    """curl of the 6 local Whitney fns per tet: 2 grad_p x grad_q -> (Nt,6,3)."""
    gp = grads[:, _PM, :]; gq = grads[:, _QM, :]
    return 2.0 * np.cross(gp, gq, axis=2)


def _quad_from_metric(metric, vol):
    """Whitney mass-type 6x6 from a 4x4 (Nt,4,4) metric m[a,b] and volumes.
    M[m,n] = m[q,s]I(p,r) - m[q,r]I(p,s) - m[p,s]I(q,r) + m[p,r]I(q,s),
    I(a,b) = vol (1+delta_ab)/20."""
    def I(a, b):
        return vol[:, None, None] * (1.0 + (a == b)) / 20.0
    return (metric[:, _QG, _SG] * I(_PG, _RG)
            - metric[:, _QG, _RG] * I(_PG, _SG)
            - metric[:, _PG, _SG] * I(_QG, _RG)
            + metric[:, _PG, _RG] * I(_QG, _SG))


def _scatter(elem_edges, elem_sign, Ae, Ne):
    """Assemble per-element (Nt,6,6) blocks into a global CSR matrix."""
    Nt = elem_edges.shape[0]
    sgn = elem_sign
    Ae = Ae * (sgn[:, :, None] * sgn[:, None, :])
    rows = np.broadcast_to(elem_edges[:, :, None], (Nt, 6, 6))
    cols = np.broadcast_to(elem_edges[:, None, :], (Nt, 6, 6))
    A = sp.coo_matrix((Ae.ravel(), (rows.ravel(), cols.ravel())),
                      shape=(Ne, Ne), dtype=Ae.dtype)
    return A.tocsr()


# ----------------------------------------------------------------------------
# Assembly of A = S - w^2 T  (Ne x Ne complex), optional UPML. Vectorised.
# ----------------------------------------------------------------------------
def assemble(nodes, tets, elem_edges, elem_sign, eps_c_elem, freq,
             mu_elem=None, pml_fn=None):
    w = 2.0 * np.pi * freq
    Ne = int(elem_edges.max()) + 1
    Nt = tets.shape[0]
    if mu_elem is None:
        mu_elem = np.full(Nt, MU0)
    eps_c_elem = np.asarray(eps_c_elem)

    vol, grads = batch_geom(nodes, tets)
    cvec = batch_curl(grads)                               # (Nt,6,3)

    if pml_fn is None:
        Lam = np.ones((Nt, 3), dtype=np.complex128)
        Laminv = Lam
    else:
        cen = nodes[tets].mean(axis=1)                    # (Nt,3)
        s = np.atleast_2d(pml_fn(cen))                     # (Nt,3) complex
        sx, sy, sz = s[:, 0], s[:, 1], s[:, 2]
        Lam = np.stack([sy * sz / sx, sz * sx / sy, sx * sy / sz], axis=1)
        Laminv = 1.0 / Lam

    # curl-curl:  (1/mu) sum_d Laminv_d cvec_m,d cvec_n,d * vol
    cw = cvec * Laminv[:, None, :]                         # (Nt,6,3) complex
    S = np.einsum('tmd,tnd->tmn', cw, cvec) * (vol / mu_elem)[:, None, None]

    # anisotropic vector mass:  eps_c sum_d Lam_d int (W_m)_d (W_n)_d
    gLmetric = np.einsum('td,tad,tbd->tab', Lam, grads, grads)   # (Nt,4,4)
    T = _quad_from_metric(gLmetric, vol) * eps_c_elem[:, None, None]

    Ae = S - w * w * T
    return _scatter(elem_edges, elem_sign, Ae, Ne), Ne


# ----------------------------------------------------------------------------
# Boundary edge DOFs = tangential circulation of a prescribed field.
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
            acc += wq * np.dot(field_fn(A + xi * d), d)
        vals[e] = acc
    return vals


def boundary_edges(nodes, edges, L, tol=1e-9):
    """Edges lying in any of the 6 cube faces (both endpoints on that plane)."""
    a, b = nodes[edges[:, 0]], nodes[edges[:, 1]]
    mask = np.zeros(edges.shape[0], dtype=bool)
    for d in range(3):
        for val in (0.0, L):
            mask |= (np.abs(a[:, d] - val) < tol) & (np.abs(b[:, d] - val) < tol)
    return np.where(mask)[0]


def solve_reduced(Aff, b, method="direct", tol=1e-9, drop_tol=1e-4,
                  fill_factor=30, verbose=False):
    """Solve Aff x = b. method='direct' (sparse LU) or 'iter' (ILU-preconditioned
    GMRES; scales to fine 3D meshes where LU fill-in is prohibitive). The ILU
    fill_factor is the key knob: 30 gives ~5 GMRES iterations on the lossy
    curl-curl system, versus hundreds at 15 (curl null-space)."""
    Aff = Aff.tocsc()
    if method == "direct":
        return spla.spsolve(Aff, b)
    ilu = spla.spilu(Aff, drop_tol=drop_tol, fill_factor=fill_factor)
    M = spla.LinearOperator(Aff.shape, ilu.solve)
    it = {"n": 0}
    cb = (lambda rk: it.__setitem__("n", it["n"] + 1)) if verbose else None
    x, info = spla.gmres(Aff, b, M=M, rtol=tol, atol=0.0, restart=200,
                         maxiter=2000, callback=cb, callback_type="pr_norm")
    if verbose:
        res = np.linalg.norm(Aff @ x - b) / np.linalg.norm(b)
        print(f"    [gmres] iters {it['n']}  rel.res {res:.2e}  info {info}")
    return x


# ---- auxiliary-space Maxwell (AMS / Hiptmair-Xu) preconditioner --------------
def discrete_gradient(edges, Nn):
    """G : nodes -> edges. Row e = (n_low, n_high) has -1 at n_low, +1 at n_high,
    so G phi is the edge circulation of grad(phi) (curl(G phi) = 0 exactly)."""
    ne = edges.shape[0]
    rows = np.repeat(np.arange(ne), 2)
    cols = edges.reshape(-1)                                # [low, high, ...]
    vals = np.tile([-1.0, 1.0], ne)
    return sp.coo_matrix((vals, (rows, cols)), shape=(ne, Nn)).tocsr()


def pi_operator(nodes, edges):
    """Pi : (H1)^3 nodal -> H(curl) edges. Maps a nodal vector field to edge
    circulations; Pi(constant c) = c . (x_high - x_low) reproduces constants."""
    ne = edges.shape[0]; Nn = nodes.shape[0]
    d = nodes[edges[:, 1]] - nodes[edges[:, 0]]            # (ne,3)
    rows, cols, vals = [], [], []
    ar = np.arange(ne)
    for c in range(3):
        rows += [ar, ar]
        cols += [3 * edges[:, 0] + c, 3 * edges[:, 1] + c]
        vals += [0.5 * d[:, c], 0.5 * d[:, c]]
    return sp.coo_matrix((np.concatenate(vals),
                          (np.concatenate(rows), np.concatenate(cols))),
                         shape=(ne, 3 * Nn)).tocsr()


def make_ams(Aff, nodes, edges, free_edges, dir_edges):
    """Hiptmair-Xu AMS preconditioner for the interior curl-curl system Aff.
    B^-1 r = D^-1 r + G_r A_G^-1 G_r^H r + Pi_r A_Pi^-1 Pi_r^H r, with the two
    NODAL auxiliary Poisson systems (much better LU fill than curl-curl) factored
    once. Nodes touching a Dirichlet edge are pinned (grounded)."""
    Nn = nodes.shape[0]
    pin = np.unique(edges[dir_edges].reshape(-1)) if len(dir_edges) else \
        np.array([], dtype=np.int64)
    freeN = np.setdiff1d(np.arange(Nn), pin)
    G = discrete_gradient(edges, Nn)[free_edges][:, freeN].tocsr()
    cols3 = (3 * freeN[:, None] + np.arange(3)).ravel()
    Pi = pi_operator(nodes, edges)[free_edges][:, cols3].tocsr()

    A_G = (G.conj().T @ Aff @ G).tocsc()
    A_Pi = (Pi.conj().T @ Aff @ Pi).tocsc()
    luG = spla.splu(A_G); luPi = spla.splu(A_Pi)
    Dinv = 1.0 / Aff.diagonal()
    GH, PiH = G.conj().T, Pi.conj().T

    def apply(r):
        return (Dinv * r + G @ luG.solve(GH @ r) + Pi @ luPi.solve(PiH @ r))
    return spla.LinearOperator(Aff.shape, apply, dtype=np.complex128), \
        (A_G.shape[0], A_Pi.shape[0])


def solve_dirichlet(A, Ne, dir_edges, dir_vals, rhs=None, method="direct",
                    tol=1e-9, geom=None):
    if rhs is None:
        rhs = np.zeros(Ne, dtype=np.complex128)
    free = np.setdiff1d(np.arange(Ne), dir_edges)
    Eb = np.zeros(Ne, dtype=np.complex128)
    Eb[dir_edges] = dir_vals
    b = rhs[free] - A[free][:, dir_edges] @ dir_vals
    Aff = A[free][:, free]
    if method == "ams":
        nodes, edges = geom
        M, sz = make_ams(Aff.tocsc(), nodes, edges, free, dir_edges)
        it = {"n": 0}
        x, info = spla.gmres(Aff.tocsc(), b, M=M, rtol=tol, atol=0.0,
                             restart=100, maxiter=500,
                             callback=lambda rk: it.__setitem__("n", it["n"] + 1),
                             callback_type="pr_norm")
        Ef = x
    else:
        Ef = solve_reduced(Aff, b, method=method, tol=tol)
    E = Eb.copy(); E[free] = Ef
    return E


# ----------------------------------------------------------------------------
# Scattered-field RHS (phantom contrast source), 3D.
# ----------------------------------------------------------------------------
def scattered_rhs(nodes, tets, elem_edges, elem_sign, Ne, freq,
                  eps_c_elem, eps_bg, Einc_fn):
    """RHS_i = int w^2 (eps_c - eps_bg) W_i . E_inc dV, E_inc at centroid.
    Vectorised; Einc_fn must accept a (M,3) point array and return (M,3)."""
    w = 2.0 * np.pi * freq
    vol, grads = batch_geom(nodes, tets)
    cen = nodes[tets].mean(axis=1)                         # (Nt,3)
    Einc = np.atleast_2d(Einc_fn(cen))                     # (Nt,3)
    intW = (vol[:, None, None] / 4.0) * (grads[:, _QM, :] - grads[:, _PM, :])
    val = (w * w * (np.asarray(eps_c_elem) - eps_bg)[:, None]
           * np.einsum('tec,tc->te', intW, Einc) * elem_sign)   # (Nt,6)
    rhs = np.zeros(Ne, dtype=np.complex128)
    np.add.at(rhs, elem_edges.ravel(), val.ravel())
    return rhs


def make_pml(L, d, strength=12.0, order=2):
    """UPML stretch s(x)=1 - j*strength*(depth/d)^order per axis (quadratic).
    pml_fn accepts a single point (3,) -> (3,) or a batch (M,3) -> (M,3)."""
    def prof(v):
        v = np.asarray(v, dtype=float)
        p = np.zeros_like(v)
        m1 = v < d
        p[m1] = ((d - v[m1]) / d) ** order
        m2 = v > L - d
        p[m2] = ((v[m2] - (L - d)) / d) ** order
        return p

    def pml_fn(xyz):
        a = np.atleast_2d(np.asarray(xyz, dtype=float))    # (M,3)
        s = 1.0 - 1j * strength * prof(a)                  # (M,3)
        return s if s.shape[0] > 1 else s[0]
    return pml_fn


# ----------------------------------------------------------------------------
# Field reconstruction at tetra centroids (lam = 1/4).
# ----------------------------------------------------------------------------
# descending-order axis permutation -> tet index in _PERMS, via code a*9+b*3+c
_PERM_LUT = np.full(27, -1, dtype=np.int64)
for _i, _pm in enumerate(_PERMS):
    _PERM_LUT[_pm[0] * 9 + _pm[1] * 3 + _pm[2]] = _i


def field_at_points(nodes, tets, elem_edges, elem_sign, E, pts, N, L):
    """Evaluate the Whitney vector field at arbitrary points on the structured
    Freudenthal mesh (vectorised over points). The tet inside a voxel is the Kuhn
    simplex selected by the descending order of the local coordinates (u,v,w)."""
    pts = np.asarray(pts, dtype=float)
    single = pts.ndim == 1
    pts = np.atleast_2d(pts)
    M = pts.shape[0]
    h = L / (N - 1)
    ijk = np.clip((pts / h).astype(int), 0, N - 2)          # (M,3)
    loc = pts / h - ijk
    order = np.argsort(-loc, axis=1)                        # descending
    lt = _PERM_LUT[order[:, 0] * 9 + order[:, 1] * 3 + order[:, 2]]
    vidx = (ijk[:, 0] * (N - 1) + ijk[:, 1]) * (N - 1) + ijk[:, 2]
    t = 6 * vidx + lt                                       # (M,)
    tet = tets[t]                                           # (M,4)
    P = nodes[tet]                                          # (M,4,3)
    Mmat = np.concatenate([np.ones((M, 4, 1)), P], axis=2)  # (M,4,4)
    Cinv = np.linalg.inv(Mmat)
    grads = np.transpose(Cinv[:, 1:4, :], (0, 2, 1))        # (M,4,3)
    lam = Cinv[:, 0, :] + np.einsum('mid,md->mi', grads, pts)   # (M,4)
    lp = lam[:, _PM]; lq = lam[:, _QM]                      # (M,6)
    gp = grads[:, _PM, :]; gq = grads[:, _QM, :]            # (M,6,3)
    W = lp[:, :, None] * gq - lq[:, :, None] * gp           # (M,6,3)
    coeff = E[elem_edges[t]] * elem_sign[t]                 # (M,6)
    vec = np.einsum('me,mec->mc', coeff, W)                 # (M,3)
    return vec[0] if single else vec


def field_at_centroids(nodes, tets, elem_edges, elem_sign, E):
    """Vectorised: Whitney field at each tet centroid (lam = 1/4)."""
    vol, grads = batch_geom(nodes, tets)
    Wc = (grads[:, _QM, :] - grads[:, _PM, :]) / 4.0       # (Nt,6,3)
    coeff = E[elem_edges] * elem_sign                      # (Nt,6)
    out = np.einsum('te,tec->tc', coeff, Wc)               # (Nt,3)
    cen = nodes[tets].mean(axis=1)
    return cen, out
