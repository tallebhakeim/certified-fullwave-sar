"""
3D voxel phantom -> certified-ready SAR map, validated against the 1D TMM.
=========================================================================

Two deliverables on the full 3D Whitney edge solver (edge_fem3d):

1. validate_slab() : a layered tissue phantom (air/skin/fat/muscle stacked along
   x) in a domain that is THIN in y,z. A y-polarised plane wave travels in +x.
   Exploiting the y,z uniformity of the field, exact boundary walls make the thin
   slab represent an infinite layered medium at 1D cost while still exercising the
   full 3D operator (3D tets, 3D curl-curl, 3D UPML in x):
     - x faces : UPML (absorb) + Dirichlet 0 on the outer wall,
     - y faces : PEC (tangential E = 0), exact since E = y_hat is face-normal,
     - z faces : natural (magnetic wall n x H = 0), exact since H = z_hat.
   SAR(x) is compared to the transfer-matrix (TMM) closed form.

2. demo_sphere() : a small cube phantom with a spherical muscle inclusion in fat,
   plane-wave illuminated, PML on all six faces -> a genuine 3D SAR volume shown
   as three orthogonal slices (the "montrable" 3D map).

Run:  python3 phantom3d.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from edge_fem3d import (build_edges, assemble, boundary_edges, solve_dirichlet,
                        scattered_rhs, field_at_centroids, field_at_points,
                        batch_geom, EPS0, MU0, C0)

# tissue properties @ ~2 GHz : (eps_r, sigma [S/m], rho [kg/m^3])
TISSUE = {
    "air":    (1.0, 0.0, 1.0),
    "skin":   (38.0, 1.30, 1100.0),
    "fat":    (5.3, 0.10, 910.0),
    "muscle": (53.0, 1.45, 1040.0),
}

_PERMS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]


def _planewave_y(k0):
    """y-polarised plane wave travelling +x; accepts a point (3,) or batch (M,3)."""
    def f(p):
        p = np.asarray(p, float); single = p.ndim == 1
        p = np.atleast_2d(p)
        out = np.zeros((p.shape[0], 3), dtype=np.complex128)
        out[:, 1] = np.exp(-1j * k0 * p[:, 0])
        return out[0] if single else out
    return f


# ----------------------------------------------------------------------------
# Rectangular-box Freudenthal-Kuhn mesh (per-axis extent and node count).
# ----------------------------------------------------------------------------
def build_box(Ls, Ns):
    Lx, Ly, Lz = Ls; Nx, Ny, Nz = Ns
    xs = np.linspace(0, Lx, Nx); ys = np.linspace(0, Ly, Ny)
    zs = np.linspace(0, Lz, Nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    def nid(i, j, k):
        return (i * Ny + j) * Nz + k

    corner = [(c & 1, (c >> 1) & 1, (c >> 2) & 1) for c in range(8)]
    cube_tets = []
    for perm in _PERMS:
        path = [0]; cur = 0
        for axis in perm:
            cur |= (1 << axis); path.append(cur)
        cube_tets.append(tuple(path))
    tets = []
    for i in range(Nx - 1):
        for j in range(Ny - 1):
            for k in range(Nz - 1):
                gc = [nid(i + dx, j + dy, k + dz) for (dx, dy, dz) in corner]
                for tt in cube_tets:
                    tets.append(tuple(gc[c] for c in tt))
    return nodes, np.asarray(tets, dtype=np.int64)


def make_pml_axes(Ls, d, axes, strength=16.0, order=2):
    """UPML stretch only on the requested axes (d = PML depth)."""
    Ls = np.asarray(Ls, float)

    def prof(v, Laxis):
        v = np.asarray(v, float); p = np.zeros_like(v)
        m1 = v < d; p[m1] = ((d - v[m1]) / d) ** order
        m2 = v > Laxis - d; p[m2] = ((v[m2] - (Laxis - d)) / d) ** order
        return p

    def pml_fn(xyz):
        a = np.atleast_2d(np.asarray(xyz, float))
        s = np.ones_like(a, dtype=np.complex128)
        for ax in axes:
            s[:, ax] = 1.0 - 1j * strength * prof(a[:, ax], Ls[ax])
        return s if s.shape[0] > 1 else s[0]
    return pml_fn


def box_face_edges(nodes, edges, Ls, which, tol=1e-9):
    """Edges lying in the requested faces. which: list of ('x',0.0)/('x',Lx)..."""
    a, b = nodes[edges[:, 0]], nodes[edges[:, 1]]
    axmap = {"x": 0, "y": 1, "z": 2}
    mask = np.zeros(edges.shape[0], dtype=bool)
    for axis, val in which:
        d = axmap[axis]
        mask |= (np.abs(a[:, d] - val) < tol) & (np.abs(b[:, d] - val) < tol)
    return np.where(mask)[0]


# ----------------------------------------------------------------------------
# TMM closed-form reference along the stacking axis (semi-infinite air both ends)
# ----------------------------------------------------------------------------
def tmm_profile(freq, xs_if, tissues, x_query):
    w = 2 * np.pi * freq

    def keta(name):
        er, sg, _ = TISSUE[name]
        epsc = EPS0 * er - 1j * sg / w
        k = w * np.sqrt(MU0 * epsc)
        return k, w * MU0 / k

    T = np.eye(2, dtype=np.complex128)
    for m in range(1, len(tissues) - 1):
        k, eta = keta(tissues[m]); dm = xs_if[m + 1] - xs_if[m]
        Tm = np.array([[np.cos(k * dm), 1j * eta * np.sin(k * dm)],
                       [1j * np.sin(k * dm) / eta, np.cos(k * dm)]])
        T = Tm @ T
    k0, eta0 = keta(tissues[0]); kl, etal = keta(tissues[-1])
    x1, xtop = xs_if[1], xs_if[-2]
    a, b = T[0]; c, dd = T[1]
    M = np.array([[a + b / eta0, -1.0], [c + dd / eta0, 1.0 / etal]],
                 dtype=np.complex128)
    rhs = np.array([-(a - b / eta0), -(c - dd / eta0)], dtype=np.complex128)
    r, t = np.linalg.solve(M, rhs)

    Eq = np.zeros(len(x_query), dtype=np.complex128)
    for i, x in enumerate(x_query):
        if x <= x1:
            Eq[i] = np.exp(-1j * k0 * (x - x1)) + r * np.exp(1j * k0 * (x - x1))
        elif x >= xtop:
            Eq[i] = t * np.exp(-1j * kl * (x - xtop))
        else:
            v = np.array([1 + r, (-1 + r) / eta0], dtype=np.complex128)
            xx = x1; m = 1
            while m < len(tissues) - 1:
                k, eta = keta(tissues[m]); top = xs_if[m + 1]
                seg = min(x, top) - xx
                Tm = np.array([[np.cos(k * seg), 1j * eta * np.sin(k * seg)],
                               [1j * np.sin(k * seg) / eta, np.cos(k * seg)]])
                v = Tm @ v; xx = min(x, top)
                if x <= top:
                    break
                m += 1
            Eq[i] = v[0]
    return Eq


# ----------------------------------------------------------------------------
# 1. Layered-slab validation vs TMM.
# ----------------------------------------------------------------------------
def validate_slab(freq=2e9, Nx=161):
    Lx = 0.20
    hx = Lx / (Nx - 1)
    n_pml = 14
    d_pml = n_pml * hx
    Lt = 6 * hx                                   # thin transverse extent
    Ls = (Lx, Lt, Lt); Ns = (Nx, 3, 3)

    x0 = d_pml + 0.030
    xs_if = [0.0, x0, x0 + 0.010, x0 + 0.010 + 0.020,
             x0 + 0.010 + 0.020 + 0.040, Lx]
    tissues = ["air", "skin", "fat", "muscle", "air"]

    nodes, tets = build_box(Ls, Ns)
    edges, ee, es = build_edges(tets)
    cen = nodes[tets].mean(axis=1)
    w = 2 * np.pi * freq
    reg = np.digitize(cen[:, 0], xs_if[1:-1])
    er = np.array([TISSUE[tissues[m]][0] for m in reg])
    sg = np.array([TISSUE[tissues[m]][1] for m in reg])
    rho = np.array([TISSUE[tissues[m]][2] for m in reg])
    eps_c = EPS0 * er - 1j * sg / w

    eps_bg = EPS0 * 1.0 + 0j
    pml = make_pml_axes(Ls, d_pml, axes=(0,), strength=16.0)
    A, Ne = assemble(nodes, tets, ee, es, eps_c, freq, pml_fn=pml)

    k0 = w * np.sqrt(MU0 * eps_bg)
    Einc_fn = _planewave_y(k0)
    rhs = scattered_rhs(nodes, tets, ee, es, Ne, freq, eps_c, eps_bg, Einc_fn)

    # Dirichlet 0 on x-outer walls (PML) and y walls (PEC); z walls left natural.
    dir_e = box_face_edges(nodes, edges, Ls,
                           [("x", 0.0), ("x", Lx), ("y", 0.0), ("y", Lt)])
    Es = solve_dirichlet(A, Ne, dir_e, np.zeros(len(dir_e), complex), rhs)

    # total field at centroids, SAR
    cen_c, Es_c = field_at_centroids(nodes, tets, ee, es, Es)
    Etot = Es_c + Einc_fn(cen_c)
    SAR_fem = sg * np.abs(Etot[:, 1]) ** 2 / (2 * rho)

    # collapse to a depth profile (bin centroids by x)
    order = np.argsort(cen_c[:, 0])
    xq = cen_c[order, 0]
    SARq = SAR_fem[order]
    Ex_fem = Etot[order, 1]

    Ex_tmm = tmm_profile(freq, xs_if, tissues, xq) * np.exp(-1j * k0 * xs_if[1])
    regq = np.digitize(xq, xs_if[1:-1])
    sgq = np.array([TISSUE[tissues[m]][1] for m in regq])
    rhoq = np.array([TISSUE[tissues[m]][2] for m in regq])
    SAR_tmm = sgq * np.abs(Ex_tmm) ** 2 / (2 * rhoq)
    tissue = sgq > 0

    errE = (np.linalg.norm((Ex_fem - Ex_tmm)[tissue])
            / np.linalg.norm(Ex_tmm[tissue]))
    errSAR = (np.linalg.norm((SARq - SAR_tmm)[tissue])
              / np.linalg.norm(SAR_tmm[tissue]))
    print("=== 3D layered slab, SAR vs TMM (2 GHz) ===")
    print(f"  domain      : {Lx*100:.0f} cm x {Lt*1e3:.1f} mm x {Lt*1e3:.1f} mm, "
          f"Nx={Nx} (hx={hx*1e3:.2f} mm), thin 3x3")
    print(f"  #tets {tets.shape[0]}  #edges {Ne}")
    print(f"  layers      : air | skin 1 | fat 2 | muscle 4 (cm) | air")
    print(f"  rel L2 err E  : {errE*100:.2f} %   (in tissue)")
    print(f"  rel L2 err SAR: {errSAR*100:.2f} %")
    print(f"  peak SAR fem {SARq[tissue].max():.3e}  tmm {SAR_tmm[tissue].max():.3e}")
    print(f"  VERDICT: {'PASS' if errSAR < 0.10 else 'CHECK'}")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(xq * 100, SAR_tmm, "-", lw=2, label="TMM (exact)")
    ax.plot(xq * 100, SARq, "o", ms=2.5, label="3D FEM (vector)")
    for xb in xs_if[1:-1]:
        ax.axvline(xb * 100, color="k", ls=":", lw=0.6)
    ax.set_xlabel("depth x [cm]"); ax.set_ylabel("SAR [W/kg]")
    ax.set_title(f"3D slab SAR vs TMM (err {errSAR*100:.2f} %)")
    ax.legend(); fig.tight_layout()
    fig.savefig("validation_slab3d.png", dpi=130)
    print("  figure      : validation_slab3d.png")
    return errSAR


# ----------------------------------------------------------------------------
# 2. Cube phantom with a spherical inclusion -> 3D SAR map (three slices).
# ----------------------------------------------------------------------------
def demo_sphere(freq=1.5e9, N=22):
    lam0 = C0 / freq
    L = 1.2 * lam0
    h = L / (N - 1)
    n_pml = 6
    d_pml = n_pml * h
    Ls = (L, L, L); Ns = (N, N, N)

    nodes, tets = build_box(Ls, Ns)
    edges, ee, es = build_edges(tets)
    cen = nodes[tets].mean(axis=1)
    w = 2 * np.pi * freq

    # fat body filling the physical region, with a muscle sphere inside
    ctr = np.array([L / 2, L / 2, L / 2])
    rad = 0.28 * L
    inside = np.linalg.norm(cen - ctr, axis=1) < rad
    body = ((cen > d_pml) & (cen < L - d_pml)).all(axis=1)
    name = np.where(body, np.where(inside, "muscle", "fat"), "air")
    er = np.array([TISSUE[n][0] for n in name])
    sg = np.array([TISSUE[n][1] for n in name])
    rho = np.array([TISSUE[n][2] for n in name])
    rho = np.where(rho > 0, rho, 1.0)
    eps_c = EPS0 * er - 1j * sg / w

    eps_bg = EPS0 * 1.0 + 0j
    pml = make_pml_axes(Ls, d_pml, axes=(0, 1, 2), strength=14.0)
    A, Ne = assemble(nodes, tets, ee, es, eps_c, freq, pml_fn=pml)
    k0 = w * np.sqrt(MU0 * eps_bg)
    Einc_fn = _planewave_y(k0)
    rhs = scattered_rhs(nodes, tets, ee, es, Ne, freq, eps_c, eps_bg, Einc_fn)
    dir_e = boundary_edges(nodes, edges, L)          # cube: PML all faces
    print(f"\n=== 3D cube phantom (muscle sphere in fat), {freq/1e9:.1f} GHz ===")
    method = "ams" if Ne > 60000 else "direct"
    print(f"  N={N}  #tets {tets.shape[0]}  #edges {Ne}  ({method})")
    Es = solve_dirichlet(A, Ne, dir_e, np.zeros(len(dir_e), complex), rhs,
                         method=method, tol=1e-8, geom=(nodes, edges))

    print(f"  peak SAR    : computing slices...")

    # smooth SAR on three orthogonal mid-planes via Whitney interpolation
    def sar_on(pts):
        Es_p = field_at_points(nodes, tets, ee, es, Es, pts, N, L)
        Et = Es_p + Einc_fn(pts)
        r = np.linalg.norm(pts - ctr, axis=1)
        inside_p = r < rad
        body_p = ((pts > d_pml) & (pts < L - d_pml)).all(axis=1)
        nm = np.where(body_p, np.where(inside_p, "muscle", "fat"), "air")
        sgp = np.array([TISSUE[n][1] for n in nm])
        rhp = np.array([TISSUE[n][2] for n in nm]); rhp = np.where(rhp > 0, rhp, 1.0)
        return sgp * np.sum(np.abs(Et) ** 2, axis=1) / (2 * rhp)

    gx = np.linspace(d_pml, L - d_pml, 120)
    XX, YY = np.meshgrid(gx, gx)
    mid = L / 2
    planes = {}
    for tag, build in {
        "xy (z=mid)": lambda: np.column_stack([XX.ravel(), YY.ravel(),
                                               np.full(XX.size, mid)]),
        "xz (y=mid)": lambda: np.column_stack([XX.ravel(), np.full(XX.size, mid),
                                               YY.ravel()]),
        "yz (x=mid)": lambda: np.column_stack([np.full(XX.size, mid),
                                               XX.ravel(), YY.ravel()]),
    }.items():
        planes[tag] = sar_on(build()).reshape(XX.shape)

    vmax = np.percentile(np.concatenate([p.ravel() for p in planes.values()]),
                         99.0)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
    ext = [d_pml * 100, (L - d_pml) * 100, d_pml * 100, (L - d_pml) * 100]
    for a_, (tag, P) in zip(ax, planes.items()):
        im = a_.imshow(P, origin="lower", extent=ext, cmap="inferno",
                       aspect="equal", vmax=vmax)
        a_.set_title(f"SAR slice {tag}")
        a_.set_xlabel("[cm]"); a_.set_ylabel("[cm]")
        fig.colorbar(im, ax=a_, fraction=0.046)
    fig.suptitle("3D SAR map: muscle sphere in fat, plane-wave illuminated "
                 f"({freq/1e9:.1f} GHz)")
    fig.tight_layout(); fig.savefig("demo_sphere3d.png", dpi=130)
    peak = max(p.max() for p in planes.values())
    print(f"  peak SAR (slices): {peak:.3e} W/kg")
    print("  figure      : demo_sphere3d.png")


if __name__ == "__main__":
    validate_slab()
    demo_sphere()
