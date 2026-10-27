"""
Real 3D XCAT head phantom -> certified-ready full-wave SAR map (WP3).
====================================================================

Loads the anatomical XCAT Head voxel volume (Interfile .h33/.i33, 1 mm), crops
and downsamples it to a solver-sized voxel grid, maps tissue labels to dielectric
properties, and runs the 3D Whitney edge solver (scattered-field plane wave +
UPML) to produce a SAR map on real head anatomy. This is the WP3 figure of the
AAP pre-project: the certified full-wave machinery on a real phantom.

The label set of this Voxelized mesh is {0,1,2,3,7}; the tissue assignment below
is a plausible dosimetric mapping (air / brain / fat / muscle-skin / bone) and is
flagged as approximate, pending the exact XCAT legend.

Run:  python3 xcat3d.py [N]
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from edge_fem3d import (build_edges, assemble, solve_dirichlet, scattered_rhs,
                        field_at_centroids, EPS0, MU0, C0)
from phantom3d import build_box, make_pml_axes, box_face_edges

XCAT_HEAD = os.path.join(os.environ.get("XCAT_DIR", "data"), "VoxHead_338x248x256.i33")
XDIM = (338, 248, 256)   # nx, ny, nz (1 mm voxels)

# label -> (name, eps_r, sigma [S/m], rho [kg/m^3]) at ~1.5 GHz (Gabriel/IT'IS).
# Assignment approximate for the 5-label Voxelized head; 7=bone is consistent
# with the XCAT thorax legend.
LABELMAP = {
    0: ("air",    1.0,  0.00, 1.2),
    1: ("brain",  48.0, 1.30, 1040.0),
    2: ("fat",    5.5,  0.10, 910.0),
    3: ("muscle", 54.0, 1.20, 1040.0),
    7: ("bone",   12.0, 0.16, 1900.0),
}


def load_head_cube(N):
    """Load the head volume, crop to the head bounding cube, downsample to N^3."""
    nx, ny, nz = XDIM
    vol = np.fromfile(XCAT_HEAD, dtype="<u2").reshape(nz, ny, nx)  # (z,y,x)
    vol = np.transpose(vol, (2, 1, 0))                            # (x,y,z)
    body = vol > 0
    xs, ys, zs = np.where(body)
    lo = np.array([xs.min(), ys.min(), zs.min()])
    hi = np.array([xs.max(), ys.max(), zs.max()])
    # bias the centre upward (smaller z index range) to focus on the head, not
    # the neck/shoulders, and take a tighter cube.
    ctr = (lo + hi) // 2
    half = int(np.max(hi - lo) * 0.40)                            # cube half-size
    lo = np.maximum(ctr - half, 0)
    hi = np.minimum(ctr + half, np.array(XDIM) - 1)
    crop = vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    side_mm = max(crop.shape)                                     # ~ cube side, mm
    # nearest-neighbour downsample to N^3 (preserve labels)
    idx = [np.linspace(0, s - 1, N).round().astype(int) for s in crop.shape]
    small = crop[np.ix_(idx[0], idx[1], idx[2])]
    L = side_mm * 1e-3                                            # domain side [m]
    return small.astype(np.int64), L


def run(freq=1.0e9, N=40, n_pml=4):
    lab, Lbody = load_head_cube(N)
    # pad the physical domain with a PML/air margin around the head cube
    hb = Lbody / (N - 1)
    margin = 1
    off = n_pml + margin                                         # body index offset
    L = Lbody + 2 * off * hb
    Nfull = N + 2 * off
    Ls = (L, L, L); Ns = (Nfull, Nfull, Nfull)
    h = L / (Nfull - 1)
    d_pml = n_pml * h

    nodes, tets = build_box(Ls, Ns)
    edges, ee, es = build_edges(tets)
    cen = nodes[tets].mean(axis=1)
    w = 2 * np.pi * freq

    # label per tet by nearest body voxel (air outside the body cube)
    ijk = np.floor(cen / h).astype(int) - off
    inside = np.all((ijk >= 0) & (ijk < N), axis=1)
    labt = np.zeros(tets.shape[0], dtype=np.int64)
    ii = np.clip(ijk, 0, N - 1)
    labt[inside] = lab[ii[inside, 0], ii[inside, 1], ii[inside, 2]]

    er = np.array([LABELMAP[l][1] for l in labt])
    sg = np.array([LABELMAP[l][2] for l in labt])
    rho = np.array([LABELMAP[l][3] for l in labt])
    eps_c = EPS0 * er - 1j * sg / w

    eps_bg = EPS0 * 1.0 + 0j
    pml = make_pml_axes(Ls, d_pml, axes=(0, 1, 2), strength=14.0)
    A, Ne = assemble(nodes, tets, ee, es, eps_c, freq, pml_fn=pml)
    k0 = w * np.sqrt(MU0 * eps_bg)

    def Einc_fn(p):                                        # y-pol plane wave, +x
        p = np.asarray(p, float); single = p.ndim == 1
        p = np.atleast_2d(p)
        out = np.zeros((p.shape[0], 3), dtype=np.complex128)
        out[:, 1] = np.exp(-1j * k0 * p[:, 0])
        return out[0] if single else out
    rhs = scattered_rhs(nodes, tets, ee, es, Ne, freq, eps_c, eps_bg, Einc_fn)
    dir_e = box_face_edges(nodes, edges, Ls,
                           [(a, v) for a in "xyz" for v in (0.0, L)])
    print(f"=== XCAT head 3D SAR, {freq/1e9:.2f} GHz ===")
    print(f"  head cube {Lbody*100:.1f} cm, domain {L*100:.1f} cm, "
          f"grid {Nfull}^3 (h={h*1e3:.1f} mm)")
    print(f"  #tets {tets.shape[0]}  #edges {Ne}")
    tissue_frac = np.mean(sg > 0)
    print(f"  tissue fill {tissue_frac*100:.0f}% ; labels {sorted(set(labt.tolist()))}")
    print(f"  solving (ams)...")
    Es = solve_dirichlet(A, Ne, dir_e, np.zeros(len(dir_e), complex), rhs,
                         method="ams", tol=1e-8, geom=(nodes, edges))

    cen_c, Es_c = field_at_centroids(nodes, tets, ee, es, Es)
    Etot = Es_c + Einc_fn(cen_c)
    SAR = np.where(sg > 0, sg * np.sum(np.abs(Etot) ** 2, axis=1)
                   / (2 * np.where(rho > 0, rho, 1.0)), 0.0)
    print(f"  peak SAR {SAR.max():.3e} W/kg  mean(tissue) "
          f"{SAR[sg>0].mean():.3e} W/kg")

    # three orthogonal mid-slices via nearest centroid, tissue outline overlaid
    from scipy.spatial import cKDTree
    tree = cKDTree(cen_c)
    g = np.linspace(d_pml, L - d_pml, 160)
    XX, YY = np.meshgrid(g, g); mid = L / 2
    labtree = cKDTree(cen_c)
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.6))
    slabs = {
        "coronal (z=mid)": np.column_stack([XX.ravel(), YY.ravel(),
                                            np.full(XX.size, mid)]),
        "sagittal (y=mid)": np.column_stack([XX.ravel(), np.full(XX.size, mid),
                                             YY.ravel()]),
        "axial (x=mid)": np.column_stack([np.full(XX.size, mid), XX.ravel(),
                                          YY.ravel()]),
    }
    Sm = {}
    for tag, pts in slabs.items():
        _, idx = tree.query(pts)
        Sm[tag] = SAR[idx].reshape(XX.shape)
    vmax = np.percentile(np.concatenate([s.ravel() for s in Sm.values()]), 99.0)
    ext = [d_pml * 100, (L - d_pml) * 100, d_pml * 100, (L - d_pml) * 100]
    for a_, (tag, S) in zip(ax, Sm.items()):
        im = a_.imshow(S.T, origin="lower", extent=ext, cmap="inferno",
                       aspect="equal", vmax=vmax)
        a_.set_title(f"SAR {tag}"); a_.set_xlabel("[cm]"); a_.set_ylabel("[cm]")
        fig.colorbar(im, ax=a_, fraction=0.046)
    fig.suptitle(f"3D full-wave SAR on the XCAT head phantom "
                 f"({freq/1e9:.2f} GHz, plane-wave, y-pol)")
    fig.tight_layout(); fig.savefig("xcat_head_sar3d.png", dpi=130)
    print("  figure   : xcat_head_sar3d.png")


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    run(N=N)
