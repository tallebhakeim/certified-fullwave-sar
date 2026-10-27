"""Figure: full-wave SAR cartography on a REAL anatomical phantom, an XCAT
thoracic slice (air/muscle/lung/bone), 900 MHz plane-wave illumination."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from edge_fem2d import (build_mesh, build_edges, assemble, boundary_edges,
                        solve_dirichlet, make_pml, scattered_rhs,
                        field_at_points, EPS0, MU0, C0)
from xcat_thorax import load_square, PROPS_900

freq = 900e6
w = 2 * np.pi * freq
Ncell = 150
lab, L = load_square(Ncell)               # (Ncell, Ncell) label map, domain L [m]
N = Ncell + 1
nodes, tris = build_mesh(L, N)
edges, ee, es = build_edges(tris)
h = L / (N - 1)
n_pml = 12
d_pml = n_pml * h

# per-element dielectric from the label map (2 triangles per cell)
eps_c = np.empty(tris.shape[0], dtype=np.complex128)
sigma = np.empty(tris.shape[0]); rho = np.empty(tris.shape[0])
for j in range(N - 1):
    for i in range(N - 1):
        name, er, sg, rr = PROPS_900[int(lab[j, i])]
        for loc in (0, 1):
            t = 2 * (j * (N - 1) + i) + loc
            eps_c[t] = EPS0 * er - 1j * sg / w
            sigma[t] = sg; rho[t] = rr

eps_bg = EPS0 + 0j
pml = make_pml(L, d_pml, strength=16.0)
A, Ne = assemble(nodes, tris, ee, es, eps_c, freq, pml_fn=pml)
k0 = w * np.sqrt(MU0 * eps_bg)
Einc_fn = lambda p: np.array([np.exp(-1j * k0 * p[1]), 0.0])   # x-pol, +y
print("solving %d edges ..." % Ne)
rhs = scattered_rhs(nodes, tris, ee, es, Ne, freq, eps_c, eps_bg, Einc_fn)
bedges = boundary_edges(nodes, edges, L)
Es = solve_dirichlet(A, Ne, bedges, np.zeros(len(bedges), complex), rhs)

# sample on physical grid (crop PML)
gx = np.linspace(d_pml, L - d_pml, 240)
XX, YY = np.meshgrid(gx, gx)
pts = np.column_stack([XX.ravel(), YY.ravel()])
Es_g = field_at_points(nodes, tris, ee, es, Es, pts, N, L)
Etot = Es_g + np.array([Einc_fn(p) for p in pts])
# tissue at sample points (nearest cell)
ii = np.clip((XX.ravel() / h).astype(int), 0, N - 2)
jj = np.clip((YY.ravel() / h).astype(int), 0, N - 2)
labg = lab[jj, ii]
sgg = np.array([PROPS_900[int(v)][2] for v in labg])
rhog = np.array([PROPS_900[int(v)][3] for v in labg])
SAR = (sgg * np.abs(Etot[:, 0]) ** 2 / (2 * rhog)).reshape(XX.shape)
Emag = np.abs(Etot[:, 0]).reshape(XX.shape)
labshow = labg.reshape(XX.shape)

ext = [d_pml * 100, (L - d_pml) * 100, d_pml * 100, (L - d_pml) * 100]
fig, ax = plt.subplots(1, 3, figsize=(14, 4.7))
im0 = ax[0].imshow(labshow, origin="lower", extent=ext, cmap="tab10")
ax[0].set_title("(a) XCAT thorax tissues"); ax[0].set_ylabel("y [cm]")
im1 = ax[1].imshow(Emag, origin="lower", extent=ext, cmap="viridis")
ax[1].set_title("(b) $|E_x|$  [V/m per unit incident]")
im2 = ax[2].imshow(SAR, origin="lower", extent=ext, cmap="inferno",
                   vmax=np.percentile(SAR[SAR > 0], 99))
ax[2].set_title("(c) SAR  [W/kg per unit incident]")
for a in ax:
    a.set_xlabel("x [cm]")
    a.contour(XX * 100, YY * 100, (labshow > 0).astype(float), levels=[0.5],
              colors="w", linewidths=0.6)
ax[1].annotate("incident wave", xy=(ext[0] + 1, ext[0] + 1),
               color="w", fontsize=8)
ax[1].annotate("", xy=(ext[0] + 3, ext[0] + 4), xytext=(ext[0] + 3, ext[0] + 1),
               arrowprops=dict(arrowstyle="->", color="w"))
for im, a in [(im1, ax[1]), (im2, ax[2])]:
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig("paper_fig_realphantom.png", dpi=140)
print("saved paper_fig_realphantom.png")
