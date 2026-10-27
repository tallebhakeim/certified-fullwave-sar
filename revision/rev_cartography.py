"""Revision R1 (Reviewer 2, Fig. 4): 2D field and SAR maps of the layered phantom,
plotted against depth below the skin surface, with ALL four interfaces drawn
(the submitted figure omitted the muscle/air interface, so the muscle appeared
to extend to the top of the plotted region). Same solver and data as before:
edge_fem2d scattered-field solution, 2 GHz, h = 1 mm, UPML 14 cells.
Also redraws the SAR profile against the transfer-matrix reference (Fig. 2).
Outputs: figures/fig_cartography.png, figures/fig_profile.png, results/cartography.json"""
import numpy as np
from rev_common import FIG, plt
from edge_fem2d import (build_mesh, build_edges, assemble, boundary_edges,
                        solve_dirichlet, make_pml, scattered_rhs,
                        field_at_points, EPS0, MU0)
from validate_voxel import TISSUE, layered_phantom, tmm_profile
import json
from rev_common import RES

freq = 2e9; w = 2 * np.pi * freq
L, N, n_pml = 0.20, 201, 14
nodes, tris = build_mesh(L, N)
edges, ee, es = build_edges(tris)
h = L / (N - 1); d_pml = n_pml * h
ys, tissues = layered_phantom(L, d_pml)
cen = nodes[tris].mean(axis=1)
reg = np.digitize(cen[:, 1], ys[1:-1])
eps_c = np.array([EPS0 * TISSUE[tissues[r]][0] - 1j * TISSUE[tissues[r]][1] / w for r in reg])
pml = make_pml(L, d_pml, strength=16.0)
A, Ne = assemble(nodes, tris, ee, es, eps_c, freq, pml_fn=pml)
k0 = w * np.sqrt(MU0 * EPS0)
Einc_fn = lambda p: np.array([np.exp(-1j * k0 * p[1]), 0.0])
rhs = scattered_rhs(nodes, tris, ee, es, Ne, freq, eps_c, EPS0 + 0j, Einc_fn)
bed = boundary_edges(nodes, edges, L)
Es = solve_dirichlet(A, Ne, bed, np.zeros(len(bed), complex), rhs)

gx = np.linspace(d_pml, L - d_pml, 220)
XX, YY = np.meshgrid(gx, gx)
pts = np.column_stack([XX.ravel(), YY.ravel()])
Et = field_at_points(nodes, tris, ee, es, Es, pts, N, L) + np.array([Einc_fn(p) for p in pts])
Ex = Et[:, 0].reshape(XX.shape)
rg = np.digitize(YY.ravel(), ys[1:-1])
sg = np.array([TISSUE[tissues[m]][1] for m in rg]); rho = np.array([TISSUE[tissues[m]][2] for m in rg])
SAR = (sg * np.abs(Et[:, 0]) ** 2 / (2 * rho)).reshape(XX.shape)

y_s = ys[1]                                        # skin surface -> depth origin
ext = [(d_pml - L / 2) * 100, (L / 2 - d_pml) * 100, (d_pml - y_s) * 100, (L - d_pml - y_s) * 100]
fig, ax = plt.subplots(1, 3, figsize=(7.2, 3.0))
panels = [(np.real(Ex), r"(a) Re$(E_x)$ (V m$^{-1}$)", "RdBu_r", None),
          (np.abs(Ex), r"(b) $|E_x|$ (V m$^{-1}$)", "viridis", None),
          (SAR, r"(c) SAR (W kg$^{-1}$)", "inferno", np.percentile(SAR, 99.5))]
for a, (F, ttl, cmap, vmax) in zip(ax, panels):
    im = a.imshow(F, origin="lower", extent=ext, cmap=cmap, aspect="equal", vmax=vmax)
    for yb in ys[1:5]:
        a.axhline((yb - y_s) * 100, color="w", lw=0.7, ls="--")
    for i, name in enumerate(tissues):
        y0, y1 = max(ys[i], d_pml), min(ys[i + 1], L - d_pml)
        a.text(ext[0] + 0.4, ((y0 + y1) / 2 - y_s) * 100, name, color="w", fontsize=6.5,
               va="center", fontweight="bold")
    a.set_title(ttl); a.set_xlabel("$x$ (cm)")
    cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=6.5)
ax[0].set_ylabel("depth below skin surface (cm)")
fig.tight_layout(); fig.savefig(f"{FIG}/fig_cartography.png")
print("interfaces (depth, cm):", [(round((y - y_s) * 100, 2)) for y in ys[1:5]])

# ---- SAR along the cut x = L/2 against the transfer-matrix reference ----
yq = np.linspace(d_pml, L - d_pml, 500)
pq = np.column_stack([np.full_like(yq, L / 2), yq])
Exq = field_at_points(nodes, tris, ee, es, Es, pq, N, L)[:, 0] + np.exp(-1j * k0 * yq)
Ext = tmm_profile(freq, ys, tissues, yq) * np.exp(-1j * k0 * ys[1])
rq = np.digitize(yq, ys[1:-1])
sq = np.array([TISSUE[tissues[m]][1] for m in rq]); rq_ = np.array([TISSUE[tissues[m]][2] for m in rq])
S_fem = sq * np.abs(Exq) ** 2 / (2 * rq_); S_ref = sq * np.abs(Ext) ** 2 / (2 * rq_)
m = sq > 0
err = np.linalg.norm((S_fem - S_ref)[m]) / np.linalg.norm(S_ref[m])
json.dump(dict(h_mm=h * 1e3, rel_L2_SAR_err=err, peak_fem=S_fem.max(), peak_ref=S_ref.max()),
          open(f"{RES}/cartography.json", "w"), indent=1)
print("SAR profile rel. L2 error %.4f" % err)
fig, a = plt.subplots(figsize=(3.6, 2.7))
dq = (yq - y_s) * 100
a.plot(dq, S_ref * 1e3, "k-", lw=1.2, label="reference (transfer-matrix)")
a.plot(dq[::4], S_fem[::4] * 1e3, "o", ms=2.2, color="C1", label="edge elements, $h=1$ mm")
for yb in ys[1:5]:
    a.axvline((yb - y_s) * 100, color="0.5", ls=":", lw=0.6)
a.set_xlim(-1, 8.5)
a.set_xlabel("depth below skin surface (cm)"); a.set_ylabel(r"SAR (mW kg$^{-1}$)")
a.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_profile.png")
