"""
Marche 2: voxel phantom -> SAR map, validated against the 1D transfer-matrix
method (TMM) for a layered phantom under normal plane-wave incidence.

Setup: free space / skin / fat / muscle / free space stacked along y, a plane
wave E = x_hat exp(-j k0 y) incident from below (2 GHz). Scattered-field FEM
with UPML; SAR = sigma |E|^2 / (2 rho) from the TOTAL field. The exact profile
E_x(y), SAR(y) comes from the TMM (the engine behind voxel.html), providing a
closed-form reference for the full vector pipeline on heterogeneous tissue.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from edge_fem2d import (build_mesh, build_edges, assemble, boundary_edges,
                        solve_dirichlet, make_pml, scattered_rhs,
                        field_at_points, EPS0, MU0, C0)

# tissue properties @ ~2 GHz : (eps_r, sigma [S/m], rho [kg/m^3])
TISSUE = {
    "air":    (1.0, 0.0, 1.0),
    "skin":   (38.0, 1.30, 1100.0),
    "fat":    (5.3, 0.10, 910.0),
    "muscle": (53.0, 1.45, 1040.0),
}


def layered_phantom(L, d_pml):
    """Return interfaces (y positions) and the tissue of each region, bottom->top."""
    y0 = d_pml + 0.030                      # free-space gap above bottom PML
    layers = [("air", y0),
              ("skin", y0 + 0.010),
              ("fat", y0 + 0.010 + 0.020),
              ("muscle", y0 + 0.010 + 0.020 + 0.040)]
    # regions: air (0..y0), skin, fat, muscle, air (last..L)
    ys = [0.0] + [ly[1] for ly in layers] + [L]
    tissues = ["air", "skin", "fat", "muscle", "air"]
    return ys, tissues


def tmm_profile(freq, ys, tissues, y_query):
    """Exact E_x(y) for a plane wave (unit amplitude) incident from below,
    semi-infinite air both ends, via 2x2 transfer matrices."""
    w = 2 * np.pi * freq
    def kη(name):
        er, sg, _ = TISSUE[name]
        epsc = EPS0 * er - 1j * sg / w
        k = w * np.sqrt(MU0 * epsc)
        eta = w * MU0 / k
        return k, eta
    # region boundaries y_1..y_{R-1}; region m spans (ys[m], ys[m+1])
    # transfer from bottom plane ys[0]-ref (top of region 0) through inner layers
    # Build total matrix across the inner stack (regions 1..R-2), from y=ys[1] up
    # to y=ys[-2]. Regions 0 and last are the semi-infinite air half-spaces.
    k0, eta0 = kη(tissues[0])
    kl, etal = kη(tissues[-1])
    T = np.eye(2, dtype=np.complex128)
    for m in range(1, len(tissues) - 1):
        k, eta = kη(tissues[m])
        dm = ys[m + 1] - ys[m]
        Tm = np.array([[np.cos(k * dm), 1j * eta * np.sin(k * dm)],
                       [1j * np.sin(k * dm) / eta, np.cos(k * dm)]])
        T = Tm @ T
    # bottom reference plane at y=ys[1] (air/skin interface), top at y=ys[-2].
    # air below: E = e^{-jk0(y-ys1)} + r e^{+jk0(y-ys1)};  v=[E,H], H=(-fwd+bwd)/eta0
    # air above: E = t e^{-jkl(y-ys_top)}; v=[t, -t/etal]
    # v_top = T v_bot  ->  solve r,t
    ys1, ystop = ys[1], ys[-2]
    # unknowns r, t : v_bot=[1+r, (-1+r)/eta0], v_top=[t,-t/etal]
    # T @ [1+r,(-1+r)/eta0] = [t,-t/etal]
    a, b = T[0]; c, dd = T[1]
    # row0: a(1+r)+b(-1+r)/eta0 = t
    # row1: c(1+r)+dd(-1+r)/eta0 = -t/etal
    # unknowns r,t: rewrite
    # (a + b/eta0) r - t = -(a - b/eta0)
    # (c + dd/eta0) r + t/etal = -(c - dd/eta0)
    M = np.array([[a + b / eta0, -1.0],
                  [c + dd / eta0, 1.0 / etal]], dtype=np.complex128)
    rhs = np.array([-(a - b / eta0), -(c - dd / eta0)], dtype=np.complex128)
    r, t = np.linalg.solve(M, rhs)

    # reconstruct E at query points by propagating the state vector
    Eq = np.zeros(len(y_query), dtype=np.complex128)
    for i, y in enumerate(y_query):
        if y <= ys1:                      # bottom air
            Eq[i] = np.exp(-1j * k0 * (y - ys1)) + r * np.exp(1j * k0 * (y - ys1))
        elif y >= ystop:                  # top air
            Eq[i] = t * np.exp(-1j * kl * (y - ystop))
        else:                             # inside stack: propagate from ys1
            v = np.array([1 + r, (-1 + r) / eta0], dtype=np.complex128)
            # find region and propagate through completed layers + partial
            yy = ys1
            m = 1
            while m < len(tissues) - 1:
                k, eta = kη(tissues[m])
                top = ys[m + 1]
                seg = min(y, top) - yy
                Tm = np.array([[np.cos(k * seg), 1j * eta * np.sin(k * seg)],
                               [1j * np.sin(k * seg) / eta, np.cos(k * seg)]])
                v = Tm @ v
                yy = min(y, top)
                if y <= top:
                    break
                m += 1
            Eq[i] = v[0]
    return Eq


def main():
    freq = 2e9
    lam0 = C0 / freq
    L = 0.20                               # 20 cm square domain
    N = 201                                # h = 1 mm ~ lambda_muscle/20
    n_pml = 14
    nodes, tris = build_mesh(L, N)
    edges, elem_edges, elem_sign = build_edges(tris)
    h = L / (N - 1)
    d_pml = n_pml * h

    ys, tissues = layered_phantom(L, d_pml)
    cen = nodes[tris].mean(axis=1)
    # assign tissue per element by y-centroid
    eps_c = np.empty(tris.shape[0], dtype=np.complex128)
    sigma = np.empty(tris.shape[0]); rho = np.empty(tris.shape[0])
    w = 2 * np.pi * freq
    reg = np.digitize(cen[:, 1], ys[1:-1])          # region index 0..R-1
    for t in range(tris.shape[0]):
        er, sg, rr = TISSUE[tissues[reg[t]]]
        eps_c[t] = EPS0 * er - 1j * sg / w
        sigma[t] = sg; rho[t] = rr

    eps_bg = EPS0 * 1.0 + 0j                          # background = air
    pml = make_pml(L, d_pml, strength=16.0)
    A, Ne = assemble(nodes, tris, elem_edges, elem_sign, eps_c, freq, pml_fn=pml)

    k0 = w * np.sqrt(MU0 * eps_bg)
    Einc_fn = lambda p: np.array([np.exp(-1j * k0 * p[1]), 0.0])   # x-pol, +y
    rhs = scattered_rhs(nodes, tris, elem_edges, elem_sign, Ne, freq,
                        eps_c, eps_bg, Einc_fn)
    bedges = boundary_edges(nodes, edges, L)
    Es = solve_dirichlet(A, Ne, bedges, np.zeros(len(bedges), complex), rhs)

    # total field on a vertical cut x = L/2
    yq = np.linspace(0.02, L - 0.02, 400)
    pts = np.column_stack([np.full_like(yq, L / 2), yq])
    Es_cut = field_at_points(nodes, tris, elem_edges, elem_sign, Es, pts, N, L)
    Einc_cut = np.array([Einc_fn(p) for p in pts])
    Ex_fem = Es_cut[:, 0] + Einc_cut[:, 0]

    # TMM references the incident phase at ys[1]; FEM references it at y=0.
    # Align the phase reference so the two describe the SAME incident wave.
    Ex_tmm = tmm_profile(freq, ys, tissues, yq) * np.exp(-1j * k0 * ys[1])

    # SAR along the cut (both), using per-y tissue props
    regq = np.digitize(yq, ys[1:-1])
    sg_q = np.array([TISSUE[tissues[m]][1] for m in regq])
    rho_q = np.array([TISSUE[tissues[m]][2] for m in regq])
    tissue_mask = sg_q > 0
    SAR_fem = sg_q * np.abs(Ex_fem) ** 2 / (2 * rho_q)
    SAR_tmm = sg_q * np.abs(Ex_tmm) ** 2 / (2 * rho_q)

    err_E = (np.linalg.norm((Ex_fem - Ex_tmm)[tissue_mask])
             / np.linalg.norm(Ex_tmm[tissue_mask]))
    err_SAR = (np.linalg.norm((SAR_fem - SAR_tmm)[tissue_mask])
               / np.linalg.norm(SAR_tmm[tissue_mask]))
    peak_fem, peak_tmm = SAR_fem.max(), SAR_tmm.max()

    print("=== Marche 2: layered voxel phantom, SAR vs TMM (2 GHz) ===")
    print(f"  mesh          : {N}x{N}, h = {h*1e3:.2f} mm")
    print(f"  layers (cm)   : air | skin 1 | fat 2 | muscle 4 | air")
    print(f"  rel L2 err  E : {err_E*100:.2f} %   (in tissue)")
    print(f"  rel L2 err SAR: {err_SAR*100:.2f} %")
    print(f"  peak SAR fem  : {peak_fem:.3e}   tmm : {peak_tmm:.3e}   "
          f"ratio {peak_fem/peak_tmm:.3f}")

    # 2D SAR map over the PHYSICAL region only (exclude the PML border, whose
    # near-wall values would otherwise saturate the colour scale)
    gx = np.linspace(d_pml, L - d_pml, 200)
    XX, YY = np.meshgrid(gx, gx)
    ptsm = np.column_stack([XX.ravel(), YY.ravel()])
    Esm = field_at_points(nodes, tris, elem_edges, elem_sign, Es, ptsm, N, L)
    Eincm = np.array([Einc_fn(p) for p in ptsm])
    Etot = Esm + Eincm
    regm = np.digitize(YY.ravel(), ys[1:-1])
    sgm = np.array([TISSUE[tissues[m]][1] for m in regm])
    rhom = np.array([TISSUE[tissues[m]][2] for m in regm])
    SARm = (sgm * np.abs(Etot[:, 0]) ** 2 / (2 * rhom)).reshape(XX.shape)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].plot(yq * 100, SAR_tmm, "-", lw=2, label="TMM (exact)")
    ax[0].plot(yq * 100, SAR_fem, "o", ms=2.5, label="FEM vector")
    for yb in ys[1:-1]:
        ax[0].axvline(yb * 100, color="k", ls=":", lw=0.6)
    ax[0].set_xlabel("y [cm]"); ax[0].set_ylabel("SAR [W/kg]")
    ax[0].set_title(f"SAR profile (err {err_SAR*100:.2f} %)"); ax[0].legend()

    im = ax[1].imshow(SARm, origin="lower",
                      extent=[d_pml * 100, (L - d_pml) * 100,
                              d_pml * 100, (L - d_pml) * 100],
                      cmap="inferno", aspect="equal",
                      vmax=np.percentile(SARm, 99.5))
    ax[1].set_title("SAR map [W/kg] (physical region)")
    ax[1].set_xlabel("x [cm]")
    ax[1].set_ylabel("y [cm]")
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    fig.tight_layout(); fig.savefig("validation_voxel.png", dpi=130)
    print("  figure        : validation_voxel.png")

    ok = err_SAR < 0.10
    print("\nVALIDATION", "PASS" if ok else "CHECK",
          f"(SAR err {err_SAR*100:.2f}%, E err {err_E*100:.2f}%)")


if __name__ == "__main__":
    main()
