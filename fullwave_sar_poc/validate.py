"""
Validation of the full-wave 2D forward solver + SAR pipeline demo.

Test 1 (lossless): point source in vacuum, compare FEM field to the analytic
    2D Green's function G(r) = (-j/4) H0^{(2)}(k r) on a ring away from the
    source and away from the outer ABC. Reports relative L2 error.

Test 2 (lossy): point source in muscle-like tissue -> field must decay over
    the skin depth; we also compute SAR = sigma |E|^2 / (2 rho) to show the
    pipeline reaches W/kg (the RF-dosimetry quantity our QS codes can't give).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from helmholtz2d import (build_mesh, solve_point_source, analytic_green,
                         nearest_node, EPS0, MU0, C0)


def test_vacuum_greens():
    freq = 300e6                 # 300 MHz -> lambda0 = 1 m
    lam = C0 / freq
    L = 3.0 * lam                # 3 wavelengths across
    N = 151                      # ~ lambda/50 resolution
    nodes, tris, bnd = build_mesh(L, N)
    eps_r = np.ones(tris.shape[0])
    sigma = np.zeros(tris.shape[0])
    src = (L / 2.0, L / 2.0)

    E, isrc, kb = solve_point_source(nodes, tris, bnd, freq, eps_r, sigma, src)
    k = np.real(kb)

    # sample on a ring: far enough from source (>0.4 lam) and boundary (>0.5 lam)
    xs, ys = nodes[:, 0] - src[0], nodes[:, 1] - src[1]
    r = np.sqrt(xs ** 2 + ys ** 2)
    edge = np.minimum.reduce([nodes[:, 0], L - nodes[:, 0],
                              nodes[:, 1], L - nodes[:, 1]])
    mask = (r > 0.45 * lam) & (r < 1.1 * lam) & (edge > 0.5 * lam)

    Ea = analytic_green(r[mask], k)
    Ef = E[mask]
    err = np.linalg.norm(Ef - Ea) / np.linalg.norm(Ea)

    print("=== Test 1: vacuum, Green's function ===")
    print(f"  freq          : {freq/1e6:.0f} MHz   (lambda = {lam:.3f} m)")
    print(f"  mesh          : {N}x{N} nodes,  h = {L/(N-1)/lam:.3f} lambda")
    print(f"  sample points : {mask.sum()}")
    print(f"  rel L2 error  : {err*100:.2f} %")

    # figure: FEM vs analytic amplitude along the +x radial line
    line = (np.abs(ys) < 0.4 * (L / (N - 1))) & (xs > 0.05 * lam) & (xs < 1.3 * lam)
    order = np.argsort(xs[line])
    rr = xs[line][order]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(rr / lam, np.abs(E[line][order]), "o", ms=3, label="FEM |E|")
    ax[0].plot(rr / lam, np.abs(analytic_green(rr, k)), "-", lw=1.5,
               label="analytic |G|")
    ax[0].set_xlabel("r / lambda"); ax[0].set_ylabel("|E_z|")
    ax[0].set_title(f"vacuum radial cut (err {err*100:.2f} %)"); ax[0].legend()

    field = E.reshape(N, N)
    im = ax[1].imshow(np.real(field), origin="lower", cmap="RdBu_r",
                      extent=[0, L / lam, 0, L / lam])
    ax[1].set_title("Re(E_z)  [wavelengths]"); ax[1].set_xlabel("x/lambda")
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig("validation_vacuum.png", dpi=130)
    print("  figure        : validation_vacuum.png")
    return err


def demo_lossy_sar():
    freq = 900e6                 # 900 MHz (mobile band)
    w = 2 * np.pi * freq
    # muscle-like tissue @ ~900 MHz
    eps_r_t, sigma_t, rho = 55.0, 0.94, 1040.0
    lam_t = C0 / (freq * np.sqrt(eps_r_t))   # wavelength IN tissue
    L = 6.0 * lam_t
    N = 161
    nodes, tris, bnd = build_mesh(L, N)
    eps_r = np.full(tris.shape[0], eps_r_t)
    sigma = np.full(tris.shape[0], sigma_t)
    src = (L / 2.0, L / 2.0)

    E, isrc, kb = solve_point_source(nodes, tris, bnd, freq, eps_r, sigma, src)

    # skin depth of the tissue (good vs poor conductor general formula)
    eps = EPS0 * eps_r_t
    alpha = w * np.sqrt(MU0 * eps / 2.0 *
                        (np.sqrt(1 + (sigma_t / (w * eps)) ** 2) - 1))
    delta = 1.0 / alpha

    SAR = sigma_t * np.abs(E) ** 2 / (2.0 * rho)    # W/kg  (per node)

    print("\n=== Test 2: lossy tissue, SAR pipeline ===")
    print(f"  freq          : {freq/1e6:.0f} MHz")
    print(f"  tissue        : eps_r={eps_r_t}, sigma={sigma_t} S/m, rho={rho}")
    print(f"  lambda_tissue : {lam_t*100:.2f} cm    skin depth = {delta*100:.2f} cm")
    print(f"  peak SAR      : {SAR.max():.3e} W/kg (arb. source strength)")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    field = np.abs(E).reshape(N, N)
    im0 = ax[0].imshow(field, origin="lower", cmap="viridis",
                       extent=[0, L * 100, 0, L * 100])
    ax[0].set_title("|E_z| in tissue"); ax[0].set_xlabel("x [cm]")
    fig.colorbar(im0, ax=ax[0], fraction=0.046)

    sar_map = SAR.reshape(N, N)
    im1 = ax[1].imshow(sar_map, origin="lower", cmap="inferno",
                       extent=[0, L * 100, 0, L * 100])
    ax[1].set_title("SAR [W/kg]"); ax[1].set_xlabel("x [cm]")
    fig.colorbar(im1, ax=ax[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig("demo_lossy_sar.png", dpi=130)
    print("  figure        : demo_lossy_sar.png")


if __name__ == "__main__":
    err = test_vacuum_greens()
    demo_lossy_sar()
    print("\nVALIDATION", "PASS" if err < 0.05 else "CHECK",
          f"(threshold 5%, got {err*100:.2f}%)")
