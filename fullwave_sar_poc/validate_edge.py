"""
Validation of the Whitney edge-element vector solver (marche 1).

Manufactured solution: a transverse plane wave  E = y_hat * exp(-j k x)  in a
(mildly lossy) homogeneous medium exactly satisfies  curl curl E - k^2 E = 0
with zero volume source. We impose its tangential circulation on the outer
boundary (Dirichlet) and solve the interior; the FEM field must reproduce the
plane wave. This isolates and validates the edge-element curl-curl machinery.

A convergence sweep confirms the error drops with mesh refinement.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from edge_fem2d import (build_mesh, build_edges, assemble, edge_circulation,
                        boundary_edges, solve_dirichlet, field_at_centroids,
                        EPS0, MU0, C0)


def run(freq, eps_r, sigma, N, L):
    w = 2 * np.pi * freq
    eps_c = EPS0 * eps_r - 1j * sigma / w
    k = w * np.sqrt(MU0 * eps_c)            # complex wavenumber (lossy)

    # exact transverse plane wave propagating +x, polarized along y
    Efn = lambda p: np.array([0.0, 1.0]) * np.exp(-1j * k * p[0])

    nodes, tris = build_mesh(L, N)
    edges, elem_edges, elem_sign = build_edges(tris)
    eps_c_elem = np.full(tris.shape[0], eps_c)
    A, Ne = assemble(nodes, tris, elem_edges, elem_sign, eps_c_elem, freq)

    bedges = boundary_edges(nodes, edges, L)
    dir_vals = edge_circulation(nodes, edges[bedges], Efn)
    E = solve_dirichlet(A, Ne, bedges, dir_vals)

    cen, Eh = field_at_centroids(nodes, tris, elem_edges, elem_sign, E)
    Eex = np.array([Efn(c) for c in cen])

    # error on interior centroids only (exclude cells touching the boundary)
    inside = np.all((cen > 0.12 * L) & (cen < 0.88 * L), axis=1)
    num = np.linalg.norm((Eh - Eex)[inside])
    den = np.linalg.norm(Eex[inside])
    return num / den, (nodes, tris, cen, Eh, Eex, k, L)


def main():
    freq = 500e6
    eps_r, sigma = 40.0, 0.6          # generic lossy tissue -> nonsingular system
    lam = C0 / (freq * np.sqrt(eps_r))
    L = 3.0 * lam

    print("=== Marche 1: edge-element curl-curl, manufactured plane wave ===")
    print(f"  freq {freq/1e6:.0f} MHz  eps_r {eps_r}  sigma {sigma} S/m  "
          f"lambda {lam*100:.2f} cm")
    print(f"  {'N':>5} {'h/lambda':>10} {'rel L2 err':>12}")
    errs, hs = [], []
    for N in [31, 51, 81, 121]:
        err, data = run(freq, eps_r, sigma, N, L)
        h = L / (N - 1) / lam
        errs.append(err); hs.append(h)
        print(f"  {N:>5} {h:>10.4f} {err*100:>11.3f}%")

    # observed convergence order
    p = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    print(f"  observed order ~ h^{p:.2f}")

    # figure on the finest mesh
    _, (nodes, tris, cen, Eh, Eex, k, L) = run(freq, eps_r, sigma, 121, L)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    xs = np.linspace(0, L, 300)
    ax[0].plot(xs / (L), np.real(np.exp(-1j * k * xs)), "-", lw=1.5,
               label="exact Re(Ey)")
    ln = np.abs(cen[:, 1] - L / 2) < 0.02 * L
    order = np.argsort(cen[ln, 0])
    ax[0].plot(cen[ln, 0][order] / L, np.real(Eh[ln, 1])[order], "o", ms=3,
               label="FEM Re(Ey)")
    ax[0].set_xlabel("x / L"); ax[0].set_ylabel("Re(Ey)")
    ax[0].set_title("plane-wave cut (y = L/2)"); ax[0].legend()

    ax[1].loglog(hs, errs, "o-", label="FEM error")
    ax[1].loglog(hs, np.array(hs) ** 2 * errs[0] / hs[0] ** 2, "--",
                 label="slope 2 ref")
    ax[1].set_xlabel("h / lambda"); ax[1].set_ylabel("rel L2 error")
    ax[1].set_title(f"convergence (~h^{p:.2f})"); ax[1].legend()
    fig.tight_layout(); fig.savefig("validation_edge.png", dpi=130)
    print("  figure : validation_edge.png")

    # lowest-order Nedelec -> theoretical O(h): validate on the CONVERGENCE
    # ORDER (and monotone decrease), not an arbitrary fixed error threshold.
    monotone = all(errs[i + 1] < errs[i] for i in range(len(errs) - 1))
    ok = (0.85 < p < 1.3) and monotone and errs[-1] < 0.05
    print("\nVALIDATION", "PASS" if ok else "CHECK",
          f"(order ~{p:.2f} [theory 1.0], monotone {monotone}, "
          f"finest err {errs[-1]*100:.2f}%)")


if __name__ == "__main__":
    main()
