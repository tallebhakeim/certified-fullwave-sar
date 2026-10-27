"""
Validation of the 3D Whitney edge solver by a manufactured lossy plane wave.
============================================================================

E = y_hat exp(-j k x) in a homogeneous lossy medium satisfies
    curl curl E - k^2 E = 0,   k = w sqrt(mu0 eps_c),  eps_c = eps0 eps_r - j sigma/w,
with ZERO volume source. Imposing the exact tangential circulation on all six
cube faces (Dirichlet) drives the interior, so the centroid field error isolates
the DISCRETIZATION error of the edge elements. Lowest-order Whitney -> O(h).

Run:  python3 validate_edge3d.py
"""

import numpy as np
from edge_fem3d import (build_mesh, build_edges, assemble, edge_circulation,
                        boundary_edges, solve_dirichlet, field_at_centroids,
                        EPS0, MU0, C0)


def run(freq=1.0e9, eps_r=40.0, sigma=0.6, ncells=(2, 3, 4, 5, 6)):
    w = 2 * np.pi * freq
    eps_c = EPS0 * eps_r - 1j * sigma / w
    k = w * np.sqrt(MU0 * eps_c)                       # Im(k) < 0 (decaying)
    lam = 2 * np.pi / np.real(k)
    L = 1.0 * lam
    khat = np.array([0.0, 1.0, 0.0])                   # E along y
    Efn = lambda p: khat * np.exp(-1j * k * p[0])

    print("=== 3D Whitney edge validation (manufactured lossy plane wave) ===")
    print(f"  freq {freq/1e9:.2f} GHz  eps_r {eps_r}  sigma {sigma}  "
          f"lam_medium {lam*100:.3f} cm  L {L*100:.3f} cm")
    print(f"{'N':>4} {'#tets':>8} {'#edges':>8} {'h/lam':>8} {'relL2%':>9} {'order':>7}")

    hs, errs = [], []
    for N in ncells:
        nodes, tets = build_mesh(L, N)
        edges, ee, es = build_edges(tets)
        epsc = np.full(tets.shape[0], eps_c)
        A, Ne = assemble(nodes, tets, ee, es, epsc, freq)
        bedges = boundary_edges(nodes, edges, L)
        dvals = edge_circulation(nodes, edges[bedges], Efn)
        E = solve_dirichlet(A, Ne, bedges, dvals)

        cen, Eh = field_at_centroids(nodes, tets, ee, es, E)
        Eex = np.array([Efn(c) for c in cen])
        num = np.sum(np.abs(Eh - Eex) ** 2)
        den = np.sum(np.abs(Eex) ** 2)
        rel = np.sqrt(num / den)
        h = L / (N - 1)
        order = ""
        if errs:
            order = f"{np.log(errs[-1]/rel)/np.log(hs[-1]/h):.2f}"
        print(f"{N:>4} {tets.shape[0]:>8} {Ne:>8} {h/lam:>8.4f} "
              f"{rel*100:>9.3f} {order:>7}")
        hs.append(h); errs.append(rel)

    # overall fitted order
    p = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    monotone = all(errs[i] > errs[i + 1] for i in range(len(errs) - 1))
    print(f"\n  fitted convergence order  = {p:.2f}  (Whitney theory 1.0)")
    print(f"  monotone decrease         = {monotone}")
    print(f"  VERDICT: {'PASS' if (0.75 < p < 1.4 and monotone) else 'CHECK'}")
    return hs, errs


if __name__ == "__main__":
    run()
