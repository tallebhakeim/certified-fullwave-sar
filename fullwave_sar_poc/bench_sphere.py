"""
Benchmark: our 3D full-wave FEM vs the EXACT Mie sphere.
========================================================

First genuinely 3D validation of the 3D edge solver against a closed-form truth
(previous 3D checks -- manufactured plane wave, TMM slab -- are 1D in substance).
This is also the benchmark CST/HFSS/COMSOL are judged against, so it puts us on
comparable ground.

Setup: lossy tissue sphere in vacuum, y-polarised plane wave travelling +x,
scattered-field formulation with UPML on all six faces. Measured quantity is the
regulated functional
    P_abs = (1/2) int_sphere sigma |E_total|^2 dV
evaluated with the SAME Whitney sigma-mass quadratic form used by the certified
bracket (certify3d), so the benchmark measures exactly what we certify.

Expected honest outcome: the structured tet mesh STAIRCASES the sphere surface,
which is the dominant error at these resolutions -- precisely the weakness that
conforming curvilinear meshes (HFSS/COMSOL) do not have. Quantifying it is the
point.

Run:  python3 bench_sphere.py
"""

import numpy as np

from edge_fem3d import (build_edges, assemble, boundary_edges, solve_dirichlet,
                        scattered_rhs, edge_circulation, EPS0, MU0, C0)
from certify3d import assemble_real
from phantom3d import build_box, make_pml_axes, _planewave_y
from mie import sphere_Pabs


def run_case(freq, radius, eps_r, sigma, L, N, n_pml=4, method="direct"):
    Ls = (L, L, L); Ns = (N, N, N)
    h = L / (N - 1)
    d_pml = n_pml * h
    w = 2 * np.pi * freq

    nodes, tets = build_box(Ls, Ns)
    edges, ee, es = build_edges(tets)
    cen = nodes[tets].mean(axis=1)
    ctr = np.array([L / 2, L / 2, L / 2])
    inside = np.linalg.norm(cen - ctr, axis=1) < radius

    er = np.where(inside, eps_r, 1.0)
    sg = np.where(inside, sigma, 0.0)
    eps_c = EPS0 * er - 1j * sg / w
    eps_bg = EPS0 * 1.0 + 0j

    pml = make_pml_axes(Ls, d_pml, axes=(0, 1, 2), strength=14.0)
    A, Ne = assemble(nodes, tets, ee, es, eps_c, freq, pml_fn=pml)
    k0 = w * np.sqrt(MU0 * eps_bg)
    Einc_fn = _planewave_y(np.real(k0))
    rhs = scattered_rhs(nodes, tets, ee, es, Ne, freq, eps_c, eps_bg, Einc_fn)
    dir_e = boundary_edges(nodes, edges, L)
    Es = solve_dirichlet(A, Ne, dir_e, np.zeros(len(dir_e), complex), rhs,
                         method=method, tol=1e-9, geom=(nodes, edges))

    # total field DOFs = scattered + incident circulations
    Einc_dofs = edge_circulation(nodes, edges, Einc_fn)
    Etot = Es + Einc_dofs

    # P_abs with the certified sigma-mass quadratic form (sigma = 0 outside)
    Msig = assemble_real(nodes, tets, ee, es, sg, 0.0)
    P_fem = 0.5 * np.real(np.vdot(Etot, Msig @ Etot))

    # discrete sphere volume actually meshed (staircase diagnostic)
    from edge_fem3d import batch_geom
    vol, _ = batch_geom(nodes, tets)
    V_disc = vol[inside].sum()
    V_true = 4.0 / 3.0 * np.pi * radius ** 3
    return P_fem, Ne, h, V_disc / V_true


def main():
    freq, radius, eps_r, sigma = 500e6, 0.05, 50.0, 1.0
    L = 0.24
    P_exact, qabs, x, m = sphere_Pabs(freq, radius, eps_r, sigma)
    lam0 = C0 / freq
    lam_in = lam0 / np.sqrt(eps_r)
    print("=== FEM 3D vs EXACT Mie sphere ===")
    print(f"  f {freq/1e6:.0f} MHz, a {radius*100:.0f} cm, eps_r {eps_r}, "
          f"sigma {sigma} S/m")
    print(f"  size param x={x:.3f}, m={m:.3f}, Q_abs={qabs:.4f}")
    print(f"  lambda0 {lam0*100:.1f} cm, lambda_tissue {lam_in*100:.2f} cm, "
          f"domain L {L*100:.0f} cm")
    print(f"  P_abs EXACT (Mie) = {P_exact:.6e} W\n")
    print(f"{'N':>4} {'#edges':>8} {'h[mm]':>7} {'a/h':>5} {'lam_in/h':>8} "
          f"{'Vdisc/V':>8} {'P_fem':>12} {'err%':>8}")
    prev = None
    for N in [12, 16, 20, 24]:
        P, Ne, h, vr = run_case(freq, radius, eps_r, sigma, L, N)
        err = (P - P_exact) / P_exact * 100
        print(f"{N:>4} {Ne:>8} {h*1e3:>7.2f} {radius/h:>5.1f} {lam_in/h:>8.1f} "
              f"{vr:>8.3f} {P:>12.5e} {err:>+8.2f}")
        prev = err
    print("\n  Note: the structured tet mesh staircases the sphere surface;")
    print("  Vdisc/V shows how well the discrete volume matches the true one.")


if __name__ == "__main__":
    main()
