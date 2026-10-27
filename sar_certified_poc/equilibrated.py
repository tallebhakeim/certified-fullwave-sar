"""
Guaranteed residual bound via equilibrated flux (1D, exact).
============================================================

Removes the saturation assumption of the hierarchical estimator: constructs an
equilibrated flux sigma_h (H(div)-conforming) that balances the discrete data
exactly element-by-element, so that

    ||R||_*  <=  eta := || E_h' - sigma_h ||_{L2}  +  C_trace |terminal defect|

is a GUARANTEED upper bound on the residual dual norm (no unknown constant:
C_trace is the explicit 1D trace constant). This closes the residual half of
the certified bracket; only the continuous inf-sup remains (see certify.py).

Construction (1D): flux sigma = E'. Balance  -sigma' = k^2 E  gives, per element,
    q_{i+1} = q_i - int_e k^2 E_h dx,
seeded at y=0 by the impedance BC  q_0 = j k0 E_h(0) - 2 j k0 a_inc.
The residual of the far BC at y=Y is the terminal defect, added as oscillation.
"""

import numpy as np


def equilibrated_eta(phantom, x, Eh, a_inc=1.0):
    k0 = phantom.k0
    n = len(x)
    q = np.empty(n, dtype=np.complex128)               # nodal equilibrated flux
    q[0] = 1j * k0 * Eh[0] - 2j * k0 * a_inc            # from impedance BC at 0
    for e in range(n - 1):
        he = x[e + 1] - x[e]
        k2 = phantom.k2_at(0.5 * (x[e] + x[e + 1]))
        react = k2 * he * (Eh[e] + Eh[e + 1]) / 2.0     # int_e k^2 E_h dx
        q[e + 1] = q[e] - react

    # ||E_h' - sigma_h||_{L2}^2 element by element
    eta2 = 0.0
    for e in range(n - 1):
        he = x[e + 1] - x[e]
        slope = (Eh[e + 1] - Eh[e]) / he                # E_h' (constant)
        di, dj = q[e] - slope, q[e + 1] - slope         # linear defect endpoints
        eta2 += he * (np.abs(di) ** 2 + np.real(di * np.conj(dj))
                      + np.abs(dj) ** 2) / 3.0
    eta_flux = np.sqrt(max(eta2, 0.0))

    # terminal defect at y=Y : BC wants sigma(Y) = -j k0 E_h(Y)
    defect_Y = q[-1] - (-1j * k0 * Eh[-1])
    Y = phantom.Y
    C_trace = np.sqrt(max(2.0 / Y, 2.0 * Y))            # explicit 1D trace const
    return eta_flux + C_trace * np.abs(defect_Y)


if __name__ == "__main__":
    # verify eta is a GUARANTEED upper bound on the true dual-norm residual
    from helmholtz1d import Phantom, solve_fem, interface_mesh, C0, DEFAULT_LAYERS
    import certify as C
    ph = Phantom(DEFAULT_LAYERS, 2e9)
    lam = C0 / 2e9 / np.sqrt(53)
    print(f"{'n':>5} {'eta_equilib':>12} {'rho_p(Riesz)':>13} {'eta>=rho_p':>11}")
    for frac in [8, 16, 32, 64]:
        x = interface_mesh(ph, lam / frac)
        _, Eh, _, _ = solve_fem(ph, x=x)
        eta = equilibrated_eta(ph, x, Eh)
        # reference true dual norm on a very fine space (Riesz)
        xf = C.refine(x, 8)
        Af, bf = C._assemble_form(ph, xf)
        Nf = C._norm_matrix(xf)
        Eh_f = np.interp(xf, x, Eh.real) + 1j * np.interp(xf, x, Eh.imag)
        rho_p = C._dual_norm(bf - Af @ Eh_f, Nf)
        print(f"{len(x)-1:>5} {eta:>12.4e} {rho_p:>13.4e} {str(eta >= rho_p):>11}")
