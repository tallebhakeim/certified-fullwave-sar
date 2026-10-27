"""
Mie series for a homogeneous (lossy) sphere: the EXACT 3D reference.
====================================================================

Our 3D validations so far (manufactured plane wave, TMM slab) are 1D in
substance. The Mie sphere gives a genuine THREE-DIMENSIONAL closed-form truth,
and it is the benchmark CST/HFSS/COMSOL are also judged against, so it lets us
compare on equal terms.

Target quantity = the ABSORBED POWER P_abs (the certified functional):
    P_abs = Q_abs * pi a^2 * S_inc,     S_inc = |E0|^2 / (2 eta0)

Convention note. Our solver uses e^{+j w t} with eps_c = eps0 eps_r - j sigma/w.
Bohren & Huffman use e^{-i w t}, where the refractive index has a POSITIVE
imaginary part. The efficiencies Q are real and convention independent, so we
simply feed BH the conjugated relative permittivity:
    eps_rel_BH = eps_r + i sigma / (w eps0),   m = sqrt(eps_rel_BH),  Im(m) > 0.

Validated against (a) the classic BH test case m=1.55, x=5.213 -> Q_ext=3.10543,
and (b) the Rayleigh limit Q_abs -> 4 x Im((m^2-1)/(m^2+2)) as x -> 0.

Run:  python3 mie.py
"""

import numpy as np

EPS0 = 8.8541878128e-12
MU0 = 4.0e-7 * np.pi
C0 = 1.0 / np.sqrt(EPS0 * MU0)
ETA0 = np.sqrt(MU0 / EPS0)


def mie_coefficients(m, x, nmax=None):
    """Bohren-Huffman a_n, b_n for refractive index m (Im m >= 0) and size x."""
    if nmax is None:
        nmax = int(np.ceil(x + 4.0 * x ** (1.0 / 3.0) + 2))
    y = m * x
    nmx = int(max(nmax, abs(y)) + 15)

    # logarithmic derivative D_n(y) by downward recurrence
    D = np.zeros(nmx + 1, dtype=np.complex128)
    for n in range(nmx, 0, -1):
        D[n - 1] = n / y - 1.0 / (D[n] + n / y)

    a = np.zeros(nmax + 1, dtype=np.complex128)
    b = np.zeros(nmax + 1, dtype=np.complex128)
    # Riccati-Bessel psi, chi by upward recurrence
    psi_m1, psi_0 = np.cos(x), np.sin(x)          # psi_{-1}, psi_0
    chi_m1, chi_0 = -np.sin(x), np.cos(x)
    for n in range(1, nmax + 1):
        psi_n = (2 * n - 1) / x * psi_0 - psi_m1
        chi_n = (2 * n - 1) / x * chi_0 - chi_m1
        xi_n = psi_n - 1j * chi_n
        xi_nm1 = psi_0 - 1j * chi_0
        Dn = D[n]
        ta = Dn / m + n / x
        tb = Dn * m + n / x
        a[n] = (ta * psi_n - psi_0) / (ta * xi_n - xi_nm1)
        b[n] = (tb * psi_n - psi_0) / (tb * xi_n - xi_nm1)
        psi_m1, psi_0 = psi_0, psi_n
        chi_m1, chi_0 = chi_0, chi_n
    return a, b


def mie_efficiencies(m, x):
    """Q_ext, Q_sca, Q_abs."""
    a, b = mie_coefficients(m, x)
    n = np.arange(1, len(a))
    w = 2 * n + 1
    qext = (2.0 / x ** 2) * np.sum(w * np.real(a[1:] + b[1:]))
    qsca = (2.0 / x ** 2) * np.sum(w * (np.abs(a[1:]) ** 2 + np.abs(b[1:]) ** 2))
    return qext, qsca, qext - qsca


def sphere_Pabs(freq, radius, eps_r, sigma, E0=1.0):
    """Exact absorbed power [W] of a lossy sphere in vacuum under a plane wave of
    amplitude E0 [V/m], via the Mie series."""
    w = 2 * np.pi * freq
    k0 = w / C0
    x = k0 * radius
    eps_rel_bh = eps_r + 1j * sigma / (w * EPS0)      # BH convention (Im m > 0)
    m = np.sqrt(eps_rel_bh)
    if np.imag(m) < 0:
        m = np.conj(m)
    _, _, qabs = mie_efficiencies(m, x)
    S_inc = E0 ** 2 / (2.0 * ETA0)                    # incident Poynting [W/m^2]
    return qabs * np.pi * radius ** 2 * S_inc, qabs, x, m


# ----------------------------------------------------------------------------
# Validation of the reference itself.
# ----------------------------------------------------------------------------
def _validate():
    ok = True
    print("=== Mie reference validation ===")

    # (a) classic Bohren-Huffman test: m=1.55 (lossless), sphere r=0.525 um at
    #     lambda=0.6328 um -> x = 2 pi r / lambda (use the exact x, not 5.213)
    m = 1.55 + 0j
    x = 2 * np.pi * 0.525 / 0.6328
    qext, qsca, qabs = mie_efficiencies(m, x)
    ref = 3.10543
    e = abs(qext - ref) / ref
    print(f"  BH case m=1.55 x={x:.5f}: Q_ext {qext:.5f} (ref {ref})  "
          f"err {e*100:.4f} %   Q_abs {qabs:.2e} (must be ~0)")
    ok &= (e < 1e-3) and (abs(qabs) < 1e-8)

    # (b) Rayleigh limit: Q_abs -> 4 x Im((m^2-1)/(m^2+2))
    m = np.sqrt(50.0 + 1j * 5.0)
    print(f"  Rayleigh limit (m^2 = 50+5j):")
    for x in [1e-2, 3e-3, 1e-3]:
        _, _, qabs = mie_efficiencies(m, x)
        ray = 4.0 * x * np.imag((m ** 2 - 1) / (m ** 2 + 2))
        e = abs(qabs - ray) / ray
        print(f"    x={x:<7g} Q_abs {qabs:.6e}  Rayleigh {ray:.6e}  "
              f"err {e*100:.3f} %")
        ok &= e < 0.02

    # (c) energy consistency: Q_abs >= 0 and Q_ext >= Q_sca for a lossy sphere
    _, _, qabs = mie_efficiencies(np.sqrt(50 + 1j * 5), 1.0)
    ok &= qabs > 0
    print(f"\n  VERDICT: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    _validate()
    print("\n=== Exact P_abs for a tissue sphere (E0 = 1 V/m) ===")
    for f, a, er, sg in [(500e6, 0.05, 50.0, 1.0),
                         (1e9, 0.03, 50.0, 1.0)]:
        P, q, x, m = sphere_Pabs(f, a, er, sg)
        print(f"  f={f/1e6:.0f} MHz a={a*100:.0f} cm eps_r={er} sigma={sg}: "
              f"x={x:.3f} m={m:.3f}  Q_abs={q:.4f}  P_abs={P:.6e} W")
