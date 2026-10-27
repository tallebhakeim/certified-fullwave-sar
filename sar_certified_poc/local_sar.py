"""
Certified PEAK LOCAL (10 g-averaged) SAR -- the actual ICNIRP-regulated quantity.
=================================================================================

ICNIRP limits the peak spatial-average SAR over a 10 g tissue cube (side
~2.15 cm for rho~1000). Here a 2.15 cm averaging window slides over the tissue;
the mass-averaged SAR in each window is a QUADRATIC output
    SAR_avg(w) = ( (1/2) int_w sigma|E|^2 ) / ( int_w rho ),
bracketed by the same DWR certificate (windowed sigma-mass as output). The
regulated peak is then enclosed by
    peak in [ max_w SAR_lo(w) ,  max_w SAR_hi(w) ]   (guaranteed),
validated against the exact peak (transfer-matrix reference).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from helmholtz1d import Phantom, interface_mesh, TISSUE, C0, DEFAULT_LAYERS
from certify import certify_Pabs

D_AVG = 0.0215                       # 10 g cube side [m] (rho ~ 1000)


def window_sigma_mass(y0, y1):
    def fn(phantom, x):
        n = len(x)
        M = np.zeros((n, n))
        for e in range(n - 1):
            ym = 0.5 * (x[e] + x[e + 1])
            if not (y0 <= ym <= y1):
                continue
            sg = phantom.sigma_at(ym)
            he = x[e + 1] - x[e]
            M[e:e + 2, e:e + 2] += sg * np.array([[2., 1.], [1., 2.]]) * he / 6.0
        return M
    return fn


def window_mass(phantom, y0, y1, nq=400):
    """int_window rho dy  (mass per unit area)."""
    yy = np.linspace(y0, y1, nq)
    rho = np.array([TISSUE[phantom.tissue_at(y)][2] for y in yy])
    return np.trapezoid(rho, yy)


def exact_window_power(phantom, y0, y1, nq=600):
    """(1/2) int_window sigma |E_exact|^2 dy."""
    yy = np.linspace(y0, y1, nq)
    E = phantom.exact_field(yy)
    sg = np.array([phantom.sigma_at(y) for y in yy])
    return 0.5 * np.trapezoid(sg * np.abs(E) ** 2, yy)


def main():
    ph = Phantom(DEFAULT_LAYERS, 2e9)
    lam = C0 / 2e9 / np.sqrt(53)
    x = interface_mesh(ph, lam / 40)
    # tissue spans edges[1]..edges[-2]; slide the window inside it
    y_lo, y_hi = ph.edges[1], ph.edges[-2]
    y0s = np.arange(y_lo, y_hi - D_AVG + 1e-9, 0.003)

    S_lo, S_hi, S_ex, centers = [], [], [], []
    print("=== Certified peak local (10 g) SAR ===")
    for y0 in y0s:
        y1 = y0 + D_AVG
        m = window_mass(ph, y0, y1)
        r = certify_Pabs(ph, x, out_mass_fn=window_sigma_mass(y0, y1))
        S_lo.append(r["P_lo"] / m); S_hi.append(r["P_hi"] / m)
        S_ex.append(exact_window_power(ph, y0, y1) / m)
        centers.append((y0 + y1) / 2)
    S_lo, S_hi, S_ex = map(np.array, (S_lo, S_hi, S_ex))

    peak_lo = S_lo.max(); peak_hi = S_hi.max(); peak_ex = S_ex.max()
    inside = peak_lo <= peak_ex <= peak_hi
    print(f"  windows           : {len(y0s)}  (2.15 cm, step 3 mm)")
    print(f"  exact peak 10g SAR: {peak_ex:.4e}")
    print(f"  certified enclosure: [{peak_lo:.4e}, {peak_hi:.4e}]")
    print(f"  peak inside bracket: {inside}")
    print(f"  relative width    : {(peak_hi-peak_lo)/peak_ex*100:.2f} %")

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    c = np.array(centers) * 100
    ax.fill_between(c, S_lo, S_hi, alpha=0.35, color="tab:orange",
                    label="certified band")
    ax.plot(c, S_ex, "k-", lw=1.6, label="exact 10 g SAR")
    ax.axhline(peak_hi, color="tab:red", ls=":", lw=1,
               label=f"certified peak <= {peak_hi:.2e}")
    ax.set_xlabel("window centre [cm]"); ax.set_ylabel("10 g-averaged SAR [W/kg]")
    ax.set_title("certified peak local SAR (ICNIRP quantity)")
    ax.legend()
    fig.tight_layout(); fig.savefig("local_sar.png", dpi=130)
    print("  figure : local_sar.png")

    print("\nVALIDATION", "PASS" if inside else "CHECK",
          f"(peak enclosed: {inside})")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
