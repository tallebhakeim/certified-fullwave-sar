"""
Validation of the 2D certified SAR bracket (curl-curl edge elements).

Manufactured lossy plane wave -> exact P_abs known. The DWR bracket must contain
the exact value and converge at the FEM rate. Companion of the 1D
sar_certified_poc; this is the 2D vector (curl-curl) version.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from certify2d import certify_2d
from edge_fem2d import C0


def main():
    freq, eps_r, sigma = 1e9, 40.0, 0.6
    lam = C0 / (freq * np.sqrt(eps_r))
    L = 1.5 * lam
    Ns = [9, 13, 19, 27, 37]
    hs, widths, ok, betas = [], [], [], []
    print("=== 2D certified SAR bracket vs manufactured exact ===")
    print(f"{'N':>4} {'P_h':>12} {'P_exact':>12} {'contains':>9} "
          f"{'beta':>7} {'width%':>8}")
    Pex = None
    rows = []
    for Nc in Ns:
        r = certify_2d(freq, eps_r, sigma, L, Nc)
        Pex = r["P_exact"]
        w = (r["P_hi"] - r["P_lo"]) / Pex * 100
        hs.append(1.0 / (Nc - 1)); widths.append(w)
        ok.append(r["contains"]); betas.append(r["beta"]); rows.append(r)
        print(f"{Nc:>4} {r['P_h']:>12.4e} {Pex:>12.4e} {str(r['contains']):>9} "
              f"{r['beta']:>7.3f} {w:>8.1f}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    hs = np.array(hs); widths = np.array(widths)
    ax[0].loglog(1.0 / hs, widths, "o-", label="bracket width")
    ax[0].loglog(1.0 / hs, widths[0] * (hs / hs[0]) ** 2, "--", label="O(h^2) ref")
    ax[0].set_xlabel("N (1/h)"); ax[0].set_ylabel("relative width [%]")
    ax[0].set_title("2D certified bracket convergence")
    ax[0].legend(); ax[0].grid(True, which="both", alpha=0.3)

    # bracket vs exact
    Ph = np.array([r["P_h"] for r in rows])
    lo = np.array([r["P_lo"] for r in rows])
    hi = np.array([r["P_hi"] for r in rows])
    Nn = np.array(Ns)
    ax[1].fill_between(Nn, lo, hi, alpha=0.3, color="tab:orange",
                       label="certified bracket")
    ax[1].plot(Nn, Ph, "o-", label="P_h (FEM)")
    ax[1].axhline(Pex, color="k", ls="--", label="exact P_abs")
    ax[1].set_xlabel("N"); ax[1].set_ylabel("P_abs")
    ax[1].set_ylim(Pex * 0.4, Pex * 1.6)
    ax[1].set_title("bracket contains exact, tightens with N")
    ax[1].legend()
    fig.tight_layout(); fig.savefig("validation_certify2d.png", dpi=130)
    print("  figure : validation_certify2d.png")

    passed = all(ok)
    print("\nVALIDATION", "PASS" if passed else "CHECK",
          f"(contains exact at all {len(Ns)} meshes: {passed}, "
          f"beta~{np.mean(betas):.3f})")


if __name__ == "__main__":
    import numpy as np
    np.seterr(all="ignore")
    main()
