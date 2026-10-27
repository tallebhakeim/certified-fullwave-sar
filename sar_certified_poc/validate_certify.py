"""
Validation & demonstration of the certified SAR bracket.

Fig 1 (convergence): the certified interval [P_lo, P_hi] contains the exact
    P_abs at every mesh and its width converges at the O(h^2) FEM rate.

Fig 2 (resonance certificate): sweeping frequency across the Fabry-Perot modes
    of a low-loss dielectric slab, the inf-sup constant beta (= distance to
    resonance) collapses at each cavity mode and the certified bracket widens
    there -- the guarantee honestly reports where it weakens. This is the SAME
    cavity-mode mechanism seen in the 2D marche-1 PML study, now quantified.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from helmholtz1d import Phantom, interface_mesh, C0, DEFAULT_LAYERS
from certify import certify_Pabs


def convergence():
    ph = Phantom(DEFAULT_LAYERS, 2e9)
    Pex = ph.exact_Pabs()
    lam = C0 / 2e9 / np.sqrt(53)
    ns, widths, ok = [], [], []
    print("=== Fig 1: certified bracket convergence (2 GHz phantom) ===")
    print(f"exact P_abs = {Pex:.6e}")
    for frac in [8, 16, 32, 64, 128]:
        x = interface_mesh(ph, lam / frac)
        r = certify_Pabs(ph, x)
        inside = r["P_lo"] <= Pex <= r["P_hi"]
        w = (r["P_hi"] - r["P_lo"]) / Pex * 100
        ns.append(len(x) - 1); widths.append(w); ok.append(inside)
        print(f"  n={len(x)-1:4d}  width={w:6.2f}%  contains_exact={inside}  "
              f"beta={r['beta']:.3f}")
    return ph, Pex, ns, widths, all(ok)


def resonance_sweep():
    # low-loss high-permittivity slab (Fabry-Perot cavity) between air
    slab_er, slab_sig, d_slab = 80.0, 0.03, 0.02      # 2 cm, weakly lossy
    def make(freq):
        from helmholtz1d import TISSUE
        TISSUE["slab"] = (slab_er, slab_sig, 1000.0)
        layers = [("air", 0.01), ("slab", d_slab), ("air", 0.01)]
        return Phantom(layers, freq)

    freqs = np.linspace(0.6e9, 3.0e9, 90)
    Pex, betas, wlo, whi = [], [], [], []
    inside_all = True
    for f in freqs:
        ph = make(f)
        lam_slab = C0 / f / np.sqrt(slab_er)
        x = interface_mesh(ph, lam_slab / 20)
        r = certify_Pabs(ph, x)
        pe = ph.exact_Pabs()
        Pex.append(pe); betas.append(r["beta"])
        wlo.append(r["P_lo"]); whi.append(r["P_hi"])
        if not (r["P_lo"] <= pe <= r["P_hi"]):
            inside_all = False
    return (freqs, np.array(Pex), np.array(betas),
            np.array(wlo), np.array(whi), inside_all)


def main():
    ph, Pex, ns, widths, all_ok = convergence()
    (freqs, Pe, betas, wlo, whi, inside_all) = resonance_sweep()

    print("\n=== Fig 2: resonance certificate (Fabry-Perot sweep) ===")
    print(f"  min beta over sweep = {betas.min():.3e} at "
          f"{freqs[np.argmin(betas)]/1e9:.2f} GHz")
    print(f"  exact inside bracket at all {len(freqs)} frequencies: {inside_all}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))

    ax[0].loglog(ns, widths, "o-", label="bracket width")
    ax[0].loglog(ns, np.array(widths)[0] * (np.array(ns)[0] / np.array(ns)) ** 2,
                 "--", label="O(h^2) ref")
    ax[0].set_xlabel("n elements"); ax[0].set_ylabel("relative width [%]")
    ax[0].set_title("certified bracket convergence")
    ax[0].legend(); ax[0].grid(True, which="both", alpha=0.3)

    axP = ax[1]; axB = axP.twinx()
    fg = freqs / 1e9
    axP.fill_between(fg, wlo, whi, color="tab:orange", alpha=0.35,
                     label="certified bracket")
    axP.plot(fg, Pe, "k-", lw=1.4, label="exact P_abs")
    axB.semilogy(fg, betas, "tab:blue", lw=1.2, label="beta (certificate)")
    axP.set_xlabel("frequency [GHz]"); axP.set_ylabel("P_abs [W/kg-ish]")
    axB.set_ylabel("inf-sup beta", color="tab:blue")
    axB.tick_params(axis="y", labelcolor="tab:blue")
    axP.set_title("resonance certificate: beta dips -> bracket widens")
    axP.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig("validation_certify.png", dpi=130)
    print("  figure : validation_certify.png")

    ok = all_ok and inside_all
    print("\nVALIDATION", "PASS" if ok else "CHECK",
          f"(convergence contains exact: {all_ok}; sweep contains exact: {inside_all})")


if __name__ == "__main__":
    main()
