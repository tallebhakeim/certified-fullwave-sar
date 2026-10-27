"""Revision R1, 1D: inf-sup enclosure (k0-weighted norm), bracket convergence and
resonance sweep with the fully computable bracket (certify_rigorous.py).
Outputs: results/bracket1d.json, figures/fig_bracket1d.png"""
import json
import numpy as np
from rev_common import FIG, RES, plt
import certify_rigorous as R
from helmholtz1d import Phantom, interface_mesh, C0, DEFAULT_LAYERS, TISSUE

np.seterr(all="ignore")


def infsup_enclosure(ph, lam, fracs):
    """Two-sided geometric-tail enclosure of the continuous inf-sup constant.
    (H1) the last increments of beta_h have one sign and decay with ratio q < 1;
    then gamma lies between beta_h(finest) and beta_h(finest) + tail."""
    betas = np.array([R.gamma_minus(ph, interface_mesh(ph, lam / f)) for f in fracs])
    d = np.diff(betas)
    ratios = d[1:] / d[:-1]
    q = float(np.max(ratios[-2:])) if len(ratios) >= 2 else float(ratios[-1])
    ok = bool(0.0 < q < 1.0 and np.all(np.sign(d[-3:]) == np.sign(d[-1])))
    tail = d[-1] * q / (1.0 - q)
    glo, ghi = sorted([betas[-1], betas[-1] + tail])
    mono = bool(np.all(d > 0))
    return betas, mono, q, glo, ghi, ok


def convergence():
    ph = Phantom(DEFAULT_LAYERS, 2e9)
    lam = C0 / 2e9 / np.sqrt(53)
    fr_enc = [16, 32, 64, 128, 256, 512]
    betas, mono, q, glo, ghi, ok = infsup_enclosure(ph, lam, fr_enc)
    beta_check = R.gamma_minus(ph, interface_mesh(ph, lam / 1024))
    Pex = R.exact_window_output(ph, 0.0, ph.Y)
    rows = []
    for fr in [8, 16, 32, 64, 128, 256]:
        x = interface_mesh(ph, lam / fr)
        r = R.bracket(ph, x, glo)
        rows.append(dict(frac=fr, n=len(x) - 1, h_mm=float(np.diff(x).max() * 1e3),
                         J_h=r["J_h"], lo=r["lo"], hi=r["hi"], lin=r["lin"], quad=r["quad"],
                         eta_p=r["eta_p"], eta_d=r["eta_d"],
                         contains=bool(r["lo"] <= Pex <= r["hi"]),
                         width_pct=(r["hi"] - r["lo"]) / Pex * 100,
                         err=abs(Pex - r["J_h"]),
                         effectivity=(r["hi"] - r["lo"]) / (2 * abs(Pex - r["J_h"]))))
    enc = dict(fracs=fr_enc, betas=betas.tolist(), monotone=mono, q=q, H1_verified=ok,
               gamma_lo=glo, gamma_hi=ghi, beta_1024=beta_check,
               check_inside=bool(glo <= beta_check <= ghi))
    return dict(P_exact=Pex, enclosure=enc, rows=rows)


def resonance():
    slab = (80.0, 0.03)
    TISSUE["slab"] = (slab[0], slab[1], 1000.0)
    freqs = np.linspace(0.6e9, 3.0e9, 90)
    out = []
    for f in freqs:
        ph = Phantom([("air", 0.01), ("slab", 0.02), ("air", 0.01)], f)
        lam_s = C0 / f / np.sqrt(slab[0])
        betas, mono, q, glo, ghi, ok = infsup_enclosure(ph, lam_s, [20, 40, 80, 160, 320])
        x = interface_mesh(ph, lam_s / 20)
        r = R.bracket(ph, x, glo)
        pe = R.exact_window_output(ph, 0.0, ph.Y)
        out.append(dict(f=f, beta320=betas[-1], gamma_lo=glo, gamma_hi=ghi, q=q,
                        H1_verified=ok, increasing=mono, P_exact=pe,
                        J_h=r["J_h"], lo=r["lo"], hi=r["hi"],
                        contains=bool(r["lo"] <= pe <= r["hi"])))
    return out


def main():
    conv = convergence()
    res = resonance()
    json.dump(dict(convergence=conv, resonance=res), open(f"{RES}/bracket1d.json", "w"), indent=1)
    e = conv["enclosure"]
    print("inf-sup (k0 norm): betas", np.round(e["betas"], 7), "monotone", e["monotone"],
          "q=%.3f" % e["q"], "gamma in [%.7f, %.7f]" % (e["gamma_lo"], e["gamma_hi"]),
          "beta_1024=%.7f inside=%s" % (e["beta_1024"], e["check_inside"]))
    for r in conv["rows"]:
        print("frac %4d n %5d  [%.5e, %.5e] in=%s width %.3f%% eff %.1f" %
              (r["frac"], r["n"], r["lo"], r["hi"], r["contains"], r["width_pct"], r["effectivity"]))
    print("resonance: all contain", all(o["contains"] for o in res),
          " H1 verified at", sum(o["H1_verified"] for o in res), "/", len(res),
          " beta increasing at", sum(o["increasing"] for o in res),
          " min gamma_lo %.4f" % min(o["gamma_lo"] for o in res),
          " max rel enclosure width %.2e" % max((o["gamma_hi"] - o["gamma_lo"]) / o["gamma_lo"] for o in res))

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    rows = conv["rows"]
    h = np.array([r["h_mm"] for r in rows]); w = np.array([r["width_pct"] for r in rows])
    ax[0].loglog(h, w, "o-", color="C1", label="bracket width")
    ax[0].loglog(h, w[-1] * (h / h[-1]) ** 2, "k--", lw=0.8, label=r"$\propto h^{2}$")
    ax[0].set_xlabel("element size $h$ (mm)"); ax[0].set_ylabel("relative width (%)")
    ax[0].set_title(r"(a) bracket on $P_{\rm abs}$, 2 GHz stack"); ax[0].legend()
    ax[0].grid(True, which="both", lw=0.3)
    f = np.array([o["f"] for o in res]) / 1e9
    pe = np.array([o["P_exact"] for o in res])
    lo = np.array([o["lo"] for o in res]); hi = np.array([o["hi"] for o in res])
    be = np.array([o["gamma_lo"] for o in res])
    ax[1].fill_between(f, lo / pe, hi / pe, color="C1", alpha=0.35, label="bracket / reference")
    ax[1].axhline(1.0, color="k", lw=0.8)
    ax[1].set_ylim(0.0, 2.0)
    ax[1].set_xlabel("frequency (GHz)"); ax[1].set_ylabel(r"$P/P_{\rm ref}$")
    ax2 = ax[1].twinx(); ax2.plot(f, be, "C0-", lw=1, label=r"$\gamma_-$")
    ax2.set_ylabel(r"inf-sup lower bound $\gamma_-$", color="C0")
    ax[1].set_title("(b) low-loss slab: resonance sweep")
    h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, loc="upper right")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_bracket1d.png")


if __name__ == "__main__":
    main()
