"""Revision R1: peak 10 g-averaged SAR (psSAR10g) bracket under refinement, the
max-over-windows lemma checked numerically, and the pass / fail / indeterminate
decision rule near the 2 W/kg local limit.
Outputs: results/peak10g.json, figures/fig_peak10g.png"""
import json
import numpy as np
from rev_common import FIG, RES, plt, ETA0
import certify_rigorous as R
from helmholtz1d import Phantom, interface_mesh, C0, DEFAULT_LAYERS

np.seterr(all="ignore")
D = 0.0215                 # side of the averaging cube (10 g at 1000 kg m^-3)
STEP = 0.0005              # window-centre grid, 0.5 mm
LIMIT = 2.0                # W/kg, ICNIRP local (head/trunk) basic restriction


def windows(ph):
    a, b = ph.edges[1], ph.edges[-2]             # tissue span (skin surface .. muscle end)
    c = np.arange(a + D / 2, b - D / 2 + 1e-12, STEP)
    return np.column_stack([c - D / 2, c + D / 2])


def main():
    ph = Phantom(DEFAULT_LAYERS, 2e9)
    lam = C0 / 2e9 / np.sqrt(53)
    W = windows(ph)
    mass = np.array([R.window_mass(ph, *w) for w in W])
    ref = np.array([R.exact_window_output(ph, *w) for w in W]) / mass
    k_ref = int(np.argmax(ref))
    # inf-sup lower bound (H1, enclosure of rev_bracket1d.py)
    gmin = R.gamma_minus(ph, interface_mesh(ph, lam / 512))
    rows, bands = [], {}
    for fr in [16, 32, 64, 128, 256]:
        x = interface_mesh(ph, lam / fr)
        solved = R.primal(ph, x)
        fine = R.primal_fine(ph, x, 16)
        lo = np.empty(len(W)); hi = np.empty(len(W)); mid = np.empty(len(W))
        for i, w in enumerate(W):
            r = R.bracket(ph, x, gmin, window=tuple(w), solved=solved, fine=fine)
            lo[i], hi[i], mid[i] = r["lo"] / mass[i], r["hi"] / mass[i], r["J_h"] / mass[i]
        all_in = bool(np.all((lo <= ref) & (ref <= hi)))
        plo, phi = lo.max(), hi.max()
        rows.append(dict(frac=fr, n=len(x) - 1, h_mm=float(np.diff(x).max() * 1e3),
                         peak_lo=plo, peak_hi=phi, peak_h=mid.max(),
                         peak_contains=bool(plo <= ref.max() <= phi), all_windows_contain=all_in,
                         width_pct=(phi - plo) / ref.max() * 100,
                         argmax_lo=int(np.argmax(lo)), argmax_hi=int(np.argmax(hi))))
        bands[fr] = (lo, hi, mid)
        print("frac %4d  peak in [%.5e, %.5e]  ref %.5e  in=%s all_windows=%s width %.3f%%" %
              (fr, plo, phi, ref.max(), rows[-1]["peak_contains"], all_in, rows[-1]["width_pct"]))

    # decision rule: SAR scales with E0^2 = 2 eta0 S (S incident power density, W m^-2)
    Sstar = LIMIT / (2 * ETA0 * ref.max())          # exposure at which the true peak = limit
    dec = []
    for r in rows:
        S_pass = LIMIT / (2 * ETA0 * r["peak_hi"])  # below: certified PASS
        S_fail = LIMIT / (2 * ETA0 * r["peak_lo"])  # above: certified FAIL
        dec.append(dict(frac=r["frac"], h_mm=r["h_mm"], S_pass=S_pass, S_fail=S_fail,
                        band_lo_rel=S_pass / Sstar, band_hi_rel=S_fail / Sstar))
    scen = {}
    for rel in [0.90, 0.95, 0.99, 1.01, 1.05, 1.10]:
        S = rel * Sstar
        out = []
        for r in rows:
            lo_, hi_ = 2 * ETA0 * S * r["peak_lo"], 2 * ETA0 * S * r["peak_hi"]
            out.append("PASS" if hi_ <= LIMIT else ("FAIL" if lo_ > LIMIT else "INDETERMINATE"))
        scen[f"{rel:.2f}"] = dict(S=S, true_peak=2 * ETA0 * S * ref.max(), decisions=out)
        print("S = %.3f S* (%.1f W/m2, true peak %.3f W/kg): %s" %
              (rel, S, 2 * ETA0 * S * ref.max(), out))
    json.dump(dict(D=D, step=STEP, n_windows=len(W), ref_peak=ref.max(),
                   ref_peak_centre_cm=float(W[k_ref].mean() * 100), gamma_minus=gmin,
                   Sstar=Sstar, rows=rows, decision_band=dec, scenarios=scen),
              open(f"{RES}/peak10g.json", "w"), indent=1)

    # ---- figure: (a) windowed SAR band at lambda/32 + reference ; (b) indeterminate band
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    depth = (W.mean(axis=1) - ph.edges[1]) * 100      # centre depth below skin surface
    scale = 2 * ETA0 * Sstar                          # normalise so that true peak = 2 W/kg
    lo, hi, mid = bands[32]
    ax[0].fill_between(depth, lo * scale, hi * scale, color="C1", alpha=0.35,
                       label=r"bracket, $h=\lambda_{\rm m}/32$")
    ax[0].plot(depth, ref * scale, "k-", lw=1.1, label="reference (transfer-matrix)")
    lo2, hi2, _ = bands[128]
    ax[0].fill_between(depth, lo2 * scale, hi2 * scale, color="C0", alpha=0.5,
                       label=r"bracket, $h=\lambda_{\rm m}/128$")
    for yb in ph.edges[2:-1]:
        for s in (-1, 1):
            dd = (yb - ph.edges[1] + s * D / 2) * 100
            if depth.min() <= dd <= depth.max():
                ax[0].axvline(dd, color="0.5", ls=":", lw=0.6)
    ax[0].axhline(LIMIT, color="C3", lw=0.8, ls="--")
    ax[0].set_xlabel("depth of the averaging-cube centre below the skin (cm)")
    ax[0].set_ylabel(r"SAR$_{10\,\rm g}$ (W kg$^{-1}$)")
    ax[0].set_title(r"(a) 10 g-averaged SAR at $S=S^\ast$")
    ax[0].legend(loc="upper right")
    h = np.array([d["h_mm"] for d in dec])
    ax[1].fill_between(h, [d["band_lo_rel"] for d in dec], [d["band_hi_rel"] for d in dec],
                       color="0.7", label="indeterminate")
    ax[1].axhline(1.0, color="k", lw=0.8)
    ax[1].set_xscale("log")
    ax[1].text(h[0] * 0.9, 0.93, "PASS", ha="right", fontsize=8)
    ax[1].text(h[0] * 0.9, 1.05, "FAIL", ha="right", fontsize=8)
    ax[1].set_ylim(0.85, 1.15)
    ax[1].set_xlabel("element size $h$ (mm)")
    ax[1].set_ylabel(r"exposure level $S/S^\ast$")
    ax[1].set_title("(b) decision regions against 2 W/kg")
    ax[1].legend(loc="upper left")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_peak10g.png")


if __name__ == "__main__":
    main()
