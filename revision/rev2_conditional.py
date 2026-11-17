"""Revision R2: summary of the 2D and 3D conditional intervals.

For each level: interval as computed (hierarchical dual norms, rf = 3 in 2D and
rf = 2 in 3D), physically clipped (rev2_clip.py), and the same interval with the
dual norms replaced by their extrapolation in rf. The hierarchical values rho(rf)
increase with rf (nested spaces); they are fitted by rho(rf) = rho_inf - C rf^(-s)
(s from three rf values, rev2_saturation2d.py / rev2_saturation3d.py), and the
ratio rho(rf_used) / rho_inf measures how far the computed value is from the
saturated one. The first-order term scales as rho_p rho_d and the remainder bound
as rho_p^2, so both are divided by the corresponding ratios. The extrapolation is
itself an estimate: it quantifies (H2), it does not remove it.
Also draws figure 7. Output: results/conditional.json, figures/fig_certify2d.png"""
import json
import numpy as np
from scipy.optimize import brentq
from rev_common import RES, FIG, plt
from rev2_clip import P3D

np.seterr(all="ignore")


def fit_s(rfs, rho):
    (r1, r2, r3), (a, b, c) = rfs[-3:], rho[-3:]
    g = lambda s: (c - b) / (b - a) - (r2 ** -s - r3 ** -s) / (r1 ** -s - r2 ** -s)
    return brentq(g, 0.05, 20.0)


def rho_inf(rfs, rho, s):
    a, b = rho[-2], rho[-1]
    C = (b - a) / (rfs[-2] ** -s - rfs[-1] ** -s)
    return b + C * rfs[-1] ** -s


def corrected(r, fp, fd, pinc=None):
    lin, quad = r["lin_r"] / (fp * fd), r["quad_r"] / fp ** 2
    lo, hi = r["P_h"] - lin, r["P_h"] + lin + quad
    clo, chi = max(0.0, lo), (min(hi, pinc) if pinc else hi)
    return dict(ratio_p=fp, ratio_d=fd, lo_x=lo, hi_x=hi, lo_xc=clo, hi_xc=chi,
                width_xc_pct=(chi - clo) / r["P_exact"] * 100, contains_xc=bool(clo <= r["P_exact"] <= chi))


def main():
    b = json.load(open(f"{RES}/brackets_2d3d.json"))
    cl = json.load(open(f"{RES}/clipped.json"))
    out = {"2d": [], "3d": []}
    s2 = json.load(open(f"{RES}/saturation2d.json"))
    rfs2 = np.array(s2["rfs"], float); iu2 = list(s2["rfs"]).index(3)
    for r, c, s in zip(b["2d"], cl["2d"], s2["rows"]):
        fp_ = []
        for key in ("rho_p", "rho_d"):
            rho = np.array(s[key]); se = fit_s(rfs2, rho)
            fp_.append(rho[iu2] / rho_inf(rfs2, rho, se))
        out["2d"].append(dict(N=r["N"], h_mm=r["h_mm"], beta=r["beta"], P_exact=r["P_exact"], P_h=r["P_h"],
                              lo_c=c["lo_c"], hi_c=c["hi_c"], width_pct=c["width_pct"], width_c_pct=c["width_c_pct"],
                              **corrected(r, *fp_)))
    try:
        s3 = json.load(open(f"{RES}/saturation3d.json"))
    except FileNotFoundError:
        s3 = []
    ratios3 = []
    if s3 and len(s3[0]["rf"]) >= 3:
        rfs = np.array(s3[0]["rf"], float)
        sp_ = fit_s(rfs, np.array(s3[0]["rho_p"])); sd_ = fit_s(rfs, np.array(s3[0]["rho_d"]))
        for rec in s3:
            rf = np.array(rec["rf"], float)
            if len(rf) < 2:
                continue
            fp = rec["rho_p"][0] / rho_inf(rf, np.array(rec["rho_p"]), sp_)
            fd = rec["rho_d"][0] / rho_inf(rf, np.array(rec["rho_d"]), sd_)
            ratios3.append((rec["N"], fp, fd, sp_, sd_))
    for r, c in zip(b["3d"], cl["3d"]):
        avail = [x for x in ratios3 if x[0] == r["N"]] or ratios3[-1:]
        row = dict(N=r["N"], h_mm=r["h"] * 1e3, beta=r["beta"], n_edges=r["n_edges"], P_exact=r["P_exact"],
                   P_h=r["P_h"], lo_c=c["lo_c"], hi_c=c["hi_c"], width_pct=c["width_pct"], width_c_pct=c["width_c_pct"])
        if avail:
            row.update(corrected(r, avail[0][1], avail[0][2], P3D), ratio_from_level=avail[0][0])
        out["3d"].append(row)
    out["ratios3d"] = ratios3
    json.dump(out, open(f"{RES}/conditional.json", "w"), indent=1)
    for k in ("2d", "3d"):
        for r in out[k]:
            print(k, "h=%.2f width %.0f%% -> clipped %.0f%% -> extrapolated norms %s  ratio_p %s" %
                  (r["h_mm"], r["width_pct"], r["width_c_pct"],
                   "%.0f%% (in=%s)" % (r["width_xc_pct"], r["contains_xc"]) if "width_xc_pct" in r else "n/a",
                   "%.3f" % r["ratio_p"] if "ratio_p" in r else "n/a"))
    # ---- figure 7
    rows = out["2d"]
    h = np.array([r["h_mm"] for r in rows]); pe = rows[0]["P_exact"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax[0].loglog(h, [r["width_c_pct"] for r in rows], "o-", color="C1", label="conditional interval")
    ax[0].loglog(h, [r["width_xc_pct"] for r in rows], "s--", color="C3", ms=3,
                 label="with extrapolated dual norms")
    w_last = rows[-1]["width_c_pct"]
    ax[0].loglog(h, w_last * (h / h[-1]) ** 2, "k:", lw=0.8, label=r"$\propto h^2$")
    ax[0].set_xlabel("element size $h$ (mm)"); ax[0].set_ylabel("relative width (%)")
    ax[0].set_title("(a) width of the 2D conditional interval"); ax[0].legend(); ax[0].grid(True, which="both", lw=0.3)
    ax[1].fill_between(h, [r["lo_c"] / pe for r in rows], [r["hi_c"] / pe for r in rows], color="C1", alpha=0.35,
                       label="conditional interval")
    ax[1].plot(h, [r["P_h"] / pe for r in rows], "o-", color="C0", ms=3, label=r"$P_h$")
    ax[1].axhline(1.0, color="k", lw=0.8, label="exact")
    ax[1].set_xscale("log"); ax[1].set_xlabel("element size $h$ (mm)"); ax[1].set_ylabel(r"$P/P_{\rm exact}$")
    ax[1].set_title("(b) interval and exact absorbed power"); ax[1].legend()
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_certify2d.png")


if __name__ == "__main__":
    main()
