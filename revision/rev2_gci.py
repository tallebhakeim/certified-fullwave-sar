"""Revision R2 (referee 1, point 7): comparison with the discretization
uncertainty obtained from a grid-convergence study.

Grid convergence index (GCI) of Roache, three-grid procedure with safety factor
Fs = 1.25 and the iteration of Celik et al (2008) for non-constant refinement
ratios. phi1 (finest), phi2, phi3 on grids h1 < h2 < h3:
  eps21 = phi2 - phi1, eps32 = phi3 - phi2, R = eps21 / eps32 (convergence ratio),
  monotone convergence if 0 < R < 1, oscillatory if R < 0, divergent if R > 1;
  p from |ln|eps32/eps21| + ln((r21^p - s)/(r32^p - s))| / ln r21, s = sign(eps32/eps21);
  GCI = Fs |eps21 / phi1| / (r21^p - 1).
The estimate "covers" the exact value if |phi1 - exact| <= GCI |phi1|.
For oscillatory or divergent triplets the GCI procedure does not apply and is
reported as such.
Cases: 1D absorbed power and psSAR10g (all consecutive triplets), 1D low-loss
slab sweep (90 frequencies, grids lambda_s/5, /10, /20, the finest being the
grid on which the bracket of figure 4(b) is computed), 2D and 3D sequences.
Output: results/gci.json"""
import json
import numpy as np
from rev_common import RES
import certify_rigorous as R
from helmholtz1d import Phantom, interface_mesh, C0, TISSUE

FS = 1.25


def gci(phi1, phi2, phi3, h1, h2, h3):
    r21, r32 = h2 / h1, h3 / h2
    e21, e32 = phi2 - phi1, phi3 - phi2
    Rr = e21 / e32 if e32 != 0 else np.inf
    kind = "monotone" if 0 < Rr < 1 else ("oscillatory" if Rr < 0 else "divergent")
    if kind != "monotone":
        return dict(kind=kind, R=float(Rr), p=None, gci=None)
    s = np.sign(e32 / e21)
    p = 2.0
    for _ in range(200):
        q = np.log((r21 ** p - s) / (r32 ** p - s))
        p_new = abs(np.log(abs(e32 / e21)) + q) / np.log(r21)
        if abs(p_new - p) < 1e-12:
            break
        p = p_new
    g = FS * abs(e21 / phi1) / (r21 ** p - 1)
    return dict(kind=kind, R=float(Rr), p=float(p), gci=float(g))


def triplets(phis, hs, exact, bracket=None):
    out = []
    for i in range(len(phis) - 2):
        d = gci(phis[i + 2], phis[i + 1], phis[i], hs[i + 2], hs[i + 1], hs[i])
        err = abs(phis[i + 2] - exact) / abs(exact)
        d.update(h_fine=hs[i + 2], true_rel_err=float(err),
                 covers=(bool(err <= d["gci"] * abs(phis[i + 2]) / abs(exact)) if d["gci"] is not None else None))
        if bracket is not None:
            lo, hi = bracket[i + 2]
            d.update(bracket_half_width_rel=float((hi - lo) / 2 / exact), bracket_contains=bool(lo <= exact <= hi))
        out.append(d)
    return out


def main():
    res = {}
    b1 = json.load(open(f"{RES}/bracket1d.json"))
    c = b1["convergence"]
    res["1d_pabs"] = triplets([r["J_h"] for r in c["rows"]], [r["h_mm"] for r in c["rows"]], c["P_exact"],
                              [(r["lo"], r["hi"]) for r in c["rows"]])
    pk = json.load(open(f"{RES}/peak10g.json"))
    res["1d_pssar"] = triplets([r["peak_h"] for r in pk["rows"]], [r["h_mm"] for r in pk["rows"]], pk["ref_peak"],
                               [(r["peak_lo"], r["peak_hi"]) for r in pk["rows"]])
    # resonance sweep: three-grid GCI ending at lambda_s/20 (the grid of figure 4(b)),
    # lambda_s/40 and lambda_s/80; the bracket is evaluated on the same finest grid
    TISSUE["slab"] = (80.0, 0.03, 1000.0)
    sweeps = {}
    for fine_fr in (20, 40, 80):
        frs = [fine_fr // 4, fine_fr // 2, fine_fr]
        sweep = []
        for o in b1["resonance"]:
            f = o["f"]
            ph = Phantom([("air", 0.01), ("slab", 0.02), ("air", 0.01)], f)
            lam_s = C0 / f / np.sqrt(80.0)
            phis, hs = [], []
            for fr in frs:
                x = interface_mesh(ph, lam_s / fr)
                _, Eh, _ = R.primal(ph, x)
                _, sig, _ = R.elem_data(ph, x)
                phis.append(R.window_output(x, Eh, sig, 0.0, ph.Y)); hs.append(float(np.diff(x).max()))
            if fine_fr == 20:
                br = (o["lo"], o["hi"])
            else:
                rb = R.bracket(ph, x, o["gamma_lo"]); br = (rb["lo"], rb["hi"])
            d = triplets(phis, hs, o["P_exact"], [None, None, br])[0]
            d.update(f=f, gamma_lo=o["gamma_lo"])
            sweep.append(d)
        sweeps[str(fine_fr)] = sweep
    sweep = sweeps["20"]
    res["1d_sweep_levels"] = sweeps
    res["1d_sweep"] = sweep
    b = json.load(open(f"{RES}/brackets_2d3d.json"))
    for k in ("2d", "3d"):
        rows = b[k]
        hs = [r["h_mm"] if "h_mm" in r else r["h"] * 1e3 for r in rows]
        res[k] = triplets([r["P_h"] for r in rows], hs, rows[0]["P_exact"], [(r["lo_r"], r["hi_r"]) for r in rows])
    json.dump(res, open(f"{RES}/gci.json", "w"), indent=1)
    for k in ("1d_pabs", "1d_pssar", "2d", "3d"):
        for d in res[k]:
            print(k, "h=%.3g %s p=%s GCI=%s err=%.4f%% covers=%s half-width=%.3f%%" %
                  (d["h_fine"], d["kind"], None if d["p"] is None else round(d["p"], 2),
                   None if d["gci"] is None else "%.4f%%" % (d["gci"] * 100), d["true_rel_err"] * 100,
                   d["covers"], d["bracket_half_width_rel"] * 100))
    for kf, sw in sweeps.items():
        kinds = [d["kind"] for d in sw]; mono = [d for d in sw if d["kind"] == "monotone"]
        print("sweep fine lambda_s/%s: monotone %d, oscillatory %d, divergent %d; GCI covers %d/%d; GCI fails %d/90; "
              "bracket contains %d/90, median half-width %.2f%%" % (kf, kinds.count("monotone"), kinds.count("oscillatory"),
              kinds.count("divergent"), sum(d["covers"] for d in mono), len(mono), 90 - sum(d["covers"] for d in mono),
              sum(d["bracket_contains"] for d in sw), 100 * np.median([d["bracket_half_width_rel"] for d in sw])))


if __name__ == "__main__":
    main()
