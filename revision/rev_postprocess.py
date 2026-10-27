"""Revision R1: apply the two-sided inf-sup rule (H1) to the 2D and 3D brackets.
beta_h was computed on each coarse space; it DEcreases slightly under refinement
in 2D and 3D, so the coarse value is not a lower bound. Every bracket is
re-evaluated with gamma_- = min over all computed levels of beta_h (the
increments are already below 1e-3 relative between the last two levels).
lin scales as 1/beta and the remainder bound as 1/beta^2, so the rescaling is exact.
Inputs: results/certify2d.json, ~/Desktop/code_propagation/fullwave_sar_poc/
certify3d_layered_results.json. Output: results/brackets_2d3d.json"""
import json
import os
from rev_common import RES, CODE


def rescale(rows, Pex_key="P_exact", Ph_key="P_h"):
    gmin = min(r["beta"] for r in rows)
    out = []
    for r in rows:
        s = r["beta"] / gmin
        lin, quad = r["lin"] * s, r["quad"] * s * s
        Ph, Pex = r[Ph_key], r[Pex_key]
        lo, hi = Ph - lin, Ph + lin + quad
        out.append(dict(r, gamma_minus=gmin, lin_r=lin, quad_r=quad, lo_r=lo, hi_r=hi,
                        contains_r=bool(lo <= Pex <= hi), width_r_pct=(hi - lo) / Pex * 100))
    return out, gmin


def main():
    res = {}
    d2 = json.load(open(f"{RES}/certify2d.json"))
    if "P_h" in d2["rows"][0]:
        res["2d"], g2 = rescale(d2["rows"])
        res["2d_gamma_minus"] = g2
    f3 = os.path.join(CODE, "fullwave_sar_poc", "certify3d_layered_results.json")
    if not os.path.exists(f3):   # run still in progress: levels parsed from the log
        f3 = f3.replace("_results.json", "_results_partial.json")
    if os.path.exists(f3):
        res["3d"], g3 = rescale(json.load(open(f3)))
        res["3d_gamma_minus"] = g3
    json.dump(res, open(f"{RES}/brackets_2d3d.json", "w"), indent=1)
    for k in ("2d", "3d"):
        for r in res.get(k, []):
            print(k, r["N"], "width %.1f%% (coarse beta: %.1f%%) contains %s" %
                  (r["width_r_pct"], (r.get("hi", r.get("P_hi")) - r.get("lo", r.get("P_lo"))) / r["P_exact"] * 100, r["contains_r"]))


if __name__ == "__main__":
    main()
