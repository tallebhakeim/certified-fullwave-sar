"""Revision R2 (referee 1, point 4): physical clipping of all intervals on the
absorbed power.

Lower end: P_abs >= 0 because sigma >= 0, so lo <- max(0, lo).
Upper end, scattering configurations only: by the power balance (12) the absorbed
power equals incident minus reflected minus transmitted power, hence
P_abs <= P_inc. With an incident amplitude of 1 V/m:
  1D (per unit area)           P_inc = 1 / (2 eta0)            W m^-2
  3D stack (cross-section L^2) P_inc = L^2 / (2 eta0),  L = 2 cm
(the 3D field (10) is the exact field of the stack exposed from air, so the
stack absorbs at most the power incident on its cross-section).
The 2D manufactured case is driven by Dirichlet data and has no incident wave:
only the lower clip applies.
Inputs: results/bracket1d.json, results/brackets_2d3d.json
Output: results/clipped.json"""
import json
from rev_common import RES, ETA0

P1D = 1.0 / (2 * ETA0)
L3 = 0.02
P3D = L3 * L3 / (2 * ETA0)


def clip(lo, hi, pex, pinc=None):
    clo = max(0.0, lo)
    chi = min(hi, pinc) if pinc is not None else hi
    return dict(lo=lo, hi=hi, lo_c=clo, hi_c=chi,
                width_pct=(hi - lo) / pex * 100, width_c_pct=(chi - clo) / pex * 100,
                lo_active=lo < 0, hi_active=pinc is not None and hi > pinc,
                contains_c=bool(clo <= pex <= chi))


def main():
    out = dict(P_inc_1d=P1D, P_inc_3d=P3D)
    b1 = json.load(open(f"{RES}/bracket1d.json"))
    pex = b1["convergence"]["P_exact"]
    out["1d"] = [dict(frac=r["frac"], h_mm=r["h_mm"], P_exact=pex, **clip(r["lo"], r["hi"], pex, P1D))
                 for r in b1["convergence"]["rows"]]
    out["1d_resonance"] = [dict(f=o["f"], P_exact=o["P_exact"],
                                **clip(o["lo"], o["hi"], o["P_exact"], P1D)) for o in b1["resonance"]]
    b = json.load(open(f"{RES}/brackets_2d3d.json"))
    out["2d"] = [dict(N=r["N"], h_mm=r["h_mm"], beta=r["beta"], P_h=r["P_h"], P_exact=r["P_exact"],
                      **clip(r["lo_r"], r["hi_r"], r["P_exact"])) for r in b["2d"]]
    out["3d"] = [dict(N=r["N"], h_mm=r["h"] * 1e3 if "h" in r else r["h_mm"], beta=r["beta"],
                      n_edges=r.get("n_edges"), n_edges_fine=r.get("n_edges_fine"),
                      P_h=r["P_h"], P_exact=r["P_exact"],
                      **clip(r["lo_r"], r["hi_r"], r["P_exact"], P3D)) for r in b["3d"]]
    json.dump(out, open(f"{RES}/clipped.json", "w"), indent=1)
    print("P_inc 1D %.4e W/m2, 3D %.4e W" % (P1D, P3D))
    for k in ("1d", "2d", "3d"):
        for r in out[k]:
            print(k, "h=%.2f mm [%.4e, %.4e] -> [%.4e, %.4e]  width %.1f%% -> %.1f%%  in=%s" %
                  (r["h_mm"], r["lo"], r["hi"], r["lo_c"], r["hi_c"], r["width_pct"], r["width_c_pct"], r["contains_c"]))
    rs = out["1d_resonance"]
    print("resonance: lo clipped at %d/90, hi clipped at %d/90, all contain %s, max width %.0f%% -> %.0f%%" %
          (sum(r["lo_active"] for r in rs), sum(r["hi_active"] for r in rs), all(r["contains_c"] for r in rs),
           max(r["width_pct"] for r in rs), max(r["width_c_pct"] for r in rs)))


if __name__ == "__main__":
    main()
