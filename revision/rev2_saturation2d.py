"""Revision R2 (referee 1, point 3): numerical indicator for the saturation
hypothesis (H2) in 2D.

The dual norms of the primal and adjoint residuals are evaluated through their
Riesz representatives on the coarse mesh refined rf times. The hierarchical
value increases with rf towards the true dual norm (nested spaces). (H2) states
that the value used (rf = 3) is not below the true dual norm by more than the
slack in the bound. This script reports rho(rf) / rho(rf_max) for rf = 2, 3, 4, 6:
a ratio close to 1 that stabilises with rf is consistent with (H2); it is an
indicator, not a proof.
Output: results/saturation2d.json"""
import json
import time
import numpy as np
from rev_common import RES
from certify2d import certify_2d, C0

np.seterr(all="ignore")
freq, eps_r, sigma = 1e9, 40.0, 0.6
lam = C0 / (freq * np.sqrt(eps_r)); L = 1.5 * lam
RFS = [2, 3, 4, 6]
rows = []
for Nc in [9, 13, 19, 27, 37]:
    rec = dict(N=Nc, h_mm=L / (Nc - 1) * 1e3, rf=RFS, rho_p=[], rho_d=[], seconds=[])
    for rf in RFS:
        t = time.time()
        r = certify_2d(freq, eps_r, sigma, L, Nc, rf=rf)
        rec["rho_p"].append(float(r["rho_p"])); rec["rho_d"].append(float(r["rho_d"]))
        rec["seconds"].append(time.time() - t)
    rp, rd = np.array(rec["rho_p"]), np.array(rec["rho_d"])
    rec["ratio_p"] = (rp / rp[-1]).tolist(); rec["ratio_d"] = (rd / rd[-1]).tolist()
    rows.append(rec)
    print("N=%d h=%.2f mm  rho_p/rho_p(6): %s  rho_d/rho_d(6): %s" %
          (Nc, rec["h_mm"], np.round(rec["ratio_p"], 4), np.round(rec["ratio_d"], 4)), flush=True)
    json.dump(dict(rfs=RFS, rows=rows), open(f"{RES}/saturation2d.json", "w"), indent=1)
