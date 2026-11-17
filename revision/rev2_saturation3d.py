"""Revision R2: saturation indicator (H2) in 3D (skin/fat/muscle stack): the
hierarchical dual norms for rf = 2, 3, 4 on the coarsest level and rf = 2, 3 on
the second level. Output: results/saturation3d.json"""
import json
import time
import numpy as np
from rev_common import RES
from certify3d_layered import certify_layered

np.seterr(all="ignore")
rows = []
for N, rfs in ((5, (2, 3, 4)), (9, (2, 3))):
    rec = dict(N=N, rf=list(rfs), rho_p=[], rho_d=[], edges_fine=[], seconds=[])
    for rf in rfs:
        t = time.time()
        r = certify_layered(N, rf=rf)
        rec["rho_p"].append(float(r["rho_p"])); rec["rho_d"].append(float(r["rho_d"]))
        rec["edges_fine"].append(int(r["n_edges_fine"])); rec["seconds"].append(time.time() - t)
        print(N, rf, r["n_edges_fine"], "%.4e %.4e %.0fs" % (r["rho_p"], r["rho_d"], time.time() - t), flush=True)
    rows.append(rec)
    json.dump(rows, open(f"{RES}/saturation3d.json", "w"), indent=1)
