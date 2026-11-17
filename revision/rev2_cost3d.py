"""Revision R2 (referee 1, points 5 and 6): cost of the 3D conditional interval
and order-of-magnitude cost for a voxel phantom.

Measured: the three 3D levels of results/certify3d_layered.log (2 cm cube,
coarse and refined edge counts, wall-clock time of one complete interval:
primal, adjoint, Riesz representatives on the refined space, inf-sup constant).
Fitted: width ~ h^p (two finest levels, unclipped widths) and time ~ n_fine^a
(two finest levels). Extrapolated: element size, edge counts and time at which
the relative width reaches 10 % and 2.5 % (in 1D a width of about 2.6 % allows
a decision for exposures 5 % away from the critical level, table 4).
Voxel phantom: tissue voxels of the XCAT head volume (1 mm) and edge counts of
the Freudenthal-Kuhn tetrahedral mesh (about 7 edges per voxel) at 1 and 2 mm.
Output: results/cost3d.json"""
import json
import os
import re
import numpy as np
from rev_common import RES

XCAT = os.path.join(os.environ.get("XCAT_DIR", "data"), "VoxHead_338x248x256.i33")


def edges_kuhn(n):
    """Edges of the 6-tetrahedra-per-cube mesh of an n x n x n grid of cubes."""
    m = n + 1
    axis = 3 * n * m * m
    face = 3 * n * n * m
    body = n ** 3
    return axis + face + body


def main():
    log = open(f"{RES}/certify3d_layered.log").read()
    lev = [dict(N=int(a), h_mm=float(b), edges=int(c), edges_fine=int(d), seconds=float(e))
           for a, b, c, d, e in re.findall(r"N=\s*(\d+) h=\s*([\d.]+)mm edges=\s*(\d+)/\s*(\d+).*?\((\d+)s\)", log)]
    cl = json.load(open(f"{RES}/clipped.json"))["3d"]
    for L_, c in zip(lev, cl):
        L_.update(width_pct=c["width_pct"], width_c_pct=c["width_c_pct"])
        assert edges_kuhn(L_["N"] - 1) == L_["edges"], (L_, edges_kuhn(L_["N"] - 1))
    a, b = lev[-2], lev[-1]
    p = np.log(a["width_pct"] / b["width_pct"]) / np.log(a["h_mm"] / b["h_mm"])
    t_exp = np.log(b["seconds"] / a["seconds"]) / np.log(b["edges_fine"] / a["edges_fine"])
    extra = []
    for target in (10.0, 2.5):
        h = b["h_mm"] * (target / b["width_pct"]) ** (1 / p)
        n = int(np.ceil(20.0 / h))
        ec, ef = edges_kuhn(n), edges_kuhn(2 * n)
        t = b["seconds"] * (ef / b["edges_fine"]) ** t_exp
        extra.append(dict(target_pct=target, h_mm=h, edges=ec, edges_fine=ef, seconds_direct=t))
    box = (338, 248, 256)
    try:
        tissue_1mm = int(np.count_nonzero(np.fromfile(XCAT, dtype="<u2")))
    except FileNotFoundError:   # licensed data absent: keep the published count
        tissue_1mm = json.load(open(f"{RES}/cost3d.json"))["xcat_head"]["tissue_voxels_1mm"]
    xcat = dict(tissue_voxels_1mm=tissue_1mm, tissue_voxels_2mm=int(round(tissue_1mm / 8)),
                box_voxels_1mm=int(np.prod(box)),
                edges_box_1mm=int(7 * np.prod(box)), edges_box_2mm=int(7 * np.prod(box) / 8))
    out = dict(levels=lev, rate_width=p, time_exponent=t_exp, extrapolation=extra, xcat_head=xcat)
    json.dump(out, open(f"{RES}/cost3d.json", "w"), indent=1)
    for L_ in lev:
        print(L_)
    print("width rate p = %.2f ; time ~ n_fine^%.2f" % (p, t_exp))
    for e in extra:
        print("width %.1f%%: h = %.3f mm, %d coarse / %d refined edges (2 cm cube), direct time ~ %.1e s" %
              (e["target_pct"], e["h_mm"], e["edges"], e["edges_fine"], e["seconds_direct"]))
    print(xcat)


if __name__ == "__main__":
    main()
