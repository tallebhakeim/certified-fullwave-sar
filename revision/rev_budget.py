"""Revision R1 (Reviewer 2, uncertainty budget): sensitivity of the reference
psSAR10g of the 2 GHz layered phantom to the tissue parameters and to the
averaging-volume definition, computed with the exact transfer-matrix solution,
for comparison with the discretization bracket.
Output: results/budget.json"""
import json
import numpy as np
from scipy.optimize import brentq
from rev_common import RES
import certify_rigorous as R
from helmholtz1d import Phantom, DEFAULT_LAYERS, TISSUE

D, STEP = 0.0215, 0.0005
BASE = dict(TISSUE)


def pssar(ph, side=None):
    a, b = ph.edges[1], ph.edges[-2]
    best = 0.0
    for c in np.arange(a + D / 2, b - D / 2 + 1e-12, STEP):
        if side is None:
            l = D
        else:   # mass-adjusted cube: side l with l^2 * int_window rho = 10 g
            f = lambda l: l * l * R.window_mass(ph, max(c - l / 2, a), min(c + l / 2, b)) - 0.010
            l = brentq(f, 0.5 * D, 1.5 * D)
        y0, y1 = max(c - l / 2, a), min(c + l / 2, b)
        best = max(best, R.exact_window_output(ph, y0, y1) / R.window_mass(ph, y0, y1))
    return best


def main():
    ph = Phantom(DEFAULT_LAYERS, 2e9)
    p0 = pssar(ph)
    out = dict(reference=p0, cases={})
    for name, idx, rel in [("eps_r", 0, 0.10), ("sigma", 1, 0.10), ("rho", 2, 0.05)]:
        for sgn in (-1, 1):
            for t in ("skin", "fat", "muscle"):
                v = list(BASE[t]); v[idx] *= 1 + sgn * rel; TISSUE[t] = tuple(v)
            p = pssar(Phantom(DEFAULT_LAYERS, 2e9))
            TISSUE.update(BASE)
            out["cases"][f"{name} {'+' if sgn > 0 else '-'}{int(rel*100)}%"] = (p - p0) / p0
    out["cases"]["mass-adjusted 10 g cube"] = (pssar(ph, side=True) - p0) / p0
    json.dump(out, open(f"{RES}/budget.json", "w"), indent=1)
    for k, v in out["cases"].items():
        print("%-26s %+.2f %%" % (k, v * 100))


if __name__ == "__main__":
    main()
