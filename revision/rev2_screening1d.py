"""Revision R2 (referee 1, point 6): screening of the averaging windows so that
adjoint problems are solved only for candidate windows.

Step 1 (no adjoint): for every window w, a first-order interval follows from one
primal solve and its refined solution. With ||.||_{sigma,w} the sigma-weighted L2
seminorm on w, sqrt(2 J_w(E)) = ||E||_{sigma,w}, so by the triangle inequality
    | sqrt(2 J_w(E)) - sqrt(2 J_w(E_h)) | <= sqrt(sigma_max,w) * l2_w,
    l2_w = ||E_f - E_h||_{L2(w)} + eta_f / (gamma_- k0)   (>= ||E - E_h||_{L2(w)}),
which gives crude bounds lo_w^c <= J_w(E) <= hi_w^c for all windows at once.
Step 2: windows are visited in decreasing order of the computed value J_w(E_h);
the current best lower bound L is the maximum of the lower bounds obtained so far.
A window with hi_w^c < L cannot contain the maximum and is discarded; otherwise its
goal-oriented (adjoint) interval is computed and L is updated.
Every window keeps a valid interval (crude or adjoint), so Lemma 1 applies.
Output: results/screening1d.json"""
import json
import time
import numpy as np
from rev_common import RES
import certify_rigorous as R
from helmholtz1d import Phantom, interface_mesh, C0, DEFAULT_LAYERS
from rev_peak10g import windows

np.seterr(all="ignore")


def main():
    ph = Phantom(DEFAULT_LAYERS, 2e9)
    lam = C0 / 2e9 / np.sqrt(53)
    W = windows(ph)
    mass = np.array([R.window_mass(ph, *w) for w in W])
    ref = np.array([R.exact_window_output(ph, *w) for w in W]) / mass
    gmin = R.gamma_minus(ph, interface_mesh(ph, lam / 512))
    w2 = R.norm_weight(ph)
    rows = []
    for fr in [16, 32, 64, 128, 256]:
        x = interface_mesh(ph, lam / fr)
        t0 = time.time()
        solved = R.primal(ph, x)
        fine = R.primal_fine(ph, x, 16)
        _, Eh, _ = solved
        _, sig, _ = R.elem_data(ph, x)
        xf, Ef, eta_f = fine
        eh_f = np.interp(xf, x, Eh.real) + 1j * np.interp(xf, x, Eh.imag)
        _, sigf, _ = R.elem_data(ph, xf)
        ind = np.where(sigf > 0, 1.0, 0.0)
        Jh = np.empty(len(W)); clo = np.empty(len(W)); chi = np.empty(len(W))
        for i, (y0, y1) in enumerate(W):
            Jh[i] = R.window_output(x, Eh, sig, y0, y1)
            inside = (x[1:] > y0) & (x[:-1] < y1)
            smax = sig[inside].max()
            d2 = 2.0 * R.window_output(xf, Ef - eh_f, ind, y0, y1)
            l2 = np.sqrt(d2) + eta_f / (gmin * np.sqrt(w2))
            a = np.sqrt(2 * Jh[i]); b = np.sqrt(smax) * l2
            chi[i] = 0.5 * (a + b) ** 2
            clo[i] = 0.5 * max(0.0, a - b) ** 2
        t_crude = time.time() - t0
        crude_ok = bool(np.all((clo / mass <= ref) & (ref <= chi / mass)))
        # screening
        t1 = time.time()
        Lbest = (clo / mass).max()
        lo = clo / mass; hi = chi / mass
        computed = []
        for i in np.argsort(-Jh / mass):
            if chi[i] / mass[i] < Lbest:
                continue
            r = R.bracket(ph, x, gmin, window=tuple(W[i]), solved=solved, fine=fine)
            lo[i] = max(lo[i], r["lo"] / mass[i]); hi[i] = min(hi[i], r["hi"] / mass[i])
            Lbest = max(Lbest, lo[i])
            computed.append(int(i))
        t_screen = time.time() - t1
        all_ok = bool(np.all((lo <= ref) & (ref <= hi)))
        rows.append(dict(frac=fr, n_windows=len(W), n_adjoint=len(computed),
                         crude_peak_lo=float((clo / mass).max()), crude_peak_hi=float((chi / mass).max()),
                         crude_width_pct=float(((chi / mass).max() - (clo / mass).max()) / ref.max() * 100),
                         crude_all_contain=crude_ok,
                         peak_lo=float(lo.max()), peak_hi=float(hi.max()),
                         width_pct=float((hi.max() - lo.max()) / ref.max() * 100),
                         peak_contains=bool(lo.max() <= ref.max() <= hi.max()),
                         all_windows_contain=all_ok))
        print("frac %3d: crude width %.1f%% (contain %s); adjoints %d/%d; peak width %.3f%% contain %s" %
              (fr, rows[-1]["crude_width_pct"], crude_ok, len(computed), len(W),
               rows[-1]["width_pct"], rows[-1]["peak_contains"]), flush=True)
    json.dump(dict(ref_peak=float(ref.max()), gamma_minus=gmin, rows=rows),
              open(f"{RES}/screening1d.json", "w"), indent=1)


if __name__ == "__main__":
    main()
