"""Revision R1: 2D bracket (TE edge elements) against the manufactured lossy plane
wave, with the corrected first-order factor 2 and the bounded remainder term.
Outputs: results/certify2d.json, figures/fig_certify2d.png (about 20 min)."""
import json
import time
import numpy as np
from rev_common import FIG, RES, plt
from certify2d import certify_2d, C0

np.seterr(all="ignore")
freq, eps_r, sigma = 1e9, 40.0, 0.6
lam = C0 / (freq * np.sqrt(eps_r)); L = 1.5 * lam
rows = []
for Nc in [9, 13, 19, 27, 37]:
    t = time.time()
    r = certify_2d(freq, eps_r, sigma, L, Nc)
    r = {k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v)) for k, v in r.items()}
    r.update(N=Nc, h_mm=L / (Nc - 1) * 1e3, width_pct=(r["P_hi"] - r["P_lo"]) / r["P_exact"] * 100,
             seconds=time.time() - t)
    rows.append(r)
    print("N=%d h=%.2f mm in=%s width=%.1f%% beta=%.3f" % (Nc, r["h_mm"], r["contains"], r["width_pct"], r["beta"]), flush=True)
json.dump(dict(freq=freq, eps_r=eps_r, sigma=sigma, L=L, rows=rows), open(f"{RES}/certify2d.json", "w"), indent=1)
h = np.array([r["h_mm"] for r in rows])
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
ax[0].loglog(h, [r["width_pct"] for r in rows], "o-", color="C1", label="bracket width")
ax[0].loglog(h, rows[-1]["width_pct"] * (h / h[-1]) ** 2, "k--", lw=0.8, label=r"$\propto h^2$")
ax[0].set_xlabel("element size $h$ (mm)"); ax[0].set_ylabel("relative width (%)")
ax[0].set_title("(a) 2D bracket width"); ax[0].legend(); ax[0].grid(True, which="both", lw=0.3)
pe = rows[0]["P_exact"]
ax[1].fill_between(h, [r["P_lo"] / pe for r in rows], [r["P_hi"] / pe for r in rows], color="C1", alpha=0.35, label="bracket")
ax[1].plot(h, [r["P_h"] / pe for r in rows], "o-", color="C0", ms=3, label=r"$P_h$")
ax[1].axhline(1.0, color="k", lw=0.8, label="exact")
ax[1].set_xscale("log"); ax[1].set_xlabel("element size $h$ (mm)"); ax[1].set_ylabel(r"$P/P_{\rm exact}$")
ax[1].set_title("(b) bracket and exact absorbed power"); ax[1].legend()
fig.tight_layout(); fig.savefig(f"{FIG}/fig_certify2d.png")
