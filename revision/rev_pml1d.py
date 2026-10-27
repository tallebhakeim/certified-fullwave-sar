"""Revision R1 (Reviewer 1, point 3): inf-sup stability of the OPEN problem versus
the PML-TRUNCATED system, and the PML modelling error, in 1D where the open
problem is represented exactly by the Dirichlet-to-Neumann (Robin) conditions.

Open problem   : [0, Y] with exact DtN conditions (certify_rigorous.py), inf-sup gamma.
Truncated      : [-d, Y + d], PML s(xi) = 1 - j tau0 (xi/d)^2 in the two air
                 layers, homogeneous Dirichlet at the outer ends, scattered-field
                 formulation  int (1/s) u'v' - s k^2 u v = int s (k^2 - k0^2) E_inc v,
                 discrete inf-sup beta_PML(d, tau0) in |||v|||^2 = ||v'||^2 + k0^2||v||^2
                 on [-d, Y + d].
tau0 is set from the nominal round-trip reflection R0 = exp(-2 k0 tau0 d / 3).
Outputs: results/pml1d.json, figures/fig_pml1d.png"""
import json
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from rev_common import FIG, RES, plt
import certify_rigorous as R
from helmholtz1d import Phantom, interface_mesh, C0, DEFAULT_LAYERS, TISSUE, k2_of

np.seterr(all="ignore")


def pml_system(ph, h, d, R0):
    k0 = ph.k0
    tau0 = 3.0 * np.log(1.0 / R0) / (2.0 * k0 * d)
    xin = interface_mesh(ph, h)
    m = max(int(np.ceil(d / h)), 1)
    x = np.concatenate([-d + np.arange(m) * d / m, xin, ph.Y + np.arange(1, m + 1) * d / m])
    mid = 0.5 * (x[:-1] + x[1:])
    xi = np.where(mid < 0, -mid, np.where(mid > ph.Y, mid - ph.Y, 0.0))
    s = 1.0 - 1j * tau0 * (xi / d) ** 2
    names = [ph.tissue_at(min(max(t, 0.0), ph.Y - 1e-12)) for t in mid]
    k2 = np.array([k2_of(n, ph.w) for n in names])
    hh = np.diff(x)
    n = len(x)
    kd = (1.0 / s) / hh - s * k2 * hh / 3.0
    ko = -(1.0 / s) / hh - s * k2 * hh / 6.0
    dg = np.zeros(n, dtype=complex); dg[:-1] += kd; dg[1:] += kd
    A = sp.diags([ko, dg, ko], [-1, 0, 1], format="csc")
    # load: s (k^2 - k0^2) E_inc, E_inc = exp(-j k0 y) (s = 1 inside the tissue)
    g, wg = np.polynomial.legendre.leggauss(4)
    b = np.zeros(n, dtype=complex)
    for e in range(n - 1):
        c = (k2[e] - k0 ** 2) * s[e]
        if abs(k2[e] - k0 ** 2) < 1e-12:
            continue
        y = 0.5 * (x[e] + x[e + 1]) + 0.5 * hh[e] * g
        t = (y - x[e]) / hh[e]
        f = c * np.exp(-1j * k0 * y) * wg * 0.5 * hh[e]
        b[e] += np.sum(f * (1 - t)); b[e + 1] += np.sum(f * t)
    inner = slice(1, n - 1)
    return x, A[inner, inner], b[inner], k2


def run_case(ph, h, d, R0, Pex):
    x, A, b, k2 = pml_system(ph, h, d, R0)
    u = np.zeros(len(x), dtype=complex); u[1:-1] = spla.spsolve(A, b)
    E = u + np.exp(-1j * ph.k0 * x)
    msk = (x >= 0) & (x <= ph.Y + 1e-12)
    xs, Es = x[msk], E[msk]
    _, sig, _ = R.elem_data(ph, xs)
    P = R.window_output(xs, Es, sig, 0.0, ph.Y)
    Nm = R.norm_matrix(x, ph.k0 ** 2)[1:-1, 1:-1]
    beta = R.infsup(A, Nm)
    return P, beta


def main():
    ph = Phantom(DEFAULT_LAYERS, 2e9)
    lam = C0 / 2e9 / np.sqrt(53)
    lam0 = C0 / 2e9
    h = lam / 128
    Pex = R.exact_window_output(ph, 0.0, ph.Y)
    # open problem: DtN on the same mesh, and the continuous enclosure
    x = interface_mesh(ph, h)
    k2, _, _ = R.elem_data(ph, x)
    beta_dtn_h = R.infsup(R.assemble(x, k2, 1j * ph.k0, 1j * ph.k0), R.norm_matrix(x, ph.k0 ** 2))
    _, Eh, _ = R.primal(ph, x)
    _, sig, _ = R.elem_data(ph, x)
    P_dtn_h = R.window_output(x, Eh, sig, 0.0, ph.Y)
    out = dict(P_exact=Pex, beta_dtn_h=beta_dtn_h, P_dtn_h=P_dtn_h, h=h, cases=[])
    print("open problem (DtN): beta_h = %.6f ; P_h err %.2e" % (beta_dtn_h, abs(P_dtn_h - Pex) / Pex))
    for dl in [0.125, 0.25, 0.5, 1.0]:
        for R0 in [1e-2, 1e-4, 1e-6, 1e-8, 1e-12]:
            P, beta = run_case(ph, h, dl * lam0, R0, Pex)
            out["cases"].append(dict(d_over_lam0=dl, R0=R0, beta=beta,
                                     P=P, rel_err_vs_exact=abs(P - Pex) / Pex,
                                     rel_dev_vs_dtn=abs(P - P_dtn_h) / P_dtn_h))
            print("d=%.3f lam0 R0=%.0e beta_PML=%.6f  |P-P_DtN|/P=%.2e" %
                  (dl, R0, beta, abs(P - P_dtn_h) / P_dtn_h))
    json.dump(out, open(f"{RES}/pml1d.json", "w"), indent=1)

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for i, dl in enumerate([0.125, 0.25, 0.5, 1.0]):
        cs = [c for c in out["cases"] if c["d_over_lam0"] == dl]
        R0s = [c["R0"] for c in cs]
        ax[0].semilogx(R0s, [c["beta"] for c in cs], "o-", color=f"C{i}",
                       label=fr"$d={dl:g}\,\lambda_0$")
        ax[1].loglog(R0s, [max(c["rel_dev_vs_dtn"], 1e-16) for c in cs], "o-", color=f"C{i}")
    ax[0].axhline(beta_dtn_h, color="k", ls="--", lw=0.8, label="open problem (DtN)")
    ax[0].invert_xaxis(); ax[1].invert_xaxis()
    ax[0].set_xlabel(r"nominal PML reflection $R_0$"); ax[0].set_ylabel(r"discrete inf-sup $\beta_h$")
    ax[0].set_title("(a) stability: truncated vs open problem"); ax[0].legend(fontsize=7)
    ax[1].set_xlabel(r"nominal PML reflection $R_0$")
    ax[1].set_ylabel(r"$|P_{\rm PML}-P_{\rm DtN}|/P_{\rm DtN}$")
    ax[1].set_title("(b) PML modelling error on $P_{\\rm abs}$")
    for a in ax:
        a.grid(True, which="both", lw=0.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_pml1d.png")


if __name__ == "__main__":
    main()
