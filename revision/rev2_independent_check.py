"""Revision R2 (referee 1, point 8): checks written independently of the scripts
that produce the figures and tables.

(a) Error representation (13) and Theorem 1 in finite dimension. For a random
    complex indefinite matrix A (full space), a Galerkin subspace V, a Hermitian
    positive semi-definite M and J(v) = v^H M v / 2:
      J(E) - J(E_h) = 2 Re (z - z_h)^H r + J(e),   r = b - A E_h,  A^H z = M E_h / 2,
    and |2 Re (z - z_h)^H r| <= 2 ||r||_* ||r_d||_* / beta, with beta the inf-sup
    constant of A in the norm N and r_d = M E_h / 2 - A^H z_h.
(b) One-dimensional residual bounds. The equilibrated bounds eta_p, eta_d of
    certify_rigorous.py are compared with the dual norm of the same residual
    computed by its Riesz representative on the mesh refined 64 times, with an
    assembly written here from scratch (piecewise-linear elements, exact
    element matrices, five-point Gauss rule split at the window ends). The Riesz
    value is a lower bound of the true dual norm, so eta >= Riesz is necessary.
Output: results/independent_check.json"""
import json
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from rev_common import RES
import certify_rigorous as R
from helmholtz1d import Phantom, interface_mesh, C0, DEFAULT_LAYERS, TISSUE
from rev_peak10g import windows

rng = np.random.default_rng(2026)


# ---------------------------------------------------------------- (a)
def check_theorem(n=60, m=25, trials=200):
    worst_id, worst_ratio = 0.0, 0.0
    for _ in range(trials):
        B = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        Nn = B @ B.conj().T / n + np.eye(n)                     # norm matrix (HPD)
        K = rng.standard_normal((n, n)); K = K @ K.T / n
        A = K - 1.5 * np.eye(n) + 0.3j * np.diag(rng.random(n))  # indefinite, lossy
        Mh = rng.standard_normal((n, n)); M = Mh @ Mh.T / n      # HPSD output matrix
        b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        V = np.linalg.qr(rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m)))[0]
        E = np.linalg.solve(A, b)
        Eh = V @ np.linalg.solve(V.conj().T @ A @ V, V.conj().T @ b)
        e = E - Eh
        J = lambda v: 0.5 * np.real(v.conj() @ M @ v)
        z = np.linalg.solve(A.conj().T, 0.5 * M @ Eh)
        zh = V @ np.linalg.solve(V.conj().T @ A.conj().T @ V, V.conj().T @ (0.5 * M @ Eh))
        r = b - A @ Eh
        lhs = J(E) - J(Eh)
        first = 2 * np.real((z - zh).conj() @ r)
        rhs = first + J(e)
        worst_id = max(worst_id, abs(lhs - rhs) / max(abs(lhs), 1e-300))
        # inf-sup: beta^2 = min eig(A^H N^-1 A, N)
        from scipy.linalg import eigh
        C = A.conj().T @ np.linalg.solve(Nn, A); C = 0.5 * (C + C.conj().T)
        beta = np.sqrt(eigh(C, Nn, eigvals_only=True)[0])
        dn = lambda g: np.sqrt(np.real(g.conj() @ np.linalg.solve(Nn, g)))
        rd = 0.5 * M @ Eh - A.conj().T @ zh
        bound = 2 * dn(r) * dn(rd) / beta
        worst_ratio = max(worst_ratio, abs(first) / bound)
    return dict(trials=trials, max_rel_identity_defect=float(worst_id), max_first_order_over_bound=float(worst_ratio))


# ---------------------------------------------------------------- (b)
G5, W5 = np.polynomial.legendre.leggauss(5)


def p1_matrices(x, kappa):
    """Stiffness K, mass M (norm) and kappa-mass on P1, exact element matrices."""
    h = np.diff(x); n = len(x)
    i = np.arange(n - 1)
    rows = np.concatenate([i, i, i + 1, i + 1]); cols = np.concatenate([i, i + 1, i, i + 1])
    k = np.concatenate([1 / h, -1 / h, -1 / h, 1 / h])
    m = np.concatenate([h / 3, h / 6, h / 6, h / 3])
    km = np.concatenate([kappa * h / 3, kappa * h / 6, kappa * h / 6, kappa * h / 3])
    mk = lambda v: sp.csr_matrix((v, (rows, cols)), shape=(n, n))
    return mk(k), mk(m), mk(km)


def window_source(x, fvals, y0, y1):
    """Load int_{[y0,y1]} f phi_i with f(y) given as callable, split at y0, y1."""
    F = np.zeros(len(x), dtype=complex)
    for e in range(len(x) - 1):
        a, b = x[e], x[e + 1]
        lo, hi = max(a, y0), min(b, y1)
        if hi <= lo:
            continue
        y = 0.5 * (lo + hi) + 0.5 * (hi - lo) * G5; w = 0.5 * (hi - lo) * W5
        fy = fvals(y)
        t = (y - a) / (b - a)
        F[e] += np.sum(fy * (1 - t) * w); F[e + 1] += np.sum(fy * t * w)
    return F


def riesz_dual_norm(ph, x, uh, adjoint=False, window=None, sigma_c=None, Eh=None, rf=64):
    xf = R.refine(x, rf)
    k2f, sigf, _ = R.elem_data(ph, xf)
    k0 = ph.k0
    kappa = np.conj(k2f) if adjoint else k2f
    K, Mn, Mk = p1_matrices(xf, kappa)
    u = np.interp(xf, x, uh.real) + 1j * np.interp(xf, x, uh.imag)
    a = -1j * k0 if adjoint else 1j * k0
    r = -(K @ u) + Mk @ u
    r[0] -= a * u[0]; r[-1] -= a * u[-1]
    if not adjoint:
        r[0] += 2j * k0
    else:
        y0, y1 = window
        sig_fn = lambda y: np.array([ph.sigma_at(t) for t in y])
        Eh_fn = lambda y: np.interp(y, x, Eh.real) + 1j * np.interp(y, x, Eh.imag)
        r += window_source(xf, lambda y: 0.5 * sig_fn(y) * Eh_fn(y), y0, y1)
    Nf = (K + R.norm_weight(ph) * Mn).tocsc()
    return float(np.sqrt(np.real(r.conj() @ spla.spsolve(Nf, r))))


def check_eta():
    out = dict(primal=[], adjoint=[], sweep=[])
    ph = Phantom(DEFAULT_LAYERS, 2e9)
    lam = C0 / 2e9 / np.sqrt(53)
    W = windows(ph)
    for fr in [8, 16, 32, 64, 128, 256]:
        x = interface_mesh(ph, lam / fr)
        luA, Eh, eta_p = R.primal(ph, x)
        rz = riesz_dual_norm(ph, x, Eh)
        out["primal"].append(dict(frac=fr, eta=float(eta_p), riesz=rz, ratio=float(eta_p / rz)))
        if fr in (16, 64, 256):
            for wi in (0, len(W) // 2, len(W) - 1):
                br = R.bracket(ph, x, 0.24, window=tuple(W[wi]), solved=(luA, Eh, eta_p))
                _, sig, _ = R.elem_data(ph, x)
                F = R.window_load(x, Eh, sig, *W[wi])
                zh = luA.solve(F, trans="H")
                rzd = riesz_dual_norm(ph, x, zh, adjoint=True, window=tuple(W[wi]), Eh=Eh)
                out["adjoint"].append(dict(frac=fr, window=int(wi), eta=float(br["eta_d"]), riesz=rzd,
                                           ratio=float(br["eta_d"] / rzd)))
    TISSUE["slab"] = (80.0, 0.03, 1000.0)
    for f in np.linspace(0.6e9, 3.0e9, 90)[::3]:
        ph = Phantom([("air", 0.01), ("slab", 0.02), ("air", 0.01)], f)
        x = interface_mesh(ph, C0 / f / np.sqrt(80.0) / 20)
        _, Eh, eta_p = R.primal(ph, x)
        rz = riesz_dual_norm(ph, x, Eh)
        out["sweep"].append(dict(f=float(f), eta=float(eta_p), riesz=rz, ratio=float(eta_p / rz)))
    return out


def main():
    a = check_theorem()
    print("(a) identity defect %.2e, max |first-order| / bound %.3f (must be <= 1)" %
          (a["max_rel_identity_defect"], a["max_first_order_over_bound"]))
    b = check_eta()
    for k in ("primal", "adjoint", "sweep"):
        rat = [d["ratio"] for d in b[k]]
        print("(b) %s: eta / Riesz in [%.3f, %.3f] over %d cases (must be >= 1)" % (k, min(rat), max(rat), len(rat)))
    json.dump(dict(theorem=a, eta=b), open(f"{RES}/independent_check.json", "w"), indent=1)


if __name__ == "__main__":
    main()
