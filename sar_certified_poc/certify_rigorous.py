"""
Fully computable two-sided bracket on (windowed) absorbed power, 1D (revision R1).
=================================================================================

Sparse re-implementation of certify.py in which EVERY ingredient of the bracket
is an upper bound by construction, except the lower bound on the continuous
inf-sup constant, which is the single explicitly stated hypothesis (H1):

  (H1)  gamma_minus <= gamma, with gamma_minus = beta_h on a fine mesh; justified by
        the verified monotone increase of beta_h under refinement and the
        geometric-tail enclosure (infsup_enclosure.py).

Problem (exp(+j w t), state E = E_x(y), y in [0, Y]):
    -E'' - k^2 E = 0,  k^2 = w^2 mu0 (eps0 eps_r - j sigma / w),
    -E'(0) + j k0 E(0) = 2 j k0 a_inc,    E'(Y) + j k0 E(Y) = 0
(the Robin conditions are the EXACT Dirichlet-to-Neumann maps of the air
half-spaces in 1D: there is no truncation error in 1D).

Output (windowed absorbed power per unit area, the numerator of the averaged SAR)
    J_w(E) = (1/2) int_{y0}^{y1} sigma |E|^2 dy.
Error representation, e = E - E_h, z the adjoint, z_h its Galerkin approximation:
    J_w(E) - J_w(E_h) = Re int_w sigma conj(E_h) e  +  J_w(e)
                      = 2 Re R_p(z - z_h)            +  J_w(e),    J_w(e) >= 0,
so with eta_p >= ||R_p||_*, eta_d >= ||R_d||_* (equilibrated fluxes below) and
gamma_minus <= gamma:
    |2 Re R_p(z - z_h)| <= 2 eta_p eta_d / gamma_minus             (first order)
    0 <= J_w(e) <= (sigma_max,w / 2) ||e||^2_L2(w)                 (remainder)
    ||e|| <= ||E_f - E_h|| + eta_f / (gamma_minus k0)       (E_f: refined solution)
    J_w(E) in [ J_h - 2 eta_p eta_d/g ,  J_h + 2 eta_p eta_d/g + remainder ].

Norm: |||v|||^2 = ||v'||^2 + k0^2 ||v||^2 (L2 on [0, Y]); k0 ||v|| <= |||v|||.

Equilibrated residual bound (generic, used for primal AND adjoint):
residual r(v) = int f conj(v) + g0 conj(v(0)) + gY conj(v(Y)) - int u_h' conj(v')
                + int kappa u_h conj(v) - a0 u_h(0) conj(v(0)) - aY u_h(Y) conj(v(Y)).
Flux q (continuous, piecewise linear): q(0) = a0 u_h(0) - g0, and on each element
q' = -mean_e(s), s = f + kappa u_h. Then exactly
    r(v) = int (q - u_h') conj(v') + sum_e int_e (s - mean_e s) conj(v - mean_e v)
           + d_Y conj(v(Y)),   d_Y = gY - aY u_h(Y) - q(Y),
and Cauchy-Schwarz + the Poincare inequality on an interval (constant h_e/pi) +
the trace inequality |v(Y)|^2 <= (1/Y + t)||v||^2 + ||v'||^2 / t give
    ||r||_* <= eta = ||q - u_h'|| + (sum_e (h_e/pi)^2 ||s - mean s||_e^2)^(1/2)
                     + C_tr |d_Y|.
All integrals are evaluated exactly (3-point Gauss on each element, split at the
window ends: integrands are polynomials of degree <= 2 on each piece).
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from helmholtz1d import Phantom, interface_mesh, TISSUE, C0, k2_of, DEFAULT_LAYERS

_G3 = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)])
_W3 = np.array([5.0, 8.0, 5.0]) / 9.0
_G20, _W20 = np.polynomial.legendre.leggauss(20)


# --------------------------------------------------------------------------
# mesh-level data
# --------------------------------------------------------------------------
def elem_data(ph, x):
    mid = 0.5 * (x[:-1] + x[1:])
    names = [ph.tissue_at(m) for m in mid]
    k2 = np.array([k2_of(n, ph.w) for n in names])
    sig = np.array([TISSUE[n][1] for n in names])
    rho = np.array([TISSUE[n][2] for n in names])
    return k2, sig, rho


def refine(x, rf):
    pts = [np.linspace(x[e], x[e + 1], rf + 1)[:-1] for e in range(len(x) - 1)]
    return np.concatenate(pts + [x[-1:]])


def assemble(x, kappa, a0, aY):
    """Matrix of a(u,v) = int u'v' - kappa u v + a0 u(0)v(0) + aY u(Y)v(Y)."""
    h = np.diff(x)
    n = len(x)
    kd = 1.0 / h - kappa * h / 3.0                     # element diagonal entries
    ko = -1.0 / h - kappa * h / 6.0                    # element off-diagonal
    d = np.zeros(n, dtype=complex)
    d[:-1] += kd; d[1:] += kd
    d[0] += a0; d[-1] += aY
    return sp.diags([ko, d, ko], [-1, 0, 1], format="csc")


def norm_matrix(x, w2=1.0):
    """|||v|||^2 = ||v'||^2 + w2 ||v||^2  (w2 = squared reference wavenumber)."""
    h = np.diff(x)
    n = len(x)
    kd = 1.0 / h + w2 * h / 3.0
    ko = -1.0 / h + w2 * h / 6.0
    d = np.zeros(n)
    d[:-1] += kd; d[1:] += kd
    return sp.diags([ko, d, ko], [-1, 0, 1], format="csc")


def _pieces(x, y0, y1):
    """Each element split at the window ends: list of (lo, hi, inside) arrays."""
    a, b = x[:-1], x[1:]
    c = np.clip(y0, a, b); d = np.clip(y1, a, b)
    return [(a, c, 0.0), (c, d, 1.0), (d, b, 0.0)]


def _p1(u, x, y):
    """P1 field u (nodal) evaluated at points y[e, g] of element e."""
    a = x[:-1, None]; h = np.diff(x)[:, None]
    t = (y - a) / h
    return u[:-1, None] * (1 - t) + u[1:, None] * t


def _gauss_pts(lo, hi):
    m = 0.5 * (lo + hi); r = 0.5 * (hi - lo)
    return m[:, None] + r[:, None] * _G3[None, :], r[:, None] * _W3[None, :]


# --------------------------------------------------------------------------
# output functional and adjoint load (exact integration)
# --------------------------------------------------------------------------
def window_output(x, u, sig, y0, y1):
    """(1/2) int_{y0}^{y1} sigma |u|^2 for P1 u (exact)."""
    J = 0.0
    for lo, hi, ins in _pieces(x, y0, y1):
        if ins == 0.0:
            continue
        y, w = _gauss_pts(lo, hi)
        J += np.sum(sig[:, None] * np.abs(_p1(u, x, y)) ** 2 * w)
    return 0.5 * J


def window_load(x, u, sig, y0, y1):
    """F_i = (1/2) int_w sigma u phi_i  (adjoint load = output derivative)."""
    n = len(x)
    F = np.zeros(n, dtype=complex)
    a = x[:-1, None]; h = np.diff(x)[:, None]
    for lo, hi, ins in _pieces(x, y0, y1):
        if ins == 0.0:
            continue
        y, w = _gauss_pts(lo, hi)
        t = (y - a) / h
        uval = _p1(u, x, y) * sig[:, None] * w * 0.5
        F[:-1] += np.sum(uval * (1 - t), axis=1)
        F[1:] += np.sum(uval * t, axis=1)
    return F


# --------------------------------------------------------------------------
# guaranteed residual dual norm by equilibrated flux (generic)
# --------------------------------------------------------------------------
def trace_constant(Y, w2=1.0):
    """|v(Y)| <= C |||v|||: |v(Y)|^2 <= (1/Y + t)||v||^2 + ||v'||^2/t, minimised
    over t of max((1/Y + t)/w2, 1/t)  ->  t^2 + t/Y - w2 = 0."""
    t = 0.5 * (-1.0 / Y + np.sqrt(1.0 / Y ** 2 + 4.0 * w2))
    return np.sqrt(1.0 / t)


def equilibrated_eta(x, uh, kappa, a0, g0, aY, gY, f_win=None, w2=1.0):
    """eta >= ||r||_* for the residual defined in the module docstring.
    f_win = (fvals_fn, y0, y1): source f = fvals_fn(y, e) on [y0, y1], 0 outside."""
    h = np.diff(x)
    ne = len(h)
    # element integral of s = f + kappa u_h, and ||s - mean||^2 needs pieces
    pieces = _pieces(x, *(f_win[1:] if f_win else (x[0], x[0])))
    Ys, Ws, Fs = [], [], []
    for lo, hi, ins in pieces:
        y, w = _gauss_pts(lo, hi)
        s = kappa[:, None] * _p1(uh, x, y)
        if f_win is not None and ins == 1.0:
            s = s + f_win[0](y)
        Ys.append(y); Ws.append(w); Fs.append(s)
    Ws = np.concatenate(Ws, axis=1); Fs = np.concatenate(Fs, axis=1)
    int_s = np.sum(Fs * Ws, axis=1)
    mean_s = int_s / h
    osc2 = (h / np.pi) ** 2 * np.sum(np.abs(Fs - mean_s[:, None]) ** 2 * Ws, axis=1)

    q = np.empty(ne + 1, dtype=complex)
    q[0] = a0 * uh[0] - g0
    q[1:] = q[0] - np.cumsum(int_s)
    slope = np.diff(uh) / h
    di, dj = q[:-1] - slope, q[1:] - slope
    flux2 = h * (np.abs(di) ** 2 + np.real(di * np.conj(dj)) + np.abs(dj) ** 2) / 3.0
    dY = gY - aY * uh[-1] - q[-1]
    Y = x[-1] - x[0]
    return (np.sqrt(np.sum(flux2)) + np.sqrt(np.sum(osc2))
            + trace_constant(Y, w2) * abs(dY))


# --------------------------------------------------------------------------
# discrete inf-sup (sparse): beta^2 = min eig (A^H N^-1 A, N)
# --------------------------------------------------------------------------
def infsup(A, N):
    """1/beta^2 = max eig of (N A^-H N A^-1 N) w = lam N w  (Hermitian, N SPD)."""
    luA = spla.splu(A.tocsc())
    luN = spla.splu(N.tocsc().astype(complex))
    n = A.shape[0]

    def op(w):
        y = luA.solve(N @ w)
        y = luA.solve(N @ y, trans="H")
        return N @ y
    Op = spla.LinearOperator((n, n), matvec=op, dtype=complex)
    Minv = spla.LinearOperator((n, n), matvec=lambda v: luN.solve(v), dtype=complex)
    Nop = spla.LinearOperator((n, n), matvec=lambda v: N @ v, dtype=complex)
    lam = spla.eigsh(Op, k=1, M=Nop, Minv=Minv, which="LA",
                     return_eigenvectors=False, tol=1e-10, maxiter=5000)[0]
    return 1.0 / np.sqrt(lam)


def norm_weight(ph):
    """Squared reference wavenumber of the norm |||v|||^2 = ||v'||^2 + k0^2 ||v||^2
    (free-space wavenumber; any positive weight gives a valid bracket, k0^2
    balances the first-order and remainder terms for the tissue phantoms)."""
    return ph.k0 ** 2


def gamma_minus(ph, x_ref):
    """Lower bound of the continuous inf-sup under (H1): beta_h on mesh x_ref."""
    k2, _, _ = elem_data(ph, x_ref)
    return infsup(assemble(x_ref, k2, 1j * ph.k0, 1j * ph.k0),
                  norm_matrix(x_ref, norm_weight(ph)))


# --------------------------------------------------------------------------
# exact references (transfer matrix + 20-point Gauss per smooth piece)
# --------------------------------------------------------------------------
def exact_window_output(ph, y0, y1):
    br = np.unique(np.concatenate([[y0, y1], ph.edges[(ph.edges > y0) & (ph.edges < y1)]]))
    J = 0.0
    for lo, hi in zip(br[:-1], br[1:]):
        sg = ph.sigma_at(0.5 * (lo + hi))
        if sg == 0.0:
            continue
        y = 0.5 * (lo + hi) + 0.5 * (hi - lo) * _G20
        J += sg * np.sum(np.abs(ph.exact_field(y)) ** 2 * _W20) * 0.5 * (hi - lo)
    return 0.5 * J


def window_mass(ph, y0, y1):
    br = np.unique(np.concatenate([[y0, y1], ph.edges[(ph.edges > y0) & (ph.edges < y1)]]))
    return sum(TISSUE[ph.tissue_at(0.5 * (lo + hi))][2] * (hi - lo)
               for lo, hi in zip(br[:-1], br[1:]))


# --------------------------------------------------------------------------
# the bracket
# --------------------------------------------------------------------------
def bracket(ph, x, gmin, window=None, solved=None, fine=None, rf=16):
    """Two-sided bracket on J_w (window=None -> whole domain = P_abs)."""
    k2, sig, _ = elem_data(ph, x)
    k0 = ph.k0
    w2 = norm_weight(ph)
    y0, y1 = window if window is not None else (x[0], x[-1])
    if solved is None:
        A = assemble(x, k2, 1j * k0, 1j * k0)
        b = np.zeros(len(x), dtype=complex); b[0] = 2j * k0
        luA = spla.splu(A)
        Eh = luA.solve(b)
        eta_p = equilibrated_eta(x, Eh, k2, 1j * k0, 2j * k0, 1j * k0, 0.0, w2=w2)
    else:
        luA, Eh, eta_p = solved
    J_h = window_output(x, Eh, sig, y0, y1)
    # adjoint  A^H z = (1/2) M_w E_h
    F = window_load(x, Eh, sig, y0, y1)
    zh = luA.solve(F, trans="H")

    def f_adj(y):
        return 0.5 * sig[:, None] * _p1(Eh, x, y)
    eta_d = equilibrated_eta(x, zh, np.conj(k2), -1j * k0, 0.0, -1j * k0, 0.0,
                             f_win=(f_adj, y0, y1), w2=w2)
    inside = (x[1:] > y0) & (x[:-1] < y1)
    smax = sig[inside].max() if inside.any() else 0.0
    lin = 2.0 * eta_p * eta_d / gmin
    # remainder J_w(e) <= (smax/2) ||e||^2_L2(w), with the guaranteed split
    # ||e|| <= ||E_f - E_h|| + ||E - E_f||,  ||E - E_f|| <= eta_f / (gmin sqrt(w2))
    # (E_f: solution on the rf-times refined mesh, eta_f its equilibrated bound)
    if fine is None:
        fine = primal_fine(ph, x, rf)
    xf, Ef, eta_f = fine
    eh_f = np.interp(xf, x, Eh.real) + 1j * np.interp(xf, x, Eh.imag)
    _, sigf, _ = elem_data(ph, xf)
    d2 = 2.0 * window_output(xf, Ef - eh_f, np.where(sigf > 0, 1.0, 0.0), y0, y1)
    l2 = np.sqrt(d2) + eta_f / (gmin * np.sqrt(w2))
    quad = 0.5 * smax * l2 ** 2
    return dict(J_h=J_h, lo=J_h - lin, hi=J_h + lin + quad, lin=lin, quad=quad,
                eta_p=eta_p, eta_d=eta_d, gmin=gmin, solved=(luA, Eh, eta_p),
                fine=fine)


def primal(ph, x):
    k2, _, _ = elem_data(ph, x)
    k0 = ph.k0
    w2 = norm_weight(ph)
    A = assemble(x, k2, 1j * k0, 1j * k0)
    b = np.zeros(len(x), dtype=complex); b[0] = 2j * k0
    luA = spla.splu(A)
    Eh = luA.solve(b)
    eta_p = equilibrated_eta(x, Eh, k2, 1j * k0, 2j * k0, 1j * k0, 0.0, w2=w2)
    return luA, Eh, eta_p


def primal_fine(ph, x, rf):
    xf = refine(x, rf)
    _, Ef, eta_f = primal(ph, xf)
    return xf, Ef, eta_f


if __name__ == "__main__":
    np.seterr(all="ignore")
    ph = Phantom(DEFAULT_LAYERS, 2e9)
    lam = C0 / 2e9 / np.sqrt(53)
    gmin = gamma_minus(ph, interface_mesh(ph, lam / 512))
    Pex = exact_window_output(ph, 0.0, ph.Y)
    print(f"gamma_minus = {gmin:.7f}   exact P_abs = {Pex:.9e}")
    for frac in [8, 16, 32, 64, 128, 256]:
        x = interface_mesh(ph, lam / frac)
        r = bracket(ph, x, gmin)
        ok = r["lo"] <= Pex <= r["hi"]
        print(f"n={len(x)-1:5d} J_h={r['J_h']:.6e} [{r['lo']:.6e}, {r['hi']:.6e}] "
              f"in={ok} width={(r['hi']-r['lo'])/Pex*100:7.3f}% "
              f"eff={(r['hi']-r['lo'])/(2*abs(Pex-r['J_h'])):.1f}")
