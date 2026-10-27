"""
1D lossy Helmholtz forward + EXACT reference (transfer matrix).
===============================================================

Physical reduction of the RF-dosimetry problem: a plane wave at normal
incidence on a layered tissue phantom. E = E_x(y), convention e^{+j w t}:

    -E'' - k(y)^2 E = 0,   k^2 = w^2 mu0 (eps0 eps_r - j sigma/w)   (Im k^2 < 0).

Impedance (radiation) boundary conditions model the semi-infinite air:
    y=0 :  E' - j k0 E = -2 j k0 a_inc      (incident a_inc + reflected)
    y=Y :  E' + j k0 E = 0                   (outgoing only)

Absorbed power (the dosimetry quantity, ~ spatially-integrated SAR):
    P_abs = (1/2) int sigma |E|^2 dy = -(1/(2 w mu0)) int Im(k^2) |E|^2 dy.

The exact solution is obtained analytically (2x2 transfer matrices on [E,E']),
giving an independent reference for P_abs to machine precision.
"""

import numpy as np

EPS0 = 8.8541878128e-12
MU0 = 4.0e-7 * np.pi
C0 = 1.0 / np.sqrt(EPS0 * MU0)

TISSUE = {                       # (eps_r, sigma [S/m], rho [kg/m^3]) @ ~2 GHz
    "air":    (1.0, 0.0, 1.0),
    "skin":   (38.0, 1.30, 1100.0),
    "fat":    (5.3, 0.10, 910.0),
    "muscle": (53.0, 1.45, 1040.0),
}


def k2_of(name, w):
    er, sg, _ = TISSUE[name]
    return w * w * MU0 * (EPS0 * er - 1j * sg / w)


class Phantom:
    """Layered phantom on [0, Y]; layers = list of (tissue, thickness)."""
    def __init__(self, layers, freq):
        self.freq = freq
        self.w = 2 * np.pi * freq
        self.k0 = self.w * np.sqrt(MU0 * EPS0)          # air wavenumber
        self.names = [n for n, _ in layers]
        self.th = np.array([t for _, t in layers])
        self.edges = np.concatenate([[0.0], np.cumsum(self.th)])
        self.Y = self.edges[-1]

    def tissue_at(self, y):
        i = min(np.searchsorted(self.edges, y, side="right") - 1, len(self.names) - 1)
        return self.names[max(i, 0)]

    def k2_at(self, y):
        return k2_of(self.tissue_at(y), self.w)

    def sigma_at(self, y):
        return TISSUE[self.tissue_at(y)][1]

    # ---- exact solution via transfer matrices on the state vector [E, E'] ----
    def _layer_T(self, name, d):
        k = np.sqrt(k2_of(name, self.w))                # complex wavenumber
        return np.array([[np.cos(k * d), np.sin(k * d) / k],
                         [-k * np.sin(k * d), np.cos(k * d)]], dtype=np.complex128)

    def exact_state0(self, a_inc=1.0):
        """Solve for [E(0), E'(0)] from the two impedance BCs."""
        T = np.eye(2, dtype=np.complex128)
        for n, d in zip(self.names, self.th):
            T = self._layer_T(n, d) @ T
        k0 = self.k0
        # unknown x = [E0, E0']; BC0: E0' - j k0 E0 = -2 j k0 a_inc
        # [EY, EY'] = T x ; BCY: EY' + j k0 EY = 0
        row0 = np.array([-1j * k0, 1.0])                # . x = -2 j k0 a_inc
        rowY = T[1] + 1j * k0 * T[0]                    # . x = 0
        M = np.array([row0, rowY], dtype=np.complex128)
        rhs = np.array([-2j * k0 * a_inc, 0.0], dtype=np.complex128)
        return np.linalg.solve(M, rhs)                  # [E0, E0']

    def exact_field(self, y_query, a_inc=1.0):
        x = self.exact_state0(a_inc)
        out = np.empty(len(y_query), dtype=np.complex128)
        for i, y in enumerate(y_query):
            # propagate from 0 to y through completed layers + partial
            v = x.copy(); yy = 0.0
            for n, d in zip(self.names, self.th):
                top = yy + d
                seg = min(y, top) - yy
                if seg > 0:
                    v = self._layer_T(n, seg) @ v
                yy = min(y, top)
                if y <= top:
                    break
            out[i] = v[0]
        return out

    def exact_Pabs(self, a_inc=1.0, nq=400):
        """Exact absorbed power by fine quadrature of the analytic field."""
        P = 0.0
        for n, d, y0 in zip(self.names, self.th, self.edges[:-1]):
            sg = TISSUE[n][1]
            if sg == 0.0:
                continue
            yy = y0 + (np.arange(nq) + 0.5) * d / nq
            E = self.exact_field(yy, a_inc)
            P += sg * np.sum(np.abs(E) ** 2) * d / nq
        return 0.5 * P


# ---------------------------------------------------------------------------
# P1 finite-element forward solver on a uniform mesh.
# ---------------------------------------------------------------------------
def interface_mesh(phantom, h_target):
    """Uniform-per-layer mesh with nodes exactly on every layer interface."""
    xs = [np.array([0.0])]
    for d, y0 in zip(phantom.th, phantom.edges[:-1]):
        m = max(int(np.ceil(d / h_target)), 1)
        xs.append(y0 + (np.arange(1, m + 1)) * d / m)
    return np.unique(np.concatenate(xs))


def solve_fem(phantom, nelem=None, a_inc=1.0, x=None):
    Y = phantom.Y
    if x is None:
        x = np.linspace(0.0, Y, nelem + 1)
    nelem = len(x) - 1
    w, k0 = phantom.w, phantom.k0
    A = np.zeros((nelem + 1, nelem + 1), dtype=np.complex128)
    b = np.zeros(nelem + 1, dtype=np.complex128)

    # local matrices: stiffness [[1,-1],[-1,1]]/he ; mass [[2,1],[1,2]]*he/6
    for e in range(nelem):
        he = x[e + 1] - x[e]
        ym = 0.5 * (x[e] + x[e + 1])
        k2 = phantom.k2_at(ym)
        Ke = np.array([[1.0, -1.0], [-1.0, 1.0]]) / he
        Me = np.array([[2.0, 1.0], [1.0, 2.0]]) * he / 6.0
        Ae = Ke - k2 * Me
        A[e:e + 2, e:e + 2] += Ae

    # impedance BC contributions: +j k0 at both endpoints; source at y=0
    A[0, 0] += 1j * k0
    A[-1, -1] += 1j * k0
    b[0] += 2j * k0 * a_inc

    E = np.linalg.solve(A, b)
    return x, E, A, b


def fem_Pabs(phantom, x, E):
    """P_abs from the FEM field via element-wise quadrature of sigma|E|^2."""
    P = 0.0
    for e in range(len(x) - 1):
        ym = 0.5 * (x[e] + x[e + 1])
        sg = phantom.sigma_at(ym)
        if sg == 0.0:
            continue
        h = x[e + 1] - x[e]
        # int over element of |linear E|^2  (exact for P1)
        Ea, Eb = E[e], E[e + 1]
        integ = h * (np.abs(Ea) ** 2 + np.real(Ea * np.conj(Eb)) + np.abs(Eb) ** 2) / 3.0
        P += sg * integ
    return 0.5 * P


DEFAULT_LAYERS = [("air", 0.030), ("skin", 0.010), ("fat", 0.020),
                  ("muscle", 0.040), ("air", 0.030)]
