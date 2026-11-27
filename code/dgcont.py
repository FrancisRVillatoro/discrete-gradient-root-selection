"""Core routines for the experiments.

Mean-value (AVF) discrete gradient by Gauss-Legendre quadrature, Newton solvers for
a single discrete-gradient step, pseudo-arclength continuation of the homotopy

    H_a(y, t) = y - x + t h G(x, y) + t (1 - t) a,     0 <= t <= 1,

from (x, 0) to t = 1, and chord and polygonal energy certificates.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def gauss01(n):
    xi, wi = leggauss(n)
    return 0.5 * (xi + 1.0), 0.5 * wi


class AVF:
    """Mean-value discrete gradient of an energy with gradient and Hessian callables."""

    def __init__(self, grad, hess, nq=24):
        self.grad, self.hess = grad, hess
        self.s, self.w = gauss01(nq)

    def G(self, x, y):
        d = y - x
        out = np.zeros_like(x, dtype=float)
        for si, wi in zip(self.s, self.w):
            out = out + wi * self.grad(x + si * d)
        return out

    def DyG(self, x, y):
        d = y - x
        out = 0.0
        for si, wi in zip(self.s, self.w):
            out = out + wi * si * self.hess(x + si * d)
        return out


def _solve(J, r, sparse):
    if sparse:
        return spla.spsolve(J.tocsc(), r)
    return np.linalg.solve(J, r)


def newton_step(x, h, G, DG, y0, tol=1e-12, maxit=60, sparse=False):
    """Plain Newton iteration for y - x + h G(y) = 0 started at y0.

    Stops when the residual is below tol (relative), or when the correction reaches
    rounding level with a residual below 1e-8 (relative), which happens for large h.
    """
    y = np.array(y0, dtype=float)
    n = y.size
    I = sp.identity(n, format="csr") if sparse else np.eye(n)
    nsol = 0
    for k in range(maxit):
        F = y - x + h * G(y)
        if not np.all(np.isfinite(F)):
            return y, nsol, False
        scale = 1.0 + np.linalg.norm(y)
        if np.linalg.norm(F) <= tol * scale:
            return y, nsol, True
        J = I + h * DG(y)
        dy = _solve(J, F, sparse)
        y = y - dy
        nsol += 1
        if np.linalg.norm(dy) <= 1e-14 * (1.0 + np.linalg.norm(y)):
            F = y - x + h * G(y)
            return y, nsol, bool(np.all(np.isfinite(F)) and np.linalg.norm(F) <= 1e-8 * (1.0 + np.linalg.norm(y)))
    F = y - x + h * G(y)
    return y, nsol, bool(np.all(np.isfinite(F)) and np.linalg.norm(F) <= 1e-8 * (1.0 + np.linalg.norm(y)))


def track(x, h, G, DG, a=None, ds0=0.05, ds_max=0.5, ds_min=1e-11, tol=1e-11,
          maxit=7, cosmin=0.9, sparse=False, max_steps=500000):
    """Pseudo-arclength continuation of H_a(y,t) = y - x + t h G(y) + t(1-t) a.

    G(y) and DG(y) are the discrete gradient G(x, y) and its y-Jacobian for the fixed x.
    Returns a dict with the accepted path points (y, t), the root at t = 1 and counters.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    a = np.zeros(n) if a is None else np.asarray(a, dtype=float)
    I = sp.identity(n, format="csr") if sparse else np.eye(n)

    def H(y, t):
        return y - x + t * h * G(y) + t * (1.0 - t) * a

    def bordered(y, t, tau):
        Hy = I + t * h * DG(y)
        Ht = h * G(y) + (1.0 - 2.0 * t) * a
        if sparse:
            return sp.bmat([[Hy, sp.csr_matrix(Ht.reshape(-1, 1))],
                            [sp.csr_matrix(tau[:n].reshape(1, -1)), sp.csr_matrix([[tau[n]]])]],
                           format="csc")
        J = np.zeros((n + 1, n + 1))
        J[:n, :n] = Hy
        J[:n, n] = Ht
        J[n, :] = tau
        return J

    w = np.concatenate([x, [0.0]])
    tau = np.concatenate([-(h * G(x) + a), [1.0]])
    tau /= np.linalg.norm(tau)
    path = [w.copy()]
    ds = ds0
    nsol, nfold, nrej, nacc = 0, 0, 0, 0
    e_last = np.zeros(n + 1)
    e_last[n] = 1.0
    for _ in range(max_steps):
        wp = w + ds * tau
        wc = wp.copy()
        conv = False
        its = 0
        for its in range(maxit):
            y, t = wc[:n], wc[n]
            r = np.concatenate([H(y, t), [tau @ (wc - wp)]])
            if not np.all(np.isfinite(r)):
                break
            if np.linalg.norm(r) <= tol * (1.0 + np.linalg.norm(wc)):
                conv = True
                break
            dw = _solve(bordered(y, t, tau), -r, sparse)
            nsol += 1
            wc = wc + dw
        tn = None
        if conv and np.linalg.norm(wc - wp) <= 0.5 * ds and wc[n] > -1e-12:
            z = _solve(bordered(wc[:n], wc[n], tau), e_last, sparse)
            nsol += 1
            tn = z / np.linalg.norm(z)
            if tn @ tau < cosmin:
                tn = None
        if tn is None:
            nrej += 1
            ds *= 0.5
            if ds < ds_min:
                raise RuntimeError("continuation step size underflow")
            continue
        if wc[n] >= 1.0:
            # land exactly on t = 1, where the perturbation vanishes
            tp, yp = w[n], w[:n]
            y0 = yp + (1.0 - tp) / (wc[n] - tp) * (wc[:n] - yp)
            yr, k, ok = newton_step(x, h, G, DG, y0, tol=1e-13, maxit=30, sparse=sparse)
            nsol += k
            if not ok or np.linalg.norm(yr - y0) > ds:
                nrej += 1
                ds *= 0.5
                continue
            path.append(np.concatenate([yr, [1.0]]))
            nacc += 1
            return dict(path=np.array(path), root=yr, nsolves=nsol, nfold=nfold,
                        nrej=nrej, nacc=nacc)
        if np.sign(tn[n]) != np.sign(tau[n]):
            nfold += 1
        w, tau = wc, tn
        path.append(w.copy())
        nacc += 1
        if its <= 2:
            ds = min(1.5 * ds, ds_max)
    raise RuntimeError("continuation did not reach t = 1")


def bk(A, B, k):
    """max_{0<=t<=1} (1-t)A + tB + (k/2) t (1-t)."""
    if k <= 0.0:
        return max(A, B)
    t = min(1.0, max(0.0, 0.5 + (B - A) / k))
    return (1.0 - t) * A + t * B + 0.5 * k * t * (1.0 - t)


def chord_check(energy, x, y, K, level=None, dmax=6):
    """Dyadic refinement of Proposition 6.1 on the chord [x, y].

    K >= 0 is a curvature bound, d^2/ds^2 E(x + s (y - x)) >= -K on [0, 1].
    With level=None the chord is compared with E(x): `certified` means that the bound of
    Proposition 6.1 equals E(x), so that the chord lies in {E <= E(x)}, and `witness` that a
    sample on the chord has energy above E(x).  With a level c > E(x), `certified` means that the
    bound is below c and `witness` that a sample has energy at or above c.  `undecided` is returned
    when neither happens up to depth dmax.  Returns (status, depth, value, energy evaluations),
    where value is the deciding bound or sample.
    """
    n = 2 ** dmax
    cache = {}

    def Ej(i):
        if i not in cache:
            cache[i] = float(energy(x + (i / n) * (y - x)))
        return cache[i]

    E0 = Ej(0)
    bound = None
    for k in range(dmax + 1):
        m = 2 ** k
        Es = [Ej(j * (n // m)) for j in range(m + 1)]
        top = max(Es) if level is not None else max(Es[1:])
        if (level is not None and top >= level) or (level is None and top > E0):
            return "witness", k, top, len(cache)
        bound = max(bk(Es[j], Es[j + 1], K / m**2) for j in range(m))
        if (level is not None and bound < level) or (level is None and bound <= E0):
            return "certified", k, bound, len(cache)
    return "undecided", dmax, bound, len(cache)


def polyline_ceiling(points, energy, kbound, m=1):
    """Certified upper bound for E on the polygon through `points`.

    kbound(p, q) must return K >= 0 with d^2/ds^2 E(p + s (q - p)) >= -K on [0, 1].
    Each segment is sampled at m+1 equally spaced points and Proposition 6.1 is applied.
    """
    ceiling = -np.inf
    for p, q in zip(points[:-1], points[1:]):
        K = kbound(p, q)
        Es = [energy(p + (j / m) * (q - p)) for j in range(m + 1)]
        for j in range(m):
            ceiling = max(ceiling, bk(Es[j], Es[j + 1], K / m**2))
    return ceiling
