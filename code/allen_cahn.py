"""Allen-Cahn gradient flow in one space dimension (Section 8.6).

Cell-centred discretization with homogeneous Neumann conditions, its energy, gradient, Hessian and
AVF discrete gradient in closed form, the discrete interface (kink), one AVF step computed by Newton's
method or by continuation, and the classification of an endpoint by energy level and sign of the mean.
Imported by exp4_allencahn.py, exp5_hybrid.py and exp6_barrier.py.
"""
import numpy as np
import scipy.sparse as sp
from dgcont import track, newton_step


class AllenCahn:
    def __init__(self, d=256, eps=0.05, L=1.0):
        self.d, self.eps, self.L = d, eps, L
        self.dx = L / d
        self.xg = (np.arange(d) + 0.5) * self.dx
        e = np.ones(d)
        Lap = sp.diags([-e[:-1], 2 * e, -e[:-1]], [-1, 0, 1], format="lil")
        Lap[0, 0] = 1.0
        Lap[d - 1, d - 1] = 1.0
        self.Lap = Lap.tocsr()          # graph Laplacian, positive semidefinite
        self.mu = eps**2 / self.dx**2

    def E(self, U):
        return 0.5 * self.mu * np.sum(np.diff(U) ** 2) + 0.25 * np.sum((U**2 - 1.0) ** 2)

    def grad(self, U):
        return self.mu * (self.Lap @ U) + U**3 - U

    def hess(self, U):
        return self.mu * self.Lap + sp.diags(3 * U**2 - 1.0)

    def G(self, X, Y):
        return 0.5 * self.mu * (self.Lap @ (X + Y)) + 0.25 * (X + Y) * (X**2 + Y**2 - 2.0)

    def DyG(self, X, Y):
        return 0.5 * self.mu * self.Lap + sp.diags(0.25 * (3 * Y**2 + 2 * X * Y + X**2 - 2.0))


def kink(ac):
    U = np.tanh((ac.xg - 0.5 * ac.L) / (np.sqrt(2.0) * ac.eps))
    for _ in range(50):
        r = ac.grad(U)
        if np.linalg.norm(r) < 1e-13:
            break
        U = U - sp.linalg.spsolve(ac.hess(U).tocsc(), r)
    return U


def step(ac, X, h, method, a_scale=1e-7, rng=None):
    G = lambda Y: ac.G(X, Y)
    DG = lambda Y: ac.DyG(X, Y)
    if method == "newton_x":
        Y, k, ok = newton_step(X, h, G, DG, X.copy(), sparse=True)
        return Y, k, ok, None
    if method == "newton_ee":
        Y, k, ok = newton_step(X, h, G, DG, X - h * ac.grad(X), sparse=True)
        return Y, k, ok, None
    rng = np.random.default_rng(0) if rng is None else rng
    a = rng.standard_normal(ac.d)
    a *= a_scale / np.linalg.norm(a)
    res = track(X, h, G, DG, a=a, ds0=0.05, ds_max=1.0, sparse=True)
    return res["root"], res["nsolves"], True, res


def classify(ac, U, c):
    e = ac.E(U)
    if not np.all(np.isfinite(U)):
        return "diverged"
    if e < c:
        return "+1 well" if U.mean() > 0 else "-1 well"
    return f"above c (mean {U.mean():+.2f})"

