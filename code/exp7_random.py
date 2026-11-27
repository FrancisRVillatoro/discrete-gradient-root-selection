"""Section 8.5: a log-sum-exp energy with random patterns in dimension 32.

Energy (6.3) with unit weights, N = 24 normalized Gaussian patterns in R^32 and beta = 8.
The energy is not separable and has no symmetry.  For each pattern the minimizer p_i next to
it and the index-one saddles reached by Newton's method from the midpoints between p_i and the
other minimizers are computed; s_i denotes the saddle of lowest energy whose unstable manifold
joins p_i to a second minimizer.  Starting points lie near s_i on the side of p_i,

    x = s_i + theta L (u + 0.5 n),   L = |p_i - s_i|,   u = (p_i - s_i)/L,

with n a random unit vector orthogonal to u and REPS independent draws of n for each pattern.
A point is kept when E(x) < E(s_i) and the
gradient flow from x converges to p_i.  One AVF step from x is then computed by Newton's method
from zero displacement, by Newton's method from the explicit Euler predictor and by the
continuation algorithm of Section 3.3.  Each root is assigned to the minimizer to which the
gradient flow converges from it, and the chord of the Newton root and the polygon of the
continuation path are checked against the level E(x) with the directional bound (6.5).
"""
import json
import time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp
from dgcont import track, newton_step, chord_check, polyline_ceiling

D, N, BETA = 32, 24, 8.0
THETAS = [0.05, 0.2, 0.5]
HS = [1.0, 2.0, 5.0, 10.0]
NQ = 64
DMAX = 6
REPS = 4
SEED = 0


class LSE:
    """E(z) = |z|^2/2 - log(sum_i exp(beta <xi_i, z>))/beta."""

    def __init__(self, Xi, beta):
        self.Xi, self.beta = Xi, beta
        self.I = np.eye(Xi.shape[1])

    def rho(self, z):
        a = self.beta * (z @ self.Xi.T)
        a = a - a.max(axis=-1, keepdims=True)
        w = np.exp(a)
        return w / w.sum(axis=-1, keepdims=True)

    def E(self, z):
        a = self.beta * (self.Xi @ z)
        m = a.max()
        return float(0.5 * (z @ z) - (m + np.log(np.exp(a - m).sum())) / self.beta)

    def grad(self, z):
        return z - self.rho(z) @ self.Xi

    def hess(self, z):
        r = self.rho(z)
        m = r @ self.Xi
        return self.I - self.beta * ((self.Xi.T * r) @ self.Xi - np.outer(m, m))


class AVF:
    """Mean-value discrete gradient of the log-sum-exp energy by Gauss-Legendre quadrature."""

    def __init__(self, lse, nq=NQ):
        xi, wi = leggauss(nq)
        self.s, self.w = 0.5 * (xi + 1.0), 0.5 * wi
        self.ws = self.w * self.s
        self.lse = lse

    def G(self, x, y):
        d = y - x
        R = self.lse.rho(x[None, :] + self.s[:, None] * d[None, :])
        return x + 0.5 * d - (self.w @ R) @ self.lse.Xi

    def DyG(self, x, y):
        d = y - x
        Xi = self.lse.Xi
        R = self.lse.rho(x[None, :] + self.s[:, None] * d[None, :])
        M = R @ Xi
        return 0.5 * self.lse.I - self.lse.beta * ((Xi.T * (self.ws @ R)) @ Xi - (M.T * self.ws) @ M)


def critical_point(lse, z, tol=1e-12, maxit=60):
    for _ in range(maxit):
        g = lse.grad(z)
        if np.linalg.norm(g) < tol:
            return z, True
        z = z - np.linalg.solve(lse.hess(z), g)
    return z, bool(np.linalg.norm(lse.grad(z)) < 1e-9)


def flow_limit(lse, P, z, T=40.0):
    """Index of the minimizer to which the gradient flow converges from z, or -1."""
    f = lambda t, y: -lse.grad(y)
    J = lambda t, y: -lse.hess(y)
    for _ in range(6):
        z = solve_ivp(f, (0.0, T), z, method="LSODA", jac=J, rtol=1e-9, atol=1e-11).y[:, -1]
        if np.linalg.norm(lse.grad(z)) < 1e-7:
            break
    dist = np.linalg.norm(P - z[None, :], axis=1)
    k = int(dist.argmin())
    return k if dist[k] < 1e-5 else -1


def kdelta(lse, p, q):
    """Curvature bound (6.5) along the segment [p, q]."""
    d = q - p
    pr = lse.Xi @ d
    om = pr.max() - pr.min()
    return max(0.0, 0.25 * lse.beta * om * om - float(d @ d))


def landscape(lse):
    """Minimizers next to the patterns and, for each, its lowest index-one saddle."""
    P = []
    for k in range(N):
        p, ok = critical_point(lse, lse.Xi[k].copy())
        assert ok and np.linalg.eigvalsh(lse.hess(p)).min() > 0
        P.append(p)
    P = np.array(P)
    assert all(flow_limit(lse, P, lse.Xi[k]) == k for k in range(N))
    cand = {}
    for i in range(N):
        for j in range(i + 1, N):
            s, ok = critical_point(lse, 0.5 * (P[i] + P[j]))
            if not ok:
                continue
            ev, V = np.linalg.eigh(lse.hess(s))
            if ev[0] < 0 < ev[1]:
                cand[(i, j)] = (lse.E(s), s, V[:, 0], float(ev[0]))
    saddles = []
    for i in range(N):
        opts = sorted((v[0], key) for key, v in cand.items() if i in key)
        for _, key in opts:
            e, s, v, mu = cand[key]
            ends = {flow_limit(lse, P, s + 1e-4 * v), flow_limit(lse, P, s - 1e-4 * v)}
            if i in ends and len(ends) == 2 and -1 not in ends:
                j = (ends - {i}).pop()
                saddles.append(dict(i=i, j=j, E=e, s=s, mu=mu))
                break
        else:
            raise RuntimeError(f"no saddle for pattern {i}")
    return P, saddles


def run():
    rng = np.random.default_rng(SEED)
    Xi = rng.standard_normal((N, D))
    Xi /= np.linalg.norm(Xi, axis=1, keepdims=True)
    lse = LSE(Xi, BETA)
    avf = AVF(lse)
    t0 = time.time()
    P, saddles = landscape(lse)
    diam = max(np.linalg.norm(Xi[i] - Xi[j]) for i in range(N) for j in range(i))
    info = dict(d=D, N=N, beta=BETA, nq=NQ, seed=SEED,
                nu_diam=float(0.25 * BETA * diam**2 - 1.0),
                min_energy=[lse.E(p) for p in P],
                saddle_energy=[sd["E"] for sd in saddles],
                saddle_partner=[sd["j"] for sd in saddles],
                saddle_eigenvalue=[sd["mu"] for sd in saddles],
                well_depth=[sd["E"] - lse.E(P[sd["i"]]) for sd in saddles],
                saddle_distance=[float(np.linalg.norm(P[sd["i"]] - sd["s"])) for sd in saddles])
    print(f"landscape: {time.time()-t0:.1f}s, nu_diam={info['nu_diam']:.3f}, "
          f"well depths {min(info['well_depth']):.3f}..{max(info['well_depth']):.3f}, "
          f"saddle eigenvalues {min(info['saddle_eigenvalue']):.2f}..{max(info['saddle_eigenvalue']):.2f}")
    kb = lambda p, q: kdelta(lse, p, q)
    rows = []
    for theta in THETAS:
        for sd, rep in [(sd, rep) for sd in saddles for rep in range(REPS)]:
            i, s = sd["i"], sd["s"]
            L = np.linalg.norm(P[i] - s)
            u = (P[i] - s) / L
            for _ in range(50):
                n = rng.standard_normal(D)
                n -= (n @ u) * u
                n /= np.linalg.norm(n)
                x = s + theta * L * (u + 0.5 * n)
                if lse.E(x) < sd["E"] and flow_limit(lse, P, x) == i:
                    break
            else:
                raise RuntimeError("no admissible starting point")
            Ex = lse.E(x)
            a = rng.standard_normal(D)
            a *= 1e-7 / np.linalg.norm(a)
            for h in HS:
                G = lambda y: avf.G(x, y)
                DG = lambda y: avf.DyG(x, y)
                row = dict(theta=theta, pattern=i, rep=rep, h=h, E_x=Ex, margin=sd["E"] - Ex)
                yn, kn, okn = newton_step(x, h, G, DG, x.copy())
                ye, ke, oke = newton_step(x, h, G, DG, x - h * lse.grad(x))
                res = track(x, h, G, DG, a=a, ds0=0.01, ds_max=0.05)
                yc = res["root"]
                where = lambda y, ok: "failed" if not ok else ("same" if flow_limit(lse, P, y) == i else "other")
                st, depth, _, nev = chord_check(lse.E, x, yn, kb(x, yn), dmax=DMAX) if okn else ("none", 0, None, 0)
                pts = [w[:-1] for w in res["path"]]
                roots = [y for y, ok in ((yn, okn), (ye, oke), (yc, True)) if ok]
                row.update(
                    newton=where(yn, okn), newton_solves=int(kn),
                    euler=where(ye, oke), euler_solves=int(ke),
                    cont=where(yc, True), cont_solves=int(res["nsolves"]), cont_folds=int(res["nfold"]),
                    cont_ceiling_minus_Ex=float(polyline_ceiling(pts, lse.E, kb, m=1) - Ex),
                    cont_points=len(pts),
                    newton_is_cont=bool(okn and np.linalg.norm(yn - yc) < 1e-8),
                    chord=st, chord_depth=int(depth), chord_energies=int(nev),
                    crossing_predicted=bool(h * abs(sd["mu"]) > 2.0),
                    defect=float(max(abs(lse.E(y) - Ex + (y - x) @ (y - x) / h) for y in roots)))
                row["checked_solves"] = row["newton_solves"] + (0 if st == "certified" else row["cont_solves"])
                rows.append(row)
    print(f"total: {time.time()-t0:.1f}s")
    summary = []
    for theta in THETAS:
        for h in HS:
            sel = [r for r in rows if r["theta"] == theta and r["h"] == h]
            cert = [r for r in sel if r["chord"] == "certified"]
            summary.append(dict(
                theta=theta, h=h, cases=len(sel),
                newton_other=sum(r["newton"] == "other" for r in sel),
                newton_failed=sum(r["newton"] == "failed" for r in sel),
                euler_other=sum(r["euler"] == "other" for r in sel),
                euler_failed=sum(r["euler"] == "failed" for r in sel),
                cont_other=sum(r["cont"] == "other" for r in sel),
                crossing_predicted=sum(r["crossing_predicted"] for r in sel),
                chord_certified=len(cert),
                chord_certified_other=sum(r["newton"] != "same" for r in cert),
                chord_witness=sum(r["chord"] == "witness" for r in sel),
                chord_undecided=sum(r["chord"] == "undecided" for r in sel),
                chord_depth_max=max([r["chord_depth"] for r in cert], default=0),
                chord_energies_max=max([r["chord_energies"] for r in cert], default=0),
                ceiling_equal=sum(r["cont_ceiling_minus_Ex"] == 0.0 for r in sel),
                folds=sum(r["cont_folds"] > 0 for r in sel),
                newton_solves_median=float(np.median([r["newton_solves"] for r in sel])),
                cont_solves_median=float(np.median([r["cont_solves"] for r in sel])),
                checked_solves_mean=float(np.mean([r["checked_solves"] for r in sel])),
                cont_solves_mean=float(np.mean([r["cont_solves"] for r in sel]))))
            q = summary[-1]
            print(f"theta={theta:4.2f} h={h:5.1f}: Newton(0) other/failed {q['newton_other']:2d}/{q['newton_failed']:2d}  "
                  f"Newton(Euler) {q['euler_other']:2d}/{q['euler_failed']:2d}  continuation {q['cont_other']:2d}  of {q['cases']} "
                  f"(h|mu|>2: {q['crossing_predicted']:2d}) | chord certified {q['chord_certified']:2d} "
                  f"(not in basin {q['chord_certified_other']}) witness {q['chord_witness']:2d} undecided {q['chord_undecided']:2d} | "
                  f"ceiling=E(x) {q['ceiling_equal']:2d} folds {q['folds']:2d} | solves: Newton {q['newton_solves_median']:.0f}, "
                  f"continuation {q['cont_solves_mean']:.1f}, checked {q['checked_solves_mean']:.1f}")
    info["max_defect"] = float(max(r["defect"] for r in rows))
    info["max_ceiling_minus_Ex"] = float(max(r["cont_ceiling_minus_Ex"] for r in rows))
    print(f"max energy-identity defect {info['max_defect']:.2e}")
    return dict(info=info, summary=summary, rows=rows)


def figures(out):
    """Outcome of the two Newton iterations against h |mu_i|, one panel per starting guess."""
    from figstyle import plt, panel, C_IN, C_OUT
    mu = np.abs(np.array(out["info"]["saddle_eigenvalue"]))
    for key, fname in (("newton", "random_newton_zero.pdf"), ("euler", "random_newton_euler.pdf")):
        fig, ax = panel()
        for b, theta in enumerate(THETAS):
            sel = [r for r in out["rows"] if r["theta"] == theta]
            xs = np.array([r["h"] * mu[r["pattern"]] for r in sel])
            ys = np.array([b + 0.72 * ((r["pattern"] * REPS + r["rep"] + 0.5) / (N * REPS) - 0.5) for r in sel])
            st = np.array([r[key] for r in sel])
            for name, m, col, ms, mew in (("same", "o", C_IN, 2.0, 0.0), ("other", "s", C_OUT, 2.2, 0.0), ("failed", "x", "k", 3.6, 0.9)):
                k = st == name
                ax.plot(xs[k], ys[k], m, ms=ms, color=col, mew=mew, ls="none")
        ax.axvline(2.0, color="k", ls=":", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlim(0.5, 22.0)
        ax.set_xticks([0.5, 1, 2, 5, 10, 20])
        ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20"])
        ax.minorticks_off()
        ax.set_ylim(-0.55, len(THETAS) - 0.45)
        ax.set_yticks(range(len(THETAS)))
        ax.set_yticklabels([f"{t:g}" for t in THETAS])
        ax.set_xlabel("$h|\\mu_i|$")
        ax.set_ylabel("$\\theta$")
        fig.savefig("../figs/" + fname)
        plt.close(fig)


if __name__ == "__main__":
    out = run()
    with open("../out/exp7_random.json", "w") as fh:
        json.dump(out, fh, indent=1)
    figures(out)
