"""Section 8.6: minimum of the discretized Allen-Cahn energy on the hyperplane of zero mean.

Every path from the constant state 1 to -1 crosses this hyperplane.  Sorting the entries of a state
does not increase the gradient term and leaves the potential term and the mean unchanged, so the
minimum on the hyperplane is attained at a monotone state.  Minimization by L-BFGS on the hyperplane
starts from random monotone states and alternates with sorting until sorting no longer lowers E.
"""
import json
import numpy as np
from scipy.optimize import minimize
from allen_cahn import AllenCahn, kink

NTRIALS = 50


def proj(v):
    return v - v.mean()


if __name__ == "__main__":
    ac = AllenCahn()
    K0 = kink(ac)
    c = float(ac.E(K0))

    def fun(v):
        u = proj(v)
        return ac.E(u), proj(ac.grad(u))

    rng = np.random.default_rng(1)
    res = []
    for trial in range(NTRIALS):
        u = np.sort(rng.uniform(-1.5, 1.5, ac.d))
        if trial % 2 == 1:
            u = u[::-1]
        u = proj(u)
        for _ in range(20):
            r = minimize(fun, u, jac=True, method="L-BFGS-B",
                         options=dict(maxiter=50000, maxcor=30, gtol=1e-11, ftol=1e-16))
            u = proj(r.x)
            us = np.sort(u) if u[-1] >= u[0] else np.sort(u)[::-1]
            if ac.E(us) < ac.E(u) - 1e-12:
                u = us
                continue
            break
        dist = min(np.max(np.abs(u - K0)), np.max(np.abs(u + K0)))
        res.append(dict(trial=trial, E=float(ac.E(u)), dist_to_interface=float(dist),
                        projected_gradient=float(np.linalg.norm(proj(ac.grad(u)), np.inf))))
    Es = np.array([r["E"] for r in res])
    out = dict(c=c, ntrials=NTRIALS, Emin=float(Es.min()), Emax=float(Es.max()),
               max_dist=float(max(r["dist_to_interface"] for r in res)),
               max_projected_gradient=float(max(r["projected_gradient"] for r in res)), trials=res)
    print(f"c = {c:.10f}; minimum on the hyperplane over {NTRIALS} starts: E in [{Es.min():.10f}, {Es.max():.10f}], "
          f"max distance to the interface {out['max_dist']:.2e}, max projected gradient {out['max_projected_gradient']:.2e}")
    with open("../out/exp6_barrier.json", "w") as fh:
        json.dump(out, fh, indent=1)
