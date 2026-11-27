"""Section 8.4: coupled AVF step for the rotated two-pattern energy of the coordinate-sweep test.

E(z) = |z|^2/2 - log(cosh(8 a.z))/8, a = (4/5, 3/5), b = (-3/5, 4/5), z0 = 0.03 a + 0.015 b, h = 4.
"""
import json
import numpy as np
from dgcont import AVF, track, newton_step

a = np.array([0.8, 0.6])
bv = np.array([-0.6, 0.8])
beta = 8.0


def logcosh(u):
    u = abs(u)
    return u + np.log1p(np.exp(-2.0 * u)) - np.log(2.0)


E = lambda z: 0.5 * z @ z - logcosh(beta * (a @ z)) / beta
grad = lambda z: z - np.tanh(beta * (a @ z)) * a
hess = lambda z: np.eye(2) - beta / np.cosh(beta * (a @ z)) ** 2 * np.outer(a, a)

if __name__ == "__main__":
    x = 0.03 * a + 0.015 * bv
    out = {"E_x": float(E(x))}
    for h in [0.25, 1.0, 2.0, 4.0]:
        avf = AVF(grad, hess, nq=64)
        G = lambda y: avf.G(x, y)
        DG = lambda y: avf.DyG(x, y)
        cont = track(x, h, G, DG, a=1e-7 * np.array([0.3, 0.9]), ds0=0.01, ds_max=0.05)
        y = cont["root"]
        yn, kn, okn = newton_step(x, h, G, DG, x.copy())
        out[str(h)] = dict(cont_q=float(a @ y), cont_E=float(E(y)), cont_solves=cont["nsolves"], cont_folds=cont["nfold"],
                           newton_q=float(a @ yn), newton_E=float(E(yn)), newton_ok=okn, newton_solves=kn)
        print(f"h={h}: continuation q={a@y:+.6f} E={E(y):+.6f} solves={cont['nsolves']} folds={cont['nfold']} | "
              f"Newton from zero displacement q={a@yn:+.6f} E={E(yn):+.6f} its={kn} ok={okn}")
    with open("../out/exp3_rotated.json", "w") as fh:
        json.dump(out, fh, indent=1)
