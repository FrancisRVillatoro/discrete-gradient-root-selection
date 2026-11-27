"""Section 8.7: Newton's method checked by the chord certificate, with continuation when the check fails.

For each single step of Sections 8.1, 8.2, 8.4 and 8.6 the checked solver applies Newton's method
from zero displacement and then Proposition 6.1 to the chord from x to the Newton root, with dyadic
refinement up to depth DMAX.  The chord is compared with the level E(x), which requires no knowledge
of the barrier, and, for comparison, with the separating level of each example.  The continuation
algorithm of Section 3.3 is run when the chord is not certified, and alone for comparison.
"""
import json
import numpy as np
from scipy.optimize import brentq
from dgcont import AVF, track, newton_step, chord_check
import exp1_twowell as s1
import exp2_ring as s2
import exp3_rotated as s3
from allen_cahn import AllenCahn, kink, step

DMAX = 6


def record(label, h, newton_ok, newton_solves, newton_well, free, level, cont_solves, cont_well):
    row = dict(example=label, h=h, newton_ok=bool(newton_ok), newton_solves=int(newton_solves), newton_well=newton_well,
               continuation_solves=int(cont_solves), continuation_well=cont_well)
    for name, (st, k, val, nev) in (("free", free), ("level", level)):
        row[name] = dict(certificate=st, depth=int(k), value=float(val), energy_evaluations=int(nev),
                         checked_solves=int(newton_solves + (0 if st == "certified" else cont_solves)),
                         checked_well=newton_well if st == "certified" else cont_well)
    print(f"{label:22s} h={h:6.2f} Newton: {newton_well:5s} ({newton_solves:3d} solves) | "
          + " | ".join(f"{name}: {row[name]['certificate']:9s} depth {row[name]['depth']} ({row[name]['energy_evaluations']:2d} energies, "
                       f"value {row[name]['value']:+.6f}), checked {row[name]['checked_solves']:4d}" for name in ("free", "level"))
          + f" | continuation {cont_solves:4d} solves, {cont_well}")
    return row


def scalar_cases():
    rows = []
    for label, beta, phi, x, h in [("scalar symmetric", 4.0, 0.0, -0.01, 1.0), ("scalar tilted", 4.0, -0.2, 0.08, 1.0)]:
        E, g, H = s1.energy(beta, phi)
        zs = 0.0 if phi == 0 else brentq(g, -0.5, 0.5)
        level = float(E(zs))
        avf = AVF(lambda z: g(z), lambda z: np.atleast_2d(H(z)), nq=32)
        X = np.atleast_1d(x)
        G = lambda y: avf.G(X, y)
        DG = lambda y: np.atleast_2d(avf.DyG(X, y))
        well = lambda y: "start" if np.sign(y[0] - zs) == np.sign(x - zs) else "other"
        yn, kn, okn = newton_step(X, h, G, DG, X.copy())
        d = yn - X
        Ef = lambda z: E(z[0])
        K = (beta - 1.0) * float(d @ d)
        cont = track(X, h, G, DG, ds0=0.01, ds_max=0.05)
        rows.append(record(label, h, okn, kn, well(yn), chord_check(Ef, X, yn, K, dmax=DMAX),
                           chord_check(Ef, X, yn, K, level=level, dmax=DMAX), cont["nsolves"], well(cont["root"])))
    return rows


def ring_cases():
    rows = []
    level = float(s2.E(np.array([0.0, np.sqrt(1.0 - 2.0 * s2.B / s2.A)])))
    th = 1.25
    rr = np.sqrt(1.0 - 2.0 * s2.B * np.sin(th) ** 2 / s2.A)
    x = rr * np.array([np.cos(th), np.sin(th)])
    well = lambda y: "start" if y[0] > 0 else "other"
    for h in [2.0, 5.0, 10.0, 20.0]:
        avf = AVF(s2.grad, s2.hess, nq=24)
        G = lambda y: avf.G(x, y)
        DG = lambda y: avf.DyG(x, y)
        yn, kn, okn = newton_step(x, h, G, DG, x.copy())
        K = s2.kbound(x, yn)
        cont = track(x, h, G, DG, a=1e-7 * np.array([0.6, -0.8]), ds0=0.01, ds_max=0.05)
        rows.append(record("ring valley", h, okn, kn, well(yn), chord_check(s2.E, x, yn, K, dmax=DMAX),
                           chord_check(s2.E, x, yn, K, level=level, dmax=DMAX), cont["nsolves"], well(cont["root"])))
    return rows


def rotated_cases():
    rows = []
    x = 0.03 * s3.a + 0.015 * s3.bv
    well = lambda y: "start" if (s3.a @ y) > 0 else "other"
    for h in [0.25, 1.0, 2.0, 4.0]:
        avf = AVF(s3.grad, s3.hess, nq=64)
        G = lambda y: avf.G(x, y)
        DG = lambda y: avf.DyG(x, y)
        yn, kn, okn = newton_step(x, h, G, DG, x.copy())
        d = yn - x
        K = max(0.0, s3.beta * float(s3.a @ d) ** 2 - float(d @ d))   # K_delta of (6.5) for the patterns +-a
        cont = track(x, h, G, DG, a=1e-7 * np.array([0.3, 0.9]), ds0=0.01, ds_max=0.05)
        rows.append(record("rotated two-well", h, okn, kn, well(yn), chord_check(s3.E, x, yn, K, dmax=DMAX),
                           chord_check(s3.E, x, yn, K, level=0.0, dmax=DMAX), cont["nsolves"], well(cont["root"])))
    return rows


def allencahn_cases():
    rows = []
    ac = AllenCahn()
    c = float(ac.E(kink(ac)))
    well = lambda Y: "start" if (ac.E(Y) < c and Y.mean() > 0) else "other"
    for name, U0 in [("Allen-Cahn cosine", 1.0 + 0.2 * np.cos(2 * np.pi * ac.xg)),
                     ("Allen-Cahn interface", np.tanh((ac.xg - 0.15) / (np.sqrt(2.0) * ac.eps)))]:
        for h in [10.0, 50.0, 200.0]:
            Y, k, ok, _ = step(ac, U0, h, "newton_x")
            d = Y - U0
            K = float(d @ d)
            Yc, kc, okc, _ = step(ac, U0, h, "cont")
            rows.append(record(name, h, ok, k, well(Y), chord_check(ac.E, U0, Y, K, dmax=DMAX),
                               chord_check(ac.E, U0, Y, K, level=c, dmax=DMAX), kc, well(Yc)))
    return rows


if __name__ == "__main__":
    rows = scalar_cases() + ring_cases() + rotated_cases() + allencahn_cases()
    with open("../out/exp5_hybrid.json", "w") as fh:
        json.dump(rows, fh, indent=1)
