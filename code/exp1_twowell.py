"""Section 8.1: scalar two-well energies, roots of one AVF step, homotopy zero set, heuristics.

E(z) = z^2/2 - log(cosh(beta z + phi))/beta, beta = 4 (nu = 3), step h = 1.
Symmetric case phi = 0, x = -0.01; tilted case phi = -0.2, x = 0.08.
"""
import json
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from dgcont import AVF, track, newton_step
from figstyle import plt, panel, C_MAIN, C_IN, C_OUT, C_GREY


def logcosh(u):
    u = np.abs(u)
    return u + np.log1p(np.exp(-2.0 * u)) - np.log(2.0)


def energy(beta, phi):
    E = lambda z: 0.5 * z * z - logcosh(beta * z + phi) / beta
    g = lambda z: z - np.tanh(beta * z + phi)
    H = lambda z: 1.0 - beta / np.cosh(beta * z + phi) ** 2
    return E, g, H


def study(beta, phi, x, h):
    E, g, H = energy(beta, phi)
    zs = 0.0 if phi == 0 else brentq(g, -0.5, 0.5)
    f = lambda y: (y - x) / h + (E(y) - E(x)) / (y - x)
    ys = np.linspace(-2.5, 2.5, 500001)
    ys = ys[np.abs(ys - x) > 1e-9]
    fv = f(ys)
    roots = [brentq(f, ys[i], ys[i + 1], xtol=1e-15)
             for i in np.where(np.sign(fv[:-1]) != np.sign(fv[1:]))[0]]
    avf = AVF(lambda z: g(z), lambda z: np.atleast_2d(H(z)), nq=32)
    X = np.atleast_1d(x)
    G = lambda y: avf.G(X, y)
    DG = lambda y: np.atleast_2d(avf.DyG(X, y))
    cont = track(X, h, G, DG, ds0=0.01, ds_max=0.05)
    yn, _, okn = newton_step(X, h, G, DG, X.copy())
    ye, _, oke = newton_step(X, h, G, DG, X - h * g(X))
    same = lambda y: bool(np.sign(y - zs) == np.sign(x - zs))
    # birth of the roots across the saddle: minimum of t(y) on the far component of {E < E(x)}
    tfun = lambda y: (y - x) ** 2 / (h * (E(x) - E(y)))
    far = [r for r in roots if not same(r)]
    tf = None
    if far:
        lo, hi = min(far), max(far)
        m = minimize_scalar(tfun, bounds=(lo, hi), method="bounded", options={"xatol": 1e-12})
        tf = (float(m.fun), float(m.x))
    out = dict(beta=beta, phi=phi, x=x, h=h, nu=beta - 1, saddle=zs, E_saddle=float(E(zs)), E_x=float(E(x)),
               roots=[dict(y=r, E=float(E(r)), decrease=float(E(x) - E(r)), dist=abs(r - x), same=same(r)) for r in roots],
               nearest=min(roots, key=lambda r: abs(r - x)), maxdecrease=max(roots, key=lambda r: abs(r - x)),
               newton_x=float(yn[0]), newton_x_ok=okn, newton_ee=float(ye[0]), newton_ee_ok=oke,
               cont=float(cont["root"][0]), cont_folds=cont["nfold"], cont_solves=cont["nsolves"], birth=tf)
    for key in ["nearest", "maxdecrease", "newton_x", "newton_ee", "cont"]:
        out[key + "_same"] = same(out[key])
    return out, cont


def figures(beta, x, h, roots, cont):
    E, g, H = energy(beta, 0.0)
    z = np.linspace(-1.4, 1.4, 2001)
    fig, ax = panel()
    ax.plot(z, E(z), color=C_MAIN)
    ax.axhline(0.0, color=C_GREY, ls=":", lw=0.8)
    ax.plot([x], [E(x)], "o", ms=4, color="k", zorder=5)
    ax.annotate("$x$", (x, E(x)), xytext=(-12, 6), textcoords="offset points")
    mk = {0: ("o", C_IN), 1: ("^", C_OUT), 2: ("s", C_OUT)}
    for i, r in enumerate(sorted(roots)):
        m, col = mk[i]
        ax.plot([r], [E(r)], m, ms=5, mfc=col if i == 0 else "white", mec=col, zorder=5)
        ax.annotate(f"$y_{i+1}$", (r, E(r)), xytext=(4, -10), textcoords="offset points", color=col)
    ax.set_xlabel("$z$")
    ax.set_ylabel("$E(z)$")
    ax.set_xlim(-1.4, 1.4)
    fig.savefig("../figs/twowell_energy.pdf")
    plt.close(fig)

    fig, ax = panel()
    y = np.linspace(-1.4, 1.4, 200001)
    y = y[np.abs(y - x) > 1e-12]
    below = E(y) < E(x)
    t = np.full_like(y, np.nan)
    t[below] = (y[below] - x) ** 2 / (h * (E(x) - E(y[below])))
    t[t > 1.35] = np.nan
    ax.plot(y, t, color=C_GREY, lw=1.0)
    path = cont["path"]
    ax.plot(path[:, 0], path[:, 1], color=C_IN, lw=2.0, alpha=0.8)
    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    ax.plot([x], [0.0], "o", ms=4, color="k")
    for i, r in enumerate(sorted(roots)):
        m, col = mk[i]
        ax.plot([r], [1.0], m, ms=5, mfc=col if i == 0 else "white", mec=col, zorder=5)
    ax.set_xlabel("$y$")
    ax.set_ylabel("$\\lambda$")
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(0.0, 1.35)
    fig.savefig("../figs/twowell_homotopy.pdf")
    plt.close(fig)


if __name__ == "__main__":
    res = {}
    sym, cont_sym = study(4.0, 0.0, -0.01, 1.0)
    tilt, _ = study(4.0, -0.2, 0.08, 1.0)
    res["symmetric"] = sym
    res["tilted"] = tilt
    figures(4.0, -0.01, 1.0, [r["y"] for r in sym["roots"]], cont_sym)
    with open("../out/exp1_twowell.json", "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    for name, r in res.items():
        print(name, f"E(x)={r['E_x']:.6e} saddle={r['saddle']:.6f} E(saddle)={r['E_saddle']:.6e} birth={r['birth']}")
        for rr in r["roots"]:
            print(f"   y={rr['y']:+.6f} E={rr['E']:+.6f} decrease={rr['decrease']:.6f} same={rr['same']}")
        for key in ["nearest", "maxdecrease", "newton_x", "newton_ee", "cont"]:
            print(f"   {key:12s} {r[key]:+.6f} same={r[key + '_same']}")
        print(f"   continuation folds={r['cont_folds']} solves={r['cont_solves']}")
