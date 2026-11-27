"""Section 8.6: Allen-Cahn gradient flow.

Scalar part: E(u) = (u^2-1)^2/4, AVF step from u0, threshold h = 4/(u0^2-2).
Semi-discrete part: d = 256 cells on [0,1], eps = 0.05, Neumann conditions,
E(U) = (eps^2/(2 dx^2)) sum (U_{j+1}-U_j)^2 + sum (U_j^2-1)^2/4, flow U' = -grad E(U).
"""
import json
import time
import numpy as np
from scipy.optimize import brentq
from allen_cahn import AllenCahn, kink, step, classify
from dgcont import polyline_ceiling
from figstyle import plt, panel, C_MAIN, C_IN, C_OUT, C_GREY


def scalar_root(u0, h):
    f = lambda y: (y - u0) / h + 0.25 * (y + u0) * (y * y + u0 * u0 - 2.0)
    return brentq(f, -10.0, 10.0, xtol=1e-15)


def integrate(ac, U0, hfun, mfun, gtol=1e-8, max_steps=6000, keep=False):
    U = U0.copy()
    n, solves, t = 0, 0, 0.0
    hist = [(0.0, float(U.mean()), float(ac.E(U)))]
    first = None
    nswitch = None
    while n < max_steps:
        if np.linalg.norm(ac.grad(U), np.inf) < gtol:
            break
        h, m = hfun(n, U), mfun(n, U)
        if m == "cont" and nswitch is None:
            nswitch = n
        Y, k, ok, res = step(ac, U, h, m)
        if not ok:
            raise RuntimeError("step failed")
        solves += k
        if n == 0:
            first = Y.copy()
        U = Y
        t += h
        n += 1
        hist.append((t, float(U.mean()), float(ac.E(U))))
    return dict(U=U, steps=n, solves=solves, hist=np.array(hist), first=first, nswitch=nswitch, t=t)


if __name__ == "__main__":
    out = {}
    # scalar threshold
    sc = []
    for u0 in [2.0, 3.0, 5.0]:
        hc = 4.0 / (u0 * u0 - 2.0)
        for fac in [0.9, 1.0, 1.1]:
            y = scalar_root(u0, fac * hc)
            sc.append(dict(u0=u0, hc=hc, h=fac * hc, root=y))
    sc.append(dict(u0=3.0, hc=4.0 / 7.0, h=1.0, root=scalar_root(3.0, 1.0)))
    out["scalar"] = sc
    for r in sc:
        print(f"scalar u0={r['u0']} hc={r['hc']:.6f} h={r['h']:.6f} root={r['root']:+.12f}")

    ac = AllenCahn()
    c = float(ac.E(kink(ac)))
    out["c"] = c
    print(f"semi-discrete: d={ac.d} eps={ac.eps} barrier c={c:.6f}")

    # below the barrier: one large step
    kb = lambda p, q: float(np.sum((q - p) ** 2))
    below = []
    for name, U0 in [("cos", 1.0 + 0.2 * np.cos(2 * np.pi * ac.xg)),
                     ("interface", np.tanh((ac.xg - 0.15) / (np.sqrt(2.0) * ac.eps)))]:
        for h in [10.0, 50.0, 200.0]:
            row = dict(case=name, E0=float(ac.E(U0)), h=h)
            for m in ["newton_x", "newton_ee", "cont"]:
                Y, k, ok, res = step(ac, U0, h, m)
                row[m] = dict(ok=bool(ok), solves=int(k), E=float(ac.E(Y)), cls=classify(ac, Y, c), mean=float(Y.mean()))
                if res is not None:
                    pts = [w[:-1] for w in res["path"]]
                    row[m]["ceiling_minus_E0"] = float(polyline_ceiling(pts, ac.E, kb, m=1) - ac.E(U0))
                    row[m]["folds"] = int(res["nfold"])
            below.append(row)
            print(f"below {name} E0={row['E0']:.4f} h={h:6.1f} " + " | ".join(
                f"{m}: {row[m]['cls']} E={row[m]['E']:.4f} solves={row[m]['solves']}" for m in ["newton_x", "newton_ee", "cont"])
                  + f" | ceiling-E0={row['cont']['ceiling_minus_E0']:.2e}")
    out["below"] = below

    # above the barrier
    U0 = 2.5 + 0.5 * np.cos(2 * np.pi * ac.xg)
    out["E0_above"] = float(ac.E(U0))
    out["hc_pointwise"] = 4.0 / (U0.max() ** 2 - 2.0)
    runs = {}
    for h in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0]:
        t0 = time.time()
        r = integrate(ac, U0, lambda n, U, h=h: h, lambda n, U: "newton_x" if h < 2.0 else "cont")
        neg = int(np.sum(r["first"] < 0.0))
        runs[f"const_{h}"] = r
        out[f"const_{h}"] = dict(steps=r["steps"], solves=r["solves"], neg_after_first=neg, final=classify(ac, r["U"], c),
                                 final_mean=float(r["U"].mean()), seconds=time.time() - t0)
        print(f"above const h={h}: negative nodes after step 1 = {neg:3d}, steps={r['steps']}, final {classify(ac, r['U'], c)} (mean {r['U'].mean():+.4f})")
    onset = {}
    for h in [0.56, 0.57, 0.575, 0.58, 0.59]:
        Y, k, ok, _ = step(ac, U0, h, "newton_x")
        onset[str(h)] = int(np.sum(Y < 0.0))
        print(f"above first step h={h}: negative nodes = {onset[str(h)]}")
    out["onset"] = onset
    for h2 in [2.0, 5.0, 20.0]:
        state = {"phase": 1}

        def hfun(n, U, h2=h2, state=state):
            if state["phase"] == 1 and ac.E(U) < 0.5 * c:
                state["phase"] = 2
            return 0.5 if state["phase"] == 1 else h2

        def mfun(n, U, state=state):
            return "newton_x" if state["phase"] == 1 else "cont"
        t0 = time.time()
        r = integrate(ac, U0, hfun, mfun)
        runs[f"two_{h2}"] = r
        out[f"two_{h2}"] = dict(steps=r["steps"], solves=r["solves"], nswitch=r["nswitch"], final=classify(ac, r["U"], c),
                                final_mean=float(r["U"].mean()), seconds=time.time() - t0)
        print(f"above two-regime 0.5 -> {h2}: switch at step {r['nswitch']}, steps={r['steps']}, solves={r['solves']}, final {classify(ac, r['U'], c)}")

    with open("../out/exp4_allencahn.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)

    # figures
    fig, ax = panel()
    ax.plot(ac.xg, U0, color=C_GREY)
    ax.plot(ac.xg, runs["const_0.5"]["first"], color=C_IN)
    ax.plot(ac.xg, runs["const_2.0"]["first"], color=C_OUT)
    ax.axhline(0.0, color="k", lw=0.6, ls=":")
    ax.text(0.5, 2.15, "$u^0$", ha="center", va="bottom", color=C_GREY)
    ax.text(0.5, 1.1, "$u^1$, $h=0.5$", ha="center", va="bottom", color=C_IN)
    ax.text(0.5, -0.45, "$u^1$, $h=2$", ha="center", va="top", color=C_OUT)
    ax.set_xlabel("$X$")
    ax.set_ylabel("$u_j$")
    fig.savefig("../figs/allencahn_first_step.pdf")
    plt.close(fig)

    fig, ax = panel()
    for key, col, ls, lab in [("const_0.5", C_IN, "-", "$h=0.5$"), ("const_1.0", "#c77c02", "-", "$h=1$"),
                              ("const_2.0", C_OUT, "-", "$h=2$"), ("two_20.0", C_MAIN, "--", "two regimes")]:
        H = runs[key]["hist"]
        ax.plot(H[1:, 0], H[1:, 1], color=col, ls=ls, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("$t$")
    ax.set_ylabel("mean of $u$")
    ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.64))
    fig.savefig("../figs/allencahn_mean.pdf")
    plt.close(fig)
