"""Section 8.2: ring valley E(z) = (A/4)(|z|^2 - 1)^2 + b z_2^2 with A = 10, b = 1.

Minima (+-1, 0), saddles (0, +-r2) with r2^2 = 1 - 2b/A and level c = 0.9.
Start on the radial minimum at angle 1.25 in the right well; one AVF step of size h.
"""
import json
import numpy as np
from numpy.polynomial import polynomial as P
from dgcont import AVF, track, newton_step, polyline_ceiling, bk
from figstyle import plt, panel, C_MAIN, C_IN, C_OUT, C_GREY

A, B = 10.0, 1.0


def E(z):
    return 0.25 * A * (z @ z - 1.0) ** 2 + B * z[1] ** 2


def grad(z):
    return A * (z @ z - 1.0) * z + np.array([0.0, 2.0 * B * z[1]])


def hess(z):
    return A * (z @ z - 1.0) * np.eye(2) + 2.0 * A * np.outer(z, z) + np.diag([0.0, 2.0 * B])


def kbound(p, q):
    """Exact K for the quartic polynomial s -> E(p + s(q-p)): minus the minimum of its second derivative."""
    s = np.linspace(0.0, 1.0, 5)
    c = P.polyfit(s, [E(p + si * (q - p)) for si in s], 4)
    c2 = P.polyder(c, 2)
    cand = [0.0, 1.0]
    if len(c2) == 3 and c2[2] != 0.0:
        sv = -c2[1] / (2.0 * c2[2])
        if 0.0 < sv < 1.0:
            cand.append(sv)
    return max(0.0, -min(P.polyval(sv, c2) for sv in cand))


def chord_depth(p, q, level, kmax=12):
    """Smallest dyadic m = 2^k for which Proposition 6.1 certifies the chord below `level`."""
    K = kbound(p, q)
    for k in range(kmax + 1):
        m = 2 ** k
        Es = [E(p + (j / m) * (q - p)) for j in range(m + 1)]
        U = max(bk(Es[j], Es[j + 1], K / m**2) for j in range(m))
        if U < level:
            return k, U
        if max(Es) >= level:
            return -k, max(Es)   # witness found at depth k
    return None, U


def run(h, th=1.25):
    r2 = np.sqrt(1.0 - 2.0 * B / A)
    c = E(np.array([0.0, r2]))
    rr = np.sqrt(1.0 - 2.0 * B * np.sin(th) ** 2 / A)
    x = rr * np.array([np.cos(th), np.sin(th)])
    avf = AVF(grad, hess, nq=24)
    G = lambda y: avf.G(x, y)
    DG = lambda y: avf.DyG(x, y)
    cont = track(x, h, G, DG, a=1e-7 * np.array([0.6, -0.8]), ds0=0.01, ds_max=0.05)
    y = cont["root"]
    pts = [w[:2] for w in cont["path"]]
    ceil = polyline_ceiling(pts, E, kbound, m=1)
    depth, Ud = chord_depth(x, y, c)
    yn, _, okn = newton_step(x, h, G, DG, x.copy())
    ye, _, oke = newton_step(x, h, G, DG, x - h * grad(x))
    roots = []
    for z1 in np.linspace(-1.5, 1.5, 31):
        for z2 in np.linspace(-1.5, 1.5, 31):
            yy, _, ok = newton_step(x, h, G, DG, np.array([z1, z2]), maxit=40)
            if ok and not any(np.linalg.norm(yy - q) < 1e-7 for q in roots):
                roots.append(yy)
    well = lambda z: "right" if z[0] > 0 else "left"
    out = dict(h=h, c=float(c), x=x.tolist(), E_x=float(E(x)), cont=y.tolist(), cont_E=float(E(y)),
               cont_well=well(y), folds=cont["nfold"], solves=cont["nsolves"], npts=len(pts),
               polyline_ceiling=float(ceil), chord_bound_depth0=float(bk(E(x), E(y), kbound(x, y))),
               chord_depth=depth, chord_U=float(Ud),
               newton_x=yn.tolist(), newton_x_ok=okn, newton_x_well=well(yn), newton_x_E=float(E(yn)),
               newton_ee=ye.tolist(), newton_ee_ok=oke, newton_ee_well=well(ye),
               roots=[dict(y=q.tolist(), E=float(E(q)), well=well(q)) for q in roots])
    return out, cont, roots, x


def figure(out, cont, roots, x):
    c = out["c"]
    g = np.linspace(-1.5, 1.5, 601)
    Z1, Z2 = np.meshgrid(g, g)
    EE = 0.25 * A * (Z1**2 + Z2**2 - 1.0) ** 2 + B * Z2**2
    fig, ax = panel()
    ax.contourf(Z1, Z2, EE, levels=[-1.0, c], colors=["#dbe7f3"])
    ax.contour(Z1, Z2, EE, levels=[0.1, 0.3, 0.5, 0.7, 1.5, 2.0, 3.0], colors=[C_GREY], linewidths=0.5)
    ax.contour(Z1, Z2, EE, levels=[c], colors=[C_MAIN], linewidths=0.9)
    path = cont["path"]
    ax.plot(path[:, 0], path[:, 1], color=C_IN, lw=2.0, alpha=0.85)
    y = np.array(out["cont"])
    ax.plot([x[0], y[0]], [x[1], y[1]], color=C_IN, ls="--", lw=0.9)
    ax.plot([x[0]], [x[1]], "o", ms=4, color="k")
    ax.plot([y[0]], [y[1]], "o", ms=5, color=C_IN)
    for q in roots:
        if np.linalg.norm(q - y) > 1e-6:
            ax.plot([q[0]], [q[1]], "s", ms=4.5, mfc="white", mec=C_OUT)
    yn = np.array(out["newton_x"])
    ax.plot([yn[0]], [yn[1]], "x", ms=6, mew=1.4, color=C_OUT)
    ax.set_aspect("equal")
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    fig.savefig("../figs/ring_valley.pdf")
    plt.close(fig)


if __name__ == "__main__":
    results = []
    for h in [2.0, 5.0, 10.0, 20.0]:
        out, cont, roots, x = run(h)
        results.append(out)
        if h == 10.0:
            figure(out, cont, roots, x)
        print(f"h={h:5.1f} E(x)={out['E_x']:.6f} c={out['c']:.6f} roots={len(roots)} "
              f"cont={np.round(out['cont'], 6)} E={out['cont_E']:.6f} well={out['cont_well']} folds={out['folds']} solves={out['solves']}")
        print(f"        polyline ceiling={out['polyline_ceiling']:.6f} chord depth0 bound={out['chord_bound_depth0']:.4f} "
              f"chord certified at depth {out['chord_depth']} (U={out['chord_U']:.6f})")
        print(f"        Newton zero displacement: {np.round(out['newton_x'], 6)} well={out['newton_x_well']} E={out['newton_x_E']:.6f} "
              f"| Newton explicit predictor: {np.round(out['newton_ee'], 6)} well={out['newton_ee_well']}")
        for q in out["roots"]:
            print(f"           root {np.round(q['y'], 6)} E={q['E']:.6f} {q['well']}")
    with open("../out/exp2_ring.json", "w") as fh:
        json.dump(results, fh, indent=1, default=float)
