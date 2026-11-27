"""Compare straight-segment sublevel tests on discrete-gradient steps.

Run python chord_test.py. Outputs chord_results.json next to this file.
All arithmetic, including energy enclosures, uses binary64. Floating-point
comparisons are experimental evidence rather than interval certificates.
"""
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq
from scipy.special import logsumexp


BETA = 6.0
TRANSVERSE_RADIUS = 0.37
DIMENSION = 16
H_VALUES = (0.75, 2.0, 5.0)
MAX_DEPTH = 10
SEED = 1907


def scalar_energy(q):
    return 0.5 * q**2 - np.logaddexp(BETA * q, -BETA * q) / BETA


def scalar_secant(q0, q1):
    if abs(q1 - q0) < 1.0e-6:
        mid = 0.5 * (q0 + q1)
        # The omitted symmetric Taylor term is below binary64 root tolerance.
        return mid - np.tanh(BETA * mid) + (
            BETA**2 * (1.0 - np.tanh(BETA * mid)**2)
            * np.tanh(BETA * mid) * (q1 - q0)**2 / 12.0
        )
    return (scalar_energy(q1) - scalar_energy(q0)) / (q1 - q0)


def scalar_roots(q0, h):
    def equation(q1):
        return q1 - q0 + h * scalar_secant(q0, q1)

    grid = np.linspace(-3.0, 3.0, 1201)
    values = np.array([equation(t) for t in grid])
    intervals = np.flatnonzero(values[:-1] * values[1:] < 0.0)
    roots = [brentq(equation, grid[i], grid[i + 1], xtol=1.0e-14) for i in intervals]
    roots.extend(grid[values == 0.0].tolist())
    return sorted(roots)


def transverse_energy(u, patterns):
    return 0.5 * (u @ u) - logsumexp(BETA * (patterns @ u)) / BETA


def transverse_avf(u0, h, patterns, nodes, weights):
    # One Newton solve for the transverse block, shared by all scalar roots.
    dim = len(u0)
    eye = np.eye(dim)
    u1 = u0.copy()
    for iteration in range(16):
        points = u0 + nodes[:, None] * (u1 - u0)
        logits = BETA * (points @ patterns.T)
        probs = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
        means = probs @ patterns
        avf = weights @ (points - means)
        residual = u1 - u0 + h * avf
        if np.linalg.norm(residual) < 5.0e-14:
            return u1, avf
        moment = np.einsum("qi,ij,ik->qjk", probs, patterns, patterns)
        covariance = moment - means[:, :, None] * means[:, None, :]
        jacobian = eye + h * np.einsum(
            "q,qij->ij", nodes * weights, eye - BETA * covariance
        )
        u1 -= np.linalg.solve(jacobian, residual)
    raise ArithmeticError("Transverse AVF Newton iteration did not converge")


def max_upper(e0, e1, curvature_length_sq):
    if curvature_length_sq <= 0.0:
        return max(e0, e1)
    vertex = np.clip(0.5 + (e1 - e0) / curvature_length_sq, 0.0, 1.0)
    return (
        (1 - vertex) * e0 + vertex * e1
        + 0.5 * curvature_length_sq * vertex * (1 - vertex)
    )


def classify(x, y, energy, c, nu):
    diff = y - x
    scale = nu * (diff @ diff)
    values = np.array([energy(x), energy(y)])
    for depth in range(MAX_DEPTH + 1):
        if np.max(values) >= c:
            return "sample_above", depth
        subdivisions = 2**depth
        envelopes = [max_upper(a, b, scale / subdivisions**2)
                     for a, b in zip(values[:-1], values[1:])]
        if max(envelopes) < c:
            return "below", depth
        if depth < MAX_DEPTH:
            refined = np.empty(2 * subdivisions + 1)
            refined[::2] = values
            refined[1::2] = [
                energy(x + ((j + 0.5) / subdivisions) * diff)
                for j in range(subdivisions)
            ]
            values = refined
    return "unresolved", MAX_DEPTH


def run():
    rng = np.random.default_rng(SEED)
    transverse = rng.normal(size=(8, DIMENSION - 1))
    transverse *= TRANSVERSE_RADIUS / np.linalg.norm(transverse, axis=1, keepdims=True)
    transverse = np.vstack((transverse, -transverse))
    patterns = np.vstack([np.column_stack((np.full(len(transverse), s), transverse))
                          for s in (-1.0, 1.0)])
    diameter = np.max(np.linalg.norm(patterns[:, None] - patterns[None, :], axis=2))
    global_nu = max(0.0, BETA * diameter**2 / 4.0 - 1.0)
    nodes, weights = leggauss(24)
    nodes, weights = (1.0 + nodes) / 2.0, weights / 2.0

    def energy(z):
        return scalar_energy(z[0]) + transverse_energy(z[1:], transverse)

    c = -np.log(len(patterns)) / BETA
    rows = []
    for q0 in (0.08, 0.27, 0.68, 1.22):
        well_depth = scalar_energy(0.0) - scalar_energy(q0)
        for fraction in (0.002, 0.05, 0.5):
            unit = rng.normal(size=DIMENSION - 1)
            unit /= np.linalg.norm(unit)
            target = c - fraction * well_depth
            radius = brentq(
                lambda r: scalar_energy(q0) + transverse_energy(r * unit, transverse) - target,
                0.0, 4.0, xtol=5.0e-15,
            )
            x = np.r_[q0, radius * unit]
            for h in H_VALUES:
                u1, avf = transverse_avf(x[1:], h, transverse, nodes, weights)
                for q1 in scalar_roots(q0, h):
                    y = np.r_[q1, u1]
                    diff = y - x
                    length = np.linalg.norm(diff)
                    width = np.ptp(patterns @ (diff / length))
                    directional_nu = max(0.0, BETA * width**2 / 4.0 - 1.0)
                    dg = np.r_[scalar_secant(q0, q1), avf]
                    rows.append(dict(
                        q0=q0, q1=q1, x=x.tolist(), y=y.tolist(), gap_fraction=fraction, h=h,
                        crossing=bool(q1 < 0.0), start_margin=c - energy(x),
                        end_margin=c - energy(y), directional_nu=directional_nu,
                        abs_chain_defect=abs(energy(y) - energy(x) - dg @ diff),
                        step_residual=float(np.linalg.norm(diff + h * dg)),
                        global_test=classify(x, y, energy, c, global_nu),
                        directional_test=classify(x, y, energy, c, directional_nu),
                    ))
    summary = {}
    for test in ("global_test", "directional_test"):
        histogram = {}
        for row in rows:
            label, level = row[test]
            by_depth = histogram.setdefault(label, {})
            by_depth[level] = by_depth.get(level, 0) + 1
        summary[test] = histogram
    result = dict(
        seed=SEED, dimension=DIMENSION, patterns=len(patterns), beta=BETA,
        transverse_radius=TRANSVERSE_RADIUS, step_sizes=H_VALUES,
        global_nu=global_nu, chosen_level=c, candidate_steps=len(rows),
        transverse_patterns=transverse.tolist(),
        across_hyperplane=sum(r["crossing"] for r in rows),
        same_sign=sum(not r["crossing"] for r in rows),
        max_chain_defect=max(r["abs_chain_defect"] for r in rows),
        max_step_residual=max(r["step_residual"] for r in rows),
        min_start_margin=min(r["start_margin"] for r in rows),
        min_end_margin=min(r["end_margin"] for r in rows), summary=summary, rows=rows,
    )
    output = Path(__file__).resolve().with_name("chord_results.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"rows", "transverse_patterns"}}, indent=2))


if __name__ == "__main__":
    run()
