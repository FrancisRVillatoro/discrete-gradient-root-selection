"""Reproduce the scalar branch experiment and the eight-dimensional AVF test.

Run: python avf_experiments.py [output.json]
Dependencies: Python 3, NumPy and SciPy. Calculations use binary64.
"""
import json
import platform
import sys
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import logsumexp


def well_energy(x, beta=4.0):
    # log(cosh u) = log(1+2*sinh(u/2)^2), including near u=0.
    return x**2 / 2 - np.log1p(2 * np.sinh(beta * x / 2)**2) / beta


def secant(x, y, beta=4.0):
    delta = beta * (y - x)
    if delta == 0:
        return x - np.tanh(beta * x)
    if abs(delta) < 0.25:
        increment = 2 * np.sinh(delta / 2)**2 + np.tanh(beta * x) * np.sinh(delta)
        return (x + y) / 2 - np.log1p(increment) / delta
    return (well_energy(y, beta) - well_energy(x, beta)) / (y - x)


def well_energy_upper(x):
    # Rational upper bound at beta=4. Truncate the positive cosh series,
    # then use log(1+v) >= 2*(z+z^3/3+z^5/5), z=v/(2+v).
    u = 4 * x
    v = u**2 / 2 + u**4 / 24 + u**6 / 720
    z = v / (2 + v)
    return x**2 / 2 - (z + z**3/3 + z**5/5) / 2


def scalar_chord_check(x, y, max_depth=12):
    # Floating endpoints are interpreted as their exact binary rationals.
    # The certificate verifies their segment, independently of the root solve.
    def check(left, right, a, b, depth):
        q = 3 * (right - left)**2 / 2
        t = min(1, max(0, (q + b - a) / (2*q)))
        if a + t * (b - a + q * (1 - t)) < 0:
            return True, 0
        if depth == max_depth:
            return False, 0
        mid = (left + right) / 2
        e_mid = well_energy_upper(mid)
        left_ok, n_left = check(left, mid, a, e_mid, depth + 1)
        right_ok, n_right = check(mid, right, e_mid, b, depth + 1)
        return left_ok and right_ok, 1 + n_left + n_right

    x, y = Fraction(x), Fraction(y)
    return check(x, y, well_energy_upper(x), well_energy_upper(y), 0)


def scalar_experiment():
    nu = 3.0
    # Predeclared disjoint brackets. At supercritical h the table lists the
    # roots in these brackets, without an exhaustive root-count assumption.
    cases = [
        (-1e-5, 1.8, [(-0.3, 0.0)]),
        (-1e-5, 2.0, [(-0.3, 0.0)]),
        (-1e-5, 2.02, [(-0.3, 0.0), (0.0, 0.01), (0.02, 0.2)]),
        (-0.01, 3.0, [(-0.7, -0.2), (0.0, 0.1), (0.2, 0.6)]),
        (-0.25, 3.0, [(-1.5, -0.25)]),
    ]
    rows = []
    for x, hnu, brackets in cases:
        h = hnu / nu
        for left, right in brackets:
            def residual(y):
                return (y - x) / h + secant(x, y)

            assert residual(left) * residual(right) < 0
            y = brentq(residual, left, right, xtol=1e-15, rtol=1e-15)
            energy_change = well_energy(y) - well_energy(x)
            margin = -well_energy(x)
            allowance = 0.5 * nu * (y - x)**2 * max(0.0, 0.5 - 1/hnu)**2
            defect = abs(energy_change + (y - x)**2 / h)
            assert energy_change < 0 and margin > 0 and defect < 2e-14
            certified = allowance < margin
            if certified:
                assert y < 0
            chord_ok, midpoints = scalar_chord_check(x, y)
            assert chord_ok == (y < 0)
            rows.append(dict(x=x, hnu=hnu, bracket=[left, right], y=y,
                             dyadic_certified=chord_ok, dyadic_midpoints=midpoints,
                             energy_change=energy_change, barrier_margin=margin,
                             chord_allowance=allowance, certified=bool(certified),
                             crosses_barrier=bool(y > 0), balance_defect=defect))
    return rows


def unit(vector):
    return vector / np.linalg.norm(vector)


def sample(seed):
    rng = np.random.default_rng(seed)
    prototypes = np.array([unit(v) for v in rng.standard_normal((4, 8))])
    patterns = []
    for prototype in prototypes:
        for _ in range(6):
            patterns.append(unit(prototype + 0.45 * rng.standard_normal(8) / np.sqrt(8)))
    start = 0.35 * prototypes[0] + 0.65 * unit(rng.standard_normal(8))
    return np.array(patterns), start


@lru_cache(maxsize=None)
def quadrature(order):
    points, weights = leggauss(order)
    return (points + 1) / 2, weights / 2


class Landscape:
    def __init__(self, patterns, beta=6.0):
        self.patterns = patterns
        self.beta = beta
        self.unit_matrix = np.eye(patterns.shape[1])

    def gradient(self, points):
        logits = self.beta * (points @ self.patterns.T)
        weights = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
        means = weights @ self.patterns
        return points - means, weights, means

    def energy(self, point):
        return point @ point / 2 - logsumexp(self.beta * (self.patterns @ point)) / self.beta

    def reference(self, start, final_time, rtol, atol):
        def velocity(time, y):
            return -self.gradient(y[None, :])[0][0]

        solution = solve_ivp(velocity, (0, final_time), start,
                             method="DOP853", rtol=rtol, atol=atol)
        if not solution.success:
            raise RuntimeError(solution.message)
        return solution.y[:, -1]

    def step(self, x, h, order):
        nodes, weights = quadrature(order)
        y = x.copy()
        for iteration in range(10):
            points = x + nodes[:, None] * (y - x)
            gradients, probabilities, means = self.gradient(points)
            residual = y - x + h * (weights @ gradients)
            if np.linalg.norm(residual) <= 2e-14:
                return y, iteration, float(np.linalg.norm(residual))
            # Sum the quadrature-node covariances before forming the Hessian.
            weighted_nodes = weights * nodes
            pattern_weights = weighted_nodes @ probabilities
            second_moment = self.patterns.T @ (pattern_weights[:, None] * self.patterns)
            mean_products = means.T @ (weighted_nodes[:, None] * means)
            jacobian = (1 + h / 2) * self.unit_matrix - h * self.beta * (second_moment - mean_products)
            y = y - np.linalg.solve(jacobian, residual)
        raise RuntimeError("AVF Newton solve failed to reach residual tolerance")

    def trajectory(self, start, h, final_time, order):
        count = int(round(final_time / h))
        assert np.isclose(count * h, final_time)
        y = start.copy()
        worst_balance, worst_residual, max_updates = 0.0, 0.0, 0
        for _ in range(count):
            x = y
            y, updates, residual = self.step(x, h, order)
            balance = self.energy(y) - self.energy(x) + np.dot(y - x, y - x) / h
            worst_balance = max(worst_balance, abs(balance))
            worst_residual = max(worst_residual, residual)
            max_updates = max(max_updates, updates)
        return y, worst_balance, max_updates, worst_residual


def accuracy_experiment():
    rows = []
    sizes = [0.25, 0.125, 0.0625, 0.03125]
    final_time = 2.0
    for seed in range(8):
        patterns, start = sample(seed)
        model = Landscape(patterns)
        diameter = np.linalg.norm(patterns[:, None, :] - patterns[None, :, :], axis=2).max()
        nu = max(0.0, model.beta * diameter**2 / 4 - 1)
        ref = model.reference(start, final_time, 4e-14, 2e-15)
        alt_ref = model.reference(start, final_time, 4e-13, 2e-14)
        reference_check = np.linalg.norm(ref - alt_ref)
        for h in sizes:
            assert h * nu < 2
            endpoint, balance, updates, residual = model.trajectory(start, h, final_time, 12)
            fine_quad, _, _, _ = model.trajectory(start, h, final_time, 24)
            rows.append(dict(seed=seed, h=h, nu=nu, error=np.linalg.norm(endpoint - ref),
                             max_balance_defect=balance, max_newton_updates=updates,
                             max_residual=residual,
                             quadrature_check=np.linalg.norm(endpoint - fine_quad),
                             reference_check=reference_check))
    summary = []
    prev_median = None
    for h in sizes:
        batch = [r for r in rows if r["h"] == h]
        errors = [r["error"] for r in batch]
        median = float(np.median(errors))
        assert max(r["max_balance_defect"] for r in batch) < 1e-12
        summary.append(dict(h=h, median_error=median, maximum_error=max(errors),
                            order=None if prev_median is None else np.log2(prev_median / median),
                            max_newton_updates=max(r["max_newton_updates"] for r in batch),
                            max_balance_defect=max(r["max_balance_defect"] for r in batch),
                            max_quadrature_check=max(r["quadrature_check"] for r in batch)))
        prev_median = median
    return rows, summary


def main():
    rows, summary = accuracy_experiment()
    data = dict(environment=dict(python=platform.python_version(), numpy=np.__version__,
                                 scipy=scipy.__version__), scalar=scalar_experiment(),
                accuracy=rows, accuracy_summary=summary)
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("avf_results.json")
    output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
