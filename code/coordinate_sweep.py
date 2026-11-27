"""Reproduce the palindromic coordinate discrete-gradient experiments.

Run with Python 3, NumPy, SciPy and mpmath. All parameters are deterministic.
Binary64 trajectories are supplemented by an independent interval proof for
the three selected coordinate roots in the h=4 branch experiment.
"""
import json
import math
from fractions import Fraction

import numpy as np
from mpmath import iv
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

BETA = 8.0
A = np.array([0.8, 0.6])
B = np.array([-0.6, 0.8])


def log_cosh(u):
    return np.logaddexp(u, -u) - np.log(2.0)


def energy(x):
    return float(0.5 * np.dot(x, x) - log_cosh(BETA * np.dot(A, x)) / BETA)


def log_cosh_difference(u, v):
    if abs(v) < 0.25:
        # cosh(u+v)/cosh(u) = 1 + 2 sinh(v/2)^2 + tanh(u) sinh(v).
        return math.log1p(2.0 * math.sinh(0.5 * v) ** 2
                          + math.tanh(u) * math.sinh(v))
    return float(log_cosh(u + v) - log_cosh(u))


def coordinate_step(x, i, tau, selected, counts):
    alpha = A[i]
    proj = float(np.dot(A, x))
    u = BETA * proj
    grad = x[i] - alpha * math.tanh(u)
    if grad == 0:
        return x.copy()
    direction = -math.copysign(1.0, grad)
    slope = 1.0 / tau + 0.5
    scalar_curvature = 4.0 * BETA ** 2 * abs(alpha) ** 3 / (9.0 * math.sqrt(3.0))
    slope_floor = slope - 0.5 * BETA * alpha ** 2

    def residual(s):
        counts["scalar_evaluations"] += 1
        if s == 0.0:
            return direction * grad
        change = log_cosh_difference(u, BETA * alpha * direction * s)
        return direction * x[i] + slope * s - change / (BETA * s)

    def unique_bracket(lo, glo, hi, ghi, depth=0):
        width = hi - lo
        if max(glo, ghi) + scalar_curvature * width ** 2 / 8.0 < 0.0:
            return None
        increasing = slope_floor > 0.0 or (ghi - glo) / width > scalar_curvature * width / 2.0
        if increasing:
            if ghi < 0.0:
                return None
            return lo, hi
        if depth >= 50:
            raise RuntimeError("Unable to isolate a simple first root")
        counts["interval_bisections"] += 1
        mid = (lo + hi) / 2.0
        gmid = residual(mid)
        left = unique_bracket(lo, glo, mid, gmid, depth + 1)
        return left if left is not None else unique_bracket(mid, gmid, hi, ghi, depth + 1)

    upper = (abs(x[i]) + abs(alpha) + 1.0) / slope
    if selected:
        lo, hi = unique_bracket(0.0, residual(0.0), upper, residual(upper))
        s = brentq(residual, lo, hi, xtol=5e-16, rtol=2e-15)
    else:
        # Unconstrained Newton on the signed secant equation from zero displacement.
        s = 0.0
        for _ in range(40):
            value = residual(s)
            if abs(value) < 2e-14:
                break
            if abs(s) < 1e-7:
                derivative = slope - 0.5 * BETA * alpha**2 * (1.0 - math.tanh(u)**2)
            else:
                v = BETA * alpha * direction * s
                dlog = BETA * alpha * direction * math.tanh(u + v)
                derivative = slope - (s * dlog - log_cosh_difference(u, v)) / (BETA * s*s)
            s -= value / derivative
        if abs(residual(s)) > 3e-13:
            raise RuntimeError("Newton failed on the benchmark trajectory")
    y = x.copy()
    y[i] += direction * s
    defect = abs(energy(y) - energy(x) + s * s / tau)
    counts["max_energy_identity_defect"] = max(counts["max_energy_identity_defect"], defect)
    return y


def sweep(x, h, selected, counts):
    stages = []
    for i, tau in [(0, h / 2.0), (1, h), (0, h / 2.0)]:
        x = coordinate_step(x, i, tau, selected, counts)
        stages.append(x)
    return stages


def empty_counts():
    return {"scalar_evaluations": 0, "interval_bisections": 0, "max_energy_identity_defect": 0.0}


def verified_large_step():
    """Enclose the first root of each stage from the exact rational initial state."""
    iv.dps = 70

    def interval(fraction):
        return iv.mpf(fraction.numerator) / fraction.denominator

    def cosh_interval(v):
        return (iv.exp(v) + iv.exp(-v)) / 2

    def tanh_interval(v):
        exp_two = iv.exp(2 * v)
        return (exp_two - 1) / (exp_two + 1)

    def rounded_out(value):
        # The displayed grid has seven decimal places; verify both endpoints.
        scale = 10**7
        lo = math.floor(float(value.a) * scale) - 1
        hi = math.ceil(float(value.b) * scale) + 1
        assert iv.mpf(lo) / scale < value < iv.mpf(hi) / scale
        return [lo / scale, hi / scale]

    a = [iv.mpf(4) / 5, iv.mpf(3) / 5]
    x = [iv.mpf(3) / 200, iv.mpf(3) / 100]
    records = []
    for i, tau, refinements in [(0, 2, 105), (1, 4, 74), (0, 2, 44)]:
        alpha = a[i]
        u = 8 * (a[0]*x[0] + a[1]*x[1])
        grad = x[i] - alpha * tanh_interval(u)
        assert grad < 0 or grad > 0
        direction = 1 if grad < 0 else -1
        slope = (iv.mpf(1) / tau) + iv.mpf(1) / 2
        curvature = 4 * 8**2 * alpha**3 / (9 * iv.sqrt(3))

        def residual(t):
            if t == 0:
                return direction * grad
            ray = interval(t)
            change = iv.log(cosh_interval(u + 8 * alpha * direction * ray)
                            / cosh_interval(u))
            return direction * x[i] + slope * ray - change / (8 * ray)

        def isolate(lo, lower, hi, upper, depth=0):
            width = interval(hi - lo)
            remainder = curvature * width**2 / 8
            if lower + remainder < 0 and upper + remainder < 0:
                return None
            is_increasing = (upper - lower) / width > curvature * width / 2
            if is_increasing and upper < 0:
                return None
            if is_increasing and lower < 0 and upper > 0:
                return lo, hi
            if depth >= 55:
                raise ArithmeticError("Interval bounds inconclusive")
            mid = (lo + hi) / 2
            val = residual(mid)
            left = isolate(lo, lower, mid, val, depth + 1)
            return left if left is not None else isolate(mid, val, hi, upper, depth + 1)

        lo, hi = Fraction(0), Fraction(4)
        lo, hi = isolate(lo, residual(lo), hi, residual(hi))
        for _ in range(refinements):
            mid = (lo + hi) / 2
            val = residual(mid)
            if val < 0:
                lo = mid
            elif val > 0:
                hi = mid
            else:
                break
        assert residual(lo) < 0 and residual(hi) > 0
        endpoints = iv.mpf([interval(lo).a, interval(hi).b])
        x[i] += direction * endpoints
        q = a[0]*x[0] + a[1]*x[1]
        assert q > 0
        records.append({"coordinate": i + 1, "stage_size": tau,
                        "positive_ray_displacement": rounded_out(endpoints),
                        "projection": rounded_out(q)})
    return records


def main():
    x0 = 0.03 * A + 0.015 * B
    branches = []
    for h in [0.25, 1.0, 2.0, 4.0]:
        for selected in [True, False]:
            counts = empty_counts()
            stages = sweep(x0, h, selected, counts)
            branches.append({"h": h, "selection": "first_descent_root" if selected else "newton_zero",
                             "stage_projections": [float(np.dot(A, x)) for x in stages],
                             "stage_energies": [energy(x) for x in stages], **counts})

    T = 1.0
    q0, p0 = 0.12, 0.06
    start = q0 * A + p0 * B
    sol = solve_ivp(lambda t, q: -q + np.tanh(BETA * q), (0.0, T), [q0],
                    method="DOP853", rtol=3e-14, atol=1e-15)
    exact = sol.y[0, -1] * A + p0 * np.exp(-T) * B
    orders = []
    for n in [8, 16, 32, 64, 128]:
        x = start.copy()
        counts = empty_counts()
        for _ in range(n):
            x = sweep(x, T / n, True, counts)[-1]
        error = float(np.linalg.norm(x - exact))
        observed = None if not orders else float(np.log2(orders[-1]["error"] / error))
        orders.append({"n": n, "h": T / n, "error": error, "observed_order": observed, **counts})
    data = {"beta": BETA, "direction": A.tolist(), "transverse_direction": B.tolist(),
            "branch_initial_coordinates": x0.tolist(), "branch_initial_energy": energy(x0),
            "branches": branches, "verified_first_root_h4": verified_large_step(),
            "accuracy_final_time": T, "accuracy_initial_qp": [q0, p0],
            "accuracy_reference": exact.tolist(), "accuracy": orders}
    print(json.dumps(data, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
