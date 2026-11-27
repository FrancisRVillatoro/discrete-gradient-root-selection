"""Outward-rounded sublevel tests for the binary64 chords in chord_results.json.

Pure Python standard library. Dot products, interpolation, and the envelope
optimization are exact rational arithmetic. exp and ln use 48-digit Decimal
arithmetic and are expanded to adjacent representable values. This verifies
barrier claims for the supplied points; it does not enclose an exact AVF root.
"""
from decimal import Context, Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN
from fractions import Fraction
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PREC = 48
DOWN = Context(prec=PREC, rounding=ROUND_FLOOR)
UP = Context(prec=PREC, rounding=ROUND_CEILING)
NEAR = Context(prec=PREC, rounding=ROUND_HALF_EVEN)


def decimal_bound(value, context):
    return context.divide(Decimal(value.numerator), Decimal(value.denominator))


def elementary_bound(name, x, upper):
    # Decimal exp and ln are correctly rounded to nearest. An adjacent
    # representable Decimal therefore provides a rigorous outer endpoint.
    nearest = getattr(NEAR, name)(x)
    return NEAR.next_plus(nearest) if upper else NEAR.next_minus(nearest)


def logsumexp_enclosure(values):
    lower = upper = Decimal(0)
    for v in values:
        low_exp = elementary_bound("exp", decimal_bound(v, DOWN), False)
        high_exp = elementary_bound("exp", decimal_bound(v, UP), True)
        lower = DOWN.add(lower, low_exp)
        upper = UP.add(upper, high_exp)
    return (Fraction(elementary_bound("ln", lower, False)),
            Fraction(elementary_bound("ln", upper, True)))


def dot(p, q):
    return sum(a * b for a, b in zip(p, q))


def max_envelope(a, b, curvature):
    if curvature == 0:
        return max(a, b)
    t = min(1, max(0, Fraction(1, 2) + (b - a) / curvature))
    return (1 - t) * a + t * b + curvature * t * (1 - t) / 2


def classify(grid_energy, c, scale, max_depth=10):
    bounds = [grid_energy(Fraction(0)), grid_energy(Fraction(1))]
    for depth in range(max_depth + 1):
        sample_margin = max(lo for lo, _ in bounds) - c[1]
        if sample_margin >= 0:
            return "sample_above", depth, sample_margin
        m = 2**depth
        curvature = scale / m**2
        upper = max(max_envelope(p[1], q[1], curvature)
                    for p, q in zip(bounds[:-1], bounds[1:]))
        if upper < c[0]:
            return "below", depth, c[0] - upper
        if depth < max_depth:
            mixed = []
            for i, p in enumerate(bounds[:-1]):
                mixed.extend((p, grid_energy(Fraction(2 * i + 1, 2 * m))))
            bounds = mixed + [bounds[-1]]
    return "unresolved", max_depth, Fraction(0)


def verify():
    source = json.loads((HERE / "chord_results.json").read_text())
    beta = Fraction(source["beta"])
    vectors = [[Fraction(v) for v in row] for row in source["transverse_patterns"]]
    if beta * max(dot(z, z) for z in vectors) >= 1:
        raise ValueError("Transverse strong convexity has not been established")
    patterns = [[Fraction(s), *z] for s in (-1, 1) for z in vectors]
    # Central symmetry makes the squared diameter exactly four times the
    # largest squared norm. Confirm central symmetry for the recorded input.
    if {tuple(v) for v in patterns} != {tuple(-a for a in v) for v in patterns}:
        raise ValueError("Pattern set must be centrally symmetric")
    global_nu = max(Fraction(0), beta * max(dot(p, p) for p in patterns) - 1)
    llo, lhi = logsumexp_enclosure([Fraction(0)] * len(patterns))
    c = -lhi / beta, -llo / beta
    checks = []
    endpoint_margins = []
    for row in source["rows"]:
        x = list(map(Fraction, row["x"]))
        y = list(map(Fraction, row["y"]))
        d = [b - a for a, b in zip(x, y)]
        xx, xd, dd = dot(x, x), dot(x, d), dot(d, d)
        projected_x = [dot(p, x) for p in patterns]
        projected_d = [dot(p, d) for p in patterns]
        width = max(projected_d) - min(projected_d)
        # This is nu_d * ||d||^2 without any square roots.
        projected_scale = max(Fraction(0), beta * width**2 / 4 - dd)
        cache = {}

        def energy(s):
            if s not in cache:
                kinetic = (xx + 2 * s * xd + s**2 * dd) / 2
                lower, upper = logsumexp_enclosure(
                    [beta * (a + s * b) for a, b in zip(projected_x, projected_d)]
                )
                cache[s] = kinetic - upper / beta, kinetic - lower / beta
            return cache[s]

        endpoint_margin = c[0] - max(energy(Fraction(0))[1], energy(Fraction(1))[1])
        if endpoint_margin <= 0:
            raise ArithmeticError("Endpoint failed strict sublevel test")
        endpoint_margins.append(endpoint_margin)
        for label, curvature in (("global_test", global_nu * dd),
                                 ("directional_test", projected_scale)):
            verdict, depth, margin = classify(energy, c, curvature)
            if [verdict, depth] != row[label] or margin <= 0:
                raise ArithmeticError("Enclosure disagrees with recorded classification")
            checks.append(dict(q0=row["q0"], q1=row["q1"], h=row["h"],
                               gap_fraction=row["gap_fraction"], bound=label,
                               verdict=verdict, depth=depth,
                               certified_margin_floor=str(decimal_bound(margin, DOWN))))
    summary = dict(input="chord_results.json", decimal_digits=PREC,
                   candidate_steps=len(source["rows"]), verified_tests=len(checks),
                   smallest_endpoint_margin=str(decimal_bound(min(endpoint_margins), DOWN)),
                   smallest_decision_margin=str(min(Decimal(v["certified_margin_floor"])
                                                    for v in checks)),
                   checks=checks)
    (HERE / "chord_verified.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "checks"}, indent=2))


if __name__ == "__main__":
    verify()
