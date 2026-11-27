# Root selection by step-size continuation for discrete-gradient integrators of nonconvex gradient flows

Programs of the numerical experiments of the article *Root selection by step-size continuation for discrete-gradient integrators of nonconvex gradient flows*, by Francisco R. Villatoro (Departamento de Lenguajes y Ciencias de la Computación, Escuela de Ingenierías Industriales, Universidad de Málaga, Spain, ORCID [0000-0003-4314-6213](https://orcid.org/0000-0003-4314-6213)).

The programs generate every number, table and figure of Section 8 of the article. The repository contains no output files, because the programs regenerate all of them.

## Requirements

Python 3.12 or later, NumPy, SciPy, Matplotlib and mpmath. From the root of the repository,

```sh
python3 -m pip install -r requirements.txt
```

The results of the article were obtained with CPython 3.12.14, NumPy 2.3.5, SciPy 1.17.0 and mpmath 1.3.0 for `chord_test.py`, `verify_chords.py`, `coordinate_sweep.py` and `avf_experiments.py`, and with CPython 3.13.18, NumPy 2.5.3, SciPy 1.18.1 and Matplotlib 3.11.2 for the other programs. Quantities computed in binary floating point may differ in the last digits with other versions or other linear-algebra libraries.

## Running the programs

From the root of the repository,

```sh
bash run_all.sh
```

creates the folders `out` and `figs` and runs every program in the order of Section 8. A single program can be run from the folder `code`, for instance

```sh
mkdir -p out figs
cd code
python3 exp1_twowell.py
```

The programs `exp*.py` write to `../out` and `../figs`, so these folders must exist.

## Programs

| Program | Section | Reproduces | Output |
|---|---|---|---|
| `exp1_twowell.py` | 8.1 | Table 1 and Figure 1 | `out/exp1_twowell.json`, `figs/twowell_energy.pdf`, `figs/twowell_homotopy.pdf` |
| `exp2_ring.py` | 8.2 | Table 2 and Figure 2 | `out/exp2_ring.json`, `figs/ring_valley.pdf` |
| `chord_test.py` | 8.3 | Table 3, step residuals and chain-rule defects | `code/chord_results.json` |
| `verify_chords.py` | 8.3 | Verification of the 108 classifications of Table 3 in exact rational arithmetic, with outward bounds on exp and log at 48 decimal digits | `code/chord_verified.json` |
| `coordinate_sweep.py` | 8.4, 8.8 | Table 4, the interval enclosures at 70 decimal digits, the sweeps with h = 0.25, 1 and 2, and the sweep columns of Table 8 | standard output, saved by `run_all.sh` as `out/coordinate_results.json` |
| `exp3_rotated.py` | 8.4 | Coupled AVF steps for the rotated two-well energy | `out/exp3_rotated.json` |
| `exp7_random.py` | 8.5 | Table 5 and Figure 3 | `out/exp7_random.json`, `figs/random_newton_zero.pdf`, `figs/random_newton_euler.pdf` |
| `exp4_allencahn.py` | 8.6 | Table 6, Figure 4 and the other values quoted in Section 8.6 | `out/exp4_allencahn.json`, `figs/allencahn_first_step.pdf`, `figs/allencahn_mean.pdf` |
| `exp6_barrier.py` | 8.6 | Minimization of the energy on the hyperplane of zero mean, which identifies the barrier level | `out/exp6_barrier.json` |
| `exp5_hybrid.py` | 8.7 | Table 7 | `out/exp5_hybrid.json` |
| `avf_experiments.py` | 8.8 | AVF columns of Table 8 and the checks of the reference solution and of the quadrature | `out/avf_results.json` |

The programs use three modules. `dgcont.py` contains the AVF discrete gradient by Gauss–Legendre quadrature, Newton's method, continuation in the step size by pseudo-arclength path following, and the energy envelopes and certificates of Section 6. `allen_cahn.py` contains the discretized Allen–Cahn equation of Section 8.6, and `figstyle.py` the common format of the figure panels.

`exp5_hybrid.py` imports `exp1_twowell.py`, `exp2_ring.py`, `exp3_rotated.py` and `allen_cahn.py` and recomputes the steps it needs. `verify_chords.py` reads the endpoints written by `chord_test.py` as exact binary rationals and uses only the Python standard library. Besides the data of Table 8, `avf_experiments.py` writes, under the key `scalar`, a study of the roots of a scalar step that the article does not use. Every random quantity is drawn from a NumPy generator with a fixed seed.

## Licence

MIT, see `LICENSE`.

## Funding

This work was supported by the Universidad de Málaga, Spain, through project PRO-B4-2026-011 of its Plan Propio de Investigación, Transferencia y Divulgación Científica.
