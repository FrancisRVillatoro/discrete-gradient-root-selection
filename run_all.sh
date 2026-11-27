#!/usr/bin/env bash
# Runs every program of Section 8 of the article, in the order of its subsections.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out figs
cd code
python3 exp1_twowell.py
python3 exp2_ring.py
python3 chord_test.py
python3 verify_chords.py
python3 coordinate_sweep.py > ../out/coordinate_results.json
python3 exp3_rotated.py
python3 exp7_random.py
python3 exp4_allencahn.py
python3 exp6_barrier.py
python3 exp5_hybrid.py
python3 avf_experiments.py ../out/avf_results.json
