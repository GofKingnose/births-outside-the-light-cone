#!/bin/sh
# Compiles the five proof files and prints the axioms used by the main results (Coq 8.18).
set -e
cd "$(dirname "$0")"
for f in CausalCone ConeLemmas ParityA1 WallDescent WeingartenWeights Assumptions; do
  coqc -Q . BOLC "$f.v"
done
if grep -nE "Admitted|admit\." *.v; then echo "unfinished proof found"; exit 1; fi
echo "Coq: all files compiled, $(grep -c 'Qed\.' CausalCone.v ConeLemmas.v ParityA1.v WallDescent.v WeingartenWeights.v | awk -F: '{s+=$2} END {print s}') proofs closed with Qed."
