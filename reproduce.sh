#!/usr/bin/env bash
# Reproduces the results of the paper.
#   ./reproduce.sh quick   unit tests, Coq, exact computations, spot checks against the stored data,
#                          analyses and figures (about 10 minutes on two cores)
#   ./reproduce.sh full    quick + every Monte Carlo campaign except the Haar state-vector campaign of
#                          Table 4 (about 30 minutes more on two cores)
#   ./reproduce.sh haar    the Haar state-vector campaign of Table 4 (several hours on two cores)
#   ./reproduce.sh paper   compiles paper/main.pdf (needs pdflatex)
# Seeds are fixed in the scripts, so the campaigns rewrite the files of data/ and field_test/ with
# identical content; when the repository is a git clone, the last step lists any file that changed.
set -euo pipefail
MODE="${1:-quick}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# fixed dates in the generated PDF files, so that figures and paper are byte-reproducible
export SOURCE_DATE_EPOCH=1791936000 FORCE_SOURCE_DATE=1
run() { echo; echo "=== $*"; "$@"; }

quick() {
  cd "$ROOT/code"
  run python3 test_theorem_A.py
  run python3 check_ctab_purity.py
  run python3 domain_wall.py
  run python3 full_spin_model.py
  run python3 optimality_A1.py
  run python3 mincut_vs_cone.py
  run python3 correlated_births_cone.py
  run python3 dual_unitary.py
  run python3 dual_unitary_fixed.py
  run python3 finite_size_theta.py
  run python3 annealed_density.py
  run python3 alpha_annealed.py
  run python3 parity_locking.py
  run python3 parity_closed_form.py
  run python3 entanglement_velocity.py
  run python3 haar_vs_clifford_analysis.py
  run python3 table_statistics.py
  run python3 spot_checks.py
  run python3 make_figures.py
  cd "$ROOT/field_test"
  run python3 tests_field.py
  run python3 analyse_field.py
  if command -v coqc >/dev/null; then run "$ROOT/coq/check_coq.sh"; else echo "coqc not found: Coq check skipped"; fi
}

full() {
  quick
  cd "$ROOT/code"
  run python3 check_two_design.py
  run python3 butterfly_velocity.py
  run python3 edge_growth_quenched.py 2
  run python3 edge_growth_large_k.py
  run python3 pair_births.py
  run python3 correlated_births.py 2
  run python3 alpha_quenched.py
  run python3 parity_closed_form_more_seeds.py
  run python3 mincut_pairs.py
  run python3 growth_campaigns.py
  run python3 table_statistics.py
  run python3 make_figures.py
  cd "$ROOT/field_test"
  run python3 run_field.py scan
  run python3 run_field.py main
  run python3 mera_depth.py
  run python3 mi_exact_vacuum.py
  run python3 analyse_field.py
}

case "$MODE" in
  quick) quick ;;
  full)  full ;;
  haar)  cd "$ROOT/code"; run python3 haar_vs_clifford.py 2; run python3 haar_vs_clifford.py 2 large; run python3 haar_vs_clifford_analysis.py ;;
  paper) cd "$ROOT/paper"; for i in 1 2 3; do pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null; done; echo "paper/main.pdf built" ;;
  *) echo "usage: $0 [quick|full|haar|paper]"; exit 2 ;;
esac

if git -C "$ROOT" rev-parse >/dev/null 2>&1; then
  echo; echo "=== stored files rewritten with different bytes by this run. Expected causes only: records of parallel"
  echo "    campaigns stored in a different order, and floating-point round-off in the free-field outputs."
  git -C "$ROOT" status --short -- data field_test paper/figures || true
fi
