# Pre-registered predictions

Every numerical campaign of the paper was preceded by written predictions with success criteria. The
files in `original_fr/` are the originals, in French, copied without modification (they were written in
French during the working sessions). This page summarises them in English and gives the outcome of each
prediction, as reported in the paper.

**What "pre-registered" means here.** Each file states the date and time at which it was written, before
the corresponding scripts were run (and, for the later campaigns, before they were written). These
dates come from the working sessions. This repository was created afterwards, in a single commit, so its
history does not independently certify them. Treat them as a declaration by the author, not as
third-party evidence.

Some files refer to internal working notes ("note 1", "note 2", "note 3") that are not part of this
repository. Notes 1 and 2 used an earlier gate ensemble, "rand2", which turned out not to be a 2-design
(paper, Section 8.1). Every number in the paper comes from uniform Clifford gates and is reproduced
by the scripts in `code/`.

| file | campaign | predictions | outcome (paper section) |
|---|---|---|---|
| `00_T1_threshold_protocol.txt` | first test, Haar circuits, n ≤ 22 | Frautschi-type threshold: the deficit stays bounded if k·v ≳ 1/2 | **refuted**: no threshold for k = 1 to 32 (Sec. 8.3) |
| `01_T1bis_large_size_protocol.txt` | same question at large size | "threshold" version (slope compatible with 0 for k ≥ k*) against "continuous" version (slope·k ≈ constant) | threshold version **refuted** (Sec. 8.3) |
| `02_addendum_scaling_parity_eden.txt` | 1D/2D scaling laws, parity mechanism, Eden | sub-volume exponents for accelerated and exponential growth; edge of the cone advances iff the number of insertions to its left is even (≥ 95 % of layers); pair insertions give v_eff = 1 and G_LB/N = 0.184 ± 5 %; Eden G/N stable within ±15 % | parity rule holds in 100 % of 7378 layers; pair control confirmed (+3 %); accelerated fronts confirmed; fast exponential front **failed as a test** (no gate crosses the cut); quantitative values for N ∝ t **failed** (Secs. 8.8, 8.9) |
| `03_bracket_alpha1.txt` | single-wall model and bracket at α = 1 (P1–P6) | P1 recursion = full spin model to 1e-10; P2 uniform Clifford purity = exact Z_A (z < 3), rand2 rejected; P3 annealed densities 0.2925 / 0.4843 / 0.6760 / 0.8168; P4 quenched values inside the proven bracket; P5 v_E in ]0.322, 0.45[ and membrane bracket; P6 chain bound A ≤ m − S2/log 2 ≤ B(t*) + 3.04 for α < 1 | all confirmed (P5 only weakly: v_E is barely above the annealed value). The paper's Theorem D has the sharper constant 1.70 bits; the chain also holds with 1.70 in all 34 cases (Secs. 8.2, 8.3, 8.7) |
| `04_uniform_clifford_rerun.txt` | rerun of earlier campaigns with uniform Clifford gates (R1–R6) | no violation of Theorem A; structural bounds unchanged; quenched values lower and inside the Theorem E bound; local pairs indistinguishable from reference, R/P2 in [2, 4.5]; v_B = 0.60 ± 0.01; k = 16, 32 inside the bracket | all confirmed, except R/P2 = 4.9 ± 0.7 at k = 32, outside [2, 4.5] but compatible within one standard error (Secs. 8.2–8.5) |
| `05_free_field.txt` | free field on an expanding lattice (F1–F7) | F1 product birth costs 0.01–0.2 Λ, independent of H; F2 vacuum-preserving birth costs < 1e-3 Λ, expansion production ∝ H^γ with γ in [1.5, 2.5]; F3 optimised MERA stays ≥ 1e-3 Λ for D ≤ 3; F4 zero mutual information for product births; F5, F6 exponentially decaying correlations and locality of the vacuum-preserving map; F7 thermal-gap diagnostic | F1, F2, F4, F6, F7 confirmed; F5 confirmed for the part created by the birth; **F3 failed** (the threshold is crossed at D = 2: the prediction was too pessimistic) (Sec. 9) |
| `06_revision_campaigns.txt` | Haar against Clifford at equal size, finite-size scaling, parity closed form, correlated births (V1–V4) | V1 (a) annealed check, (b) S > S2 growing with m, (c) \|E[S2]_Haar − E[S]_Clifford\| ≤ 0.2 bit, (d) ordering, (e) positions; V2 convergence of c(m), non-universal constant; V3 closed form within 0.02 at 4 seeds; V4 local correlations without effect, non-local reduction ≤ 10 % at f = 0.01 and in [5 %, 75 %] at f = 0.1 | V1 (c) **failed** from N = 16 (0.23 bit); V2 confirmed, but the limit depends on the parity of m and not on N0, contrary to what was announced; V3 **failed** for N ∝ t² and exponential growth at 4 seeds (agreement within one standard error with 44 seeds, added afterwards); V4 confirmed except k = 4, f = 0.1 (4.8(10) %, just below 5 %) (Secs. 8.4, 8.6, 8.8, and the remark on finite-size corrections in Sec. 6) |

The scripts that ran the first two protocols are not part of this repository: T1 used Haar state-vector
circuits at n ≤ 22, and T1-bis used the rand2 ensemble. The refutation of the threshold reported in the
paper rests on the uniform-Clifford campaigns (`code/edge_growth_quenched.py`, `code/edge_growth_large_k.py`).
