(* ======================================================================
   ConeLemmas.v -- algebraic lemmas behind the two-path membrane estimate
   (paper "Births outside the light cone", remark on the membrane estimate)
   and the no-scrambling limit of non-local pairs.

   This file checks the elementary part of the derivation (geometry of the
   two cut paths and combination of the bounds). The premises, i.e. that
   the entropy is at most the cost of each cut of the network, are taken
   from the literature on entanglement membranes (Nahum et al. 2017;
   Jonay, Huse and Nahum 2018) and are NOT formalised here: the resulting
   expression is an estimate, not a theorem of the paper.
   ====================================================================== *)

Require Import Reals Lra.
Open Scope R_scope.

(* --- 1. End point of the diagonal path ------------------------------
   Cut in the middle m = n/2 at the final time T = n k (edge growth, one
   qubit per step, k layers per step). The path leaves (m, T) to the right
   at speed v and meets the birth line t = k x at the point
   xs = (m + v T)/(1 + v k). *)

Lemma diagonal_meets_births :
  forall n k v : R, 0 < v -> 0 <= k ->
  let T := n * k in let m := n / 2 in
  let xs := (m + v * T) / (1 + v * k) in
  xs - m = v * (T - k * xs).
Proof.
  intros n k v Hv Hk T m xs. unfold xs, T, m.
  assert (1 + v * k <> 0) by nra.
  field. assumption.
Qed.

(* Cost of the diagonal path = number of world lines crossed. *)
Lemma diagonal_cost :
  forall n k v : R, 0 < v -> 0 <= k ->
  let T := n * k in let m := n / 2 in
  let xs := (m + v * T) / (1 + v * k) in
  xs - m = (n / 2) * (v * k) / (1 + v * k).
Proof.
  intros n k v Hv Hk T m xs. unfold xs, T, m.
  assert (1 + v * k <> 0) by nra.
  field. assumption.
Qed.

(* --- 2. Fraction never reached ---------------------------------------- *)

Lemma unreached_fraction :
  forall a m : R, 0 <= a -> m - (a * m) / (1 + a) = m / (1 + a).
Proof.
  intros a m Ha. assert (1 + a <> 0) by lra. field. assumption.
Qed.

(* A qubit at distance d from the cut, born (m - d) steps ago, is reached
   by the cone (effective speed a = k v per step) iff d (1 + a) <= a m. *)
Lemma reached_iff :
  forall a m d : R, d <= a * (m - d) <-> d * (1 + a) <= a * m.
Proof. intros a m d. split; intro H; nra. Qed.

(* --- 3. Combination of two upper bounds on the entropy ----------------- *)

Lemma gap_two_paths :
  forall m S c1 c2 : R, S <= c1 -> S <= c2 ->
  m - S >= Rmax (m - c1) (m - c2).
Proof.
  intros m S c1 c2 H1 H2. unfold Rmax.
  destruct (Rle_dec (m - c1) (m - c2)); lra.
Qed.

(* --- 4. Final form of the two-path estimate --------------------------
   With c1 = (n/2) k vE (vertical path) and c2 = diagonal cost:
   G = n/2 - S >= (n/2) * max(1 - k vE, 1/(1 + k vB)). *)

Lemma two_path_estimate :
  forall n k vE vB S : R, 0 < vB -> 0 <= k ->
  S <= (n / 2) * (k * vE) ->
  S <= (n / 2) * (vB * k) / (1 + vB * k) ->
  n / 2 - S >= Rmax ((n / 2) * (1 - k * vE)) ((n / 2) / (1 + vB * k)).
Proof.
  intros n k vE vB S HvB Hk H1 H2.
  assert (Hpos : 1 + vB * k <> 0) by nra.
  assert (E1 : n / 2 - (n / 2) * (k * vE) = (n / 2) * (1 - k * vE)) by ring.
  assert (E2 : n / 2 - (n / 2) * (vB * k) / (1 + vB * k) = (n / 2) / (1 + vB * k))
    by (field; assumption).
  rewrite <- E1, <- E2.
  apply gap_two_paths; assumption.
Qed.

(* --- 5. No-scrambling limit of non-local pairs ------------------------
   Per pair birth: n grows by 2 and the half m by 1; one pair in two
   (probability p = 1/2) straddles the cut and contributes 1 bit.
   Slope per degree of freedom = (1 - p)/2 = 1/4, against 1/2 without pairs. *)

Lemma slope_P2_no_scrambling : (1 - 1/2) / 2 = 1/4.
Proof. field. Qed.

Lemma ratio_k0 : (1/2) / (1/4) = 2.
Proof. field. Qed.
