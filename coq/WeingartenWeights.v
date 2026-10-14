(* WeingartenWeights.v -- gate weights of the input gauge (paper, proposition "input gauge"),
   computed from the two-copy Weingarten values (Collins and Sniady), and identities used in
   Theorems D and E. Spins: false = e (identity), true = s (swap). Coq 8.18, Reals library. *)
Require Import Reals Lra.
Open Scope R_scope.

Section Weights.
Variable q : R.
Hypothesis Hq : q > 1.

(* Wg(e) = 1/(q^4-1), Wg(s) = -1/(q^2 (q^4-1)); Wg(sigma^-1 tau) is Wg(e) if sigma = tau, Wg(s) otherwise. *)
Definition Wg (sigma tau : bool) : R :=
  if Bool.eqb sigma tau then 1 / (q^4 - 1) else - (1 / (q^2 * (q^4 - 1))).
(* contraction of one leg: <<sigma|pi>> = q^{c(sigma^-1 pi)} = q^2 if sigma = pi, q otherwise *)
Definition ov (sigma pi : bool) : R := if Bool.eqb sigma pi then q^2 else q.
(* weight of a gate in the input gauge: sum over the output spin sigma *)
Definition w (tau a a' : bool) : R :=
  Wg false tau * ov false a * ov false a' + Wg true tau * ov true a * ov true a'.

Definition K := q / (q^2 + 1).

Lemma q4m1_pos : q^4 - 1 > 0.
Proof. assert (q^2 > 1) by nra. nra. Qed.

Lemma w_equal : forall t, w t t t = 1.
Proof.
  pose proof q4m1_pos as H4. assert (Hq2 : q^2 > 0) by nra.
  intros [|]; unfold w, Wg, ov; simpl; field; lra.
Qed.

Lemma w_forbidden : forall t, w t (negb t) (negb t) = 0.
Proof.
  pose proof q4m1_pos as H4. assert (Hq2 : q^2 > 0) by nra.
  intros [|]; unfold w, Wg, ov; simpl; field; lra.
Qed.

Lemma w_wall : forall t a, w t a (negb a) = K.
Proof.
  pose proof q4m1_pos as H4. assert (Hq2 : q^2 + 1 > 0) by nra. assert (q^2 > 0) by nra.
  assert (Hf : q^4 - 1 = (q^2 - 1) * (q^2 + 1)) by ring. assert (q^2 - 1 > 0) by nra.
  intros [|] [|]; unfold w, Wg, ov, K; simpl; field; repeat split; lra.
Qed.

(* Theorem E (b): for k = 1, e^{2 theta} = (q^2-q+1)/q solves K (e^{2 theta} + 1) = 1 *)
Lemma theta1_root : K * ((q^2 - q + 1) / q + 1) = 1.
Proof. assert (q^2 + 1 > 0) by nra. unfold K; field; lra. Qed.

(* annealed line tension at the butterfly point: E(v_B) = v_B ln q, with E(v) = ln(1/K) - h((1+v)/2),
   (1+v_B)/2 = a = q^2/(q^2+1), (1-v_B)/2 = b = 1/(q^2+1). *)
Definition a_ := q^2 / (q^2 + 1).
Definition b_ := 1 / (q^2 + 1).
Definition vB := (q^2 - 1) / (q^2 + 1).
Definition E_vB := ln (/ K) + a_ * ln a_ + b_ * ln b_.

Lemma butterfly_tension : E_vB = vB * ln q.
Proof.
  assert (H1 : q^2 + 1 > 0) by nra. assert (H0 : q > 0) by lra. assert (Hq2 : q^2 > 0) by nra.
  unfold E_vB, a_, b_, vB, K.
  replace (/ (q / (q^2 + 1))) with ((q^2 + 1) * / q) by (field; lra).
  rewrite ln_mult by (try apply Rinv_0_lt_compat; lra).
  rewrite ln_Rinv by lra.
  unfold Rdiv. rewrite ln_mult by (try apply Rinv_0_lt_compat; lra).
  rewrite Rmult_1_l. rewrite ln_Rinv by lra.
  replace (q^2) with (q * q) by ring. rewrite ln_mult by lra.
  field; lra.
Qed.

End Weights.
