(* WallDescent.v -- Coq 8.18; the combinatorial part uses no axiom.
   Annealed Renyi-2 model in the input gauge (paper, Section 6): a configuration is a list of
   binary spins (true = s, false = e), one per living leg, read just above the current slice.
   Elementary steps, going down:
     - gate on (i, i+1) with input spin tau, ALLOWED if the weight 1/K/0 is non-zero,
       i.e. if a_i <> a_(i+1) or tau = a_i;
     - removal of a leg (a birth read backwards), anywhere in the chain.
   Theorem 1 (walls_nonincreasing): the number of walls never increases.
   Theorem 2 (path_bundle): weight of a bundle of wall paths:
     C(j+l, j) * K^(j+l) * q^j <= q^l,  K = q/(q^2+1)   (net displacement X = j - l). *)

Require Import List Arith Lia Bool.
Import ListNotations.

Fixpoint walls (l : list bool) : nat :=
  match l with
  | x :: ((y :: _) as t) => (if Bool.eqb x y then 0 else 1) + walls t
  | _ => 0
  end.

Definition allowed (a b tau : bool) : Prop := a <> b \/ tau = a.

Fixpoint gate (i : nat) (tau : bool) (l : list bool) : list bool :=
  match i, l with
  | 0, _ :: _ :: r => tau :: tau :: r
  | S i', x :: r => x :: gate i' tau r
  | _, _ => l
  end.

Definition allowed_at (l : list bool) (i : nat) (tau : bool) : Prop :=
  forall a b, nth_error l i = Some a -> nth_error l (S i) = Some b -> allowed a b tau.

Fixpoint remove_leg (i : nat) (l : list bool) : list bool :=
  match i, l with
  | 0, _ :: r => r
  | S i', x :: r => x :: remove_leg i' r
  | _, [] => []
  end.

Lemma gate_creates_no_wall : forall l i tau, allowed_at l i tau -> walls (gate i tau l) <= walls l.
Proof.
  induction l as [|x t IH]; intros i tau H.
  - destruct i; simpl; lia.
  - destruct i as [|[|i]].
    + (* gate on the first two legs *)
      destruct t as [|y r]; [simpl; lia|].
      specialize (H x y eq_refl eq_refl). unfold allowed in H.
      destruct r as [|z r'];
        destruct x, y, tau; try destruct z; destruct H as [H|H]; try congruence; simpl; lia.
    + (* gate on legs 1 and 2: the head x sees the new spin *)
      destruct t as [|y r]; [simpl; lia|].
      destruct r as [|z r']; [simpl; lia|].
      specialize (H y z eq_refl eq_refl). unfold allowed in H.
      destruct r' as [|w r''];
        destruct x, y, z, tau; try destruct w; destruct H as [H|H]; try congruence; simpl; lia.
    + (* gate further away: the head is unchanged, induction *)
      destruct t as [|y r]; [simpl; lia|].
      change (gate (S (S i)) tau (x :: y :: r)) with (x :: gate (S i) tau (y :: r)).
      change (gate (S i) tau (y :: r)) with (y :: gate i tau r).
      assert (Hr : allowed_at (y :: r) (S i) tau).
      { intros a b Ha Hb. apply H; simpl in *; assumption. }
      specialize (IH (S i) tau Hr).
      change (gate (S i) tau (y :: r)) with (y :: gate i tau r) in IH.
      simpl in IH |- *. lia.
Qed.

Lemma removal_creates_no_wall : forall l i, walls (remove_leg i l) <= walls l.
Proof.
  induction l as [|x t IH]; intros i.
  - destruct i; simpl; lia.
  - destruct i as [|[|i]].
    + destruct t as [|y r]; simpl; [lia|]. destruct r; simpl; lia.
    + (* removal of leg 1: triangle inequality [x<>z] <= [x<>y] + [y<>z] *)
      destruct t as [|y r]; [simpl; lia|].
      destruct r as [|z r']; [simpl; destruct x, y; simpl; lia|].
      destruct r' as [|w r''];
        destruct x, y, z; try destruct w; simpl; lia.
    + destruct t as [|y r]; [simpl; lia|].
      change (remove_leg (S (S i)) (x :: y :: r)) with (x :: remove_leg (S i) (y :: r)).
      change (remove_leg (S i) (y :: r)) with (y :: remove_leg i r).
      specialize (IH (S i)).
      change (remove_leg (S i) (y :: r)) with (y :: remove_leg i r) in IH.
      simpl in IH |- *. lia.
Qed.

Inductive step : list bool -> list bool -> Prop :=
  | step_gate : forall l i tau, allowed_at l i tau -> step l (gate i tau l)
  | step_removal : forall l i, step l (remove_leg i l).

Inductive history : list bool -> list bool -> Prop :=
  | h_nil : forall l, history l l
  | h_cons : forall l l' l'', step l l' -> history l' l'' -> history l l''.

Theorem walls_nonincreasing : forall l l', history l l' -> walls l' <= walls l.
Proof.
  intros l l' H. induction H as [l|l l' l'' Hp Hh IH].
  - lia.
  - inversion Hp; subst.
    + pose proof (gate_creates_no_wall l i tau H). lia.
    + pose proof (removal_creates_no_wall l i). lia.
Qed.

(* Corollary: region A = prefix (a single wall on the top boundary) => at most one wall. *)
Corollary single_wall : forall l l', walls l <= 1 -> history l l' -> walls l' <= 1.
Proof. intros l l' H1 H2. pose proof (walls_nonincreasing l l' H2). lia. Qed.

(* ------------------------------------------------------------------------------------ *)
(* Theorem 2: weight of a bundle of paths.                                               *)
Require Import Reals Binomial Lra.
Open Scope R_scope.

Lemma pow_div_split : forall (q D : R) (l : nat), D <> 0 -> (q / D) ^ l = (1 / D) ^ l * q ^ l.
Proof. intros q D l HD. induction l as [|l IH]; simpl; [ring|]. rewrite IH. field. exact HD. Qed.

(* K^(j+l) q^j = a^j b^l q^l  with K = q/D, a = q*q/D, b = 1/D, D = q*q+1: biased walk *)
Lemma weight_identity : forall (q D : R) (j l : nat), D <> 0 ->
  (q / D) ^ (j + l) * q ^ j = (q * q / D) ^ j * (1 / D) ^ l * q ^ l.
Proof.
  intros q D j l HD. induction j as [|j IH].
  - simpl. rewrite pow_div_split by exact HD. ring.
  - rewrite Nat.add_succ_l. simpl.
    replace (q / D * (q / D) ^ (j + l) * (q * q ^ j))
      with (q * q / D * ((q / D) ^ (j + l) * q ^ j)) by (field; exact HD).
    rewrite IH. ring.
Qed.

Lemma term_le_sum : forall (f : nat -> R) (n k : nat),
  (forall i, 0 <= f i) -> (k <= n)%nat -> f k <= sum_f_R0 f n.
Proof.
  intros f n k Hpos Hk. induction n as [|n IH].
  - assert (k = 0)%nat by lia. subst. simpl. lra.
  - simpl. destruct (Nat.eq_dec k (S n)) as [E|E].
    + subst. assert (0 <= sum_f_R0 f n).
      { clear IH Hk. induction n as [|n IHn]; simpl; [apply Hpos|]. pose proof (Hpos (S n)). lra. }
      lra.
    + assert (k <= n)%nat by lia. pose proof (Hpos (S n)). specialize (IH H). lra.
Qed.

Lemma C_pos : forall n i, 0 <= C n i.
Proof.
  intros n i. unfold C. apply Rmult_le_pos.
  - apply pos_INR.
  - left. apply Rinv_0_lt_compat. apply Rmult_lt_0_compat; apply lt_0_INR; apply lt_O_fact.
Qed.

Theorem path_bundle : forall (q : R) (j l : nat), q > 0 ->
  C (j + l) j * (q / (q * q + 1)) ^ (j + l) * q ^ j <= q ^ l.
Proof.
  intros q j l Hq.
  assert (Hq2 : 0 <= q * q) by nra.
  assert (HD : q * q + 1 <> 0) by lra.
  set (a := q * q / (q * q + 1)). set (b := 1 / (q * q + 1)).
  assert (Ha : 0 <= a) by (unfold a; apply Rmult_le_pos; [lra|left; apply Rinv_0_lt_compat; lra]).
  assert (Hb : 0 <= b) by (unfold b; apply Rmult_le_pos; [lra|left; apply Rinv_0_lt_compat; lra]).
  assert (Hab : a + b = 1) by (unfold a, b; field; lra).
  rewrite Rmult_assoc, (weight_identity q (q * q + 1) j l HD). fold a b.
  assert (Hbin : C (j + l) j * a ^ j * b ^ l <= 1).
  { rewrite <- (pow1 (j + l)), <- Hab, binomial.
    replace l with ((j + l) - j)%nat at 2 by lia.
    apply (term_le_sum (fun i => C (j + l) i * a ^ i * b ^ (j + l - i))).
    - intros i. apply Rmult_le_pos; [apply Rmult_le_pos; [apply C_pos|apply pow_le; lra]|apply pow_le; lra].
    - lia. }
  assert (0 <= q ^ l) by (apply pow_le; lra).
  replace (C (j + l) j * (a ^ j * b ^ l * q ^ l)) with ((C (j + l) j * a ^ j * b ^ l) * q ^ l) by ring.
  rewrite <- (Rmult_1_l (q ^ l)) at 2. apply Rmult_le_compat_r; assumption.
Qed.

(* No preferred direction: (2K)^t = sum of the weights of all paths of t steps;
   2K = 2q/(q*q+1) <= 1 gives the time penalty exp(-t E(0)). *)
Lemma twoK_le_1 : forall q : R, q > 0 -> 2 * (q / (q * q + 1)) <= 1.
Proof.
  intros q Hq. assert (HD : q * q + 1 > 0) by nra.
  apply (Rmult_le_reg_r (q * q + 1)); [lra|].
  replace (2 * (q / (q * q + 1)) * (q * q + 1)) with (2 * q) by (field; lra).
  pose proof (Rle_0_sqr (q - 1)) as Hs. unfold Rsqr in Hs. nra.
Qed.

