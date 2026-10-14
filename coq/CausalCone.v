(* ======================================================================
   CausalCone.v -- combinatorial part of Theorem A and elementary lemmas
   (paper "Births outside the light cone", Sections 3 and 4, Appendix B).

   What is formalised:
   (I)  For an ARBITRARY circuit (gates of arbitrary support, on qudits
        labelled by integers, which may be "born" at any time) and a region
        A, the backward sweep splits the circuit into cone gates and
        out-of-cone gates, and:
          - A is contained in the infected set C;
          - every cone gate acts inside C;
          - no out-of-cone gate acts on A;
          - the circuit is EQUAL to "cone gates, then out-of-cone gates",
            for every semantics in which gates with disjoint supports
            commute (unitaries, isometries, channels...).
   (II) Real-number lemmas: criterion for a site to be reached by the cone
        (front), positivity of the asymptotic unreached fraction for N ~ t.

   What is NOT formalised: the final quantum step (pure state on C, hence
   S(A) = S(C \ A) <= |C \ A| log q), which is the standard Schmidt
   decomposition.
   ====================================================================== *)

Require Import List Bool Arith Lia.
Import ListNotations.

Definition gate := list nat.                  (* support of the gate *)
Definition qset := nat -> bool.               (* set of qudits *)

Definition touches (C : qset) (g : gate) : bool := existsb C g.
Definition add (C : qset) (g : gate) : qset :=
  fun q => C q || existsb (Nat.eqb q) g.
Definition disjoint (g h : gate) : Prop := forall q, In q g -> ~ In q h.

(* Splitting by backward sweep. The circuit is listed in temporal order
   (head = first gate applied); the tail is processed first. *)
Fixpoint split (c : list gate) (A : qset) : list gate * list gate * qset :=
  match c with
  | [] => ([], [], A)
  | g :: rest =>
      let '(cn, out, C) := split rest A in
      if touches C g then (g :: cn, out, add C g) else (cn, g :: out, C)
  end.

Lemma touches_false : forall C g, touches C g = false ->
  forall q, In q g -> C q = false.
Proof.
  intros C g H q Hq. unfold touches in H.
  destruct (C q) eqn:E; [|reflexivity].
  assert (existsb C g = true) by (apply existsb_exists; exists q; auto).
  congruence.
Qed.

Lemma add_contains : forall C g q, In q g -> add C g q = true.
Proof.
  intros C g q Hq. unfold add. apply orb_true_iff. right.
  apply existsb_exists. exists q. split; [assumption|]. apply Nat.eqb_refl.
Qed.

Lemma add_ext : forall C g q, C q = true -> add C g q = true.
Proof. intros. unfold add. rewrite H. reflexivity. Qed.

(* (I.1) A is contained in C *)
Lemma split_ext : forall c A cn out C, split c A = (cn, out, C) ->
  forall q, A q = true -> C q = true.
Proof.
  induction c as [|g rest IH]; intros A cn out C Hs q Hq; simpl in Hs.
  - injection Hs as <- <- <-. assumption.
  - destruct (split rest A) as [[cn' out'] C'] eqn:E.
    destruct (touches C' g); injection Hs as <- <- <-.
    + apply add_ext. exact (IH A cn' out' C' E q Hq).
    + exact (IH A cn' out' C' E q Hq).
Qed.

(* (I.2) every cone gate acts inside C *)
Lemma split_cone_inside : forall c A cn out C, split c A = (cn, out, C) ->
  forall g, In g cn -> forall q, In q g -> C q = true.
Proof.
  induction c as [|h rest IH]; intros A cn out C Hs g Hg q Hq; simpl in Hs.
  - injection Hs as <- <- <-. destruct Hg.
  - destruct (split rest A) as [[cn' out'] C'] eqn:E.
    destruct (touches C' h); injection Hs as <- <- <-.
    + destruct Hg as [<-|Hg].
      * apply add_contains; assumption.
      * apply add_ext. exact (IH A cn' out' C' E g Hg q Hq).
    + exact (IH A cn' out' C' E g Hg q Hq).
Qed.

(* (I.3) no out-of-cone gate acts on A *)
Lemma split_out_avoids_A : forall c A cn out C, split c A = (cn, out, C) ->
  forall g, In g out -> forall q, In q g -> A q = false.
Proof.
  induction c as [|h rest IH]; intros A cn out C Hs g Hg q Hq; simpl in Hs.
  - injection Hs as <- <- <-. destruct Hg.
  - destruct (split rest A) as [[cn' out'] C'] eqn:E.
    destruct (touches C' h) eqn:T; injection Hs as <- <- <-.
    + exact (IH A cn' out' C' E g Hg q Hq).
    + destruct Hg as [<-|Hg].
      * destruct (A q) eqn:Aq; [|reflexivity].
        pose proof (split_ext rest A cn' out' C' E q Aq) as HC.
        rewrite (touches_false C' h T q Hq) in HC. discriminate.
      * exact (IH A cn' out' C' E g Hg q Hq).
Qed.

(* (I.4) reordering: abstract semantics in which gates with disjoint
   supports commute. *)
Section Reorder.
  Variable State : Type.
  Variable apply : gate -> State -> State.
  Hypothesis comm : forall g h s, disjoint g h -> apply g (apply h s) = apply h (apply g s).

  Fixpoint run (c : list gate) (s : State) : State :=
    match c with [] => s | g :: r => run r (apply g s) end.

  Lemma commute_through : forall cn g s, (forall h, In h cn -> disjoint g h) ->
    run cn (apply g s) = apply g (run cn s).
  Proof.
    induction cn as [|h cn IH]; intros g s H; simpl; [reflexivity|].
    rewrite <- comm by (apply H; left; reflexivity).
    apply IH. intros h' Hh'. apply H. right. assumption.
  Qed.

  Theorem split_run : forall c A cn out C, split c A = (cn, out, C) ->
    forall s, run c s = run out (run cn s).
  Proof.
    induction c as [|g rest IH]; intros A cn out C Hs s; simpl in Hs.
    - injection Hs as <- <- <-. reflexivity.
    - destruct (split rest A) as [[cn' out'] C'] eqn:E.
      destruct (touches C' g) eqn:T; injection Hs as <- <- <-; simpl.
      + exact (IH A cn' out' C' E (apply g s)).
      + rewrite (IH A cn' out' C' E (apply g s)).
        f_equal. apply commute_through.
        intros h Hh q Hqg Hqh.
        pose proof (split_cone_inside rest A cn' out' C' E h Hh q Hqh) as HC.
        rewrite (touches_false C' g T q Hqg) in HC. discriminate.
  Qed.
End Reorder.

(* ---------------------------------------------------------------------- *)
Require Import Reals Lra.
Open Scope R_scope.

(* (II.1) Front of speed s, cone of speed v, lam = v/s. A site at distance r
   from the centre, with direction cosine u > 0 with respect to the normal
   of the cut, born at t_b = r/s, is NOT reached at T = R/s iff
   r u > lam (R - r)  <=>  r > lam R / (lam + u). *)
Lemma front_not_reached : forall lam u r R0, 0 < lam -> 0 <= u ->
  (r * u > lam * (R0 - r) <-> r * (lam + u) > lam * R0).
Proof. intros. split; intro; nra. Qed.

Lemma front_threshold : forall lam u r R0, 0 < lam -> 0 <= u ->
  (r * (lam + u) > lam * R0 <-> r > lam * R0 / (lam + u)).
Proof.
  intros lam u r R0 Hl Hu.
  assert (Hp : 0 < lam + u) by lra.
  split; intro H.
  - apply Rmult_lt_reg_r with (lam + u); [assumption|].
    unfold Rdiv. rewrite Rmult_assoc, Rinv_l by lra. lra.
  - apply Rmult_lt_compat_r with (r := lam + u) in H; [|assumption].
    unfold Rdiv in H. rewrite Rmult_assoc, Rinv_l in H by lra. lra.
Qed.

(* (II.2) Homogeneous 1D expansion with N ~ t (c = v tau / N0 > 0):
   asymptotic unreached fraction f(c) = 1 - 2c (1 - exp(-1/(2c))),
   strictly between 0 and 1. *)
Lemma fraction_p1_pos : forall c, 0 < c ->
  0 < 1 - 2 * c * (1 - exp (- (1 / (2 * c)))).
Proof.
  intros c Hc. set (x := 1 / (2 * c)).
  assert (Hx : 0 < x) by (unfold x; apply Rdiv_lt_0_compat; lra).
  assert (H1 : 1 + - x < exp (- x)) by (apply exp_ineq1; lra).
  assert (H2 : 2 * c * x = 1) by (unfold x; field; lra).
  nra.
Qed.

Lemma fraction_p1_lt1 : forall c, 0 < c ->
  1 - 2 * c * (1 - exp (- (1 / (2 * c)))) < 1.
Proof.
  intros c Hc. set (x := 1 / (2 * c)).
  assert (Hx : 0 < x) by (unfold x; apply Rdiv_lt_0_compat; lra).
  assert (H1 : exp (- x) < exp 0) by (apply exp_increasing; lra).
  rewrite exp_0 in H1. nra.
Qed.

(* (II.3) Constant global rate in dimension 2: radius R = sqrt(g t / pi),
   edge speed dR/dt = g / (2 pi R) -> 0. The ratio (causal speed) /
   (edge speed) = 2 pi R v / g grows without bound: algebraic identity. *)
Lemma speed_ratio : forall g v R0, 0 < g -> 0 < R0 ->
  v / (g / (2 * PI * R0)) = 2 * PI * R0 * v / g.
Proof. intros. assert (PI > 0) by apply PI_RGT_0. field. lra. Qed.
