(* ======================================================================
   ParityA1.v -- exact arithmetic bookkeeping of Proposition A1
   (edge growth, 1D, brickwork circuit, cut in the middle).

   Notation (integers):
     k  >= 1: layers per birth;  N0: initial sites;  n = 2m;
     Lf = k (2m - N0): number of layers;  d0 in {0,1}: parity offset
     (d0 = (m - Lf) mod 2);  E = floor((k(m+1) - d0)/(k+1));  D* = 2E - 1 + d0.
   The ray dynamics (brickwork of SWAPs) is NOT formalised here: only the
   arithmetic inequalities on which the proof relies.
   ====================================================================== *)

Require Import ZArith Lia.
Open Scope Z_scope.

Section A1.
  Variables k m N0 d0 : Z.
  Hypothesis Hk : 1 <= k.
  Hypothesis Hkm : k < m.
  Hypothesis HN0 : 0 <= N0 <= m.
  Hypothesis Hd0 : d0 = 0 \/ d0 = 1.

  Definition Lf := k * (2 * m - N0).
  Definition E := (k * (m + 1) - d0) / (k + 1).
  Definition Dst := 2 * E - 1 + d0.

  (* Bracketing of the integer part. *)
  Lemma E_bounds : (k + 1) * E <= k * (m + 1) - d0 < (k + 1) * (E + 1).
  Proof.
    unfold E.
    pose proof (Z.div_mod (k * (m + 1) - d0) (k + 1) ltac:(lia)) as H.
    pose proof (Z.mod_pos_bound (k * (m + 1) - d0) (k + 1) ltac:(lia)) as Hr.
    nia.
  Qed.

  (* (1) The e-th widening of the cone (cone edge on site m-1+e, at layer
     Lf - d0 - e) requires this site to be born: k(m-1+e-N0) <= layer. *)
  Lemma alive_iff : forall e, 1 <= e ->
    (k * (m - 1 + e - N0) <= Lf - d0 - e <-> (k + 1) * e <= k * (m + 1) - d0).
  Proof. intros e He. unfold Lf. split; intro; nia. Qed.

  (* (2) These widenings exist exactly for e = 1..E: the cone contains E sites of the complement of A. *)
  Lemma E_max : forall e, 1 <= e -> ((k + 1) * e <= k * (m + 1) - d0 <-> e <= E).
  Proof. intros e He. pose proof E_bounds. split; intro; nia. Qed.

  (* (3) Bijection: pair j (1 <= j <= E) is created at depth 2j - 1 + d0 <= D*. *)
  Lemma depths : forall j, (1 <= j <= E) <-> (1 <= 2 * j - 1 + d0 <= Dst).
  Proof. intro j. unfold Dst. lia. Qed.

  (* (4) The layer of edge gate e and that of the associated Bell gate differ
     by e - 1 layers (right ray of length e - 1). *)
  Lemma layer_gap : forall e,
    (Lf - d0 - e) - (Lf - (2 * e - 1 + d0)) = e - 1.
  Proof. intro e. lia. Qed.

  (* (5) Half-pair B: final position >= m (stays in the complement of A). r = remaining layers. *)
  Lemma half_B : forall e, 1 <= e <= E ->
    let r := d0 + e - 1 in
    (r = 0 /\ m - 1 + e = m) \/ (r >= 1 /\ m - 1 + e - (r - 1) >= m).
  Proof. intros e He r. unfold r. lia. Qed.

  (* (6) E <= m - 1, hence D* <= 2m - 1: the A half-pairs do not come back
     to the cut before the final time, and stay in A. *)
  Lemma E_lt_m : E <= m - 1.
  Proof. pose proof E_bounds. nia. Qed.

  Lemma Dst_le : Dst <= 2 * m - 1.
  Proof. pose proof E_lt_m. unfold Dst. lia. Qed.

  Lemma half_A : forall d, 1 <= d <= Dst ->
    (d <= m /\ 0 <= m - d) \/ (m < d /\ d - m - 1 <= m - 1).
  Proof. intros d Hd. pose proof Dst_le. lia. Qed.

  (* (7) The deepest cut gate exists: site m is born before layer Lf - D*,
     i.e. D* <= k m. *)
  Lemma cut_gate_exists : Dst <= k * m.
  Proof.
    pose proof E_bounds as [HE _]. unfold Dst.
    assert (H1 : 0 <= (k - 1) * (k * m - 1 - d0)) by (apply Z.mul_nonneg_nonneg; nia).
    assert (H2 : (k + 1) * (2 * E - 1 + d0) <= (k + 1) * (k * m)) by nia.
    apply Z.mul_le_mono_pos_l in H2; lia.
  Qed.

  (* (8) Monotonicity: if site m-1+e is alive at the layer of the edge gate,
     all intermediate sites are alive along the right ray. *)
  Lemma ray_alive : forall Lc j1 j2, j1 <= j2 ->
    k * (m + j1 - N0) - (Lc + j1) <= k * (m + j2 - N0) - (Lc + j2).
  Proof. intros. nia. Qed.
End A1.
