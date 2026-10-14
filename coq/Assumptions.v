(* Assumptions.v -- prints the axioms on which the main results depend.
   Expected output (Coq 8.18):
   - "Closed under the global context" (no axiom at all) for the combinatorial and
     arithmetic statements: split_*, ParityA1.*, walls_nonincreasing, single_wall;
   - for the real-number lemmas, only axioms that the standard Reals library itself
     relies on: ClassicalDedekindReals.sig_forall_dec, ClassicalDedekindReals.sig_not_dec,
     FunctionalExtensionality.functional_extensionality_dep and, for the three lemmas that
     use exp or ln (fraction_p1_pos, fraction_p1_lt1, butterfly_tension), Classical_Prop.classic. *)
From BOLC Require Import CausalCone ConeLemmas ParityA1 WallDescent WeingartenWeights.

Print Assumptions CausalCone.split_ext.
Print Assumptions CausalCone.split_cone_inside.
Print Assumptions CausalCone.split_out_avoids_A.
Print Assumptions CausalCone.split_run.
Print Assumptions CausalCone.front_threshold.
Print Assumptions CausalCone.fraction_p1_pos.
Print Assumptions CausalCone.fraction_p1_lt1.
Print Assumptions CausalCone.speed_ratio.
Print Assumptions ConeLemmas.two_path_estimate.
Print Assumptions ParityA1.E_max.
Print Assumptions ParityA1.depths.
Print Assumptions ParityA1.cut_gate_exists.
Print Assumptions ParityA1.ray_alive.
Print Assumptions WallDescent.walls_nonincreasing.
Print Assumptions WallDescent.single_wall.
Print Assumptions WallDescent.path_bundle.
Print Assumptions WallDescent.twoK_le_1.
Print Assumptions WeingartenWeights.w_equal.
Print Assumptions WeingartenWeights.w_forbidden.
Print Assumptions WeingartenWeights.w_wall.
Print Assumptions WeingartenWeights.theta1_root.
Print Assumptions WeingartenWeights.butterfly_tension.
