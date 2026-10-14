"""Replaces the gate of tableau.PTab by a UNIFORMLY random two-qubit Clifford gate (Sp(4,F2), modulo Paulis).
The original PTab.layer composes SL(2,F2) layers and CNOTs ("rand2"), which is NOT a 2-design.
Import this module BEFORE creating tableaux (and before creating a multiprocessing Pool: child
processes inherit the patch). All results reported in the paper use this patch."""
import numpy as np
import tableau
from clifford_uniform import SP4, CTab

def _layer_unif(self, a, b, rng=None, choices=None, record=True):
    a = np.asarray(a, dtype=np.int64); b = np.asarray(b, dtype=np.int64)
    if len(a) == 0:
        return
    CTab.layer_sym(self, a, b, SP4[rng.integers(720, size=len(a))], record)

tableau.PTab.layer = _layer_unif
tableau.Tab.layer = None          # the unpacked tableau is no longer used
