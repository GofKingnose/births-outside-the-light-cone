#!/usr/bin/env python3
"""Proposition A1 (edge growth, 1D, brickwork): the minimal cut of the space-time network equals the number
of cone sites.
(i)  Check by maximum flow (scipy) on the exact DAG, k = 1..6, N <= 400: mincut == |Ā ∩ Inf(A)| ?
(ii) Explicit construction (k = 1 only; the general construction is checked in optimality_A1.py): SWAP
     everywhere, except on the gates that straddle the cut at even depths 2tau, where (H⊗I) then CNOT is
     applied (it makes a Bell pair from |0>|0>). Each pair is carried by the SWAPs along the two light rays
     ("V" shape) up to the top of A and of Ā. If S(A) = |Ā ∩ Inf(A)|, the circuit is a certificate:
     mincut >= S >= cone, hence equality."""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import maximum_flow
from tableau import PTab, infected

def schedule(k, n_final, N0=16):
    """Edge growth: one site added (on the right) every k layers; returns the layers [(a_ids, b_ids)]
    and the final size. Positions = identifiers (appended at the end)."""
    layers, n, L, births = [], N0, 0, {}
    while True:
        if L % k == 0:
            if n == n_final: break
            births[n] = L; n += 1
        par = L % 2
        a = np.arange(par, n - 1, 2); layers.append((a, a + 1)); L += 1
    return layers, n

def cone_count(layers, n):
    m = n // 2; A = np.arange(m); B = np.arange(m, n)
    iA = infected(layers, A, n); iB = infected(layers, B, n)
    return min(int(iA[B].sum()), int(iB[A].sum()), m)

def mincut(layers, n):
    m = n // 2; last = -np.ones(n, dtype=np.int64); rows, cols = [], []; g = 0
    for a, b in layers:
        for x, y in zip(a, b):
            for q in (x, y):
                if last[q] >= 0: rows += [last[q], g]; cols += [g, last[q]]
                last[q] = g
            g += 1
    S, T = g, g + 1; caps = [1] * len(rows); cnt = {}
    for q in range(n):
        if last[q] >= 0:
            key = (last[q], S if q < m else T); cnt[key] = cnt.get(key, 0) + 1
    for (u, t), c in cnt.items(): rows += [u, t]; cols += [t, u]; caps += [c, c]
    M = sparse.coo_matrix((caps, (rows, cols)), shape=(g + 2, g + 2)).tocsr(); M.sum_duplicates()
    return int(maximum_flow(M.astype(np.int32), S, T).flow_value)

class Route(PTab):
    """SWAP everywhere; (H⊗I)·CNOT on the marked gates."""
    def apply(self, a, b, bell):
        w = self._w(); a = np.asarray(a, dtype=np.int64); b = np.asarray(b, dtype=np.int64)
        sw = ~bell
        if sw.any():
            for M in (self.X, self.Z):
                tmp = M[a[sw], :w].copy(); M[a[sw], :w] = M[b[sw], :w]; M[b[sw], :w] = tmp
        if bell.any():
            self._singles(a[bell], np.ones(bell.sum(), dtype=np.int64)); self._cnot(a[bell], b[bell])

def construction(k, n_final, Dmax=None):
    layers, n = schedule(k, n_final); m = n // 2; Lf = len(layers)
    tab = Route(n); N0 = 16
    for _ in range(N0): tab.born()
    L = 0; nb = N0
    for L, (a, b) in enumerate(layers):
        while tab.n < n and tab.n < N0 + L // k + 1: tab.born()
        depth = Lf - L                      # depth of the layer below the final time
        if Dmax is None: Dmax = int(2 * m * k / (k + 1))     # half-pair in Ā: reflected by the moving edge, stays in Ā
        bell = (a == m - 1) & (depth > 0) & (depth <= Dmax)   # half-pair in A: reflected by the wall, stays in A (d < 2m)
        tab.apply(a, b, bell)
    while tab.n < n: tab.born()
    return int(tab.entropy(np.arange(m))), cone_count(layers, n)

if __name__ == '__main__':
    print("(i) maximum flow:")
    for k in range(1, 7):
        line = []
        for nf in (60, 120, 240, 400):
            layers, n = schedule(k, nf); c = cone_count(layers, n); f = mincut(layers, n)
            line.append(f"N={n}: min cut {f} / cone {c}{'' if f == c else '  <-- DIFFERENT'}")
        print(f"  k={k}: " + " | ".join(line), flush=True)
    print("(ii) explicit construction by routed Bell pairs (k = 1):")
    for nf in (60, 120, 240, 400, 800):
        S, c = construction(1, nf); print(f"  N={nf}: S(A) = {S}, cone = {c}{'  OK' if S == c else '  <-- mismatch'}", flush=True)
