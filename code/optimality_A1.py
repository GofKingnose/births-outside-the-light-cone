#!/usr/bin/env python3
"""Proposition A1 — check of the exact discrete formulas (edge growth, 1D, brickwork):
   L_f = k (n - N0),  delta0 = (m - L_f) mod 2,  E = floor((k(m+1) - delta0)/(k+1)),  D* = 2E - 1 + delta0,
and of the construction (SWAP everywhere, Bell pair on the cut gates of depth <= D*): S(A) = E."""
import numpy as np
from tableau import PTab, infected

def schedule(k, n_final, N0):
    layers, n, L = [], N0, 0
    while True:
        if L % k == 0:
            if n == n_final: break
            n += 1
        a = np.arange(L % 2, n - 1, 2); layers.append((a, a + 1)); L += 1
    return layers, n

class Route(PTab):
    def apply(self, a, b, bell):
        w = self._w(); sw = ~bell
        if sw.any():
            for M in (self.X, self.Z):
                tmp = M[a[sw], :w].copy(); M[a[sw], :w] = M[b[sw], :w]; M[b[sw], :w] = tmp
        if bell.any():
            self._singles(a[bell], np.ones(bell.sum(), dtype=np.int64)); self._cnot(a[bell], b[bell])

def check(k, n, N0):
    layers, n = schedule(k, n, N0); m = n // 2; Lf = len(layers)
    assert Lf == k * (n - N0)
    A, B = np.arange(m), np.arange(m, n)
    cone_B = int(infected(layers, A, n)[B].sum()); cone_A = int(infected(layers, B, n)[A].sum())
    d0 = (m - Lf) % 2; E = (k * (m + 1) - d0) // (k + 1); Dst = 2 * E - 1 + d0
    tab = Route(n)
    for _ in range(N0): tab.born()
    for L, (a, b) in enumerate(layers):
        while tab.n < min(n, N0 + L // k + 1): tab.born()
        bell = (a == m - 1) & (Lf - L <= Dst)
        tab.apply(a, b, bell)
    while tab.n < n: tab.born()
    S = int(tab.entropy(A))
    return dict(E_cone=min(cone_B, cone_A, m), cone_B=cone_B, E_formula=E, S=S, d0=d0)

if __name__ == '__main__':
    bad, tot, d0s = 0, 0, set()
    for k in range(1, 9):
        for N0 in (4, 10, 16):
            for m in range(max(N0, k + 1), 61, 3):
                r = check(k, 2 * m, N0); tot += 1; d0s.add(r['d0'])
                ok = r['E_cone'] == r['cone_B'] == r['E_formula'] == r['S']
                if not ok:
                    bad += 1; print("FAILURE", k, N0, m, r)
    print(f"{tot} cases (k = 1..8, N0 in {{4,10,16}}, m = N0..60, delta0 in {sorted(d0s)}): {bad} failure(s)")
