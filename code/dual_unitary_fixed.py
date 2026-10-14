#!/usr/bin/env python3
"""Which class of gates saturates Theorem A EXACTLY (edge growth, 1D)?  (Remark on structured dynamics.)
Deterministic variants (no randomness): V0 = SWAP·CZ; V1 = (H⊗H)·SWAP·CZ (self-dual Ising, Clifford point);
V2 = SWAP·CZ·(H⊗I) (asymmetric dressing); and the randomly dressed dual-unitary Clifford gate (DU).
For each checkpoint, G - G_LB is printed (0 = exact saturation)."""
import numpy as np
from tableau import PTab, causal_bound
from dual_unitary import DU

class Fixed(PTab):
    pre, post = (), ()
    def layer(self, a, b, rng=None, choices=None, record=True):
        a = np.asarray(a, dtype=np.int64); b = np.asarray(b, dtype=np.int64)
        if len(a) == 0: return
        w = self._w(); one = lambda q: np.ones(len(q), dtype=np.int64)
        for who in self.pre:
            q = a if who == 'a' else b; self._singles(q, one(q))
        self._singles(b, one(b)); self._cnot(a, b); self._singles(b, one(b))       # CZ
        for M in (self.X, self.Z):                                                   # SWAP
            tmp = M[a, :w].copy(); M[a, :w] = M[b, :w]; M[b, :w] = tmp
        for who in self.post:
            q = a if who == 'a' else b; self._singles(q, one(q))
        if record: self.log.append((a.copy(), b.copy()))

def make(pre, post):
    return type('F', (Fixed,), dict(pre=pre, post=post))

def edge_run(cls, k, Nmax=800, seed=0):
    rng = np.random.default_rng(seed); tab = cls(Nmax); pos = [tab.born() for _ in range(16)]
    L, out = 0, []; checks = [100, 200, 400, 800]
    while checks:
        if L % k == 0 and len(pos) < Nmax:
            pos.append(tab.born())
            if len(pos) == checks[0]:
                n = len(pos); A, B = pos[:n // 2], pos[n // 2:]; S = tab.entropy(A); b1, b2 = causal_bound(tab.log, A, B, Nmax)
                out.append((n, (n // 2 - S) - (n // 2 - min(b1, b2, n // 2)))); checks.pop(0)
        par = L % 2; n = len(pos)
        tab.layer([pos[x] for x in range(par, n - 1, 2)], [pos[x + 1] for x in range(par, n - 1, 2)], rng); L += 1
    return out

if __name__ == '__main__':
    variants = {'V0 SWAP·CZ': make((), ()), 'V1 (H⊗H)·SWAP·CZ': make((), ('a', 'b')),
                'V2 SWAP·CZ·(H⊗I)': make(('a',), ()), 'random DU': DU}
    for name, cls in variants.items():
        for k in (1, 2, 3, 4):
            seeds = range(3) if name == 'random DU' else range(1)
            res = [edge_run(cls, k, seed=s) for s in seeds]
            print(f"{name:20s} k={k}: G - G_LB = " + " | ".join(f"N={r[0]}: {[x[i][1] for x in res]}" for i, r in enumerate(res[0])), flush=True)
