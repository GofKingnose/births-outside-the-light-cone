#!/usr/bin/env python3
"""Exploratory test (not pre-registered): is Theorem A saturated by DUAL-UNITARY circuits?
Gate: (u1⊗u2) · SWAP · CZ · (u3⊗u4), u_i random single-qubit Cliffords (dual-unitary, maximally entangling).
Comparison of G/N (dual-unitary) with G_LB/N (exact bound)."""
import numpy as np, json
from tableau import PTab, causal_bound
from growth_1d import birth_times

class DU(PTab):
    def layer(self, a, b, rng=None, choices=None, record=True):
        a = np.asarray(a, dtype=np.int64); b = np.asarray(b, dtype=np.int64)
        if len(a) == 0: return
        qs = np.concatenate([a, b]); w = self._w()
        self._singles(qs, rng.integers(6, size=len(qs)))
        H = np.ones(len(b), dtype=np.int64)              # SL2[1] = Hadamard (x <-> z)
        self._singles(b, H); self._cnot(a, b); self._singles(b, H)          # CZ
        for M in (self.X, self.Z):                        # SWAP
            tmp = M[a, :w].copy(); M[a, :w] = M[b, :w]; M[b, :w] = tmp
        self._singles(qs, rng.integers(6, size=len(qs)))
        if record: self.log.append((a.copy(), b.copy()))

def run(mode, seed, Nmax, cls):
    rng = np.random.default_rng(seed); N0 = 16; tb = birth_times(mode, Nmax)
    tab = cls(Nmax); pos = [tab.born() for _ in range(N0)]; nb, L, rec = 0, 0, []
    checks = [c for c in (100, 200, 400, 800) if c <= Nmax]
    while checks:
        while nb < len(tb) and tb[nb] <= L:
            q = tab.born(); p = len(pos) if mode.startswith('edge') else int(rng.integers(0, len(pos) + 1))
            pos.insert(p, q); nb += 1
            if mode.endswith('pair') and nb < len(tb):     # insertion by adjacent pairs (parity preserved)
                pos.insert(p + 1, tab.born()); nb += 1
            if checks and len(pos) == checks[0]:
                n = len(pos); A, B = pos[:n // 2], pos[n // 2:]; S = tab.entropy(A); b1, b2 = causal_bound(tab.log, A, B, Nmax)
                rec.append((n, n // 2 - S, n // 2 - min(b1, b2, n // 2), S > min(b1, b2))); checks.pop(0)
        par = L % 2; n = len(pos)
        tab.layer([pos[x] for x in range(par, n - 1, 2)], [pos[x + 1] for x in range(par, n - 1, 2)], rng); L += 1
    return rec

if __name__ == '__main__':
    out = {}
    for mode in ('edge-1', 'edge-4', 'p1', 'p0.5'):
        Nm = 800 if mode != 'p0.5' else 400
        R = [run(mode, s, Nm, DU) for s in range(3)]
        Ns = [r[0] for r in R[0]]
        G = np.mean([[r[1] for r in rec] for rec in R], 0); L = np.mean([[r[2] for r in rec] for rec in R], 0)
        v = sum(r[3] for rec in R for r in rec)
        out[mode] = dict(N=Ns, G=G.tolist(), G_LB=L.tolist())
        print(f"{mode:6s} dual-unitary: " + " | ".join(f"N={n} G/N={g/n:.4f} G_LB/N={l/n:.4f}" for n, g, l in zip(Ns, G, L)) + f"   violations={v}", flush=True)
    json.dump(out, open('../data/dual_unitary.json', 'w'), indent=1)
