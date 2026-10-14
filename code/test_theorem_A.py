#!/usr/bin/env python3
"""Tests: (1) packed GF(2) rank = reference rank; (2) vectorised transposed tableau =
reference simulator stabilizer_reference.Stab on identical circuits with births;
(3) Theorem A on these circuits; (4) Theorem A for Haar gates (state vector, q = 2);
(5) packed tableau = unpacked tableau."""
import sys, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, '.')
from stabilizer_reference import Stab, gf2_rank as rank_ref, SL2 as SL2ref
from tableau import Tab, gf2_rank, SL2, infected, causal_bound
import statevector

rng = np.random.default_rng(1)
# (1)
for _ in range(300):
    r, c = rng.integers(1, 90, size=2)
    M = (rng.random((r, c)) < rng.random()).astype(np.uint8)
    assert gf2_rank(M) == rank_ref(M)
print("(1) packed rank: OK")

# (2)+(3)
viol = 0; checks = 0
for trial in range(40):
    Nmax = 40
    t = Tab(Nmax); s = None
    for _ in range(3):
        t.born()
    s = Stab(3)
    for step in range(60):
        if t.n < Nmax and rng.random() < 0.3:
            t.born(); s.insert_fresh(s.n)
        n = t.n
        perm = rng.permutation(n); m = int(rng.integers(0, n // 2 + 1))
        a, b = perm[:m], perm[m:2 * m]
        ch = rng.integers(6, size=(3, 2 * m))
        t.layer(a, b, choices=ch)
        qs = np.concatenate([a, b])
        for j, q in enumerate(qs): s.single(q, SL2ref[ch[0, j]])
        for i in range(m): s.cnot(a[i], b[i])
        for j, q in enumerate(qs): s.single(q, SL2ref[ch[1, j]])
        for i in range(m): s.cnot(b[i], a[i])
        for j, q in enumerate(qs): s.single(q, SL2ref[ch[2, j]])
        A = rng.permutation(n)[:int(rng.integers(1, n))]
        SA = t.entropy(A)
        assert SA == s.entropy(np.sort(A)), (trial, step)
        B = np.setdiff1d(np.arange(n), A)
        b1, b2 = causal_bound(t.log, A, B, Nmax)
        checks += 1; viol += SA > min(b1, b2)
print(f"(2) vectorised tableau = reference: OK; (3) Theorem A, Clifford: {viol} violation(s) / {checks}")

# (4) Haar, state vector, growth by insertion at random positions
viol = 0; checks = 0; slack = []
for trial in range(30):
    psi = np.zeros((2, 2), complex); psi[0, 0] = 1
    pos = [0, 1]; nid = 2; log = []
    while len(pos) < 14:
        p = int(rng.integers(0, len(pos) + 1))
        psi = statevector.insert_fresh(psi, p); pos.insert(p, nid); nid += 1
        par = int(rng.integers(2))
        aa, bb = [], []
        for x in range(par, len(pos) - 1, 2):
            if rng.random() < 0.7:
                psi = statevector.apply2(psi, statevector.haar2(rng), x, x + 1); aa.append(pos[x]); bb.append(pos[x + 1])
        log.append((np.array(aa, dtype=np.int64), np.array(bb, dtype=np.int64)))
        n = len(pos)
        for m in range(1, n):
            SA = statevector.entropy_cut(psi, m)
            A = pos[:m]; B = pos[m:]
            b1, b2 = causal_bound(log, A, B, 64)
            checks += 1; viol += SA > min(b1, b2) + 1e-9; slack.append(min(b1, b2) - SA)
print(f"(4) Theorem A, Haar: {viol} violation(s) / {checks}; minimal slack = {min(slack):.3e} bit")

# (5) packed tableau = unpacked tableau (identical circuits, births, > 64 qubits)
from tableau import PTab
for trial in range(10):
    Nmax = 150; t1 = Tab(Nmax); t2 = PTab(Nmax)
    for _ in range(5): t1.born(); t2.born()
    for step in range(120):
        if t1.n < Nmax and rng.random() < 0.6: t1.born(); t2.born()
        n = t1.n; perm = rng.permutation(n); m = int(rng.integers(0, n // 2 + 1))
        a, b = perm[:m], perm[m:2 * m]; ch = rng.integers(6, size=(3, 2 * m))
        t1.layer(a, b, choices=ch); t2.layer(a, b, choices=ch)
        if step % 10 == 0:
            A = rng.permutation(n)[:int(rng.integers(1, n))]
            assert t1.entropy(A) == t2.entropy(A)
print("(5) packed tableau = unpacked tableau: OK")
