#!/usr/bin/env python3
"""Butterfly velocity v_B for uniform Clifford gates: right front of a local Pauli operator.
Chain of L sites, initial operator = random non-trivial Pauli at site L/4; Heisenberg evolution
(symplectic action) in brickwork; position of the rightmost non-trivial site after t layers."""
import numpy as np, json
from clifford_uniform import SP4
L, T, SEEDS = 800, 300, 200
rng = np.random.default_rng(424242)
fronts = np.zeros((SEEDS, T))
for s in range(SEEDS):
    x = np.zeros(L, np.uint8); z = np.zeros(L, np.uint8)
    c = L // 4
    while x[c] == 0 and z[c] == 0:
        x[c], z[c] = rng.integers(2), rng.integers(2)
    for t in range(T):
        a = np.arange(t % 2, L - 1, 2); b = a + 1
        M = SP4[rng.integers(720, size=len(a))].astype(np.int32)
        V = np.stack([x[a], z[a], x[b], z[b]], axis=1).astype(np.int32)        # (G, 4)
        W = np.einsum('gij,gj->gi', M, V) & 1
        x[a], z[a], x[b], z[b] = W[:, 0], W[:, 1], W[:, 2], W[:, 3]
        nz = np.flatnonzero(x | z); fronts[s, t] = nz.max()
mean = fronts.mean(0); tt = np.arange(1, T + 1)
sel = tt >= 50
vB = np.polyfit(tt[sel], mean[sel], 1)[0]
# error by resampling
boot = [np.polyfit(tt[sel], fronts[rng.integers(SEEDS, size=SEEDS)].mean(0)[sel], 1)[0] for _ in range(200)]
print(f"v_B (uniform Clifford) = {vB:.4f} ± {np.std(boot):.4f} site/layer  (exact for a 2-design: 0.6)")
json.dump(dict(vB=vB, err=float(np.std(boot)), mean=mean.tolist()), open('../data/butterfly_velocity.json', 'w'))
