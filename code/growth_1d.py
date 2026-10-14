#!/usr/bin/env python3
"""
1D growth classes and exact check of Theorem A (run through growth_campaigns.py, which applies
uniform_gate_patch so that all gates are uniformly random two-qubit Cliffords).

Modes:
  edge-k            : one site added at the right edge every k layers;
  p0.5, p1, p2, exp : insertion at a uniformly random position, N(t) = N0 (1+t/tau)^p or N0 e^{t/tau},
                      tau = c N0 (strict causal speed v = 1 site/layer), N0 = 16, c = 1/2;
  p1pair            : as p1, but insertions come in adjacent pairs (parity preserved).
One layer = one unit of time (brickwork, alternating parity). At each checkpoint N:
A = left half, exact S(A) (stabilizer), exact causal bounds.
"""
import sys, json, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, '.')
from tableau import PTab as Tab, causal_bound

CHECK = (50, 100, 200, 400, 800, 1600)

def birth_times(mode, Nmax, N0=16, c=0.5):
    tau = c * N0
    j = np.arange(N0 + 1, Nmax + 1, dtype=float)
    if mode.startswith('edge'):
        k = int(mode.split('-')[1]); return (j - N0) * k
    if mode == 'exp':
        return tau * np.log(j / N0)
    p = float(mode[1:].replace('pair', ''))
    return tau * ((j / N0) ** (1 / p) - 1)

def run(args):
    mode, seed, Nmax = args
    rng = np.random.default_rng(seed)
    N0 = 16
    tb = birth_times(mode, Nmax)
    tab = Tab(Nmax)
    pos = [tab.born() for _ in range(N0)]
    nb, L, rec = 0, 0, []
    checks = [c for c in CHECK if c <= Nmax]
    t0 = time.time()
    while True:
        while nb < len(tb) and tb[nb] <= L:
            q = tab.born()
            p = len(pos) if mode.startswith('edge') else int(rng.integers(0, len(pos) + 1))
            pos.insert(p, q); nb += 1
            if mode.endswith('pair') and nb < len(tb):     # insertion by adjacent pairs: parity preserved
                pos.insert(p + 1, tab.born()); nb += 1
            if checks and len(pos) == checks[0]:
                n = len(pos); A = pos[:n // 2]; B = pos[n // 2:]
                S = tab.entropy(A)
                b1, b2 = causal_bound(tab.log, A, B, Nmax)
                m = n // 2
                rec.append(dict(n=n, t=L, S=int(S), G=int(m - S), bA=b1, bB=b2,
                                G_LB=int(m - min(b1, b2, m)), viol=bool(S > min(b1, b2))))
                checks.pop(0)
        if not checks:
            break
        par = L % 2; n = len(pos)
        a = [pos[x] for x in range(par, n - 1, 2)]; b = [pos[x + 1] for x in range(par, n - 1, 2)]
        tab.layer(a, b, rng)
        L += 1
    return mode, seed, rec, time.time() - t0

