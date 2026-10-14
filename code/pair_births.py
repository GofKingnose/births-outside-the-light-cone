#!/usr/bin/env python3
"""Births in pairs, edge growth, uniform Clifford gates.
ref: one pure qubit per birth event; P1: Bell pair adjacent at the edge; P2: one member at the edge,
the other inserted at a uniformly random position. k scrambling layers per added DEGREE OF FREEDOM.
Measurement: G = m - S(A), A = left half; slope dG/dn fitted on n in [200, 800]."""
import sys, json, time
import numpy as np
sys.path.insert(0, '.')
from multiprocessing import Pool
from clifford_uniform import CTab, SP4

NMAX = 800

def run(args):
    variant, k, seed = args
    rng = np.random.default_rng(7000 + 100 * k + seed + {'ref': 0, 'P1': 10000, 'P2': 20000}[variant])
    tab = CTab(NMAX + 2)
    pos = []
    def bell(qa, qb):
        tab._singles(np.array([qa]), np.array([1], dtype=np.int64))      # H (swaps x <-> z)
        tab._cnot(np.array([qa]), np.array([qb]))
    q0, q1 = tab.born(), tab.born(); pos += [q0, q1]; bell(q0, q1)       # first birth event (a pair)
    credit, par, rec = 0.0, 0, []
    while len(pos) < NMAX:
        if variant == 'ref':
            pos.append(tab.born()); added = 1
        elif variant == 'P1':
            qa, qb = tab.born(), tab.born(); pos += [qa, qb]; bell(qa, qb); added = 2
        else:
            qa = tab.born(); pos.append(qa)
            qb = tab.born(); pos.insert(int(rng.integers(0, len(pos) + 1)), qb); bell(qa, qb); added = 2
        credit += k * added
        while credit >= 1 - 1e-12:
            n = len(pos)
            a = np.array([pos[x] for x in range(par, n - 1, 2)], dtype=np.int64)
            tab.layer_sym(a, np.array([pos[x + 1] for x in range(par, n - 1, 2)], dtype=np.int64),
                          SP4[rng.integers(720, size=len(a))], record=False)
            par ^= 1; credit -= 1
        n = len(pos)
        if n % 20 == 0 or n % 20 == 1:
            if not rec or rec[-1][0] // 20 != n // 20:
                m = n // 2
                rec.append((n, int(m - tab.entropy(np.array(pos[:m], dtype=np.int64)))))
    return variant, k, seed, rec

if __name__ == '__main__':
    t = time.time()
    jobs = [(v, k, s) for k in (0, 1, 2, 4, 8, 16, 32) for v in ('ref', 'P1', 'P2') for s in range(8)]
    out = []
    with Pool(2) as P:
        for r in P.imap_unordered(run, jobs):
            out.append(r); json.dump(out, open('../data/pair_births.json', 'w'))
    res = {}
    for v, k, s, rec in out:
        n = np.array([a for a, b in rec], float); G = np.array([b for a, b in rec], float)
        sel = (n >= 200) & (n <= NMAX)
        res.setdefault((v, k), []).append(np.polyfit(n[sel], G[sel], 1)[0])
    for k in (0, 1, 2, 4, 8, 16, 32):
        line = f"k={k:2d} "
        for v in ('ref', 'P1', 'P2'):
            x = np.array(res[(v, k)]); line += f"| {v}: {x.mean():.4f} ± {x.std()/np.sqrt(len(x)):.4f} "
        r = np.mean(res[('ref', k)]) / np.mean(res[('P2', k)])
        print(line + f"| R/P2 = {r:.2f}", flush=True)
    print(f"[{time.time()-t:.0f}s]")
