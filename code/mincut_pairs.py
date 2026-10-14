#!/usr/bin/env python3
"""
Non-local pairs analysed by MINIMAL CUT (limit of large local dimension).

The circuit is seen as a tensor network: each gate = a node, each world-line segment = an edge of
capacity 1, each Bell pair = a node joining the two qubits, a fresh qubit |0> = a free leg (rank 1,
carries no entanglement). The entropy of a cut is bounded by (and, for Haar gates of large local
dimension, equal to) the minimal cut separating the left output legs from the right ones
(Nahum-Ruhman-Vijay-Haah 2017).

Question: does the geometry of minimal cuts alone, with Bell pairs acting as shortcuts, reproduce the
weakening of the arrow by non-local pairs observed on qubits (Table 5)? If yes, the effect is purely
geometric (shortcuts in the network); if not, it depends on the finite-dimensional dynamics (q = 2).
"""
import sys, json, time
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import maximum_flow
from multiprocessing import Pool

class Net:
    def __init__(self):
        self.nn = 0; self.E = []; self.last = []   # last[i] = node of the last tensor on qubit i (None = fresh)
    def node(self):
        self.nn += 1; return self.nn - 1
    def insert(self, pos):
        self.last.insert(pos, None)
    def bell(self, a, b):
        u = self.node(); self.last[a] = u; self.last[b] = u
    def gate(self, a, b):
        g = self.node()
        for q in (a, b):
            if self.last[q] is not None:
                self.E.append((self.last[q], g))
        self.last[a] = g; self.last[b] = g
    def layer(self, par):
        n = len(self.last)
        for a in range(par, n - 1, 2):
            self.gate(a, a + 1)
    def cut(self, m):
        n = len(self.last); S, T = self.nn, self.nn + 1
        cnt = {}
        for q in range(n):
            u = self.last[q]
            if u is None:
                continue
            key = (u, S if q < m else T)
            cnt[key] = cnt.get(key, 0) + 1
        rows, cols, caps = [], [], []
        for (u, v) in self.E:
            rows += [u, v]; cols += [v, u]; caps += [1, 1]
        for (u, t), c in cnt.items():
            rows += [u, t]; cols += [t, u]; caps += [c, c]
        N = self.nn + 2
        A = sparse.coo_matrix((caps, (rows, cols)), shape=(N, N)).tocsr()
        A.sum_duplicates()
        return maximum_flow(A.astype(np.int32), S, T).flow_value

def run(args):
    variant, k, seed, nmax = args
    rng = np.random.default_rng(seed)
    net = Net(); net.insert(0); net.insert(1); net.bell(0, 1)
    credit, par, rec, nextm = 0.0, 0, [], 40
    while len(net.last) < nmax:
        n = len(net.last)
        if variant == 'ref':
            net.insert(n); added = 1
        elif variant == 'P1':
            net.insert(n); net.insert(n + 1); net.bell(n, n + 1); added = 2
        else:
            net.insert(n); p2 = int(rng.integers(0, n + 2)); net.insert(p2)
            a = n + 1 if p2 <= n else n; net.bell(a, p2); added = 2
        credit += k * added
        while credit >= 1 - 1e-12:
            net.layer(par); par ^= 1; credit -= 1
        if len(net.last) >= nextm:
            m = len(net.last) // 2
            rec.append((len(net.last), m - net.cut(m))); nextm += 20
    return variant, k, seed, rec

if __name__ == '__main__':
    t = time.time(); out = {}
    jobs = [(v, k, s, 160) for v in ('ref', 'P1', 'P2') for k in (1, 2, 4, 8, 16) for s in range(3)]
    with Pool(2) as p:
        res = p.map(run, jobs)
    for v in ('ref', 'P1', 'P2'):
        out[v] = {}
        for k in (1, 2, 4, 8, 16):
            sl = []
            for vv, kk, ss, rec in res:
                if vv == v and kk == k:
                    ns = np.array([x[0] for x in rec]); G = np.array([x[1] for x in rec]); sel = ns >= 60
                    sl.append(np.polyfit(ns[sel], G[sel], 1)[0])
            out[v][k] = float(np.mean(sl))
    print(f"{'k':>3} {'ref':>8} {'P1':>8} {'P2':>8} {'ref/P2':>8}   (minimal cut, n <= 160)")
    for k in (1, 2, 4, 8, 16):
        r, p1, p2 = out['ref'][k], out['P1'][k], out['P2'][k]
        print(f"{k:3d} {r:8.4f} {p1:8.4f} {p2:8.4f} {r/p2 if p2>1e-9 else float('inf'):8.2f}")
    json.dump(out, open('../data/mincut_pairs.json', 'w'), indent=1)
    print(f"[{time.time()-t:.0f}s]")
