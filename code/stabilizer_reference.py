#!/usr/bin/env python3
"""
Minimal stabilizer simulator (phases ignored) for entanglement entropy; used as a
reference to validate the vectorised tableau of tableau.py.

A pure stabilizer state of n qubits is described by n generators, rows (x | z) of an
n x 2n binary matrix. Phases do not affect entropies and are ignored. Entropy (bits)
of a region A:
    S_A = rank_GF2(generators restricted to the columns of A) - |A|.
A birth inserts a qubit in |0>: a zero column plus a Z generator.
"""
import numpy as np

# The 6 elements of SL(2, F2) acting on (x, z) of one qubit
SL2 = [np.array(m, dtype=np.uint8) for m in
       ([[1, 0], [0, 1]], [[0, 1], [1, 0]], [[1, 1], [0, 1]],
        [[1, 0], [1, 1]], [[0, 1], [1, 1]], [[1, 1], [1, 0]])]

class Stab:
    def __init__(self, n):
        self.n = n
        self.X = np.zeros((n, n), dtype=np.uint8)
        self.Z = np.eye(n, dtype=np.uint8)          # |0...0>: generators Z_i

    def insert_fresh(self, pos):
        self.X = np.insert(self.X, pos, 0, axis=1)
        self.Z = np.insert(self.Z, pos, 0, axis=1)
        newx = np.zeros((1, self.n + 1), dtype=np.uint8)
        newz = np.zeros((1, self.n + 1), dtype=np.uint8); newz[0, pos] = 1
        self.X = np.vstack([self.X, newx]); self.Z = np.vstack([self.Z, newz])
        self.n += 1

    def single(self, q, M):
        x, z = self.X[:, q].copy(), self.Z[:, q].copy()
        self.X[:, q] = (M[0, 0] * x + M[0, 1] * z) & 1
        self.Z[:, q] = (M[1, 0] * x + M[1, 1] * z) & 1

    def cnot(self, c, t):
        self.X[:, t] ^= self.X[:, c]
        self.Z[:, c] ^= self.Z[:, t]

    def rand2(self, a, b, rng):
        """Random two-qubit Clifford gate (scrambling composition; NOT a 2-design, see the paper)."""
        for q in (a, b):
            self.single(q, SL2[rng.integers(6)])
        self.cnot(a, b)
        for q in (a, b):
            self.single(q, SL2[rng.integers(6)])
        self.cnot(b, a)
        for q in (a, b):
            self.single(q, SL2[rng.integers(6)])

    def layer(self, parity, rng):
        for a in range(parity, self.n - 1, 2):
            self.rand2(a, a + 1, rng)

    def entropy(self, cols):
        cols = np.asarray(cols)
        M = np.concatenate([self.X[:, cols], self.Z[:, cols]], axis=1).copy()
        return gf2_rank(M) - len(cols)

def gf2_rank(M):
    M = M.copy()
    rows, cols = M.shape
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        piv = np.nonzero(M[r:, c])[0]
        if len(piv) == 0:
            continue
        p = r + piv[0]
        if p != r:
            M[[r, p]] = M[[p, r]]
        mask = M[:, c].astype(bool); mask[r] = False
        M[mask] ^= M[r]
        r += 1
    return r
