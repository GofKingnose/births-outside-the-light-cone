#!/usr/bin/env python3
"""
Stabilizer-tableau library for growing registers and the causal bound.

* TRANSPOSED, pre-allocated stabilizer tableau: X[q, g], Z[q, g]
  (q = qubit identifier, g = generator). Qubits receive identifiers in order of
  birth; spatial positions are handled separately (a list of positions in 1D,
  lattice coordinates in 2D).
  Birth of qubit q: new generator Z_q (state |0>, product with the rest).
* VECTORISED layers of random Clifford gates (disjoint gates).
* GF(2) rank on packed bits (uint64).
* Gate log per sub-layer and backward sweep of the causal cone:
  Inf(A) = qubits whose world line meets the causal past of A.
  Theorem A:  S(A) <= |Ā ∩ Inf(A)|  (bits, q = 2), for ANY circuit.
"""
import numpy as np

SL2 = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[1, 1], [0, 1]],
                [[1, 0], [1, 1]], [[0, 1], [1, 1]], [[1, 1], [1, 0]]], dtype=np.uint8)


class Tab:
    def __init__(self, Nmax):
        self.X = np.zeros((Nmax, Nmax), dtype=np.uint8)
        self.Z = np.zeros((Nmax, Nmax), dtype=np.uint8)
        self.n = 0
        self.log = []                       # [(a_ids, b_ids), ...] per sub-layer, in time order

    def born(self):
        q = self.n
        self.Z[q, q] = 1
        self.n += 1
        return q

    def _singles(self, qs, idx):
        n = self.n
        M = SL2[idx]
        x = self.X[qs, :n]; z = self.Z[qs, :n]
        self.X[qs, :n] = (M[:, 0, 0, None] & x) ^ (M[:, 0, 1, None] & z)
        self.Z[qs, :n] = (M[:, 1, 0, None] & x) ^ (M[:, 1, 1, None] & z)

    def _cnot(self, c, t):
        n = self.n
        self.X[t, :n] ^= self.X[c, :n]
        self.Z[c, :n] ^= self.Z[t, :n]

    def layer(self, a, b, rng=None, choices=None, record=True):
        """Random Clifford gates on the disjoint pairs (a[i], b[i])."""
        a = np.asarray(a, dtype=np.int64); b = np.asarray(b, dtype=np.int64)
        if len(a) == 0:
            return
        qs = np.concatenate([a, b])
        if choices is None:
            choices = rng.integers(6, size=(3, len(qs)))
        self._singles(qs, choices[0])
        self._cnot(a, b)
        self._singles(qs, choices[1])
        self._cnot(b, a)
        self._singles(qs, choices[2])
        if record:
            self.log.append((a.copy(), b.copy()))

    def entropy(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        if len(ids) == 0:
            return 0
        n = self.n
        M = np.concatenate([self.X[ids, :n], self.Z[ids, :n]], axis=0)
        return gf2_rank(M) - len(ids)


def gf2_rank(M):
    """Rank over GF(2) of a 0/1 matrix (uint8), columns packed into uint64."""
    r, c = M.shape
    if r == 0 or c == 0:
        return 0
    P = np.packbits(M.astype(np.uint8), axis=1)
    pad = (-P.shape[1]) % 8
    if pad:
        P = np.concatenate([P, np.zeros((r, pad), dtype=np.uint8)], axis=1)
    P = np.ascontiguousarray(P).view(np.uint64).copy()
    rank = 0
    W = P.shape[1]
    for w in range(W):
        colw = P[:, w]
        for bit in range(64):
            if rank == r:
                return rank
            m = np.uint64(1) << np.uint64(bit)
            nz = np.flatnonzero(P[rank:, w] & m)
            if nz.size == 0:
                continue
            p = rank + nz[0]
            if p != rank:
                P[[rank, p]] = P[[p, rank]]
            rest = rank + 1 + np.flatnonzero(P[rank + 1:, w] & m)
            if rest.size:
                P[rest] ^= P[rank]
            rank += 1
    return rank


def infected(log, A_ids, Nmax):
    """Backward sweep: set of qubits whose world line meets the (strict) causal
    past of A. Closed under predecessors."""
    inf = np.zeros(Nmax, dtype=bool)
    inf[np.asarray(A_ids, dtype=np.int64)] = True
    for a, b in reversed(log):
        m = inf[a] | inf[b]
        inf[a] |= m
        inf[b] |= m
    return inf


def causal_bound(log, A_ids, B_ids, Nmax):
    """Returns (|B ∩ Inf(A)|, |A ∩ Inf(B)|): two upper bounds on S(A)."""
    infA = infected(log, A_ids, Nmax)
    infB = infected(log, B_ids, Nmax)
    return int(infA[np.asarray(B_ids, dtype=np.int64)].sum()), int(infB[np.asarray(A_ids, dtype=np.int64)].sum())


class PTab(Tab):
    """Same tableau, columns (generators) packed by 64 into uint64 words:
    gates are bitwise operations on rows -> about 8x less memory traffic.
    Same interface as Tab."""
    def __init__(self, Nmax):
        self.Wd = (Nmax + 63) // 64
        self.X = np.zeros((Nmax, self.Wd), dtype=np.uint64)
        self.Z = np.zeros((Nmax, self.Wd), dtype=np.uint64)
        self.n = 0
        self.log = []

    def born(self):
        q = self.n
        self.Z[q, q // 64] |= np.uint64(1) << np.uint64(q % 64)
        self.n += 1
        return q

    def _w(self):
        return (self.n + 63) // 64

    def _singles(self, qs, idx):
        w = self._w()
        M = SL2[idx].astype(np.uint64) * np.uint64(0xFFFFFFFFFFFFFFFF)   # all-or-nothing masks
        x = self.X[qs, :w]; z = self.Z[qs, :w]
        self.X[qs, :w] = (M[:, 0, 0, None] & x) ^ (M[:, 0, 1, None] & z)
        self.Z[qs, :w] = (M[:, 1, 0, None] & x) ^ (M[:, 1, 1, None] & z)

    def _cnot(self, c, t):
        w = self._w()
        self.X[t, :w] ^= self.X[c, :w]
        self.Z[c, :w] ^= self.Z[t, :w]

    def entropy(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        if len(ids) == 0:
            return 0
        w = self._w()
        P = np.concatenate([self.X[ids, :w], self.Z[ids, :w]], axis=0).copy()
        return gf2_rank_words(P) - len(ids)


def gf2_rank_words(P):
    """GF(2) rank of a matrix already packed into uint64 words (modified in place)."""
    r = P.shape[0]; rank = 0
    for w in range(P.shape[1]):
        for bit in range(64):
            if rank == r:
                return rank
            m = np.uint64(1) << np.uint64(bit)
            nz = np.flatnonzero(P[rank:, w] & m)
            if nz.size == 0:
                continue
            p = rank + nz[0]
            if p != rank:
                P[[rank, p]] = P[[p, rank]]
            rest = rank + 1 + np.flatnonzero(P[rank + 1:, w] & m)
            if rest.size:
                P[rest] ^= P[rank]
            rank += 1
    return rank
