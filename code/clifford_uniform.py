#!/usr/bin/env python3
"""Two-qubit Clifford gates drawn UNIFORMLY (modulo Paulis): Sp(4, F2), 720 elements.
Coordinate order: (x_a, z_a, x_b, z_b). Signs (Paulis) do not affect any entropy, so drawing
the symplectic part uniformly reproduces the law of entropies under the uniform Clifford group,
which is a unitary 2-design (and even a 3-design on qubits)."""
import itertools, numpy as np
from tableau import PTab, SL2

OMEGA = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.uint8)


def _enumerate_sp4():
    out = []
    for bits in itertools.product((0, 1), repeat=16):
        M = np.array(bits, dtype=np.uint8).reshape(4, 4)
        if np.array_equal((M.T.astype(int) @ OMEGA @ M) % 2, OMEGA):
            out.append(M)
    return np.array(out, dtype=np.uint8)


SP4 = _enumerate_sp4()
assert len(SP4) == 720, len(SP4)


def _blk(i, j):
    M = np.zeros((4, 4), dtype=np.uint8); M[:2, :2] = SL2[i]; M[2:, 2:] = SL2[j]; return M


CNOT_AB = np.array([[1, 0, 0, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 0, 1]], dtype=np.uint8)  # x_b^=x_a, z_a^=z_b
CNOT_BA = np.array([[1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 1, 0, 1]], dtype=np.uint8)  # x_a^=x_b, z_b^=z_a


def rand2_matrix(c):
    """Symplectic matrix of the rand2 gate of tableau.py; c = (a0,b0,a1,b1,a2,b2) SL2 indices."""
    M = _blk(c[0], c[1])
    M = (CNOT_AB.astype(int) @ M) % 2
    M = (_blk(c[2], c[3]).astype(int) @ M) % 2
    M = (CNOT_BA.astype(int) @ M) % 2
    M = (_blk(c[4], c[5]).astype(int) @ M) % 2
    return M.astype(np.uint8)


class CTab(PTab):
    """PTab with arbitrary symplectic gates (uniform draw in Sp(4,F2) by default)."""
    def layer_sym(self, a, b, Ms, record=True):
        a = np.asarray(a, dtype=np.int64); b = np.asarray(b, dtype=np.int64)
        if len(a) == 0:
            return
        w = self._w()
        V = np.stack([self.X[a, :w], self.Z[a, :w], self.X[b, :w], self.Z[b, :w]])     # (4, G, w)
        Mk = Ms.astype(np.uint64) * np.uint64(0xFFFFFFFFFFFFFFFF)                       # (G, 4, 4)
        new = []
        for r in range(4):
            acc = Mk[:, r, 0, None] & V[0]
            for j in range(1, 4):
                acc = acc ^ (Mk[:, r, j, None] & V[j])
            new.append(acc)
        self.X[a, :w], self.Z[a, :w], self.X[b, :w], self.Z[b, :w] = new
        if record:
            self.log.append((a.copy(), b.copy()))

    def layer_unif(self, a, b, rng, record=True):
        self.layer_sym(a, b, SP4[rng.integers(720, size=len(a))], record)


# ---------- batch simulator (n <= 64), for small-size Monte-Carlo averages ----------
def batch_entropy(N0, n_final, k, m, S, rng, gate='unif'):
    """Returns S(A) (bits) for S independent circuits, A = [0, m). Schedule of full_spin_model."""
    from domain_wall import events
    n_gen = n_final
    X = np.zeros((S, n_final, n_gen), dtype=np.uint8); Z = np.zeros_like(X)
    for q in range(N0):
        Z[:, q, q] = 1
    n = N0
    for e in events(N0, n_final, k):
        if e < 0:
            Z[:, n, n] = 1; n += 1; continue
        a = np.arange(e, n - 1, 2); b = a + 1
        G = len(a)
        if G == 0:
            continue
        if gate == 'unif':
            M = SP4[rng.integers(720, size=(S, G))]
        else:
            c = rng.integers(6, size=(S, G, 6))
            M = RAND2_TABLE[np.ravel_multi_index(np.moveaxis(c, -1, 0), (6,) * 6)]
        V = np.stack([X[:, a], Z[:, a], X[:, b], Z[:, b]], axis=2)                        # (S, G, 4, n_gen)
        Wn = np.einsum('sgij,sgjk->sgik', M.astype(np.int32), V.astype(np.int32)) & 1
        X[:, a], Z[:, a], X[:, b], Z[:, b] = (Wn[:, :, i].astype(np.uint8) for i in range(4))
    rows = np.concatenate([X[:, :m], Z[:, :m]], axis=1)                                     # (S, 2m, n_gen)
    packed = (rows.astype(np.uint64) << np.arange(n_gen, dtype=np.uint64)).sum(axis=2)       # (S, 2m)
    return batch_rank(packed, n_gen) - m


def batch_rank(P, nbits):
    S, R = P.shape
    basis = np.zeros((S, nbits), dtype=np.uint64)
    rank = np.zeros(S, dtype=np.int64)
    idx = np.arange(S)
    for r in range(R):
        v = P[:, r].copy()
        for bit in range(nbits - 1, -1, -1):
            m = np.uint64(1) << np.uint64(bit)
            has = (v & m) != 0
            occ = basis[:, bit] != 0
            red = has & occ
            v[red] ^= basis[red, bit]
            ins = has & ~occ
            basis[ins, bit] = v[ins]; rank += ins; v[ins] = 0
    return rank


RAND2_TABLE = np.array([rand2_matrix(c) for c in itertools.product(range(6), repeat=6)], dtype=np.uint8)
