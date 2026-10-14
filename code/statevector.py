#!/usr/bin/env python3
"""
State-vector tools for a growing register (q = 2).

Convention: a pure state |psi> is stored as a numpy tensor of shape (2,)*n,
axis i = qubit i in the current spatial order of the chain.
A birth inserts a fresh qubit in |0> (isometry H_n -> H_n ⊗ C^2): the global state
stays pure and the total entropy stays zero.
"""
import numpy as np

LN2 = np.log(2.0)

def haar2(rng):
    """Haar-random 4x4 unitary."""
    z = (rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    return q * (d / np.abs(d))

def apply2(psi, U, a, b):
    """Applies U (4x4) to qubits a, b (a != b)."""
    n = psi.ndim
    T = U.reshape(2, 2, 2, 2)
    out = np.tensordot(T, psi, axes=([2, 3], [a, b]))
    # out has axes (a', b', remaining axes in order) -> put them back in place
    rest = [i for i in range(n) if i not in (a, b)]
    order = [None] * n
    order[a] = 0
    order[b] = 1
    for k, i in enumerate(rest):
        order[i] = k + 2
    return np.transpose(out, order)

def insert_fresh(psi, pos):
    """Inserts a qubit |0> at position pos (0..n)."""
    e = np.expand_dims(psi, pos)
    return np.concatenate([e, np.zeros_like(e)], axis=pos)

def brickwork_layer(psi, parity, rng):
    n = psi.ndim
    for a in range(parity, n - 1, 2):
        psi = apply2(psi, haar2(rng), a, a + 1)
    return psi

def entropy_cut(psi, m):
    """Entropy (bits) of the first m qubits."""
    n = psi.ndim
    M = psi.reshape(2 ** m, 2 ** (n - m))
    s = np.linalg.svd(M, compute_uv=False)
    p = s ** 2
    p = p[p > 1e-15]
    return float(-(p * np.log2(p)).sum())

def page_bits(m, n):
    """Page average entropy (bits) of a subsystem of m qubits out of n."""
    a, b = m, n - m
    if a > b:
        a, b = b, a
    return a - 2.0 ** (a - b) / (2 * LN2)

def rdm(psi, keep):
    """Reduced density matrix of the qubits 'keep' (sorted list)."""
    n = psi.ndim
    other = [i for i in range(n) if i not in keep]
    A = np.transpose(psi, list(keep) + other).reshape(2 ** len(keep), -1)
    return A @ A.conj().T

def vn_bits(rho):
    w = np.linalg.eigvalsh(rho)
    w = w[w > 1e-14]
    return float(-(w * np.log2(w)).sum())

def mutual_info_matrix(psi):
    n = psi.ndim
    S1 = np.array([vn_bits(rdm(psi, [i])) for i in range(n)])
    I = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            Sij = vn_bits(rdm(psi, [i, j]))
            I[i, j] = I[j, i] = S1[i] + S1[j] - Sij
    return S1, I
