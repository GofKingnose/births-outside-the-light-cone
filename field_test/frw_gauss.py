#!/usr/bin/env python3
"""
Free scalar field on a 1+1D lattice growing according to an FRW law: exact Gaussian computation.

Model (fixed in ../preregistration/field_test_predictions.txt before any computation)
-------------------------------------------------------------------------------------
Periodic lattice of N comoving cells, physical spacing ell(t) = a(t)/a(t_ref) in [1, 2] (cutoff Lambda = 1).
Discretised 1+1 FRW Lagrangian: L = sum ell [phidot^2/2 - (Dphi/ell)^2/2 - m^2 phi^2/2], hence
    H(t) = sum_i [ P_i^2/(2 ell) + (phi_{i+1}-phi_i)^2/(2 ell) + ell m^2 phi_i^2/2 ],   [phi_i, P_j] = i delta_ij.
Mode k: mass M = ell, frequency Omega_k = sqrt(4 sin^2(k/2)/ell^2 + m^2).
When ell reaches 2: CANONICAL Haar refinement N -> 2N (ell: 2 -> 1)
    phi_{2i} = phi_i + w_i/2, phi_{2i+1} = phi_i - w_i/2,  P_{2i} = P_i/2 + p_i, P_{2i+1} = P_i/2 - p_i,
where (w_i, p_i) are the newborn degrees of freedom. Covariance Gamma = 1/2 <{R,R}>, R = (phi, P).
Birth rules: 'prod' (local ground state, product), 'vacuum' (Gaussian isometry that preserves the
vacuum exactly), 'mera' (product + D layers of optimised Gaussian gates).
"""
import numpy as np
from scipy import sparse
from scipy.linalg import circulant, eigh, expm
from scipy.optimize import minimize, brentq

# ------------------------------------------------------------------ Gaussian tools
def omegas(N, ell, m):
    k = 2 * np.pi * np.arange(N) / N
    return np.sqrt(4 * np.sin(k / 2) ** 2 / ell ** 2 + m ** 2)

def circ_apply(eig, X):
    """Product of a circulant matrix (eigenvalues eig, symmetric in k) with X (along axis 0)."""
    return np.fft.ifft(eig[:, None] * np.fft.fft(X, axis=0), axis=0).real

def circ_matrix(eig):
    return circulant(np.fft.ifft(eig).real)

def vacuum(N, ell, m):
    Om = omegas(N, ell, m)
    G = np.zeros((2 * N, 2 * N))
    G[:N, :N] = circ_matrix(1 / (2 * ell * Om))
    G[N:, N:] = circ_matrix(ell * Om / 2)
    return G

def thermal(N, ell, m, beta):
    Om = omegas(N, ell, m)
    nb = 0.0 if beta == np.inf else 1 / np.expm1(beta * Om)
    G = np.zeros((2 * N, 2 * N))
    G[:N, :N] = circ_matrix((nb + 0.5) / (ell * Om))
    G[N:, N:] = circ_matrix(ell * Om * (nb + 0.5))
    return G

def energy(G, ell, m):
    N = G.shape[0] // 2
    Gf, Gp = G[:N, :N], G[N:, N:]
    d = np.diag(Gf); off = Gf[np.arange(N), (np.arange(N) + 1) % N]
    return 0.5 * (np.trace(Gp) / ell + (2 * d.sum() - 2 * off.sum()) / ell + ell * m ** 2 * d.sum())

def e_vac(N, ell, m):
    return 0.5 * omegas(N, ell, m).sum()

def symp_eigs(G):
    """Symplectic eigenvalues (>= 1/2) of a 2n x 2n covariance matrix."""
    n = G.shape[0] // 2
    J = np.block([[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]])
    w, V = eigh(G)
    w = np.clip(w, 1e-300, None)
    Gh = (V * np.sqrt(w)) @ V.T
    ev = eigh(1j * (Gh @ J @ Gh), eigvals_only=True)     # values +-nu
    nu = np.sort(ev)[n:]                                   # the n positive values
    return np.maximum(nu, 0.5)

def entropy_from_nu(nu):
    a = nu + 0.5; b = nu - 0.5
    s = a * np.log(a)
    mask = b > 1e-14
    s[mask] -= b[mask] * np.log(b[mask])
    return float(s.sum())

def entropy(G, idx):
    """Entropy (nats) of the subset of modes idx (mode indices 0..n-1) of a 2n x 2n covariance matrix."""
    n = G.shape[0] // 2
    idx = np.asarray(idx)
    sel = np.concatenate([idx, idx + n])
    return entropy_from_nu(symp_eigs(G[np.ix_(sel, sel)]))

def single_mode_S(G, i):
    n = G.shape[0] // 2
    det = G[i, i] * G[i + n, i + n] - G[i, i + n] ** 2
    nu = np.sqrt(max(det, 0.25))
    return entropy_from_nu(np.array([nu]))

def two_mode_MI(G, i, js):
    """MI(i : j) for all j in js (single modes)."""
    n = G.shape[0] // 2
    out = []
    Si = single_mode_S(G, i)
    for j in js:
        sel = np.array([i, j, i + n, j + n])
        nu = symp_eigs(G[np.ix_(sel, sel)])
        out.append(Si + single_mode_S(G, j) - entropy_from_nu(nu))
    return np.array(out)

# ------------------------------------------------------------------ exact propagation per mode
def step_propagator(N, ell, m, dt):
    Om = omegas(N, ell, m); M = ell
    c, s = np.cos(Om * dt), np.sin(Om * dt)
    return np.stack([c, s / (M * Om), -M * Om * s, c])     # u11, u12, u21, u22 per mode

def compose(U2, U1):
    """U2 . U1 (2x2 matrices per mode)."""
    a1, b1, c1, d1 = U1; a2, b2, c2, d2 = U2
    return np.stack([a2 * a1 + b2 * c1, a2 * b1 + b2 * d1, c2 * a1 + d2 * c1, c2 * b1 + d2 * d1])

def apply_U(G, U):
    N = G.shape[0] // 2
    def left(X):
        top, bot = X[:N], X[N:]
        return np.vstack([circ_apply(U[0], top) + circ_apply(U[1], bot), circ_apply(U[2], top) + circ_apply(U[3], bot)])
    Y = left(G)
    return left(Y.T)

# ------------------------------------------------------------------ refinement
def T_matrix(Nc):
    """R_fine = T R_cw, R_cw = (phi_c, w, P_c, p); R_fine = (phi_f, P_f), N_f = 2 Nc."""
    Nf = 2 * Nc; rows, cols, vals = [], [], []
    for i in range(Nc):
        for (r, s) in ((2 * i, 1.0), (2 * i + 1, -1.0)):
            rows += [r, r]; cols += [i, Nc + i]; vals += [1.0, 0.5 * s]                      # phi
            rows += [Nf + r, Nf + r]; cols += [Nf + i, Nf + Nc + i]; vals += [0.5, 1.0 * s]    # P
    return sparse.csr_matrix((vals, (rows, cols)), shape=(2 * Nf, 2 * Nf))

def Tinv_matrix(Nc):
    Nf = 2 * Nc; rows, cols, vals = [], [], []
    for i in range(Nc):
        rows += [i, i]; cols += [2 * i, 2 * i + 1]; vals += [0.5, 0.5]                          # phi_c
        rows += [Nc + i, Nc + i]; cols += [2 * i, 2 * i + 1]; vals += [1.0, -1.0]                # w
        rows += [Nf + i, Nf + i]; cols += [Nf + 2 * i, Nf + 2 * i + 1]; vals += [1.0, 1.0]       # P_c
        rows += [Nf + Nc + i, Nf + Nc + i]; cols += [Nf + 2 * i, Nf + 2 * i + 1]; vals += [0.5, -0.5]  # p
    return sparse.csr_matrix((vals, (rows, cols)), shape=(2 * Nf, 2 * Nf))

def sandwich(Ts, G):
    """T G T^T for sparse T and dense symmetric G."""
    Y = np.asarray(Ts @ G)
    return np.asarray(Ts @ Y.T).T

def hamiltonian_matrix(N, ell, m):
    Lap = sparse.diags([2 * np.ones(N), -np.ones(N - 1), -np.ones(N - 1)], [0, 1, -1], format='lil')
    Lap[0, N - 1] = -1; Lap[N - 1, 0] = -1
    K = Lap.tocsr() / ell + ell * m ** 2 * sparse.identity(N)
    return sparse.block_diag([K, sparse.identity(N) / ell], format='csr')

def detail_local_state(Nc, m, ell_f=1.0):
    """Local coefficients (a_w, b_p) of the newborn modes in H_fine, and their local ground state."""
    T = T_matrix(Nc); Hf = hamiltonian_matrix(2 * Nc, ell_f, m)
    A = (T.T @ Hf @ T).tocsr(); Nf = 2 * Nc
    aw = A.diagonal()[Nc:Nf]; bp = A.diagonal()[Nf + Nc:]
    return aw, bp, 0.5 * np.sqrt(bp / aw), 0.5 * np.sqrt(aw / bp)

def assemble_cw(Gc, ww, pp):
    """(Gamma_c + Gamma_w) in the order (phi_c, w, P_c, p)."""
    Nc = Gc.shape[0] // 2; Nf = 2 * Nc
    G = np.zeros((2 * Nf, 2 * Nf))
    ic = np.r_[0:Nc, Nf:Nf + Nc]
    G[np.ix_(ic, ic)] = Gc
    G[np.arange(Nc, Nf), np.arange(Nc, Nf)] = ww
    G[np.arange(Nf + Nc, 2 * Nf), np.arange(Nf + Nc, 2 * Nf)] = pp
    return G

_VP_CACHE = {}
def vacuum_preserving_map(Nc, m):
    """M = X1^{1/2} X0^{-1/2} (positions, coordinates (c,w)); S = diag(M, M^{-T})."""
    key = (Nc, m)
    if key in _VP_CACHE: return _VP_CACHE[key]
    Nf = 2 * Nc
    aw, bp, ww, pp = detail_local_state(Nc, m)
    G0 = assemble_cw(vacuum(Nc, 2.0, m), ww, pp)
    Ti = Tinv_matrix(Nc)
    G1 = sandwich(Ti, vacuum(Nf, 1.0, m))
    X0 = 2 * G0[:Nf, :Nf]; X1 = 2 * G1[:Nf, :Nf]
    w0, V0 = eigh(X0); w1, V1 = eigh(X1)
    X1h = (V1 * np.sqrt(w1)) @ V1.T; X1mh = (V1 / np.sqrt(w1)) @ V1.T
    X0h = (V0 * np.sqrt(w0)) @ V0.T; X0mh = (V0 / np.sqrt(w0)) @ V0.T
    M = X1h @ X0mh; Q = X1mh @ X0h                          # Q = M^{-T}
    _VP_CACHE[key] = (M, Q, ww, pp)
    return _VP_CACHE[key]

# ------------------------------------------------------------------ MERA-like (Gaussian) layer
J4 = np.block([[np.zeros((2, 2)), np.eye(2)], [-np.eye(2), np.zeros((2, 2))]])
def gate_from_params(x):
    A = np.zeros((4, 4)); iu = np.triu_indices(4)
    A[iu] = x; A = A + A.T - np.diag(np.diag(A))
    return expm(J4 @ A)

def apply_gate_layer(G, g, parity):
    N = G.shape[0] // 2
    a = np.arange(parity, N, 2); b = (a + 1) % N
    idx = np.stack([a, b, a + N, b + N])                    # (4, npairs)
    def left(X):
        R = X[idx]                                          # (4, np, 2N)
        X = X.copy(); X[idx] = np.einsum('ij,jpk->ipk', g, R); return X
    Y = left(G)
    return left(Y.T)

def mera_birth(Gc_cw_fine, gates):
    G = Gc_cw_fine
    for d, g in enumerate(gates):
        G = apply_gate_layer(G, g, d % 2)
    return G

def optimise_mera(D, m, Nc=64, seed=0, restarts=3):
    aw, bp, ww, pp = detail_local_state(Nc, m)
    T = T_matrix(Nc); Nf = 2 * Nc
    G0 = sandwich(T, assemble_cw(vacuum(Nc, 2.0, m), ww, pp))
    Ev = e_vac(Nf, 1.0, m)
    def f(x):
        gates = [gate_from_params(x[10 * d:10 * (d + 1)]) for d in range(D)]
        return (energy(mera_birth(G0, gates), 1.0, m) - Ev) / Nc
    rng = np.random.default_rng(seed); best = None
    for r in range(restarts):
        x0 = np.zeros(10 * D) if r == 0 else 0.1 * rng.standard_normal(10 * D)
        res = minimize(f, x0, method='L-BFGS-B', options=dict(maxiter=2000))
        if best is None or res.fun < best.fun: best = res
    return [gate_from_params(best.x[10 * d:10 * (d + 1)]) for d in range(D)], best.fun, f(np.zeros(10 * D))

# ------------------------------------------------------------------ birth
def birth(Gc, rule, m, gates=None, return_cw=False):
    Nc = Gc.shape[0] // 2; Nf = 2 * Nc
    T = T_matrix(Nc)
    if rule == 'vacuum':
        M, Q, ww, pp = vacuum_preserving_map(Nc, m)
        G = assemble_cw(Gc, ww, pp)
        S11, S12, S21, S22 = G[:Nf, :Nf], G[:Nf, Nf:], G[Nf:, :Nf], G[Nf:, Nf:]
        Gcw = np.block([[M @ S11 @ M.T, M @ S12 @ Q.T], [Q @ S21 @ M.T, Q @ S22 @ Q.T]])
        Gf = sandwich(T, Gcw)
    else:
        aw, bp, ww, pp = detail_local_state(Nc, m)
        Gf = sandwich(T, assemble_cw(Gc, ww, pp))
        if rule == 'mera':
            Gf = mera_birth(Gf, gates)
        Gcw = sandwich(Tinv_matrix(Nc), Gf) if return_cw else None
    return (Gf, Gcw) if return_cw else Gf

# ------------------------------------------------------------------ expansion laws
def law(kind, H0, p=None):
    """Returns (t0, a(t), H(t), t_doubling(t_ref))."""
    if kind == 'dS':
        return 0.0, (lambda t: np.exp(H0 * t)), (lambda t: H0), (lambda tr: tr + np.log(2) / H0)
    t0 = p / H0
    return t0, (lambda t: (t / t0) ** p), (lambda t: p / t), (lambda tr: tr * 2 ** (1 / p))

def evolve_interval(G, N, m, a, Hf, t1, t2, eps=0.005):
    """Propagates from t1 to t2 (ell = a(t)/a(t1) goes from 1 to 2); returns Gamma and the number of steps."""
    U = np.stack([np.ones(N), np.zeros(N), np.zeros(N), np.ones(N)])
    t = t1; nsteps = 0; aref = a(t1)
    while t < t2 - 1e-12:
        dt = min(eps / Hf(t), t2 - t)
        ell = a(t + dt / 2) / aref
        U = compose(step_propagator(N, ell, m, dt), U)
        t += dt; nsteps += 1
    return apply_U(G, U), nsteps

def quanta(G, ell, m):
    """Total number of instantaneous quanta sum_k n_k, n_k = E_k/Omega_k - 1/2 (invariant under adiabatic redshift)."""
    N = G.shape[0] // 2
    i = np.arange(N); Om = omegas(N, ell, m)
    def fdiag(B):
        c = B[i[:, None], (i[:, None] + i[None, :]) % N].sum(axis=0)       # c(d) = sum_i B[i, i+d]
        return np.fft.fft(c).real / N                                      # symmetrised <|x_k|^2>
    gf, gp = fdiag(G[:N, :N]), fdiag(G[N:, N:])
    Ek = 0.5 * (gp / ell + ell * Om ** 2 * gf)
    return float((Ek / Om - 0.5).sum())
