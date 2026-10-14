#!/usr/bin/env python3
"""Check of the statistical-mechanics model (annealed Renyi-2) WITH births, in the output gauge.
Model: one spin sigma_g in {e, s} per gate; exact weight of a gate
    W(sigma | inputs) = sum_tau Wg(sigma tau^{-1}, q^2) f_l(tau) f_r(tau),
    f(tau) = q^{c(tau^{-1} sigma_below)} (leg coming from a gate),  f(tau) = 1 (fresh leg: birth or t = 0);
    top boundary: q^{c(sigma_last^{-1} tau_top)}, tau_top = s on A, e on Ā; leg never touched: 1.
Prediction: E_Haar[Tr rho_A^2] = Z_A exactly (no renormalisation). Check: Z_empty = 1.
Compared with a Monte-Carlo average over Haar gates (state vector)."""
import sys, numpy as np
sys.path.insert(0, '.')
import statevector

def weights(q):
    D = q * q; Wg = {0: 1 / (D * D - 1), 1: -1 / (D * (D * D - 1))}   # 0 = same permutation, 1 = different
    ov = lambda a, b: q ** (2 if a == b else 1)
    def f(tau, below):                       # below: 0 = free, 1 = e, 2 = s
        return 1.0 if below == 0 else ov(tau, below)
    def W(sig, bl, br):
        return sum(Wg[0 if sig == tau else 1] * f(tau, bl) * f(tau, br) for tau in (1, 2))
    return W, ov

def schedule(N0, n_final, k):
    ev, n, L = [], N0, 0
    while True:
        if L % k == 0:
            if n == n_final: break
            ev.append(('birth',)); n += 1
        ev.append(('layer', [(a, a + 1) for a in range(L % 2, n - 1, 2)])); L += 1
    return ev

def Z_statmech(q, N0, n_final, k, A):
    W, ov = weights(q); states = {tuple([0] * N0): 1.0}
    for e in schedule(N0, n_final, k):
        if e[0] == 'birth':
            states = {s + (0,): w for s, w in states.items()}; continue
        for (a, b) in e[1]:
            new = {}
            for s, w in states.items():
                for sig in (1, 2):
                    ww = W(sig, s[a], s[b])
                    if ww == 0: continue
                    t = list(s); t[a] = t[b] = sig; t = tuple(t); new[t] = new.get(t, 0) + w * ww
            states = new
    tot = 0.0
    for s, w in states.items():
        f = 1.0
        for x, sp in enumerate(s):
            if sp != 0: f *= ov(sp, 2 if x in A else 1)
        tot += w * f
    return tot

def purity_mc(N0, n_final, k, m, samples, rng):
    vals = []
    for _ in range(samples):
        psi = np.zeros([2] * N0, complex); psi[(0,) * N0] = 1; n = N0
        for e in schedule(N0, n_final, k):
            if e[0] == 'birth': psi = statevector.insert_fresh(psi, n); n += 1; continue
            for (a, b) in e[1]: psi = statevector.apply2(psi, statevector.haar2(rng), a, b)
        sv = np.linalg.svd(psi.reshape(2 ** m, -1), compute_uv=False); vals.append(np.sum(sv ** 4))
    return np.mean(vals), np.std(vals) / np.sqrt(samples)

if __name__ == '__main__':
    rng = np.random.default_rng(7)
    for (N0, nf, k) in ((2, 6, 1), (2, 8, 1), (2, 8, 2), (4, 10, 1)):
        m = nf // 2
        Z0 = Z_statmech(2, N0, nf, k, set())
        ZA = Z_statmech(2, N0, nf, k, set(range(m)))
        mc, err = purity_mc(N0, nf, k, m, 3000, rng)
        print(f"N0={N0} n={nf} k={k}: Z_empty = {Z0:.6f} | Z_A (spin model) = {ZA:.5f} | Haar Monte Carlo = {mc:.5f} ± {err:.5f} | deviation = {(ZA - mc) / err:+.2f} sigma", flush=True)
