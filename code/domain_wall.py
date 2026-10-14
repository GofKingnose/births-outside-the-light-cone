#!/usr/bin/env python3
"""Exact single-domain-wall representation of the annealed Renyi-2 model (input gauge tau).

For each gate we keep the INPUT spin tau in {e, s}; the output spin sigma is summed over:
    w(tau; a_l, a_r) = sum_sigma Wg(sigma^-1 tau, q^2) q^{c(sigma^-1 a_l)} q^{c(sigma^-1 a_r)}  in {1, K, 0},
a_l, a_r = spins of the gates located above each of the two output legs (or of the boundary).
w = 1 if a_l = a_r = tau; K = q/(q^2+1) if a_l != a_r; 0 if a_l = a_r != tau.
Fresh legs (births, t = 0): weight 1. Hence no wall is ever created going down.
A = [0, m): a single wall, at position p (legs [0,p) in e, [p,n) in s, after the global symmetry e<->s).
"""
import numpy as np
from full_spin_model import schedule


def events(N0, n_final, k):
    """Same schedule as full_spin_model.schedule, without storing gates: -1 (birth) or layer parity."""
    ev, n, L = [], N0, 0
    while True:
        if L % k == 0:
            if n == n_final: break
            ev.append(-1); n += 1
        ev.append(L % 2); L += 1
    return ev


def logZ_wall(q, N0, n_final, k, m):
    """log Z_A (natural), Z_A = E[Tr rho_A^2], A = [0, m). Computed in the log domain
    (dominant paths are large deviations of the free walk: no underflow)."""
    lK = np.log(q / (q * q + 1.0))
    ev = events(N0, n_final, k)
    n = n_final
    lw = np.full(n_final + 2, -np.inf); lw[m] = 0.0
    for e in reversed(ev):
        if e < 0:                       # removal of leg n-1: p -> min(p, n-1)
            lw[n - 1] = np.logaddexp(lw[n - 1], lw[n]); lw[n] = -np.inf; n -= 1
            continue
        ps = np.arange(e + 1, n, 2)     # wall at p under the gate (p-1, p)
        if len(ps):
            src = lw[ps] + lK; lw[ps] = -np.inf
            lw[ps - 1] = np.logaddexp(lw[ps - 1], src)
            lw[ps + 1] = np.logaddexp(lw[ps + 1], src)
    assert n == N0
    fin = lw[np.isfinite(lw)]
    return float(np.logaddexp.reduce(fin))


if __name__ == '__main__':
    from full_spin_model import Z_statmech
    print("Single-wall recursion vs full spin model (sigma gauge), q = 2 and q = 3")
    worst = 0.0
    for q in (2, 3):
        for (N0, nf, k) in ((2, 6, 1), (2, 8, 1), (2, 8, 2), (4, 10, 1), (2, 10, 3), (3, 9, 2), (4, 12, 1), (2, 12, 4), (5, 11, 1)):
            for m in range(1, nf):
                ZA = Z_statmech(q, N0, nf, k, set(range(m)))
                Zp = np.exp(logZ_wall(q, N0, nf, k, m))
                rel = abs(ZA - Zp) / ZA; worst = max(worst, rel)
            Z0 = Z_statmech(q, N0, nf, k, set())
            print(f"  q={q} N0={N0} n={nf} k={k}: Z_empty = {Z0:.12f}; m = {nf//2}: Z_A = {Z_statmech(q,N0,nf,k,set(range(nf//2))):.10f} | wall = {np.exp(logZ_wall(q,N0,nf,k,nf//2)):.10f}", flush=True)
    print(f"  maximal relative deviation (all cuts m = 1..n-1): {worst:.2e}")


# ----------------------------------------------------------------------------------------------
# General schedule: births at the right edge at layers L_1 <= L_2 <= ...; end at layer T.
def events_general(N0, birth_layers, T):
    """List of events (-1 = birth at the right edge, otherwise the layer parity);
    a birth of date L happens just before layer L."""
    ev, bi = [], 0
    for L in range(T):
        while bi < len(birth_layers) and birth_layers[bi] <= L:
            ev.append(-1); bi += 1
        ev.append(L % 2)
    while bi < len(birth_layers):
        ev.append(-1); bi += 1
    return ev


def logZ_events(q, N0, ev, m):
    """log Z_A for A = [0, m) at the final time, arbitrary schedule of births at the right edge."""
    lK = np.log(q / (q * q + 1.0))
    n = N0 + sum(1 for e in ev if e < 0)
    lw = np.full(n + 2, -np.inf); lw[m] = 0.0
    for e in reversed(ev):
        if e < 0:
            lw[n - 1] = np.logaddexp(lw[n - 1], lw[n]); lw[n] = -np.inf; n -= 1
            continue
        ps = np.arange(e + 1, n, 2)
        if len(ps):
            src = lw[ps] + lK; lw[ps] = -np.inf
            lw[ps - 1] = np.logaddexp(lw[ps - 1], src)
            lw[ps + 1] = np.logaddexp(lw[ps + 1], src)
    fin = lw[np.isfinite(lw)]
    return float(np.logaddexp.reduce(fin))


def cone_lower(N0, ev, m):
    """Theorem-A bound for A = [0, m): |Ā \\ Inf(A)| = m_Ā - |Ā ∩ Inf(A)| (in qudits).
    Backward sweep: Inf(A) ∩ time slice = [0, r]; a gate (r, r+1) extends it."""
    n = N0 + sum(1 for e in ev if e < 0)
    nA_bar = n - m
    r = m - 1; inside = 0
    for e in reversed(ev):
        if e < 0:
            x = n - 1                                   # removed leg (birth read backwards)
            if x >= m and x <= r: inside += 1
            n -= 1; r = min(r, n - 1); continue
        if r + 1 < n and (r - e) % 2 == 0 and r >= e:   # gate (r, r+1) in this layer
            r += 1
    inside += sum(1 for x in range(m, n) if x <= r)     # initial qudits of Ā in the cone
    return nA_bar - inside
