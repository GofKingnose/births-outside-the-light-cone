#!/usr/bin/env python3
"""
Diagnostic of the parity locking of the strict cone under insertion (1D, brickwork pattern).

Definition: going backwards in time from A = final left half, the reached set is a prefix [0, b]
of the chain. A layer is "unsaturated" if b < n-2 before the layer.
v_eff = average, over unsaturated layers (except the first), of the advance b_after - b_before in {0, 1}.

Mechanism tested: after any layer, the edge is "aligned" for the next layer (going backwards);
each removal (= insertion forward in time) of a site at position <= b shifts b by one unit and
flips the alignment. Prediction: the edge advances IFF the number m of such removals since the
previous layer is even. We measure the layer-by-layer agreement, P(m even) and v_eff.
Controls: edge growth (m = 0 always, v_eff = 1) and insertion by pairs (m always even).
"""
import json
import numpy as np
from growth_1d import birth_times
from continuum_geometry import f_insert

def run_struct(mode, seed, Nmax, N0=16):
    """Same random draws as growth_1d.run (identical seeds) but without the quantum simulation."""
    rng = np.random.default_rng(seed); tb = birth_times(mode, Nmax)
    pos = list(range(N0)); nid = N0; nb, L = 0, 0; events = []
    while nb < len(tb):
        while nb < len(tb) and tb[nb] <= L:
            p = len(pos) if mode.startswith('edge') else int(rng.integers(0, len(pos) + 1))
            pos.insert(p, nid); events.append(('ins', nid)); nid += 1; nb += 1
            if mode.endswith('pair') and nb < len(tb):
                pos.insert(p + 1, nid); events.append(('ins', nid)); nid += 1; nb += 1
        if nb == len(tb): break
        par = L % 2; n = len(pos)
        events.append(('layer', np.array([pos[x] for x in range(par, n - 1, 2)], dtype=np.int64),
                       np.array([pos[x + 1] for x in range(par, n - 1, 2)], dtype=np.int64)))
        L += 1
    return pos, events, nid

def diagnose(pos, events, nid):
    pos = list(pos); n = len(pos)
    inf = np.zeros(nid, bool); inf[pos[:n // 2]] = True
    adv, meven, first, m = [], [], True, 0
    for e in reversed(events):
        if e[0] == 'layer':
            P = np.array(pos); b0 = np.flatnonzero(inf[P]).max()
            a, b = e[1], e[2]; mm = inf[a] | inf[b]; inf[a] |= mm; inf[b] |= mm
            b1 = np.flatnonzero(inf[P]).max()
            if b0 < len(pos) - 2 and not first:
                adv.append(int(b1 - b0)); meven.append(m % 2 == 0)
            first = False; m = 0
        else:
            P = np.array(pos); b = np.flatnonzero(inf[P]).max()
            if pos.index(e[1]) <= b: m += 1
            pos.remove(e[1])
    adv = np.array(adv); meven = np.array(meven)
    return adv, meven

if __name__ == '__main__':
    out = {}
    for mode in ('edge-1', 'p1', 'p2', 'exp', 'p0.5', 'p1pair'):
        A, M = [], []
        for seed in range(4):
            pos, ev, nid = run_struct(mode, seed, 1600 if mode != 'p0.5' else 400)
            a, m = diagnose(pos, ev, nid); A.append(a); M.append(m)
        a = np.concatenate(A); m = np.concatenate(M)
        v = float(a.mean()); agree = float(np.mean((a == 1) == m)); pev = float(m.mean())
        out[mode] = dict(v_eff=v, P_m_even=pev, agreement=agree, layers=int(len(a)))
        line = f"{mode:7s} v_eff = {v:.3f}  P(m even) = {pev:.3f}  agreement advance<->m even = {agree:.4f}  ({len(a)} layers)"
        if mode in ('p1', 'p2', 'exp', 'p1pair'):
            law = mode.replace('pair', '')
            line += f"  | continuum G_LB/N: v=1 -> {f_insert(law, 0.5, 16, 1600)/2:.4f}; v_eff -> {f_insert(law, 0.5 * v, 16, 1600)/2:.4f}"
        print(line, flush=True)
    json.dump(out, open('../data/parity_locking.json', 'w'), indent=1)
