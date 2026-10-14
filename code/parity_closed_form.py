#!/usr/bin/env python3
"""Closed form for the effective speed of the strict cone under 1D insertion (brickwork pattern).
Same random draws as parity_locking.py. For each unsaturated layer, the realised advance (0 or 1) is compared
with the prediction P(even number of insertions to the left of the edge) = 1/2 [1 + (1 - 2 pi)^r], pi = (b+1)/n,
r = number of insertions between the two layers (read backwards). The strict cone depends only on the circuit
structure: no gate is drawn here."""
import sys, json
import numpy as np
sys.path.insert(0, '.')
from parity_locking import run_struct


def diagnose_pred(pos, events, nid):
    pos = list(pos); n = len(pos)
    inf = np.zeros(nid, bool); inf[pos[:n // 2]] = True
    adv, pred, first, m, r, ctx = [], [], True, 0, 0, None
    for e in reversed(events):
        if e[0] == 'layer':
            P = np.array(pos); b0 = np.flatnonzero(inf[P]).max()
            a, b = e[1], e[2]; mm = inf[a] | inf[b]; inf[a] |= mm; inf[b] |= mm
            b1 = np.flatnonzero(inf[P]).max()
            if b0 < len(pos) - 2 and not first and ctx is not None:
                bb, nn = ctx
                pi = (bb + 1) / nn
                adv.append(int(b1 - b0)); pred.append(0.5 * (1 + (1 - 2 * pi) ** r))
            first = False; m = 0; r = 0
            P = np.array(pos); ctx = (int(np.flatnonzero(inf[P]).max()), len(pos))
        else:
            r += 1
            pos.remove(e[1])
    return np.array(adv), np.array(pred)


if __name__ == '__main__':
    out = {}
    for mode in ('p1', 'p2', 'exp', 'p0.5', 'p1pair', 'edge-1'):
        A, Pp = [], []
        for seed in range(4):
            pos, ev, nid = run_struct(mode, seed, 1600 if mode != 'p0.5' else 400)
            a, p = diagnose_pred(pos, ev, nid); A.append(a); Pp.append(p)
        a = np.concatenate(A); p = np.concatenate(Pp)
        out[mode] = dict(v_eff=float(a.mean()), prediction=float(p.mean()), layers=int(len(a)),
                         se=float(a.std(ddof=1) / np.sqrt(len(a))))
        print(f"{mode:7s} measured v_eff = {a.mean():.3f} ± {a.std(ddof=1)/np.sqrt(len(a)):.3f} | prediction 1/2[1+(1-2pi)^r] = {p.mean():.3f} | {len(a)} layers")
    json.dump(out, open('../data/parity_closed_form.json', 'w'), indent=1)
