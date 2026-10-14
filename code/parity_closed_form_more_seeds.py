#!/usr/bin/env python3
"""Additional seeds (4..43) for the closed form of v_eff (laws p2, exp, p1). Added AFTER reading the 4-seed
results of parity_closed_form.py, to separate statistical noise from a defect of the closed form."""
import json, numpy as np
import sys; sys.path.insert(0, ".")
from parity_locking import run_struct
from parity_closed_form import diagnose_pred
out = {}
for mode in ('p2', 'exp', 'p1'):
    A, Pp = [], []
    for seed in range(4, 44):
        pos, ev, nid = run_struct(mode, seed, 1600)
        a, p = diagnose_pred(pos, ev, nid); A.append(a); Pp.append(p)
    a = np.concatenate(A); p = np.concatenate(Pp)
    out[mode] = dict(v_eff=float(a.mean()), se=float(a.std(ddof=1) / np.sqrt(len(a))), prediction=float(p.mean()), layers=int(len(a)))
    print(f"{mode:4s} v_eff = {a.mean():.3f} ± {a.std(ddof=1)/np.sqrt(len(a)):.3f} | prediction = {p.mean():.3f} | {len(a)} layers")
json.dump(out, open('../data/parity_closed_form_44seeds.json', 'w'), indent=1)
