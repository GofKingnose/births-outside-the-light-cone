#!/usr/bin/env python3
"""Quenched entanglement velocity v_E of uniform Clifford circuits (Section 8.2): fixed chain of 512 qubits,
central cut, 8 seeds (records of kind 'vE' in ../data/edge_growth_quenched.json, produced by edge_growth_quenched.py).
Slope of S(t) per seed over three fit windows; mean and standard error (Bessel) over seeds.
Compared with the annealed value v_E^(2) = log2((q^2+1)/(2q)) = 0.3219 for q = 2, a lower bound for every 2-design."""
import json, numpy as np
R = [r for r in json.load(open('../data/edge_growth_quenched.json')) if r['kind'] == 'vE']
T = np.array([t for t, s in R[0]['rec']]); S = np.array([[s for t, s in r['rec']] for r in R], float)
for t0, t1 in ((50, 300), (100, 400), (200, 500)):
    sel = (T >= t0) & (T <= t1)
    sl = np.array([np.polyfit(T[sel], s[sel], 1)[0] for s in S])
    print(f"v_E (layers {t0}-{t1}) = {sl.mean():.4f} ± {sl.std(ddof=1) / np.sqrt(len(sl)):.4f} bit/layer ({len(sl)} seeds)")
print(f"annealed v_E^(2) = {np.log2(5 / 4):.4f} bit/layer")
