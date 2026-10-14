#!/usr/bin/env python3
"""Edge growth, k = 16 and 32, uniform Clifford gates: E[G]/N against the proven bracket."""
import sys, json
sys.path.insert(0, '.')
from multiprocessing import Pool
from edge_growth_quenched import edge_run
if __name__ == '__main__':
    specs = [(k, m, 9000 + 100 * k + s, 2) for k in (16, 32) for m in (256, 512) for s in range(6)]
    with Pool(2) as P:
        out = P.map(edge_run, specs)
    json.dump(out, open('../data/edge_growth_large_k.json', 'w'))
    import numpy as np
    for k in (16, 32):
        for m in (256, 512):
            S = np.array([r['S'] for r in out if r['k'] == k and r['m'] == m], float) / m
            print(f"k={k} m={m}: E[S]/m = {S.mean():.4f} ± {S.std()/np.sqrt(len(S)):.4f} ; E[G]/N = {0.5*(1-S.mean()):.4f}")
