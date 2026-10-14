"""Exact annealed Renyi-2 density -log2 Z_A / m for edge growth (k layers per birth), against theta_k (Theorem E)."""
import numpy as np, json, time
from domain_wall import logZ_wall
from scipy.optimize import minimize_scalar
def E(v, q):
    p = (1 + v) / 2; h = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return np.log((q * q + 1) / q) - h
def s2(k, q=2):
    r = minimize_scalar(lambda v: E(v, q) / (v + 1 / k), bounds=(0, 0.999999), method='bounded')
    return min(1.0, r.fun / np.log(q))
out = {}
for k in (1, 2, 4, 8):
    pred = s2(k); rows = []
    for m in (64, 256, 1024, 4096, 16384):
        t = time.time()
        val = -logZ_wall(2, 2, 2 * m, k, m) / np.log(2) / m
        rows.append((m, val))
        print(f"k={k} m={m:5d}: -log2 Z_A / m = {val:.5f}  (predicted {pred:.5f}, deviation {val-pred:+.5f}, m·deviation/log2 m = {(val-pred)*m/np.log2(m):+.3f}) [{time.time()-t:.1f}s]", flush=True)
    out[k] = dict(pred=pred, rows=rows)
json.dump(out, open('../data/annealed_density.json', 'w'), indent=1)
