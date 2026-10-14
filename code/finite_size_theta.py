#!/usr/bin/env python3
"""Finite-size scaling of the annealed density (Theorem E, remark on finite-size corrections), exact
single-wall recursion. c(m) = -ln Z_A - theta_k m for A = [0, m), N = 2m, one birth every k layers, q = 2."""
import sys, json
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0, '.')
from domain_wall import logZ_wall

q = 2.0; K = q / (q * q + 1)
theta = lambda k: brentq(lambda t: 2 * K * np.cosh(t) - np.exp(-t / k), 1e-9, np.log(q) - 1e-12)

ms = sorted(set([8, 9, 12, 13, 16, 17, 24, 25, 32, 33, 48, 49, 64, 65, 96, 97, 128, 129, 192, 193, 256, 257,
                 384, 385, 512, 513, 768, 769, 1024, 1025, 2048, 2049, 4096, 4097]))
out = {}
for k in (1, 2, 4, 8):
    th = theta(k)
    for N0 in (2, 4, 10):
        rows = []
        for m in ms:
            if 2 * m <= N0: continue
            lz = logZ_wall(q, N0, 2 * m, k, m)
            rows.append((m, -lz - th * m, -lz / (th * m) - 1))
        out[f'k{k}_N0{N0}'] = dict(theta=th, rows=rows)
        r = np.array(rows)
        lim = {par: r[(r[:, 0] % 2 == par)][-1, 1] for par in (0, 1)}
        bound_lo, bound_hi = -np.log(np.exp(2 * th) + 2), 2 * th + np.log(2)
        # first size from which |c(m) - c(inf)| < 1e-6 (same parity) and relative error < 1 %
        conv = {par: next((int(m) for m, c, e in rows if m % 2 == par and
                           all(abs(c2 - lim[par]) < 1e-6 for m2, c2, e2 in rows if m2 >= m and m2 % 2 == par)), None)
                for par in (0, 1)}
        rel1 = next((int(m) for m, c, e in rows if all(abs(e2) < 0.01 for m2, c2, e2 in rows if m2 >= m)), None)
        out[f'k{k}_N0{N0}'].update(lim=lim, conv=conv, rel1=rel1, bounds=(bound_lo, bound_hi))
        print(f"k={k} N0={N0:2d} theta={th:.6f} nats | c(inf) even m={lim[0]:+.6f} odd m={lim[1]:+.6f} | bounds [{bound_lo:+.3f}, {bound_hi:+.3f}] "
              f"| |c-c_inf|<1e-6 from m={conv[0]}/{conv[1]} | rel. error < 1 % from m={rel1}")
json.dump(out, open('../data/finite_size_theta.json', 'w'), indent=1)
