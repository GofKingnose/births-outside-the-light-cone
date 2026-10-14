#!/usr/bin/env python3
"""
2D growth: constant-speed front vs constant rate vs Eden; exact check of Theorem A
(run through growth_campaigns.py, which applies uniform_gate_patch).

Square lattice, centre (-1/2, -1/2), cut A = {x <= -1}, Ā = {x >= 0}.
One round = 4 sub-layers of random Clifford gates (h-even, h-odd, v-even, v-odd)
between neighbours that are both born; 1 round = 1 unit of time; sub-layer u at time u/4.
Strict causal speed normal to the cut: 2 sites/round.
Growth:
  front-s : a site is born when the disc of radius s t reaches it (s sites/round);
  acc-a   : accelerated front R = a t^2;   expo-H : exponential front R = r0 e^{H t};
  rate-g  : sites ordered by distance (ties broken at random), g births/round;
  eden-r  : each empty site adjacent to the cluster is born at rate r (Gillespie).
"""
import sys, json, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, '.')
from tableau import PTab as Tab, causal_bound

CHECK = (100, 200, 400, 800, 1600, 3200, 6400)
LH = 60                                   # coordinates x, y in [-LH, LH-1]
W = 2 * LH

def coords():
    xs, ys = np.meshgrid(np.arange(-LH, LH), np.arange(-LH, LH), indexing='ij')
    return xs.ravel(), ys.ravel()

X, Y = coords()
R = np.hypot(X + 0.5, Y + 0.5)
idx = lambda x, y: (x + LH) * W + (y + LH)

def pairs():
    P = []
    for orient, par in (('h', 0), ('h', 1), ('v', 0), ('v', 1)):
        a, b = [], []
        for x in range(-LH, LH):
            for y in range(-LH, LH):
                if orient == 'h' and x % 2 == par and x + 1 < LH:
                    a.append(idx(x, y)); b.append(idx(x + 1, y))
                if orient == 'v' and y % 2 == par and y + 1 < LH:
                    a.append(idx(x, y)); b.append(idx(x, y + 1))
        P.append((np.array(a), np.array(b)))
    return P
PAIRS = pairs()

def schedule(mode, Nmax, rng):
    kind, val = mode.split('-'); val = float(val)
    if kind == 'front':
        order = np.lexsort((rng.random(len(R)), R))[:Nmax]
        return (R[order] - R[order[0]]) / val, order
    if kind == 'acc':                       # accelerated front R = a t^2 (alpha = 2)
        order = np.lexsort((rng.random(len(R)), R))[:Nmax]
        return np.sqrt((R[order] - R[order[0]]) / val), order
    if kind == 'expo':                      # exponential front R = r0 e^{H t}
        order = np.lexsort((rng.random(len(R)), R))[:Nmax]
        return np.log(R[order] / R[order[0]]) / val, order
    if kind == 'rate':
        order = np.lexsort((rng.random(len(R)), R))[:Nmax]
        return np.arange(Nmax) / val, order
    if kind == 'eden':
        occ = np.zeros(W * W, bool); times, order = [], []
        per = []; inper = np.zeros(W * W, bool)
        def add(i, t):
            occ[i] = True; times.append(t); order.append(i)
            x, y = X[i], Y[i]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                xx, yy = x + dx, y + dy
                if -LH <= xx < LH and -LH <= yy < LH:
                    j = idx(xx, yy)
                    if not occ[j] and not inper[j]:
                        inper[j] = True; per.append(j)
        t = 0.0
        for (x, y) in ((-1, -1), (-1, 0), (0, -1), (0, 0)):
            i = idx(x, y)
            if inper[i]:
                per.remove(i); inper[i] = False
            add(i, 0.0)
        per[:] = [j for j in per if not occ[j]]
        for j in range(W * W): inper[j] = False
        for j in per: inper[j] = True
        while len(order) < Nmax:
            t += rng.exponential(1 / (val * len(per)))
            k = int(rng.integers(len(per))); i = per[k]; per[k] = per[-1]; per.pop(); inper[i] = False
            add(i, t)
        return np.array(times), np.array(order)

def run(args):
    mode, seed, Nmax = args
    rng = np.random.default_rng(1000 + seed)
    tb, order = schedule(mode, Nmax, rng)
    alive = np.zeros(W * W, bool); idof = -np.ones(W * W, np.int64)
    tab = Tab(Nmax); nb, u, rec = 0, 0, []
    checks = [c for c in CHECK if c <= Nmax]; t0 = time.time()
    while checks:
        tu = u / 4
        while nb < Nmax and tb[nb] <= tu + 1e-12:
            i = order[nb]; alive[i] = True; idof[i] = tab.born(); nb += 1
            if checks and tab.n == checks[0]:
                sites = np.flatnonzero(alive)
                A = idof[sites[X[sites] <= -1]]; B = idof[sites[X[sites] >= 0]]
                S = tab.entropy(A if len(A) <= len(B) else B)
                b1, b2 = causal_bound(tab.log, A, B, Nmax)
                m = min(len(A), len(B)); rmax = R[sites].max()
                rec.append(dict(n=int(tab.n), t=tu, nA=len(A), nB=len(B), S=int(S), G=int(m - S),
                                bA=b1, bB=b2, G_LB=int(m - min(b1, b2, m)), viol=bool(S > min(b1, b2)),
                                rmax=float(rmax), rmean=float(R[sites].mean())))
                checks.pop(0)
        a, b = PAIRS[u % 4]
        ok = alive[a] & alive[b]
        tab.layer(idof[a[ok]], idof[b[ok]], rng)
        u += 1
    return mode, seed, rec, time.time() - t0

