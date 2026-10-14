#!/usr/bin/env python3
"""Births correlated with the past (Table 6): edge growth (framework of pair_births.py), uniform Clifford gates.
With probability f, the newborn (born |0> at the right edge) immediately undergoes a uniform Clifford gate with an
EXISTING site: its immediate neighbour (variant 'loc') or a uniformly drawn existing site ('nl').
k scrambling layers per added degree of freedom. Measurement: G = m - S(A), A = left half; slope on [200, 800].
The mutual information newborn/past right after each correlated birth is also recorded: I = 2 S(x)."""
import sys, json, time
import numpy as np
sys.path.insert(0, '.')
from multiprocessing import Pool
from clifford_uniform import CTab, SP4

NMAX = 800
FS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0)


def run(args):
    variant, k, f, seed = args
    rng = np.random.default_rng(31_000_000 + 1_000_000 * k + int(round(1000 * f)) * 1000 + seed + (0 if variant == 'nl' else 500))
    tab = CTab(NMAX + 2)
    pos = [tab.born(), tab.born()]
    credit, par, rec, mis = 0.0, 0, [], []
    while len(pos) < NMAX:
        x = tab.born(); pos.append(x)
        if f > 0 and rng.random() < f:
            y = pos[-2] if variant == 'loc' else pos[int(rng.integers(0, len(pos) - 1))]
            tab.layer_sym(np.array([x]), np.array([y]), SP4[rng.integers(720, size=1)], record=False)
            mis.append(2 * int(tab.entropy(np.array([x], dtype=np.int64))))
        credit += k
        while credit >= 1 - 1e-12:
            n = len(pos)
            a = np.array([pos[i] for i in range(par, n - 1, 2)], dtype=np.int64)
            b = np.array([pos[i + 1] for i in range(par, n - 1, 2)], dtype=np.int64)
            tab.layer_sym(a, b, SP4[rng.integers(720, size=len(a))], record=False)
            par ^= 1; credit -= 1
        n = len(pos)
        if n % 20 == 0:
            m = n // 2
            rec.append((n, int(m - tab.entropy(np.array(pos[:m], dtype=np.int64)))))
    return variant, k, f, seed, rec, (float(np.mean(mis)) if mis else None), len(mis)


if __name__ == '__main__':
    t = time.time()
    jobs = [('nl', k, f, s) for k in (1, 4) for f in FS for s in range(8)]
    jobs += [('loc', k, f, s) for k in (1, 4) for f in (0.1, 1.0) for s in range(8)]
    out = []
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 2) as P:
        for r in P.imap_unordered(run, jobs):
            out.append(r); json.dump(out, open('../data/correlated_births.json', 'w'))
    res = {}
    for v, k, f, s, rec, mi, nmi in out:
        n = np.array([a for a, b in rec], float); G = np.array([b for a, b in rec], float)
        sel = (n >= 200) & (n <= NMAX)
        res.setdefault((v, k, f), []).append((np.polyfit(n[sel], G[sel], 1)[0], mi))
    for k in (1, 4):
        base = np.array([x for x, _ in res[('nl', k, 0.0)]])
        for v in ('nl', 'loc'):
            for f in FS:
                if (v, k, f) not in res: continue
                x = np.array([a for a, _ in res[(v, k, f)]]); mi = [b for _, b in res[(v, k, f)] if b is not None]
                print(f"k={k} {v:3s} f={f:<5} slope = {x.mean():.4f} ± {x.std(ddof=1)/np.sqrt(len(x)):.4f} | "
                      f"relative reduction = {1 - x.mean()/base.mean():+.3f} | mean I per correlated birth = "
                      f"{(np.mean(mi) if mi else float('nan')):.2f} bit", flush=True)
    print(f"[{time.time()-t:.0f}s]")
