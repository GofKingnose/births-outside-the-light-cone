#!/usr/bin/env python3
"""1D and 2D growth campaigns with UNIFORM two-qubit Clifford gates (Tables 6 and 7, Figures 5 and 6),
with an exact check of Theorem A at every checkpoint. Output: ../data/growth_campaigns.json."""
import json, time, sys
sys.path.insert(0, '.')
import uniform_gate_patch                      # noqa: F401  (patch applied before anything else)
from multiprocessing import Pool
import growth_1d, growth_2d

def job(a):
    kind, mode, seed, N = a
    r = growth_1d.run((mode, seed, N)) if kind == '1d' else growth_2d.run((mode, seed, N))
    return kind, r[0], r[1], r[2], r[3]

if __name__ == '__main__':
    t = time.time()
    jobs = [('1d', m, s, 1600) for m in ['edge-1', 'edge-4', 'p0.5', 'p1', 'p2', 'exp', 'p1pair'] for s in range(4)]
    jobs += [('2d', m, s, 6400) for m in ['front-0.25', 'front-0.5', 'front-1', 'rate-4', 'rate-16'] for s in range(4)]
    jobs += [('2d', 'eden-0.25', s, 6400) for s in range(10, 18)] + [('2d', 'eden-1', s, 6400) for s in range(20, 28)]
    jobs += [('2d', m, s, 6400) for m in ('acc-0.5', 'expo-1', 'acc-0.05', 'expo-0.1') for s in range(4)]
    out = {'1d': {}, '2d': {}}
    with Pool(2) as p:
        for kind, mode, seed, rec, dt in p.imap_unordered(job, jobs):
            out[kind].setdefault(mode, []).append(rec)
            json.dump(out, open('../data/growth_campaigns.json', 'w'), indent=1)
            print(f"{kind} {mode:10s} seed {seed} [{dt:5.0f}s] " + " | ".join(
                f"N={r['n']} G/N={r['G']/r['n']:.3f} LB/N={r['G_LB']/r['n']:.3f}{' VIOL' if r['viol'] else ''}" for r in rec[-2:]), flush=True)
    nv = sum(r['viol'] for k in out for v in out[k].values() for rec in v for r in rec)
    nc = sum(len(rec) for k in out for v in out[k].values() for rec in v)
    print(f"Theorem A: {nv} violation(s) / {nc} checkpoints   [{time.time()-t:.0f}s]", flush=True)
