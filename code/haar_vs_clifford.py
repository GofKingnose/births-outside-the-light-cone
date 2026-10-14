#!/usr/bin/env python3
"""Haar circuits (state vector) against uniform Clifford circuits at EQUAL size, edge growth (Table 4).

Geometry identical to domain_wall.events: N0 qudits, a birth at the right edge before layer L if L = 0 (mod k),
layer L = gates (x, x+1) with x = L (mod 2), stop at N = 2m; A = [0, m).
Per circuit: S_vN, S_2 (Haar); S (Clifford, where S = S_2). Exact bounds: LB (cone), UB (single-wall recursion).
Usage: python3 haar_vs_clifford.py [n_cores] [large]   (set OPENBLAS_NUM_THREADS=1 when using several cores)
"""
import sys, json, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, '.')
from domain_wall import events, logZ_wall, cone_lower
from clifford_uniform import batch_entropy

LN2 = np.log(2.0)


def haar_gates(rng, G):
    z = (rng.normal(size=(G, 4, 4)) + 1j * rng.normal(size=(G, 4, 4))) / np.sqrt(2)
    Q, R = np.linalg.qr(z)
    d = np.diagonal(R, axis1=1, axis2=2)
    return Q * (d / np.abs(d))[:, None, :]


def haar_run(args):
    """One Haar circuit; returns (S_vN, S_2) in bits for A = [0, m)."""
    N0, k, m, seed = args
    rng = np.random.default_rng(seed)
    n = N0
    psi = np.zeros(2 ** N0, complex); psi[0] = 1.0           # qubit 0 = most significant bit
    for e in events(N0, 2 * m, k):
        if e < 0:                                            # birth at the right edge: |psi> ⊗ |0>
            new = np.zeros(2 * psi.size, complex); new[0::2] = psi; psi = new; n += 1
            continue
        xs = range(e, n - 1, 2)
        U = haar_gates(rng, len(xs))
        for U1, x in zip(U, xs):
            v = psi.reshape(2 ** x, 4, 2 ** (n - x - 2))
            psi = np.matmul(U1, v).reshape(-1)
    s = np.linalg.svd(psi.reshape(2 ** m, 2 ** (n - m)), compute_uv=False)
    p = s ** 2; p = p[p > 1e-16]; p /= p.sum()
    return float(-(p * np.log2(p)).sum()), float(-np.log2((p ** 2).sum()))


def geometry_bounds(N0, k, m):
    ev = events(N0, 2 * m, k)
    lz = logZ_wall(2, N0, 2 * m, k, m)
    return dict(LB=int(cone_lower(N0, ev, m)), UB=m + lz / LN2, Z=float(np.exp(lz)))


if __name__ == '__main__':
    ncpu = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    N0 = 2
    plan = [(6, 4000), (8, 2000), (10, 1000)]               # (m, number of Haar circuits)
    if len(sys.argv) > 2 and sys.argv[2] == 'large':
        plan = [(11, 150)]
    out = []
    t0 = time.time()
    with Pool(ncpu) as P:
        for m, nH in plan:
            for k in (1, 2, 4):
                b = geometry_bounds(N0, k, m)
                seeds = [(N0, k, m, 10_000_000 + 100_000 * k + 1000 * m + s) for s in range(nH)]
                H = np.array(P.map(haar_run, seeds, chunksize=max(1, nH // (4 * ncpu))))
                rng = np.random.default_rng(20_000 + 100 * k + m)
                C = np.concatenate([batch_entropy(N0, 2 * m, k, m, 2000, rng) for _ in range(10)]).astype(float)
                rec = dict(N0=N0, k=k, m=m, nH=nH, nC=len(C), **b,
                           H_SvN=H[:, 0].tolist(), H_S2=H[:, 1].tolist(), C_S=C.tolist())
                out.append(rec)
                json.dump(out, open('../data/haar_vs_clifford.json' if plan[0][0] != 11 else '../data/haar_vs_clifford_m11.json', 'w'))
                se = lambda x: x.std(ddof=1) / np.sqrt(len(x))
                pH = 2.0 ** (-H[:, 1]); pC = 2.0 ** (-C)
                print(f"m={m} k={k} | LB={b['LB']} UB={b['UB']:.3f} | Haar: G_vN={m-H[:,0].mean():.3f}±{se(H[:,0]):.3f} "
                      f"G_2={m-H[:,1].mean():.3f}±{se(H[:,1]):.3f} | Clifford: G={m-C.mean():.3f}±{se(C):.3f} | "
                      f"z_annealed Haar={(pH.mean()-b['Z'])/se(pH):+.2f} Clifford={(pC.mean()-b['Z'])/se(pC):+.2f} [{time.time()-t0:.0f}s]",
                      flush=True)
    print('DONE', flush=True)
