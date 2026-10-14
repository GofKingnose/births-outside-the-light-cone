"""Edge growth with uniform Clifford gates: quenched S(A) for A = [0, m), N = 2m (Table 3), and the quenched
entanglement velocity on a fixed chain of 512 qubits. Usage: python3 edge_growth_quenched.py <n_cores>"""
import numpy as np, json, time, sys
from multiprocessing import Pool
from clifford_uniform import CTab
from domain_wall import events

def edge_run(spec):
    k, m, seed, N0 = spec
    rng = np.random.default_rng(seed)
    n_final = 2 * m
    t = CTab(n_final)
    for _ in range(N0): t.born()
    n = N0
    for e in events(N0, n_final, k):
        if e < 0:
            t.born(); n += 1; continue
        a = np.arange(e, n - 1, 2)
        t.layer_unif(a, a + 1, rng, record=False)
    S = t.entropy(np.arange(m))
    return dict(kind='edge', k=k, m=m, seed=seed, S=int(S))

def ve_run(spec):
    N, Tmax, seed = spec
    rng = np.random.default_rng(seed)
    t = CTab(N)
    for _ in range(N): t.born()
    rec = []
    for L in range(Tmax):
        a = np.arange(L % 2, N - 1, 2)
        t.layer_unif(a, a + 1, rng, record=False)
        if (L + 1) % 10 == 0:
            rec.append((L + 1, int(t.entropy(np.arange(N // 2)))))
    return dict(kind='vE', N=N, seed=seed, rec=rec)

if __name__ == '__main__':
    specs_ve = [(512, 600, 1000 + s) for s in range(8)]
    specs = [(k, m, 100 * k + s, 2) for k in (1, 2, 4, 8) for m in (128, 256, 512) for s in range(12)]
    specs += [(k, 1024, 5000 + 100 * k + s, 2) for k in (1, 2, 4, 8) for s in range(6)]
    out = []
    with Pool(int(sys.argv[1])) as P:
        for r in P.imap_unordered(ve_run, specs_ve):
            out.append(r); print('vE', r['seed'], r['rec'][-1], flush=True)
        json.dump(out, open('../data/edge_growth_quenched.json', 'w'))
        for r in P.imap_unordered(edge_run, specs):
            out.append(r); print('edge', r['k'], r['m'], r['seed'], r['S'], flush=True)
            json.dump(out, open('../data/edge_growth_quenched.json', 'w'))
    print('DONE', flush=True)
