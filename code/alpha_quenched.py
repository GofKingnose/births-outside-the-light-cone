"""Quenched E[G] for uniform Clifford circuits with growth n(L) - N0 = floor(L^alpha) at the right edge."""
import numpy as np, sys, json
sys.path.insert(0, '.')
from multiprocessing import Pool
from clifford_uniform import CTab
from domain_wall import events_general
def run(spec):
    alpha, m, seed = spec
    rng = np.random.default_rng(seed); N0 = 2; nb = 2 * m - N0
    bl = [int(np.ceil(i ** (1 / alpha))) for i in range(1, nb + 1)]
    nxt = int(np.ceil((nb + 1) ** (1 / alpha))); T = bl[-1] + (nxt - bl[-1]) // 2 + 1
    t = CTab(2 * m)
    for _ in range(N0): t.born()
    n = N0
    for e in events_general(N0, bl, T):
        if e < 0: t.born(); n += 1; continue
        a = np.arange(e, n - 1, 2); t.layer_unif(a, a + 1, rng, record=False)
    return dict(alpha=alpha, m=m, seed=seed, G=m - int(t.entropy(np.arange(m))))
specs = [(a, m, 1000 * int(100 * a) + 10 * m + s) for (a, m) in ((0.5, 64), (0.75, 64), (0.75, 128), (0.75, 256), (0.9, 128), (0.9, 256)) for s in range(8)]
with Pool(2) as P:
    out = P.map(run, specs)
json.dump(out, open('../data/alpha_quenched.json', 'w'))
for (a, m) in sorted(set((r['alpha'], r['m']) for r in out)):
    G = np.array([r['G'] for r in out if r['alpha'] == a and r['m'] == m], float)
    print(f"alpha={a} m={m}: E[G] (uniform Clifford, bits) = {G.mean():.2f} ± {G.std()/np.sqrt(len(G)):.2f}")
