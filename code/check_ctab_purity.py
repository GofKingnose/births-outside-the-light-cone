# Check: the CTab simulator (used at large sizes) reproduces the exact Z_A at small size.
import numpy as np
from clifford_uniform import CTab
from domain_wall import events, logZ_wall
rng = np.random.default_rng(99)
for (N0, n_final, k) in ((2, 16, 2), (2, 20, 1)):
    m = n_final // 2; vals = []
    for _ in range(12000):
        t = CTab(n_final)
        for _ in range(N0): t.born()
        n = N0
        for e in events(N0, n_final, k):
            if e < 0: t.born(); n += 1; continue
            a = np.arange(e, n - 1, 2); t.layer_unif(a, a + 1, rng, record=False)
        vals.append(2.0 ** -t.entropy(np.arange(m)))
    v = np.array(vals); Z = np.exp(logZ_wall(2, N0, n_final, k, m))
    print(f"CTab N0={N0} n={n_final} k={k}: E[2^-S] = {v.mean():.5f} ± {v.std()/np.sqrt(len(v)):.5f} | exact Z = {Z:.5f} | z = {(v.mean()-Z)/(v.std()/np.sqrt(len(v))):+.2f}", flush=True)
