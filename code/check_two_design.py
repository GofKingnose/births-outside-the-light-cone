"""Average purity E[2^{-S(A)}] of uniform Clifford circuits (and of the rand2 ensemble) against the
exact annealed value Z_A of the single-wall recursion (Section 8.2). Output: ../data/two_design_check.json.
Also prints how the rand2 construction covers Sp(4, F2): it is not uniform, hence not a 2-design.
The original campaign also listed the geometry (N0, n, k) = (2, 32, 1); it was interrupted by a time limit
and is not part of the reported results, so it is not listed here."""
import numpy as np, json, time
from collections import Counter
from clifford_uniform import batch_entropy, SP4, RAND2_TABLE
from domain_wall import logZ_wall
index = {M.tobytes(): i for i, M in enumerate(SP4)}
cover = Counter(index[M.tobytes()] for M in RAND2_TABLE)
print(f"rand2: {len(RAND2_TABLE)} combinations reach {len(cover)} of {len(SP4)} elements of Sp(4,F2), "
      f"with multiplicities from {min(cover.values())} to {max(cover.values())} (uniform would be {len(RAND2_TABLE) / len(SP4):.1f} each)")
rng = np.random.default_rng(20261008)
geoms = [(2, 8, 1), (2, 8, 2), (4, 10, 1), (2, 12, 4), (2, 16, 1), (2, 16, 2), (2, 20, 1), (2, 24, 2), (2, 24, 4)]
out = dict(rand2_cover=dict(reached=len(cover), total=len(SP4), min=min(cover.values()), max=max(cover.values())), rows=[])
for (N0, n, k) in geoms:
    m = n // 2
    Z = np.exp(logZ_wall(2, N0, n, k, m))
    line = f"N0={N0:2d} n={n:2d} k={k} m={m:2d}: exact Z_A = {Z:.5f}"
    rec = dict(N0=N0, n=n, k=k, m=m, Z=Z)
    for gate in ('unif', 'rand2'):
        vals = []
        t = time.time()
        for _ in range(10):
            vals.append(2.0 ** (-batch_entropy(N0, n, k, m, 20000, rng, gate)))
        v = np.concatenate(vals); mu = v.mean(); se = v.std() / np.sqrt(len(v))
        rec[gate] = (mu, se)
        line += f" | {gate}: {mu:.5f} ± {se:.5f} (z = {(mu - Z) / se:+.1f})"
    print(line, flush=True); out['rows'].append(rec)
json.dump(out, open('../data/two_design_check.json', 'w'), indent=1)
