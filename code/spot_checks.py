#!/usr/bin/env python3
"""Quick reproducibility check (a few minutes): re-runs single jobs of the long campaigns with their stored seeds
and compares the result with the records stored in ../data/ and ../field_test/. Integer results (stabilizer
entropies) must agree exactly; floating-point results (Haar state vectors, Gaussian field) to 1e-9.
Usage: python3 spot_checks.py            (from the code/ directory)"""
import sys, json, pickle, time, numpy as np
sys.path.insert(0, '.')
D = '../data/'
ok_all = True


def report(name, ok, t0, detail=''):
    global ok_all
    ok_all &= bool(ok)
    print(f"{'OK  ' if ok else 'FAIL'} {name:58s} [{time.time() - t0:5.0f}s] {detail}", flush=True)


def load_module_until(path, marker):
    """Executes a script up to a marker line (for scripts whose campaign runs at import time)."""
    src = open(path).read().split(marker)[0]
    ns = {'__name__': 'spot'}; exec(compile(src, path, 'exec'), ns); return ns


# 1. Edge growth, quenched (Table 3): one circuit k = 4, m = 128, seed 400
t0 = time.time()
from edge_growth_quenched import edge_run
r = edge_run((4, 128, 400, 2))
ref = [x for x in json.load(open(D + 'edge_growth_quenched.json')) if x.get('kind') == 'edge' and x['k'] == 4 and x['m'] == 128 and x['seed'] == 400]
report('edge growth k=4 m=128 seed 400', len(ref) == 1 and ref[0]['S'] == r['S'], t0, f"S = {r['S']}")

# 2. Edge growth, large k (Table 3): k = 16, m = 256, seed 10600
t0 = time.time()
r = edge_run((16, 256, 9000 + 1600, 2))
ref = [x for x in json.load(open(D + 'edge_growth_large_k.json')) if x['k'] == 16 and x['m'] == 256 and x['seed'] == 10600]
report('edge growth k=16 m=256 seed 10600', len(ref) == 1 and ref[0]['S'] == r['S'], t0, f"S = {r['S']}")

# 3. Pair births (Table 5): variant P2, k = 4, seed 0
t0 = time.time()
from pair_births import run as pair_run
v, k, s, rec = pair_run(('P2', 4, 0))
ref = [x for x in json.load(open(D + 'pair_births.json')) if x[0] == 'P2' and x[1] == 4 and x[2] == 0]
report('pair births P2 k=4 seed 0', len(ref) == 1 and [list(p) for p in rec] == [list(p) for p in ref[0][3]], t0, f"{len(rec)} checkpoints")

# 4. Correlated births (Table 6): non-local, k = 4, f = 0.1, seed 0
t0 = time.time()
from correlated_births import run as corr_run
out = corr_run(('nl', 4, 0.1, 0))
ref = [x for x in json.load(open(D + 'correlated_births.json')) if x[0] == 'nl' and x[1] == 4 and abs(x[2] - 0.1) < 1e-12 and x[3] == 0]
same = len(ref) == 1 and [list(p) for p in out[4]] == [list(p) for p in ref[0][4]] and abs((out[5] or 0) - (ref[0][5] or 0)) < 1e-9
report('correlated births non-local k=4 f=0.1 seed 0', same, t0, f"{len(out[4])} checkpoints, mean MI = {out[5]:.4f} bit")

# 5. Quenched alpha < 1 (Figure 4): alpha = 0.75, m = 64, seed 75640
t0 = time.time()
ns = load_module_until('alpha_quenched.py', 'specs = ')
r = ns['run']((0.75, 64, 1000 * 75 + 10 * 64 + 0))
ref = [x for x in json.load(open(D + 'alpha_quenched.json')) if x['alpha'] == 0.75 and x['m'] == 64 and x['seed'] == 75640]
report('alpha=0.75 m=64 seed 75640', len(ref) == 1 and ref[0]['G'] == r['G'], t0, f"G = {r['G']}")

# 6. Haar against Clifford (Table 4): first Haar circuit and first Clifford batch, k = 1, m = 6
t0 = time.time()
from haar_vs_clifford import haar_run
from clifford_uniform import batch_entropy
H = haar_run((2, 1, 6, 10_000_000 + 100_000 * 1 + 1000 * 6 + 0))
rec = [x for x in json.load(open(D + 'haar_vs_clifford.json')) if x['k'] == 1 and x['m'] == 6][0]
C = batch_entropy(2, 12, 1, 6, 2000, np.random.default_rng(20_000 + 100 * 1 + 6)).astype(float)
same = abs(H[0] - rec['H_SvN'][0]) < 1e-9 and abs(H[1] - rec['H_S2'][0]) < 1e-9 and C.tolist() == rec['C_S'][:2000]
report('Haar circuit 0 and Clifford batch 0, k=1 m=6', same, t0, f"S_vN = {H[0]:.6f}, S_2 = {H[1]:.6f} bits")

# 7. 1D insertion campaign (Table 7): N proportional to t, seed 0, up to N = 1600 (matched against the 4 stored runs)
t0 = time.time()
from growth_campaigns import job
kind, mode, seed, rec, dt = job(('1d', 'p1', 0, 1600))
stored = json.load(open(D + 'growth_campaigns.json'))['1d']['p1']
report('1D insertion N~t seed 0 (N <= 1600)', any(rec == x for x in stored), t0, f"G/N(1600) = {rec[-1]['G'] / rec[-1]['n']:.4f}")

# 8. Free field (Section 9): scan run H = 0.05, product rule
t0 = time.time()
sys.path.insert(0, '../field_test')
import run_field
r = run_field.run(('dS', None, 0.05, 0.05, 'prod', 0, 32, 4, -1))
ref = [x for x in pickle.load(open('../field_test/field_scan.pkl', 'rb')) if x['spec']['rule'] == 'prod' and x['spec']['H0'] == 0.05]
e1 = np.array([b['eps'] for b in r['births']]); e0 = np.array([b['eps'] for b in ref[0]['births']])
report('free field, de Sitter H=0.05, product births', len(ref) == 1 and np.allclose(e1, e0, rtol=1e-9, atol=1e-12), t0,
       f"eps per newborn = {e1.mean():.5f} Lambda")

print('ALL SPOT CHECKS PASSED' if ok_all else 'SOME SPOT CHECKS FAILED')
sys.exit(0 if ok_all else 1)
