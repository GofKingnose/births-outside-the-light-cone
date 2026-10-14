#!/usr/bin/env python3
"""Homogeneous computation of the statistics quoted in the tables of the paper, from the campaign data files
(uniform Clifford gates). Single convention: mean over independent seeds, standard error = sample standard
deviation (ddof = 1) / sqrt(n). Exponents: bootstrap over seeds (10 000 resamples).
Slopes: ordinary least squares per seed, then mean and standard error across seeds."""
import sys, json
import numpy as np
sys.path.insert(0, '.')
from domain_wall import events, logZ_wall, cone_lower
from continuum_geometry import f_insert

D = '../data/'
se = lambda x: np.std(x, ddof=1) / np.sqrt(len(x))
LN2 = np.log(2)
rng = np.random.default_rng(12345)
res = {}

print('=== Edge growth table: E[G]/N against the exact bounds at the same size ===')
P45 = json.load(open(D + 'edge_growth_quenched.json')); EB = json.load(open(D + 'edge_growth_large_k.json'))
rows = []
for k, m, src in [(1, 1024, P45), (2, 1024, P45), (4, 1024, P45), (8, 1024, P45), (16, 512, EB), (32, 512, EB)]:
    S = np.array([r['S'] for r in src if r.get('kind', 'edge') == 'edge' and r['k'] == k and r['m'] == m], float)
    g = 0.5 * (1 - S / m)
    lz = logZ_wall(2, 2, 2 * m, k, m); UB = 0.5 * (1 + lz / LN2 / m); LB = cone_lower(2, events(2, 2 * m, k), m) / (2 * m)
    dq = S / m - (-lz / LN2 / m)                                  # quenched density - annealed density (bit/qudit)
    rows.append(dict(k=k, m=m, n=len(S), G=g.mean(), seG=se(g), UB=UB, LB=LB, rel=(UB - g.mean()) / UB, serel=se(g) / UB,
                     dq=dq.mean(), sedq=se(dq)))
    print(f"k={k:2d} m={m} ({len(S)} seeds) E[G]/N = {g.mean():.4f} ± {se(g):.4f} | LB = {LB:.4f} | UB(m) = {UB:.4f} | "
          f"(UB-G)/UB = {100*(UB-g.mean())/UB:+.2f} ± {100*se(g)/UB:.2f} % | quenched-annealed = {dq.mean():+.4f} ± {se(dq):.4f} bit/qudit")
res['edge'] = rows

print('\n=== Pairs table: slopes dG/dN on [200, 800], 8 seeds ===')
PU = json.load(open(D + 'pair_births.json')); sl = {}
for v, k, s, rec in PU:
    n = np.array([a for a, b in rec], float); G = np.array([b for a, b in rec], float); sel = (n >= 200) & (n <= 800)
    sl.setdefault((v, k), []).append(np.polyfit(n[sel], G[sel], 1)[0])
res['pairs'] = {}
for k in (0, 1, 2, 4, 8, 16, 32):
    line = f"k={k:2d}"
    for v in ('ref', 'P1', 'P2'):
        x = np.array(sl[(v, k)]); line += f" | {v}: {x.mean():.4f} ± {se(x):.4f}"
        res['pairs'][f'{v}_{k}'] = (x.mean(), se(x))
    r, p2 = np.array(sl[('ref', k)]), np.array(sl[('P2', k)])
    ratio = r.mean() / p2.mean(); eratio = ratio * np.hypot(se(r) / r.mean(), se(p2) / p2.mean()) if p2.mean() > 0 else np.nan
    res['pairs'][f'ratio_{k}'] = (ratio, eratio)
    print(line + f" | R/P2 = {ratio:.2f} ± {eratio:.2f}")

U = json.load(open(D + 'growth_campaigns.json'))
def tab(d, mode):
    v = d[mode]; Ns = [r['n'] for r in v[0]]
    G = np.array([[r['G'] for r in rec] for rec in v], float); L = np.array([[r['G_LB'] for r in rec] for rec in v], float)
    S = np.array([[r['S'] for r in rec] for rec in v], float)
    return np.array(Ns), G, L, S

print('\n=== 1D table (insertion), 4 seeds ===')
V3 = json.load(open(D + 'parity_closed_form_44seeds.json')); V3a = json.load(open(D + 'parity_closed_form.json'))
veff = {'p1': V3['p1']['v_eff'], 'p2': V3['p2']['v_eff'], 'exp': V3['exp']['v_eff'], 'p0.5': V3a['p0.5']['v_eff'], 'p1pair': 1.0}
res['1d'] = {}
for mode in ['edge-1', 'edge-4', 'p0.5', 'p1', 'p1pair', 'p2', 'exp']:
    N, G, L, S = tab(U['1d'], mode); i4, i16 = list(N).index(400), list(N).index(1600)
    g4, g16, l16, s16 = G[:, i4] / 400, G[:, i16] / 1600, L[:, i16] / 1600, S[:, i16]
    law = mode.replace('pair', '')
    pred = f_insert(law, 0.5, 16, 1600) / 2 if not mode.startswith('edge') else None
    predv = f_insert(law, 0.5 * veff[mode], 16, 1600) / 2 if mode in veff else None
    res['1d'][mode] = dict(g4=(g4.mean(), se(g4)), g16=(g16.mean(), se(g16)), l16=(l16.mean(), se(l16)), s16=(s16.mean(), se(s16)),
                           pred=pred, predv=predv, veff=veff.get(mode))
    print(f"{mode:7s} G/N(400) = {g4.mean():.3f} ± {se(g4):.3f} | G/N(1600) = {g16.mean():.4f} ± {se(g16):.4f} | "
          f"G_LB/N(1600) = {l16.mean():.4f} ± {se(l16):.4f} | S(1600) = {s16.mean():.1f} ± {se(s16):.1f} | pred. v=1: {pred} | pred. v_eff ({veff.get(mode)}): {predv}")

print('\n=== 2D table ===')
res['2d'] = {}
for mode in ['front-0.25', 'front-0.5', 'front-1', 'eden-0.25', 'eden-1', 'rate-16', 'rate-4']:
    N, G, L, S = tab(U['2d'], mode); i4, i16, i64 = [list(N).index(x) for x in (400, 1600, 6400)]
    vals = {x: G[:, i] / x for x, i in ((400, i4), (1600, i16), (6400, i64))}
    l64 = L[:, i64] / 6400
    ex = lambda Gs: np.log(Gs[:, i64].mean() / Gs[:, i16].mean()) / np.log(4)
    boots = [ex(G[rng.integers(0, len(G), len(G))]) for _ in range(10000)]
    rat = G[:, i64] / L[:, i64]
    res['2d'][mode] = dict(n=len(G), g={x: (v.mean(), se(v)) for x, v in vals.items()}, l64=(l64.mean(), se(l64)),
                           expo=(ex(G), float(np.std(boots))), ratio=(G[:, i64].mean() / L[:, i64].mean(), se(rat)))
    print(f"{mode:11s} ({len(G)} seeds) G/N: " + ' '.join(f"{v.mean():.3f}±{se(v):.3f}" for v in vals.values()) +
          f" | G_LB/N(6400) = {l64.mean():.3f} ± {se(l64):.3f} | G/G_LB = {G[:, i64].mean()/L[:, i64].mean():.3f} | "
          f"exponent = {ex(G):.2f} ± {np.std(boots):.2f} (bootstrap)")

print('\n=== Regime alpha < 1, quenched values (8 seeds) ===')
P6 = json.load(open(D + 'alpha_quenched.json')); res['alpha'] = {}
for (a, m) in sorted(set((r['alpha'], r['m']) for r in P6)):
    G = np.array([r['G'] for r in P6 if r['alpha'] == a and r['m'] == m], float)
    res['alpha'][f'{a}_{m}'] = (G.mean(), se(G))
    print(f"alpha={a} m={m}: E[G] = {G.mean():.2f} ± {se(G):.2f} bits")

json.dump(res, open(D + 'table_statistics.json', 'w'), indent=1, default=float)
