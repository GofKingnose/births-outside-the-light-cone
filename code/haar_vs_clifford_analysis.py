#!/usr/bin/env python3
"""Analysis of haar_vs_clifford.py (Table 4): annealed check, Theorem A, positions in the bracket,
the gap E[S] - E[S_2] for Haar circuits and simple fits of its size dependence."""
import json
import numpy as np
from scipy.optimize import curve_fit
se = lambda x: np.std(x, ddof=1) / np.sqrt(len(x))
recs = []
for f in ('../data/haar_vs_clifford.json', '../data/haar_vs_clifford_m11.json'):
    try:
        recs += json.load(open(f))
    except FileNotFoundError:
        pass
recs.sort(key=lambda r: (r['m'], r['k']))
viol, ntot = 0, 0
rows = []
for r in recs:
    m, k, LB, UB, Z = r['m'], r['k'], r['LB'], r['UB'], r['Z']
    SvN, S2, C = np.array(r['H_SvN']), np.array(r['H_S2']), np.array(r['C_S'])
    viol += int((SvN > m - LB + 1e-9).sum() + (C > m - LB + 1e-9).sum()); ntot += len(SvN) + len(C)
    GvN, G2, GC = m - SvN, m - S2, m - C
    pH, pC = 2.0 ** (-S2), 2.0 ** (-C)
    zH, zC = (pH.mean() - Z) / se(pH), (pC.mean() - Z) / se(pC)
    pos = lambda g: (g.mean() - LB) / (UB - LB)
    Dg = SvN - S2
    rows.append(dict(m=m, k=k, nH=len(SvN), nC=len(C), LB=LB, UB=UB, GvN=(GvN.mean(), se(GvN)), G2=(G2.mean(), se(G2)),
                     GC=(GC.mean(), se(GC)), zH=zH, zC=zC, rH=pos(GvN), rH2=pos(G2), rC=pos(GC), D=(Dg.mean(), se(Dg)),
                     dS2=(S2.mean() - C.mean(), np.hypot(se(S2), se(C)))))
    print(f"N={2*m:2d} k={k} | LB={LB} UB={UB:.3f} | Haar vN {GvN.mean():.3f}({se(GvN)*1e3:.0f}) r={pos(GvN):.2f} | Haar R2 {G2.mean():.3f}({se(G2)*1e3:.0f}) r={pos(G2):.2f} | "
          f"Clifford {GC.mean():.3f}({se(GC)*1e3:.0f}) r={pos(GC):.2f} | D=S-S2 {Dg.mean():.3f} (D/m {Dg.mean()/m:.3f}) | E[S2]_H-E[S]_C {S2.mean()-C.mean():+.3f} | z_H={zH:+.2f} z_C={zC:+.2f}")
print(f"Theorem A: {viol} violation(s) in {ntot} circuits (Haar + Clifford)")

print('\nSize dependence of D(m) = E[S] - E[S_2] (Haar), weighted least squares:')
fits = {}
models = {'linear a + b m': lambda m, a, b: a + b * m,
          'logarithmic a + b ln m': lambda m, a, b: a + b * np.log(m),
          'saturating a - b/m': lambda m, a, b: a - b / m}
for k in (1, 2, 4):
    sub = [r for r in rows if r['k'] == k]
    m = np.array([r['m'] for r in sub], float); y = np.array([r['D'][0] for r in sub]); s = np.array([r['D'][1] for r in sub])
    fits[k] = {}
    for name, f in models.items():
        p, cov = curve_fit(f, m, y, sigma=s, absolute_sigma=True)
        chi2 = float((((f(m, *p) - y) / s) ** 2).sum())
        fits[k][name] = dict(params=p.tolist(), errors=np.sqrt(np.diag(cov)).tolist(), chi2=chi2, dof=len(m) - 2)
        print(f"  k={k} {name:24s}: params = {np.round(p, 4).tolist()} ± {np.round(np.sqrt(np.diag(cov)), 4).tolist()} | chi2 = {chi2:.1f} for {len(m) - 2} dof")
json.dump(dict(rows=rows, fits=fits, theorem_A=dict(violations=viol, circuits=ntot)),
          open('../data/haar_vs_clifford_summary.json', 'w'), indent=1)
