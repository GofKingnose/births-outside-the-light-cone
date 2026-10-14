#!/usr/bin/env python3
"""Campaigns: energy injected per birth, production by the expansion, newborn/existing correlations,
locality of the vacuum-preserving isometry, entropy of one half and gap to the thermal entropy at equal energy.
Usage: python3 run_field.py main | scan"""
import sys, json, time, pickle
import numpy as np
from multiprocessing import Pool
from scipy.optimize import brentq
from frw_gauss import *

def thermal_beta(N, ell, m, Eexc):
    Om = omegas(N, ell, m)
    if Eexc <= 1e-12:
        return np.inf
    def f(lb):
        with np.errstate(over='ignore'):
            return (Om / np.expm1(np.minimum(np.exp(lb) * Om, 700))).sum() - Eexc
    return np.exp(brentq(f, np.log(1e-6), np.log(1e6)))

def half_measures(G, ell, m):
    N = G.shape[0] // 2; A = np.arange(N // 2)
    Eexc = energy(G, ell, m) - e_vac(N, ell, m)
    S = entropy(G, A)
    beta = thermal_beta(N, ell, m, Eexc)
    Sth = entropy(thermal(N, ell, m, beta), A)
    Svac = entropy(vacuum(N, ell, m), A)
    return dict(N=N, Eexc=Eexc, S=S, Sth=Sth, Svac=Svac, GE=Sth - S, T=(1 / beta if np.isfinite(beta) else 0.0))

def mi_profiles(Gcw, Nc, rs=(1, 2, 4, 8, 16, 32, 64, 128)):
    """Correlations between the central newborn w_i0 and the existing modes c_j, just after birth."""
    i0 = Nc // 2; wi = Nc + i0
    js = np.arange(Nc)
    dist = np.minimum(np.abs(js - i0), Nc - np.abs(js - i0))
    sel = js[np.argsort(dist)][:min(Nc, 260)]           # the 260 nearest suffice for the profile
    mi_c = two_mode_MI(Gcw, wi, sel)
    mi_w = two_mode_MI(Gcw, wi, Nc + sel[1:])
    prof = {}
    for r in rs:
        far = js[dist >= r]
        if len(far) == 0: continue
        Sw = single_mode_S(Gcw, wi); Sf = entropy(Gcw, far); Sj = entropy(Gcw, np.r_[wi, far])
        prof[r] = Sw + Sf - Sj
    S_w = single_mode_S(Gcw, wi)
    S_W = entropy(Gcw, np.arange(Nc, 2 * Nc))            # total entanglement newborns / existing (pure state)
    return dict(dist_c=dist[sel].tolist(), mi_c=mi_c.tolist(), dist_w=dist[sel[1:]].tolist(), mi_w=mi_w.tolist(),
                mi_far=prof, S_newborn=S_w, S_all_newborns_per=S_W / Nc)

def run(spec):
    kind, p, H0, m, rule, D, N0, nd, measure_at = spec
    t_start = time.time()
    t0, a, Hf, tdouble = law(kind, H0, p)
    gates = None
    if rule == 'mera':
        gates, fopt, f0 = optimise_mera(D, m, Nc=32, restarts=2)
    G = vacuum(N0, 1.0, m); N = N0; t = t0
    log = dict(spec=dict(kind=kind, p=p, H0=H0, m=m, rule=rule, D=D, N0=N0, nd=nd), intervals=[], births=[], half=[], mi=None, locality=None)
    for d in range(nd):
        E_start = energy(G, 1.0, m) - e_vac(N, 1.0, m); Q_start = quanta(G, 1.0, m)
        t2 = tdouble(t)
        G, nst = evolve_interval(G, N, m, a, Hf, t, t2)
        E_end = energy(G, 2.0, m) - e_vac(N, 2.0, m); Q_end = quanta(G, 2.0, m)
        log['intervals'].append(dict(N=N, t1=t, t2=t2, H1=Hf(t), H2=Hf(t2), steps=nst, Eexc_start=E_start, Eexc_end=E_end,
                                     prod_per_site=(E_end - E_start) / N, Q_start=Q_start, Q_end=Q_end, quanta_per_site=(Q_end - Q_start) / N))
        if N >= 128:
            hm = half_measures(G, 2.0, m); hm['t'] = t2; hm['H'] = Hf(t2); log['half'].append(hm)
        t = t2
        want_cw = (N == measure_at)
        out = birth(G, rule, m, gates=gates, return_cw=want_cw) if rule != 'vacuum' else None
        if rule == 'vacuum':
            Gf = birth(G, 'vacuum', m)
            Gcw = sandwich(Tinv_matrix(N), Gf) if want_cw else None
        else:
            Gf, Gcw = (out if want_cw else (out, None))
        E_after = energy(Gf, 1.0, m) - e_vac(2 * N, 1.0, m); Q_after = quanta(Gf, 1.0, m)
        log['births'].append(dict(Nc=N, t=t, H=Hf(t), Eexc_before=E_end, Eexc_after=E_after, eps=(E_after - E_end) / N,
                                  Q_before=Q_end, Q_after=Q_after, quanta_per_newborn=(Q_after - Q_end) / N))
        if want_cw:
            log['mi'] = mi_profiles(Gcw, N)
            if rule == 'vacuum':
                M, Q, ww, pp = vacuum_preserving_map(N, m)
                i0 = N // 2; dist = np.arange(0, N // 2)
                row_c = np.abs(M[i0, :N]); row_w = np.abs(M[N + i0, N:])
                dc = np.minimum(np.abs(np.arange(N) - i0), N - np.abs(np.arange(N) - i0))
                prof = [float(max(np.abs(M[i0, :N])[dc == r].max(), np.abs(M[i0, N:])[dc == r].max(),
                                  np.abs(M[N + i0, :N])[dc == r].max(), np.abs(M[N + i0, N:])[dc == r].max())) for r in range(0, N // 2, 1)]
                log['locality'] = dict(dist=list(range(0, N // 2)), maxabs=prof)
        G = Gf; N = 2 * N
    hm = half_measures(G, 1.0, m); hm['t'] = t; hm['H'] = Hf(t); hm['final'] = True; log['half'].append(hm)
    log['time'] = time.time() - t_start
    return log

if __name__ == '__main__':
    which = sys.argv[1]
    specs = []
    if which == 'main':
        for kind, p in (('pl', 0.5), ('pl', 3.0), ('dS', None)):
            for rule, D in (('prod', 0), ('vacuum', 0), ('mera', 2), ('mera', 3)):
                specs.append((kind, p, 0.05, 0.05, rule, D, 32, 6, 512))
        for rule, D in (('prod', 0), ('vacuum', 0)):                      # light field (m < H) in de Sitter
            specs.append(('dS', None, 0.05, 0.01, rule, D, 32, 6, 512))
    elif which == 'scan':
        for H in (0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
            for rule, D in (('prod', 0), ('vacuum', 0), ('mera', 2)):
                specs.append(('dS', None, H, 0.05, rule, D, 32, 4, -1))
    out = []
    with Pool(2) as P:
        for r in P.imap_unordered(run, specs):
            s = r['spec']; b = r['births']; hf = r['half'][-1]
            print(f"{s['kind']} p={s['p']} H0={s['H0']} m={s['m']} {s['rule']}{s['D'] or ''}: "
                  f"birth eps = {np.mean([x['eps'] for x in b]):.3e} | quanta/newborn = {np.mean([x['quanta_per_newborn'] for x in b]):.3e} | expansion quanta/site/doubling = {np.mean([x['quanta_per_site'] for x in r['intervals']]):.3e} "
                  f"| final N={hf['N']} S={hf['S']:.2f} Sth={hf['Sth']:.2f} GE/|A|={hf['GE']/(hf['N']//2):.4f} [{r['time']:.0f}s]", flush=True)
            out.append(r)
            pickle.dump(out, open(f'field_{which}.pkl', 'wb'))
    print('DONE', flush=True)
