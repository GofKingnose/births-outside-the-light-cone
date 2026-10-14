#!/usr/bin/env python3
"""Tables and figures of the campaigns "free field on a growing FRW lattice" (Section 9 of the paper).
Inputs: field_scan.pkl, field_main.pkl, mera_depth.json, mi_exact_vacuum.json. Outputs: field_results.txt and
fig_field_energy.pdf, fig_field_correlations.pdf, fig_field_gap.pdf, fig_field_entanglement.pdf."""
import pickle, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9, 'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
                     'axes.linewidth': 0.7, 'legend.fontsize': 7.5, 'legend.frameon': False, 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})
B = pickle.load(open('field_scan.pkl', 'rb')); P = pickle.load(open('field_main.pkl', 'rb'))
MD = json.load(open('mera_depth.json'))
COL = {'prod': '#b5462d', 'mera2': '#d9a02b', 'mera3': '#7a9a3a', 'vacuum': '#2a5d8f'}
LAB = {'prod': 'product (R1)', 'mera2': 'MERA, D=2 (R3)', 'mera3': 'MERA, D=3 (R3)', 'vacuum': 'vacuum-preserving (R2)'}
key = lambda s: s['rule'] + (str(s['D']) if s['rule'] == 'mera' else '')
lines = []
def say(x=''):
    print(x); lines.append(x)

# ------------------------------------------------------------------ 1. energy and quanta injected per birth
say('=== 1. Scan in H (de Sitter, m = 0.05, N: 32 -> 512) ===')
say(f"{'H':>7} {'rule':>8} {'eps/Lambda per newborn':>24} {'quanta/newborn':>16} {'expansion quanta/site/doubling':>32}")
scan = {}
for r in sorted(B, key=lambda r: (r['spec']['H0'], key(r['spec']))):
    s = r['spec']; k = key(s)
    eps = np.mean([b['eps'] for b in r['births']]); qn = np.mean([b['quanta_per_newborn'] for b in r['births']])
    qe = np.mean([x['quanta_per_site'] for x in r['intervals']])
    scan.setdefault(k, []).append((s['H0'], eps, qn, qe))
    say(f"{s['H0']:7.4f} {k:>8} {eps:24.3e} {qn:16.3e} {qe:32.3e}")
for k in scan:
    a = np.array(scan[k]); sel = a[:, 0] <= 0.02
    if k == 'vacuum':
        g = np.polyfit(np.log(a[sel, 0]), np.log(np.abs(a[sel, 1])), 1)[0]; say(f"log-log slope of eps_vacuum (H <= 0.02): {g:.2f}")
    g2 = np.polyfit(np.log(a[sel, 0]), np.log(a[sel, 3]), 1)[0]; say(f"log-log slope of the production by expansion ({k}, H <= 0.02): {g2:.2f}")
say('MERA, eps(D) (birth from the coarse vacuum): ' + '; '.join(f"m={m}: " + ', '.join(f"D={d}: {e:.2e}" for d, e in v) for m, v in MD.items()))

say('\n=== 2. Main campaign (N: 32 -> 2048) ===')
for r in P:
    s = r['spec']; k = key(s); hf = r['half'][-1]
    eps = np.mean([b['eps'] for b in r['births']]); qn = np.mean([b['quanta_per_newborn'] for b in r['births']])
    qe = [x['quanta_per_site'] for x in r['intervals']]
    lawn = f"a ~ t^{s['p']}" if s['kind'] == 'pl' else 'de Sitter'
    say(f"{lawn:12s} m={s['m']:<5} {k:>6}: eps = {eps:.3e} Lambda | quanta/newborn = {qn:.2e} | expansion quanta/site/doubling (mean) = {np.mean(qe):.2e} "
        f"| final H = {r['intervals'][-1]['H2']:.2e}")
    say('      N, S(A), S_vac(A), S_th(A), G_E/|A|: ' + ' | '.join(f"{h['N']}: {h['S']:.2f}, {h['Svac']:.2f}, {h['Sth']:.2f}, {h['GE'] / (h['N'] // 2):.4f}" for h in r['half']))

# ------------------------------------------------------------------ figure 1: quanta and energy injected
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.75))
ax = axs[0]
for k in ('prod', 'mera2'):
    a = np.array(sorted(scan[k])); ax.loglog(a[:, 0], a[:, 2], 'o-', color=COL[k], ms=3.5, lw=1.1, label='birth ' + LAB[k] + ', per newborn')
a = np.array(sorted(scan['vacuum']))
ax.loglog(a[:, 0], a[:, 3], 's--', color='0.4', ms=3, lw=0.9, label="expansion, per site and per doubling")
HH = np.array([0.002, 0.03]); ax.loglog(HH, 0.6 * HH ** 2, 'k:', lw=0.8); ax.text(0.0035, 2.0e-6, r'$\propto H^2$', fontsize=8)
qmax = max(abs(x[2]) for x in scan['vacuum'])
ax.text(0.011, 6e-6, r'R2: $|\Delta n| \leq %d\times10^{-15}$' % int(np.ceil(qmax * 1e15)) + '\n(exact conservation)', fontsize=6.5, color=COL['vacuum'])
ax.set_xlabel(r'$H/\Lambda$ (de Sitter)'); ax.set_ylabel('instantaneous quanta created')
ax.set_title('(a) quanta injected', fontsize=8.5)
ax = axs[1]
for k in ('prod', 'mera2', 'vacuum'):
    a = np.array(sorted(scan[k])); ax.loglog(a[:, 0], np.abs(a[:, 1]), 'o-', color=COL[k], ms=3.5, lw=1.1, label=LAB[k])
ax.loglog(HH, 4e-4 * HH ** 2, 'k:', lw=0.8); ax.text(0.004, 1.5e-9, r'$\propto H^2$', fontsize=8)
ax.set_xlabel(r'$H/\Lambda$ (de Sitter)'); ax.set_ylabel(r'$\varepsilon/\Lambda$ per newborn')
ax.set_title('(b) energy injected at birth', fontsize=8.5)
ax = axs[2]
for m, mk in (('0.05', 'o-'), ('0.01', 's--')):
    v = np.array(MD[m]); ax.semilogy(v[:, 0], v[:, 1], mk, color='#7a9a3a', ms=4, lw=1, label=f'MERA, $m={m}$')
ax.axhline(np.mean([abs(x[1]) for x in scan['vacuum'] if x[0] == 0.05]), color=COL['vacuum'], lw=0.9, ls=':', label='R2, $H=0.05$')
ax.set_xlabel('depth $D$ (0 = product)'); ax.set_ylabel(r'$\varepsilon/\Lambda$ per newborn')
ax.set_xticks(range(5)); ax.set_title('(c) optimised MERA layer', fontsize=8.5); ax.legend(fontsize=6.5, loc='upper right')
h, l = axs[0].get_legend_handles_labels(); h2, l2 = axs[1].get_legend_handles_labels()
fig.legend(h + h2[2:], l + ['birth ' + LAB['vacuum'] + ' (energy only)'], loc='lower center', ncol=2, fontsize=6.8)
plt.tight_layout(rect=(0, 0.13, 1, 1)); plt.savefig('fig_field_energy.pdf'); plt.close()

# ------------------------------------------------------------------ figure 2: correlations at birth and locality
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9))
ax = axs[0]
for r in P:
    s = r['spec']
    if not (s['kind'] == 'dS' and s['m'] == 0.05) or r['mi'] is None: continue
    k = key(s); mi = r['mi']
    if k == 'prod':
        say(f"\nMI newborn/existing, product rule: max = {max(abs(x) for x in mi['mi_c']):.1e} (zero to machine precision)")
        continue
    rr = np.array(sorted(int(x) for x in mi['mi_far'])); vv = np.array([mi['mi_far'][x] for x in rr])
    ax.loglog(rr, np.maximum(vv, 1e-16), 'o-', color=COL[k], ms=3.5, lw=1.1, label=LAB[k] + r': $I(w_{i}:\{c_j\}_{|j-i|\geq r})$')
    say(f"MI(newborn : existing at distance >= r), {k}: " + ', '.join(f"r={a}: {b:.2e}" for a, b in zip(rr, vv)) + f" | S(newborn) = {mi['S_newborn']:.4f} | S(all newborns)/N = {mi['S_all_newborns_per']:.4f}")
EX = json.load(open('mi_exact_vacuum.json'))['0.05']
ax.loglog(EX['r'][:7], EX['mi'][:7], 'o:', color=COL['vacuum'], mfc='none', ms=3.5, lw=0.9, label='vacuum-preserving, from the exact vacuum')
say(f"MI from the exact vacuum (m = 0.05): " + ', '.join(f"r={a}: {b:.2e}" for a, b in zip(EX['r'], EX['mi'])) + f" | tail length {EX['tail_length_fine']:.1f} fine sites (1/(2m) = 10)")
ax.set_xlabel(r'distance $r$ (coarse cells)'); ax.set_ylabel('mutual information (nats)')
ax.set_title('correlations of a newborn with existing cells ($N_c=512$)', fontsize=8.5); ax.legend(fontsize=6.3)
ax = axs[1]
for r in P:
    if r['locality'] is not None and r['spec']['kind'] == 'dS' and r['spec']['m'] == 0.05:
        L = r['locality']; d = np.array(L['dist']); v = np.array(L['maxabs'])
        ax.semilogy(d[d <= 120], v[d <= 120], '-', color=COL['vacuum'], lw=1.2, label=r'$\max|M_{ij}|$, $m=0.05$')
        sel = (d >= 5) & (d <= 60); sl = np.polyfit(d[sel], np.log(v[sel]), 1)[0]
        say(f"Locality of the vacuum-preserving isometry (m = 0.05): exponential decay, length {-1 / sl:.1f} coarse cells (= {-2 / sl:.1f} fine sites; 1/m = 20 sites)")
ax.set_xlabel('distance (coarse cells)'); ax.set_ylabel(r'$\max |M|$ at this distance')
ax.set_title('range of the vacuum-preserving isometry', fontsize=8.5); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig('fig_field_correlations.pdf'); plt.close()

# ------------------------------------------------------------------ figure 3: gap to the thermal entropy at equal energy
# Homogeneous points: end of each expansion interval (ell = 2), just BEFORE the birth. The last logged point
# ("final", just after the last birth, ell = 1) is not comparable and is discarded.
from matplotlib.ticker import NullFormatter, FixedLocator
pre = lambda r: [h for h in r['half'] if not h.get('final')]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.8))
for ax, (kind, p, title) in zip(axs, (('pl', 0.5, r'$a\propto t^{1/2}$ (decelerated)'), ('pl', 3.0, r'$a\propto t^{3}$ (accelerated)'), ('dS', None, r'de Sitter, $H=0.05$'))):
    for r in P:
        s = r['spec']
        if s['kind'] != kind or (p is not None and s['p'] != p): continue
        k = key(s); hs = pre(r)
        N = np.array([h['N'] for h in hs]); g = np.array([h['GE'] / (h['N'] // 2) for h in hs])
        ls = '-' if s['m'] == 0.05 else '--'
        ax.loglog(N, np.maximum(g, 1e-6), 'o' + ls, color=COL[k], ms=3, lw=1, mfc=COL[k] if s['m'] == 0.05 else 'none',
                  label=LAB[k] + ('' if s['m'] == 0.05 else ', $m=0.01$'))
    ax.set_title(title, fontsize=8.5); ax.set_xlabel('$N$ (sites, before birth)')
    ax.xaxis.set_major_locator(FixedLocator([128, 256, 512, 1024])); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticklabels(['128', '256', '512', '1024']); ax.set_ylim(5e-5, 0.6)
axs[0].set_ylabel(r'$G_E/|A|$ (nats per site)')
h, l = axs[2].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=3, fontsize=6.8)
plt.tight_layout(rect=(0, 0.13, 1, 1)); plt.savefig('fig_field_gap.pdf'); plt.close()

# ------------------------------------------------------------------ figure 4: entanglement of one half (R1) and gap (R2) vs instantaneous H
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8))
ax = axs[0]
STY = {('pl', 0.5): ('#b5462d', '-', r'R1, $a\propto t^{1/2}$'), ('pl', 3.0): ('#b5462d', '--', r'R1, $a\propto t^{3}$'), ('dS', None): ('#b5462d', ':', 'R1, de Sitter')}
say('\n=== 3. Entropy of one half S(A), points before birth ===')
for r in P:
    s = r['spec']
    if s['m'] != 0.05 or s['rule'] not in ('prod', 'vacuum'): continue
    hs = pre(r); N = np.array([h['N'] for h in hs]); S = np.array([h['S'] for h in hs])
    say(f"{s['rule']:6s} {s['kind']} p={s['p']}: " + ', '.join(f"N={n}: S={x:.2f} (S/|A| = {x / (n // 2):.3f})" for n, x in zip(N, S))
        + ' | interval durations: ' + ', '.join(f"{x['t2'] - x['t1']:.1f}" for x in r['intervals']))
    if s['rule'] == 'prod':
        c, ls, lab = STY[(s['kind'], s['p'])]; ax.loglog(N, S, 'o' + ls, color=c, ms=3, lw=1, label=lab)
    elif s['kind'] == 'dS':
        ax.loglog(N, S, 'o-', color=COL['vacuum'], ms=3, lw=1, label='R2, all laws (here de Sitter)')
ax.loglog([128, 1024], [6.08, 6.08 * 8], color='0.6', lw=0.6); ax.text(300, 22, r'$\propto N$', fontsize=7.5, color='0.4')
ax.set_xlabel('$N$'); ax.set_ylabel(r'$S(A)$, $|A|=N/2$ (nats)'); ax.set_title('(a) entanglement of one half', fontsize=8.5)
ax.xaxis.set_major_locator(FixedLocator([128, 256, 512, 1024])); ax.xaxis.set_minor_formatter(NullFormatter()); ax.set_xticklabels(['128', '256', '512', '1024'])
ax.legend(fontsize=6.5)
ax = axs[1]
say('\n=== 4. Gap G_E/|A| of rule R2 as a function of the instantaneous H ===')
mk = {('pl', 0.5): ('v', r'$a\propto t^{1/2}$'), ('pl', 3.0): ('^', r'$a\propto t^{3}$'), ('dS', None): ('o', r'de Sitter, $N\leq 1024$')}
for r in P:
    s = r['spec']
    if s['rule'] != 'vacuum' or s['m'] != 0.05: continue
    hs = pre(r); m_, lab = mk[(s['kind'], s['p'])]
    ax.loglog([h['H'] for h in hs], [max(h['GE'] / (h['N'] // 2), 1e-6) for h in hs], m_, color=COL['vacuum'], ms=4, mfc='none', label=lab)
    say(f"{s['kind']} p={s['p']}: " + ', '.join(f"(N={h['N']}, H={h['H']:.2e}) -> {h['GE'] / (h['N'] // 2):.2e}" for h in hs))
sc = [(r['spec']['H0'], pre(r)[-1]) for r in B if r['spec']['rule'] == 'vacuum']
sc.sort(key=lambda x: x[0])
ax.loglog([x[0] for x in sc], [max(x[1]['GE'] / (x[1]['N'] // 2), 1e-6) for x in sc], 's-', color='0.45', ms=3, lw=0.8, label=r'de Sitter scan ($N=256$)')
say('de Sitter scan (N = 256, before birth): ' + ', '.join(f"H={h:.4f} -> {x['GE'] / (x['N'] // 2):.2e}" for h, x in sc))
ax.set_xlabel(r'instantaneous $H/\Lambda$'); ax.set_ylabel(r'$G_E/|A|$ (nats per site), rule R2'); ax.set_title('(b) gap of rule R2', fontsize=8.5)
ax.legend(fontsize=6.5)
plt.tight_layout(); plt.savefig('fig_field_entanglement.pdf'); plt.close()
open('field_results.txt', 'w').write('\n'.join(lines) + '\n')
print('figures OK')
