#!/usr/bin/env python3
"""Figures of the paper, from the stored data in ../data/. Output: ../paper/figures/
   fig_cone_geometry.pdf (Fig. 1), fig_edge.pdf (Fig. 3), fig_alpha.pdf (Fig. 4),
   fig_dichotomy.pdf (Fig. 5), fig_scaling.pdf (Fig. 6).
Standard errors use the Bessel correction (ddof = 1)."""
import json, os, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
from scipy.optimize import brentq
from continuum_geometry import f_d
plt.rcParams.update({'font.size': 9, 'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
                     'axes.linewidth': 0.7, 'legend.fontsize': 7.5, 'legend.frameon': False, 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})
D = '../data/'; OUT = '../paper/figures/'; os.makedirs(OUT, exist_ok=True)
q = 2; K = q / (q * q + 1)
theta = lambda k: brentq(lambda t: np.log(2 * K * np.cosh(t)) + t / k, 1e-12, np.log(q) - 1e-12)   # nats
sem = lambda x: np.std(x, ddof=1) / np.sqrt(len(x))

# ---------------------------------------------------------------- Fig. 1: geometry of the cone
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw=dict(width_ratios=[1, 1.15]))
ax = axs[0]
T, v = 1.0, 0.7                                       # linear front R(t) = t (s = 1), cone of speed v
x = np.linspace(0, 1, 400); tb = x                    # birth when the front passes
ax.fill_between(x, tb, T, color='0.93', lw=0)
ax.plot(x, tb, 'k-', lw=1.3, label='births (front)')
tt = np.linspace(0, T, 50); ax.plot(v * (T - tt), tt, 'C3--', lw=1.1, label=r'edge of the causal past of $A$')
un = x > v * (T - tb)
ax.plot(x[un], tb[un], color='C0', lw=5, alpha=0.5, solid_capstyle='butt', label='sites never reached')
ax.axvline(0, color='k', lw=0.8); ax.text(-0.03, 1.01 * T, 'cut', ha='left', va='bottom', fontsize=7.5)
ax.set_xlim(-0.05, 1.12); ax.set_ylim(0, 1.1); ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel(r'distance from the cut, on the $\bar A$ side'); ax.set_ylabel(r'time (final time $T$)')
ax.legend(loc='lower right', fontsize=6.5); ax.set_title(r'(a) geometry of the cone ($\alpha=1$, 1D)', fontsize=8.5)
ax = axs[1]
def frac(alpha, T, v=1.0):
    R = T ** alpha; xs = np.linspace(0, R, 20001)[1:]
    return np.mean(xs > v * (T - xs ** (1 / alpha)))
Ts = np.geomspace(2, 2e4, 60)
for alpha, c in ((0.5, 'C3'), (0.75, 'C1'), (1.0, 'C2'), (1.5, 'C0'), (2.0, 'C4')):
    N = Ts ** alpha; ax.plot(N, [frac(alpha, T) for T in Ts], color=c, lw=1.3, label=rf'$\alpha={alpha:g}$')
ax.set_xscale('log'); ax.set_xlabel(r'$N \propto R = t^\alpha$ (1D, $v=1$)'); ax.set_ylabel('fraction never reached')
ax.set_ylim(-0.02, 1.02); ax.set_xlim(2, 1e5); ax.legend(fontsize=7, loc='lower right')
ax.set_title(r'(b) $\alpha<1$: $\to0$;  $\alpha=1$: constant;  $\alpha>1$: $\to1$', fontsize=8.5)
plt.tight_layout(); plt.savefig(OUT + 'fig_cone_geometry.pdf'); plt.close()

# ---------------------------------------------------------------- Fig. 3: edge growth, bracket
P45 = json.load(open(D + 'edge_growth_quenched.json')); EB = json.load(open(D + 'edge_growth_large_k.json'))
pts = []
for k in (1, 2, 4, 8):
    S = np.array([r['S'] for r in P45 if r.get('kind') == 'edge' and r['k'] == k and r['m'] == 1024], float) / 1024
    G = 0.5 * (1 - S); pts.append((k, G.mean(), sem(G)))
for k in (16, 32):
    S = np.array([r['S'] for r in EB if r['k'] == k and r['m'] == 512], float) / 512
    G = 0.5 * (1 - S); pts.append((k, G.mean(), max(sem(G), 1e-5)))
kk = np.exp(np.linspace(0, np.log(40), 300))
lo = 1 / (2 * (1 + kk)); hi = np.array([0.5 * (1 - theta(k) / np.log(2)) for k in kk])
mem = 0.5 * np.maximum(1 - kk * 0.325, 1 / (1 + kk * 0.6))
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9))
for ax, scale in zip(axs, ('lin', 'log')):
    ax.fill_between(kk, lo, hi, color='#d9e6f2', lw=0, label='proven bracket')
    ax.plot(kk, lo, color='#2a5d8f', lw=1.1, label=r'Theorem A: $1/(2(1+k))$')
    ax.plot(kk, hi, color='#b5462d', lw=1.1, label=r'Theorem E: $\frac{1}{2}(1-\theta_k/\ln 2)$')
    ax.plot(kk, mem, color='0.45', lw=0.9, ls='--', label='two-path membrane estimate (not a bound)')
    k_, g_, e_ = np.array(pts).T
    ax.errorbar(k_, g_, yerr=e_, fmt='o', color='k', ms=3.5, capsize=2, label='uniform Clifford (2-design)')
    ax.set_xscale('log', base=2); ax.set_xticks([1, 2, 4, 8, 16, 32]); ax.set_xticklabels(['1', '2', '4', '8', '16', '32'])
    ax.set_xlabel(r'$k$ (layers per birth)')
    if scale == 'log':
        ax.set_yscale('log'); ax.set_ylim(0.012, 0.45)
        ax.yaxis.set_minor_formatter(NullFormatter()); ax.set_yticks([0.02, 0.05, 0.1, 0.2, 0.4]); ax.set_yticklabels(['0.02', '0.05', '0.1', '0.2', '0.4'])
    else:
        ax.set_ylabel(r'$G/(N\log 2)$'); ax.legend(fontsize=6.4, loc='upper right')
axs[0].set_title(r'edge growth, $\alpha=1$ (linear scale)', fontsize=8.5)
axs[1].set_title('same data, logarithmic scale', fontsize=8.5)
plt.tight_layout(); plt.savefig(OUT + 'fig_edge.pdf'); plt.close()

# ---------------------------------------------------------------- Fig. 4: alpha < 1
P = json.load(open(D + 'alpha_annealed.json')); PT = json.load(open(D + 'alpha_quenched.json'))
fig, ax = plt.subplots(figsize=(5.6, 3.8))
cols = {0.5: '#7a7a7a', 0.6: '#1b9e77', 0.75: '#d95f02', 0.9: '#7570b3', 1.0: '#000000'}
for a in (0.5, 0.6, 0.75, 0.9, 1.0):
    rr = [r for r in P if r['alpha'] == a]
    m = np.array([r['m'] for r in rr]); U = np.array([r['U'] for r in rr]); LA = np.array([r['LA'] for r in rr])
    ax.plot(m, U, '-', color=cols[a], lw=1.4)
    sel = LA > 0
    ax.plot(m[sel], LA[sel], ':', color=cols[a], lw=1.2)
    tt = [r for r in PT if r['alpha'] == a]
    for mm in sorted(set(r['m'] for r in tt)):
        G = np.array([r['G'] for r in tt if r['m'] == mm], float)
        ax.errorbar([mm], [G.mean()], yerr=[sem(G)], fmt='o', color=cols[a], ms=3.5, capsize=2)
    if 0.5 < a < 1:
        xx = np.array([64, 512]); ax.plot(xx, 1.6 * U[-1] * (xx / 512) ** (2 - 1 / a), '--', color=cols[a], lw=0.7, alpha=0.7)
ax.set_xscale('log', base=2); ax.set_yscale('log')
ax.set_xlabel(r'$m=|A|$ ($N=2m$)'); ax.set_ylabel('bits')
ax.text(17, 330, 'solid: $m-S_2^{\\rm ann}/\\log 2 \\geq \\mathbb{E}[G]$\ndotted: Theorem A bound $\\leq G$\npoints: quenched $\\mathbb{E}[G]$ (Clifford)\ndashed: slope $2-1/\\alpha$', fontsize=7.5, va='top')
h = [Line2D([0], [0], color=cols[a], lw=1.6) for a in (0.5, 0.6, 0.75, 0.9, 1.0)]
ax.legend(h, [fr'$\alpha={a}$' for a in (0.5, 0.6, 0.75, 0.9, 1.0)], fontsize=8, loc='lower right')
fig.tight_layout(); fig.savefig(OUT + 'fig_alpha.pdf'); plt.close()

# ---------------------------------------------------------------- Fig. 5: 1D/2D dichotomy
U = json.load(open(D + 'growth_campaigns.json'))
def arr(d, m, key):
    v = d[m]; Ns = np.array([r['n'] for r in v[0]])
    X = np.array([[min(r['bA'], r['bB']) if key == 'b' else r[key] for r in rec] for rec in v], float)
    return Ns, X.mean(0), X.std(0, ddof=1) / np.sqrt(len(X))
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9))
ax = axs[0]
for m, c, lab in (('front-1', 'C0', r'front $s=1$'), ('front-0.5', 'C1', r'front $s=\frac{1}{2}$'), ('front-0.25', 'C2', r'front $s=\frac{1}{4}$'),
                  ('eden-1', 'C4', r'Eden $r=1$'), ('eden-0.25', 'C5', r'Eden $r=\frac{1}{4}$'),
                  ('rate-16', 'C3', r'rate $\gamma=16$'), ('rate-4', 'C7', r'rate $\gamma=4$')):
    N, G, Gs = arr(U['2d'], m, 'G'); _, L, _ = arr(U['2d'], m, 'G_LB')
    ax.errorbar(N, G / N, yerr=Gs / N, fmt='o-', color=c, ms=3.0, lw=1, label=lab)
    ax.plot(N, L / N, 's:', color=c, ms=2.6, mfc='none', lw=0.8)
    if m.startswith('front'):
        s = float(m.split('-')[1]); ax.axhline(f_d(2 / s, 2) / 2, color=c, lw=0.6, ls='--')
    if m.startswith('rate'):
        g = float(m.split('-')[1]); NN = np.geomspace(100, 6400); ax.plot(NN, g * np.sqrt(NN / np.pi) / (2 * np.pi) / NN, color=c, lw=0.6, ls='--')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'$N$'); ax.set_ylabel(r'$G/(N\log 2)$')
ax.set_title('2D: filled = uniform Clifford, open = Theorem A bound', fontsize=8)
ax.legend(ncol=2, fontsize=6.2, loc='lower left')
ax = axs[1]
for m, c, lab in (('edge-1', 'C0', r'edge, $k=1$'), ('edge-4', 'C1', r'edge, $k=4$'), ('p1', 'C2', r'$N\propto t$ (inserted)'),
                  ('p1pair', 'C8', r'$N\propto t$ (pairs)'), ('p2', 'C4', r'$N\propto t^2$'), ('exp', 'C5', r'$N\propto e^{t/\tau}$'), ('p0.5', 'C3', r'$N\propto t^{1/2}$')):
    N, G, Gs = arr(U['1d'], m, 'G'); _, L, _ = arr(U['1d'], m, 'G_LB')
    ax.errorbar(N, G / N, yerr=Gs / N, fmt='o-', color=c, ms=3.0, lw=1, label=lab)
    ax.plot(N, L / N, 's:', color=c, ms=2.6, mfc='none', lw=0.8)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'$N$'); ax.set_title('1D', fontsize=8)
ax.legend(ncol=2, fontsize=6.2, loc='lower left')
plt.tight_layout(); plt.savefig(OUT + 'fig_dichotomy.pdf'); plt.close()

# ---------------------------------------------------------------- Fig. 6: 2D scaling laws
fig, ax = plt.subplots(figsize=(3.6, 2.8))
for m, c, lab in (('front-0.5', 'C1', r'front $\alpha=1$'), ('acc-0.5', 'C4', r'front $\alpha=2$'),
                  ('acc-0.05', 'C9', r'front $\alpha=2$ (slow)'), ('expo-0.1', 'C5', 'exponential front')):
    N, S, Ss = arr(U['2d'], m, 'S'); _, B, _ = arr(U['2d'], m, 'b')
    ax.plot(N, S, 'o-', color=c, ms=3.0, lw=1, label=lab); ax.plot(N, B, 's:', color=c, ms=2.6, mfc='none', lw=0.8)
NN = np.geomspace(400, 6400)
for p, y0, txt in ((1.0, 300, r'$N^1$'), (0.75, 60, r'$N^{3/4}$'), (0.5, 9, r'$N^{1/2}$')):
    ax.plot(NN, y0 * (NN / 400) ** p, 'k-', lw=0.5); ax.text(NN[-1] * 1.05, y0 * 16 ** p, txt, fontsize=7, va='center')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'$N$'); ax.set_ylabel(r'$S(A)$ (bits)')
ax.set_xlim(80, 13000); ax.legend(fontsize=6.3, loc='upper left'); ax.set_title('2D: entropy (filled) and causal bound (open)', fontsize=7.5)
plt.tight_layout(); plt.savefig(OUT + 'fig_scaling.pdf'); plt.close()
print('figures OK; edge-growth points (k, E[G]/(N log 2), s.e.):', [(int(a), round(float(b), 4), round(float(c), 4)) for a, b, c in pts])
