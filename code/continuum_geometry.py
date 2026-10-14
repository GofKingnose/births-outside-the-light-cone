#!/usr/bin/env python3
"""Analytic (continuum) predictions of the fraction of Ā outside the strict causal cone."""
import numpy as np
from scipy.integrate import quad
def f_d(lam, d):
    # hemispherical average of 1-(lam/(lam+u))^d, u = cosine of the angle to the normal
    if d == 1: return 1/(1+lam)
    if d == 2:
        return quad(lambda th: 1-(lam/(lam+np.cos(th)))**2, -np.pi/2, np.pi/2)[0]/np.pi
    if d == 3: return 1 - lam*(2*lam+1)/(2*(lam+1)**2)
def f_insert(law, c, N0, NT):
    # homogeneous 1D expansion, v = 1 site/layer, tau = c N0; y = N_b/N_T
    def Delta(Nb):
        if law == 'p0.5': return 2*c*(NT-Nb)/N0
        if law == 'p1':   return c*np.log(NT/Nb)
        if law == 'p2':   return c*(np.sqrt(N0/Nb)-np.sqrt(N0/NT))
        if law == 'exp':  return c*N0*(1/Nb-1/NT)
    tot = N0*max(0, 1-2*Delta(N0))                      # initial sites (born at t = 0)
    tot += quad(lambda Nb: max(0.0, 1-2*Delta(Nb)), N0, NT, limit=400)[0]
    return tot/NT                                         # fraction of Ā never reached
if __name__ == '__main__':
    for lam in (8, 4, 2): print(f"2D front, lambda={lam}: f_2 = {f_d(lam,2):.4f}  -> G_LB/N ~ {f_d(lam,2)/2:.4f}")
    for g in (4, 16):
        for N in (400, 1600):
            R = np.sqrt(N/np.pi); print(f"constant rate gamma={g}, N={N}: G_LB ~ gamma R/(2 pi) = {g*R/(2*np.pi):.1f}  (G_LB/N = {g*R/(2*np.pi)/N:.4f})")
    for law in ('p0.5', 'p1', 'p2', 'exp'):
        print(law, " ".join(f"N={N}: G_LB/N={f_insert(law,0.5,16,N)/2:.4f}" for N in (100, 200, 400, 800, 1e6)))
    print("p=1 asymptotic: f = 1-2c(1-exp(-1/(2c))) =", 1-2*0.5*(1-np.exp(-1)), "-> G_LB/N =", (1-(1-np.exp(-1)))/2)
