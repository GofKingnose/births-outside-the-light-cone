"""Growth n(L) - N0 = floor(L^alpha) at the right edge: exact annealed bound m - S2rec/log2 against the
Theorem-A lower bound and the Theorem-D upper bound B(t*) + log2(2 + (q^2+1)/(2q)) (= 1.70 bit for q = 2)."""
import numpy as np, json, time, sys
sys.path.insert(0, '.')
from domain_wall import events_general, logZ_events, cone_lower
q = 2; K = q / (q * q + 1); E0 = np.log((q * q + 1) / (2 * q))
cq_bits = np.log2(2 + 1 / (2 * K))
out = []
for alpha in (0.4, 0.5, 0.6, 0.75, 0.9, 1.0):
    for m in (16, 32, 64, 128, 256, 512):
        N0 = 2; nb = 2 * m - N0
        bl = [int(np.ceil(i ** (1 / alpha))) for i in range(1, nb + 1)]      # n(L) - N0 = floor(L^alpha)
        nxt = int(np.ceil((nb + 1) ** (1 / alpha)))
        T = bl[-1] + (nxt - bl[-1]) // 2 + 1
        if T > 3e6: continue
        t0 = time.time()
        ev = events_general(N0, bl, T)
        U = m - (-logZ_events(q, N0, ev, m) / np.log(q))                      # m - S2/log q (bits)
        LA = cone_lower(N0, ev, m)
        tstar = int(np.ceil(m * np.log(q) / E0))
        B = sum(1 for L in bl if L > T - tstar)
        ok = (LA <= U + 1e-9) and (U <= B + cq_bits + 1e-9)
        out.append(dict(alpha=alpha, m=m, T=T, U=U, LA=LA, B=B, tstar=tstar))
        print(f"alpha={alpha:4.2f} m={m:4d} T={T:8d}: Theorem A {LA:5d} <= m - S2/log2 = {U:9.3f} <= B(t*) + {cq_bits:.2f} = {B + cq_bits:9.2f}  [{'OK' if ok else 'FAIL'}]  U/m = {U/m:.4f}  ({time.time()-t0:.0f}s)", flush=True)
json.dump(out, open('../data/alpha_annealed.json', 'w'), indent=1)
