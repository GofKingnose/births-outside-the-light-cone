"""Energy per newborn eps of the optimised Gaussian MERA layer as a function of its depth D (birth from the coarse vacuum)."""
import json, numpy as np
from frw_gauss import *
out = {}
for m in (0.05, 0.01):
    res = []
    for D in (0, 1, 2, 3, 4):
        if D == 0:
            aw, bp, ww, pp = detail_local_state(128, m)
            Gf = sandwich(T_matrix(128), assemble_cw(vacuum(128, 2.0, m), ww, pp))
        else:
            gates, fopt, f0 = optimise_mera(D, m, Nc=32, restarts=3)
            aw, bp, ww, pp = detail_local_state(128, m)
            Gf = mera_birth(sandwich(T_matrix(128), assemble_cw(vacuum(128, 2.0, m), ww, pp)), gates)
        eps = (energy(Gf, 1.0, m) - e_vac(256, 1.0, m)) / 128
        res.append((D, float(eps)))
        print(f"m={m} D={D}: eps = {eps:.3e}", flush=True)
    out[str(m)] = res
json.dump(out, open('mera_depth.json', 'w'), indent=1)
