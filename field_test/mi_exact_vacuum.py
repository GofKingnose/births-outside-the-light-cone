"""Control: newborn/existing mutual-information profile for a vacuum-preserving birth from the EXACT vacuum
(no excitations created by the expansion), to separate the structure of the vacuum from the effect of expansion."""
import json, numpy as np
from frw_gauss import *
from run_field import mi_profiles
out = {}
for m in (0.05, 0.1):
    Nc = 512
    Gf = birth(vacuum(Nc, 2.0, m), 'vacuum', m)
    mi = mi_profiles(sandwich(Tinv_matrix(Nc), Gf), Nc)
    rr = sorted(int(x) for x in mi['mi_far']); vv = [mi['mi_far'][r] for r in rr]
    ok = [(r, v) for r, v in zip(rr, vv) if v > 1e-11 and r >= 16]
    tail = np.polyfit(np.array([r for r, v in ok]) * 2.0, np.log([v for r, v in ok]), 1)[0]   # in fine sites (beyond r = 16)
    print(f"m={m}: MI(r) = " + ', '.join(f"{r}:{v:.2e}" for r, v in zip(rr, vv)) + f" | tail length = {-1/tail:.1f} fine sites (1/m = {1/m:.0f}, 1/(2m) = {1/(2*m):.0f})")
    out[str(m)] = dict(r=rr, mi=vv, tail_length_fine=-1 / tail)
json.dump(out, open('mi_exact_vacuum.json', 'w'), indent=1)
