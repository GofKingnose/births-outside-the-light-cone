#!/usr/bin/env python3
"""Exact check (cone only, no quantum state) of the remark on LOCAL correlated births: in edge growth, if each
newborn undergoes at birth a gate with the site r positions to its left (worst case of a correlation of range r),
|Ā ∩ Inf(A)| increases by at most r/(k+1) + 2 with respect to the uncorrelated case (asserted), and the
worst case over m <= 200 and N0 in {2, 4, 10} is exactly ceil(r/(k+1)) (printed)."""
import sys, math
sys.path.insert(0, '.')
from domain_wall import events

def cone_count(N0, ev, m, r):
    n = N0 + sum(1 for e in ev if e < 0)
    b = m - 1; inside = 0
    for e in reversed(ev):
        if e < 0:
            x = n - 1                                    # newborn read backwards (at the right edge)
            if r > 0 and x - r <= b:                     # birth gate (x - r, x) inside the cone
                b = x
            if m <= x <= b: inside += 1
            n -= 1; b = min(b, n - 1); continue
        if b + 1 < n and (b - e) % 2 == 0 and b >= e:
            b += 1
    inside += sum(1 for x in range(m, n) if x <= b)
    return inside

worst = {}
for k in range(1, 9):
    for m in range(5, 201, 5):
        for N0 in (2, 4, 10):
            ev = events(N0, 2 * m, k)
            base = cone_count(N0, ev, m, 0)
            for r in range(1, 11):
                d = cone_count(N0, ev, m, r) - base
                worst[(k, r)] = max(worst.get((k, r), -99), d)
                assert d <= r / (k + 1) + 2, (k, m, N0, r, d)
exact = all(d == math.ceil(r / (k + 1)) for (k, r), d in worst.items())
print('maximal increase of |Ā ∩ Inf(A)|, k = 1..8 (rows), r = 1..10 (columns):')
for k in range(1, 9):
    print(f'  k={k}:', [worst[(k, r)] for r in range(1, 11)])
print('worst case equals ceil(r/(k+1)) for every (k, r):', exact)
