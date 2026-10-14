import numpy as np
from frw_gauss import *
m = 0.05
# 1. T is symplectic
for Nc in (4, 7):
    T = T_matrix(Nc).toarray(); Nf = 2 * Nc
    J = np.block([[np.zeros((Nf, Nf)), np.eye(Nf)], [-np.eye(Nf), np.zeros((Nf, Nf))]])
    print('T symplectic:', np.abs(T @ J @ T.T - J).max() < 1e-12, '| Tinv T = I:', np.abs(Tinv_matrix(Nc).toarray() @ T - np.eye(2 * Nf)).max() < 1e-12)
# 2. stationary vacuum at fixed ell; vacuum energy
N = 32
G = vacuum(N, 1.3, m)
print('E(vacuum) - E_vac:', energy(G, 1.3, m) - e_vac(N, 1.3, m))
U = step_propagator(N, 1.3, m, 7.0)
print('stationary vacuum (fixed ell):', np.abs(apply_U(G, U) - G).max())
# 3. purity and symplecticity of the propagation
nu = symp_eigs(apply_U(G, compose(step_propagator(N, 1.7, m, 3.0), step_propagator(N, 1.1, m, 2.0))))
print('purity after propagation:', np.abs(nu - 0.5).max())
# 4. births from the coarse vacuum (ell=2): excitation energy per newborn, purity, exactness of R2
Nc = 32; Gc = vacuum(Nc, 2.0, m)
for rule in ('prod', 'vacuum'):
    Gf = birth(Gc, rule, m)
    eps = (energy(Gf, 1.0, m) - e_vac(2 * Nc, 1.0, m)) / Nc
    print(f"{rule:6s}: eps = {eps:.6f} | purity {np.abs(symp_eigs(Gf) - 0.5).max():.1e}")
print('R2: coarse vacuum -> fine vacuum exactly:', np.abs(birth(Gc, 'vacuum', m) - vacuum(2 * Nc, 1.0, m)).max())
# 5. eps_prod nearly independent of N
for Nc in (16, 64, 128):
    Gf = birth(vacuum(Nc, 2.0, m), 'prod', m); print('Nc', Nc, 'eps_prod =', (energy(Gf, 1.0, m) - e_vac(2 * Nc, 1.0, m)) / Nc)
# 6. entropy of an interval of the vacuum (log law, c = 1: S ~ (1/3) ln l + const for l << 1/m)
Gv = vacuum(512, 1.0, 0.001)
print('S(vacuum, l=8,16,32,64):', [round(entropy(Gv, np.arange(L)), 3) for L in (8, 16, 32, 64)], '(slope per doubling: ln2/3 = 0.231 in the conformal limit; about 0.19 is measured, reduced by the log-log correction of the light non-compact boson, cf. Casini-Huerta 2005; not quantified here)')
# 7. Proposition (R2 conserves quanta): for ANY coarse Gaussian state, R2 conserves exactly
#    Q = sum_k n_k (instantaneous quanta), because M X0 M^T = X1 makes Tr(Gamma_phi X^{-1}) and Tr(Gamma_P X) invariant.
#    Test on random excited states (random symplectic map applied to the vacuum); R1 does not conserve Q.
from scipy.linalg import expm
rng = np.random.default_rng(1)
for Nc in (16, 48):
    Nf2 = 2 * Nc
    Jc = np.block([[np.zeros((Nc, Nc)), np.eye(Nc)], [-np.eye(Nc), np.zeros((Nc, Nc))]])
    A = rng.normal(size=(Nf2, Nf2)) * 0.15; A = A + A.T
    S = expm(Jc @ A); Gc = S @ vacuum(Nc, 2.0, m) @ S.T
    Q0 = quanta(Gc, 2.0, m)
    Qv = quanta(birth(Gc, 'vacuum', m), 1.0, m); Qp = quanta(birth(Gc, 'prod', m), 1.0, m)
    print(f'Nc={Nc}: Q before = {Q0:.6f} | after R2 = {Qv:.6f} (difference {Qv - Q0:.1e}) | after R1 = {Qp:.6f}')
