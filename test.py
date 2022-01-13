from fal import Falc82
from rh_atoms import CaII_atom, H_6_atom, Fe_simple_atom
from atomic_set import RadiativeSet, lte_pops
from rh_method import Context
from background import Background
import matplotlib.pyplot as plt
import numpy as np

atmosConst = Falc82()
atmosConst.quadrature(5)
atmos = atmosConst.convert_scales()

aSet = RadiativeSet([H_6_atom(), CaII_atom()])
aSet.set_active('H')
aSet.set_active('Ca')
spect = aSet.compute_wavelength_grid()
# eqPops = aSet.compute_eq_pops(atmos)
eqPops = aSet.iterate_lte_ne_eq_pops(atmos)

background = Background(atmos, spect)
ctx = Context(atmos, spect, eqPops, background)

dJ = 1.0
dPops = 1.0
dPopsNr = 1.0
i = 0

# while dJ > 2e-3 or dPops > 1e-3:
#     i += 1
#     dJ = ctx.formal_sol_gamma_matrices(fixCol=False)

#     if i > 3:
#         dPops = ctx.stat_equil()
#         # dPopsNr = ctx.nr_post_update()

#         # print(dPops)
#     ctx.update_J()
#     print('Iteration %.3d: dJ: %.2e, dPops: %s' % (i, dJ, 'Just iterating Jbar' if i <= 3 else '%.2e (%.2e)' % (dPops, dPopsNr)))
#     # if i > 6:
#     #     break

GaussSeidel = True
NrPostIter = True
sorOmega = 1.0
dPopsHist = []

while dJ > 2e-2 or dPops > 1e-3:
    i += 1
    if (not GaussSeidel) or i < 3:
        dJ = ctx.formal_sol_gamma_matrices()
    else:
        dJ, dPops = ctx.formal_sol_gauss_seidel(sorOmega=sorOmega)

    if i >= 3 and (not GaussSeidel):
        dPops = ctx.stat_equil()
    if i >= 3 and NrPostIter:
        dPopsNr = ctx.nr_post_update(sorOmega=sorOmega)
        for p in eqPops:
            p.nStar[:] = lte_pops(p.model, atmos, p.nTotal)
    dPopsHist.append(dPops)

    if GaussSeidel and i >= 3 and dPopsHist[-1] < 1e-1:
        delta = dPopsHist[-1] / dPopsHist[-2]
        sorOmega = 2.0 / (1.0 + np.sqrt(1.0 - delta))

        # print(dPops)
    ctx.update_J()
    print('Iteration %.3d: dJ: %.2e, dPops: %s' % (i, dJ, 'Just iterating Jbar' if i < 3 else '%.2e (%.2e)' % (dPops, dPopsNr)))
    if GaussSeidel:
        print(f'sorOmega: {sorOmega:.2f}')
    # if i > 6:
    #     break

refI = np.load('Imu.npy')
plt.plot(spect.wavelength, ctx.I[:, -1])
plt.plot(spect.wavelength, refI[:, -1], '--')
plt.show()