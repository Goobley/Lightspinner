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
# atmos.ne.value[:] = eqPops['H'].nStar[-1, :]
# neStart = np.copy(atmos.ne.value)
# atmos.nTotal = atmos.nHTot.value + atmos.ne.value

background = Background(atmos, spect)
ctx = Context(atmos, spect, eqPops, background)

dJ = 1.0
dPops = 1.0
dPopsNr = 1.0
i = 0
# ctx.fdjac()
# ctx.F()
# atom = ctx.activeAtoms[0]
# Fs = [np.copy(atom.F)]
# L2 = [np.mean(Fs[-1]**2)]
# BroyI = np.zeros_like(atom.Broy)
# for k in range(atmos.Nspace):
#     BroyI[:, :, k] = np.linalg.inv(atom.Broy[:, :, k])

while dJ > 2e-3 or dPops > 1e-3:
    i += 1
    dJ = ctx.formal_sol_gamma_matrices(fixCol=False)

    if i > 3:
        dPops = ctx.stat_equil()
        # print(dPops)
        # dJ = ctx.formal_sol_gamma_matrices(fixCol=True)
        dPopsNr = ctx.nr_post_update()
        for p in eqPops:
            p.nStar[:] = lte_pops(p.model, atmos, p.nTotal)
            
        # print(dPops)
    ctx.update_J()
    print('Iteration %.3d: dJ: %.2e, dPops: %s' % (i, dJ, 'Just iterating Jbar' if i <= 3 else '%.2e (%.2e)' % (dPops, dPopsNr)))
    if i > 3: break
    # if i > 6:
    #     break

# refI = np.load('Imu.npy')
# plt.plot(spect.wavelength, ctx.I[:, -1])
# # plt.plot(spect.wavelength, refI[:, -1], '--')
# plt.show()