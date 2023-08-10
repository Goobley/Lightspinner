from fal import Falc82
from rh_atoms import CaII_atom, H_6_atom
from atomic_set import RadiativeSet
from rh_method import Context
from background import Background
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

atmosConst = Falc82()
atmosConst.quadrature(5)
atmos = atmosConst.convert_scales()
atmos.vlos[:] += np.sin(np.linspace(0, 4 * np.pi, 82)) * 100e3 << (u.m / u.s)

aSet = RadiativeSet([CaII_atom(), H_6_atom()])
aSet.set_active('Ca')
spect = aSet.compute_wavelength_grid()
eqPops = aSet.compute_eq_pops(atmos)
background = Background(atmos, spect)
ctx = Context(atmos, spect, eqPops, background)

dJ = 1.0
dPops = 1.0
i = 0
while dJ > 2e-3 or dPops > 1e-3:
    i += 1
    dJ = ctx.formal_sol_gamma_matrices()

    if i > 3:
        dPops = ctx.stat_equil()
    print('Iteration %.3d: dJ: %.2e, dPops: %s' % (i, dJ, 'Just iterating Jbar' if i < 3 else '%.2e' % dPops))

aSet = RadiativeSet([CaII_atom(), H_6_atom()])
aSet.set_active('Ca')
spect = aSet.compute_wavelength_grid()
eqPops = aSet.compute_eq_pops(atmos)

background = Background(atmos, spect)
ctx_rest = Context(atmos, spect, eqPops, background)

dJ = 1.0
dPops = 1.0
i = 0
while dJ > 2e-3 or dPops > 1e-3:
    i += 1
    dJ = ctx_rest.formal_sol_gamma_matrices_rest()

    if i > 3:
        dPops = ctx_rest.stat_equil()
    print('Iteration %.3d: dJ: %.2e, dPops: %s' % (i, dJ, 'Just iterating Jbar' if i < 3 else '%.2e' % dPops))
    # if i > 4:
    #     break
    # break
ctx_rest.formal_sol_gamma_matrices()



plt.plot(spect.wavelength, ctx.I[:, -1])
plt.plot(spect.wavelength, ctx_rest.I[:, -1], '--')
plt.show()