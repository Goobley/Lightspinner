from fal import Falc82
from rh_atoms import CaII_atom, H_6_atom
from atomic_set import RadiativeSet
from rh_method import Context, tm_prod, make_block_tridiag
from background import Background
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve

atmosConst = Falc82()
atmosConst.quadrature(5)
atmos = atmosConst.convert_scales()

aSet = RadiativeSet([CaII_atom(), H_6_atom()])
aSet.set_active('Ca')
spect = aSet.compute_wavelength_grid()
eqPops = aSet.compute_eq_pops(atmos)

background = Background(atmos, spect)
ctx = Context(atmos, spect, eqPops, background)

dJ = 1.0
dPops = 1.0
i = 0
# while dJ > 2e-3 or dPops > 1e-3:
i += 1
dJ = ctx.formal_sol_gamma_matrices()
td, sol = make_block_tridiag(ctx.activeAtoms[0].Gamma, ctx.activeAtoms[0].n)
s = solve(td, sol)
s = s.reshape(82, 6)

#     if i > 3:
#         dPops = ctx.stat_equil()
#     print('Iteration %.3d: dJ: %.2e, dPops: %s' % (i, dJ, 'Just iterating Jbar' if i < 3 else '%.2e' % dPops))

# plt.plot(spect.wavelength, ctx.I[:, -1])
# plt.show()