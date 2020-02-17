from fal import Falc82
from rh_atoms import CaII_atom, H_6_atom
from atomic_set import RadiativeSet
from rh_method import Context
from background import Background
import matplotlib.pyplot as plt

atmosConst = Falc82()
atmosConst.quadrature(5)
atmos = atmosConst.convert_scales()

aSet = RadiativeSet([CaII_atom(), H_6_atom()])
aSet.set_active('H')
spect = aSet.compute_wavelength_grid()
eqPops = aSet.compute_eq_pops(atmos)

background = Background(atmos, spect)
ctx = Context(atmos, spect, eqPops, background)

dJ = 1.0
dPops = 1.0
i = 0
while dJ > 2e-3 or dPops > 1e-3:
    i += 1
    print(i, dJ, dPops)
    dJ = ctx.formal_sol_gamma_matrices()

    if i <= 3:
        continue

    dPops = ctx.stat_equil()

plt.plot(spect.wavelength, ctx.I[:, -1])
plt.show()