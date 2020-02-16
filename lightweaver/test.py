from fal import Falc82
from rh_atoms import CaII_atom, H_6_atom
from atomic_set import RadiativeSet
from rh_method import Context
from background import Background

atmosConst = Falc82()
atmosConst.quadrature(5)
atmos = atmosConst.convert_scales()

aSet = RadiativeSet([CaII_atom(), H_6_atom()])
aSet.set_active('Ca', 'H')
spect = aSet.compute_wavelength_grid()
eqPops = aSet.compute_eq_pops(atmos)
# NOTE(cmo): Doesn't fix -- line cores are too deep
# eqPops['H'].nStar[:] = atmosConst.hydrogenPops.value

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



