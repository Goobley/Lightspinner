from fal import Falc82
from rh_atoms import CaII_atom, H_6_atom
from atomic_set import RadiativeSet
from rh_method import Context
from background import Background

atmos = Falc82()
atmos.quadrature(5)
atmos.convert_scales()

aSet = RadiativeSet([CaII_atom(), H_6_atom()])
aSet.set_active('Ca')
spect = aSet.compute_wavelength_grid()
eqPops = aSet.compute_eq_pops(atmos)

background = Background(atmos, spect)
ctx = Context(atmos, spect, eqPops, background)



