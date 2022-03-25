from fal import Falc82
from rh_atoms import CaII_atom, H_6_atom
from atomic_set import RadiativeSet
from rh_method import Context
from background import Background
import matplotlib.pyplot as plt
import pdb
import time

atmosConst = Falc82()
atmosConst.quadrature(5)
atmos = atmosConst.convert_scales()

aSet = RadiativeSet([CaII_atom(), H_6_atom()])
aSet.set_active('Ca')
spect = aSet.compute_wavelength_grid()
eqPops = aSet.compute_eq_pops(atmos)
eqPops2 = aSet.compute_eq_pops(atmos)

background = Background(atmos, spect)
ctx = Context(atmos, spect, eqPops, background)
ctxTrapez = Context(atmos, spect, eqPops2, background, formalSolver='trapezoidal')
dJ = ctx.formal_sol_gamma_matrices()
print(dJ)
dJ = ctxTrapez.formal_sol_gamma_matrices()
print(dJ)

# pdb.set_trace()

start = time.time()
dJ = 1.0
dPops = 1.0
i = 0
while dJ > 2e-3 or dPops > 1e-3:
    i += 1
    dJ = ctxTrapez.formal_sol_gamma_matrices()

    if i > 3:
        dPops = ctxTrapez.stat_equil()
    print('Iteration %.3d: dJ: %.2e, dPops: %s' % (i, dJ, 'Just iterating Jbar' if i < 3 else '%.2e' % dPops))
end = time.time()

print(f'{end - start}')
print('-'*80)

start = time.time()
dJ = 1.0
dPops = 1.0
i = 0
while dJ > 2e-3 or dPops > 1e-3:
    i += 1
    dJ = ctx.formal_sol_gamma_matrices()

    if i > 3:
        dPops = ctx.stat_equil()
    print('Iteration %.3d: dJ: %.2e, dPops: %s' % (i, dJ, 'Just iterating Jbar' if i < 3 else '%.2e' % dPops))
end = time.time()
print(f'{end - start}')

import numpy as np
Imu = np.load('Imu.npy')
plt.plot(spect.wavelength, ctx.I[:, -1])
plt.plot(spect.wavelength, ctxTrapez.I[:, -1])
plt.plot(spect.wavelength, Imu[:, -1])
plt.show()