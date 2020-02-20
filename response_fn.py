import numpy as np
from fal import Falc82
from rh_atoms import CaII_atom, H_6_atom
from atomic_set import RadiativeSet
from rh_method import Context
from background import Background
import matplotlib.pyplot as plt
from copy import deepcopy
import astropy.units as u

def iterate_ctx(ctx):
    dJ = 1.0
    dPops = 1.0
    i = 0
    while dJ > 2e-3 or dPops > 1e-3:
        i += 1
        dJ = ctx.formal_sol_gamma_matrices()

        if i > 3:
            dPops = ctx.stat_equil()
        print('Iteration %.3d: dJ: %.2e, dPops: %s' % (i, dJ, 'Just iterating Jbar' if i < 3 else '%.2e' % dPops))

def iterate_temperature_perturbation(startingPops, k, pertSize):
    atmosConst = Falc82()
    atmosConst.quadrature(5)
    atmosConst.temperature[k] += pertSize
    atmos = atmosConst.convert_scales()

    aSet = RadiativeSet([CaII_atom(), H_6_atom()])
    aSet.set_active('Ca')
    spect = aSet.compute_wavelength_grid()
    eqPops = aSet.compute_eq_pops(atmos)
    eqPops['Ca'].pops = np.copy(startingPops)

    background = Background(atmos, spect)
    ctx = Context(atmos, spect, eqPops, background)

    iterate_ctx(ctx)
    return ctx.I




atmosConst = Falc82()
atmosConst.quadrature(5)
atmos = atmosConst.convert_scales()

aSet = RadiativeSet([CaII_atom(), H_6_atom()])
aSet.set_active('Ca')
spect = aSet.compute_wavelength_grid()
eqPops = aSet.compute_eq_pops(atmos)

background = Background(atmos, spect)
ctx = Context(atmos, spect, eqPops, background)

iterate_ctx(ctx)

tempPert = 50 << u.K
Iplus = np.zeros((spect.wavelength.shape[0], atmos.Nspace))
Iminus = np.zeros((spect.wavelength.shape[0], atmos.Nspace))
for k in range(atmos.Nspace):
    Ip = iterate_temperature_perturbation(eqPops['Ca'].n, k, 0.5 * tempPert)[:,-1]
    Im = iterate_temperature_perturbation(eqPops['Ca'].n, k, -0.5 * tempPert)[:, -1]
    Iplus[:, k] = Ip
    Iminus[:, k] = Im

rf = (Iplus - Iminus) / ctx.I[:, -1][:, None]

yEdges = np.arange(atmos.Nspace+1) - 0.5
xEdges = 0.5 * (spect.wavelength[1:] + spect.wavelength[:-1])
xEdges = np.insert(xEdges, 0, xEdges[0] - (xEdges[1] - xEdges[0]))
xEdges = np.insert(xEdges, -1, xEdges[-1] + (xEdges[-1] - xEdges[-2]))
plt.pcolormesh(xEdges, yEdges, rf.T)
plt.xlim(853.944, 854.944)
