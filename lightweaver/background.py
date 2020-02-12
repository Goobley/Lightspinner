import numpy as np
from dataclasses import dataclass
from .atmosphere import Atmosphere
from .atomic_set import SpectrumConfiguration
import witt
import constants as Const
from .atomic_table import get_global_atomic_table
from .utils import planck

def thomson_scattering(atmos):
    sigma = 8.0 * np.pi / 3.0 * (Const.QElectron / (np.sqrt(4.0 * np.pi * Const.Epsilon0) * (np.sqrt(Const.MElectron) * Const.CLight)))**4
    sca = atmos.ne * sigma
    return sca

class Background:
    def __init__(self, atmos: Atmosphere, spect: SpectrumConfiguration):
        Nspace = atmos.Nspace
        Nspect = spect.wavelength.shape[0]

        at = get_global_atomic_table()
        eos = witt.witt()
        pgas = np.zeros(Nspace)
        pe = np.zeros(Nspace)
        rho = Const.Amu * at.weightPerH * atmos.nHTot * Const.CM_TO_M**3 / Const.G_TO_KG
        for k in range(Nspace):
            pgas[k] = eos.pg_from_rho(atmos.temperature[k], rho[k])
            pe[k] = eos.pe_from_rho(atmos.temperature[k], rho[k])

        chi = np.zeros((Nspect, Nspace))
        eta = np.zeros((Nspect, Nspace))
        for k in range(Nspace):
            chi[:, k] = eos.contOpacity(atmos.temperature[k], pgas[k], pe[k], spect.wavelength*10) / Const.CM_TO_M
            eta[:, k] = planck(atmos.temperature[k], spect.wavelength) * chi[:, k]

        sca = np.zeros((Nspect, Nspace))
        thomson = thomson_scattering(atmos)
        sca[None, :] = thomson

        self.chi = chi
        self.eta = eta
        self.sca = sca
