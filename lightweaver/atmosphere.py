from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Sequence, TYPE_CHECKING, Optional, Union
import numpy as np
from witt import witt
import constants as Const
from scipy.interpolate import interp1d
from numpy.polynomial.legendre import leggauss
from utils import ConvergenceError
from atomic_table import get_global_atomic_table
import astropy.units as u

class ScaleType(Enum):
    Geometric = 0
    ColumnMass = auto()
    Tau500 = auto()

class BoundaryCondition(Enum):
    Zero = auto()
    Thermalised = auto()

uvel = u.m / u.s
unumdens = u.m**(-3)
udens = u.kg / u.m**3

def conditional_unit(x, unit):
    if x is None:
        return x
    return x << unit

def conditional_deunit(x):
    if x is None:
        return x
    return x.value

class AtmosphereConstructor:

    @u.quantity_input(depthScale=[u.m, u.kg/u.m**2, u.one], temperature=u.K, vlos=uvel,
                      vturb=uvel, ne=unumdens, hydrogenPops=unumdens, nHTot=unumdens)
    def __init__(self, depthScale, temperature, vlos, vturb, ne=None,
                 hydrogenPops=None, nHTot=None, 
                 lowerBc=BoundaryCondition.Thermalised, upperBc=BoundaryCondition.Zero):

        if depthScale.unit.is_equivalent(u.kg / u.m**2):
            self.depthScale = depthScale << u.kg / u.m**2
            self.scale = ScaleType.ColumnMass
        elif depthScale.unit.is_equivalent(u.m):
            self.depthScale = depthScale << u.m
            self.scale = ScaleType.Geometric
        else:
            self.depthScale = depthScale << u.one
            self.scale = ScaleType.Tau500
        self.temperature = temperature << u.K
        self.vlos = vlos << uvel
        self.vturb = vturb << uvel
        self.ne = conditional_unit(ne, unumdens)
        self.hydrogenPops = conditional_unit(hydrogenPops, unumdens)
        self.nHTot = conditional_unit(nHTot, unumdens)
        self.lowerBc = lowerBc
        self.upperBc = upperBc

        if self.hydrogenPops is not None:
            if self.nHTot is None:
                self.nHTot = np.sum(self.hydrogenPops, axis=0)
            else:
                raise ValueError('AtmosphereConstructor: Both nHTot and hydrogenPops are set, only one should be used')

        self.dimensioned = True

    def convert_scales(self, atomicTable=None, logG=2.44) -> 'Atmosphere':
        self.nondimensionalise()
        if atomicTable is None:
            atomicTable = get_global_atomic_table()

        if np.any(self.temperature < 2500):
            raise ValueError('Extremely low temperature (< 2500 K)')
        Nspace = self.Nspace

        eos = witt()

        rhoSI = Const.Amu * atomicTable.weightPerH * self.nHTot
        rho = Const.Amu * atomicTable.weightPerH * self.nHTot * Const.CM_TO_M**3 / Const.G_TO_KG
        pgas = np.zeros_like(self.depthScale)
        pe = np.zeros_like(self.depthScale)
        for k in range(self.depthScale.shape[0]):
            pgas[k] = eos.pg_from_rho(self.temperature[k], rho[k])
            pe[k] = eos.pe_from_rho(self.temperature[k], rho[k])

        chi_c = np.zeros_like(self.depthScale)
        for k in range(self.depthScale.shape[0]):
            chi_c[k] = eos.contOpacity(self.temperature[k], pgas[k], pe[k], np.array([5000.0])) / Const.CM_TO_M

        if self.scale == ScaleType.ColumnMass:
            height = np.zeros_like(self.depthScale)
            tau_ref = np.zeros_like(self.depthScale)
            cmass = self.depthScale

            height[0] = 0.0
            tau_ref[0] = chi_c[0] / rhoSI[0] * cmass[0]
            for k in range(1, cmass.shape[0]):
                height[k] = height[k-1] - 2.0 * (cmass[k] - cmass[k-1]) / (rhoSI[k-1] + rhoSI[k])
                tau_ref[k] = tau_ref[k-1] + 0.5 * (chi_c[k-1] + chi_c[k]) * (height[k-1] - height[k])

            hTau1 = np.interp(1.0, tau_ref, height)
            height -= hTau1

            self.cmass = cmass
            self.height = height
            self.tau_ref = tau_ref
        elif self.scale == ScaleType.Geometric:
            cmass = np.zeros(Nspace)
            tau_ref = np.zeros(Nspace)
            height = self.depthScale

            cmass[0] = (self.nHTot[0] * atomicTable.weightPerH + self.ne[0]) * (Const.KBoltzmann * self.temperature[0] / 10**logG)
            tau_ref[0] = 0.5 * chi_c[0] * (height[0] - height[1])
            if tau_ref[0] > 1.0:
                tau_ref[0] = 0.0

            for k in range(1, Nspace):
                cmass[k] = cmass[k-1] + 0.5 * (rhoSI[k-1] + rhoSI[k]) * (height[k-1] - height[k])
                tau_ref[k] = tau_ref[k-1] + 0.5 * (chi_c[k-1] + chi_c[k]) * (height[k-1] - height[k])
            self.cmass = cmass
            self.height = height
            self.tau_ref = tau_ref
        elif self.scale == ScaleType.Tau500:
            cmass = np.zeros(Nspace)
            height = np.zeros(Nspace)
            tau_ref = self.depthScale

            cmass[0] = (tau_ref[0] / chi_c[0]) * rhoSI[0]
            for k in range(1, Nspace):
                height[k] = height[k-1] - 2.0 * (tau_ref[k] - tau_ref[k-1]) / (chi_c[k-1] + chi_c[k])
                cmass[k] = cmass[k-1] + 0.5 * (chi_c[k-1] + chi_c[k]) * (height[k-1] - height[k])

            hTau1 = np.interp(1.0, tau_ref, height)
            height -= hTau1

            self.cmass = cmass
            self.height = height
            self.tau_ref = tau_ref

        self.dimensionalise()
        return Atmosphere(self)

    def quadrature(self, Nrays: Optional[int]=None, mu: Optional[Sequence[float]]=None, wmu: Optional[Sequence[float]]=None):

        if Nrays is not None and mu is None:
            if Nrays >= 1:        
                x, w = leggauss(Nrays)
                mid, halfWidth = 0.5, 0.5
                x = mid + halfWidth * x
                w *= halfWidth

                self.muz = x
                self.wmu = w
            else:
                raise ValueError('Unsupported Nrays=%d' % Nrays)
        elif Nrays is not None and mu is not None:
            if wmu is None:
                raise ValueError('Must provide wmu when providing mu')
            if Nrays != len(mu):
                raise ValueError('mu must be Nrays long if Nrays is provided')
            if len(mu) != len(wmu):
                raise ValueError('mu and wmu must be the same shape')

            self.muz = np.array(mu)
            self.wmu = np.array(wmu)

        self.muy = np.zeros_like(self.muz)
        self.mux = np.sqrt(1.0 - self.muz**2)

    def rays(self, mu: Union[float, Sequence[float]]):
        if isinstance(mu, float):
            mu = [mu]

        self.muz = np.array(mu)
        self.wmu = np.zeros_like(self.muz)
        self.muy = np.zeros_like(self.muz)
        self.mux = np.sqrt(1.0 - self.muz**2)

    @property
    def Nspace(self):
        if self.depthScale is not None:
            return self.depthScale.shape[0]
        elif self.cmass is not None:
            return self.cmass.shape[0]
        elif self.height is not None:
            return self.height.shape[0]
        elif self.tau_ref is not None:
            return self.tau_ref.shape[0]

    @property
    def Nrays(self):
        if self.muz is None:
            raise AttributeError('Nrays not set, call atmos.rays or .quadrature first')

        return self.muz.shape[0]

    def dimensionalise(self):
        if self.dimensioned:
            return

        if self.scale is ScaleType.Tau500:
            self.depthScale = self.depthScale << u.one
        elif self.scale is ScaleType.Geometric:
            self.depthScale = self.depthScale << u.m
        else:
            self.depthScale = self.depthScale << u.kg / u.m**2

        self.temperature = self.temperature << u.K
        self.vlos = self.vlos << uvel
        self.vturb = self.vturb << uvel
        self.ne = conditional_unit(self.ne, unumdens)
        self.hydrogenPops = conditional_unit(self.hydrogenPops, unumdens)
        self.nHTot = conditional_unit(self.nHTot, unumdens)
        try:
            self.cmass = self.cmass << u.g / u.m**2
        except AttributeError:
            pass
        try:
            self.height = self.height << u.m
        except AttributeError:
            pass
        try:
            self.tau_ref = self.tau_ref << u.one
        except AttributeError:
            pass
        self.dimensioned = True

    def nondimensionalise(self):
        if not self.dimensioned:
            return

        self.depthScale = self.depthScale.value
        self.temperature = self.temperature.value
        self.vlos = self.vlos.value
        self.vturb = self.vturb.value
        self.ne = conditional_deunit(self.ne)
        self.hydrogenPops = conditional_deunit(self.hydrogenPops)
        self.nHTot = conditional_deunit(self.nHTot)
        try:
            self.cmass = self.cmass.value
        except AttributeError:
            pass
        try:
            self.height = self.height.value
        except AttributeError:
            pass
        try:
            self.tau_ref = self.tau_ref.value
        except AttributeError:
            pass
        self.dimensioned = False


class Atmosphere:

    def __init__(self, atmos: AtmosphereConstructor):
        if not atmos.dimensioned:
            atmos.dimensionalise()

        self.temperature: u.Quantity = atmos.temperature
        self.vlos: u.Quantity = atmos.vlos
        self.vturb: u.Quantity = atmos.vturb
        self.ne: u.Quantity = atmos.ne
        self.nHTot: u.Quantity = atmos.nHTot

        self.refScale = atmos.scale
        self.height: u.Quantity = atmos.height
        self.cmass: u.Quantity = atmos.cmass
        self.tau_ref: np.ndarray = atmos.tau_ref

        self.mux = atmos.mux
        self.muy = atmos.muy
        self.muz = atmos.muz
        self.wmu = atmos.wmu

        self.dimensioned = True

    def convert_scales(self, atomicTable=None, logG=2.44):
        self.nondimensionalise()
        if atomicTable is None:
            atomicTable = get_global_atomic_table()

        if np.any(self.temperature < 2500):
            raise ValueError('Extremely low temperature (< 2500 K)')

        eos = witt()
        Nspace = self.Nspace

        rhoSI = Const.Amu * atomicTable.weightPerH * self.nHTot
        rho = Const.Amu * atomicTable.weightPerH * self.nHTot * Const.CM_TO_M**3 / Const.G_TO_KG
        pgas = np.zeros(Nspace)
        pe = np.zeros(Nspace)
        for k in range(Nspace):
            pgas[k] = eos.pg_from_rho(self.temperature[k], rho[k])
            pe[k] = eos.pe_from_rho(self.temperature[k], rho[k])

        chi_c = np.zeros(Nspace)
        for k in range(Nspace):
            chi_c[k] = eos.contOpacity(self.temperature[k], pgas[k], pe[k], np.array([5000.0])) / Const.CM_TO_M

        if self.refScale == ScaleType.ColumnMass:
            height = np.zeros(Nspace)
            tau_ref = np.zeros(Nspace)
            cmass = self.cmass

            height[0] = 0.0
            tau_ref[0] = chi_c[0] / rhoSI[0] * cmass[0]
            for k in range(1, cmass.shape[0]):
                height[k] = height[k-1] - 2.0 * (cmass[k] - cmass[k-1]) / (rhoSI[k-1] + rhoSI[k])
                tau_ref[k] = tau_ref[k-1] + 0.5 * (chi_c[k-1] + chi_c[k]) * (height[k-1] - height[k])

            hTau1 = np.interp(1.0, tau_ref, height)
            height -= hTau1

            self.cmass[:] = cmass
            self.height[:] = height
            self.tau_ref[:] = tau_ref
        elif self.refScale == ScaleType.Geometric:
            cmass = np.zeros(Nspace)
            tau_ref = np.zeros(Nspace)
            height = self.height

            cmass[0] = (self.nHTot[0] * atomicTable.weightPerH + self.ne[0]) * (Const.KBoltzmann * self.temperature[0] / 10**logG)
            tau_ref[0] = 0.5 * chi_c[0] * (height[0] - height[1])
            if tau_ref[0] > 1.0:
                tau_ref[0] = 0.0

            for k in range(1, Nspace):
                cmass[k] = cmass[k-1] + 0.5 * (rhoSI[k-1] + rhoSI[k]) * (height[k-1] - height[k])
                tau_ref[k] = tau_ref[k-1] + 0.5 * (chi_c[k-1] + chi_c[k]) * (height[k-1] - height[k])
            self.cmass[:] = cmass
            self.height[:] = height
            self.tau_ref[:] = tau_ref
        elif self.scale == ScaleType.Tau500:
            cmass = np.zeros(Nspace)
            height = np.zeros(Nspace)
            tau_ref = self.tau_ref

            cmass[0] = (tau_ref[0] / chi_c[0]) * rhoSI[0]
            for k in range(1, Nspace):
                height[k] = height[k-1] - 2.0 * (tau_ref[k] - tau_ref[k-1]) / (chi_c[k-1] + chi_c[k])
                cmass[k] = cmass[k-1] + 0.5 * (chi_c[k-1] + chi_c[k]) * (height[k-1] - height[k])

            hTau1 = np.interp(1.0, tau_ref, height)
            height -= hTau1

            self.cmass[:] = cmass
            self.height[:] = height
            self.tau_ref[:] = tau_ref
        
        self.dimensionalise()

    def quadrature(self, Nrays: Optional[int]=None, mu: Optional[Sequence[float]]=None, wmu: Optional[Sequence[float]]=None):

        if Nrays is not None and mu is None:
            if Nrays >= 1:        
                x, w = leggauss(Nrays)
                mid, halfWidth = 0.5, 0.5
                x = mid + halfWidth * x
                w *= halfWidth

                self.muz = x
                self.wmu = w
            else:
                raise ValueError('Unsupported Nrays=%d' % Nrays)
        elif Nrays is not None and mu is not None:
            if wmu is None:
                raise ValueError('Must provide wmu when providing mu')
            if Nrays != len(mu):
                raise ValueError('mu must be Nrays long if Nrays is provided')
            if len(mu) != len(wmu):
                raise ValueError('mu and wmu must be the same shape')

            self.muz = np.array(mu)
            self.wmu = np.array(wmu)

        self.muy = np.zeros_like(self.muz)
        self.mux = np.sqrt(1.0 - self.muz**2)

    def rays(self, mu: Union[float, Sequence[float]]):
        if isinstance(mu, float):
            mu = [mu]

        self.muz = np.array(mu)
        self.wmu = np.zeros_like(self.muz)
        self.muy = np.zeros_like(self.muz)
        self.mux = np.sqrt(1.0 - self.muz**2)

    @property
    def Nspace(self):
        if self.cmass is not None:
            return self.cmass.shape[0]
        elif self.height is not None:
            return self.height.shape[0]
        elif self.tau_ref is not None:
            return self.tau_ref.shape[0]

    @property
    def Nrays(self):
        if self.muz is None:
            raise AttributeError('Nrays not set, call atmos.rays or .quadrature first')

        return self.muz.shape[0]

    def dimensionalise(self):
        if self.dimensioned:
            return
        self.temperature = self.temperature << u.K
        self.vlos = self.vlos << uvel
        self.vturb = self.vturb << uvel
        self.ne = conditional_unit(self.ne, unumdens)
        self.nHTot = conditional_unit(self.nHTot, unumdens)
        self.cmass = self.cmass << u.g / u.m**2
        self.height = self.height << u.m
        self.tau_ref = self.tau_ref << u.one
        self.dimensioned = True

    def nondimensionalise(self):
        if not self.dimensioned:
            return
        
        self.temperature = self.temperature.value
        self.vlos = self.vlos.value
        self.vturb = self.vturb.value
        self.ne = conditional_deunit(self.ne)
        self.nHTot = conditional_deunit(self.nHTot)
        self.cmass = self.cmass.value
        self.height = self.height.value
        self.tau_ref = self.tau_ref.value
        self.dimensioned = False
