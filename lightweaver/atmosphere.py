from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Sequence, TYPE_CHECKING, Optional, Union
import numpy as np
from .witt import witt
import lightweaver.constants as Const
from scipy.interpolate import interp1d
from numpy.polynomial.legendre import leggauss
from .utils import ConvergenceError
from .atomic_table import get_global_atomic_table

class ScaleType(Enum):
    Geometric = 0
    ColumnMass = auto()
    Tau500 = auto()

class BoundaryCondition(Enum):
    Zero = auto()
    Thermalised = auto()

@dataclass
class Atmosphere:
    scale: ScaleType
    depthScale: np.ndarray
    temperature: np.ndarray
    vlos: np.ndarray
    vturb: np.ndarray
    ne: Optional[np.ndarray] = None
    hydrogenPops: Optional[np.ndarray] = None
    B: Optional[np.ndarray] = None
    gammaB: Optional[np.ndarray] = None
    chiB: Optional[np.ndarray] = None
    nHTot: Optional[np.ndarray] = None
    lowerBc: BoundaryCondition = field(default=BoundaryCondition.Thermalised)
    upperBc: BoundaryCondition = field(default=BoundaryCondition.Zero)

    def __post_init__(self):
        if self.hydrogenPops is not None:
            self.nHTot = np.sum(self.hydrogenPops, axis=0)

    def convert_scales(self, atomicTable=None, logG=2.44, Pgas=None, Pe=None, Ptop=None, PeTop=None):
        if atomicTable is None:
            atomicTable = get_global_atomic_table()

        if np.any(self.temperature < 2500):
            raise ValueError('Extremely low temperature (< 2500 K)')

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
        else:
            raise ValueError("Other scales not handled yet")

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





