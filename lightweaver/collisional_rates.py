from atomic_model import CollisionalRates, AtomicModel
import constants as Const
from dataclasses import dataclass
import numpy as np
from typing import Sequence
from scipy.special import exp1
from numba import njit
from scipy.interpolate import interp1d

@dataclass(eq=False)
class TemperatureInterpolationRates(CollisionalRates):
    temperature: Sequence[float]
    rates: Sequence[float]

    def setup_interpolator(self):
        if len(self.rates) <  3:
            self.interpolator = interp1d(self.temperature, self.rates, fill_value=(self.rates[0], self.rates[-1]), bounds_error=False)
        else:
            self.interpolator = interp1d(self.temperature, self.rates, kind=3, fill_value=(self.rates[0], self.rates[-1]), bounds_error=False)

@dataclass(eq=False)
class Omega(TemperatureInterpolationRates):
    def __repr__(self):
        s = 'Omega(j=%d, i=%d, temperature=%s, rates=%s)' % (self.j, self.i, repr(self.temperature), repr(self.rates))
        return s

    def setup(self, atom):
        i, j = self.i, self.j
        self.i = min(i, j)
        self.j = max(i, j)
        self.atom = atom
        self.jLevel = atom.levels[self.j]
        self.iLevel = atom.levels[self.i]
        self.C0 = Const.ERydberg / np.sqrt(Const.MElectron) * np.pi * Const.RBohr**2 * np.sqrt(8.0 / (np.pi * Const.KBoltzmann))

    def compute_rates(self, atmos, nstar, Cmat):
        try:
            C = self.interpolator(atmos.temperature)
        except AttributeError:
            self.setup_interpolator()
            C = self.interpolator(atmos.temperature)

        Cdown = self.C0 * atmos.ne * C / (self.jLevel.g * np.sqrt(atmos.temperature))
        Cmat[self.i, self.j, :] += Cdown
        Cmat[self.j, self.i, :] += Cdown * nstar[self.j] / nstar[self.i]

@dataclass(eq=False)
class CI(TemperatureInterpolationRates):
    def __repr__(self):
        s = 'CI(j=%d, i=%d, temperature=%s, rates=%s)' % (self.j, self.i, repr(self.temperature), repr(self.rates))
        return s

    def setup(self, atom):
        i, j = self.i, self.j
        self.i = min(i, j)
        self.j = max(i, j)
        self.atom = atom
        self.jLevel = atom.levels[self.j]
        self.iLevel = atom.levels[self.i]
        self.dE = self.jLevel.E_SI - self.iLevel.E_SI

    def compute_rates(self, atmos, nstar, Cmat):
        try:
            C = self.interpolator(atmos.temperature)
        except AttributeError:
            self.setup_interpolator()
            C = self.interpolator(atmos.temperature)
        Cup = C * atmos.ne * np.exp(-self.dE / (Const.KBoltzmann * atmos.temperature)) * np.sqrt(atmos.temperature)
        Cmat[self.j, self.i, :] += Cup
        Cmat[self.i, self.j, :] += Cup * nstar[self.i] / nstar[self.j]


@dataclass(eq=False)
class CE(TemperatureInterpolationRates):
    def __repr__(self):
        s = 'CE(j=%d, i=%d, temperature=%s, rates=%s)' % (self.j, self.i, repr(self.temperature), repr(self.rates))
        return s

    def setup(self, atom):
        i, j = self.i, self.j
        self.i = min(i, j)
        self.j = max(i, j)
        self.atom = atom
        self.jLevel = atom.levels[self.j]
        self.iLevel = atom.levels[self.i]
        self.gij = self.iLevel.g / self.jLevel.g

    def compute_rates(self, atmos, nstar, Cmat):
        try:
            C = self.interpolator(atmos.temperature)
        except AttributeError:
            self.setup_interpolator()
            C = self.interpolator(atmos.temperature)
        Cdown = C * atmos.ne * self.gij * np.sqrt(atmos.temperature)
        Cmat[self.i, self.j, :] += Cdown
        Cmat[self.j, self.i, :] += Cdown * nstar[self.j] / nstar[self.i]
