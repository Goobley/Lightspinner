from dataclasses import dataclass
from enum import Enum, auto
import numpy as np
from typing import Union, Optional
from .atomic_model import AtomicLine, AtomicContinuum
from .atmosphere import Atmosphere
from .atomic_set import SpectrumConfiguration
import constants as Const

class TransitionType(Enum):
    Line = auto()
    Continuum = auto()

@dataclass
class UV:
    Uji: np.ndarray
    Vij: np.ndarray
    Vji: np.ndarray

def voigt_H(a, v):
    z = (v + 1j * a)
    return special.wofz(z).real

class ComputationalTransition:
    def __init__(self, trans: Union[AtomicLine, AtomicContinuum], compAtom: 'ComputationalAtom', atmos: Atmosphere, spect: SpectrumConfiguration):
        self.transModel = trans
        self.atom = compAtom
        self.wavelength = trans.wavelength
        
        if isinstance(trans, AtomicLine):
            self.type = TransitionType.Line
            self.Aji = trans.Aji
            self.Bji = trans.Bji
            self.Bij = trans.Bij
            self.lambda0 = trans.lambda0
        else:
            self.type = TransitionType.Continuum

        self.i = trans.i
        self.j = trans.j

        self.Nblue = np.searchsorted(spect.wavelength, self.wavelength[0])
        self.compute_phi(atmos)
        self.active = np.zeros(spect.wavelength.shape[0], np.bool)
        for i, s in enumerate(spect.activeSet):
            if trans in s:
                self.active[i] = True

    def lt(self, la: int) -> int:
        return la - self.Nblue

    def wlambda(self, la: Optional[int]=None) -> Union[float, np.ndarray]:
        pass

    def compute_phi(self, atmos: Atmosphere):
        if self.type == TransitionType.Continuum:
            return

        sqrtPi = np.sqrt(np.pi)
        aDamp = self.transModel.damping(atmos, self.atom.vBroad)
        phi = np.zeros((self.transModel.Nlambda, atmos.Nrays, 2, atmos.Nspace))
        wPhi = np.zeros(atmos.Nspace)

        wLambda = self.wlambda()

        vlosDop = np.zeros((atmos.Nrays, atmos.Nspace))
        for mu in range(atmos.Nrays):
            vlosDop[mu, :] = atmos.muz[mu] * atmos.vlos / self.atom.vBroad

        for la in range(self.wavelength.shape[0]):
            v = (self.wavelength[la] - self.lambda0) * Const.CLight / (self.atom.vBroad * self.lambda0)
            for mu in range(atmos.Nrays):
                wlamu = wLambda * 0.5 * atmos.wmu[mu]
                for toFrom, sign in enumerate([-1.0, 1.0]):
                    vk = v + sign * vlosDop[mu]
                    phi[la, mu, toFrom, :] = voigt_H(aDamp, vk) / (sqrtPi * self.atom.vBroad)
                    wPhi[:] += phi[la, mu, toFrom, :] * wlamu[la]

        self.wphi = 1.0 / wPhi
        self.phi = phi


class ComputationalAtom:
    vBroad: np.ndarray
    pass

        