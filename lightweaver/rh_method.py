from dataclasses import dataclass
from enum import Enum, auto
import numpy as np
from numpy.linalg import solve
from typing import Union, Optional
from atomic_model import AtomicLine, AtomicContinuum, AtomicModel, VoigtLine
from atmosphere import Atmosphere
from atomic_set import SpectrumConfiguration, AtomicStateTable
from atomic_table import AtomicTable
import constants as Const
from background import Background
from numba import njit
from utils import voigt_H, planck
from typing import cast

@dataclass
class UV:
    Uji: np.ndarray
    Vij: np.ndarray
    Vji: np.ndarray

class ComputationalTransition:
    def __init__(self, trans: Union[AtomicLine, AtomicContinuum], compAtom: 'ComputationalAtom', atmos: Atmosphere, spect: SpectrumConfiguration, gij: float):
        self.transModel = trans
        self.atom = compAtom
        self.wavelength = trans.wavelength
        
        if isinstance(trans, AtomicLine):
            self.Aji = trans.Aji
            self.Bji = trans.Bji
            self.Bij = trans.Bij
            self.lambda0 = trans.lambda0
        else:
            self.alpha = trans.alpha

        self.i = trans.i
        self.j = trans.j

        self.Nblue = np.searchsorted(spect.wavelength, self.wavelength[0])
        self.compute_phi(atmos)
        self.active = np.zeros(spect.wavelength.shape[0], np.bool)
        for i, s in enumerate(spect.activeSet):
            if trans in s:
                self.active[i] = True

        self.gij = gij
        self.Rij = np.zeros(atmos.Nspace)
        self.Rji = np.zeros(atmos.Nspace)

    def lt(self, la: int) -> int:
        return la - self.Nblue

    def wlambda(self, la: Optional[int]=None) -> Union[float, np.ndarray]:
        if isinstance(self.transModel, AtomicLine):
            dopplerWidth = Const.CLight / self.lambda0
        else:
            dopplerWidth = 1.0

        if la is not None:
            if la == 0:
                return 0.5 * (self.wavelength[1] - self.wavelength[0]) * dopplerWidth
            elif la == self.wavelength.shape[0]-1:
                return 0.5 * (self.wavelength[-1] - self.wavelength[-2]) * dopplerWidth
            else:
                return 0.5 * (self.wavelength[la+1] - self.wavelength[la-1]) * dopplerWidth

        wla = np.zeros_like(self.wavelength)
        wla[0] = 0.5 * (self.wavelength[1] - self.wavelength[0])
        wla[-1] = 0.5 * (self.wavelength[-1] - self.wavelength[-2])
        wla[1:-1] = 0.5 * (self.wavelength[2:] - self.wavelength[:-2])

        return dopplerWidth * wla

    def compute_phi(self, atmos: Atmosphere):
        if isinstance(self.transModel, AtomicContinuum):
            return

        sqrtPi = np.sqrt(np.pi)
        aDamp, Qelast = self.transModel.damping(atmos, self.atom.vBroad, self.atom.hPops.nStar[0])
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

    def uv(self, la, mu, toFrom) -> UV:
        lt = self.lt(la)

        hc_4pi = 0.25 * Const.HC / np.pi

        if isinstance(self.transModel, AtomicLine):
            phi = self.phi[lt, mu, toFrom, :]
            Vij = hc_4pi * self.Bij * phi
            Vji = self.gij * Vij
            Uji = self.Aji / self.Bji * Vji
        else:
            Vij = self.alpha[lt]
            Vji = self.gij * Vij
            Uji = 2.0 * Const.HC / (Const.NM_TO_M * self.wavelength[lt])**3 * Vji

        return UV(Uji=Uji, Vij=Vij, Vji=Vji)



class ComputationalAtom:
    def __init__(self, atom: AtomicModel, atmos: Atmosphere, spect: SpectrumConfiguration, eqPops: AtomicStateTable):
        self.atomicModel = atom
        self.atomicTable = eqPops.atomicTable
        self.spect = spect
        self.atmos = atmos

        self.vBroad = atom.vBroad(atmos)

        self.pops = eqPops[atom.name]
        self.hPops = eqPops['H']
        self.nTotal = self.pops.nTotal

        self.trans = []
        # TODO(cmo): Fix gij
        gij = 1.0
        for l in atom.lines:
            if l in spect.transitions:
                self.trans.append(ComputationalTransition(l, self, atmos, spect, gij))

        for c in atom.continua:
            if c in spect.transitions:
                self.trans.append(ComputationalTransition(c, self, atmos, spect, gij))

        Nlevel = len(atom.levels)
        self.Gamma = np.zeros((Nlevel, Nlevel, atmos.Nspace))
        self.C = np.zeros((Nlevel, Nlevel, atmos.Nspace))
        self.nStar = self.pops.nStar
        self.n = np.copy(self.nStar)

        self.Nlevel = Nlevel
        self.Ntrans = len(self.trans)
        self.Nspace = atmos.Nspace

    def setup_wavelength(self, laIdx: int):
        self.eta = np.zeros(self.Nspace)
        self.gij = np.zeros((self.Ntrans, self.Nspace))
        self.wla = np.zeros((self.Ntrans, self.Nspace))
        self.V = np.zeros((self.Nlevel, self.Nspace))
        self.U = np.zeros((self.Nlevel, self.Nspace))
        self.chi = np.zeros((self.Nlevel, self.Nspace))

        hc_k = Const.HC / (Const.KBoltzmann * Const.NM_TO_M)
        h_4pi = 0.25 * Const.HPlanck / np.pi
        hc_4pi = h_4pi * Const.CLight
        for kr, t in enumerate(self.trans):
            if not t.active[laIdx]:
                continue

            if isinstance(t, AtomicLine):
                self.gij[kr, :] = t.Bji / t.Bij
                self.wla[kr, :] = t.wlambda(t.lt(laIdx)) * t.wphi / hc_4pi
            else:
                self.gij[kr, :] = self.nStar[t.i] / self.nStar[t.j] \
                    * np.exp(-hc_k / self.spect.wavelength[laIdx] / self.atmos.temperature)
                self.wla[kr, :] = t.wlambda(t.lt(laIdx)) / self.spect.wavelength[laIdx] / h_4pi
            t.gij = self.gij[kr]

    def zero_angle_dependent_vars(self):
        self.eta.fill(0.0)
        self.V.fill(0.0)
        self.U.fill(0.0)
        self.chi.fill(0.0)

    def compute_collisions(self):
        self.C = np.zeros_like(self.Gamma)
        for col in self.atomicModel.collisions:
            col.compute_rates(self.atmos, self.nstar, self.C)
        self.C[self.C < 0.0] = 0.0

@dataclass
class IPsi:
    I: np.ndarray
    PsiStar: np.ndarray

@njit
def piecewise_1d_impl(muz, toFrom, Istart, z, chi, S):
    Nspace = chi.shape[0]
    zmu = 0.5 / muz

    if toFrom:
        dk = -1
        kStart = Nspace - 1
        kEnd = 0
    else:
        dk = 1
        kStart = 0
        kEnd = Nspace - 1

    dtau_uw = zmu * (chi[kStart] + chi[kStart + dk]) * np.abs(z[kStart] - z[kStart + dk])
    dS_uw = (S[kStart] - S[kStart + dk]) / dtau_uw

    Iupw = Istart
    I = np.empty(Nspace)
    Psi = np.empty(Nspace)
    I[kStart] = Iupw
    Psi[kStart] = 0.0

    for k in range(kStart + dk, kEnd + dk, dk):
        w = w2(dtau_uw)

        if k != kEnd:
            dtau_dw = zmu * (chi[k] + chi[k+dk]) * np.abs(z[k] - z[k+dk])
            dS_dw = (S[k] - S[k+dk]) / dtau_dw
            # print(dS_uw, dtau_uw, Iupw, w)
            I[k] = (1.0 - w[0]) * Iupw + w[0] * S[k] + w[1] * dS_uw
            Psi[k] = w[0] - w[1] / dtau_uw
        else:
            I[k] = (1.0 - w[0]) * Iupw + w[0] * S[k] + w[1] * dS_uw
            Psi[k] = w[0] - w[1] / dtau_uw
        
        Iupw = I[k]
        dS_uw = dS_dw
        dtau_uw = dtau_dw

    return I, Psi / chi

def piecewise_linear_1d(atmos, mu, toFrom, wav, chi, S):
    zmu = 0.5 / atmos.muz[mu]
    z = atmos.height

    if toFrom:
        dk = -1
        kStart = atmos.Nspace - 1
        kEnd = 0
    else:
        dk = 1
        kStart = 0
        kEnd = atmos.Nspace - 1

    dtau_uw = zmu * (chi[kStart] + chi[kStart + dk]) * np.abs(z[kStart] - z[kStart + dk])

    if toFrom:
        Bnu = planck(atmos.temperature[-2:], wav)
        Iupw = Bnu[1] - (Bnu[0] - Bnu[1]) / dtau_uw
    else:
        Iupw = 0.0

    I, Psi = piecewise_1d_impl(atmos.muz[mu], toFrom, Iupw, z, chi, S)
    return IPsi(I, Psi)


class Context:
    def __init__(self, atmos: Atmosphere, spect: SpectrumConfiguration, eqPops: AtomicStateTable, background: Background):
        self.atmos = atmos
        self.spect = spect
        self.background = background
        self.eqPops = eqPops

        self.activeAtoms = []
        for a in spect.radSet.activeAtoms:
            self.activeAtoms.append(ComputationalAtom(a, atmos, spect, eqPops))

        self.J = np.zeros((spect.wavelength.shape[0], atmos.Nspace))
        self.I = np.zeros((spect.wavelength.shape[0], atmos.Nrays))

    def formal_sol_gamma_matrices(self):
        Nspace = self.atmos.Nspace
        Nrays = self.atmos.Nrays
        Nspect = self.spect.wavelength.shape[0]

        activeAtoms = self.activeAtoms

        for atom in activeAtoms:
            atom.setup_Gamma()
            atom.compute_collisions()
            atom.Gamma += atom.C

        JDag = np.copy(self.J)
        self.J.fill(0.0)

        for la, wav in enumerate(self.spect.wavelength):
            for atom in activeAtoms:
                atom.setup_wavelength(la)

            for mu in range(Nrays):
                for toFrom, sign in enumerate([-1.0, 1.0]):
                    chiTot = np.zeros(Nspace)
                    etaTot = np.zeros(Nspace)

                    for atom in activeAtoms:
                        atom.zero_angle_dependent_vars()
                        for t in atom.trans:
                            if not t.active[la]:
                                continue

                            uv = t.uv(la, mu, toFrom)
                            chi = atom.n[t.i] * uv.Vij - atom.n[t.j] * uv.Vji
                            eta = atom.n[t.j] * uv.Uji

                            atom.chi[t.i] += chi
                            atom.chi[t.j] -= chi
                            atom.U[t.j] += uv.Uji
                            atom.V[t.i] += uv.Vij
                            atom.V[t.j] += uv.Vji
                            chiTot += chi
                            etaTot += eta
                            atom.eta += eta

                    chiTot += background.chi[la]
                    S = (etaTot + background.eta[la] + background.sca[la] * JDag) / chiTot

                    iPsi = piecewise_linear_1d(atmos, mu, toFrom, chi, S)
                    self.I[la, mu] = iPsi.I[0]
                    self.J[la] += 0.5 * self.atmos.wmu[mu] * iPsi.I

                    for atom in self.activeAtoms:
                        Ieff = iPsi.I - iPsi.PsiStar * atom.eta

                        for kr, t in enumerate(atom.trans):
                            if not t.active[la]:
                                continue

                            wmu = 0.5 * self.atmos.wmu[mu]
                            wlamu = atom.wla[kr] * wmu

                            uv = t.uv(la, mu, toFrom)

                            integrand = (uv.Uji + uv.Vji * Ieff) - (iPsi.PsiStar * atom.chi[t.i] * atom.U[t.j])
                            atom.Gamma[t.i, t.j] += integrand * wlamu

                            integrand = (uv.Vij * Ieff) - (iPsi.PsiStar * atom.chi[t.j] * atom.U[t.i])
                            atom.Gamma[t.j, t.i] += integrand * wlamu

                            t.Rij += I * uv.Vij * wlamu
                            t.Rji += (uv.Uji + I * uv.Vij) * wlamu

        dJ = np.abs(1.0 - JDag / self.J)
        dJMax = dJ.max()

        return dJMax

    def stat_equil(self):
        Nspace = self.atmos.Nspace

        maxRelChange = 0.0
        for atom in self.activeAtoms:
            Nlevel = atom.Nlevel
            for k in range(Nspace):
                iEliminate = np.argmax(atom.n[:, k])
                Gamma = np.copy(atom.Gamma[:, :, k])

                Gamma[iEliminate, :] = 1.0
                nk = np.zeros(Nlevel)
                nk[iEliminate] = atom.nTotal[k]

                nOld = np.copy(atom.n[:, k])
                nNew = solve(Gamma, nk)
                change = np.abs(1.0 - nOld / nNew)
                maxRelChange = max(maxRelChange, change.max())
                atom.n[:, k] = nNew

        return maxRelChange