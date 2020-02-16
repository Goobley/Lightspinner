from dataclasses import dataclass
from enum import Enum, auto
import numpy as np
from scipy.linalg import solve
from typing import Union, Optional, List
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
    def __init__(self, trans: Union[AtomicLine, AtomicContinuum], compAtom: 'ComputationalAtom', atmos: Atmosphere, spect: SpectrumConfiguration):
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

        self.gij = None
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
            # NOTE(cmo): (2) of [U01] under the assumption of CRD (i.e. phi == psi).
            # Implented as per (26) of [U01].
            # However, the assumption of CRD can be lifted without changing any code.
            # here by defining gij as gi/gj * rhoPrd.
            # We use the Einstein relation Aji/Bji = (2h*nu**3) / c**2
            phi = self.phi[lt, mu, toFrom, :]
            Vij = hc_4pi * self.Bij * phi
            Vji = self.gij * Vij
            Uji = self.Aji / self.Bji * Vji
        else:
            # NOTE(cmo): (3) of [U01] using the expression for gij given in (26)
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

        self.trans: List[ComputationalTransition] = []
        for l in atom.lines:
            if l in spect.transitions:
                self.trans.append(ComputationalTransition(l, self, atmos, spect))

        for c in atom.continua:
            if c in spect.transitions:
                self.trans.append(ComputationalTransition(c, self, atmos, spect))

        Nlevel = len(atom.levels)
        self.Gamma = np.zeros((Nlevel, Nlevel, atmos.Nspace))
        self.C = np.zeros((Nlevel, Nlevel, atmos.Nspace))
        self.nStar = self.pops.nStar
        self.n = np.copy(self.nStar)

        self.Nlevel = Nlevel
        self.Ntrans = len(self.trans)

    def setup_wavelength(self, laIdx: int):
        Nspace = self.atmos.Nspace
        self.eta = np.zeros(Nspace)
        self.gij = np.zeros((self.Ntrans, Nspace))
        self.wla = np.zeros((self.Ntrans, Nspace))
        self.U = np.zeros((self.Nlevel, Nspace))
        self.chi = np.zeros((self.Nlevel, Nspace))

        hc_k = Const.HC / (Const.KBoltzmann * Const.NM_TO_M)
        h_4pi = 0.25 * Const.HPlanck / np.pi
        hc_4pi = h_4pi * Const.CLight
        for kr, t in enumerate(self.trans):
            if not t.active[laIdx]:
                continue

            # NOTE(cmo): gij for transition kr following (26) in [U01]
            # NOTE(cmo): wla for transition kr: weighting term in wavelength integral for transition
            if isinstance(t.transModel, AtomicLine):
                self.gij[kr, :] = t.Bji / t.Bij
                self.wla[kr, :] = t.wlambda(t.lt(laIdx)) * t.wphi / hc_4pi
            else:
                self.gij[kr, :] = self.nStar[t.i] / self.nStar[t.j] \
                    * np.exp(-hc_k / self.spect.wavelength[laIdx] / self.atmos.temperature)
                self.wla[kr, :] = t.wlambda(t.lt(laIdx)) / self.spect.wavelength[laIdx] / h_4pi
            t.gij = self.gij[kr]

    def zero_angle_dependent_vars(self):
        self.eta.fill(0.0)
        self.U.fill(0.0)
        self.chi.fill(0.0)

    def setup_Gamma(self):
        self.Gamma.fill(0.0)

    def compute_collisions(self):
        self.C = np.zeros_like(self.Gamma)
        # NOTE(cmo): Get colllisional rates from each term on the atomic model.
        # They are added to the correct location in the C matrix so it can be added directly to Gamma. 
        # i.e. A rate from j to i is put into C[i, j]
        for col in self.atomicModel.collisions:
            col.compute_rates(self.atmos, self.nStar, self.C)
        # NOTE(cmo): Some rates use splines that can, in odd cases, go negative, so make sure that doesn't happen
        self.C[self.C < 0.0] = 0.0

@dataclass
class IPsi:
    I: np.ndarray
    PsiStar: np.ndarray

@njit
def w2(dtau):
    w = np.empty(2)
    if dtau < 5e-4:
        w[0] = dtau * (1.0 - 0.5*dtau)
        w[1] = dtau**2 * (0.5 - dtau / 3.0)
    elif dtau > 50.0:
        w[0] = 1.0
        w[1] = 1.0
    else:
        expdt = np.exp(-dtau)
        w[0] = 1.0 - expdt
        w[1] = w[0] - dtau * expdt
    return w

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
    I = np.zeros(Nspace)
    Psi = np.zeros(Nspace)
    I[kStart] = Iupw
    Psi[kStart] = 0.0

    for k in range(kStart + dk, kEnd + dk, dk):
        w = w2(dtau_uw)

        if k != kEnd:
            dtau_dw = zmu * (chi[k] + chi[k+dk]) * np.abs(z[k] - z[k+dk])
            dS_dw = (S[k] - S[k+dk]) / dtau_dw
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

    if toFrom:
        dtau_uw = zmu * (chi[kStart] + chi[kStart + dk]) * np.abs(z[kStart] - z[kStart + dk])
        Bnu = planck(atmos.temperature[-2:], wav)
        Iupw = Bnu[1] - (Bnu[0] - Bnu[1]) / dtau_uw
    else:
        Iupw = 0.0

    I, Psi = piecewise_1d_impl(atmos.muz[mu], toFrom, Iupw, z, chi, S)
    return IPsi(I, Psi)

class Context:
    def __init__(self, atmos: Atmosphere, spect: SpectrumConfiguration, eqPops: AtomicStateTable, background: Background):
        self.atmos = atmos
        self.atmos.nondimensionalise()
        self.spect = spect
        self.background = background
        self.eqPops = eqPops

        self.activeAtoms: List[ComputationalAtom] = []
        for a in spect.radSet.activeAtoms:
            self.activeAtoms.append(ComputationalAtom(a, atmos, spect, eqPops))

        self.J = np.zeros((spect.wavelength.shape[0], atmos.Nspace))
        self.I = np.zeros((spect.wavelength.shape[0], atmos.Nrays))

    def formal_sol_gamma_matrices(self):
        Nspace = self.atmos.Nspace
        Nrays = self.atmos.Nrays
        Nspect = self.spect.wavelength.shape[0]
        background = self.background

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
                            # NOTE(cmo): Compute opacity and emissivity in transition t
                            # Equation (1) in [U01]
                            chi = atom.n[t.i] * uv.Vij - atom.n[t.j] * uv.Vji
                            eta = atom.n[t.j] * uv.Uji

                            # NOTE(cmo): Add the opacity to the lower level
                            # Subtract the opacity from the upper level (stimulated emission)
                            # The atom.chi matrix is then simply the total "effective opacity" for each atomic level
                            atom.chi[t.i] += chi
                            atom.chi[t.j] -= chi
                            # NOTE(cmo): Accumulate U, this is like the chi matrix
                            atom.U[t.j] += uv.Uji
                            # NOTE(cmo): Accumulate onto total opacity and emissivity 
                            # as well as total emissivity in the atom -- needed for Ieff
                            chiTot += chi
                            etaTot += eta
                            atom.eta += eta

                    # NOTE(cmo): Accumulate background opacity
                    chiTot += background.chi[la]
                    # NOTE(cmo): Compute source function
                    S = (etaTot + background.eta[la] + background.sca[la] * JDag[la]) / chiTot

                    # NOTE(cmo): Compute formal solution and approximate operator PsiStar
                    iPsi = piecewise_linear_1d(self.atmos, mu, toFrom, wav, chiTot, S)
                    # NOTE(cmo): Save outgoing intensity -- this is done for both ingoing and outgoing rays, 
                    # but the outgoing rays happen after so we can simply save it every subiteration
                    self.I[la, mu] = iPsi.I[0]
                    # NOTE(cmo): Add contribution to local mean intensity field Jbar
                    self.J[la] += 0.5 * self.atmos.wmu[mu] * iPsi.I

                    # NOTE(cmo): Construct Gamma matrix for each atom
                    for atom in self.activeAtoms:
                        # NOTE(cmo): Compute Ieff as per (20) in [U01], or (2.20) in [RH92] using 
                        # \sum_j \sum_{i<j} n_j\dagger U_{ji}\dagger is simply 
                        # the sum of \eta in the currently active transitions in an atom
                        Ieff = iPsi.I - iPsi.PsiStar * atom.eta

                        for kr, t in enumerate(atom.trans):
                            if not t.active[la]:
                                continue

                            # NOTE(cmo): wlamu is the product of angle and wavelength integration weights (*0.5 for up/down)
                            wmu = 0.5 * self.atmos.wmu[mu]
                            wlamu = atom.wla[kr] * wmu

                            uv = t.uv(la, mu, toFrom)

                            # NOTE(cmo): Add the contributions to the Gamma matrix
                            # Follow (2.19) in [RH92] or (24) in [U01] -- we use the properties
                            # of the Gamma matrix to construct the off-diagonal components later 
                            # and can therefore ignore all terms with a Kronecker delta.

                            # NOTE(cmo): The accumulated chi and U matrices per atom are
                            # extremely handy here, as they essentially serve as all of the
                            # bookkeeping for filling Gamma
                            integrand = (uv.Uji + uv.Vji * Ieff) - (atom.chi[t.i] * iPsi.PsiStar * atom.U[t.j])
                            atom.Gamma[t.i, t.j] += integrand * wlamu

                            integrand = (uv.Vij * Ieff) - (atom.chi[t.j] * iPsi.PsiStar * atom.U[t.i])
                            atom.Gamma[t.j, t.i] += integrand * wlamu
                            # NOTE(cmo): Compare equations (2.16) and (2.19) in [RH92] -- note how all non-dagger terms are of
                            # course gone from (2.19) since the new populations are evaluated when in the matrix solution, and
                            # the n_l\daggers in the _critical summations_ (the final terms on both sides of (2.16)) -- the
                            # populations we preserve from the previous iteration are already wrapped up in the atom.chi
                            # term, and all other terms will be multiplied by their associated new populations when we do
                            # \Gamma \cdot \vec{n} = \vec{0}.

                            # NOTE(cmo): Radiative rates component for this angle and frequency
                            # as per (28) of [U01] or (2.8) or [RH92]
                            t.Rij += iPsi.I * uv.Vij * wlamu
                            t.Rji += (uv.Uji + iPsi.I * uv.Vij) * wlamu

        # NOTE(cmo): "Finish" constructing the Gamma matrices by computing the diagonal.
        # Looking the contributions to the Gamma matrix that contain Kronecker deltas and using U_{i,i} = V_{i,i} = 0,
        # we have \sum_l \Gamma_{l l\prime} = 0, and these terms are simply the additive inverses of the sums 
        # of every other entry on their column in Gamma.
        for atom in activeAtoms:
            for k in range(Nspace):
                np.fill_diagonal(atom.Gamma[:, :, k], 0.0)
                for i in range(atom.Nlevel):
                    GamDiag = np.sum(atom.Gamma[:, i, k])
                    atom.Gamma[i, i, k] = -GamDiag

        dJ = np.abs(1.0 - JDag / self.J)
        dJMax = dJ.max()

        return dJMax

    def stat_equil(self):
        Nspace = self.atmos.Nspace

        maxRelChange = 0.0
        for atom in self.activeAtoms:
            Nlevel = atom.Nlevel
            for k in range(Nspace):
                # NOTE(cmo): Find the level with the maximum population at this depth point
                iEliminate = np.argmax(atom.n[:, k])
                # NOTE(cmo): Copy the Gamma matrix so we can modify it to contain the total number conservation equation
                Gamma = np.copy(atom.Gamma[:, :, k])

                # NOTE(cmo): Set all entries on the row to eliminate to 1.0 for number conservation
                Gamma[iEliminate, :] = 1.0
                # NOTE(cmo): Set solution vector to 0 (as per stat. eq.) other than entry for which we are conserving population
                nk = np.zeros(Nlevel)
                nk[iEliminate] = atom.nTotal[k]

                # NOTE(cmo): Solve Gamma . n = 0 (constrained by conservation equation)
                nOld = np.copy(atom.n[:, k])
                nNew = solve(Gamma, nk)
                # NOTE(cmo): Compute relative change and update populations
                change = np.abs(1.0 - nOld / nNew)
                maxRelChange = max(maxRelChange, change.max())
                atom.n[:, k] = nNew

        return maxRelChange