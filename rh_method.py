from dataclasses import dataclass
from enum import Enum, auto
import numpy as np
from scipy.linalg import solve
from typing import Union, Optional, List
from atomic_model import AtomicLine, AtomicContinuum, AtomicModel, VoigtLine
from atmosphere import Atmosphere
from atomic_set import SpectrumConfiguration, AtomicStateTable
from atomic_table import AtomicTable
from formal_solver import piecewise_linear_1d, IPsi
import constants as Const
from background import Background
from numba import njit
from utils import voigt_H
from typing import cast

@dataclass
class UV:
    """A contained for the [RH92]/[U01] Uji, Vij and Vji terms.
    """
    Uji: np.ndarray
    Vij: np.ndarray
    Vji: np.ndarray

class ComputationalTransition:
    """A container to hold the computational state of any transition (line or
    continuum).

    ...
    Attributes
    ----------
    transModel : Union[AtomicLine, AtomicContinuum]
        The model of the transition present in the atomic model.
    atom : ComputationAtom
        The "parent" atom of this transition.
    i : int
        The index of the lower level of this transition in the parent atom.
    j : int
        The index of the upper level of this transition in the parent atom.
    wavelength :  np.ndarray
        The wavelength grid over which this transition is integrated.
    isLine : bool
        True if the transition represents an atomic line.
    Aji : float
        The Einstein A for the transition (lines only).
    Bji : float
        The Einstein Bji for the transition (lines only).
    Bij : float
        The Einstein Bij for the transition (lines only).
    lambda0 : float
        The rest wavelenght for this transition (lines only).
    phi : np.ndarray (4D)
        The line profile in for each [lambda, mu, toFrom, depthPoint] (lines
        only).
    wphi : np.ndarray
        The weighting factor for integrating the line profile s.t.
        \int d\nu d\Omega phi = 1
        at each depth in the atmosphere (lines only).
    alpha : np.ndarray
        The cross section at each wavelength in the transition's wavelength
        grid (continua only).
    Nblue : int
        The index of the bluest (smallest) entry of the line's wavelength
        grid in the global wavelength array.
    active :  np.ndarray of bool
        An array of bools with the same length as the global wavelength array
        indicating the wavelengths when the transition is "active" and should
        be taken considered in the RT calculation.
    gij : np.ndarray
        g_{ij} for each depth point as defined in (26) of [U01], attached to
        this class as a view from the "parent" atom.
    Rij : np.ndarray
        The radiative excitation rate for each depth point.
    Rji : np.ndarray
        The radiative deexcitation rate for each depth point.

    Methods
    -------
    lt(la: int) -> int
        Returns the index into the local wavelength array for a given index
        in the global array. Does no verification.
    wlambda(self, la: Optional[int]=None) -> Union[float, np.ndarray]
        Returns the wavelength integration weight for a single point (if
        index provided), or the whole wavelength grid.
    compute_phi(self, atmos: Atmosphere)
        Computes the line profile and normalisation terms for lines. Does
        nothing for continua.
    uv(self, la, mu, toFrom) -> UV
        Returns the U and V terms from [RH92]/[U01] for the transition at a
        given frequency, ray and direction index.
    """

    def __init__(self, trans: Union[AtomicLine, AtomicContinuum], compAtom: 'ComputationalAtom', atmos: Atmosphere, spect: SpectrumConfiguration):
        """
        Parameters
        ----------
        trans : Union[AtomicLine, AtomicContinuum]
            The transition model from the atomic model object.
        compAtom : ComputationalAtom
            The "parent" atom to this transition.
        atmos :  Atmosphere
            The stratified atmosphere we will be solving the RTE over.
        spect : SpectrumConfiguration
            The object describing the wavelength discretisation and set of
            active transitions at each frequency.
        """
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
        """Map from the index into the global wavelength array to the index
        in the local wavelength array.

        Parameters
        ----------
        la : int
            Index into global wavelength array

        Returns
        -------
        lt : int
            Index into the transition's wavelength array
        """

        return la - self.Nblue

    @property
    def isLine(self):
        """Return if the transition represents a line.
        """

        return isinstance(self.transModel, AtomicLine)

    def wlambda(self, la: Optional[int]=None) -> Union[float, np.ndarray]:
        """Return the wavelength integration weights for the transition.

        For continua the weight is simply \Delta\lambda, whereas for lines we
        choose to use Doppler units (without thermal velocity factor) of c /
        \lambda_0, as this preserves wphi \approx 1.0. The thermal velocity
        factor appears in the line profile.

        Parameters
        ----------
        la : Optional[int]=None
            An index into the transition's wavelength array. If provided then
            only the weight for this index is returned.

        Returns
        -------
        float or np.ndarray
            The wavelength integration weight, either for the requested
            point, or the entire wavelength grid.
        """

        if self.isLine:
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
        """Compute the absorption profile and normalisation coefficient for a
        spectral line.

        This function computes self.phi and self.wphi if the associated
        transition is an atomic line. Here phi is a 4D array [lambda, mu,
        up/down, depth] and wphi an array of the (multiplicative)
        normalisation coefficient per depth point. The normalisation of this
        Voigt profile is slightly different to many but is equivalent to RH &
        Lightweaver in that it isn't normalised to be integrated over
        frequency, but instead in Doppler units and is therefore a factor of
        lambda_0 smaller than is often used. This has an effect on the
        expression used for the U and V coefficients for bound-bound
        transitions.

        Parameters
        ----------
        atmos : Atmosphere
            The stratified atmosphere over which to compute the line profile.
        """

        if isinstance(self.transModel, AtomicContinuum):
            return

        sqrtPi = np.sqrt(np.pi)
        aDamp, Qelast = self.transModel.damping(atmos, self.atom.vBroad, self.atom.hPops.n[0])
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
        """Compute the Uji, Vij and Vji coefficients from [RH92] and [U01]
        for this transition for an wavelength, angle and direction.

        The bound-bound coefficients appear a factor of \lambda_0 larger than
        those presented in the previous papers. This term is encompassed in
        the line profile phi.

        Parameters
        ----------
        la : int
            Index of the current wavelength to be solved (in the global
            wavelength array).
        mu : int
            Index of the current mu (cosine of angle to normal) in the
            atmos.muz array.
        toFrom : bool
            If the desired solution is along the ray towards the observer
            (upgoing/True) or away (downgoing/False).
        """

        lt = self.lt(la)

        hc_4pi = 0.25 * Const.HC / np.pi

        if self.isLine:
            # NOTE(cmo): (2) of [U01] under the assumption of CRD (i.e. phi == psi).
            # Implented as per (26) of [U01]. These appear a factor of
            # \lambda_0 larger than is correct, but this factor is encompassed
            # in phi.
            # The assumption of CRD can be lifted without changing any code
            # here by defining gij as gi/gj * rhoPrd.
            # We use the Einstein relation Aji/Bji = (2h*nu**3) / c**2
            phi = self.phi[lt, mu, int(toFrom), :]
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
    """Stores the state of an atom and its transitions during iteration.

    ...
    Attributes
    ----------
    atomicModel : AtomicModel
        The complete model atom.
    atomicTable : AtomicTable
        The atomic table (with correct abundances etc.) to be used in the
        calculations.
    spect : SpectrumConfiguration
        The object describing the wavelength discretisation and set of
        active transitions at each frequency.
    atmos : Atmosphere
        The stratified atmosphere on which the RTE and ESE are to be solved.
    vBroad : np.ndarray
        The broadening velocity of the atom at each depth point in the
        atmosphere (thermal and microturbulence).
    pops : AtomicState
        The populations object for this atom from the AtomicStateTable.
    hPops : AtomicState
        The populations object for H from the AtomicStateTable.
    nStar : np.ndarray (2D)
        The LTE populations for this species [level, depth].
    n : np.ndarray (2D)
        The NLTE populations for this species [level, depth].
    nTotal : np.ndarray
        The total population of this species per depth point.
    trans : List[ComputationalTransition]
        List of objects used to hold the computational state of each
        transition.
    Ntrans : int
        Number of transitions in atomic model over the wavelength range
        defined in spect.
    Nlevel : int
        Number of levels in atomic model.
    Gamma : np.ndarray (3D)
        RH Gamma matrix for atom [level, level, depth].
    C : np.ndarray (3D)
        RH-style C matrix for atom [level, level, depth]. Collisional rates
        are loaded into this matrix in "transposed" form, i.e. Cij in C[j,
        i], so that it can be added directly to Gamma.
    eta : np.ndarray
        Scratch array for storing the emissivity in all of the transitions of
        an atom in one iteration of the formal solver.
    gij : np.ndarray (2D)
        Array for storing gij per transition [trans, depth] as described by
        (26) of [U01].
    wla : np.ndarray (2D)
        Array for storing the "d\nu"/"dwavelength" integration weights per
        transition [trans, depth].
    U : np.ndarray
        Scratch array for storing the \sum_{ji}U_{ji} term that appears in
        the MALI method (per level) [level, depth].
    chi : np.ndarray
        Scratch array for storing the effective opacity term that appears in
        the MALI method (per level) [level, depth].

    Methods
    -------
    setup_wavelength(self, laIdx: int)
        Configure the scratch matrices for the coming wavelength (of index
        laIdx in the global array) and precalculate the terms that are
        constant over angle (gij & wla).
    zero_angle_dependent_vars(self)
        Zero the scratch matrices that are needed for afresh for every
        directional formal solution whilst computing the terms for the Gamma
        matrix (eta, U, and chi).
    setup_Gamma(self)
        Zero the Gamma matrix.
    compute_collisions(self)
        Compute the collisional rates from the AtomicModel and Atmosphere,
        and store the results in their transposed form in C.
    """

    def __init__(self, atom: AtomicModel, atmos: Atmosphere, spect: SpectrumConfiguration, eqPops: AtomicStateTable):
        """
        Parameters
        ----------
        atom : AtomicModel
            The atomic model for which this object is being constructed to
            hold its computation data during iteration of the NLTE
            populations.
        atmos : Atmosphere
            The atmosphere over which a NLTE solution is to be found.
        spect : SpectrumConfiguration
            The object describing the wavelength discretisation and set of
            active transitions at each frequency.
        eqPops : AtomicStateTable
            The LTE (and possibly NLTE) populations of each species present
            in the atmosphere that is being considered in detail. If the NLTE
            populations are set for this atom then the NLTE computational
            populations will start there, otherwise the system will start
            from LTE populations.
        """

        self.atomicModel = atom
        self.atomicTable = eqPops.atomicTable
        self.spect = spect
        self.atmos = atmos

        self.vBroad = atom.v_broad(atmos)

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
        # NOTE(cmo): if NLTE populations are specified then use them
        if self.pops.pops is not None:
            self.n = self.pops.pops
        else:
            self.n = np.copy(self.nStar)
            self.pops.pops = self.n

        self.Nlevel = Nlevel
        self.Ntrans = len(self.trans)

        # NOTE(cmo): Call setup wavelength so that all of the transitions
        # correctly have all of their arrays initialised
        self.setup_wavelength(0)

    def setup_wavelength(self, laIdx: int):
        """Configure the scratch matrices for the coming wavelength (of index
        laIdx in the global array) and precalculate the terms that are
        constant over angle (gij & wla).
        """

        Nspace = self.atmos.Nspace
        self.eta = np.zeros(Nspace)
        self.gij = np.zeros((self.Ntrans, Nspace))
        self.wla = np.zeros((self.Ntrans, Nspace))
        self.U = np.zeros((self.Nlevel, Nspace))
        self.chi = np.zeros((self.Nlevel, Nspace))

        hc_k = Const.HC / (Const.KBoltzmann * Const.NM_TO_M)
        for kr, t in enumerate(self.trans):
            t.gij = self.gij[kr]
            if not t.active[laIdx]:
                continue

            # NOTE(cmo): gij for transition kr following (26) in [U01]
            # NOTE(cmo): wla for transition kr: weighting term in wavelength integral for transition
            if t.isLine:
                # NOTE(cmo): Like for U and V, these rates also appear to be
                # off by a factor of \lambda, but aren't due to the use Doppler
                # units (inside wlambda).
                self.gij[kr, :] = t.Bji / t.Bij
                self.wla[kr, :] = t.wlambda(t.lt(laIdx)) * t.wphi / Const.HC
            else:
                self.gij[kr, :] = self.nStar[t.i] / self.nStar[t.j] \
                    * np.exp(-hc_k / self.spect.wavelength[laIdx] / self.atmos.temperature)
                self.wla[kr, :] = t.wlambda(t.lt(laIdx)) / self.spect.wavelength[laIdx] / Const.HPlanck

    def zero_angle_dependent_vars(self):
        """
        Zero the scratch matrices that are needed for afresh for every
        directional formal solution whilst computing the terms for the Gamma
        matrix (eta, U, and chi).
        """

        self.eta.fill(0.0)
        self.U.fill(0.0)
        self.chi.fill(0.0)

    def setup_Gamma(self):
        """Zero the Gamma matrix.
        """

        self.Gamma.fill(0.0)

    def compute_collisions(self):
        """Compute the collisional rates from the AtomicModel and Atmosphere,
        and store the results in their transposed form in C.
        """

        self.C = np.zeros_like(self.Gamma)
        # NOTE(cmo): Get colllisional rates from each term on the atomic model.
        # They are added to the correct location in the C matrix so it can be added directly to Gamma.
        # i.e. The rate from j to i is put into C[i, j]
        for col in self.atomicModel.collisions:
            col.compute_rates(self.atmos, self.nStar, self.C)
        # NOTE(cmo): Some rates use spline interpolants that can, in odd cases,
        # go negative, so make sure that doesn't happen
        self.C[self.C < 0.0] = 0.0


class Context:
    """Context to adminster the formal solution and compute the Gamma matrix
    for a collection of active atoms over a provided set of wavelengths.

    ...
    Attributes
    ----------
    atmos : Atmosphere
        The stratified atmosphere on which the RTE and ESE are to be solved.
    spect : SpectrumConfiguration
        The object containing the atomic models and wavelength
        discretisation and set of active transitions at each frequency.
    background : Background
        An object containing the background opacity (chi), emissivity
        (eta) and scattering (sca) for each wavelength to be solved for
        in spect (and depth in the atmosphere).
    eqPops : AtomicStateTable
        The LTE (and possibly NLTE) populations of each species present
        in the atmosphere that is being considered in detail. If the NLTE
        populations are set for this atom then the NLTE computational
        populations will start there, otherwise the system will start
        from LTE populations.
    activeAtoms : List[ComputationalAtom]
        The computational data for the atoms that are being for which the ESE
        and RTE are being solved to find their NLTE populations.
    I : np.ndarray (2D)
        The outgoing radiation as per wavelength point and ray [lambda, mu].
    J : np.ndarray (2D)
        The angle averaged intensity at each wavelength and depth [lambda,
        depth].

    Methods
    -------
    formal_sol_gamma_matrices(self)
        Perform the formal solution over all specified wavelengths and angles
        and fill in the Gamma matrix.
    stat_equil(self)
        Update the populations of all active species towards statistical
        equilibrium, using the current version of the Gamma matrix.
    """

    def __init__(self, atmos: Atmosphere, spect: SpectrumConfiguration, 
                 eqPops: AtomicStateTable, background: Background, 
                 formalSolver: str='piecewise'):
        """
        Parameters
        ----------
        atmos : Atmosphere
            The stratified atmosphere on which the RTE and ESE are to be solved.
        spect : SpectrumConfiguration
            The object containing the atomic models and wavelength
            discretisation and set of active transitions at each frequency.
        eqPops : AtomicStateTable
            The LTE (and possibly NLTE) populations of each species present
            in the atmosphere that is being considered in detail. If the NLTE
            populations are set for this atom then the NLTE computational
            populations will start there, otherwise the system will start
            from LTE populations.
        background : Background
            An object containing the background opacity (chi), emissivity
            (eta) and scattering (sca) for each wavelength to be solved for
            in spect (and depth in the atmosphere).
        """

        self.atmos = atmos
        self.atmos.nondimensionalise()
        self.spect = spect
        self.background = background
        self.eqPops = eqPops
        self.formalSolver = formalSolver

        self.activeAtoms: List[ComputationalAtom] = []
        for a in spect.radSet.activeAtoms:
            self.activeAtoms.append(ComputationalAtom(a, atmos, spect, eqPops))

        self.J = np.zeros((spect.wavelength.shape[0], atmos.Nspace))
        self.I = np.zeros((spect.wavelength.shape[0], atmos.Nrays))

    def formal_sol_gamma_matrices(self):
        """Perform the formal solution over all specified wavelengths and angles
        and fill in the Gamma matrix.

        This function applies the method of [RH92] and [U01] to construct the
        Gamma matrix for the given, possibly overlapping, transitions on the
        given common wavelength grid, with specified angle quadrature for a
        plane-parallel atmosphere.

        Returns
        -------
        dJ : float
            The maximum relative update of the local angle-averaged intensity J
        """

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
                    iPsi = piecewise_linear_1d(self.atmos, mu, toFrom, wav, chiTot, S, 
                                               method=self.formalSolver)
                    # NOTE(cmo): Save outgoing intensity -- this is done for both ingoing and outgoing rays,
                    # but the outgoing rays happen after so we can simply save it every subiteration
                    self.I[la, mu] = iPsi.I[0]
                    # NOTE(cmo): Add contribution to local mean intensity field Jbar
                    self.J[la] += 0.5 * self.atmos.wmu[mu] * iPsi.I

                    # NOTE(cmo): Construct Gamma matrix for each atom
                    for atom in self.activeAtoms:
                        # NOTE(cmo): Compute Ieff as per (20) in [U01], or (2.20) in [RH92] using
                        # \sum_j \sum_{i<j} n_j\dagger U_{ji}\dagger is simply
                        # the sum of \eta in the currently active transitions in an atom.
                        # I find (2.20) of RH92 a little confusing here, as I
                        # believe they are working on the assumption of only
                        # one active species, with possible overlapping
                        # transitions. To me, the expression in [U01] is
                        # clearer.
                        Ieff = iPsi.I - iPsi.PsiStar * atom.eta

                        for kr, t in enumerate(atom.trans):
                            if not t.active[la]:
                                continue

                            # NOTE(cmo): wlamu is the product of angle and
                            # wavelength integration weights (*0.5 for up/down
                            # component)
                            wmu = 0.5 * self.atmos.wmu[mu]
                            # NOTE(cmo): The 4pi term here represents the
                            # factor for integrating over all solid angles,
                            # rather than simply averaging.
                            wlamu = atom.wla[kr] * wmu * 4 * np.pi

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
                            # \Gamma \cdot \vec{n} = \vec{0} i.e. statistical equilbirum.

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
        """Update the populations of all active species towards statistical
        equilibrium, using the current version of the Gamma matrix.

        Returns
        -------
        maxRelChange : float
            The maximum relative change in any of the atomic populations (at
            the depth point with maximum population change).
        """
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