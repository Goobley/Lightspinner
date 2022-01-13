import numpy as np
from numba import njit
from dataclasses import dataclass
from utils import planck

@dataclass
class IPsi:
    """Holder for the I and PsiStar arrays in a single object.
    """

    I: np.ndarray
    PsiStar: np.ndarray

@njit
def w2(dtau):
    """Compute the analytic integration factors w0, w1 for the piecewise
    linear short characteristics method.

    Parameters
    ----------
    dtau : float
        Delta tau over the region for which the integration factors are
        needed.

    Returns
    -------
    w : np.ndarray
        A 2 element array containing w0 and w1 -- these are calculated using
        a Taylor series for small dtau, and set to constant values for large
        dtau.
    """

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
    """Formal solver core integration function.

    Compute the one-dimensional (plane parallel) formal solution and
    approximate Psi operator (diagonal) for the specified parameters up or
    down one ray for the source function and opacity at a given frequency.
    This function uses the piecewise linear short characteristics method --
    see [HM pp. 404-408], Auer & Paletou (1994) A&A, 285, 675-686, and Kunasz
    & Auer (1988) JQSRT, 39, 67-79. For the approximate operator see [HM pp.
    436-440] and [RH91/92]. Also see the notes attached to this code. The
    piecewise linear method is somewhat crude, and requires a densely grid in
    tau. Due to the nature of assuming piecewise linear variations of the
    source function between the points this method can overestimate
    substantially in regions where the source function has upwards curvature
    and underestimate similarly in regions where the curvature is downwards.
    This is however the simplest formal solver to implement, and performs
    well enough for many problems.

    Parameters
    ----------
    muz : float
        The cosine of the ray to the normal of the atmospheric slabs.
    toFrom : bool
        Whether the ray is upgoing (towards the observer), or downgoing
        (True/False respectively).
    Istart : float
        The incoming intensity at the boundary where the integration starts.
    z : np.ndarray
        The height grid (1D array).
    chi :  np.ndarray
        The total opacity grid (1D array).
    S :  np.ndarray
        The total source function grid (1D array).

    Returns
    -------
    I : np.ndarray
        The intensity at each point on the grid (1D array).
    PsiStar : np.ndarray
        The approximate Psi operator (diagonal of the Lambda operator) at
        each point on the grid (1D array).
    """

    Nspace = chi.shape[0]
    # NOTE(cmo): Since a smaller mu "increases" the perceived thickness of the slab, the factor we need to use if 1/mu
    zmu = 1.0 / muz

    # NOTE(cmo): It is simplest to set up the looping criterea separately, with dk being the loop step
    if toFrom:
        # NOTE(cmo): Upgoing ray / to observer
        dk = -1
        kStart = Nspace - 1
        kEnd = 0
    else:
        # NOTE(cmo): Downgoing ray / away from observer
        dk = 1
        kStart = 0
        kEnd = Nspace - 1

    # dtau_uw = average opacity          *             slab thickness
    dtau_uw = 0.5 * (chi[kStart] + chi[kStart + dk]) * zmu * np.abs(z[kStart] - z[kStart + dk])
    # NOTE(cmo): dS_uw = dS / dtau i.e. c_1 on slides. Direction is opposite to
    # forward derivative in z as "increases" away from the point at which we're
    # solving -- in all directions.
    dS_uw = (S[kStart] - S[kStart + dk]) / dtau_uw

    Iupw = Istart
    I = np.zeros(Nspace)
    LambdaStar = np.zeros(Nspace)
    # NOTE(cmo): Initial point is equation to boundary condition
    I[kStart] = Iupw
    LambdaStar[kStart] = 0.0

    for k in range(kStart + dk, kEnd, dk):
        # NOTE(cmo): Get analytic integration terms
        w = w2(dtau_uw)

        # NOTE(cmo): Compute I and LambdaStar
        # (1.0 - w[0]) = exp(-dtau) as w[0] as w[0] = 1 - exp(-dtau) and this saves us recomputing the exp
        I[k] = Iupw * (1.0 - w[0]) + w[0] * S[k] + w[1] * dS_uw
        LambdaStar[k] = w[0] - w[1] / dtau_uw
        # NOTE(cmo): dtau_dw and dS_dw like uw for next iteration
        dtau_dw = 0.5 * (chi[k] + chi[k+dk]) * zmu * np.abs(z[k] - z[k+dk])
        dS_dw = (S[k] - S[k+dk]) / dtau_dw

        # NOTE(cmo): Set values (Iupw, dS_uw, dtau_uw) for next iteration
        Iupw = I[k]
        dS_uw = dS_dw
        dtau_uw = dtau_dw

    # NOTE(cmo): Do final point (exactly the same in this linear scheme)
    I[kEnd] = (1.0 - w[0]) * Iupw + w[0] * S[k] + w[1] * dS_uw
    LambdaStar[kEnd] = w[0] - w[1] / dtau_uw

    # NOTE(cmo): Correctly make PsiStar by dividing LambdaStar by chi
    return I, LambdaStar / chi

@njit
def piecewise_linear_1d_single_up(z, muz, chi, S, Iupw, k):
    '''Integrate RTE in the upgoing direction for a single step between k-dk and k
    Linear SC method
    '''
    zmu = 1.0 / muz

    # NOTE(cmo): Upgoing ray / to observer
    dk = -1
    dtau_uw = 0.5 * (chi[k-dk] + chi[k]) * zmu * np.abs(z[k-dk] - z[k])
    dS_uw = (S[k-dk] - S[k]) / dtau_uw
    w = w2(dtau_uw)
    Ik = Iupw * (1.0 - w[0]) + w[0] * S[k] + w[1] * dS_uw
    PsiStark = (w[0] - w[1] / dtau_uw) / chi[k]

    return Ik, PsiStark

def piecewise_linear_1d(atmos, mu, toFrom, wav, chi, S):
    """One-dimensional Piecewise linear formal solver

    Compute the one-dimensional (plane parallel) formal solution and
    approximate Psi operator (diagonal) for the given atmosphere up or
    down one ray for the source function and opacity at a given frequency.
    The radiation boundary conditions assume that the lower boundary is
    thermalised and the upper has no incident radiation.
    This function uses the piecewise linear short characteristics method --
    see [HM pp. 404-408], Auer & Paletou (1994) A&A, 285, 675-686, and Kunasz
    & Auer (1988) JQSRT, 39, 67-79. For the approximate operator see [HM pp.
    436-440] and [RH91/92]. Also see the notes attached to this code. The
    piecewise linear method is somewhat crude, and requires a densely grid in
    tau. Due to the nature of assuming piecewise linear variations of the
    source function between the points this method can overestimate
    substantially in regions where the source function has upwards curvature
    and underestimate similarly in regions where the curvature is downwards.
    This is however the simplest formal solver to implement, and performs
    well enough for many problems.

    Parameters
    ----------
    atmos :  Atmosphere
        The atmosphere object containing the stratification to compute the
        formal solution through.
    mu : int
        The index of the ray to compute the formal solution for in atmos.muz.
    toFrom : bool
        Whether the ray is upgoing (towards the observer), or downgoing
        (True/False respectively).
    wav : float
        The wavelength at which the formal solution is being computed (needed for
        boundary conditions).
    chi :  np.ndarray
        The total opacity for each depth point in the stratified
        atmosphere(1D array).
    S :  np.ndarray
        The total source function grid for each depth point in the stratified
        atmosphere (1D array).

    Returns
    -------
    IPsi
        A dataclass containing the intensity approximate Psi operator at each
        point in the atmosphere.
    """

    zmu = 1.0 / atmos.muz[mu]
    z = atmos.height

    if toFrom:
        dk = -1
        kStart = atmos.Nspace - 1
        kEnd = 0
    else:
        dk = 1
        kStart = 0
        kEnd = atmos.Nspace - 1

    # NOTE(cmo): Set up supper simple boundary conditions -- thermalised for upgoing and zero for downgoing
    if toFrom:
        dtau_uw = zmu * (chi[kStart] + chi[kStart + dk]) * 0.5 * np.abs(z[kStart] - z[kStart + dk])
        Bnu = planck(atmos.temperature[-2:], wav)
        Iupw = Bnu[1] - (Bnu[0] - Bnu[1]) / dtau_uw
    else:
        Iupw = 0.0

    I, PsiStar = piecewise_1d_impl(atmos.muz[mu], toFrom, Iupw, z, chi, S)
    return IPsi(I, PsiStar)