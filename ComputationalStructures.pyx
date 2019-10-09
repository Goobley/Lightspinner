from AtomicModel import AtomicModel, LineType, AtomicContinuum, AtomicLine, ExplicitContinuum, HydrogenicContinuum
import Constants as Const

from copy import deepcopy
import numpy as np
from numpy.polynomial.legendre import leggauss
cimport numpy as np
from libc.stdlib cimport malloc, calloc, free

ctypedef int bool_t
ctypedef void Ng
ctypedef void RLK_Line
ctypedef void FixedTransition

cdef extern from "../RhCoreData.h":
    struct RhAtomicLine:
        bool_t polarizable
        bool_t PRD

        int i
        int j
        int Nlambda
        int Nblue
        int Ncomponent
        int Nxrd
        int nLine

        double lambda0
        double* wavelength
        double g_Lande_eff
        double Aji
        double Bji
        double Bij
        double* Rij
        double* Rji
        double** phi
        double** phi_Q
        double** phi_U
        double** phi_V
        double** psi_Q
        double** psi_U
        double** psi_V
        double* wphi
        double* Qelast
        double Grad
        # double qcore
        # double qwing
        double** rho_prd
        double* c_shift
        double* c_fraction
        double** gII
        int** id0
        int** id1
        double** frac
        Ng* Ng_prd
        Atom* atom
        RhAtomicLine** xrd
    
    struct RhAtomicContinuum:
        bool_t hydrogenic
        int i
        int j
        int Nlambda
        int Nblue
        int nCont
        double lambda0
        double* wavelength
        double alpha0
        double* alpha
        double* Rij
        double* Rji
        Atom *atom

    struct RhAccumulate:
        double** gij
        double** Vij
        double** wla
        double** chi_up
        double** chi_down
        double** Uji_down
        double*  eta
        double** Gamma
        double** RjiLine
        double** RijLine
        double** RjiCont
        double** RijCont
        bool_t* lineRatesDirty

    struct Atom:
        bool_t active
        bool_t NLTEpops
        int Nlevel
        int Nline
        int Ncont
        int Nfixed
        int Nprd
        int* stage
        int activeIndex
        double*  g
        double*  E
        double** C
        double*  vbroad
        double** n
        double** nstar
        double*  ntotal
        double** Gamma
        RhAtomicLine *line
        RhAtomicContinuum *continuum
        FixedTransition *ft
        Ng* Ng_n
        RhAccumulate* accumulate

    enum angleset:
        SET_VERTICAL
        SET_GL
        SET_A2
        SET_A4
        SET_A6
        SET_A8
        SET_B4
        SET_B6
        SET_B8
        NO_SET

    struct AngleSet:
        angleset set
        int Ninclination, Nazimuth

    struct Atmosphere:
        bool_t H_LTE
        bool_t Stokes
        bool_t hydrostatic
        int Ndim
        int* N
        int Nspace
        int Nrays
        int Natom
        int Nmolecule
        int NPRDactive
        int Nactiveatom
        int Nactivemol
        double* T
        double* ne
        double* vturb
        double** v_los # // Moved from spectrum because it feels out of place -- Should probably go to Geometry though
        double gravity
        double vmicro_char
        double lambda_ref
        double* wmu
        double* nHtot
#   double** nH; // nH is only used during atmospheric setup -- can probably go.
        double* nHmin
        double* B
        double* gamma_B
        double* chi_B
        double B_char
        double** cos_gamma
        double** cos_2chi
        double** sin_2chi
        AngleSet angleSet
        Atom* H
        Atom** atoms
        Atom** activeatoms
        Molecule* H2
        Molecule* OH
        Molecule* CH
        Molecule** molecules
        Molecule** activemols
        AtomicTable* atomicTable

    struct Element:
        char ID[3]
        int abundance_set
        int Nstage
        # int* mol_index # Can actually chuck mol_index and Nmolecule -- they're only used in the informational printout at the end of ChemEq
        # int Nmolecule
        double weight
        double abund
        double* ionpot
        double** pf
        double** n    
    
    struct AtomicTable:
        double* Tpf
        Element* elements
        int Nelem
        int Npf
        double totalAbund
        double avgMolWeight
        double weightPerH

    enum boundcond:
        ZERO
        THERMALIZED
        IRRADIATED
        REFLECTIVE

    enum mass_scale:
        GEOMETRIC
        COLUMN_MASS
        TAU500

    enum vertical:
        TOP
        BOTTOM

    struct Geometry:
        mass_scale scale
        boundcond vboundary[2]
        int Ndep
        int Nrays
        double* height
        double* cmass
        double* tau_ref
        double* mux
        double* muy
        double* muz
        double* wmu
        double* vel
        double** Itop
        double** Ibottom

    struct Molecule:
        # char ID[MOLECULE_ID_WIDTH + 1], *popsFile, *configs;
        bool_t active
        # int* pt_index
        # int* pt_count
        # int Nelement
        # int Nnuclei
        # int Npf
        # int Neqc
        int Nrt
        # int charge
        int Nconfig,
        int Nv
        int NJ
        int activeindex
        double Ediss
        # double Tmin
        # double Tmax
        # double weight
        double* vbroad
        double* pf_coef
        double* eqc_coef
        double* pf
        double** pfv,
        double* n
        double** nv
        double** nvstar
        double* C_ul
        double** Gamma
        MolecularLine* mrt
        Ng* Ng_nv
        # rhthread* rhth
        RhAccumulate* rhAcc

    struct ZeemanMultiplet:
        int Ncomponent
        int* q
        double* shift
        double* strength

    enum type:
        ATOMIC_LINE
        ATOMIC_CONTINUUM
        VIBRATION_ROTATION
        MOLECULAR_ELECTRONIC

    enum Hund:
        CASE_A
        CASE_B

    struct MolecularLine:
        type type
        Hund Hundi
        Hund Hundj
        bool_t symmetric
        bool_t Voigt
        bool_t polarizable
        char configi[3]
        char configj[3]
        char parityi[2]
        char parityj[2]
        int vi
        int vj
        int Nlambda
        int Nblue
        int subi
        int subj
        int Lambdai
        int Lambdaj
        int ecnoi,
        int ecnoj
        double lambda0
        double* wavelength
        double isotope_frac
        double Ei
        double Ej
        double gi
        double gj
        double Si
        double Sj
        double Omegai
        double Omegaj,
        double** phi
        double* wphi
        double g_Lande_eff
        double Grad
        double qcore
        double qwing
        double Aji
        double Bji
        double Bij
        Molecule* molecule
        ZeemanMultiplet* zm


    struct Background:
        int Nspect
        int Nray
        int Ndir
        int Nrlk
        RLK_Line* rlk_lines
        bool_t polarized
        double** chi
        double** eta
        double** sca
        double** chip

    struct DepthData:
        double** IDepth
        double** SDepth
        double** chiDepth

    struct Options:
        pass

    union LineOrCont:
        RhAtomicLine* line
        RhAtomicContinuum* continuum

    union MolLine:
        MolecularLine* vrline

    struct AtomicTransition:
        type type
        LineOrCont ptype
        Atom *atom

    struct MolTransition:
        type type
        MolLine ptype
        Molecule *molecule

    struct ActiveSet:
        int* Nactiveatomrt
        int* Nactivemolrt
        int* Nlower
        int* Nupper
        int** lower_levels
        int** upper_levels
        double* chi
        double* eta
        double* chip
        AtomicTransition** art
        MolTransition** mrt

    struct Spectrum:
        bool_t updateJ
        int Nspect
        int* PRDindex
        double* wavelength
        double** J
        double** I
        double** Stokes_Q
        double** Stokes_U
        double** Stokes_V
        double** J20
        double** Jgas 
        ActiveSet* aset
        int* nc
        int* iprdh
        double* cprdh

    struct RhContext:
        Atmosphere* atmos
        Geometry* geo
        Spectrum* spectrum
        DepthData* depth
        Background* background
        Options* options



cdef convert_element(ele, Element* cEle):
    name = ele.name.encode()
    cEle.ID[0] = name[0]
    cEle.ID[1] = name[1]
    cEle.ID[2] = '\0'

    # cEle.Nmolecule = 0
    # cEle.mol_index = NULL
    cEle.abundance_set = True
    cEle.weight = ele.weight
    cEle.abund = ele.abundance

    ele.ionpot = np.ascontiguousarray(ele.ionpot)
    cdef np.ndarray[np.double_t, ndim=1] ionpot = np.ascontiguousarray(ele.ionpot)
    cEle.ionpot = &ionpot[0]

    cEle.Nstage = ele.pf.shape[0] 

    ele.pf = np.ascontiguousarray(ele.pf)
    cdef np.ndarray[np.double_t, ndim=2] pf = np.ascontiguousarray(ele.pf)
    cEle.pf = <double**> malloc(cEle.Nstage * sizeof(double*))
    for i in range(cEle.Nstage):
        cEle.pf[i] = <double*> &pf[i, 0]

    cEle.n = NULL


cdef class CAtomicTable:
    cdef AtomicTable* atomicTable

    def __init__(self, table):
        # TODO(cmo): This should actually take a LtePopulations, rather than an AtomicTable, so it can set the LtePops
        self.atomicTable = <AtomicTable*> malloc(sizeof(AtomicTable))
        self.atomicTable.totalAbund = table.totalAbundance
        self.atomicTable.weightPerH = table.weightPerH
        self.atomicTable.avgMolWeight = table.avgMolWeight

        cdef int Npf = len(table.Tpf)
        table.Tpf = np.ascontiguousarray(table.Tpf)
        cdef np.ndarray[np.double_t, ndim=1] Tpf = np.ascontiguousarray(table.Tpf)
        self.atomicTable.Tpf = &Tpf[0]
        self.atomicTable.Npf = Npf

        cdef int Nelem = len(table.elements)
        self.atomicTable.elements = <Element*> malloc(Nelem * sizeof(Element))
        self.atomicTable.Nelem = Nelem

        cdef Element* cEle
        for i, ele in enumerate(table.elements):
            cEle = &self.atomicTable.elements[i]
            convert_element(ele, cEle)

    def __dealloc__(self):
        for i in range(self.atomicTable.Nelem):
            free(self.atomicTable.elements[i].pf)
            free(self.atomicTable.elements[i].n)
        free(self.atomicTable.elements)
        free(self.atomicTable)
 
cdef init_atom(Atom* atom):
    atom.active = False
    atom.NLTEpops = False
    atom.Nlevel = 0
    atom.Nline = 0
    atom.Ncont = 0
    atom.Nfixed = 0
    atom.Nprd = 0
    atom.stage = NULL
    atom.activeIndex = 0

    atom.g = NULL
    atom.E = NULL

    atom.C = NULL
    atom.vbroad = NULL
    atom.n = NULL
    atom.nstar = NULL
    atom.Gamma = NULL
    atom.line = NULL
    atom.continuum = NULL
    atom.ft = NULL
    atom.Ng_n = NULL
    atom.accumulate = NULL

cdef init_atomic_line(RhAtomicLine* line):
    line.polarizable = False
    line.PRD = False

    line.i = 0
    line.j = 0
    line.Nlambda = 0
    line.Nblue = 0
    line.Ncomponent = 0
    line.Nxrd = 0
    line.nLine = 0

    line.lambda0 = 0.0
    line.wavelength = NULL
    line.g_Lande_eff = 0.0
    line.Aji = 0.0
    line.Bji = 0.0
    line.Bij = 0.0
    line.Rij = NULL
    line.Rji = NULL
    line.phi = NULL
    line.phi_Q = NULL
    line.phi_U = NULL
    line.phi_V = NULL
    line.psi_Q = NULL
    line.psi_U = NULL
    line.psi_V = NULL
    line.wphi = NULL
    line.Qelast = NULL
    line.Grad = 0.0
    # line.qcore = 0.0
    # line.qwing = 0.0
    line.rho_prd = NULL
    line.c_shift = NULL
    line.c_fraction = NULL
    line.gII = NULL
    line.id0 = NULL
    line.id1 = NULL
    line.frac = NULL
    line.Ng_prd = NULL
    line.atom = NULL
    line.xrd = NULL


cdef init_atomic_continuum(RhAtomicContinuum* cont):
    cont.hydrogenic = True
    cont.i = 0
    cont.j = 0
    cont.Nlambda = 0
    cont.Nblue = 0
    cont.nCont = 0
    cont.lambda0 = 0.0
    cont.wavelength = NULL
    cont.alpha0 = 0.0
    cont.alpha = NULL
    cont.Rij = NULL
    cont.Rji = NULL
    cont.atom = NULL

cdef init_accumulate(RhAccumulate* acc):
    acc.gij = NULL
    acc.Vij = NULL
    acc.wla = NULL
    acc.chi_up = NULL
    acc.chi_down = NULL
    acc.Uji_down = NULL
    acc.eta = NULL
    acc.Gamma = NULL
    acc.RjiLine = NULL
    acc.RijLine = NULL
    acc.RjiCont = NULL
    acc.RijCont = NULL
    acc.lineRatesDirty = NULL

cdef free_accumulate(RhAccumulate* acc):
    free(<void*> acc.gij)
    free(<void*> acc.Vij)
    free(<void*> acc.wla)
    free(<void*> acc.chi_up)
    free(<void*> acc.chi_down)
    free(<void*> acc.Uji_down)
    free(<void*> acc.eta)
    free(<void*> acc.Gamma)
    free(<void*> acc.RjiLine)
    free(<void*> acc.RijLine)
    free(<void*> acc.RjiCont)
    free(<void*> acc.RijCont)
    free(<void*> acc.lineRatesDirty)


cdef init_atmosphere(Atmosphere* atmos):
    atmos.H_LTE = False
    atmos.Stokes = False
    atmos.hydrostatic = False
    atmos.Ndim = 0
    atmos.N = NULL
    atmos.Nspace = 0
    atmos.Nrays = 0
    atmos.Natom = 0
    atmos.Nmolecule = 0
    atmos.NPRDactive = 0
    atmos.Nactiveatom = 0
    atmos.Nactivemol = 0
    atmos.T = NULL
    atmos.ne = NULL
    atmos.vturb = NULL
    atmos.v_los = NULL
    atmos.gravity = 0.0
    atmos.vmicro_char = 0.0
    atmos.lambda_ref = 0.0
    atmos.wmu = NULL
    atmos.nHtot = NULL
    atmos.nHmin = NULL
    atmos.B = NULL
    atmos.gamma_B = NULL
    atmos.chi_B = NULL
    atmos.B_char = 0.0
    atmos.cos_gamma = NULL
    atmos.cos_2chi = NULL
    atmos.sin_2chi = NULL
    atmos.angleSet.set = NO_SET
    atmos.angleSet.Nazimuth = 0
    atmos.angleSet.Ninclination = 0
    atmos.H = NULL
    atmos.atoms = NULL
    atmos.activeatoms = NULL
    atmos.H2 = NULL
    atmos.OH = NULL
    atmos.CH = NULL
    atmos.molecules = NULL
    atmos.activemols = NULL
    atmos.atomicTable = NULL

cdef init_geometry(Geometry* geo):
    geo.scale = mass_scale
    geo.vboundary[0] = ZERO
    geo.vboundary[1] = THERMALIZED
    geo.Ndep = 0
    geo.Nrays = 0
    geo.height = NULL
    geo.cmass = NULL
    geo.tau_ref = NULL
    geo.mux = NULL
    geo.muy = NULL
    geo.muz = NULL
    geo.wmu = NULL
    geo.vel = NULL
    geo.Itop = NULL
    geo.Ibottom = NULL

cdef init_molecule(Molecule* mol):
    mol.active = False
    mol.Nrt = 0
    mol.Nconfig = 0
    mol.Nv = 0
    mol.NJ = 0
    mol.activeindex = 0
    mol.Ediss = 0.0
    mol.vbroad = NULL
    mol.pf_coef = NULL
    mol.eqc_coef = NULL
    mol.pf = NULL
    mol.pfv = NULL
    mol.n = NULL
    mol.nv = NULL
    mol.nvstar = NULL
    mol.C_ul = NULL
    mol.Gamma = NULL
    mol.mrt = NULL
    mol.Ng_nv = NULL
    mol.rhAcc = NULL

cdef init_background(Background* bg):
    bg.Nspect = 0
    bg.Nray = 0
    bg.Ndir = 0
    bg.Nrlk = 0
    bg.rlk_lines = NULL
    bg.polarized = False
    bg.chi = NULL
    bg.eta = NULL
    bg.sca = NULL
    bg.chip = NULL

cdef init_depthdata(DepthData* d):
    d.IDepth = NULL
    d.SDepth = NULL
    d.chiDepth = NULL

cdef init_rhcontext(RhContext* ctx):
    ctx.atmos = NULL
    ctx.geo = NULL
    ctx.spectrum = NULL
    ctx.depth = NULL
    ctx.background = NULL
    ctx.options = NULL

cdef init_activeset(ActiveSet* aset):
    aset.Nactiveatomrt = NULL
    aset.Nactivemolrt = NULL
    aset.Nlower = NULL
    aset.Nupper = NULL
    aset.lower_levels = NULL
    aset.upper_levels = NULL
    aset.chi = NULL
    aset.eta = NULL
    aset.chip = NULL
    aset.art = NULL
    aset.mrt = NULL

cdef init_spectrum(Spectrum* spect):
    spect.updateJ = False
    spect.Nspect = 0
    spect.PRDindex = NULL
    spect.wavelength = NULL
    spect.J = NULL
    spect.I = NULL
    spect.Stokes_Q = NULL
    spect.Stokes_U = NULL
    spect.Stokes_V = NULL
    spect.J20 = NULL
    spect.Jgas = NULL
    spect.aset = NULL
    spect.nc = NULL
    spect.iprdh = NULL
    spect.cprdh = NULL

cdef class ComputationalAtomicContinuum:
    cdef RhAtomicContinuum* cCont

    @staticmethod
    cdef new(ComputationalAtom atom, cont, RhAtomicContinuum* cCont, options):
        self = ComputationalAtomicContinuum()
        self.atomicModel = atom
        self.continuumModel = cont
        self.cCont = cCont

        init_atomic_continuum(cCont)
        cCont.atom = &atom.cAtom

        cCont.i = cont.i
        cCont.j = cont.j
        cCont.nCont = atom.atomicModel.continua.index(cont)
        cCont.lambda0 = cont.lambda0

        if type(cont) is ExplicitContinuum:
            cCont.hydrogenic = False
        
        self.wavelength = np.ascontiguousarray(cont.wavelength)
        cdef int Nlambda = self.wavelength.shape[0]
        cCont.Nlambda = Nlambda
        self.alpha = np.ascontiguousarray(cont.alpha)
        cdef np.ndarray[np.double_t, ndim=1] ptr
        ptr = np.ascontiguousarray(self.wavelength)
        cCont.wavelength = &ptr[0]
        ptr = np.ascontiguousarray(self.alpha)
        cCont.alpha = &ptr[0]

        if atom.active:
            Nspace = atom.atmos.depthScale.shape[0]
            self.Rij = np.ascontiguousarray(np.zeros(Nspace))
            self.Rji = np.ascontiguousarray(np.zeros(Nspace))

            ptr = np.ascontiguousarray(self.Rij)
            cCont.Rij = &ptr[0]
            ptr = np.ascontiguousarray(self.Rji)
            cCont.Rji = &ptr[0]

        @property
        def i(self):
            return self.cCont.i

        @property
        def j(self):
            return self.cCont.j



cdef class ComputationalAtomicLine:
    cdef RhAtomicLine* cLine

    # def __dealloc__(self):
    #     free(<void*> self.cLine.c_shift)
    #     free(<void*> self.cLine.c_fraction)

    @staticmethod
    cdef new(ComputationalAtom atom, line, RhAtomicLine* cLine, options):
        self = ComputationalAtomicLine()
        self.atomicModel = atom
        self.lineModel = line
        self.cLine = cLine

        init_atomic_line(cLine)

        cLine.atom = &atom.cAtom

        cLine.i = line.i
        cLine.j = line.j
        cLine.nLine = atom.atomicModel.lines.index(line)
        cLine.Nlambda = line.Nlambda
        cLine.Grad = line.gRad
        cLine.g_Lande_eff = line.gLandeEff
        
        cLine.Aji = line.Aji
        cLine.Bji = line.Bji
        cLine.Bij = line.Bij

        cLine.lambda0 = line.lambda0

        if line.type == LineType.PRD and options.PRD.enable:
            cLine.PRD = True
            atom.Nprd += 1

        cLine.Ncomponent = 1
        cLine.c_shift = <double*> malloc(sizeof(double))
        cLine.c_fraction = <double*> malloc(sizeof(double))
        cLine.c_shift[0] = 0.0
        cLine.c_fraction[0] = 1.0

        cdef np.ndarray[np.double_t, ndim=1] wavelength
        if options.stokes:
            self.wavelength = np.ascontiguousarray(line.polarized_wavelength(options.stokes.b_char))
            wavelength = np.ascontiguousarray(self.wavelength)

            cLine.polarizable = np.any(line.wavelength != self.wavelength)
            cLine.wavelength = &wavelength[0]
            cLine.Nlambda = self.wavelength.shape[0]
        else:
            self.wavelength = np.ascontiguousarray(line.wavelength)
            wavelength = np.ascontiguousarray(self.wavelength)
            cLine.wavelength = &wavelength[0]
            cLine.Nlambda = self.wavelength.shape[0]

        cdef np.ndarray[np.double_t, ndim=1] ptr
        if atom.active:
            Nspace = atom.atmos.depthScale.shape[0]
            self.Rij = np.ascontiguousarray(np.zeros(Nspace))
            self.Rji = np.ascontiguousarray(np.zeros(Nspace))

            ptr = np.ascontiguousarray(self.Rij)
            cLine.Rij = &ptr[0]
            ptr = np.ascontiguousarray(self.Rji)
            cLine.Rji = &ptr[0]
        
            # if options.XRD.enable and len(line.xrd) > 0:

        @property
        def PRD(self):
            return self.cLine.PRD

        @property
        def i(self):
            return self.cLine.i

        @property
        def i(self):
            return self.cLine.j

        @property
        def Aji(self):
            return self.cLine.Aji

        @property
        def Bji(self):
            return self.cLine.Bji

        @property
        def Bij(self):
            return self.cLine.Bij

        
cdef class ComputationalAtom:
    cdef Atom cAtom
    cdef int Nthread
    cdef object atomicModel
    cdef object atmos
    cdef object active
    cdef object atomicTable
    cdef object nstar
    cdef object ntotal
    cdef object n
    cdef object g
    cdef object E
    cdef object vbroad
    cdef object lines
    cdef object continua

    def __dealloc__(self):
        for i in range(self.cAtom.Nline):
            free(<void*> self.cAtom.line[i].c_shift)
            free(<void*> self.cAtom.line[i].c_fraction)
            free(<void*> self.cAtom.line[i].xrd)
            free(<void*> self.cAtom.line)
            free(<void*> self.cAtom.continuum)
            free(<void*> self.cAtom.n)
            free(<void*> self.cAtom.nstar)
            free(<void*> self.cAtom.C)

            if self.Nthread > 1:
                for i in range(self.Nthread):
                    free_accumulate(&self.cAtom.accumulate[i])

    def __init__(self, atom, atmos, active, atomicTable, options):
        init_atom(&self.cAtom)
        self.atomicModel = atom
        self.atmos = atmos
        self.active = active
        self.atomicTable = atomicTable
        self.Nthread = int(options['Nthread'])

        atomicTable[atom.name].atom = self

        cdef int Nspace = atmos.depthScale.shape[0]
        cdef int Nlevel = len(atom.levels)
        self.nstar = np.ascontiguousarray(np.zeros((Nlevel, Nspace)))
        self.ntotal = np.ascontiguousarray(atomicTable[atom.name].abundance * atmos.nHTot)
        
        vtherm = 2.0 * Const.KBOLTZMANN / (Const.AMU * atomicTable[atom.name].weight)
        self.vbroad = np.ascontiguousarray(np.sqrt(vtherm * atmos.temperature + atmos.vturb**2))

        self.cAtom.active = active
        self.cAtom.Nlevel = Nlevel
        cdef np.ndarray[np.double_t, ndim=1] ptr
        self.g = np.ascontiguousarray(np.zeros(Nlevel))
        self.E = np.ascontiguousarray(np.zeros(Nlevel))
        for i, l in enumerate(atom.levels):
            self.g[i] = l.g
            self.E[i] = l.E_SI

        ptr = np.ascontiguousarray(self.g)
        self.cAtom.g = <double*> &ptr[0]
        ptr = np.ascontiguousarray(self.E)
        self.cAtom.E = <double*> &ptr[0]

        cdef np.ndarray[np.double_t, ndim=2] ptr2 = np.ascontiguousarray(self.nstar)
        self.cAtom.nstar = <double**> malloc(Nlevel * sizeof(double*))
        # As we used i in the enumerate before, Cython treats it as a python object here and won't do the cast-y stuff. Hence idx -- I'm not sure that's the reasoning actually. It seems in part linked to the type of the argument of the range
        for idx in range(Nlevel):
            self.cAtom.nstar[idx] = <double*> &ptr2[idx, 0]
        
        self.ntotal = np.ascontiguousarray(self.ntotal)
        ptr = np.ascontiguousarray(self.ntotal)
        self.cAtom.ntotal = &ptr[0]

        self.vbroad = np.ascontiguousarray(self.vbroad)
        ptr = np.ascontiguousarray(self.vbroad)
        self.cAtom.vbroad = &ptr[0]

        # TODO(cmo): Copy the levels, lines etc to this object (deepcopy), then add the extras like radiative rates to the entries in those arrays, leaving the model untouched. Then copy from those new models to the C Models. Given that we need to keep variables like Nblue updated later, we can use a setter property on this object to update them.

        cdef int Nline = len(atom.lines)
        self.cAtom.Nline = Nline
        self.cAtom.line = <RhAtomicLine*> malloc(Nline * sizeof(RhAtomicLine))
        cdef RhAtomicLine* cLine = NULL
        self.lines = []
        for i, l in enumerate(atom.lines):
            cLine = &self.cAtom.line[i]
            self.lines.append(ComputationalAtomicLine.new(self, l, cLine, options))
        
        if options.xrd.enable:
            for i, l in enumerate(atom.lines):
                if len(l.xrd) > 0:
                    length = len(l.xrd)
                    self.cAtom.line[i].Nxrd = length
                    self.cAtom.line[i].xrd = <RhAtomicLine**> malloc(length * sizeof(RhAtomicLine*))
                    for x in l.xrd:
                        xIdx = atom.lines.index(x)
                        self.cAtom.line[i].xrd[i] = &self.cAtom.line[xIdx]


        cdef int Ncont = len(atom.continua)
        self.cAtom.Ncont = Ncont
        self.cAtom.continuum = <RhAtomicContinuum*> malloc(Nline * sizeof(RhAtomicContinuum))
        cdef RhAtomicContinuum* cCont = NULL
        self.continua = []
        for i, l in enumerate(atom.lines):
            cCont = &self.cAtom.continuum[i]
            self.continua.append(ComputationalAtomicContinuum.new(self, l, cCont, options))

        self.collisions = deepcopy(atom.collisions)

        if self.active:
            self.cAtom.n = <double**> malloc(Nlevel * sizeof(double*))
            self.n = np.ascontiguousarray(np.zeros((Nlevel, Nspace)))
            ptr2 = np.ascontiguousarray(self.n)

            for idx in range(Nlevel):
                self.cAtom.n[idx] = <double*> &ptr2[idx, 0]

            if self.Nthread > 1:
                self.cAtom.accumulate = <RhAccumulate*> malloc(self.Nthread * sizeof(RhAccumulate))
                for i in range(self.Nthread):
                    init_accumulate(&self.cAtom.accumulate[i])


            self.C = np.ascontiguousarray(np.zeros((Nlevel*Nlevel, Nspace)))
            ptr2 = np.ascontiguousarray(self.C)
            self.cAtom.C = <double**> malloc(Nlevel*Nlevel*sizeof(double*))
            for idx in range(Nlevel*Nlevel):
                self.cAtom.C[idx] = <double*> &ptr2[idx, 0]

        else:
            self.n = self.nstar
            self.cAtom.n = self.cAtom.nstar

        @property
        def name(self):
            return self.atomicModel.name

cdef class ComputationalMolecule:
    cdef Molecule cMol

    def __init__(self, mol, atmos, pops, active, options):
        self.mol = mol
        if active:
            raise ValueError('Active Molecules NYI')
        init_molecule(&self.cMol)
        self.cMol.Ediss = mol.Ediss
        self.active = active
        self.cMol.active = active
        self.n = np.ascontiguousarray(pops)
        cdef np.ndarray[np.double_t, ndim=1] n = np.ascontiguousarray(self.n)
        self.cMol.n = <double*> &n[0]

cdef class ComputationalAtmosphere:
    cdef Atmosphere cAtmos
    cdef Geometry cGeo

    def __dealloc__(self):
        free(<void*> self.cAtmos.N)
        free(<void*> self.cAtmos.atoms)
        free(<void*> self.cAtmos.activeatoms)
        free(<void*> self.cAtmos.molecules)
        free(<void*> self.cAtmos.activemols)

    def __init__(self, atmos, atoms, mols, nRays, **kwargs):
        init_atmosphere(&self.cAtmos)
        init_geometry(&self.cGeo)
        # assume that the incoming atmosphere is already in the right units
        # Also assume that all depth scales are filled in? Yes for now
        self.atmos = atmos
        self.cAtoms = atoms
        self.cMolecules = mols
        # self.atoms = atoms
        # self.molecules = mols

        # Set up the atmosphere and geometry structures based on the the inputs.
        # What order between atoms and atmosphere?
        # If we go atom->atmosphere we only have to do one pass (CAtoms are already in place)
        # However atmosphere->atom->atmosphere may be tidier
        # Upon reflection, I think atoms->atmosphere will be easiest to pull off
        cdef int Nspace = atmos.depthScale.shape[0]
        self.cAtmos.Ndim = 1
        self.cAtmos.N = <int*> malloc(sizeof(int))
        self.cAtmos.N[0] = Nspace
        self.geo.Ndep = Nspace

        self.cAtmos.gravity = atmos.gravity
        self.cAtmos.lambda_ref = 500.0

        cdef np.ndarray[np.double_t, ndim=1] ptr
        self.tau_ref = np.ascontiguousarray(atmos.tau_ref)
        ptr = np.ascontiguousarray(self.tau_ref)
        self.cGeo.tau_ref = <double*> &ptr[0]

        self.cmass = np.ascontiguousarray(atmos.cmass)
        ptr = np.ascontiguousarray(self.cmass)
        self.cGeo.cmass = <double*> &ptr[0]

        self.height = np.ascontiguousarray(atmos.height)
        ptr = np.ascontiguousarray(self.height)
        self.cGeo.height = <double*> &ptr[0]

        self.temperature = np.ascontiguousarray(atmos.temperature)
        ptr = np.ascontiguousarray(self.temperature)
        self.cAtmos.T = <double*> &ptr[0]

        self.ne = np.ascontiguousarray(atmos.ne)
        ptr = np.ascontiguousarray(self.ne)
        self.cAtmos.ne = <double*> &ptr[0]

        self.vturb = np.ascontiguousarray(atmos.vturb)
        ptr = np.ascontiguousarray(self.vturb)
        self.cAtmos.vturb = <double*> &ptr[0]

        self.v_los = np.ascontiguousarray(atmos.v_los)
        ptr = np.ascontiguousarray(self.v_los)
        self.cGeo.vel = <double*> &ptr[0]

        # Copy properties, set up atmosphere.

        if nRays > 1:
            # Get quadrature
            self.nRays = nRays
            x, w = leggauss(nRays)
            mid, halfWidth = 0.5, 0.5
            x = mid + halfWidth * x
            w *= halfWidth

            self.muz = np.ascontiguousarray(x)
            self.muy = np.ascontiguousarray(np.zeros_like(x))
            self.mux = np.ascontiguousarray(np.sqrt(1.0 - x**2))
            self.wmu = np.ascontiguousarray(w)

            self.cGeo.Nrays = self.nRays
            self.cAtmos.Nrays = self.nRays

            ptr = np.ascontiguousarray(self.muz)
            self.cGeo.muz = <double*> &ptr[0]
            ptr = np.ascontiguousarray(self.muy)
            self.cGeo.muy = <double*> &ptr[0]
            ptr = np.ascontiguousarray(self.mux)
            self.cGeo.mux = <double*> &ptr[0]
            ptr = np.ascontiguousarray(self.wmu)
            self.cGeo.wmu = <double*> &ptr[0]
            self.cAtmos.wmu = self.cGeo.wmu

        # TODO(cmo): Fix ray handling
        elif nRays == 1:
            raise ValueError("Needs special handling for one ray")
        else:
            raise ValueError("Unsupported nRays=%d"%nRays)

        if atmos.B is not None:
            self.cAtmos.Stokes = True
            raise ValueError("Not yet supporting magnetic atmospheres")

        self.nHTot = np.ascontiguousarray(atmos.nHTot)
        ptr = np.ascontiguousarray(self.nHTot)
        self.cAtmos.nHtot = <double*> &ptr[0]

        # Only supporting most basic BCs for now
        # TODO(cmo): Fix BC handling
        self.cGeo.vboundary[int(TOP)] = ZERO
        self.cGeo.vboundary[int(BOTTOM)] = THERMALIZED

        # Put atoms and molecules into the atmosphere
        cdef int Natom = len(self.cAtoms)
        self.cAtmos.Natom = Natom
        self.cAtmos.atoms = <Atom**> malloc(Natom * sizeof(Atom*))
        cdef ComputationalAtom atom
        for i in range(Natom):
            atom = self.cAtoms[i]
            self.cAtmos.atoms[i] = &atom.cAtom
            if atom.atomicModel.name == 'H' or  atom.atomicModel.name == 'H ':
                self.cAtmos.H = &atom.cAtom

        cdef int Nmolecule = len(self.cMolecules)
        self.cAtmos.Nmolecule = Nmolecule
        self.cAtmos.molecules = <Molecule**> malloc(Nmolecule * sizeof(Molecule*))
        cdef ComputationalMolecule mol
        for i in range(Nmolecule):
            mol = self.cMolecules[i]
            self.cAtmos.molecules[i] = &mol.cMol
            if mol.mol.name == 'H2':
                self.cAtmos.H2 = &mol.cMol
            elif mol.mol.name == 'OH':
                self.cAtmos.OH = &mol.cMol
            elif mol.mol.name == 'CH':
                self.cAtmos.CH = &mol.cMol

        # Special case for H^-
        # TODO

        @property
        def activeAtoms(self):
            return [a for a in self.cAtoms if a.active]

cdef class ComputationalBackground:
    cdef Background cBg

    def __init__(self, ComputationalAtmosphere atomsphere, thing):
        init_background(&self.cBg)

cdef class ComputationalSpectrum:
    cdef Spectrum cSpect
    cdef int Nactiveatom

    def __dealloc__(self):
        pass
        # Cleanup activeset

    def __init__(self, radiativeSet, computationalAtoms, atomicTable, atmos):
        init_spectrum(&self.cSpect)

        self.activeInfo = radiativeSet.compute_wavelength_grid()
        activeAtoms = list(radiativeSet.activeAtoms)
        activeAtoms = sorted(activeAtoms, key=lambda x: atomicTable[x].weight)

        self.activeCompAtoms = atmos.activeAtoms
        activeIdx = {atom.atomicModel: i for i, atom in enumerate(self.activeCompAtoms)}
        if len(self.activeCompAtoms) != len(radiativeSet.activeAtoms):
            raise ValueError('Have all atoms be converted to their computational counterparts?')

        cdef int Nspect = self.activeInfo.wavelength.shape[0]
        self.cSpect.Nspect = Nspect
        self.wavelength = np.ascontiguousarray(self.activeInfo.wavelength)
        cdef np.ndarray[np.double_t, ndim=1] ptr = np.ascontiguousarray(self.activeInfo.wavelength)
        self.cSpect.wavelength = &ptr[0]


        self.Nactiveatom = len(self.activeCompAtoms)
        self.cSpect.aset = <ActiveSet*> malloc(Nspect * sizeof(ActiveSet))

        self.transitions = self.activeInfo.transitions
        self.compTransitions = []
        for t in self.transitions:
            atom = self.activeCompAtoms[activeIdx[t.atom]]
            if isinstance(t, AtomicContinuum):
                idx = atom.atomicModel.continua.index(t)
                self.compTransitions.append(atom.continua[idx])
            elif isinstance(t, AtomicLine):
                idx = atom.atomicModel.lines.index(t)
                self.compTransitions.append(atom.lines[idx])

        cdef ActiveSet* aset
        cdef int count = 0
        cdef ComputationalAtomicLine cLine
        cdef ComputationalAtomicContinuum cCont
        for i in range(Nspect):
            aset = &self.cSpect.aset[i]
            init_activeset(aset)
            aset.Nactiveatomrt = <int*> malloc(self.Nactiveatom * sizeof(int))
            aset.art = <AtomicTransition**> malloc(self.Nactiveatom * sizeof(AtomicTransition*))

            for nact in range(self.Nactiveatom):
                aset.Nactiveatomrt[nact] = 0

                for cont in self.activeInfo.continuaPerAtom[self.activeCompAtoms[nact].name][i]:
                    aset.art[nact][aset.Nactiveatomrt[nact]].type = type.ATOMIC_CONTINUUM
                    cCont = self.compTransitions[self.transitions.index(cont)]
                    aset.art[nact][aset.Nactiveatomrt[nact]].ptype.continuum = cCont.cCont
                    aset.Nactiveatomrt[nact] += 1

                for line in self.activeInfo.linesPerAtom[self.activeCompAtoms[nact].name][i]:
                    aset.art[nact][aset.Nactiveatomrt[nact]].type = type.ATOMIC_LINE
                    cLine = self.compTransitions[self.transitions.index(line)]
                    aset.art[nact][aset.Nactiveatomrt[nact]].ptype.line = cLine.cLine
                    aset.Nactiveatomrt[nact] += 1

        # Do a second pass for upper and lower levels
        for i in range(Nspect):
            aset = &self.cSpect.aset[i]

            aset.Nlower = <int*> malloc(self.Nactiveatom * sizeof(int))
            aset.Nupper = <int*> malloc(self.Nactiveatom * sizeof(int))
            aset.upper_levels = <int**> malloc(self.Nactiveatom * sizeof(int*))
            aset.lower_levels = <int**> malloc(self.Nactiveatom * sizeof(int*))

            for nact in range(self.Nactiveatom):
                lowerLevels = list(self.activeInfo.lowerLevels[self.activeCompAtoms[nact].name][i])
                upperLevels = list(self.activeInfo.upperLevels[self.activeCompAtoms[nact].name][i])
                aset.Nlower[nact] = int(len(lowerLevels))
                aset.Nupper[nact] = int(len(upperLevels))
                aset.upper_levels[nact] = <int*> malloc(aset.Nupper[nact] * sizeof(int))
                for j, l in enumerate(upperLevels):
                    aset.upper_levels[nact][j] = int(l)
                aset.lower_levels[nact] = <int*> malloc(aset.Nlower[nact] * sizeof(int))
                for j, l in enumerate(lowerLevels):
                    aset.lower_levels[nact][j] = int(l)
        

        # Collect the wavelength grid for all of the atoms and molecules
        # Sort and remove duplicates
        # Adjust the wavelengths for each transition
        # Recompute the wavelength/alpha grids for the continua
        # Fill the active set and set the upper/lower levels

    @property
    def updateJ(self):
        return self.cSpect.updateJ

    @updateJ.setter
    def updateJ(self, updateJ):
        self.cSpect.updateJ = updateJ


cdef class Context:
    cdef RhContext ctx
    cdef ComputationalAtmosphere cAtmos
    cdef ComputationalSpectrum cSpect
    cdef object radiativeSet
    cdef object molecules
    cdef object atmosphere
    cdef object atomicTable
    cdef object populations
    cdef object nRays
    cdef object options
    cdef object cAtoms
    cdef object cMolecules

    def __init__(self, radiativeSet, molecules, atmosphere, equilibriumPops, nRays, options):
        init_rhcontext(&self.ctx)

        self.radiativeSet = radiativeSet
        self.molecules = molecules
        self.atmosphere =  atmosphere
        self.atomicTable = equilibriumPops.atomicTable
        self.populations = equilibriumPops
        self.nRays = nRays
        self.options = options

        self.cAtoms = []
        for atom in radiativeSet.atoms:
            self.cAtoms.append(ComputationalAtom(atom, atmosphere, atom in radiativeSet.activeAtoms, self.atomicTable, options))

        self.cMolecules = []
        for mol in molecules:
            self.cMolecules.append(ComputationalMolecule(mol.molecularModel, atmosphere, mol.active, options))

        self.cAtmos = ComputationalAtmosphere(atmosphere, self.cAtoms, self.cMolecules, nRays)

        self.ctx.atmos = &self.cAtmos.cAtmos
        self.ctx.geo = &self.cAtmos.cGeo

        # Spectrum stuff
        # TODO(cmo): Configure the Spectrum object once, so we can actually not mess around with the computational atoms for the spectrum stuff once done
        self.cSpect = ComputationalSpectrum(radiativeSet, self.cAtoms, self.atomicTable, self.cAtmos)
        self.ctx.spectrum = &self.cSpect.cSpect

        # Background stuff




# TODO(cmo): RLK binding -- we can probably actually ignore this for now
# TODO(cmo): Basic molecule handling
# TODO(cmo): RhContext

