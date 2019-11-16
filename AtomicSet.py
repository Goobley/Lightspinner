from dataclasses import dataclass, field
from AtomicTable import atomic_weight_sort, Element, AtomicTable
from AtomicModel import *
from Atmosphere import Atmosphere
from Molecule import Molecule, MolecularTable
from typing import List, Sequence, Set, Optional, Any, Union, Dict
from copy import copy
from collections import OrderedDict
import numpy as np
from scipy.linalg import solve

@dataclass
class SpectrumConfiguration:
    wavelength: np.ndarray
    transitions: List[Union[AtomicLine, AtomicContinuum]]
    blueIdx: List[int]
    redIdx: List[int]
    activeSet: List[List[Union[AtomicLine, AtomicContinuum]]]
    activeLines: List[List[AtomicLine]]
    activeContinua: List[List[AtomicContinuum]]
    contributors: List[List[AtomicModel]]
    continuaPerAtom: Dict[str, List[List[AtomicContinuum]]]
    linesPerAtom: Dict[str, List[List[AtomicLine]]]
    lowerLevels: Dict[str, List[Set[int]]]
    upperLevels: Dict[str, List[Set[int]]]


def lte_pops(atomicModel, atmos, nTotal, debeye=True):
    Nlevel = len(atomicModel.levels)
    c1 = (Const.HPLANCK / (2.0 * np.pi * Const.M_ELECTRON)) * (Const.HPLANCK / Const.KBOLTZMANN)

    c2 = 0.0
    if debeye:
        c2 = np.sqrt(8.0 * np.pi / Const.KBOLTZMANN) * (Const.Q_ELECTRON**2 / (4.0 * np.pi * Const.EPSILON_0))**1.5
        nDebeye = np.zeros(Nlevel)
        for i in range(1, Nlevel):
            stage = atomicModel.levels[i].stage
            Z = stage
            for m in range(1, stage - atomicModel.levels[0].stage + 1):
                nDebeye[i] += Z
                Z += 1
    # print(atomicModel.name)
    # print([l.stage for l in atomicModel.levels])
    # print(nDebye)

    dEion = c2 * np.sqrt(atmos.ne / atmos.temperature)
    cNe_T = 0.5 * atmos.ne * (c1 / atmos.temperature)**1.5
    total = np.ones(atmos.Nspace)

    nStar = np.zeros((Nlevel, atmos.temperature.shape[0]))
    ground = atomicModel.levels[0]
    for i in range(1, Nlevel):
        dE = atomicModel.levels[i].E_SI - ground.E_SI
        gi0 = atomicModel.levels[i].g / ground.g
        dZ = atomicModel.levels[i].stage - ground.stage
        if debeye:
            dE_kT = (dE - nDebeye[i] * dEion) / (Const.KBOLTZMANN * atmos.temperature)
        else:
            dE_kT = dE / (Const.KBOLTZMANN * atmos.temperature)

        nst = gi0 * np.exp(-dE_kT)
        nStar[i, :] = nst
        nStar[i, :] /= cNe_T**dZ
        # for m in range(1, dZ + 1):
        #     nStar[i, :] /= cNe_T
        total += nStar[i]

    nStar[0] = nTotal / total

    for i in range(1, Nlevel):
        nStar[i] *= nStar[0]

    return nStar

class AtomicPopsTable:
    def __init__(self, atoms: List['AtomicModelPops']):
        self.atoms = atoms
        self.indices = OrderedDict(zip([a.model.name.upper().ljust(2) for a in atoms], list(range(len(atoms)))))

    def __contains__(self, name: str) -> bool:
        name = name.upper().ljust(2)
        return name in self.indices.keys()

    def __len__(self) -> int:
        return len(self.atoms)

    def __getitem__(self, name: str) -> 'AtomicModelPops':
        name = name.upper().ljust(2)
        return self.atoms[self.indices[name]]

    def __iter__(self) -> 'AtomicPopsTableIterator':
        return AtomicPopsTableIterator(self)

class AtomicPopsTableIterator:
    def __init__(self, table: AtomicPopsTable):
        self.table = table
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.table):
            a = self.table.atoms[self.index]
            self.index += 1
            return a

        raise StopIteration



@dataclass
class AtomicModelPops:
    model: AtomicModel
    nStar: np.ndarray
    nTotal: np.ndarray
    pops: Optional[np.ndarray] = None

    def __repr__(self):
        return 'AtomicModelPops(name=%s(%d), nStar=%s, nTotal=%s, pops=%s)' % (self.model.name, hash(self.model), repr(self.nStar), repr(self.nTotal), repr(self.pops))

    def __hash__(self):
        return hash(repr(self))

    @property
    def name(self):
        return self.model.name

    @property
    def abundance(self):
        return self.model.atomicTable[self.model.name].abundance

    @property
    def weight(self):
        return self.model.atomicTable[self.model.name].weight

    @property
    def n(self) -> np.ndarray:
        if self.pops is not None:
            return self.pops
        else:
            return self.nStar

    def fjk(self, atmos, k):
        # Nstage: int = (self.model.levels[-1].stage - self.model.levels[0].stage) + 1
        Nstage: int = self.model.levels[-1].stage + 1

        fjk = np.zeros(Nstage)
        dfjk = np.zeros(Nstage)

        for i, l in enumerate(self.model.levels):
            fjk[l.stage] += self.n[i, k]

        fjk /= self.nTotal[k]

        return fjk, dfjk

    def fj(self, atmos):
        # Nstage: int = (self.model.levels[-1].stage - self.model.levels[0].stage) + 1
        Nstage: int = self.model.levels[-1].stage + 1
        Nspace: int = atmos.depthScale.shape[0]

        fj = np.zeros((Nstage, Nspace))
        dfj = np.zeros((Nstage, Nspace))

        for i, l in enumerate(self.model.levels):
            fj[l.stage] += self.n[i]

        fjk /= self.nTotal

        return fj, dfj


@dataclass
class EquilibriumPopulations:
    atmosphere: Atmosphere
    atomicTable: AtomicTable
    atomicPops: AtomicPopsTable
    molecularTable: MolecularTable
    molecularPops: List[np.ndarray]
    HminPops: np.ndarray

    def __getitem__(self, name: str) -> np.ndarray:
        if name == 'H-':
            return self.HminPops
        else:
            if name in self.molecularTable:
                key = self.molecularTable.indices[name]
                return self.molecularPops[key]
            elif name in self.atomicPops:
                return self.atomicPops[name].n
            else:
                raise KeyError('Unknown key: %s' % name)

    def __contains__(self, name: Union[str, AtomicModelPops]) -> bool:
        if isinstance(name, AtomicModel):
            name = name.name
        if name == 'H-':
            return True
        
        if name in self.molecularTable:
            return True

        if name in self.atomicPops:
            return True

        return False

    def atomic_population(self, name: str) -> np.ndarray:
        return self.atomicPops[name].n

    def molecular_population(self, name: str) -> np.ndarray:
        name = name.upper()
        key = self.molecularTable.indices[name]
        return self.molecularPops[key]

@dataclass
class RadiativeSet:
    atoms: List[AtomicModel]
    activeSet: Set[AtomicModel] = field(default_factory=set)
    detailedLteSet: Set[AtomicModel] = field(default_factory=set)
    passiveSet: Set[AtomicModel] = field(init=False)

    def __post_init__(self):
        self.passiveSet = set(self.atoms)
        self.atomicNames = []
        for atom in self.atoms:
            self.atomicNames.append(atom.name)

        if len(self.atomicNames) > len(set(self.atomicNames)):
            raise ValueError('Multiple entries for an atom: %s' % self.atoms)

    def __contains__(self, name: str) -> bool:
        return name in self.atomicNames

    def is_active(self, name: str) -> bool:
        if name in self.atomicNames:
            return self.atoms[self.atomicNames.index(name)] in self.activeSet
        raise ValueError('Atom %s not present in RadiativeSet' % name)

    def is_passive(self, name: str) -> bool:
        if name in self.atomicNames:
            return self.atoms[self.atomicNames.index(name)] in self.passiveSet
        raise ValueError('Atom %s not present in RadiativeSet' % name)

    def is_lte(self, name: str) -> bool:
        if name in self.atomicNames:
            return self.atoms[self.atomicNames.index(name)] in self.detailedLteSet
        raise ValueError('Atom %s not present in RadiativeSet' % name)

    @property
    def activeAtoms(self) -> List[AtomicModel]:
        activeAtoms : List[AtomicModel] = [a for a in self.activeSet]
        activeAtoms = sorted(activeAtoms, key=atomic_weight_sort)
        return activeAtoms

    @property
    def lteAtoms(self) -> List[AtomicModel]:
        lteAtoms : List[AtomicModel] = [a for a in self.detailedLteSet]
        lteAtoms = sorted(lteAtoms, key=atomic_weight_sort)
        return lteAtoms

    @property
    def passiveAtoms(self) -> List[AtomicModel]:
        passiveAtoms : List[AtomicModel] = [a for a in self.passiveSet]
        passiveAtoms = sorted(passiveAtoms, key=atomic_weight_sort)
        return passiveAtoms

    def __getitem__(self, name: str) -> AtomicModel:
        name = name.upper()
        if len(name) == 1 and name not in self.atomicNames:
            name += ' '

        return self.atoms[self.atomicNames.index(name)]

    def validate_sets(self):
        if self.activeSet | self.passiveSet | self.detailedLteSet != set(self.atoms):
            raise ValueError('Problem with distribution of Atoms inside AtomicSet')
    
    def set_active(self, *args: str):
        names = set(args)
        for atomName in names:
            self.activeSet.add(self[atomName])
            self.detailedLteSet.discard(self[atomName])
            self.passiveSet.discard(self[atomName])
        self.validate_sets()

    def set_detailed_lte(self, *args: str):
        names = set(args)
        for atomName in names:
            self.detailedLteSet.add(self[atomName])
            self.activeSet.discard(self[atomName])
            self.passiveSet.discard(self[atomName])
        self.validate_sets()

    def set_passive(self, *args: str):
        names = set(args)
        for atomName in names:
            self.passiveSet.add(self[atomName])
            self.activeSet.discard(self[atomName])
            self.detailedLteSet.discard(self[atomName])
        self.validate_sets()

    def set_atomic_table(self, table: AtomicTable):
        for a in self.atoms:
            a.replace_atomic_table(table)

    def compute_eq_pops(self, mols: MolecularTable, atmos: Atmosphere):
        atomicPops = []
        for a in sorted(self.atoms, key=atomic_weight_sort):
            nTotal = a.atomicTable[a.name].abundance * atmos.nHTot
            nStar = lte_pops(a, atmos, nTotal, debeye=True)
            atomicPops.append(AtomicModelPops(a, nStar, nTotal))

        table = AtomicPopsTable(atomicPops)
        eqPops = chemical_equilibrium_fixed_ne(atmos, mols, table, self.atoms[0].atomicTable)
        return eqPops

    def compute_wavelength_grid(self, extraWavelengths: Optional[np.ndarray]=None, lambdaReference=500.0) -> SpectrumConfiguration:
        if len(self.activeSet) == 0 and len(self.detailedLteSet) == 0:
            raise ValueError('Need at least one atom active or in detailed LTE')
        grids = []
        if extraWavelengths is not None:
            grids.append(extraWavelengths)
        grids.append(np.array([lambdaReference]))

        transitions: List[Union[AtomicLine, AtomicContinuum]] = []
        continuaPerAtom: Dict[str, List[List[AtomicContinuum]]] = {}
        linesPerAtom: Dict[str, List[List[AtomicLine]]]= {}
        upperLevels: Dict[str, List[Set[int]]] = {}
        lowerLevels: Dict[str, List[Set[int]]] = {}

        for atom in self.activeSet:
            continuaPerAtom[atom.name] = []
            linesPerAtom[atom.name] = []
            lowerLevels[atom.name] = []
            upperLevels[atom.name] = []
            for line in atom.lines:
                transitions.append(line)
                grids.append(line.wavelength)
            for cont in atom.continua:
                transitions.append(cont)
                grids.append(np.array([cont.lambdaEdge]))
                grids.append(cont.wavelength[cont.wavelength <= cont.lambdaEdge])

        for atom in self.detailedLteSet:
            for line in atom.lines:
                grids.append(line.wavelength)
            for cont in atom.continua:
                grids.append(cont.wavelength)

        grid = np.concatenate(grids)
        grid = np.sort(grid)
        grid = np.unique(grid)
        # grid = np.unique(np.floor(1e10*grid)) / 1e10
        blueIdx = []
        redIdx = []
        # Nlambda = []

        for t in transitions:
            blueIdx.append(np.searchsorted(grid, t.wavelength[0]))
            redIdx.append(np.searchsorted(grid, t.wavelength[-1])+1)
            # Nlambda.append(redIdx[-1] - blueIdx[-1])

        for i, t in enumerate(transitions):
            # NOTE(cmo): Some continua have wavelength grids that go past their edge. Let's avoid that.
            if isinstance(t, AtomicContinuum):
                while grid[redIdx[i]-1] > t.lambdaEdge:
                    redIdx[i] -= 1
            wavelength = np.copy(grid[blueIdx[i]:redIdx[i]])
            if isinstance(t, AtomicContinuum):
                t.alpha = t.compute_alpha(wavelength)
            t.wavelength = wavelength
            t.Nlambda = wavelength.shape[0] # type: ignore

        activeSet: List[List[Union[AtomicLine, AtomicContinuum]]] = []
        activeLines: List[List[AtomicLine]] = []
        activeContinua: List[List[AtomicContinuum]] = []
        contributors: List[List[AtomicModel]] = []
        for i in range(grid.shape[0]):
            activeSet.append([])
            activeLines.append([])
            activeContinua.append([])
            contributors.append([])
            for atom in self.activeSet:
                continuaPerAtom[atom.name].append([])
                linesPerAtom[atom.name].append([])
                upperLevels[atom.name].append(set())
                lowerLevels[atom.name].append(set())
            for kr, t in enumerate(transitions):
                if blueIdx[kr] <= i < redIdx[kr]:
                    activeSet[-1].append(t)
                    contributors[-1].append(t.atom)
                    if isinstance(t, AtomicContinuum):
                        activeContinua[-1].append(t)
                        continuaPerAtom[t.atom.name][-1].append(t)
                        upperLevels[t.atom.name][-1].add(t.j)
                        lowerLevels[t.atom.name][-1].add(t.i)
                    elif isinstance(t, AtomicLine):
                        activeLines[-1].append(t)
                        linesPerAtom[t.atom.name][-1].append(t)
                        upperLevels[t.atom.name][-1].add(t.j)
                        lowerLevels[t.atom.name][-1].add(t.i)


        return SpectrumConfiguration(grid, transitions, blueIdx, redIdx, activeSet, activeLines, activeContinua, contributors, continuaPerAtom, linesPerAtom, lowerLevels, upperLevels)



def chemical_equilibrium_fixed_ne(atmos: Atmosphere, molecules: MolecularTable, atomicPops: AtomicPopsTable, table: AtomicTable) -> EquilibriumPopulations:
    nucleiSet: Set[Element] = set()
    for mol in molecules:
        nucleiSet |= set(mol.elements)
    nuclei: List[Union[Element, AtomicModelPops]]= list(nucleiSet)
    nuclei = sorted(nuclei, key=atomic_weight_sort)
    if not nuclei[0].name.startswith('H'):
        raise ValueError('H not list of nuclei -- check H2 molecule')
    print([n.name for n in nuclei])

    nuclIndex = [[nuclei.index(ele) for ele in mol.elements] for mol in molecules]

    # Replace basic elements with full Models if present
    for i, nuc in enumerate(nuclei):
        if nuc.name in atomicPops:
            nuclei[i] = atomicPops[nuc.name]
    Nnuclei = len(nuclei)

    Neqn = Nnuclei + len(molecules)
    f = np.zeros(Neqn)
    n = np.zeros(Neqn)
    df = np.zeros((Neqn, Neqn))
    a = np.zeros(Neqn)

    # Equilibrium constant per molecule
    Phi = np.zeros(len(molecules))
    # Neutral fraction
    fn0 = np.zeros(Nnuclei)

    CI = (Const.HPLANCK / (2.0 * np.pi * Const.M_ELECTRON)) * (Const.HPLANCK / Const.KBOLTZMANN)
    Nspace = atmos.depthScale.shape[0]
    HminPops = np.zeros(Nspace)
    molPops = [np.zeros(Nspace) for mol in molecules]
    maxIter = 0
    for k in range(Nspace):
        for i in range(Nnuclei):
            a[i] = nuclei[i].abundance * atmos.nHTot[k]
            fjk, dfjk = nuclei[i].fjk(atmos, k)
            fn0[i] = fjk[0]

        PhiHmin = 0.25 * (CI / atmos.temperature[k])**1.5 \
                    * np.exp(Const.E_ION_HMIN / (Const.KBOLTZMANN * atmos.temperature[k]))
        fHmin = atmos.ne[k] * fn0[0] * PhiHmin


        # Eq constant for each molecule at this location
        for i, mol in enumerate(molecules):
            Phi[i] = mol.equilibrium_constant(atmos.temperature[k])

        # Setup initial solution. Everything dissociated
        # n[:Nnuclei] = a[:Nnuclei]
        # n[Nnuclei:] = 0.0
        n[:] = a[:]
        # print('a', a)

        nIter = 1
        NmaxIter = 50
        IterLimit = 1e-3
        prevN = n.copy()
        while nIter < NmaxIter:
            # print(k, ',', nIter)
            # Save previous solution
            prevN[:] = n[:]

            # Set up iteration
            f[:] = n - a
            df[:, :] = 0.0
            np.fill_diagonal(df, 1.0)

            # Add nHmin to number conservation for H
            f[0] += fHmin * n[0]
            df[0, 0] += fHmin

            # Fill population vector f and derivative matrix df
            for i, mol in enumerate(molecules):
                saha = Phi[i]
                for j, ele in enumerate(mol.elements):
                    nu = nuclIndex[i][j]
                    saha *= (fn0[nu] * n[nu])**mol.elementCount[j]
                    # Contribution to conservation for each nucleus in this molecule
                    f[nu] += mol.elementCount[j] * n[Nnuclei + i]

                saha /= atmos.ne[k]**mol.charge
                f[Nnuclei + i] -= saha
                # if Nnuclei + i == f.shape[0]-1:
                #     print(i)
                #     print(saha)

                # Compute derivative matrix
                for j, ele in enumerate(mol.elements):
                    nu = nuclIndex[i][j]
                    df[nu, Nnuclei + i] += mol.elementCount[j]
                    df[Nnuclei + i, nu] = -saha * (mol.elementCount[j] / n[nu])

            correction = solve(df, f)
            n -= correction

            # correction = dgesvx(df, f.reshape(f.shape[0], 1))[7]
            # n -= correction.squeeze()
            # correction = newton_krylov(lambda x: df@x - f, np.zeros_like(f))
            # n -= correction
            # print(correction)

            # dnMax = np.nanmax(np.abs((n - prevN) / n))
            dnMax = np.nanmax(np.abs(1.0 - prevN / n))
            # print(dnMax)
            if dnMax <= IterLimit:
                maxIter = max(maxIter, nIter)
                break

            nIter += 1
        if dnMax > IterLimit:
            raise ValueError("ChemEq iteration not converged: T: %e [K], density %e [m^-3], dnmax %e" % (atmos.temperature[k], atmos.nHTot[k], dnMax))

        for i, ele in enumerate(nuclei):
            if ele.name in atomicPops:
                atomPop = atomicPops[ele.name].n
                fraction = n[i] / np.sum(atomPop[:, k])
                atomPop[:, k] *= fraction

        HminPops[k] = atmos.ne[k] * n[0] * PhiHmin

        for i, pop in enumerate(molPops):
            pop[k] = n[Nnuclei + i] 

    result = EquilibriumPopulations(atmos, table, atomicPops, molecules, molPops, HminPops)
    print("Maximum number of iterations taken: %d" % maxIter)
    return result