from dataclasses import dataclass, field
from atomic_table import atomic_weight_sort, Element, AtomicTable, get_global_atomic_table
from atomic_model import AtomicLine, AtomicModel, AtomicContinuum
from atmosphere import Atmosphere
import constants as Const
from typing import List, Sequence, Set, Optional, Any, Union, Dict
from copy import copy, deepcopy
from collections import OrderedDict
import numpy as np

@dataclass
class SpectrumConfiguration:
    radSet: 'RadiativeSet'
    wavelength: np.ndarray
    transitions: List[Union[AtomicLine, AtomicContinuum]]
    models: List[AtomicModel]
    blueIdx: List[int]
    activeSet: List[List[Union[AtomicLine, AtomicContinuum]]]
    freq: np.ndarray = field(init=False)

    def __post_init__(self):
        self.freq = Const.CLight / (self.wavelength * Const.NM_TO_M)

    def subset_configuration(self, wavelengths, expandLineGridsNm=0.0) -> 'SpectrumConfiguration':
        Nblue = np.searchsorted(self.wavelength, wavelengths[0])
        Nred = min(np.searchsorted(self.wavelength, wavelengths[-1])+1, self.wavelength.shape[0]-1)

        trans: List[Union[AtomicLine, AtomicContinuum]] = []
        continuaPerAtom: Dict[str, List[List[AtomicContinuum]]] = {}
        linesPerAtom: Dict[str, List[List[AtomicLine]]]= {}
        upperLevels: Dict[str, List[Set[int]]] = {}
        lowerLevels: Dict[str, List[Set[int]]] = {}

        radSet = self.radSet
        models = [m for m in (radSet.activeSet | radSet.detailedLteSet)]
        for atom in self.models:
            for l in atom.lines:
                if l.wavelength[-1] < wavelengths[0]:
                    continue
                if l.wavelength[0] > wavelengths[-1]:
                    continue
                trans.append(l)
                if expandLineGridsNm != 0.0:
                    l.wavelength = np.concatenate([[l.wavelength[0]-expandLineGridsNm, l.wavelength, l.wavelength[-1]+expandLineGridsNm]])
            for c in atom.continua:
                if c.wavelength[-1] < wavelengths[0]:
                    continue
                if c.wavelength[0] > wavelengths[-1]:
                    continue
                trans.append(c)
        activeAtoms = [t.atom for t in trans]

        for atom in activeAtoms:
            continuaPerAtom[atom.name] = []
            linesPerAtom[atom.name] = []
            upperLevels[atom.name] = []
            lowerLevels[atom.name] = []

        blueIdx = []
        redIdx = []
        for t in trans:
            blueIdx.append(np.searchsorted(wavelengths, t.wavelength[0]))
            redIdx.append(min(np.searchsorted(wavelengths, t.wavelength[-1])+1, wavelengths.shape[-1]))

        for i, t in enumerate(trans):
            if isinstance(t, AtomicContinuum):
                while wavelengths[redIdx[i]-1] > t.lambdaEdge and redIdx[i] > 0:
                    redIdx[i] -= 1
            wavelength = np.copy(wavelengths[blueIdx[i]:redIdx[i]])
            if isinstance(t, AtomicContinuum):
                t.alpha = t.compute_alpha(wavelength)
            t.wavelength = wavelength

        activeSet: List[List[Union[AtomicLine, AtomicContinuum]]] = []
        activeLines: List[List[AtomicLine]] = []
        activeContinua: List[List[AtomicContinuum]] = []
        contributors: List[List[AtomicModel]] = []
        for i in range(wavelengths.shape[0]):
            activeSet.append([])
            activeLines.append([])
            activeContinua.append([])
            contributors.append([])
            for atom in activeAtoms:
                continuaPerAtom[atom.name].append([])
                linesPerAtom[atom.name].append([])
                upperLevels[atom.name].append(set())
                lowerLevels[atom.name].append(set())
            for kr, t in enumerate(trans):
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


        return SpectrumConfiguration(radSet=radSet, wavelength=wavelengths, transitions=trans, models=activeAtoms, blueIdx=blueIdx,
                                     activeSet=activeSet)



def lte_pops(atomicModel, atmos, nTotal, debye=True):
    Nlevel = len(atomicModel.levels)
    c1 = (Const.HPlanck / (2.0 * np.pi * Const.MElectron)) * (Const.HPlanck / Const.KBoltzmann)

    c2 = 0.0
    if debye:
        c2 = np.sqrt(8.0 * np.pi / Const.KBoltzmann) * (Const.QElectron**2 / (4.0 * np.pi * Const.Epsilon0))**1.5
        nDebye = np.zeros(Nlevel)
        for i in range(1, Nlevel):
            stage = atomicModel.levels[i].stage
            Z = stage
            for m in range(1, stage - atomicModel.levels[0].stage + 1):
                nDebye[i] += Z
                Z += 1

    dEion = c2 * np.sqrt(atmos.ne / atmos.temperature)
    cNe_T = 0.5 * atmos.ne * (c1 / atmos.temperature)**1.5
    total = np.ones(atmos.Nspace)

    nStar = np.zeros((Nlevel, atmos.temperature.shape[0]))
    ground = atomicModel.levels[0]
    for i in range(1, Nlevel):
        dE = atomicModel.levels[i].E_SI - ground.E_SI
        gi0 = atomicModel.levels[i].g / ground.g
        dZ = atomicModel.levels[i].stage - ground.stage
        if debye:
            dE_kT = (dE - nDebye[i] * dEion) / (Const.KBoltzmann * atmos.temperature)
        else:
            dE_kT = dE / (Const.KBoltzmann * atmos.temperature)

        nst = gi0 * np.exp(-dE_kT)
        nStar[i, :] = nst
        nStar[i, :] /= cNe_T**dZ
        total += nStar[i]

    nStar[0] = nTotal / total

    for i in range(1, Nlevel):
        nStar[i] *= nStar[0]

    return nStar

class AtomicStateTable:
    def __init__(self, atmos: Atmosphere, atomicTable: AtomicTable, atoms: List['AtomicState']):
        self.atmos = atmos
        self.atomicTable = atomicTable
        self.atoms = atoms
        self.indices = OrderedDict(zip([a.model.name.upper().ljust(2) for a in atoms], list(range(len(atoms)))))

    def __contains__(self, name: str) -> bool:
        name = name.upper().ljust(2)
        return name in self.indices.keys()

    def __len__(self) -> int:
        return len(self.atoms)

    def __getitem__(self, name: str) -> 'AtomicState':
        name = name.upper().ljust(2)
        return self.atoms[self.indices[name]]

    def __iter__(self) -> 'AtomicStateTableIterator':
        return AtomicStateTableIterator(self)

class AtomicStateTableIterator:
    def __init__(self, table: AtomicStateTable):
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
class AtomicState:
    model: AtomicModel
    nStar: np.ndarray
    nTotal: np.ndarray
    pops: Optional[np.ndarray] = None

    def __repr__(self):
        return 'AtomicModelPops(name=%s(%d), nStar=%s, nTotal=%s, pops=%s)' % (self.model.name, hash(self.model), repr(self.nStar), repr(self.nTotal), repr(self.pops))

    def __hash__(self):
        return hash(repr(self))

    def update_nTotal(self, atmos : Atmosphere):
        dim = False
        if atmos.dimensioned:
            dim = True
            atmos.nondimensionalise()
        self.nTotal[:] = self.model.atomicTable[self.name].abundance * atmos.nHTot
        if dim:
            atmos.dimensionalise()

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

    @n.setter
    def n(self, val: np.ndarray):
        if val.shape != self.nStar.shape:
            raise ValueError('Incorrect dimensions for population array, expected %s, got %s.' % (repr(self.nStar.shape), repr(val.shape)))
        self.pops = val


@dataclass
class RadiativeSet:
    atoms: List[AtomicModel]
    atomicTable: AtomicTable = field(default_factory=get_global_atomic_table)
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

        self.set_atomic_table(self.atomicTable)

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
        if (self.activeSet | self.passiveSet | self.detailedLteSet) != set(self.atoms):
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
        self.atomicTable = table
        for a in self.atoms:
            if a.atomicTable is self.atomicTable:
                continue
            a.replace_atomic_table(self.atomicTable)

    def iterate_lte_ne_eq_pops(self, atmos: Atmosphere):
        atmos.nondimensionalise()
        maxIter = 500
        prevNe = np.copy(atmos.ne)
        ne = np.copy(atmos.ne)
        for it in range(maxIter):
            atomicPops = []
            prevNe[:] = ne
            ne.fill(0.0)
            for a in sorted(self.atoms, key=atomic_weight_sort):
                nTotal = self.atomicTable[a.name].abundance * atmos.nHTot
                nStar = lte_pops(a, atmos, nTotal, debye=True)
                atomicPops.append(AtomicState(a, nStar, nTotal))
                stages = np.array([l.stage for l in a.levels])
                # print(stages)
                ne += np.sum(nStar * stages[:, None], axis=0)
                # print(ne)
            atmos.ne[:] = ne

            relDiff = np.nanmax(np.abs(1.0 - prevNe / ne))
            print(relDiff)
            maxRelDiff = np.nanmax(relDiff)
            if maxRelDiff < 1e-3:
                print("Iterate LTE: %d iterations" % it)
                break
        else:
            print("LTE ne failed to converge")

        atmos.dimensionalise()
        table = AtomicStateTable(atmos, self.atomicTable, atomicPops)
        return table

    def compute_eq_pops(self, atmos: Atmosphere):
        dim = False
        if atmos.dimensioned:
            dim = True
            atmos.nondimensionalise()
        atomicPops = []
        for a in sorted(self.atoms, key=atomic_weight_sort):
            nTotal = self.atomicTable[a.name].abundance * atmos.nHTot
            nStar = lte_pops(a, atmos, nTotal, debye=True)
            atomicPops.append(AtomicState(a, nStar, nTotal))

        table = AtomicStateTable(atmos, self.atomicTable, atomicPops)
        if dim:
            atmos.dimensionalise
        return table

    def compute_wavelength_grid(self, extraWavelengths: Optional[np.ndarray]=None, lambdaReference=500.0) -> SpectrumConfiguration:
        if len(self.activeSet) == 0 and len(self.detailedLteSet) == 0:
            raise ValueError('Need at least one atom active or in detailed LTE')
        grids = []
        if extraWavelengths is not None:
            grids.append(extraWavelengths)
        grids.append(np.array([lambdaReference]))

        models: List[AtomicModel] = []
        transitions: List[Union[AtomicLine, AtomicContinuum]] = []
        continuaPerAtom: Dict[str, List[List[AtomicContinuum]]] = {}
        linesPerAtom: Dict[str, List[List[AtomicLine]]]= {}
        upperLevels: Dict[str, List[Set[int]]] = {}
        lowerLevels: Dict[str, List[Set[int]]] = {}

        for atom in (self.activeSet | self.detailedLteSet):
            models.append(atom)
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

        grid = np.concatenate(grids)
        grid = np.sort(grid)
        grid = np.unique(grid)
        blueIdx = []
        redIdx = []

        for t in transitions:
            blueIdx.append(np.searchsorted(grid, t.wavelength[0]))
            redIdx.append(np.searchsorted(grid, t.wavelength[-1])+1)

        for i, t in enumerate(transitions):
            # NOTE(cmo): Some continua have wavelength grids that go past their edge. Let's avoid that.
            if isinstance(t, AtomicContinuum):
                while grid[redIdx[i]-1] > t.lambdaEdge:
                    redIdx[i] -= 1
            wavelength = np.copy(grid[blueIdx[i]:redIdx[i]])
            if isinstance(t, AtomicContinuum):
                t.alpha = t.compute_alpha(wavelength)
            t.wavelength = wavelength

        activeSet: List[List[Union[AtomicLine, AtomicContinuum]]] = []
        activeLines: List[List[AtomicLine]] = []
        activeContinua: List[List[AtomicContinuum]] = []
        contributors: List[List[AtomicModel]] = []
        for i in range(grid.shape[0]):
            activeSet.append([])
            activeLines.append([])
            activeContinua.append([])
            contributors.append([])
            for atom in (self.activeSet | self.detailedLteSet):
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

        return SpectrumConfiguration(radSet=self, wavelength=grid, transitions=transitions, models=models, blueIdx=blueIdx, activeSet=activeSet)