from dataclasses import dataclass, field
from AtomicTable import atomic_weight_sort
from AtomicModel import *
from Molecule import Molecule
from typing import List, Sequence, Set, Optional, Any, Union, Dict
from copy import copy

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
                grids.append(cont.wavelength)

        for atom in self.detailedLteSet:
            for line in atom.lines:
                grids.append(line.wavelength)
            for cont in atom.continua:
                grids.append(cont.wavelength)

        grid = np.concatenate(grids)
        grid = np.sort(grid)
        grid = np.unique(grid)
        blueIdx = []
        redIdx = []
        Nlambda = []

        for t in transitions:
            blueIdx.append(np.searchsorted(grid, t.wavelength[0]))
            redIdx.append(np.searchsorted(grid, t.wavelength[-1])+1)
            Nlambda.append(redIdx[-1] - blueIdx[-1])

        for i, t in enumerate(transitions):
            wavelength = np.copy(grid[blueIdx[i]:redIdx[i]])
            if isinstance(t, AtomicContinuum):
                t.alpha = t.compute_alpha(wavelength)
            t.wavelength = wavelength
            t.Nlambda = Nlambda[i] # type: ignore

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


