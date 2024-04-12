from pymatgen.io.vasp import DictSet, Incar, Kpoints, Potcar, Poscar, MPRelaxSet


class GGADMCRelaxSet(DictSet):

    CONFIG = MPRelaxSet.CONFIG

    def incar_updates(self):
        return {}

    def kpoints_updates(self):
        return {}

    def potcar_updates(self):
        return {}
