from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram, CompoundPhaseDiagram
from pymatgen.core.composition import Composition
from pymatgen.analysis.interface_reactions import InterfacialReactivity

from pydmclab.utils.handy import read_json, write_json


class MixingHull(object):
    def __init__(self, input_energies, end_members, energy_key):
        for c in input_energies:
            input_energies[c]["comp"] = Composition(c)
            input_energies[c]["entry"] = PDEntry(
                input_energies[c]["comp"], input_energies[c][energy_key]
            )

        self.input_energies = input_energies
        self.end_members = [Composition(c) for c in end_members]

    @property
    def pd(self):
        input_energies = self.input_energies
        pd_entries = [input_energies[c]["entry"] for c in input_energies]
        return CompoundPhaseDiagram(pd_entries, self.end_members, False)


if __name__ == "__main__":
    mix = main()
