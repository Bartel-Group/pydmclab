import numpy as np
import os
import matplotlib.pyplot as plt

from ase.phonons import Phonons
from ase.thermochemistry import CrystalThermo

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.eos import Murnaghan

from chgnet.model import CHGNetCalculator

from pydmclab.core.struc import StrucTools
from pydmclab.plotting.utils import set_rc_params, get_colors, get_label
from pydmclab.utils.handy import read_json, write_json

from pydmclab.mlp.relax import Relaxer


class HarmonicPhonons(object):
    """
    Obtain the harmonic phonon DOS for one structure with a ML potential
    """

    def __init__(
        self,
        structure,
        supercell_grid=(1, 1, 1),
        calculator=CHGNetCalculator(),
        displacement_delta=5e-2,
        kpts_for_dos=(40, 40, 40),
        npts_for_dos=3000,
        dos_delta=5e-4,
        data_dir="data",
        displacement_dir="phonons",
        fjson_dos="phonon_dos.json",
        remake_displacements=False,
        remake_dos=False,
    ):
        """
        Args:
            structure (pymatgen.Structure)
                The structure for which to calculate the phonon DOS

            supercell_grid (tuple)
                The supercell grid for the phonon calculations

            calculator (ase Calculator)
                The calculator to use for the phonon calculations

            displacement_delta (float)
                The displacement delta for the phonon calculations (for finite displacements method)

            kpts_for_dos (tuple)
                The k-points for the DOS calculations

            npts_for_dos (int)
                The number of points for the DOS calculations

            dos_delta (float)
                The delta for the DOS calculations

            data_dir (str)
                The directory where the data is stored

            displacement_dir (str):
                The directory where the displacements are stored
                    within data_dir

            fjson_dos (str)
                The filename for the phonon DOS data
                    within data_dir

            remake_displacements (bool)
                Whether to remake the displacements

            remake_dos (bool)
                Whether to remake the DOS
        """

        self.structure = structure
        self.atoms = AseAtomsAdaptor.get_atoms(structure)
        self.supercell_grid = supercell_grid
        self.calculator = calculator
        self.displacement_delta = displacement_delta
        self.kpts_for_dos = kpts_for_dos
        self.npts_for_dos = npts_for_dos
        self.dos_delta = dos_delta
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        self.displacement_dir = os.path.join(data_dir, displacement_dir)
        self.fjson_dos = os.path.join(data_dir, fjson_dos)
        self.remake_displacements = remake_displacements
        self.remake_dos = remake_dos

    @property
    def E0(self):
        """
        Returns:
            The 0 K internal energy (eV/cell) computed with the Calculator
        """
        calc = self.calculator
        atoms = self.atoms
        atoms.calc = calc
        return atoms.get_potential_energy()

    @property
    def phonons(self):
        """
        Returns:
            ase Phonons object
                starts from scratch if self.remake_displacements
                otherwise reads cached displacements from data_dir/displacement_dir
        """
        calc = self.calculator
        atoms = self.atoms
        atoms.calc = calc

        ph = Phonons(
            atoms,
            calc,
            supercell=self.supercell_grid,
            delta=self.displacement_delta,
            name=self.displacement_dir,
        )
        if self.remake_displacements:
            ph.clean()
        ph.run()

        ph.read(acoustic=True)
        return ph

    @property
    def phonon_dos(self):
        """
        Returns:
            {'E0' : 0 K internal energy (eV/cell),
             'dos' :
                [{'E' : energy level (eV),
                  'dos' : phonon DOS at E (float)}]
                  }

            reads from data_dir/fjson_dos if available and not remake_dos
                otherwise writes to data_dir/fjson_dos

        """
        fjson = self.fjson_dos
        if not self.remake_dos and os.path.exists(fjson):
            return read_json(fjson)
        ph = self.phonons
        phonon_energies, phonon_dos = ph.dos(
            kpts=self.kpts_for_dos, npts=self.npts_for_dos, delta=self.dos_delta
        )
        out = {
            "dos": [
                {"E": phonon_energies[i], "dos": phonon_dos[i]}
                for i in range(len(phonon_energies))
            ],
            "E0": self.E0,
        }
        write_json(out, fjson)
        return read_json(fjson)


class Helmholtz(object):
    def __init__(
        self,
        phonon_dos,
        temperatures=np.linspace(0, 2000, 100),
        formula_units=1,
        fjson="helmholtz.json",
        remake=False,
    ):
        """
        Args:
            phonon_dos (HarmonicPhonons.phonon_dos)
                {'E0' : 0 K internal energy (eV/cell),
                'dos' :
                    [{'E' : energy level (eV),
                    'dos' : phonon DOS at E (float)}]
                    }

            temperatures (np.array)
                list of temperatures (K)

            formula_units (int)
                not sure what this is (?)

            fjson (str)
                save data to fjson (full relative path should be passed)

            remake (bool)
                whether to remake the data

        """
        self.temperatures = temperatures

        self.E0 = phonon_dos["E0"]

        self.phonon_energies = [
            phonon_dos["dos"][i]["E"] for i in range(len(phonon_dos["dos"]))
        ]
        self.phonon_dos = [
            phonon_dos["dos"][i]["dos"] for i in range(len(phonon_dos["dos"]))
        ]

        self.formula_units = formula_units
        self.fjson = fjson
        self.remake = remake

    @property
    def thermo(self):
        """
        Returns:
            ase CrystalThermo object
        """
        return CrystalThermo(
            phonon_energies=self.phonon_energies,
            phonon_DOS=self.phonon_dos,
            potentialenergy=self.E0,
            formula_units=self.formula_units,
        )

    @property
    def free_energies(self):
        """
        Returns:
            {'data' :
                [{'T' : temperature (K),
                  'F' : Helmholtz free energy (eV/cell)}]

            reads from fjson if available and not remake
                otherwise writes to fjson
        """
        fjson = self.fjson
        if not self.remake and os.path.exists(fjson):
            return read_json(fjson)
        temperatures = self.temperatures
        thermo = self.thermo
        Fs = [thermo.get_helmholtz_energy(temperature=T) for T in temperatures]
        Fs[0] = self.E0
        out = {
            "data": [
                {"T": temperatures[i], "F": Fs[i]} for i in range(len(temperatures))
            ]
        }
        write_json(out, fjson)
        return read_json(fjson)


class Gibbs(object):
    def __init__(
        self,
        structure,
        scales=np.linspace(0.95, 1.05, 11),
        supercell_grid=(1, 1, 1),
        calculator=CHGNetCalculator(),
        displacement_delta=5e-2,
        kpts_for_dos=(40, 40, 40),
        npts_for_dos=3000,
        dos_delta=5e-4,
        temperatures=np.linspace(0, 2000, 100),
        data_dir="data",
        displacement_dir="phonons",
        fjson_dos="phonon_dos.json",
        fjson_helmholtz="helmholtz.json",
        fjson_gibbs="gibbs.json",
        remake_displacements=False,
        remake_dos=False,
        remake_helmholtz=False,
        remake_gibbs=False,
    ):
        """
        Args:
            structure (pymatgen.Structure)
                The structure for which to calculate the Gibbs free energy

            scales (np.array)
                Isotropic volume scaling for QHA
                    what fractions of the equilibrium volume do we want to assess

            supercell_grid (tuple)
                The supercell grid for the phonon calculations

            calculator (ase Calculator)
                The calculator to use for the phonon calculations

            displacement_delta (float)
                The displacement delta for the phonon calculations (for finite displacements method)

            kpts_for_dos (tuple)
                The k-points for the DOS calculations

            npts_for_dos (int)
                The number of points for the DOS calculations

            dos_delta (float)
                The delta for the DOS calculations

            temperatures (np.array)
                The temperatures for the Gibbs free energy calculations

            data_dir (str)
                The directory where the data is stored

            displacement_dir (str):
                The directory where the displacements are stored
                    within data_dir

            fjson_dos (str)
                The filename for the phonon DOS data
                    within data_dir

            fjson_helmholtz (str)
                The filename for the Helmholtz free energy data
                    within data_dir

            fjson_gibbs (str)
                The filename for the Gibbs free energy data
                    within data_dir

            remake_displacements (bool)
                Whether to remake the displacements

            remake_dos (bool)
                Whether to remake the DOS

            remake_helmholtz (bool)
                Whether to remake the Helmholtz free energy

            remake_gibbs (bool)
                Whether to remake the Gibbs free energy
        """

        self.scales = scales
        self.structure = structure
        self.supercell_grid = supercell_grid
        self.calculator = calculator
        self.displacement_delta = displacement_delta
        self.kpts_for_dos = kpts_for_dos
        self.npts_for_dos = npts_for_dos
        self.dos_delta = dos_delta
        self.temperatures = temperatures
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        self.displacement_dir = displacement_dir
        self.fjson_dos = fjson_dos
        self.fjson_helmholtz = fjson_helmholtz
        self.fjson_gibbs = os.path.join(data_dir, fjson_gibbs)
        self.remake_displacements = remake_displacements
        self.remake_dos = remake_dos
        self.remake_helmholtz = remake_helmholtz
        self.remake_gibbs = remake_gibbs

    @property
    def structures(self):
        """
        Returns:
            list of isotropically strained structures

        """
        scales = self.scales
        structure = self.structure
        return [
            StrucTools(structure).scale_structure(scale_factor=scale)
            for scale in scales
        ]

    @property
    def volumes(self):
        """
        Returns:
            list of volumes (A**3) for the structures
        """
        structures = self.structures
        return [StrucTools(structure).structure.volume for structure in structures]

    @property
    def harmonic(self):
        """
        Returns:
            dict
                {scaling (float) : HarmonicPhonons object for structure at that scaling}
        """
        structures = self.structures
        scales = self.scales
        return {
            scales[i]: HarmonicPhonons(
                structure=structures[i],
                supercell_grid=self.supercell_grid,
                calculator=self.calculator,
                displacement_delta=self.displacement_delta,
                kpts_for_dos=self.kpts_for_dos,
                npts_for_dos=self.npts_for_dos,
                dos_delta=self.dos_delta,
                data_dir=self.data_dir,
                displacement_dir=self.displacement_dir + "-%s" % str(scales[i]),
                fjson_dos=self.fjson_dos.replace(".json", "-%s.json" % str(scales[i])),
                remake_displacements=self.remake_displacements,
                remake_dos=self.remake_dos,
            )
            for i in range(len(structures))
        }

    @property
    def dos(self):
        """
        Returns:
            dict
                {scaling (float) :
                    {'E0' : 0 K internal energy (eV/cell),
                    'dos' :
                        [{'E' : energy level (eV),
                        'dos' : phonon DOS at E (float)}]
                    }
                }
        """
        harmonic = self.harmonic
        return {scale: harmonic[scale].phonon_dos for scale in harmonic}

    @property
    def helmholtz(self):
        """
        Returns:
            dict
                {volume (A**3) :
                    {'data' :
                        [{'T' : temperature (K),
                        'F' : Helmholtz free energy (eV/cell)}]
                    }
                }
        """
        scales = self.scales
        volumes = self.volumes
        dos = self.dos
        return {
            volumes[i]: Helmholtz(
                phonon_dos=dos[scales[i]],
                temperatures=self.temperatures,
                fjson=os.path.join(
                    self.data_dir,
                    self.fjson_helmholtz.replace(".json", "-%s.json" % str(scales[i])),
                ),
                remake=self.remake_helmholtz,
            ).free_energies
            for i in range(len(volumes))
        }

    @property
    def gibbs(self):
        """
        Returns:
            {'data' :
                [{'T' : temperature (K),
                'G' : Gibbs free energy (eV/cell)}]
            }
        """
        fjson = self.fjson_gibbs
        if not self.remake_gibbs and os.path.exists(fjson):
            return read_json(fjson)
        temperatures = self.temperatures
        volumes = self.volumes
        Fs = self.helmholtz
        Gs = []
        for i in range(len(temperatures)):
            T = temperatures[i]
            F = [Fs[vol]["data"][i]["F"] for vol in volumes]
            V = [float(vol) for vol in volumes]
            eos = Murnaghan(V, F)
            try:
                eos.fit()
                min_F = eos.e0
                Gs.append({"T": T, "G": min_F})
            except:
                print("Failed to fit Murnaghan EOS at T = %s K" % T)
                continue
        out = {"data": Gs}
        write_json(out, fjson)
        return read_json(fjson)


def main():
    plot_Gs()
    return


if __name__ == "__main__":
    main()
