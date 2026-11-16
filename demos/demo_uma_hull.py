import os

from pymatgen.core import Lattice, Structure

from pydmclab.core.struc import StrucTools
from pydmclab.core.energies import UMAFormationEnergy
from pydmclab.core.hulls import UMAHull

"""
Purpose:
    Evaluate the stability of new compounds against UMA reference data
        (UMA formation energies for MP structures, packaged w/ pydmclab)

    Workflow:
        - relax a structure w/ UMA (atoms + cell) and get its total energy
        - UMAFormationEnergy applies an MP2020-style (OMat24) correction and returns Ef (eV/atom)
        - UMAHull places Ef on the hull of every UMA reference compound in that chemical space

    Compatible UMA settings (new calculations must match the reference data):
        model = uma-s-1p2p1
        task = omat
        relaxation = atoms and cell relaxed
        convergence = fmax < 0.03 eV/A

    This demo has three parts (turn each on/off in main()):
        1) a hypothetical LiCoO2 polymorph w/ a made-up UMA energy (no UMA needed)
        2) three made-up compounds w/ made-up formation energies, evaluated together as a list
        3) relaxing LiMnO2 w/ UMA, then evaluating it (needs fairchem-core + the UMA checkpoint)

Unit tests:
    - see pydmclab/tests/test_energies.py and pydmclab/tests/test_hulls.py
"""

# UMA settings that match the reference data
UMA_MODEL = "uma-s-1p2p1"
UMA_TASK = "omat"
FMAX = 0.03

TEST_STRUC = os.path.join("cifs", "mp-18767-LiMnO2.cif")


def print_hull_results(hull_out):
    for results in hull_out:
        print(f"Stability results for {results['formula']}:")
        for k, v in results.items():
            print(f"\t{k}: {v}")


def demo_hypothetical_polymorph():
    """
    A hypothetical LiCoO2 polymorph, 30 meV/atom below UMA's LiCoO2 ground state
        structure can be a Structure, Structure.as_dict(), or a path to a structure file
    """
    structure = Structure.from_spacegroup(
        "R-3m",
        Lattice.hexagonal(2.82, 14.05),
        ["Li", "Co", "O"],
        [[0, 0, 0], [0, 0, 0.5], [0, 0, 0.24]],
    )

    # UMA total energy (eV/atom), uncorrected
    E_per_atom = -5.7276 - 0.030

    # formation energy w/ MP2020-style corrections (uses packaged UMA elemental references)
    Ef = UMAFormationEnergy().formation_energy(structure, E_per_atom)
    formula = StrucTools(structure).compact_formula

    # a single (formula, Ef) or a list of them, [(formula, Ef), ...]
    hull_out = UMAHull((formula, Ef)).hull_output_data
    print_hull_results(hull_out)

    return hull_out


def demo_compound_list():
    """
    Three made-up compounds (not in the UMA reference data) w/ made-up formation energies,
        evaluated together as a list of (formula, Ef)

    Compounds passed together go on the same hull:
        - LiCo3O5 and Li2Co7O11 are both in Li-Co-O, so they compete w/ each other
            Li2Co7O11 alone would be stable (Ed = -0.004 eV/atom), but LiCo3O5 makes it
            unstable; its rxn includes 'Co3Li1O5 (input 0)'
        - Ca3Ti2S7 is in an unrelated chemical space, so it is unaffected by the other two
    """
    # [(formula, Ef (eV/atom)), ...]
    compounds = [
        ("LiCo3O5", -1.72),
        ("Li2Co7O11", -1.69),
        ("Ca3Ti2S7", -2.02),
    ]

    # results come back in the same order as the input list
    hull_out = UMAHull(compounds).hull_output_data
    print_hull_results(hull_out)

    return hull_out


def demo_uma_relaxation():
    """
    Relax LiMnO2 w/ UMA, then evaluate its stability
    """
    from pydmclab.mlp.fairchem.dynamics import FAIRChemRelaxer

    initial_structure = StrucTools(TEST_STRUC).structure

    relaxer = FAIRChemRelaxer(name_or_path=UMA_MODEL, task_name=UMA_TASK)
    results = relaxer.relax(
        initial_structure, fmax=FMAX, steps=500, relax_cell=True, verbose=False
    )
    if not results["converged"]:
        print(f"WARNING: relaxation did not reach fmax < {FMAX} eV/A")

    relaxed_structure = StrucTools(results["final_structure"]).structure
    E_per_atom = results["final_energy"] / len(relaxed_structure)

    Ef = UMAFormationEnergy().formation_energy(relaxed_structure, E_per_atom)
    formula = StrucTools(relaxed_structure).compact_formula

    hull_out = UMAHull((formula, Ef)).hull_output_data
    print_hull_results(hull_out)

    return hull_out


def main():
    run_hypothetical_polymorph = True
    run_compound_list = True
    # requires fairchem-core + access to the UMA checkpoint
    run_uma_relaxation = False

    if run_hypothetical_polymorph:
        print("\n1) hypothetical LiCoO2 polymorph")
        out_1 = demo_hypothetical_polymorph()

    if run_compound_list:
        print("\n2) three made-up compounds as a list")
        out_2 = demo_compound_list()

    if run_uma_relaxation:
        print("\n3) UMA-relaxed LiMnO2")
        out_3 = demo_uma_relaxation()

    return


if __name__ == "__main__":
    main()
