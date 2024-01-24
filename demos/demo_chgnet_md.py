import os
import numpy as np
from pydmclab.core.struc import StrucTools
from pydmclab.mlp.chgnet_md import CHGNetRelaxer, CHGNetObserver
from pydmclab.hpc.helpers import get_query, get_strucs

API_KEY = "azRTXUxQ8u2qcyfl"

DATA_DIR = os.path.join("outputs", "mlp-relaxation", "chgnet")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

COMPS = ["IrO2"]


def main():
    query = get_query(
        comp=COMPS,
        api_key=API_KEY,
        only_gs=True,
        include_structure=True,
        supercell_structure=False,
        max_Ehull=0.05,
        max_sites_per_structure=65,
        max_strucs_per_cmpd=4,
        data_dir=DATA_DIR,
        savename="chgnet-relax-query.json",
        remake=True,
    )
    strucs = get_strucs(
        query=query,
        data_dir=DATA_DIR,
        savename="chgnet-relax-strucs.json",
        remake=True,
    )

    for cmpd in strucs:
        for mpid in strucs[cmpd]:
            st = StrucTools(strucs[cmpd][mpid])  # Load IrO2
            initial_structure = st.perturb(
                0.1
            )  # Perturb to get non-equilibrium structure
            chgnet_relaxer = CHGNetRelaxer(
                initial_structure=initial_structure
            )  # Initialize Relaxer
            chgnet_observer = CHGNetObserver(
                relaxer=chgnet_relaxer
            )  # Pass converged relaxer to observer

            print(f"\n{cmpd} {mpid}")
            print(
                f"CHGNetObserver took {len(chgnet_observer.trajectory)} steps to converge."
            )
            print(f"Initial energy: {chgnet_observer.energies[0]:.4f} eV/atom")
            print(f"Final energy: {chgnet_observer.energies[-1]:.4f} eV/atom")

    return strucs


if __name__ == "__main__":
    strucs = main()
