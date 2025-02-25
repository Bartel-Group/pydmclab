import os
from shutil import copyfile
from pydmclab.data.configs import load_base_configs

from pydmclab.hpc.helpers import (
    check_strucs,
    check_magmoms,
    get_launch_dirs,
    check_launch_dirs,
    submit_calcs,
    get_results,
    check_results,
    get_gs,
    check_gs,
    get_thermo_results,
    check_thermo_results,
    get_launch_configs,
    get_sub_configs,
    get_slurm_configs,
    get_vasp_configs,
    get_analysis_configs,
    make_sub_for_launcher,
)

from pymatgen.core import Structure, Lattice, Molecule
from pymatgen.core.surface import SlabGenerator
from pymatgen.analysis.adsorption import *
from pymatgen.core.surface import generate_all_slabs
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.structure import Structure
from pymatgen.core.surface import SlabGenerator

# set up some paths that will point to where your data/calculations will live
#  these are just defaults, you can change the `_DIR` variables to point to wherever you want
#
# The home directory path is used to point to your local copy of the pydmclab repo
#   pydmclab is assumed to be in /users/{number}/{username}/bin/pydmclab
#   and $HOME points to /users/{number}/{username}
HOME_PATH = os.environ["HOME"]
_, _, _, USER_NAME = HOME_PATH.split("/")
SCRATCH_PATH = os.path.join(os.environ["SCRATCH_GLOBAL"], USER_NAME)

# where is this file
SCRIPTS_DIR = os.getcwd()

# where are my calculations going to live (defaults to scratch)
CALCS_DIR = SCRIPTS_DIR.replace("scripts", "calcs").replace(HOME_PATH, SCRATCH_PATH)

# where is my data going to live
DATA_DIR = SCRIPTS_DIR.replace("scripts", "data")

# make our calcs dir and data dir
for d in [CALCS_DIR, DATA_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)


