# import standard libraries and functions used in the workflow
import os
from shutil import copyfile
from pydmclab.data.configs import load_base_configs

from pydmclab.hpc.helpers import (
    get_query,
    check_query,
    get_strucs,
    check_strucs,
    get_magmoms,
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

from pydmclab.utils.handy import read_json, write_json

# import any custom functions that might be needed for this workflow below
#  (ie non-default helpers)
from pydmclab.hpc.helpers import get_strucs_from_cifs

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

# copy our passer.py file to your scripts_dir
#  if you want to use a custom passer, just set CUSTOM_PASSER = True and put your passer.py in the scripts dir
#  this file handles passing data between calculations after they finish
#  eg copying WAVECAR/CONTCAR, making INCAR changes based on magmoms, etc
#  see pydmclab.hpc.passer
CUSTOM_PASSER = False
if not CUSTOM_PASSER:
    copyfile(f"{HOME_PATH}/bin/pydmclab/pydmclab/hpc/passer.py", "passer.py")

# copy our collector.py file to your scripts_dir
#  if you want to use a custom collector, just set CUSTOM_COLLECTOR = True and put your collector.py in the scripts dir
#  this file handles collecting data from calculations after they finish
#  writes results.json file to each calc_dir to aggregate during analysis phase
#  see pydmclab.hpc.collector
CUSTOM_COLLECTOR = False
if not CUSTOM_COLLECTOR:
    copyfile(f"{HOME_PATH}/bin/pydmclab/pydmclab/hpc/collector.py", "collector.py")

# load our baseline configs
#  see pydmclab.data.data._hpc_configs.yaml
BASE_CONFIGS = load_base_configs()

# if you need data from MP as a starting point (often the case), you need your API key
#  see pydmclab.core.query.MPQuery
API_KEY = "{{API_KEY}}"

# what to query MP for (if you need MP data)
#  e.g., 'MnO2', ['MnO2', 'TiO2'], 'Ca-Ti-O, etc
#  see pydmclab.hpc.helpers.get_query
COMPOSITIONS = "{{COMPOSITIONS}}"

# any configurations related to LaunchTools
#  see pydmclab.hpc.helpers.get_launch_configs
#  see pydmclab.data.data._hpc_configs.yaml (LAUNCH_CONFIGS)
#  see pydmclab.hpc.launch.LaunchTools
LAUNCH_CONFIGS = get_launch_configs(
    n_afm_configs=0,
    override_mag=False,
    ID_specific_vasp_configs=None,
)

# any configurations related to SubmitTools
#  see pydmclab.hpc.helpers.get_sub_configs
#  see pydmclab.data.data._hpc_configs.yaml (SUB_CONFIGS)
#  see pydmclab.hpc.submit.SubmitTools
SUB_CONFIGS = get_sub_configs(
    relaxation_xcs=["gga"],
    static_addons={},
    prioritize_relaxes=True,
    start_with_loose=False,
    custom_calc_list=None,
    restart_these_calcs=None,
    submit_calculations_in_parallel=False,
    machine="msi",
    mpi_command="mpirun",
    vasp_version=6,
)

# any configurations related to SLURM (job execution on compute nodes)
#  see pydmclab.hpc.helpers.get_slurm_configs
#  see pydmclab.data.data._hpc_configs.yaml (SLURM_CONFIGS)
#  see pydmclab.hpc.submit.SubmitTools
SLURM_CONFIGS = get_slurm_configs(
    total_nodes=1,
    cores_per_node=8,
    walltime_in_hours=95,
    mem_per_core="all",
    partition="agsmall,msismall,msidmc",
    error_file="log.e",
    output_file="log.o",
    account="cbartel",
)

# any configurations related to VASPSetUp
#  see pydmclab.hpc.helpers.get_vasp_configs
#  see pydmclab.data.data._hpc_configs.yaml (VASP_CONFIGS)
#  see pydmclab.hpc.vasp.VASPSetUp
#  see pydmclab.hpc.sets.GetSet
INCAR_MODS = {"all-all": {"NELM": 60}}
KPOINTS_MODS = None
POTCAR_MODS = None

VASP_CONFIGS = get_vasp_configs(
    standard="dmc",
    dont_relax_cell=False,
    incar_mods=INCAR_MODS,
    kpoints_mods=KPOINTS_MODS,
    potcar_mods=POTCAR_MODS,
    flexible_convergence_criteria=False,
    compare_static_and_relax_energies=0.1,
    special_functional=False,
    COHPSteps=2000,
    reciprocal_kpoints_density_for_lobster=100,
    bandstructure_symprec=0.1,
    bandstructure_kpoints_line_density=20,
)

# any configurations related to AnalyzeVASP, AnalyzeBatch
#  see pydmclab.hpc.helpers.get_analysis_configs
#  see pydmclab.data.data._hpc_configs.yaml (ANALYSIS_CONFIGS)
#  see pydmclab.hpc.analyze.AnalyzeVASP
#  see pydmclab.hpc.analyze.AnalyzeBatch
ANALYSIS_CONFIGS = get_analysis_configs(
    only_calc="static",
    only_xc=None,
    analyze_structure=True,
    analyze_mag=False,
    analyze_charge=False,
    analyze_dos=False,
    analyze_bonding=False,
    analyze_phonons=False,
    exclude=None,
    remake_results=False,
    verbose=False,
)

# update our configs based on the specific configs we've nust created
CONFIGS = BASE_CONFIGS.copy()
CONFIGS.update(VASP_CONFIGS)
CONFIGS.update(SLURM_CONFIGS)
CONFIGS.update(SUB_CONFIGS)
CONFIGS.update(LAUNCH_CONFIGS)
CONFIGS.update(ANALYSIS_CONFIGS)

# write our configs to a file
CONFIGS = write_json(CONFIGS, os.path.join(SCRIPTS_DIR, "configs.json"))

# whether or not you want to generate MAGMOMs (only if you're running AFM)
# if you are setting `override_mag` at this point, you need to write your own function to generate magmoms
GEN_MAGMOMS = (
    True
    if (LAUNCH_CONFIGS["n_afm_configs"] or LAUNCH_CONFIGS["override_mag"])
    else False
)


def main():
    # remake a submission script so you can execute launcher.py on the cluster
    remake_sub_for_launcher = False

    # remake query? print query summary?
    remake_query = False
    print_query_check = True

    # remake strucs? print strucs summary?
    remake_strucs = False
    print_strucs_check = True

    # remake magmoms? print magmoms summary?
    remake_magmoms = False
    print_magmoms_check = True

    # remake launch directories? print launch_dirs summary?
    remake_launch_dirs = False
    print_launch_dirs_check = True

    # remake submission scripts? ready to launch them?
    remake_subs = True
    ready_to_launch = True

    # remake compiled results? print results summary?
    remake_results = True
    print_results_check = True

    # remake ground-state data? print gs summary?
    remake_gs = True
    print_gs_check = True

    # remake thermo results? print thermo summary?
    remake_thermo_results = True
    print_thermo_results_check = True

    # remake submission script for launcher?
    # make one if it doesn't exist
    if (
        not os.path.exists(os.path.join(SCRIPTS_DIR, "sub_launcher.sh"))
        or remake_sub_for_launcher
    ):
        make_sub_for_launcher()

    # get data from Materials Project
    #  or replace with your own data retrieval function (i.e. get_strucs_from_cifs)
    #  pay attention to arguments here. are these the right filters?
    query = get_query(
        api_key=API_KEY,
        search_for=COMPOSITIONS,
        max_Ehull=0.05,
        max_polymorph_energy=0.1,
        max_strucs_per_cmpd=1,
        max_sites_per_structure=41,
        include_sub_phase_diagrams=False,
        include_structure=True,
        properties=None,
        data_dir=DATA_DIR,
        savename="query.json",
        remake=remake_query,
    )
    if print_query_check:
        check_query(query)

    # retrieve the structures from your query
    #  replace with your own function for making structures
    #  if you don't want to just use the structures as is from the query
    #  eg make substitutions, create defects, make supercells, etc
    strucs = get_strucs(
        query=query,
        data_dir=DATA_DIR,
        savename="strucs.json",
        remake=remake_strucs,
        force_supercell=True,
    )
    if print_strucs_check:
        check_strucs(strucs)

    # make magmoms if you're running AFM calcs
    #  replace with your own function for making magmoms if you'd like or of you're setting override_mag to True
    if GEN_MAGMOMS:
        magmoms = get_magmoms(
            strucs=strucs,
            max_afm_combos=50,
            treat_as_nm=None,
            data_dir=DATA_DIR,
            savename="magmoms.json",
            remake=remake_magmoms,
        )
        if print_magmoms_check:
            check_magmoms(strucs=strucs, magmoms=magmoms)
    else:
        magmoms = None

    # get launch directories dictionary
    #  and make the launch directories
    #  this rarely gets replaced with a custom function
    #  we're setting up our directory tree here
    #  convenient to do this the pydmclab way
    launch_dirs = get_launch_dirs(
        strucs=strucs,
        magmoms=magmoms,
        user_configs=CONFIGS,
        make_launch_dirs=True,
        data_dir=DATA_DIR,
        calcs_dir=CALCS_DIR,
        savename="launch_dirs.json",
        remake=remake_launch_dirs,
    )
    if print_launch_dirs_check:
        check_launch_dirs(launch_dirs)

    # write/update submission scripts in each launch directory
    #  submit them if you're ready
    if remake_subs:
        submit_calcs(
            launch_dirs=launch_dirs,
            user_configs=CONFIGS,
            ready_to_launch=ready_to_launch,
            n_procs=CONFIGS["n_procs_for_submission"],
        )

    # analyze calculations
    #  see if they're done
    #  compile their results
    #  note: a lot of the analysis happens within each submission script, so this should be fast
    results = get_results(
        launch_dirs=launch_dirs,
        user_configs=CONFIGS,
        data_dir=DATA_DIR,
        remake=remake_results,
    )
    if print_results_check:
        check_results(results)

    # get smaller dataset just for your calculated ground-state entries at each composition
    gs = get_gs(
        results=results,
        calc_types_to_search=("static",),
        data_dir=DATA_DIR,
        remake=remake_gs,
    )
    if print_gs_check:
        check_gs(gs)

    # get thermo results for all entries (smaller than results, bigger than gs)
    thermo = get_thermo_results(
        results=results, gs=gs, data_dir=DATA_DIR, remake=remake_thermo_results
    )
    if print_thermo_results_check:
        check_thermo_results(thermo)

    return


if __name__ == "__main__":
    main()
