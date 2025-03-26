# import standard libraries and functions used in the workflow
import os
from shutil import copyfile
from pydmclab.data.configs import load_base_configs

from pydmclab.hpc.helpers import (
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
    get_results_with_slabs,
    get_adsorbed_slabs,
    get_adsorption_energy_results
)

from pydmclab.utils.handy import read_json, write_json

# import any custom functions that might be needed for this workflow below
#  (ie non-default helpers)
from pydmclab.hpc.helpers import (
    check_slabs,
    set_magmoms_from_template,
)

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

# any configurations related to LaunchTools
#  see pydmclab.hpc.helpers.get_launch_configs
#  see pydmclab.data.data._hpc_configs.yaml (LAUNCH_CONFIGS)
#  see pydmclab.hpc.launch.LaunchTools
LAUNCH_CONFIGS = get_launch_configs(
    n_afm_configs=0,
    override_mag="{{OVERRIDE_MAG}}",
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
    custom_calc_list=['gga-relax', 'gga-static', 'gga-static_dipole', 'gga-static_ldipole'],
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
    cores_per_node=32,
    walltime_in_hours=48,
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
#
#  kpoint grid is based on previous calculations and is not necessarily the best choice
#  consider additional convergence testing if you are unsure of what is best for your system
INCAR_MODS = {"all-all": {"NELM": 60}}
KPOINTS_MODS = {"all-all": {"grid": [5, 5, 1]}}
POTCAR_MODS = None

VASP_CONFIGS = get_vasp_configs(
    standard="dmc",
    dont_relax_cell=True,
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
    analyze_trajectory=False,
    analyze_mag=False,
    analyze_charge=False,
    analyze_dos=False,
    analyze_bonding=False,
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
# if the template structure is magnetic, `override_mag` should be set
GEN_MAGMOMS = (
    True
    if (LAUNCH_CONFIGS["n_afm_configs"] or LAUNCH_CONFIGS["override_mag"])
    else False
)

# What kind of molecule do we want to adsorb?
# Can either be a string or a list of strings
# Example: 'O' or ['O','N','H']

ADSORBATE_TYPE = 'O'

def main():
    # make a submission script so you can execute launcher.py on the cluster
    remake_sub_for_launcher = False

    # remake slabs? print slabs summary?
    remake_slabs = False
    print_slabs_check = True

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

    # remake results with adsorption energies
    remake_ads_res = True

    # remake submission script for launcher?
    # make one if it doesn't exist
    if (
        not os.path.exists(os.path.join(SCRIPTS_DIR, "sub_launcher.sh"))
        or remake_sub_for_launcher
    ):
        make_sub_for_launcher()
    
    slabs = get_adsorbed_slabs(
        adsorbate_type = ADSORBATE_TYPE,
        data_dir = DATA_DIR, 
        slab_dir = None,
        selective_dynamics = True,
        height = 0.9,
        super_cell = [2,2,1],
        savename = 'ads_slabs.json',
        remake = remake_slabs
        )

    if print_slabs_check:
        check_slabs(slabs)

    # If your template ground state structure is magnetic, magmoms should be set via `override_mag`
    if GEN_MAGMOMS:
        magmoms = set_magmoms_from_template(
            strucs=slabs,
            data_dir=DATA_DIR,
            savename="magmoms.json",
            remake=remake_magmoms,
        )
        if print_magmoms_check:
            check_magmoms(strucs=slabs, magmoms=magmoms)
    else:
        magmoms = None

    # get launch directories dictionary
    #  and make the launch directories
    #  this rarely gets replaced with a custom function
    #  we're setting up our directory tree here
    #  convenient to do this the pydmclab way
    launch_dirs = get_launch_dirs(
        strucs=slabs,
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

    slab_result = get_results_with_slabs(
        data_dir = DATA_DIR, 
        remake = False, 
        ref_dir = None,
        savename = 'results_with_slabs.json'
    )

    ads_res = get_adsorption_energy_results(
        data_dir = DATA_DIR,
        slab_dir = None,
        remake = remake_ads_res,
        savename = 'results.json'
    )
    
    if print_thermo_results_check:
        check_thermo_results(thermo)
        
    return


if __name__ == "__main__":
    main()
