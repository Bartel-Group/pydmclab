import os
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
from pydmclab.core.comp import CompTools
from pydmclab.core.struc import StrucTools
from pydmclab.hpc.analyze import AnalyzeVASP

from shutil import copyfile

# where is this file
SCRIPTS_DIR = os.getcwd()

# where are my calculations going to live (maybe on scratch)
CALCS_DIR = SCRIPTS_DIR.replace("scripts", "calcs")

# where is my data going to live
DATA_DIR = SCRIPTS_DIR.replace("scripts", "data")

for d in [CALCS_DIR, DATA_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

# copy our passer.py file to your scripts_dir. if you want to use a custom passer, just set this = True and put your passer.py in the scripts dir
USER_NAME = "cbartel"
CUSTOM_PASSER = False
if not CUSTOM_PASSER:
    copyfile(
        "/home/cbartel/%s/bin/pydmclab/pydmclab/hpc/passer.py" % USER_NAME, "passer.py"
    )

CUSTOM_COLLECTOR = False
if not CUSTOM_COLLECTOR:
    copyfile(
        "/home/cbartel/%s/bin/pydmclab/pydmclab/hpc/collector.py" % USER_NAME,
        "collector.py",
    )

# if you need data from MP as a starting point (often the case), you need your API key
# if API_KEY = None, it will expect you've configured your API KEY using pmg config --add PMG_MAPI_KEY <USER_API_KEY>
API_KEY = None

# what to query MP for (if you need MP data)
## e.g., 'MnO2', ['MnO2', 'TiO2'], 'Ca-Ti-O, etc
COMPOSITIONS = None

# any configurations related to LaunchTools
LAUNCH_CONFIGS = get_launch_configs(
    n_afm_configs=0,
    override_mag=False,
    ID_specific_vasp_configs=None,
)

# any configurations related to SubmitTools
SUB_CONFIGS = get_sub_configs(
    relaxation_xcs=["gga"],
    static_addons={"gga": ["lobster"]},
    prioritize_relaxes=True,
    custom_calc_list=None,
    restart_these_calcs=[],
    start_with_loose=False,
    machine="msi",
    mpi_command="mpirun",
    vasp_version=6,
    submit_calculations_in_parallel=False,
)

# any configurations related to Slurm
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
VASP_CONFIGS = get_vasp_configs(
    standard="dmc",
    dont_relax_cell=False,
    special_functional=None,
    incar_mods={},
    kpoints_mods={},
    potcar_mods={},
    lobster_configs={"COHPSteps": 2000, "reciprocal_kpoints_density_for_lobster": 100},
    bs_configs={"bs_symprec": 0.1, "bs_line_density": 20},
    flexible_convergence_criteria=False,
    compare_static_and_relax_energies=0.1,
)

# any configurations related to AnalyzeBatch
ANALYSIS_CONFIGS = get_analysis_configs(
    only_xc=None,
    only_calc="static",
    analyze_calculations_in_parallel=False,
    analyze_structure=True,
    analyze_trajectory=False,
    analyze_mag=False,
    analyze_charge=False,
    analyze_dos=False,
    analyze_bonding=False,
    exclude=[],
    remake_results=False,
)

CONFIGS = ANALYSIS_CONFIGS.copy()
CONFIGS.update(VASP_CONFIGS)
CONFIGS.update(SLURM_CONFIGS)
CONFIGS.update(SUB_CONFIGS)
CONFIGS.update(LAUNCH_CONFIGS)

CONFIGS = write_json(CONFIGS, os.path.join(SCRIPTS_DIR, "configs.json"))

# whether or not you want to generate MAGMOMs (only if you're running AFM)
GEN_MAGMOMS = True if LAUNCH_CONFIGS["n_afm_configs"] else False

# NOTE: the default is to use the imported functions from pydmclab.hpc.helpers
# You will often want to write your own "get_query" and/or "get_strucs" functions instead
# See below (or within pydmclab.hpc.helpers) for some more detailed docs


def get_custom_data(savename="custom.json", remake=False):
    fjson = os.path.join(DATA_DIR, savename)
    if not remake and os.path.exists(fjson):
        return read_json(fjson)

    d = {}
    write_json(d, fjson)
    return read_json(fjson)


def main():
    remake_sub_for_launcher = False

    remake_query = False
    print_query_check = True

    remake_strucs = False
    print_strucs_check = True

    remake_magmoms = False
    print_magmoms_check = True

    remake_launch_dirs = False
    print_launch_dirs_check = True

    remake_subs = True
    ready_to_launch = True

    remake_results = True
    print_results_check = True

    remake_gs = True
    print_gs_check = True

    remake_thermo_results = True
    print_thermo_results_check = True

    if remake_sub_for_launcher:
        make_sub_for_launcher()

    query = get_query(
        api_key=API_KEY,
        search_for=COMPOSITIONS,
        properties=None,
        max_Ehull=0.05,
        max_sites_per_structure=41,
        max_polymorph_energy=0.1,
        only_gs=False,
        include_structure=True,
        max_strucs_per_cmpd=1,
        include_sub_phase_diagrams=False,
        data_dir=DATA_DIR,
        savename="query.json",
        remake=remake_query,
    )
    if print_query_check:
        check_query(query)

    strucs = get_strucs(
        query=query,
        data_dir=DATA_DIR,
        savename="strucs.json",
        remake=remake_strucs,
    )
    if print_strucs_check:
        check_strucs(strucs)

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

    if remake_subs:
        submit_calcs(
            launch_dirs=launch_dirs,
            user_configs=CONFIGS,
            ready_to_launch=ready_to_launch,
            n_procs=CONFIGS["n_procs_for_submission"],
        )

    results = get_results(
        launch_dirs=launch_dirs,
        user_configs=CONFIGS,
        data_dir=DATA_DIR,
        remake=remake_results,
    )
    if print_results_check:
        check_results(results)

    gs = get_gs(results=results, data_dir=DATA_DIR, remake=remake_gs)

    if print_gs_check:
        check_gs(gs)

    thermo = get_thermo_results(
        results=results, gs=gs, data_dir=DATA_DIR, remake=remake_thermo_results
    )

    if print_thermo_results_check:
        check_thermo_results(thermo)

    return


if __name__ == "__main__":
    main()
