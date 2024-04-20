import multiprocessing as multip
import os
from pydmclab.hpc.launch import LaunchTools
from pydmclab.hpc.submit import SubmitTools
from pydmclab.hpc.analyze import AnalyzeVASP, AnalyzeBatch
from pydmclab.core.comp import CompTools
from pydmclab.core.query import MPQuery, MPLegacyQuery
from pydmclab.core.struc import StrucTools
from pydmclab.core.mag import MagTools
from pydmclab.core.energies import ChemPots, FormationEnthalpy, MPFormationEnergy
from pydmclab.utils.handy import read_json, write_json
from pydmclab.data.configs import load_partition_configs


def get_vasp_configs(
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
):
    """
    configs related to particular VASP calculations

    Args:
        standard (str):
            "dmc" for group standard, "mp" for Materials Project standard

        dont_relax_cell (bool)
            if True, sets ISIF = 2 for all calculations

        special_functional (str)
            if you're not using r2SCAN, PBE, or HSE06, specify here as str (must be acceptable value for GGA, METAGGA, etc in INCAR)

        incar_mods (dict):
            modifications to INCAR
                {xc-calc (str) : {INCAR tag (str) : value (str, bool, float, int)}}
                    use xc = 'all' or calc = 'all' to apply to all xcs or all calcs

            e.g., {'metagga-all' : {'LMAXMIX' : 0.1,
                                    'EDIFF' : 1e-7},
                    'gga-relax' : {'EDIFFG' : 1e-4}}

        kpoints_mods (dict):
            modifications to KPOINTS
                {xc-calc (str) : {KPOINTS generator tag ('grid', 'auto', 'density', 'reciprocal_density') : value (float, list)}}

            e.g., {'gga-all' : {'grid' : [8, 8, 8]},

        potcar_mods (dict):
            modifications to POTCAR
                {xc-calc (str) : {element (str) : potcar symbol (str)}}

            e.g.,
                {'all-all' : {'W' : 'W_pv'}}

        lobster_configs (dict):
            modifications to LOBSTER (only affects 'all-lobster' xc-calcs)

        bs_configs (dict):
            modifications to band structure calculations (only affects 'all-bs' xc-calcs)
                {'bs_symprec' : float, 'bs_line_density' : int}
                    symprec for finding primitive cell for band structure calculations, line density for band structure kpoints

        flexible_convergence_criteria (bool):
            if True, reduce EDIFF/EDIFFG if convergence is taking very many steps
            if False, only update NELM and NSW if convergence is taking very many steps

        compare_static_and_relax_energies (float):
            if the difference between static and relax energies is greater than this, rerun the static calculation
                if False, don't compare the energies

    """
    vasp_configs = {}
    if dont_relax_cell:
        incar_mods.update({"all-all": {"ISIF": 2}})
    vasp_configs["standard"] = standard
    vasp_configs["incar_mods"] = incar_mods
    vasp_configs["kpoints_mods"] = kpoints_mods
    vasp_configs["potcar_mods"] = potcar_mods
    vasp_configs["functional"] = special_functional
    vasp_configs["bs_symprec"] = bs_configs["bs_symprec"]
    vasp_configs["bs_line_density"] = bs_configs["bs_line_density"]
    vasp_configs["COHPSteps"] = lobster_configs["COHPSteps"]
    vasp_configs["flexible_convergence_criteria"] = flexible_convergence_criteria
    vasp_configs["relax_static_energy_diff_tol"] = compare_static_and_relax_energies

    return vasp_configs


def get_slurm_configs(
    total_nodes=1,
    cores_per_node=8,
    walltime_in_hours=95,
    mem_per_core="all",
    partition="agsmall,msismall,msidmc",
    error_file="log.e",
    output_file="log.o",
    account="cbartel",
):
    """

    configs related to HPC settings for each submission script


    Args:
        total_nodes (int):
            how many nodes to run each VASP job on

        cores_per_node (int):
            how many cores per node to use for each VASP job

        walltime_in_hours (int):
            how long to run each VASP job

        mem_per_core (str):
            if 'all', try to use all avaiable mem; otherwise use specified memory (int, MB) per core

        partition (str):
            what part of the cluster to run each VASP job on

        error_file (str):
            where to send each VASP job errors

        output_file (str):
            where to send each VASP job outputs

        account (str):
            what account to charge for your VASP jobs

    Returns:
        {slurm config name : slurm config value}
    """
    slurm_configs = {}

    slurm_configs["nodes"] = total_nodes
    slurm_configs["ntasks"] = int(total_nodes * cores_per_node)

    if account == "cbartel":
        # convert MSI to minutes
        slurm_configs["time"] = int(walltime_in_hours * 60)

    if total_nodes > 1:
        if "small" in partition:
            print("WARNING: cant use small partition on > 1 node; switching to large")
        partition = partition.replace("small", "large")

    slurm_configs["partition"] = partition

    slurm_configs["error_file"] = error_file
    slurm_configs["output_file"] = output_file
    slurm_configs["account"] = account

    if total_nodes > 4:
        print("WARNING: are you sure you need more than 4 nodes??")

    if (total_nodes > 1) and (cores_per_node < 32):
        print("WARNING: this seems like a small job. are you sure you need > 1 node??")

    # figure out how much memory to use per core
    if mem_per_core == "all":
        partitions = load_partition_configs()
        if partition in partitions:
            mem_per_cpu = partitions[partition]["mem_per_core"]
            if isinstance(mem_per_cpu, str):
                if "GB" in mem_per_cpu:
                    mem_per_cpu = int(mem_per_cpu.replace("GB", "")) * 1000
        elif partition == "agsmall,msidmc":
            mem_per_cpu = 4000
        else:
            mem_per_cpu = 1900
    else:
        mem_per_cpu = mem_per_core

    slurm_configs["mem-per-cpu"] = str(int(mem_per_cpu)) + "M"
    return slurm_configs


def get_sub_configs(
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
):
    """

    configs related to generating submission scripts

    Args:
        relaxation_xcs (list):
            list of xcs you want to at least run relax + static for
                e.g., ['gga', 'metaggau']
        static_addons (dict):
            {xc : [list of additional calculations to run after static]}
                e.g., {'gga' : ['lobster', 'parchg']}

        relaxation_xcs and static_addons generate a calc_list (order of xc-calcs to be executed)
            e.g., ['gga-relax', 'gga-static', 'metagga-relax', 'metagga-static', 'metagga-lobster', 'metagga-parchg']

        prioritize_relaxes (bool):
            if True, run relaxes before static addons

        custom_calc_list (list):
            if you don't want to autogenerate a calc_list, you can specify the full list you want to run here

        restart_these_calcs (list):
            list of xc-calcs you want to start over (e.g., ['gga-lobster'])

        start_with_loose (bool):
            prepend your calc_list with a loose calc if True

        submit_calculations_in_parallel (bool or int):
            whether to prepare submission scripts in parallel or not
                False: use 1 processor
                True: use all available processors - 1
                int: use that many processors
            if this is not False, you should not run this on a login node

        machine (str):
            name of supercomputer

        mpi_command (str):
            the command to use for mpi (eg mpirun, srun, etc)

        vasp_version (int):
            5 for 5.4.4 or 6 for 6.4.1



    Returns:
        {config_name : config_value}

    """
    sub_configs = {}

    if not submit_calculations_in_parallel:
        n_procs = 1
    else:
        if submit_calculations_in_parallel is True:
            n_procs = multip.cpu_count() - 1
        else:
            n_procs = int(submit_calculations_in_parallel)

    sub_configs["n_procs_for_submission"] = n_procs
    sub_configs["mpi_command"] = mpi_command

    if custom_calc_list:
        sub_configs["custom_calc_list"] = custom_calc_list

    sub_configs["fresh_restart"] = restart_these_calcs
    sub_configs["start_with_loose"] = start_with_loose
    sub_configs["relaxation_xcs"] = relaxation_xcs
    sub_configs["static_addons"] = static_addons
    sub_configs["machine"] = machine
    sub_configs["vasp_version"] = vasp_version

    if prioritize_relaxes:
        sub_configs["run_static_addons_before_all_relaxes"] = False
    else:
        sub_configs["run_static_addons_before_all_relaxes"] = True

    return sub_configs


def get_launch_configs(
    n_afm_configs=0,
    override_mag=False,
    ID_specific_vasp_configs=None,
):
    """

    configs related to launching chains of calculations

    Args:

        n_afm_configs (int):
            number of antiferromagnetic configurations to run for each structure (0 if you don't want to run AFM)

        override_mag (bool):
            if True, let user decide mag completely (rather than auto figuring out whether magnetic)
                NOTE: not sure if this is working

        ID_specific_vasp_configs (dict):
            {<formula_indicator>_<struc_indicator> : {'incar_mods' : {<INCAR tag> : <value>}, {'kpoints' : <kpoints value>}, {'potcar' : <potcar value>}}

    Returns:
        dictionary of launch configurations
            {config param : config value}
    """

    if ID_specific_vasp_configs is None:
        ID_specific_vasp_configs = {}
    return {
        "override_mag": override_mag,
        "n_afm_configs": n_afm_configs,
        "ID_specific_vasp_configs": ID_specific_vasp_configs,
    }


def get_analysis_configs(
    only_calc="static",
    only_xc=None,
    analyze_calculations_in_parallel=False,
    analyze_structure=True,
    analyze_trajectory=False,
    analyze_mag=False,
    analyze_charge=False,
    analyze_dos=False,
    analyze_bonding=False,
    exclude=[],
):
    """

    configs related to parsing calculations and compiling results

    Args:

        only_calc (bool or str):
            if str, only analyze this calc (eg 'static')

        only_xc (bool or str):
            if str, only analyze this xc (eg 'gga')

        analyze_calculations_in_parallel (bool or int): whether to analyze calculation results in parallel or not
            - False: use 1 processor
            - True: use all available processors
            - int: use that many processors

        analyze_structure (bool, optional):
            True to include structure in your results

        analyze_trajectory (bool, optional):
            True to include trajectory in your results

        analyze_mag (bool, optional):
            True to include magnetization in your results

        analyze_charge (bool, optional):
            True to include bader charge + lobster charges + madelung in your results

        analyze_dos (bool, optional):
            True to include pdos, tdos in your results

        analyze_bonding (bool, optional):
            True to include tcohp, pcohp, tcoop, pcoop, tcobi, pcobi in your results

        exclude (list, optional):
            list of strings to exclude from analysis. Defaults to [].
                - overwrites other options
    Returns:
        dictionary of ANALYSIS_CONFIGS
            {'include_*' : True or False}
    """

    analysis_configs = {}

    if not analyze_calculations_in_parallel:
        n_procs = 1
    else:
        if analyze_calculations_in_parallel == True:
            n_procs = multip.cpu_count() - 1
        elif analyze_calculations_in_parallel == False:
            n_procs = 1
        else:
            n_procs = analyze_calculations_in_parallel

    analysis_configs["n_procs_for_analysis"] = n_procs

    includes = []
    if analyze_structure:
        includes.append("structure")

    if analyze_trajectory:
        includes.append("trajectory")

    if analyze_mag:
        includes.append("mag")

    if analyze_charge:
        includes.extend(["charge", "madelung"])

    if analyze_dos:
        includes.extend(["tdos", "pdos"])

    if analyze_bonding:
        includes.extend(["tcohp", "pcohp", "tcoop", "pcoop", "tcobi", "pcobi"])

    for include in includes:
        analysis_configs["include_" + include] = True

    if exclude:
        for ex in exclude:
            analysis_configs["include_" + ex] = False

    analysis_configs["only_calc"] = only_calc
    analysis_configs["only_xc"] = only_xc

    return analysis_configs


def get_query(
    api_key,
    search_for,
    properties=None,
    max_Ehull=0.05,
    max_sites_per_structure=41,
    max_polymorph_energy=0.1,
    only_gs=False,
    include_structure=True,
    max_strucs_per_cmpd=1,
    include_sub_phase_diagrams=False,
    data_dir=os.getcwd().replace("scripts", "data"),
    savename="query.json",
    remake=False,
):
    """
    Args:
        api_key (str)
            your API key (should be 32 characters)
        search_for (str or list)
            can either be:
                - a chemical system (str) of elements joined by "-"
                - a chemical formula (str)
                - an MP ID (str)
            can either be a list of:
                - chemical systems (str) of elements joined by "-"
                - chemical formulas (str)
                - MP IDs (str)

        properties (list or None)
            list of properties to query
                - if None, then use typical_properties
                - if 'all', then use all properties
                - if a string, then add that property to typical_properties
                - if a list, then add those properties to typical_properties

        band_gap (tuple)
            band gap range to query

        max_Ehull (float)
            upper bound on energy above hull to query

        max_sites_per_structure (int)
            upper bound on number of sites to query

        max_polymorph_energy (float)
            upper bound on polymorph energy to query

        only_gs (bool)
            if True, remove non-ground state polymorphs for each unique composition

        include_structure (bool)
            if True, include the structure (as a dictionary) for each entry

        max_strucs_per_cmpd (int)
            if not None, only retain the lowest energy structures for each composition until you reach max_strucs_per_cmpd

        include_sub_phase_diagrams (bool)
            if True, include all sub-phase diagrams for a given composition
                e.g., if comp = "Sr-Zr-S", then also include "Sr-S" and "Zr-S" in the query
        data_dir (str)
            directory to save fjson
        savename (str)
            filename for fjson in data_dir
        remake (bool)
            write (True) or just read (False) fjson
    Returns:
        {ID (str) : {'structure' : Pymatgen Structure as dict,
                    < any other data you want to keep track of >}}
    """

    if len(api_key) < 32:
        raise ValueError("API key should be 32 characters")

    fjson = os.path.join(data_dir, savename)
    if os.path.exists(fjson) and not remake:
        return read_json(fjson)

    # initialize MPQuery with your API key
    mpq = MPQuery(api_key=api_key)

    # get the data from MP
    data = mpq.get_data(
        search_for=search_for,
        properties=properties,
        max_Ehull=max_Ehull,
        max_sites_per_structure=max_sites_per_structure,
        max_polymorph_energy=max_polymorph_energy,
        only_gs=only_gs,
        include_structure=include_structure,
        max_strucs_per_cmpd=max_strucs_per_cmpd,
        include_sub_phase_diagrams=include_sub_phase_diagrams,
    )

    write_json(data, fjson)
    return read_json(fjson)


def get_legacy_query(
    comp,
    api_key,
    properties=None,
    criteria=None,
    only_gs=True,
    include_structure=True,
    supercell_structure=False,
    max_Ehull=0.05,
    max_sites_per_structure=65,
    max_strucs_per_cmpd=4,
    data_dir=os.getcwd().replace("scripts", "data"),
    savename="query.json",
    remake=False,
):
    """
    Args:
        comp (list or str)
            can either be:
                - a chemical system (str) of elements joined by "-"
                - a chemical formula (str)
            can either be a list of:
                - chemical systems (str) of elements joined by "-"
                - chemical formulas (str)

        api_key (str):
            your API key for Materials Project

        properties (list or None)
            list of properties to query
                - if None, then use typical_properties
                - if 'all', then use supported_properties

        criteria (dict or None)
            dictionary of criteria to query
                - if None, then use {}

        only_gs (bool)
            if True, remove non-ground state polymorphs for each unique composition

        include_structure (bool)
            if True, include the structure (as a dictionary) for each entry

        supercell_structure (bool)
            only runs if include_structure = True
            if False, just retrieve the MP structure
            if not False, must be specified as [a,b,c] to make an a x b x c supercell of the MP structure

        max_Ehull (float)
            if not None, remove entries with Ehull_mp > max_Ehull

        max_sites_per_structure (int)
            if not None, remove entries with more than max_sites_per_structure sites

        max_strucs_per_cmpd (int)
            if not None, only retain the lowest energy structures for each composition until you reach max_strucs_per_cmpd

        data_dir (str)
            directory to save fjson

        savename (str)
            filename for fjson in data_dir

        remake (bool)
            write (True) or just read (False) fjson

    Returns:
        {ID (str) : {'structure' : Pymatgen Structure as dict,
                    < any other data you want to keep track of >}}
    """

    fjson = os.path.join(data_dir, savename)
    if os.path.exists(fjson) and not remake:
        return read_json(fjson)

    # initialize MPQuery with your API key
    mpq = MPLegacyQuery(api_key=api_key)

    # get the data from MP
    data = mpq.get_data_for_comp(
        comp=comp,
        properties=properties,
        criteria=criteria,
        only_gs=only_gs,
        include_structure=include_structure,
        supercell_structure=supercell_structure,
        max_Ehull=max_Ehull,
        max_sites_per_structure=max_sites_per_structure,
        max_strucs_per_cmpd=max_strucs_per_cmpd,
    )

    write_json(data, fjson)
    return read_json(fjson)


def check_query(query):
    for mpid in query:
        print("\nmpid: %s" % mpid)
        print("\tcmpd: %s" % query[mpid]["cmpd"])
        print("\tstructure formula: %s" % StrucTools(query[mpid]["structure"]).formula)


def get_strucs(
    query,
    data_dir=os.getcwd().replace("scripts", "data"),
    savename="strucs.json",
    remake=False,
):
    """
    Args:
        query (dict)
            {mpid : {DATA}}

        data_dir (str)
            directory to save fjson

        savename (str)
            filename for fjson in DATA_DIR

        remake (bool)
            write (True) or just read (False) fjson

    Returns:
        {formula identifier (str) :
            {structure identifier for that formula (str) :
                Pymatgen Structure object as dict}}
    """

    fjson = os.path.join(data_dir, savename)
    if os.path.exists(fjson) and not remake:
        return read_json(fjson)

    # get all unique chemical formulas in the query
    formulas_in_query = sorted(list(set([query[mpid]["cmpd"] for mpid in query])))

    data = {}
    for formula in formulas_in_query:
        # get all MP IDs in your query having that formula
        mpids = [mpid for mpid in query if query[mpid]["cmpd"] == formula]
        data[formula] = {mpid: query[mpid]["structure"] for mpid in mpids}

    write_json(data, fjson)
    return read_json(fjson)


def check_strucs(strucs):
    for formula in strucs:
        for ID in strucs[formula]:
            print("\nformula: %s" % formula)
            print("\tID: %s" % ID)
            struc = strucs[formula][ID]
            print("\tstructure formula: %s" % StrucTools(struc).formula)


def get_magmoms(
    strucs,
    max_afm_combos=50,
    treat_as_nm=None,
    data_dir=os.getcwd().replace("scripts", "data"),
    savename="magmoms.json",
    remake=False,
):
    """
    Args:
        strucs (dict)
            {formula : {ID : structure}}

        max_afm_combos (int)
            maximum number of AFM spin configurations to generate

        treat_as_nm (list)
            any normally mag els you'd like to treat as nonmagnetic for AFM enumeration

        data_dir (str)
            directory to save fjson

        savename (str)
            filename for fjson in data_dir

        remake (bool)
            write (True) or just read (False) fjson

    Returns:
        {formula identifier (str) :
            {structure identifier for that formula (str) :
                {AFM ordering identifier (str) :
                    [list of magmoms (floats) for each site in the structure]}}}"""

    fjson = os.path.join(data_dir, savename)
    if not remake and os.path.exists(fjson):
        return read_json(fjson)

    if treat_as_nm is None:
        treat_as_nm = []

    magmoms = {}
    for formula in strucs:
        magmoms[formula] = {}
        for ID in strucs[formula]:
            # for each unique structure, get AFM magmom orderings
            structure = strucs[formula][ID]
            magtools = MagTools(
                structure=structure,
                max_afm_combos=max_afm_combos,
                treat_as_nm=treat_as_nm,
            )
            curr_magmoms = magtools.get_afm_magmoms
            magmoms[formula][ID] = curr_magmoms

    write_json(magmoms, fjson)
    return read_json(fjson)


def check_magmoms(strucs, magmoms):
    for formula in strucs:
        for ID in strucs[formula]:
            structure_formula = StrucTools(strucs[formula][ID]).formula
            n_afm_configs = len(magmoms[formula][ID])
            print("%s: %i AFM configs\n" % (structure_formula, n_afm_configs))


def get_launch_dirs(
    strucs,
    magmoms,
    user_configs,
    make_launch_dirs=True,
    data_dir=os.getcwd().replace("scripts", "data"),
    calcs_dir=os.getcwd().replace("scripts", "calcs"),
    savename="launch_dirs.json",
    remake=False,
):
    """
    Args:
        strucs (dict)
            {formula : {ID : structure}}

        magmoms (dict)
            {formula : {ID : {AFM configuration index : [list of magmoms on each site]}}

        user_configs (dict)
            optional launch configurations

        make_launch_dirs (bool)
            make launch directories (True) or just return launch dict (False)

        refresh_configs (bool)
            refresh configs (True) or just use existing configs (False)

        data_dir (str)
            directory to save fjson

        calcs_dir (str)
            directory above all your calculations

        savename (str)
            filename for fjson in data_dir

        remake (bool)
            write (True) or just read (False) fjson

    Returns:

        {launch_dir (str) : {'magmom' : [list of magmoms for the structure in that launch_dir (list)],
                             'ID_specific_vasp_configs' : {<formula_indicator>_<struc_indicator> : {}}}

        also makes launch_dir and populates with POSCAR using strucs if make_dirs=True

        NOTE: ID_specific_vasp_configs is not working as intended

    """

    fjson = os.path.join(data_dir, savename)
    if os.path.exists(fjson) and not remake:
        return read_json(fjson)

    all_launch_dirs = {}
    for formula in strucs:
        for ID in strucs[formula]:
            # for each unique structure, generate our launch directories
            structure = strucs[formula][ID]
            if magmoms:
                curr_magmoms = magmoms[formula][ID]
            else:
                curr_magmoms = None

            launch = LaunchTools(
                calcs_dir=calcs_dir,
                structure=structure,
                formula_indicator=formula,
                struc_indicator=ID,
                initial_magmoms=curr_magmoms,
                user_configs=user_configs,
            )

            launch_dirs = launch.launch_dirs(make_dirs=make_launch_dirs)

            for launch_dir in launch_dirs:
                all_launch_dirs[launch_dir] = launch_dirs[launch_dir]

    write_json(all_launch_dirs, fjson)
    return read_json(fjson)


def check_launch_dirs(launch_dirs):
    print("\nanalyzing launch directories")
    for d in launch_dirs:
        print("\nlaunching from %s" % d)


def submit_one_calc(submit_args):
    """
    Prepares VASP inputs, writes submission script, and launches job for one launch_dir

    Args:
        submit_args (dict) should contain:
            {'launch_dir' :
                launch_dir (str)
                    (formula/ID/standard/mag) to write and launch submission script in,
            'launch_dirs' :
                launch_dirs (dict)
                    {launch_dir (formula/ID/standard/mag) : {'xcs' : [list of final_xcs], 'magmoms' : [list of magmoms for each site in structure in launch_dir], 'ID_specific_vasp_configs' : {options}}},
            'user_configs' :
                user_configs (dict)
                    optional sub, slurm, or VASP configurations,
            'ready_to_launch':
                ready_to_launch (bool)
                    write and launch (True) or just write submission scripts (False)

            'parallel':
                running_in_parallel (bool)
                    whether to run in parallel (True) or not (False)
                }

    Returns:
        None

    """
    launch_dir = submit_args["launch_dir"]
    launch_dirs = submit_args["launch_dirs"]
    user_configs = submit_args["user_configs"]
    ready_to_launch = submit_args["ready_to_launch"]
    running_in_parallel = submit_args["parallel"]

    curr_user_configs = user_configs.copy()

    # what magmoms apply to that launch_dir
    initial_magmom = launch_dirs[launch_dir]["magmom"]

    ID_specific_vasp_configs = launch_dirs[launch_dir]["ID_specific_vasp_configs"]

    if running_in_parallel:
        try:
            sub = SubmitTools(
                launch_dir=launch_dir,
                initial_magmom=initial_magmom,
                ID_specific_vasp_configs=ID_specific_vasp_configs,
                user_configs=curr_user_configs,
            )

            print(sub.calc_list)
            # prepare VASP directories and write submission script
            sub.write_sub

            # submit submission script to the queue
            if ready_to_launch:
                sub.launch_sub

            success = True
        except TypeError:
            print("\nERROR: %s\n   will submit without multiprocessing" % launch_dir)
            success = False
    else:
        sub = SubmitTools(
            launch_dir=launch_dir,
            initial_magmom=initial_magmom,
            ID_specific_vasp_configs=ID_specific_vasp_configs,
            user_configs=curr_user_configs,
        )

        # prepare VASP directories and write submission script
        sub.write_sub

        # submit submission script to the queue
        if ready_to_launch:
            sub.launch_sub

        success = True

    return {"launch_dir": launch_dir, "success": success}


def submit_calcs(
    launch_dirs,
    user_configs={},
    ready_to_launch=True,
    n_procs=1,
):
    """
    Prepares VASP inputs, writes submission script, and launches job for all launch_dirs

    Args:
        launch_dirs (dict)
            {launch_dir (formula/ID/standard/mag) : {'xcs' : [list of final_xcs], 'magmoms' : [list of magmoms for each site in structure in launch_dir]}}

        user_configs (dict)
            optional sub, slurm, or VASP configurations

        ready_to_launch (bool)
            write and launch (True) or just write submission scripts (False)

        n_procs (int or str)
            if parallelizing, how many processors

    Returns:
        None

    """

    submit_args = {
        "launch_dirs": launch_dirs,
        "user_configs": user_configs,
        "ready_to_launch": ready_to_launch,
        "parallel": False if n_procs == 1 else True,
    }

    if n_procs == 1:
        print("\n\n submitting calculations in serial\n\n")
        for launch_dir in launch_dirs:
            curr_submit_args = submit_args.copy()
            curr_submit_args["launch_dir"] = launch_dir
            submit_one_calc(curr_submit_args)
        return
    elif n_procs == "all":
        n_procs = multip.cpu_count() - 1

    print("\n\n submitting calculations in parallel\n\n")
    print("not refreshing configs for parallel --> causes trouble")
    list_of_submit_args = []
    for launch_dir in launch_dirs:
        curr_submit_args = submit_args.copy()
        curr_submit_args["launch_dir"] = launch_dir
        list_of_submit_args.append(curr_submit_args)
    pool = multip.Pool(processes=n_procs)
    statuses = pool.map(submit_one_calc, list_of_submit_args)
    pool.close()

    submitted_w_multiprorcessing = [status for status in statuses if status["success"]]
    failed_w_multiprocessing = [status for status in statuses if not status["success"]]

    print(
        "%i/%i calculations submitted with multiprocessing"
        % (len(submitted_w_multiprorcessing), len(statuses))
    )
    for status in failed_w_multiprocessing:
        launch_dir = status["launch_dir"]
        curr_submit_args = submit_args.copy()
        curr_submit_args["launch_dir"] = launch_dir
        submit_one_calc(curr_submit_args)

    return


def get_results(
    launch_dirs,
    user_configs,
    data_dir=os.getcwd().replace("scripts", "data"),
    savename="results.json",
    remake=False,
):
    """
    Args:
        launch_dirs (dict)
            {launch_dir (formula_indicator/struc_indicator/mag) : {'magmoms' : [list of magmoms for each site in structure in launch_dir], 'ID_specific_vasp_configs' : {}}}

        user_configs (dict)
            optional analysis configurations

        data_dir (str)
            directory to save fjson

        savename (str)
            filename for fjson in data_dir

        remake (bool)
            write (True) or just read (False) fjson

    Returns:
        {formula--ID--standard--mag--xc-calc (str) :
            {scraped results from VASP calculation}}
    """

    fjson = os.path.join(data_dir, savename)
    if os.path.exists(fjson) and not remake:
        return read_json(fjson)

    analyzer = AnalyzeBatch(launch_dirs, user_configs=user_configs)

    data = analyzer.results

    write_json(data, fjson)
    return read_json(fjson)


def check_results(results):
    keys_to_check = list(results.keys())

    converged = 0
    static_converged = 0
    total_static = 0
    for key in keys_to_check:
        formula_indicator, struc_indicator, mag, xc_calc = key.split("--")
        xc, calc = xc_calc.split("-")
        data = results[key]
        convergence = results[key]["results"]["convergence"]
        print("\n%s" % key)
        print("convergence = %s" % convergence)
        if convergence:
            converged += 1
            # print("\n%s" % key)
            print("E (%s) = %.2f" % (xc_calc, data["results"]["E_per_at"]))
            if calc == "static":
                static_converged += 1
        if calc == "static":
            total_static += 1

    print(
        "\n\n SUMMARY: %i/%i of all calcs converged" % (converged, len(keys_to_check))
    )
    print(
        "\n\n SUMMARY: %i/%i of static calcs converged"
        % (static_converged, total_static)
    )


def get_gs(
    results,
    include_structure=False,
    non_default_functional=None,
    compute_Ef=True,
    standard="dmc",
    data_dir=os.getcwd().replace("scripts", "data"),
    savename="gs.json",
    remake=False,
):
    """
    Args:
        results (dict)
            {formula--ID--standard--mag--xc-calc (str) : {scraped results from VASP calculation}}

        include_structure (bool)
            include the structure or not

        non_default_functional (str)
            if you're not using r2SCAN or PBE

        compute_Ef (bool)
            if True, compute formation enthalpy

        data_dir (str)
            directory to save fjson

        savename (str)
            filename for fjson in data_dir

        remake (bool)
            write (True) or just read (False) fjson

    Returns:
        {xc (str, the exchange-correlation method) :
            {formula (str) :
                {'E' : energy of the ground-structure,
                'key' : formula.ID.standard.mag.xc_calc for the ground-state structure,
                'structure' : structure of the ground-state structure,
                'n_started' : how many polymorphs you tried to calculate,
                'n_converged' : how many polymorphs are converged,
                'complete' : True if n_converged = n_started (i.e., all structures for this formula at this xc are done),
                'Ef' : formation enthalpy at 0 K}
    """
    fjson = os.path.join(data_dir, savename)
    if os.path.exists(fjson) and not remake:
        return read_json(fjson)

    results = {
        key: results[key]
        for key in results
        if results[key]["meta"]["setup"]["calc"] == "static"
    }

    gs = {
        xc: {}
        for xc in sorted(
            list(set([results[key]["meta"]["setup"]["xc"] for key in results]))
        )
    }

    for xc in gs:
        keys = [
            k
            for k in results
            if results[k]["meta"]["setup"]["xc"] == xc
            if results[k]["results"]["formula"]
        ]

        unique_formulas = sorted(
            list(set([results[key]["results"]["formula"] for key in keys]))
        )
        for formula in unique_formulas:
            gs[xc][formula] = {}
            formula_keys = [
                k for k in keys if results[k]["results"]["formula"] == formula
            ]
            converged_keys = [
                k for k in formula_keys if results[k]["results"]["convergence"]
            ]
            if not converged_keys:
                gs_energy, gs_structure, gs_key = None, None, None
            else:
                energies = [results[k]["results"]["E_per_at"] for k in converged_keys]
                gs_energy = min(energies)
                gs_key = converged_keys[energies.index(gs_energy)]
                gs_structure = results[gs_key]["structure"]
            complete = True if len(formula_keys) == len(converged_keys) else False
            gs[xc][formula] = {
                "E": gs_energy,
                "key": gs_key,
                "n_started": len(formula_keys),
                "n_converged": len(converged_keys),
                "complete": complete,
            }
            if include_structure:
                gs[xc][formula]["structure"] = gs_structure

    if compute_Ef:
        for xc in gs:
            if not non_default_functional:
                functional = "r2scan" if xc == "metagga" else "pbe"
            else:
                functional = non_default_functional
            mus = ChemPots(functional=functional, standard=standard).chempots
            for formula in gs[xc]:
                E = gs[xc][formula]["E"]
                if E:
                    Ef = FormationEnthalpy(formula=formula, E_DFT=E, chempots=mus).Ef
                else:
                    Ef = None
                gs[xc][formula]["Ef"] = Ef
    write_json(gs, fjson)
    return read_json(fjson)


def check_gs(gs):
    """
    checks that this dictionary is generated properly

    Args:
        gs (_type_): _description_

    """

    print("\nchecking ground-states")
    xcs = list(gs.keys())
    for xc in xcs:
        print("  xc = %s" % xc)
        formulas = list(gs[xc].keys())
        n_formulas = len(formulas)
        n_formulas_complete = len([k for k in formulas if gs[xc][k]["complete"]])
        print(
            "%i/%i formulas with all calculations completed"
            % (n_formulas_complete, n_formulas)
        )
        for formula in gs[xc]:
            if "Ef" in gs[xc][formula]:
                if gs[xc][formula]["Ef"]:
                    print("%s : %.2f eV/at" % (formula, gs[xc][formula]["Ef"]))


def get_thermo_results(
    results,
    gs,
    data_dir=os.getcwd().replace("scripts", "data"),
    savename="thermo_results.json",
    remake=False,
):
    """

    Args:
        results (dict):
            full results dictionary

        gs (dict):
            dictionary of ground-state data

        data_dir (str)
            directory to save fjson

        savename (str)
            fjson name in data_dir

        remake (bool)
            Read (False) or write (True) json

    Returns:
        {xc (str) :
            {formula (str) :
                {ID (str) :
                    {'E' : energy of the structure (DFT total energy in eV/atom),
                    'Ef' : formation enthalpy at 0 K (eV/atom),
                    'is_gs' : True if this is the lowest energy polymorph for this formula,
                    'dE_gs' : how high above the ground-state this structure is in energy (eV/atom)
                    'all_polymorphs_converged' : True if every structure that was computed for this formula is converged}}
    """
    fjson = os.path.join(data_dir, savename)
    if os.path.exists(fjson) and not remake:
        return read_json(fjson)

    thermo_results = {xc: {formula: {} for formula in gs[xc]} for xc in gs}

    for key in results:
        tmp_thermo = {}

        xc = results[key]["meta"]["setup"]["xc"]
        formula = results[key]["results"]["formula"]
        ID = "__".join(
            [
                results[key]["meta"]["setup"]["formula_indicator"],
                results[key]["meta"]["setup"]["struc_indicator"],
                results[key]["meta"]["setup"]["mag"],
            ]
        )
        E = results[key]["results"]["E_per_at"]
        formula = results[key]["results"]["formula"]
        if ("structure" in results[key]) and results[key]["structure"]:
            calcd_formula = StrucTools(results[key]["structure"]).formula
        else:
            calcd_formula = None
        tmp_thermo["E"] = E
        tmp_thermo["key"] = key
        tmp_thermo["formula"] = formula
        tmp_thermo["calculated_formula"] = calcd_formula

        if E:
            gs_key = gs[xc][formula]["key"]
            if "Ef" in gs[xc][formula]:
                gs_Ef = gs[xc][formula]["Ef"]
            else:
                gs_Ef = None
            gs_E = gs[xc][formula]["E"]
            if gs_E:
                delta_E_gs = E - gs_E
            else:
                delta_E_gs = None

            if key == gs_key:
                tmp_thermo["is_gs"] = True
            else:
                tmp_thermo["is_gs"] = False

            tmp_thermo["dE_gs"] = delta_E_gs
            if gs_Ef:
                tmp_thermo["Ef"] = gs_Ef + delta_E_gs
            else:
                tmp_thermo["Ef"] = None
            tmp_thermo["all_polymorphs_converged"] = gs[xc][formula]["complete"]

        else:
            tmp_thermo["dE_gs"] = None
            tmp_thermo["Ef"] = None
            tmp_thermo["is_gs"] = False
            tmp_thermo["all_polymorphs_converged"] = False

        thermo_results[xc][formula][ID] = tmp_thermo

    write_json(thermo_results, fjson)
    return read_json(fjson)


def check_thermo_results(thermo):
    print("\nchecking thermo results")

    for xc in thermo:
        print("\nxc = %s" % xc)
        for formula in thermo[xc]:
            print("formula = %s" % formula)
            print(
                "%i polymorphs converged"
                % len([k for k in thermo[xc][formula] if thermo[xc][formula][k]["E"]])
            )
            gs_ID = [k for k in thermo[xc][formula] if thermo[xc][formula][k]["is_gs"]]
            if gs_ID:
                gs_ID = gs_ID[0]
                print("%s is the ground-state structure" % gs_ID)

    print("\n\n  SUMMARY  ")
    for xc in thermo:
        print("~~%s~~" % xc)
        converged_formulas = []
        for formula in thermo[xc]:
            for ID in thermo[xc][formula]:
                if thermo[xc][formula][ID]["all_polymorphs_converged"]:
                    converged_formulas.append(formula)

        converged_formulas = list(set(converged_formulas))
        print(
            "%i/%i formulas have all polymorphs converged"
            % (len(converged_formulas), len(thermo[xc].keys()))
        )


def get_dos_results(
    results,
    thermo_results,
    only_gs=True,
    only_xc="metagga",
    only_formulas=[],
    dos_to_store=["tdos", "tcohp"],
    regenerate_dos=False,
    regenerate_cohp=False,
    data_dir=os.getcwd().replace("scripts", "data"),
    savename="dos_results.json",
    remake=False,
):
    """
    Args:
        results (dict)
            from get_results

        thermo_results (dict)
            from get_thermo_results

        only_gs (bool)
            if True, only get DOS/COHP for the ground-state polymorphs

        only_xc (str)
            if not None, only get DOS/COHP for this XC

        only_formulas (list)
            if not None, only get DOS/COHP for these formulas

        only_standard (str)
            if not None, only get DOS/COHP for this standard

        dos_to_store (list)
            which DOS/COHP to store ['tcohp', 'pcohp', 'tdos', 'pdos', etc]

        regenerate_dos (bool)
            if True, make pdos/tdos jsons again even if it exists

        regenerate_cohp (bool)
            if True, make pcohp/tcohp jsons again even if it exists

        data_dir (str)
            path to data directory

        savename (str)
            name of json file to save results to

        remake (bool)
            if True, remake the json file
    """
    fjson = os.path.join(data_dir, savename)
    if os.path.exists(fjson) and not remake:
        return read_json(fjson)

    for key in results:
        calc_dir = results[key]["meta"]["calc_dir"]
        xc = results[key]["meta"]["setup"]["xc"]
        calc = results[key]["meta"]["setup"]["calc"]
        if calc not in ["lobster", "bs"]:
            continue
        ID = "__".join(
            [
                results[key]["meta"]["setup"]["formula_indicator"],
                results[key]["meta"]["setup"]["struc_indicator"],
                results[key]["meta"]["setup"]["mag"],
            ]
        )
        formula = results[key]["results"]["formula"]
        thermo_result = thermo_results[xc][formula][ID]
        if only_gs:
            if not thermo_result["is_gs"]:
                continue
        if only_formulas:
            if thermo_result["formula"] not in only_formulas:
                continue
        if only_xc:
            if xc != only_xc:
                continue
        av = AnalyzeVASP(calc_dir)
        if "tdos" in dos_to_store:
            pdos = av.pdos(remake=regenerate_dos)
            tdos = av.tdos(pdos=pdos, remake=regenerate_dos)
            thermo_results[xc][formula][ID]["tdos"] = tdos
        if "pdos" in dos_to_store:
            thermo_results[xc][formula][ID]["pdos"] = pdos
        if "tcohp" in dos_to_store:
            pcohp = av.pcohp(remake=regenerate_cohp)
            tcohp = av.tcohp(pcohp=pcohp, remake=regenerate_cohp)
            thermo_results[xc][formula][ID]["tcohp"] = tcohp
        if "pcohp" in dos_to_store:
            thermo_results[xc][formula][ID]["pcohp"] = pcohp
        if "tcoop" in dos_to_store:
            pcohp = av.pcohp(are_coops=True, remake=regenerate_cohp)
            tcohp = av.tcohp(pcohp=pcohp, remake=regenerate_cohp)
            thermo_results[xc][formula][ID]["tcoop"] = tcohp
        if "pcoop" in dos_to_store:
            thermo_results[xc][formula][ID]["pcoop"] = pcohp
        if "tcobi" in dos_to_store:
            pcohp = av.pcohp(are_cobis=True, remake=regenerate_cohp)
            tcohp = av.tcohp(pcohp=pcohp, remake=regenerate_cohp)
            thermo_results[xc][formula][ID]["tcobi"] = tcohp
        if "pcobi" in dos_to_store:
            thermo_results[xc][formula][ID]["pcobi"] = pcohp

    write_json(thermo_results, fjson)
    return read_json(fjson)


def get_entries(
    results,
    data_dir=os.getcwd().replace("scripts", "data"),
    savename="entries.json",
    remake=False,
):
    """
    Args:
        results (dict)
            from get_results
                {formula--ID--standard--mag--xc-calc (str) : {scraped results from VASP calculation}}

        data_dir (str)
            path to data directory

        savename (str)
            name of json file to save results to

        remake (bool)
            if True, remake the json file

    Returns:
        dictionary with ComputedStructureEntry objects for each of your completed calculations
            {'entries' : [list of ComputedStructureEntry.as_dict() objects]}

        note: each of these entries has the "key" from the results dictionary stored as entry['data']['material_id']
    """
    fjson = os.path.join(data_dir, savename)
    if os.path.exists(fjson) and not remake:
        return read_json(fjson)
    d = {"entries": [results[k]["entry"] for k in results if results[k]["entry"]]}
    write_json(d, fjson)
    return read_json(fjson)


def get_mp_entries(
    chemsyses,
    api_key,
    thermo_types=None,
    data_dir=os.getcwd().replace("scripts", "data"),
    savename="mp_entries.json",
    remake=False,
):
    """
    Args:
        chemsyses (list)
            list of chemical systems to get entries for (e.g., ['Li-Fe-O', 'Mg-Cr-O'])
                will include "sub phase diagrams" in query
                e.g., for 'Li-Fe-O', will include 'Li-Fe-O', 'Li-Fe', 'Li-O', 'Fe-O', 'Li', 'Fe', 'O'

        api_key (str)
            your Materials Project API key

        thermo_types (list)
            list of thermo types to get entries for
                this could be ['GGA_GGA+U'], ['R2SCAN'], ['GGA_GGA+U', 'R2SCAN']
            if None, will get all data regardless of thermo_type (note: this should be equivalent to thermo_types=['GGA_GGA+U', 'R2SCAN'])

        data_dir (str)
            path to data directory

        savename (str)
            name of json file to save results to

        remake (bool)
            if True, remake the json file

    Returns:
        dictionary of ComputedStructureEntry objects from the Materials Project
            {chemsys (str) :
                [list of ComputedStructureEntry.as_dict() objects]}
            note: the mp-id is stored as entry['data']['material_id']
            note: the xc is stored in entry['parameters']['run_type']

    """
    fjson = os.path.join(data_dir, savename)
    if os.path.exists(fjson) and not remake:
        return read_json(fjson)

    mpq = MPQuery(api_key=api_key)
    out = {}
    for chemsys in chemsyses:
        if thermo_types:
            data = mpq.get_entries_for_chemsys(chemsys, thermo_types=thermo_types)
        else:
            data = mpq.get_entries_for_chemsys(chemsys)
        out[chemsys] = list(data.values())
    write_json(out, fjson)
    return read_json(fjson)


def get_merged_entries(
    my_entries,
    mp_entries,
    restrict_my_xc_to=None,
    data_dir=os.getcwd().replace("scripts", "data"),
    savename="merged_entries_for_mp_Ef.json",
    remake=False,
):
    """
    Args:
        my_entries (dict)
            from get_entries
                {'entries' : [list of ComputedStructureEntry.as_dict() objects]}

        mp_entries (dict)
            from get_mp_entries
                {chemsys (str) : [list of ComputedStructureEntry.as_dict() objects]}

        restrict_my_xc_to (str)
            if not None, only include my entries with this xc
                e.g., 'GGA', 'GGA+U', 'r2SCAN'

        data_dir (str)
            path to data directory

        savename (str)
            name of json file to save results to

        remake (bool)
            if True, remake the json file

    Returns:
        dictionary of ComputedStructureEntry objects from the Materials Project and your calculations
            {chemsys (str) :
                [list of ComputedStructureEntry.as_dict() objects]}
            note: this will exclude any of your calculations where standard != 'mp' because the purpose of this is to compare to MP
                if you're not comparing to MP, then you can just get your own entries from get_entries
    """
    fjson = os.path.join(data_dir, savename)
    if os.path.exists(fjson) and not remake:
        return read_json(fjson)

    if restrict_my_xc_to == "GGA":
        my_allowed_xcs = ["gga"]
    elif restrict_my_xc_to == "GGA+U":
        my_allowed_xcs = ["ggau"]
    elif restrict_my_xc_to == "r2SCAN":
        my_allowed_xcs = ["r2scan"]
    elif restrict_my_xc_to == "GGA_GGA+U":
        my_allowed_xcs = ["gga", "ggau"]
    else:
        my_allowed_xcs = None

    entries = {}
    for chemsys in mp_entries:
        entries[chemsys] = []
        mp_entries_for_chemsys = mp_entries[chemsys]
        for e in mp_entries_for_chemsys:
            entries[chemsys].append(e)

    relevant_chemsyses = list(entries.keys())

    my_entries = my_entries["entries"]
    for e in my_entries:
        if e["data"]["standard"] != "mp":
            continue
        if my_allowed_xcs and (e["data"]["xc"] not in my_allowed_xcs):
            continue
        formula = e["data"]["formula"]
        for chemsys in relevant_chemsyses:
            if set(CompTools(formula).els).issubset(set(chemsys.split("-"))):
                entries[chemsys].append(e)

    write_json(entries, fjson)
    return read_json(fjson)


def get_mp_compatible_Efs(
    merged_entries,
    data_dir=os.getcwd().replace("scripts", "data"),
    savename="mp_compatible_Efs.json",
    remake=False,
):
    """
    Args:
        merged_entries (dict)
            from get_merged_entries
                {chemsys (str) : [list of ComputedStructureEntry.as_dict() objects]}

        data_dir (str)
            path to data directory

        savename (str)
            name of json file to save results to

        remake (bool)
            if True, remake the json file

    Returns:
        dictionary of compatible formation energies for each chemsys
            {chemsys (str) :
                {formula (str) :
                    ID (str) : formation energy (eV/atom)}
            note: this will include all polymorphs
            note: this will include MP data (ID = mp-id) and your data (ID = formula--ID--standard--mag--xc-calc)
    """
    fjson = os.path.join(data_dir, savename)
    if os.path.exists(fjson) and not remake:
        return read_json(fjson)

    out = {}
    for chemsys in merged_entries:
        mpfe = MPFormationEnergy(merged_entries[chemsys])
        Efs = mpfe.Efs
        out[chemsys] = Efs

    write_json(out, fjson)
    return read_json(fjson)


def crawl_and_purge(
    head_dir,
    files_to_purge=[
        "WAVECAR",
        "CHGCAR",
        "CHG",
        "PROCAR",
        "LOCPOT",
        "AECCAR0",
        "AECCAR1",
        "AECCAR2",
    ],
    safety="on",
    check_convergence=True,
    verbose=False,
):
    """
    Args:
        head_dir (str)
            directory to start crawling beneath

        files_to_purge (list)
            list of file names to purge

        safety (str)
            'on' or 'off' to turn on/off safety
                - if safety is on, won't actually delete files
    """
    purged_files = []
    mem_created = 0
    for subdir, dirs, files in os.walk(head_dir):
        ready = False
        if check_convergence:
            if "POTCAR" in files:
                av = AnalyzeVASP(subdir)
                if av.is_converged:
                    ready = True
                else:
                    ready = False
            else:
                ready = False
        else:
            ready = True
        if ready:
            for f in files:
                if f in files_to_purge:
                    path_to_f = os.path.join(subdir, f)
                    if verbose:
                        print(path_to_f)
                    mem_created += os.stat(path_to_f).st_size
                    purged_files.append(path_to_f)
                    if safety == "off":
                        os.remove(path_to_f)
    if safety == "off":
        print(
            "You purged %i files, freeing up %.2f GB of memory"
            % (len(purged_files), mem_created / 1e9)
        )
    if safety == "on":
        print(
            "You had the safety on\n If it were off, you would have purged %i files, freeing up %.2f GB of memory"
            % (len(purged_files), mem_created / 1e9)
        )


def make_sub_for_launcher():
    """
    Creates sub_launcher.sh file to launch launcher on compute node
    """
    flauncher_sub = os.path.join(os.getcwd(), "sub_launcher.sh")
    launch_job_name = "-".join([os.getcwd().split("/")[-2], "launcher"])
    with open(flauncher_sub, "w") as f:
        f.write("#!/bin/bash -l\n")
        f.write("#SBATCH --nodes=1\n")
        f.write("#SBATCH --ntasks=8\n")
        f.write("#SBATCH --time=4:00:00\n")
        f.write("#SBATCH --mem=8G\n")
        f.write("#SBATCH --error=_log_launcher.e\n")
        f.write("#SBATCH --output=_log_launcher.o\n")
        f.write("#SBATCH --account=cbartel\n")
        f.write("#SBATCH --job-name=%s\n" % launch_job_name)
        f.write("#SBATCH --partition=msismall\n")
        f.write("\npython launcher.py\n")


def main():
    return


if __name__ == "__main__":
    main()
