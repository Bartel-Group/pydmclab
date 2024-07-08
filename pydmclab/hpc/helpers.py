import multiprocessing as multip
import os
import warnings

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
    incar_mods=None,
    kpoints_mods=None,
    potcar_mods=None,
    flexible_convergence_criteria=False,
    compare_static_and_relax_energies=0.1,
    special_functional=False,
    COHPSteps=2000,
    reciprocal_kpoints_density_for_lobster=100,
    bandstructure_symprec=0.1,
    bandstructure_kpoints_line_density=20,
):
    """
    configs related to VASP calculations
    Args:
        standard (str):
            "dmc" for group standard, "mp" for Materials Project standard
                This affects how pymatgen VaspSets get modified
                See pydmclab.hpc.sets.GetSet.user_*_settings for more details
        dont_relax_cell (bool)
            if True, sets ISIF = 2 for all calculations
            if False, do nothing
        <input_file>_mods (dict)
            <input_file> in [incar, kpoints, potcar]
            all of these mods expect a dictionary of the format
                {xc-calc (str) : {setting (str) : value (str, bool, float, int, list)}}
                    xc can be 'gga', 'ggau', 'metagga', 'metaggau', 'hse06', etc
                    calc can be 'loose', 'relax', 'static', 'lobster', 'dfpt', etc
                use xc-calc = 'all-<calc>' to apply modification to all xcs for a given <calc>
                use xc-calc = '<xc>-all' to apply modifications to all calcs for a given <xc>
                use xc-calc = 'all-all' to apply modifications to all xcs and all calcs
        incar_mods (dict)
            modifications to INCAR (on top of the VaspSet determined by pydmclab.hpc.sets.GetSet)
                e.g., to modify INCAR settings for all of your metagga calculations:
                    {'metagga-all' : {'LMAXMIX' : 0.1,
                                    'EDIFF' : 1e-7}}
                e.g., to modify only your gga-relax calculations
                    {'gga-relax' : {'EDIFFG' : -0.05}}
        kpoints_mods (dict):
            modifications to KPOINTS (see pydmclab.hpc.sets.GetSet.user_kpoints_settings for options)
                e.g., to set an 8x8x8 grid for all gga calculations
                    {'gga-all' : {'grid' : [8, 8, 8]}}
        potcar_mods (dict):
            modifications to POTCAR
                e.g., to change your W pseudopotential for all calculations
                    {'all-all' : {'W' : 'W_pv'}}
        flexible_convergence_criteria (bool):
            if True, reduce EDIFF/EDIFFG if convergence is taking very many steps
            if False, only update NELM and NSW if convergence is taking very many steps
        compare_static_and_relax_energies (float):
            if True, if the difference between static and relax energies is greater than this, rerun the static calculation
            if False, don't compare the energies
        special_functional (dict or bool)
            if you're not using r2SCAN, PBE, or HSE06, specify here
                e.g., {'gga' : 'PS'} would use PBESol for gga calculations
                e.g., {'metagga' : 'SCAN'} would use SCAN for metagga calculations
            if False, do nothing
        COHPSteps (int):
            how many (E, DOS) points do you want in LOBSTER outputs
                only applies to xc-calc='all-lobster'
        reciprocal_kpoints_density_for_lobster (int):
            kppra for LOBSTER calculations (higher is denser grid)
                only applies to xc-calc='all-lobster'
        bandstructure_symprec (float):
            symmetry precision for finding primitive cell before bandstructure calculation
                only applies to xc-calc='all-bs'
        bandstructure_kpoints_line_density (int):
            number of KPOINTS between each high-symmetry k-point for bandstructure calculation
                only applies to xc-calc='all-bs'
    Returns:
        dictionary of configs relevant to running VASP
    """
    if incar_mods is None:
        incar_mods = {}
    if kpoints_mods is None:
        kpoints_mods = {}
    if potcar_mods is None:
        potcar_mods = {}

    vasp_configs = {}
    if dont_relax_cell:
        incar_mods.update({"all-all": {"ISIF": 2}})
    vasp_configs["standard"] = standard
    vasp_configs["incar_mods"] = incar_mods
    vasp_configs["kpoints_mods"] = kpoints_mods
    vasp_configs["potcar_mods"] = potcar_mods
    vasp_configs["functional"] = special_functional
    vasp_configs["bs_symprec"] = bandstructure_symprec
    vasp_configs["bs_line_density"] = bandstructure_kpoints_line_density
    vasp_configs["COHPSteps"] = COHPSteps
    vasp_configs["reciprocal_kpoints_density_for_lobster"] = (
        reciprocal_kpoints_density_for_lobster
    )
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
    start_with_loose=False,
    custom_calc_list=None,
    restart_these_calcs=None,
    submit_calculations_in_parallel=False,
    machine="msi",
    mpi_command="mpirun",
    vasp_version=6,
    struc_src_for_hse="metagga-relax",
):
    """
    configs related to generating submission scripts
    Args:
        relaxation_xcs (list):
            list of xcs you want to run relax + static for
                e.g., ['gga', 'metaggau'] if you want to run PBE and R2SCAN+U calculations
        static_addons (dict):
            {xc : [list of additional calculations to run after static]}
                e.g., {'gga' : ['lobster', 'parchg']} if you want to run LOBSTER and PARCHG analysis after your PBE calculation
        prioritize_relaxes (bool):
            the combination of relaxation_xcs and static_addons generates a calc_list (the order of calculations to be executed)
                if True, run relax+static for all relaxation_xcs first, then run static_addons
                if False, run xc by xc, so once a relax+static finishes for an xc, run all of that xc's static_addons next
            e.g.,
                if relaxation_xcs = ['gga', 'metagga']
                and static_addons = {'gga' : ['lobster'], 'metagga' : ['bs']}
                if prioritize_relaxes = True,
                    calc_list = ['gga-relax', 'gga-static', 'metagga-relax', 'metagga-static', 'gga-lobster', 'metagga-bs']
                if prioritize_relaxes = False,
                    calc_list = ['gga-relax', 'gga-static', 'gga-lobster', 'metagga-relax', 'metagga-static', 'metagga-lobster']
        start_with_loose (bool):
            if True, your first relaxation_xc will start with a loose calc before relax --> static
            if False, no loose calculations will be performed
        custom_calc_list (list):
            if you don't want to autogenerate a calc_list, you can specify the full list you want to run here
                e.g., ['metagga-static'] would only run this single xc-calc
        restart_these_calcs (list):
            list of xc-calcs you want to start over (e.g., ['gga-lobster'])
        submit_calculations_in_parallel (bool or int):
            whether to prepare submission scripts in parallel or not
                False: use 1 processor
                True: use all available processors - 1
                int: use that many processors
            you should only execute this function on a login node if submit_calculations_in_parallel = False
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

    if restart_these_calcs is None:
        restart_these_calcs = []

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
    sub_configs["struc_src_for_hse"] = struc_src_for_hse

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
                these are generated using pydmclab.core.mag.MagTools
                each configuration will be its own calculation directory (afm_0, afm_1, ...)
        override_mag (str or bool):
            if False, do nothing
            if str, set mag to override_mag (rather than letting pydmclab.core.mag.MagTools figure out if it might be magnetic)
                NOTE: not sure if this is working
        ID_specific_vasp_configs (dict):
            use this to modify VASP configs for a subset of the structures you're calculating
                {<formula_indicator>_<struc_indicator> : {'incar_mods' : {<INCAR tag> : <value>}, {'kpoints' : <kpoints value>}, {'potcar' : <potcar value>}}
            e.g.,
                to change EDIFF for the perovskite polymorph of SrZrS3
                    {'SrZrS3_perovskite' : {'incar_mods' : {'EDIFF' : 1e-5}}}
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
    analyze_structure=True,
    analyze_trajectory=False,
    analyze_mag=False,
    analyze_charge=False,
    analyze_dos=False,
    analyze_bonding=False,
    exclude=None,
    remake_results=False,
    verbose=False,
):
    """
    configs related to parsing calculations and compiling results
    Args:
        only_calc (bool or str):
            if str, only analyze this calc (eg 'static')
            if list, only analyze these calculations (eg ['static','lobster'])
            if None, analyze all calculations
        only_xc (bool or str):
            if str, only analyze this xc (eg 'gga')
            if list, only analyze these xcs (eg ['gga','ggau'])
            if None, analyze all xcs
        analyze_structure (bool):
            True to include structure in your results
        analyze_trajectory (bool):
            True to include ionic relaxation trajectory in your results
        analyze_mag (bool):
            True to include magnetization in your results
        analyze_charge (bool):
            True to include bader charge + lobster charges + madelung in your results
        analyze_dos (bool):
            True to include pdos, tdos in your results
        analyze_bonding (bool):
            True to include tcohp, pcohp, tcoop, pcoop, tcobi, pcobi in your results
        exclude (list):
            list of strings to exclude from analysis
                overwrites other options
        remake_results (bool):
            if True, regeneration calc_dir/results.json file even if it exists and calc hasn't re-started
        verbose (bool):
            print ('analyzing %s' % calc_dir) whenever one is being analyzed
    Returns:
        dictionary of configs related to analysis
            {config param (str) : config value (str, bool)}
    """

    if exclude is None:
        exclude = []

    analysis_configs = {}

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
    analysis_configs["remake_results"] = remake_results
    analysis_configs["verbose"] = verbose

    return analysis_configs


def get_query(
    api_key,
    search_for,
    max_Ehull=0.05,
    max_polymorph_energy=0.1,
    max_strucs_per_cmpd=1,
    max_sites_per_structure=41,
    include_sub_phase_diagrams=False,
    include_structure=True,
    properties=None,
    data_dir=os.getcwd().replace("scripts", "data"),
    savename="query.json",
    remake=False,
):
    """
    Use this to retrieve data + structures from the Materials Project (next-gen)
    Args:
        api_key (str or None)
            your API key (should be 32 characters for next-gen database)
                can be None, but you need to configure pymatgen w/ your MP API key
                     `pmg config --add PMG_MAPI_KEY <USER_API_KEY>`
        search_for (str or list)
            can either be:
                a chemical system (str) of elements joined by "-"
                    eg 'Ca-Ti-O' for all ternary calcium titanium oxides
                a chemical formula (str)
                    eg 'CaTiO3' for this formula
                an MP ID (str)
                    eg 'mp-1234' for this ID
            or a list of:
                chemical systems (str) of elements joined by "-"
                    eg ['Ca-Ti-O', 'Sr-Ti-O']
                chemical formulas (str)
                    eg ['CaO', 'CaTiO3']
                MP IDs (str)
                    eg ['mp-1', 'mp-2']
        max_Ehull (float)
            upper bound on energy above hull for retrieved entries
        max_polymorph_energy (float)
            upper bound on polymorph energy for retrieved entries
                set to 0 to only retrieve ground-state structures for all compositions
        max_strucs_per_cmpd (int)
            upper bound on number of polymorphs to retrieve for each queried composition
                retains the lowest energy ones
        max_sites_per_structure (int)
            upper bound on number of sites in retrieved structures
        include_sub_phase_diagrams (bool)
            if True, include all sub-phase diagrams for a given composition
                e.g., if search_for = "Sr-Zr-S", then also include "Sr-S" and "Zr-S" in the query
        include_structure (bool)
            if True, include the structure (as a dictionary) for each entry
        properties (list or None)
            list of properties to query
                if None, then use pydmclab.core.query.MPQuery.typical_properties
                if 'all', then use all properties
                if a string, then add that property to typical_properties
                if a list, then add those properties to typical_properties
        data_dir (str)
            directory to save fjson
        savename (str)
            filename for fjson in data_dir
        remake (bool)
            write (True) or just read (False) fjson
    Returns:
        {ID (str) : {'structure' : Pymatgen Structure as dict,
                    '<other property>' : whatever you queried for}}
        e.g.,
            {'mp-1234' : {'structure' : Structure.as_dict,
                          'E_mp' : -5.4321}}
        Note: if you don't want to use MP data, you can create your own `get_query` function that gets whatever data you want
            e.g., it might return
            {'SrZrS3_perovskite' : {'structure' : Structure.as_dict,
                                    'E_per_at' : -6.54}}
    """

    if api_key and len(api_key) < 32:
        raise ValueError("API key should be 32 characters or NoneType")

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
    NOTE: this is deprecated for get_query
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
    warnings.warn(
        "DeprecationWarning: Are you sure you want to use the legacy MP database? The next-gen MP database is preferred. Use pydmclab.hpc.helpers.get_query"
    )

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
    force_supercell=False,
):
    """
    You should rarely use this default function, but it should give you an idea how to make your own structures
    Args:
        query (dict)
            {<unique structure indicator> (e.g., MP ID) :
                {'structure' : Pymatgen Structure as dict,
                 '<any other info>' : ...}}
            usually generated with get_query (or similar custom function)
        data_dir (str)
            directory to save fjson
        savename (str)
            filename for fjson in DATA_DIR
        remake (bool)
            write (True) or just read (False) fjson
        force_supercell (bool)
            whether to create a 2x2x2 supercell for structures with only 1 atom
    Returns:
        {formula_indicator (str) :
            {struc_indicator (str) :
                Pymatgen Structure object as dict}}
        e.g., if you got some MP data, this might return something like:
            {'Cl3Cs1Pb1' : {'mp-1234' : Structure.as_dict}}
        e.g., if you made this yourself, it might look like:
            {'Li2FeP2S6' : {'ordering_0' : Structure.as_dict}}
        All structures within a formula_indicator should have the same composition
        It's fine if the struc_indicator alone does not define a material, but formula_indicator + struc_indicator should
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

    if force_supercell:
        # if this is True, make a 2x2x2 if input structure is a 1 atom unit cell
        for formula, formula_data in data.items():
            for mpid, struc_dict in formula_data.items():
                struc = StrucTools(struc_dict).structure
                if len(struc) < 2:
                    supercell = StrucTools(struc).make_supercell([2, 2, 2])
                    data[formula][mpid] = supercell.as_dict()

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
            {formula_indicator :
                {struc_indicator : structure}}
            usually generated with get_strucs (or similar custom function)
        max_afm_combos (int)
            maximum number of AFM configurations to generate
        treat_as_nm (list)
            any normally mag els you'd like to treat as nonmagnetic for AFM enumeration
                e.g., if you know Ti won't be magnetic, you could set to ['Ti']
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
                    [list of magmoms (floats) for each site in the structure]}}}
        e.g.,
        {'Cr2O3' :
            {'mp-4321' :
                {'0' : [-5, 5, -5, 5, -5, 5, 0, 0, 0, 0, 0, 0, 0, 0]}}}
    """

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
    user_configs=None,
    make_launch_dirs=True,
    data_dir=os.getcwd().replace("scripts", "data"),
    calcs_dir=os.getcwd().replace("scripts", "calcs"),
    savename="launch_dirs.json",
    remake=False,
):
    """
    Args:
        strucs (dict)
            {formula_indicator
                : {struc_indicator
                    : structure}}
            usually generated w/ get_strucs (or similar custom function)
        magmoms (dict)
            {formula_indicator :
                {struc_indicator :
                    {AFM configuration index :
                        [list of magmoms on each site]}}
            usually generated w/ get_magmoms (or similar custom function)
        user_configs (dict)
            optional configs that apply to launch directories
                these usually get generated using get_launch_configs
                n_afm_configs = 0 by default
                    how many AFM configurations to run
                override_mag = False by default
                    could be 'nm' if you only want to run nonmagnetic,
                    won't check for whether structure is mag or not mag,
                    it will just do as you say
                    (not sure if this is properly implemented)
                ID_specific_vasp_configs: None by default
                    {<formula_indicator>_<struc_indicator> :
                        {'incar_mods' : {<incar_key> : <incar_val>},
                        {'kpoints_mods' : {<kpoints_key> : <kpoints_val>},
                        {'potcar_mods' : {<potcar_key> : <potcar_val>}}
        make_launch_dirs (bool)
            make launch directories (True) or just return launch dict (False)
        data_dir (str)
            directory to save fjson
                usually is this ../data
        calcs_dir (str)
            directory that holds all your calculations
                usually this is ../calcs or /scratch/..../calcs
        savename (str)
            filename for fjson in data_dir
        remake (bool)
            write (True) or just read (False) fjson
    Returns:
        {launch_dir (str) :
            {'magmom' : [list of magmoms for the structure in that launch_dir (list)],
             'ID_specific_vasp_configs' : {<formula_indicator>_<struc_indicator> : {desired configs for this entry}}}
        Returns the minimal list of directories that will house submission files (each of which launch a chain of calcs)
            note a chain of calcs must have the same structure and magnetic information, otherwise, there's no reason to chain them
                so the launch_dir defines: structure, magmom, ID-specific configs
        These launch_dirs have a very prescribed structure:
            calcs_dir / formula_indicator / struc_indicator / mag
            e.g.,
                ../calcs/Nd2O7Ru2/mp-19930/fm
                ../calcs/LiMn2O4_1/3/afm_4
                    (if LiMn2O4_1 was a unique compositional indicator and 3 was a unique structural indicator)
        also makes launch_dir and populates with POSCAR using strucs if make_dirs=True
    """

    fjson = os.path.join(data_dir, savename)
    if os.path.exists(fjson) and not remake:
        return read_json(fjson)

    if user_configs is None:
        user_configs = {}

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

            for launch_dir, params in launch_dirs.items():
                all_launch_dirs[launch_dir] = params

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
                    (calcs_dir/formula/ID/mag) to write and launch submission script in,
            'launch_dirs' :
                launch_dirs (dict)
                    {launch_dir (calcs_dir/formula/ID/mag) : {'magmoms' : [list of magmoms for each site in structure in launch_dir], 'ID_specific_vasp_configs' : {options}}},
            'user_configs' :
                user_configs (dict)
                    optional sub or slurm configs
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

            # prepare VASP directories and write submission script
            sub.write_sub

            # submit submission script to the queue
            if ready_to_launch:
                sub.launch_sub

            success = True
        except TypeError:
            # print("\nERROR: %s\n   will submit without multiprocessing" % launch_dir)
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
    user_configs=None,
    ready_to_launch=True,
    n_procs=1,
):
    """
    Prepares VASP inputs, writes submission script, and launches job for all launch_dirs
    Args:
        launch_dirs (dict)
            {launch_dir (str) :
                {'magmom' : [list of magmoms for the structure in that launch_dir (list)],
                 'ID_specific_vasp_configs' : {<formula_indicator>_<struc_indicator> : {desired configs for this entry}}}
             usually generated with get_launch_dirs
        user_configs (dict)
            optional sub or slurm configs
                these normally get generated using get_sub_configs and get_slurm_configs
                relaxation_xcs: default is ['gga']
                    list of xcs you want to at least run relax + static for
                static_addons: default is {'gga' : ['lobster']}
                    dictionary of things you want to do after a static is converged. eg {'metagga' : ['lobster', 'bs']}
                run_static_addons_before_all_relaxes: default is False
                    if False, prioritize relaxes finishing; if True, run static addons as soon as possible
                custom_calc_list: default is None
                    complete list of calcs in the order you want to run (if you don't want these generated automatically)
                start_with_loose: default is False
                    if True, add gga-loose or ggau-loose as your very first calc in the list to calculate
                fresh_restart: default is None
                    if you want to start certain calcs over set them in a list here. eg ['metaggau-lobster', 'metagga-bs']
                vasp: default is vasp_std
                    which vasp do you want to use
                        use vasp_gam for "loose" calcs (havent implemented yet)
                vasp_version: default is 6
                    version of VASP (can be 5 for 5.4.4 or 6 for 6.4.1)
                mpi_command: default is mpirun
                    how to launch on multicore/multinode (may be mpirun depending on compilation)
                manager: default is '#SBATCH'
                    how to manage interactions with the queue (some machines dont use slurm)
                machine: default is msi
                    which supercomputer
                execute_flags: default is ['srun', 'python', 'bin/lobster', 'bin/vasp', 'bader', 'mpirun']
                    how to figure out if a submission script needs to be launched
                n_procs_for_submission: default is 1
                    how many cores to parallelize the submission part of the launcher on
                        note: this has nothing to do w/ how many cores each VASP calc runs on
                        only affects how many cores this function gets executed on
                nodes: default is 1
                    how many nodes
                ntasks: default is 8
                    how many total cores
                time: default is 1440
                    how long in minutes before hitting walltime
                error: default is log.e
                    where to write slurm errors to in launch_dir
                output: default is log.o
                    where to write slurm output to in launch_dir
                account: default is cbartel
                    account to charge
                partition: default is agsmall,msismall,msidmc
                    partition to use
                job-name: default is None
                    unique job name; if none provided, will default to formula_indicator.struc_indicator.mag.project_dir
                        e.g., default might be SrZrS3.mp-1234.afm_0.perovskite_chalcs
                mem-per-cpu: default is None
                    specify mem per core
                mem-per-gpu: default is None
                    specify mem per gpu core
                constraint: default is None
                    may not need this ever on MSI
                qos: default is None
                    may not need this ever on MSI
        ready_to_launch (bool)
            write and launch (True) or just write submission scripts (False)
        n_procs (int or str)
            if parallelizing, how many processors
    Returns:
        None
    """

    if user_configs is None:
        user_configs = {}

    submit_args = {
        "launch_dirs": launch_dirs,
        "user_configs": user_configs,
        "ready_to_launch": ready_to_launch,
        "parallel": False if n_procs == 1 else True,
    }

    if n_procs == 1:
        # print("\n\n submitting calculations in serial\n\n")
        for launch_dir in launch_dirs:
            curr_submit_args = submit_args.copy()
            curr_submit_args["launch_dir"] = launch_dir
            submit_one_calc(curr_submit_args)
        return
    elif n_procs == "all":
        n_procs = multip.cpu_count() - 1

    # print("\n\n submitting calculations in parallel\n\n")
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

    # print(
    #     "%i/%i calculations submitted with multiprocessing"
    #     % (len(submitted_w_multiprorcessing), len(statuses))
    # )
    for status in failed_w_multiprocessing:
        launch_dir = status["launch_dir"]
        curr_submit_args = submit_args.copy()
        curr_submit_args["launch_dir"] = launch_dir
        submit_one_calc(curr_submit_args)

    return


def get_results(
    launch_dirs,
    user_configs=None,
    data_dir=os.getcwd().replace("scripts", "data"),
    savename="results.json",
    remake=False,
):
    """
    Args:
        launch_dirs (dict)
            {launch_dir (str) :
                {'magmom' : [list of magmoms for the structure in that launch_dir (list)],
                 'ID_specific_vasp_configs' : {<formula_indicator>_<struc_indicator> : {desired configs for this entry}}}
        user_configs (dict)
            optional analysis configurations
                usually generated using get_analysis_configs
                only_calc: default is 'static'
                    only retrieve data from the static calculations
                only_xc: default is None
                    if None, retrieve all xcs, else retrieve only the one specified
                check_relax_energy: default is True
                    make sure the relax calculation and the static have similar energies
                include_metadata: default is True
                    include metadata like INCAR, KPOINTS, POTCAR settings
                include_calc_setup: default is True
                    include things related to the calculation setup -- mag, xc, etc
                include_structure: default is True
                    include the relaxed crystal structure as a dict
                include_trajectory: default is False
                    include the compact trajectory from the vasprun.xml
                include_mag: default is False
                    include the relaxed magnetization info as as dict
                include_tdos: default is False
                    include the light version of the density of states
                include_pdos: default is False
                    include heavy dos
                include_charge: default is False
                    include partial chage info
                include_madelung: default is False
                    include Madelung energies
                include_tcohp: default is False
                    include light COHPCAR data
                include_pcohp: default is False
                    include heavy COHPCAR data
                include_tcoop: default is False
                    include light COOPCAR data
                include_pcoop: default is False
                    include heavy COOPCAR data
                include_tcobi: default is False
                    include light COBICAR data
                include_pcobi: default is False
                    include heavy COBICAR data
                include_entry: default is False
                    include pymatgen computed structure entry
                create_cif: default is True
                    create a .cif file for each CONTCAR
                verbose: default is True
                    print stuff as things get analyzed
                remake_results: default is False
                    whether or not to rerun analyzer in each calculation directory
        data_dir (str)
            directory to save fjson
        savename (str)
            filename for fjson in data_dir
        remake (bool)
            write (True) or just read (False) fjson
    Returns:
        {formula--ID--mag--xc-calc (str) :
            {scraped results from VASP calculation}}
    """

    fjson = os.path.join(data_dir, savename)
    if os.path.exists(fjson) and not remake:
        return read_json(fjson)

    if user_configs is None:
        user_configs = {}

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
    calc_types_to_search=("static",),
    compute_Ef=True,
    standard="dmc",
    data_dir=os.getcwd().replace("scripts", "data"),
    savename="gs.json",
    remake=False,
):
    """
    Args:
        results (dict)
            {formula--ID--mag--xc-calc (str) : {scraped results from VASP calculation}}
            usually generated with get_results
        include_structure (bool)
            include the structure or not
        non_default_functional (str)
            if you're not using r2SCAN or PBE
        calc_types_to_search (tuple)
            tuple of calculation types to include, e.g., ("static", "defect_neutral, "defect_charged_p1")
        compute_Ef (bool)
            if True, compute formation enthalpy
        data_dir (str)
            directory to save fjson
        savename (str)
            filename for fjson in data_dir
        remake (bool)
            write (True) or just read (False) fjson
    Returns:
        for "static" only:
            {xc (str, the exchange-correlation method) :
                {formula (str) :
                    {'E' : energy of the ground-structure,
                    'key' : formula--ID--mag--xc-calc for the ground-state structure,
                    'structure' : structure of the ground-state structure,
                    'n_started' : how many polymorphs you tried to calculate,
                    'n_converged' : how many polymorphs are converged,
                    'complete' : True if n_converged = n_started (i.e., all structures for this formula at this xc are done),
                    'Ef' : formation enthalpy at 0 K}}}
        otherwise:
            {calc_type (str, [static, defect_neutral, ...]) :
                {xc (str, the exchange-correlation method) :
                    {formula (str) :
                        {'E' : energy of the ground-structure,
                        'key' : formula--ID--mag--xc-calc for the ground-state structure,
                        'structure' : structure of the ground-state structure,
                        'n_started' : how many polymorphs you tried to calculate,
                        'n_converged' : how many polymorphs are converged,
                        'complete' : True if n_converged = n_started (i.e., all structures for this formula at this xc are done),
                        'Ef' : formation enthalpy at 0 K}}}}
    """
    fjson = os.path.join(data_dir, savename)
    if os.path.exists(fjson) and not remake:
        return read_json(fjson)

    results = {
        key: results[key]
        for key in results
        if results[key]["meta"]["setup"]["calc"] in calc_types_to_search
    }

    calc_types = sorted(
        list(set([results[key]["meta"]["setup"]["calc"] for key in results]))
    )

    gs = {}
    for calc_type in calc_types:
        gs[calc_type] = {
            xc: {}
            for xc in sorted(
                list(
                    set(
                        [
                            results[key]["meta"]["setup"]["xc"]
                            for key in results
                            if results[key]["meta"]["setup"]["calc"] == calc_type
                        ]
                    )
                )
            )
        }

    for calc_type in gs:
        for xc in gs[calc_type]:
            keys = [
                k
                for k in results
                if results[k]["meta"]["setup"]["calc"] == calc_type
                if results[k]["meta"]["setup"]["xc"] == xc
                if results[k]["results"]["formula"]
            ]

            unique_formulas = sorted(
                list(set([results[key]["results"]["formula"] for key in keys]))
            )
            for formula in unique_formulas:
                gs[calc_type][xc][formula] = {}
                formula_keys = [
                    k for k in keys if results[k]["results"]["formula"] == formula
                ]
                converged_keys = [
                    k for k in formula_keys if results[k]["results"]["convergence"]
                ]
                if not converged_keys:
                    gs_energy, gs_structure, gs_key = None, None, None
                else:
                    energies = [
                        results[k]["results"]["E_per_at"] for k in converged_keys
                    ]
                    gs_energy = min(energies)
                    gs_key = converged_keys[energies.index(gs_energy)]
                    if include_structure:
                        gs_structure = results[gs_key]["structure"]
                complete = True if len(formula_keys) == len(converged_keys) else False
                gs[calc_type][xc][formula] = {
                    "E": gs_energy,
                    "key": gs_key,
                    "n_started": len(formula_keys),
                    "n_converged": len(converged_keys),
                    "complete": complete,
                }
                if include_structure:
                    gs[calc_type][xc][formula]["structure"] = gs_structure

    if compute_Ef:
        for calc_type in gs:
            for xc in gs[calc_type]:
                if not non_default_functional:
                    functional = "r2scan" if xc == "metagga" else "pbe"
                else:
                    functional = non_default_functional
                mus = ChemPots(functional=functional, standard=standard).chempots
                for formula in gs[calc_type][xc]:
                    E = gs[calc_type][xc][formula]["E"]
                    if E:
                        Ef = FormationEnthalpy(
                            formula=formula, E_DFT=E, chempots=mus
                        ).Ef
                    else:
                        Ef = None
                    gs[calc_type][xc][formula]["Ef"] = Ef

    if calc_types_to_search in (("static",), "static", ["static"]):
        gs = gs["static"]

    write_json(gs, fjson)
    return read_json(fjson)


def check_gs(gs):
    """
    checks that this dictionary is generated properly
    """

    print("\nchecking ground-states")

    if gs == {}:
        return

    calc_types_or_xcs = list(gs.keys())

    static_gs_only = not any(
        isinstance(gs[calc_type_or_xc][xc_or_formula][formula_or_info], dict)
        for calc_type_or_xc in calc_types_or_xcs
        for xc_or_formula in gs[calc_type_or_xc]
        for formula_or_info in gs[calc_type_or_xc][xc_or_formula]
    )

    if static_gs_only:
        gs = {"static": gs}
        calc_types = ["static"]
    else:
        calc_types = calc_types_or_xcs

    for calc_type in calc_types:
        xcs = list(gs[calc_type].keys())
        print("  calc_type = %s" % calc_type)
        for xc in xcs:
            print("    xc = %s" % xc)
            formulas = list(gs[calc_type][xc].keys())
            n_formulas = len(formulas)
            n_formulas_complete = len(
                [k for k in formulas if gs[calc_type][xc][k]["complete"]]
            )
            print(
                "      %i/%i formulas with all calculations completed"
                % (n_formulas_complete, n_formulas)
            )
            for formula in gs[calc_type][xc]:
                if "Ef" in gs[calc_type][xc][formula]:
                    if gs[calc_type][xc][formula]["Ef"]:
                        print(
                            "      %s : Ef = %.2f eV/at"
                            % (formula, gs[calc_type][xc][formula]["Ef"])
                        )


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
            {formula--ID--mag--xc-calc (str) : {scraped results from VASP calculation}}
            usually generated with get_results
        gs (dict):
            {xc (str, the exchange-correlation method) :
                {formula (str) :
                    {'E' : energy of the ground-structure,
                    'key' : formula--ID--mag--xc-calc for the ground-state structure,
                    'structure' : structure of the ground-state structure,
                    'n_started' : how many polymorphs you tried to calculate,
                    'n_converged' : how many polymorphs are converged,
                    'complete' : True if n_converged = n_started (i.e., all structures for this formula at this xc are done),
                    'Ef' : formation enthalpy at 0 K}
            usually generated with get_gs
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
    
    if "lobster" in gs:
        thermo_results = {calc: {xc: {formula: {} for formula in gs[calc][xc]} for xc in gs[calc]} for calc in gs}
    
    thermo_results = {xc: {formula: {} for formula in gs[xc]} for xc in gs}

    for key in results:
        tmp_thermo = {}

        xc = results[key]["meta"]["setup"]["xc"]
        formula = results[key]["results"]["formula"]
        calc = results[key]["meta"]["setup"]["calc"]
        
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
            if "lobster" in gs:
                gs = gs[calc]
                
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
        
        print(calc, xc , formula, ID, tmp_thermo)
        if "lobster" in thermo_results:
                thermo_results[calc][xc][formula][ID] = tmp_thermo
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
    only_formulas=None,
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
    if only_formulas is None:
        only_formulas = []

    fjson = os.path.join(data_dir, savename)
    if os.path.exists(fjson) and not remake:
        return read_json(fjson)

    for key in results:
        calc_dir = results[key]["meta"]["calc_dir"]
        xc = results[key]["meta"]["setup"]["xc"]
        calc = results[key]["meta"]["setup"]["calc"]
        if calc not in ["lobster", "bs"]:
            continue
        if not results[key]["results"]["convergence"]:
            continue
        ID = "__".join(
            [
                results[key]["meta"]["setup"]["formula_indicator"],
                results[key]["meta"]["setup"]["struc_indicator"],
                results[key]["meta"]["setup"]["mag"],
            ]
        )
        formula = results[key]["results"]["formula"]
        thermo_result =thermo_results[calc][xc][formula][ID]
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
    with open(flauncher_sub, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash -l\n")
        f.write("#SBATCH --nodes=1\n")
        f.write("#SBATCH --ntasks=32\n")
        f.write("#SBATCH --time=4:00:00\n")
        f.write("#SBATCH --mem=8G\n")
        f.write("#SBATCH --error=_log_launcher.e\n")
        f.write("#SBATCH --output=_log_launcher.o\n")
        f.write("#SBATCH --account=cbartel\n")
        f.write("#SBATCH --job-name=%s\n" % launch_job_name)
        f.write("#SBATCH --partition=agsmall,msidmc\n")
        f.write("\npython launcher.py\n")


def main():
    return


if __name__ == "__main__":
    main()