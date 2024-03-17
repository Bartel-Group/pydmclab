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

# from pymatgen.entries.computed_entries import ComputedStructureEntry


def get_vasp_configs(
    run_lobster=False,
    run_bandstructure=False,
    run_parchg=False,
    run_dielectric=False,
    run_phonons=False,
    detailed_dos=False,
    modify_loose_incar=False,
    modify_relax_incar=False,
    modify_static_incar=False,
    modify_loose_kpoints=False,
    modify_relax_kpoints=False,
    modify_static_kpoints=False,
    modify_loose_potcar=False,
    modify_relax_potcar=False,
    modify_static_potcar=False,
):
    """

    how to modify VASP calculations from the defaults

    see pydmclab.hpc.vasp for defaults

    Args:
        run_lobster (bool):
            True to run LOBSTER for static calculations
                sets vasp_configs['lobster_static'] = True

        run_bandstructure (bool):
            True to run a bandstructure calculation after your static calculations
                sets vasp_configs['generate_bandstructure'] = True

        run_parchg (bool):
            True to run a parchg calculation after your static calculations
                sets vasp_configs['generate_parchg'] = True

        run_dielectric (bool):
            True to get the dielectric tensor on your static calculation

        run_phonons (False or list):
            False = don't do phonons
            list (3 elements) = supercell expansion for finite displacements (eg [2, 3, 4])
            True = use default supercell expansion = [2, 2, 2]

        detailed_dos (bool or int):
            if you're running LOBSTER, this will determine how many (E, DOS/COHP) points you get
                if False, COHPSteps = 400
                if True, COHPSteps = 4000
                if int, COHPSteps = detailed_dos
            if run_lobster is False, this is ignored

        modify_< calc >_incar (False or dict):

            < calc > could be loose, relax, or static

            dictionary of {incar flag (str) : setting for that flag}
                modifies only that < calc >
                gets passed to vasp_configs["< calc >_incar"]

            e.g., modify_relax_incar = {'ISIF' : 2, 'ISYM' : -1}
                would update those INCAR settings in your relax calculations

        modify_< calc >_kpoints (False or dict):

            < calc > could be loose, relax, or static

            dictionary representation of the KPOINTS you want to use if non-default
                modifies only that < calc >
                gets passed to vasp_configs["< calc >_kpoints"]

            e.g., modify_static_kpoints = {'reciprocal_density' : 1000}
                would update the KPOINT generation for your static calculations

        modify_< calc >_potcar (False or dict):

            < calc > could be loose, relax, or static

            dictionary representation of the POTCAR you want to use if non-default
                modifies only that < calc >
                gets passed to vasp_configs["< calc >_potcar"]

            e.g., modify_static_potcar = {'W' : 'W_pv', 'O' : 'O_s}
                would update the POTCARs as requested for your static calculations

    Returns:
        dictionary of VASP_CONFIGS
            {config_key : config_value}

            to get passed as user_configs to VASPSetUp (through SubmitTools)
    """
    vasp_configs = {
        "lobster_static": run_lobster,
        "generate_bandstructure": run_bandstructure,
        "generate_parchg": run_parchg,
        "generate_dielectric": run_dielectric,
    }

    if isinstance(run_phonons, list):
        vasp_configs["supercell_grid_for_finite_displacements"] = run_phonons
        generate_finite_displacements = True
    else:
        generate_finite_displacements = run_phonons

    vasp_configs["generate_finite_displacements"] = generate_finite_displacements

    if vasp_configs["generate_bandstructure"]:
        vasp_configs["lobster_static"] = True

    # INCARs
    if modify_loose_incar:
        vasp_configs["loose_incar"] = modify_loose_incar
    if modify_relax_incar:
        vasp_configs["relax_incar"] = modify_relax_incar
    if modify_static_incar:
        vasp_configs["static_incar"] = modify_static_incar

    # KPOINTSs
    if modify_loose_kpoints:
        vasp_configs["loose_kpoints"] = modify_loose_kpoints
    if modify_relax_kpoints:
        vasp_configs["relax_kpoints"] = modify_relax_kpoints
    if modify_static_kpoints:
        vasp_configs["static_kpoints"] = modify_static_kpoints

    # POTCARs
    if modify_loose_potcar:
        vasp_configs["loose_potcar"] = modify_loose_potcar
    if modify_relax_potcar:
        vasp_configs["relax_potcar"] = modify_relax_potcar
    if modify_static_potcar:
        vasp_configs["static_potcar"] = modify_static_potcar

    if detailed_dos:
        if isinstance(detailed_dos, bool):
            vasp_configs["COHPSteps"] = 4000
        else:
            vasp_configs["COHPSteps"] = detailed_dos

        if ("static_incar" in vasp_configs) and isinstance(
            vasp_configs["static_incar"], dict
        ):
            vasp_configs["static_incar"]["NEDOS"] = vasp_configs["COHPSteps"]

        else:
            vasp_configs["static_incar"] = {"NEDOS": vasp_configs["COHPSteps"]}

    return vasp_configs


def get_slurm_configs(
    total_nodes=1,
    cores_per_node=8,
    walltime_in_hours=95,
    mem_per_core="all",
    partition="agsmall,msidmc",
    error_file="log.e",
    output_file="log.o",
    account="cbartel",
):
    """

    how to modify slurm configurations for each VASP job

    (see pydmclab.data.data._slurm_configs.yaml for defaults)

    see pydmclab.hpc.submit.SubmitTools for more info


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
    machine="msi",
    submit_calculations_in_parallel=False,
    delete_all_calculations_and_start_over=False,
    rerun_lobster=False,
    mpi_command="mpirun",
    special_packing=False,
    vasp_version=6,
    skip_loose=False,
):
    """

    configs related to preparing submission scripts and submitting VASP calculations

        see defaults in pydmclab.data.data._sub_configs.yaml
        see pydmclab.hpc.submit.SubmitTools for more info

    Args:
        submit_calculations_in_parallel (bool or int):
            whether to prepare submission scripts in parallel or not
                False: use 1 processor
                True: use all available processors - 1
                int: use that many processors
            if this is not False, you should not run this on a login node

        delete_all_calculations_and_start_over (bool):
            if True, start all calculations over (ie delete all outputs)
                ** you should rarely use this! **

        rerun_lobster (bool) :
            if True, rerun lobster even if it has already been run
                ** you should rarely use this! **

        mpi_command (str):
            the command to use for mpi (eg mpirun, srun, etc)

        special_packing (dict):
            if you want to change the loose --> relax --> static flow for some functional
                e.g., {'metagga' : ['metagga-loose', 'metagga-static']}

    Returns:
        {config_name : config_value}

    """
    sub_configs = {}

    if not submit_calculations_in_parallel:
        n_procs = 1
    else:
        if submit_calculations_in_parallel == True:
            n_procs = multip.cpu_count() - 1
        else:
            n_procs = int(submit_calculations_in_parallel)

    sub_configs["n_procs"] = n_procs

    if delete_all_calculations_and_start_over:
        sub_configs["fresh_restart"] = True

    if rerun_lobster:
        sub_configs["force_postprocess"] = True

    sub_configs["mpi_command"] = mpi_command

    if special_packing:
        sub_configs["packing"] = {}
        for xc in special_packing:
            sub_configs["packing"][xc] = special_packing[xc]

    sub_configs["machine"] = machine

    sub_configs["vasp_version"] = vasp_version

    if skip_loose:
        sub_configs["skip_loose"] = True

    return sub_configs


def get_launch_configs(
    standards=["dmc"],
    xcs=["metagga"],
    use_mp_thermo_data=False,
    n_afm_configs=0,
    skip_xcs_for_standards={"mp": ["gga", "metagga"]},
):
    """

    configs related to launching chains of calculations

    see defaults in pydmclab.data.data._launch_configs.yaml

    see pydmclab.hpc.launch.LaunchTools for more info

    Args:
        standards (list):
            list of standards (str) you'd like to calculate
                e.g., ['dmc']

        xcs (list):
            list of xcs (str) you'd like to calculate for each standard
                e.g., ['metagga', 'ggau']

        use_mp_thermo_data (bool):
            True if you are going to use formation energies provided in Materials Project for phase stability analysis
                will automatically update the xcs/standards you want to launch to run these calcs

        n_afm_configs (int):
            number of antiferromagnetic configurations to run for each structure (0 if you don't want to run AFM)

        skip_xcs_for_standards (dict):
            dictionary of xcs to skip for a given standard
                in principle, you may not want to run every xc for every standard. this gives you a mechanism to encode skipping combinations
                Defaults to {"mp": ["gga", "metagga"]}.
                    - e.g., we don't want to run GGA or MetaGGA MP calculations because MP uses GGA+U (for now)

    Returns:
        dictionary of launch configurations
            {config param : config value}
    """

    launch_configs = {}

    to_launch = {}

    for standard in standards:
        to_launch[standard] = []
        for xc in xcs:
            if (standard in skip_xcs_for_standards) and (
                xc in skip_xcs_for_standards[standard]
            ):
                continue
            to_launch[standard].append(xc)

    standards_to_keep = [standard for standard in to_launch if to_launch[standard]]
    to_launch = {standard: to_launch[standard] for standard in standards_to_keep}

    if use_mp_thermo_data:
        # make sure we run GGA+U for MP consistency
        to_launch["mp"] = ["ggau"]

    launch_configs["to_launch"] = to_launch

    launch_configs["n_afm_configs"] = n_afm_configs

    return launch_configs


def get_analysis_configs(
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

    function for modifying analysis configs from the defaults (see pydmclab.data.data._batch_analysis_configs.yaml for defaults)

    Args:
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

    analysis_configs["n_procs"] = n_procs

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

    return analysis_configs


def get_query(
    api_key,
    search_for,
    properties=None,
    max_Ehull=0.1,
    max_sites_per_structure=100,
    max_polymorph_energy=0.1,
    only_gs=False,
    include_structure=True,
    max_strucs_per_cmpd=5,
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
    treat_as_nm=[],
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
    ID_specific_vasp_configs={},
    make_launch_dirs=True,
    refresh_configs=True,
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

        ID_specific_vasp_configs (dict)
            optional VASP configurations for specific IDs

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
        {launch dir (str, formula/ID/standard/mag) :
            {'xcs' : [list of final xcs to run (str)],
            'magmoms' : [list of magmoms (floats) for each site in the structure]}}

        also makes launch_dir and populates with POSCAR using strucs if make_dirs=True

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
            top_level = formula
            unique_ID = ID

            launch = LaunchTools(
                calcs_dir=calcs_dir,
                user_configs=user_configs,
                magmoms=curr_magmoms,
                ID_specific_vasp_configs=ID_specific_vasp_configs,
                structure=structure,
                top_level=top_level,
                unique_ID=unique_ID,
                refresh_configs=refresh_configs,
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
        print("   these final xcs: %s" % launch_dirs[d]["xcs"])


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
            'refresh_configs' :
                refresh_configs (list)
                    list of which configs to refresh,
            'ready_to_launch':
                ready_to_launch (bool)
                    write and launch (True) or just write submission scripts (False)
                }

    Returns:
        None

    """
    launch_dir = submit_args["launch_dir"]
    launch_dirs = submit_args["launch_dirs"]
    user_configs = submit_args["user_configs"]
    refresh_configs = submit_args["refresh_configs"]
    ready_to_launch = submit_args["ready_to_launch"]
    running_in_parallel = submit_args["parallel"]

    curr_user_configs = user_configs.copy()

    if "ID_specific_vasp_configs" in launch_dirs[launch_dir]:
        if launch_dirs[launch_dir]["ID_specific_vasp_configs"]:
            curr_user_configs.update(
                launch_dirs[launch_dir]["ID_specific_vasp_configs"]
            )

    # what are our terminal xcs for that launch_dir
    final_xcs = launch_dirs[launch_dir]["xcs"]

    # what magmoms apply to that launch_dir
    magmom = launch_dirs[launch_dir]["magmom"]

    if running_in_parallel:
        try:
            sub = SubmitTools(
                launch_dir=launch_dir,
                final_xcs=final_xcs,
                magmom=magmom,
                user_configs=curr_user_configs,
                refresh_configs=refresh_configs,
            )

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
            final_xcs=final_xcs,
            magmom=magmom,
            user_configs=curr_user_configs,
            refresh_configs=refresh_configs,
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
    refresh_configs=["vasp", "sub", "slurm"],
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

        refresh_configs (list)
            list of which configs to refresh

        ready_to_launch (bool)
            write and launch (True) or just write submission scripts (False

    Returns:
        None

    """

    submit_args = {
        "launch_dirs": launch_dirs,
        "user_configs": user_configs,
        "refresh_configs": refresh_configs,
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
    submit_args["refresh_configs"] = refresh_configs
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
    refresh_configs=True,
    data_dir=os.getcwd().replace("scripts", "data"),
    savename="results.json",
    remake=False,
):
    """
    Args:
        launch_dirs (dict)
            {launch_dir (formula/ID/standard/mag) : {'xcs' : [list of final_xcs], 'magmoms' : [list of magmoms for each site in structure in launch_dir]}}

        user_configs (dict)
            optional analysis configurations

        refresh_configs (bool)
            refresh configs (True) or just use existing configs (False)

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

    analyzer = AnalyzeBatch(
        launch_dirs, user_configs=user_configs, refresh_configs=refresh_configs
    )

    data = analyzer.results

    write_json(data, fjson)
    return read_json(fjson)


def check_results(results):
    keys_to_check = list(results.keys())

    converged = 0
    for key in keys_to_check:
        if "--" in key:
            delimiter = "--"
        else:
            delimiter = "."
        top_level, ID, standard, mag, xc_calc = key.split(delimiter)
        data = results[key]
        convergence = results[key]["results"]["convergence"]
        print("\n%s" % key)
        print("convergence = %s" % convergence)
        if convergence:
            converged += 1
            # print("\n%s" % key)
            print("E (static) = %.2f" % data["results"]["E_per_at"])

    print("\n\n SUMMARY: %i/%i converged" % (converged, len(keys_to_check)))


def get_gs(
    results,
    include_structure=False,
    non_default_functional=None,
    compute_Ef=True,
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
    {standard (str, the calculation standard) :
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

    standards = sorted(
        list(set([results[key]["meta"]["setup"]["standard"] for key in results]))
    )

    gs = {
        standard: {
            xc: {}
            for xc in sorted(
                list(
                    set(
                        [
                            results[key]["meta"]["setup"]["xc"]
                            for key in results
                            if results[key]["meta"]["setup"]["standard"] == standard
                        ]
                    )
                )
            )
        }
        for standard in standards
    }

    for standard in gs:
        for xc in gs[standard]:
            keys = [
                k
                for k in results
                if results[k]["meta"]["setup"]["standard"] == standard
                if results[k]["meta"]["setup"]["xc"] == xc
                if results[k]["results"]["formula"]
            ]

            unique_formulas = sorted(
                list(set([results[key]["results"]["formula"] for key in keys]))
            )
            for formula in unique_formulas:
                gs[standard][xc][formula] = {}
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
                    gs_structure = results[gs_key]["structure"]
                complete = True if len(formula_keys) == len(converged_keys) else False
                gs[standard][xc][formula] = {
                    "E": gs_energy,
                    "key": gs_key,
                    "n_started": len(formula_keys),
                    "n_converged": len(converged_keys),
                    "complete": complete,
                }
                if include_structure:
                    gs[standard][xc][formula]["structure"] = gs_structure

    if compute_Ef:
        for standard in gs:
            for xc in gs[standard]:
                if not non_default_functional:
                    functional = "r2scan" if xc == "metagga" else "pbe"
                else:
                    functional = non_default_functional
                mus = ChemPots(functional=functional, standard=standard).chempots
                for formula in gs[standard][xc]:
                    E = gs[standard][xc][formula]["E"]
                    if E:
                        Ef = FormationEnthalpy(
                            formula=formula, E_DFT=E, chempots=mus
                        ).Ef
                    else:
                        Ef = None
                    gs[standard][xc][formula]["Ef"] = Ef
    write_json(gs, fjson)
    return read_json(fjson)


def check_gs(gs):
    """
    checks that this dictionary is generated properly

    Args:
        gs (_type_): _description_

    """

    print("\nchecking ground-states")
    standards = gs.keys()
    print("standards = ", standards)
    for standard in standards:
        print("\nworking on %s standard" % standard)
        xcs = list(gs[standard].keys())
        for xc in xcs:
            print("  xc = %s" % xc)
            formulas = list(gs[standard][xc].keys())
            n_formulas = len(formulas)
            n_formulas_complete = len(
                [k for k in formulas if gs[standard][xc][k]["complete"]]
            )
            print(
                "%i/%i formulas with all calculations completed"
                % (n_formulas_complete, n_formulas)
            )
            for formula in gs[standard][xc]:
                if "Ef" in gs[standard][xc][formula]:
                    if gs[standard][xc][formula]["Ef"]:
                        print(
                            "%s : %.2f eV/at"
                            % (formula, gs[standard][xc][formula]["Ef"])
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
        {standard (str) :
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

    thermo_results = {
        standard: {
            xc: {formula: {} for formula in gs[standard][xc]} for xc in gs[standard]
        }
        for standard in gs
    }

    for key in results:
        tmp_thermo = {}

        standard = results[key]["meta"]["setup"]["standard"]
        xc = results[key]["meta"]["setup"]["xc"]
        formula = results[key]["results"]["formula"]
        ID = "__".join(
            [
                results[key]["meta"]["setup"]["formula_tag"],
                results[key]["meta"]["setup"]["ID"],
                results[key]["meta"]["setup"]["mag"],
            ]
        )
        E = results[key]["results"]["E_per_at"]
        formula = results[key]["results"]["formula"]
        structure = results[key]["structure"]
        if structure:
            calcd_formula = StrucTools(structure).formula
        else:
            calcd_formula = None
        tmp_thermo["E"] = E
        tmp_thermo["key"] = key
        tmp_thermo["formula"] = formula
        tmp_thermo["calculated_formula"] = calcd_formula

        if E:
            gs_key = gs[standard][xc][formula]["key"]
            if "Ef" in gs[standard][xc][formula]:
                gs_Ef = gs[standard][xc][formula]["Ef"]
            else:
                gs_Ef = None
            gs_E = gs[standard][xc][formula]["E"]
            delta_E_gs = E - gs_E

            if key == gs_key:
                tmp_thermo["is_gs"] = True
            else:
                tmp_thermo["is_gs"] = False

            tmp_thermo["dE_gs"] = delta_E_gs
            if gs_Ef:
                tmp_thermo["Ef"] = gs_Ef + delta_E_gs
            else:
                tmp_thermo["Ef"] = None
            tmp_thermo["all_polymorphs_converged"] = gs[standard][xc][formula][
                "complete"
            ]

        else:
            tmp_thermo["dE_gs"] = None
            tmp_thermo["Ef"] = None
            tmp_thermo["is_gs"] = False
            tmp_thermo["all_polymorphs_converged"] = False

        thermo_results[standard][xc][formula][ID] = tmp_thermo

    write_json(thermo_results, fjson)
    return read_json(fjson)


def check_thermo_results(thermo):
    print("\nchecking thermo results")

    for standard in thermo:
        print("\n\nworking on %s standard" % standard)
        for xc in thermo[standard]:
            print("\nxc = %s" % xc)
            for formula in thermo[standard][xc]:
                print("formula = %s" % formula)
                print(
                    "%i polymorphs converged"
                    % len(
                        [
                            k
                            for k in thermo[standard][xc][formula]
                            if thermo[standard][xc][formula][k]["E"]
                        ]
                    )
                )
                gs_ID = [
                    k
                    for k in thermo[standard][xc][formula]
                    if thermo[standard][xc][formula][k]["is_gs"]
                ]
                if gs_ID:
                    gs_ID = gs_ID[0]
                    print("%s is the ground-state structure" % gs_ID)

    print("\n\n  SUMMARY  ")
    for standard in thermo:
        for xc in thermo[standard]:
            print("~~%s~~" % "_".join([standard, xc]))
            converged_formulas = []
            for formula in thermo[standard][xc]:
                for ID in thermo[standard][xc][formula]:
                    if thermo[standard][xc][formula][ID]["all_polymorphs_converged"]:
                        converged_formulas.append(formula)

            converged_formulas = list(set(converged_formulas))
            print(
                "%i/%i formulas have all polymorphs converged"
                % (len(converged_formulas), len(thermo[standard][xc].keys()))
            )


def get_dos_results(
    results,
    thermo_results,
    only_gs=True,
    only_xc="metagga",
    only_formulas=[],
    only_standard="dmc",
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
        standard, xc = (
            results[key]["meta"]["setup"]["standard"],
            results[key]["meta"]["setup"]["xc"],
        )
        ID = "__".join(
            [
                results[key]["meta"]["setup"]["formula_tag"],
                results[key]["meta"]["setup"]["ID"],
                results[key]["meta"]["setup"]["mag"],
            ]
        )
        formula = results[key]["results"]["formula"]
        thermo_result = thermo_results[standard][xc][formula][ID]
        if only_gs:
            if not thermo_result["is_gs"]:
                continue
        if only_formulas:
            if thermo_result["formula"] not in only_formulas:
                continue
        if only_xc:
            if xc != only_xc:
                continue
        if only_standard:
            if standard != only_standard:
                continue
        av = AnalyzeVASP(calc_dir)
        if "tdos" in dos_to_store:
            pdos = av.pdos(remake=regenerate_dos)
            tdos = av.tdos(pdos=pdos, remake=regenerate_dos)
            thermo_results[standard][xc][formula][ID]["tdos"] = tdos
        if "pdos" in dos_to_store:
            thermo_results[standard][xc][formula][ID]["pdos"] = pdos
        if "tcohp" in dos_to_store:
            pcohp = av.pcohp(remake=regenerate_cohp)
            tcohp = av.tcohp(pcohp=pcohp, remake=regenerate_cohp)
            thermo_results[standard][xc][formula][ID]["tcohp"] = tcohp
        if "pcohp" in dos_to_store:
            thermo_results[standard][xc][formula][ID]["pcohp"] = pcohp
        if "tcoop" in dos_to_store:
            pcohp = av.pcohp(are_coops=True, remake=regenerate_cohp)
            tcohp = av.tcohp(pcohp=pcohp, remake=regenerate_cohp)
            thermo_results[standard][xc][formula][ID]["tcoop"] = tcohp
        if "pcoop" in dos_to_store:
            thermo_results[standard][xc][formula][ID]["pcoop"] = pcohp
        if "tcobi" in dos_to_store:
            pcohp = av.pcohp(are_cobis=True, remake=regenerate_cohp)
            tcohp = av.tcohp(pcohp=pcohp, remake=regenerate_cohp)
            thermo_results[standard][xc][formula][ID]["tcobi"] = tcohp
        if "pcobi" in dos_to_store:
            thermo_results[standard][xc][formula][ID]["pcobi"] = pcohp

    write_json(thermo_results, fjson)
    return read_json(fjson)


def get_entries(
    results,
    data_dir=os.getcwd().replace("scripts", "data"),
    savename="entries.json",
    remake=False,
):
    fjson = os.path.join(data_dir, savename)
    if os.path.exists(fjson) and not remake:
        return read_json(fjson)
    d = {"entries": [results[k]["entry"] for k in results]}
    write_json(d, fjson)
    return read_json(fjson)


def get_mp_entries(
    chemsyses,
    api_key,
    data_dir=os.getcwd().replace("scripts", "data"),
    savename="mp_entries.json",
    remake=False,
):
    fjson = os.path.join(data_dir, savename)
    if os.path.exists(fjson) and not remake:
        return read_json(fjson)

    mpq = MPQuery(api_key=api_key)
    out = {}
    for chemsys in chemsyses:
        data = mpq.get_entries_for_chemsys(chemsys)
        out[chemsys] = list(data.values())
    write_json(out, fjson)
    return read_json(fjson)


def get_merged_entries(
    my_entries,
    mp_entries,
    restrict_mp_xc_to=None,
    restrict_my_xc_to=None,
    data_dir=os.getcwd().replace("scripts", "data"),
    savename="merged_entries_for_mp_Ef.json",
    remake=False,
):
    fjson = os.path.join(data_dir, savename)
    if os.path.exists(fjson) and not remake:
        return read_json(fjson)

    if restrict_my_xc_to == "GGA":
        my_allowed_xcs = ["gga"]
    elif restrict_my_xc_to == "GGA+U":
        my_allowed_xcs = ["ggau"]
    elif restrict_my_xc_to == "r2SCAN":
        my_allowed_xcs = ["r2scan"]
    else:
        my_allowed_xcs = None

    if restrict_mp_xc_to in ["GGA", "GGA+U"]:
        mp_allowed_xcs = ["GGA", "GGA+U", "PBE", "PBE+U"]
    elif restrict_my_xc_to == "r2SCAN":
        mp_allowed_xcs = ["r2SCAN"]
    else:
        mp_allowed_xcs = None

    entries = {}
    for chemsys in mp_entries:
        entries[chemsys] = []
        mp_entries_for_chemsys = mp_entries[chemsys]
        for e in mp_entries_for_chemsys:
            if mp_allowed_xcs and (e["parameters"]["run_type"] not in mp_allowed_xcs):
                continue
            entries[chemsys].append(e)
    entries = mp_entries.copy()
    relevant_chemsyses = list(mp_entries.keys())

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
    override_mp_with_my_calcs=False,
    data_dir=os.getcwd().replace("scripts", "data"),
    savename="mp_compatible_Efs.json",
    remake=False,
):
    fjson = os.path.join(data_dir, savename)
    if os.path.exists(fjson) and not remake:
        return read_json(fjson)

    out = {}
    for chemsys in merged_entries:
        mpfe = MPFormationEnergy(
            merged_entries[chemsys], override_mp_with_my_calcs=override_mp_with_my_calcs
        )
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
