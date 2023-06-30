import os

from pymatgen.core.structure import Structure

from pydmclab.utils.handy import read_yaml, write_yaml, is_calc_valid
from pydmclab.core.mag import MagTools
from pydmclab.data.configs import load_launch_configs
from pydmclab.core.struc import StrucTools

HERE = os.path.dirname(os.path.abspath(__file__))


class LaunchTools(object):
    """
    This is a class to figure out:
        what launch_dirs need to be created
            i.e., which directories will house submission scripts
        what calculation chains need to be run in each launch_dir

        a launch_dir pertains to a particular structure with a particular magnetic configuration calculated with a particular method

    The output is going to be:
        {launch_dir (str) : {'xcs' : [list of final xcs for each chain (str)],
                             'magmom' : [list of magmoms for the structure in that launch_dir (list)],}}
    """

    def __init__(
        self,
        calcs_dir,
        structure,
        top_level,
        unique_ID,
        magmoms=None,
        user_configs={},
        refresh_configs=True,
        launch_configs_yaml=os.path.join(os.getcwd(), "_launch_configs.yaml"),
    ):
        """
        Args:

            calcs_dir (os.path):
                top directory where all calculations to be launched will be stored

                usually if I'm writing a "launch" script to configure and run a bunch of calcs from  a directory
                    os.getcwd() = */scripts:
                        then calcs_dir will be os.getcwd().replace('scripts', 'calcs')
                        unless you want to run calcs on /scratch then you might replace path_to_scripts with path_to_similar_location_on_scratch

                    I should also probably have a directory to store data called scripts_dir.replace('scripts', 'data')

                    these are best practices but not strictly enforced in the code anywhere

            structure (Structure):
                pymatgen structure object
                    usually I want to run a series of calculations for some input structure
                        this is the input structure

            top_level (str):
                top level directory within calcs_dir

                could be whatever you want, but usually this will be a chemical formula
                    could also be a placeholder for a formula (e.g., Li1S2Ti1_bigsupercell)


            unique_ID (str):
                level below top_level (should uniquely define this particular structure for the top_level formula)
                    could be a material ID in materials project (for standard geometry relaxations, this makes sense)
                    could be x in the LiCoO2 example I described previously
                    it's really up to you, but it must be unique within the top_level directory

            magmoms (dict):
                if you are running AFM calculations
                    {index of configuration index (int) : magmom (list)} generated using MagTools
                        best practice is to save this as a json in data_dir so you only call MagTools once

                if you are not running AFM calculations (you don't need a specific MAGMOM)
                    can be None or {}

            user_configs (dict):
                any setting you want to pass that's not default in pydmclab.data.data._launch_configs.yaml
                    these configs pertain to how you want to set up the launch of many calculations
                    e.g., how many AFM configs to launch

            refresh_configs (bool)
                if True, will copy pydmclab baseline configs to your local directory
                this is useful if you've made changes to the configs files in the directory you're working in and want to start over

            launch_configs_yaml (os.pathLike)
                path to yaml file containing launch configs
                    there's usually no reason to change this
                    this holds some default configs for LaunchTools
                    can always be changed with user_configs

        Returns:
            configs (dict):
                dictionary of all configs and arguments to LaunchTools
        """

        # make our calcs_dir if it doesn't exist (this will hold all the launch_dirs)
        if not os.path.exists(calcs_dir):
            os.mkdir(calcs_dir)

        # make our local launch_configs file if it doesn't exist
        if not os.path.exists(launch_configs_yaml) or refresh_configs:
            _launch_configs = load_launch_configs()
            write_yaml(_launch_configs, launch_configs_yaml)

        # initialize our baseline launch_configs
        _launch_configs = read_yaml(launch_configs_yaml)

        # update our baseline launch_configs with user_configs
        configs = {**_launch_configs, **user_configs}

        # make structure a dict() for easier handling
        if not isinstance(structure, dict):
            structure = structure.as_dict()

        # check to make sure we have magmoms if we're running AFM calcs
        if configs["n_afm_configs"] > 0:
            if MagTools(structure).could_be_afm:
                if not magmoms:
                    raise ValueError(
                        "You are running afm calculations but provided no magmoms, generate these first, then pass to LaunchTools"
                    )

        # add the required arguments to our configs file
        configs["top_level"] = top_level
        configs["unique_ID"] = unique_ID
        configs["calcs_dir"] = calcs_dir

        # store our magmoms and structure
        self.magmoms = magmoms
        self.structure = structure

        # make a copy of our configs to prevent unwanted changes
        self.configs = configs.copy()

    @property
    def valid_mags(self):
        """
        Returns:
            list of magnetic configuration names that make sense to run based on the inputs

        e.g.,
            if we have a nonmagnetic system, this should be ['nm']
            if we set n_afm_configs = 100, but our magmoms only has 3 configs, then this will just hold ['fm', 'afm_0', 'afm_1', 'afm_2']

        Note:
            configs['override_mag'] will force that we use configs['override_mag'] as our mag

        """
        # copy our configs
        configs = self.configs.copy()

        # return override_mag if we set it
        if configs["override_mag"]:
            return configs["override_mag"]

        structure = self.structure

        # if we're not magnetic, return nm
        if not MagTools(structure).could_be_magnetic:
            return ["nm"]

        # if we can't be AFM or we didn't ask for AFM, but we are magnetic, return fm
        if not MagTools(structure).could_be_afm or not configs["n_afm_configs"]:
            return ["fm"]

        # figure out the max AFM index we can run based on what we asked for

        # shift for 0 index
        max_desired_afm_idx = configs["n_afm_configs"] - 1

        magmoms = self.magmoms

        # figure out what configs we have magmoms for
        configs_in_magmoms = list(magmoms.keys())
        configs_in_magmoms = sorted([int(i) for i in configs_in_magmoms])
        max_available_afm_idx = max(configs_in_magmoms)

        max_afm_idx = min(max_desired_afm_idx, max_available_afm_idx)

        # create placeholders afm_0, afm_1, ... to define each AFM configuration
        afm_indices = ["afm_%s" % str(i) for i in range(max_afm_idx + 1)]

        # return FM + AFM configs for AFM calcs
        return ["fm"] + afm_indices

    def launch_dirs(self, make_dirs=True):
        """
        Args:
            make_dirs (bool)
                if True, make the launch_dir and populate each with the relevant POSCAR

        Returns:
            a dictionary of:
                {launch_dir (str) : {'xcs': [list of final_xcs to submit w/ SubmitTools],
                                     'magmom' : [list of magmoms for the structure in launch_dir to pass to SubmitTools]}}

        Returns the minimal list of directories that will house submission files (each of which launch a chain of calcs)
            note a chain of calcs must have the same structure and magnetic information, otherwise, there's no reason to chain them
                so the launch_dir defines: structure, standard, magmom

        These launch_dirs have a very prescribed structure:
            calcs_dir / top_level / unique_ID / standard / mag

            e.g.,
                ../calcs/Nd2O7Ru2/mp-19930/dmc/fm
                ../calcs/2/3/dmc/afm_4
                    (if (2) was a unique compositional indicator and (3) was a unique structural indicator)
        """
        structure = self.structure
        magmoms = self.magmoms

        # make a copy of our configs to prevent unwanted changes
        configs = self.configs.copy()

        # the list of mags we can run
        mags = self.valid_mags

        # final_xcs we want to run for each standard
        to_launch = configs["to_launch"]

        # level0 houses all our launch_dirs
        level0 = configs["calcs_dir"]

        # level1 describes the composition
        level1 = configs["top_level"]

        # level2 describes the structure
        level2 = configs["unique_ID"]

        launch_dirs = {}
        for standard in to_launch:
            # for each standard we asked for, use that as level3
            level3 = standard

            # we asked for certain xcs at each standard, hold them here
            xcs = to_launch[standard]

            for mag in mags:
                # for each mag we can run, use that as level4
                level4 = mag

                # start w/ magmom = None, then check for updating if we have AFM configs
                magmom = None
                if "afm" in mag:
                    # grab the magmom if our calc is AFM
                    idx = mag.split("_")[1]
                    if str(idx) in magmoms:
                        magmom = magmoms[str(idx)]
                    elif int(idx) in magmoms:
                        magmom = magmoms[int(idx)]

                # our launch_dir is now defined
                launch_dir = os.path.join(level0, level1, level2, level3, level4)

                # save the final_xcs we want to submit in this launch_dir
                # SubmitTools will make 1 submission script for each final_xc
                # save the magmom as well. VASPSetUp will need that to set the INCARs for all calcs in this launch_dir
                launch_dirs[launch_dir] = {"xcs": xcs, "magmom": magmom}

                # if make_dirs, make the launch_dir and put a POSCAR in there
                if make_dirs:
                    # make the launch_dir if it doesn't exist
                    if not os.path.exists(launch_dir):
                        os.makedirs(launch_dir)

                    # make the POSCAR if it doesn't exist
                    fposcar = os.path.join(launch_dir, "POSCAR")
                    if not os.path.exists(fposcar):
                        struc = Structure.from_dict(structure)

                        # perturb if requested
                        if configs["perturb_launch_poscar"]:
                            initial_structure = struc.copy()
                            if isinstance(configs["perturb_launch_poscar"], bool):
                                perturbation = 0.05
                            else:
                                perturbation = configs["perturb_launch_poscar"]
                            perturbed_structure = StrucTools(initial_structure).perturb(
                                perturbation
                            )

                            # write it
                            perturbed_structure.to(fmt="poscar", filename=fposcar)
                        else:
                            # write it (w/o perturbing)
                            struc.to(fmt="poscar", filename=fposcar)

        # return the dictionary (to be passed to SubmitTools)
        return launch_dirs
