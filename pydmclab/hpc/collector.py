import os
import sys

from pydmclab.hpc.analyze import AnalyzeVASP
from pydmclab.core.struc import StrucTools
from pydmclab.utils.handy import read_json, write_json


class Collector(object):
    """
    Executed on the cluster after calculations run to collect results
    """

    def __init__(self, calc_dir, path_to_configs):
        """
        Args:
            calc_dir (str):
                path to the calculation directory

            path_to_configs (str):
                path to the json file that contains the configuration for the analysis

        Returns:
            configs (dict)
                dictionary of configs
        """
        self.calc_dir = calc_dir
        self.path_to_configs = path_to_configs
        configs = read_json(path_to_configs)
        self.configs = configs

    @property
    def key(self):
        """
        Returns:
            key to be stored in the results dictionary
                formula_indicator--struc_indicator--mag--xc-calc
        """
        calc_dir = self.calc_dir
        key = "--".join(calc_dir.split("/")[-4:])
        return key

    @property
    def results(self):
        """
        Returns:
            a dictionary of results for that calculation directory
                format varies based on self.configs
                see AnalyzeVASP.summary() for more info
        """
        calc_dir = self.calc_dir
        key = self.key
        xc_calc = key.split("--")[-1]
        xc, calc = xc_calc.split("-")

        configs = self.configs.copy()

        if calc != "relax":
            configs["include_trajectory"] = False
        if calc not in ["lobster", "bs"]:
            configs["include_tcohp"] = False
            configs["include_pcohp"] = False
            configs["include_tcoop"] = False
            configs["include_pcoop"] = False
            configs["include_tcobi"] = False
            configs["include_pcobi"] = False
            configs["include_tdos"] = False
            configs["include_pdos"] = False
        if calc != "static":
            configs["include_mag"] = False
            configs["include_entry"] = False
            configs["include_structure"] = False

        verbose = configs["verbose"]
        include_meta = configs["include_metadata"]
        include_calc_setup = configs["include_calc_setup"]
        include_structure = configs["include_structure"]
        include_trajectory = configs["include_trajectory"]
        include_mag = configs["include_mag"]
        include_tdos = configs["include_tdos"]
        include_pdos = configs["include_pdos"]
        include_tcohp = configs["include_tcohp"]
        include_pcohp = configs["include_pcohp"]
        include_tcoop = configs["include_tcoop"]
        include_pcoop = configs["include_pcoop"]
        include_tcobi = configs["include_tcobi"]
        include_pcobi = configs["include_pcobi"]
        include_entry = configs["include_entry"]
        check_relax = configs["check_relax_energy"]
        create_cif = configs["create_cif"]

        if verbose:
            print("analyzing %s" % calc_dir)
        analyzer = AnalyzeVASP(calc_dir)

        # collect the data we asked for
        summary = analyzer.summary(
            include_meta=include_meta,
            include_calc_setup=include_calc_setup,
            include_structure=include_structure,
            include_trajectory=include_trajectory,
            include_mag=include_mag,
            include_tdos=include_tdos,
            include_pdos=include_pdos,
            include_tcohp=include_tcohp,
            include_pcohp=include_pcohp,
            include_tcoop=include_tcoop,
            include_pcoop=include_pcoop,
            include_tcobi=include_tcobi,
            include_pcobi=include_pcobi,
            include_entry=include_entry,
        )

        # store the relax energy if we asked to
        if check_relax:
            relax_energy = AnalyzeVASP(calc_dir.replace(calc, "relax")).E_per_at
            summary["results"]["E_relax"] = relax_energy

        if create_cif and summary["results"]["convergence"] and include_structure:
            if summary["structure"]:
                s = StrucTools(summary["structure"]).structure
                s.to(fmt="cif", filename=os.path.join(calc_dir, key + ".cif"))
            else:
                print("no structure, cant make cif")
        return summary

    @property
    def to_dict(self):
        """
        Returns:
            writes the results dictionary to a file.
            also includes logic on when to repeat the results generation process.
                repeat if:
                    asked to remake_results (configs)
                    results.json does not exist
                    OUTCAR has been touched since results.json was last written
                otherwise:
                    just read the results.json file
        """
        savename = "results.json"
        fjson = os.path.join(self.calc_dir, savename)
        if os.path.exists(fjson) and not self.configs["remake_results"]:
            results = read_json(fjson)
            foutcar = os.path.join(self.calc_dir, "OUTCAR")
            if results[self.key]["results"]["convergence"]:
                if not os.path.exists(foutcar):
                    return results
                results_gen_time = os.path.getmtime(fjson)
                outcar_gen_time = os.path.getmtime(foutcar)
                if outcar_gen_time < results_gen_time:
                    return results
        results = self.results
        write_json(results, fjson)
        return read_json(fjson)


def debug():
    """
    Execute this to avoid try/except to see error messages
    """
    calc_dir, path_to_configs = sys.argv[1], sys.argv[2]
    collector = Collector(calc_dir=calc_dir, path_to_configs=path_to_configs)
    results = collector.to_dict
    return results


def main():
    # get info that pertains to the present calculation
    calc_dir, path_to_configs = sys.argv[1], sys.argv[2]

    # initialize the Collector for this claculation
    collector = Collector(calc_dir=calc_dir, path_to_configs=path_to_configs)

    # try to collect results and read or write results.json
    try:
        results = collector.to_dict

    # if this fails for some reason, populate collection.o with python error message that caused failure
    except Exception as e:
        fcollection = os.path.join(calc_dir, "collection.o")
        with open(fcollection, "w", encoding="utf-8") as f:
            f.write("something went wrong\n\n\n")
            f.write(str(e))
        results = None
    return results


if __name__ == "__main__":
    results = main()
