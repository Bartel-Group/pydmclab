import numpy as np
import os, json

from pydmclab.core.comp import CompTools
from pydmclab.core.query import MPQuery
from pydmclab.utils.handy import write_json, read_json

this_dir, this_filename = os.path.split(__file__)
DATA_PATH = os.path.join(this_dir, "data")


def mus_at_0K():
    """
    These were run by Bartel in January 20223
    """
    fjson = os.path.join(DATA_PATH, "mus_from_dmc_no_corrections.json")
    if os.path.exists(fjson):
        return read_json(fjson)
    mus = {}
    d = read_json(os.path.join(DATA_PATH, "230122_dmc-mus.json"))
    for xc in d:
        if xc == "gga":
            functional = "pbe"
        elif xc == "metagga":
            functional = "r2scan"
        mus[functional] = d[xc]
    return write_json(mus, fjson)


def mus_at_T():
    """
    These come from Bartel 2018 Nat Comm
    """
    with open(os.path.join(DATA_PATH, "elemental_gibbs_energies_T.json")) as f:
        return json.load(f)


def mp2020_compatibility_dmus():
    """
    from MP2020Compatibility (https://github.com/materialsproject/pymatgen/blob/master/pymatgen/entries/MP2020Compatibility.yaml)
    """
    fjson = os.path.join(DATA_PATH, "mp2020_compatibility_dmus.json")
    if os.path.exists(fjson):
        return read_json(fjson)
    data = {
        "U": {
            "V": -1.7,
            "Cr": -1.999,
            "Mn": -1.668,
            "Fe": -2.256,
            "Co": -1.638,
            "Ni": -2.541,
            "W": -4.438,
            "Mo": -3.202,
        },
        "anions": {
            "O": -0.687,
            "S": -0.503,
            "F": -0.462,
            "Cl": -0.614,
            "Br": -0.534,
            "I": -0.379,
            "N": -0.361,
            "Se": -0.472,
            "Si": 0.071,
            "Sb": -0.192,
            "Te": -0.422,
            "H": -0.179,
        },
        "peroxide": {"O": -0.465},
        "superoxide": {"O": -0.161},
    }

    return write_json(data, fjson)


def mus_from_mp_no_corrections():
    """
    Last collected Dec 2022 from old API

    Returns:
        _type_: _description_
    """
    fjson = os.path.join(DATA_PATH, "mus_from_mp_no_corrections.json")
    if os.path.exists(fjson):
        return read_json(fjson)

    mus = mus_at_0K()

    mp_pbe_mus = mus["mp"]["pbe"]

    mpq = MPQuery(api_key="***REMOVED***")

    mp_mus = {}
    for el in mp_pbe_mus:
        print(el)
        my_mu = mp_pbe_mus[el]
        el += "1"
        query = mpq.get_data_for_comp(el, only_gs=True)

        mp_mu = query[el]["E_mp"]
        mp_mus[el[:-1]] = mp_mu

    return write_json(mp_mus, fjson)


def ssub():
    fjson = os.path.join(DATA_PATH, "ssub.json")
    if os.path.exists(fjson):
        return read_json(fjson)
    data = {}
    with open(os.path.join(DATA_PATH, "ssub.dat")) as f:
        for line in f:
            if "cmpd" in line:
                continue
            cmpd, H = line[:-1].split(" ")
            cmpd = CompTools(cmpd).clean
            if len(CompTools(cmpd).els) > 1:
                if cmpd not in data:
                    data[cmpd] = H
                else:
                    if H < data[cmpd]:
                        data[cmpd] = H
    return write_json(data, fjson)


def mus_from_bartel2019_npj():
    fjson = os.path.join(DATA_PATH, "mus_from_bartel2019_npj.json")
    if os.path.exists(fjson):
        return read_json(fjson)

    import pandas as pd

    df = pd.read_csv(os.path.join(DATA_PATH, "bartel2019_npj_reference-energies.csv"))

    els = df.el.values
    gga = df.PBE.values
    gga_fit = df["PBE+"].values
    metagga = df.SCAN.values
    metagga_fit = df["SCAN+"].values

    xcs = {
        "pbe": gga,
        "pbe_fit": gga_fit,
        "scan": metagga,
        "scan_fit": metagga_fit,
    }

    data = {xc: {el: xcs[xc][i] for i, el in enumerate(els)} for xc in xcs}
    return write_json(data, fjson)


def main():
    mus_from_bartel2019_npj()
    ssub()
    mus_from_mp_no_corrections()
    mus_at_0K()
    mus_at_T()


if __name__ == "__main__":
    main()
