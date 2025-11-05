import numpy as np
import os, json

from pydmclab.core.comp import CompTools
from pydmclab.core.query import MPQuery
from pydmclab.utils.handy import write_json, read_json

this_dir, this_filename = os.path.split(__file__)
DATA_PATH = os.path.join(this_dir, "data")


def mus_at_0K():
    """
    These were run by Szymanski in May 2024
    """
    fjson = os.path.join(DATA_PATH, "mus_from_dmc_no_corrections.json")
    if os.path.exists(fjson):
        return read_json(fjson)
    mus = {}
    mus = read_json(os.path.join(DATA_PATH, "240528_dmc-mus.json"))
    #    for xc in d:
    #        if xc == "gga":
    #            functional = "pbe"
    #        elif xc == "metagga":
    #            functional = "r2scan"
    #        mus[functional] = d[xc]
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


def omat2024_compatibility_dmus():
    """
    from OMat2024Compatibility (https://huggingface.co/datasets/facebook/OMAT24/tree/main/references)
    """
    fjson = os.path.join(DATA_PATH, "omat2024_compatibility_dmus.json")
    if os.path.exists(fjson):
        return read_json(fjson)
    data = {
        "U": {
            "V": -1.813,
            "Cr": -2.037,
            "Mn": -1.701,
            "Fe": -2.428,
            "Co": -2.151,
            "Ni": -2.58,
            "W": -4.445,
            "Mo": -2.972,
        },
        "anions": {
            "O": -0.657,
            "S": -0.487,
            "F": -0.436,
            "Cl": -0.6,
            "Br": -0.317,
            "I": -0.194,
            "N": -0.303,
            "Se": -0.474,
            "Si": 0.028,
            "Sb": -0.194,
            "Te": -0.418,
            "H": -0.173,
        },
        "peroxide": {"O": -0.433},
        "superoxide": {"O": -0.152},
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


def mus_from_omat():
    fjson = os.path.join(DATA_PATH, "mus_from_omat.json")
    if os.path.exists(fjson):
        return read_json(fjson)

    mus_omat = {
        "Ac": -4.07814994,
        "Ag": -2.717566636666667,
        "Al": -3.72747094,
        "Ar": -0.04024922,
        "As": -4.682053975,
        "Au": -3.22543615,
        "B": -6.7049832025,
        "Ba": -1.909760855,
        "Be": -3.7599265875,
        "Bi": -3.86884349,
        "Br": -1.44212653,
        "C": -9.199231915,
        "Ca": -1.93153898,
        "Cd": -0.74560783,
        "Ce": -5.921598105,
        "Cl": -1.8458800525,
        "Co": -7.03719571,
        "Cr": -9.515461885,
        "Cs": -0.8951742186206897,
        "Cu": -3.74513504,
        "Dy": -4.529309136666667,
        "Er": -4.495170506666667,
        "Eu": -10.29636708,
        "F": -1.80965965,
        "Fe": -8.27161473,
        "Ga": -2.9132388475,
        "Gd": -13.981797415,
        "Ge": -4.507745255,
        "H": -3.394378795,
        "He": -0.02165726,
        "Hf": -9.923830525,
        "Hg": -0.183946075,
        "Ho": -4.515334846666667,
        "I": -1.359441135,
        "In": -2.561892916666667,
        "Ir": -8.8565415,
        "K": -1.060647223,
        "Kr": -0.04490603,
        "La": -4.894495445,
        "Li": -1.9062828033333332,
        "Lu": -4.449524705,
        "Mg": -1.49982141,
        "Mn": -8.992716942413793,
        "Mo": -10.92071152,
        "N": -8.32644572,
        "Na": -1.2761079241379312,
        "Nb": -10.10334713,
        "Nd": -4.723421585,
        "Ne": -0.02128077,
        "Ni": -5.48422812,
        "Np": -12.73119343375,
        "O": -4.93913603,
        "Os": -11.223396495,
        "P": -5.409239460952381,
        "Pa": -9.41125813,
        "Pb": -3.57680932,
        "Pd": -5.21837678,
        "Pm": -4.6949224075,
        "Pr": -4.7434366675,
        "Pt": -6.08256756,
        "Pu": -13.9884109,
        "Rb": -0.94818047875,
        "Re": -12.39400358,
        "Rh": -7.27165212,
        "Ru": -9.236562915,
        "S": -4.1303699215625,
        "Sb": -4.146554885,
        "Sc": -6.245474435,
        "Se": -3.5072754221875,
        "Si": -5.42070396,
        "Sm": -4.6557698025,
        "Sn": -3.83029594,
        "Sr": -1.6272253,
        "Ta": -11.81537868,
        "Tb": -4.56844973,
        "Tc": -10.344412205,
        "Te": -3.1402593133333334,
        "Th": -7.44188808,
        "Ti": -7.806284746666667,
        "Tl": -2.2203468566666666,
        "Tm": -4.469228606666666,
        "U": -11.13607304,
        "V": -8.96201137,
        "W": -12.95412021,
        "Xe": -0.04027387,
        "Y": -6.42697752,
        "Yb": -4.4680880499999995,
        "Zn": -1.1150354,
        "Zr": -8.518544045,
    }

    return write_json(mus_omat, fjson)


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


def gas_thermo_data():
    fjson = os.path.join(DATA_PATH, "gas_thermo_data_nist.json")
    if os.path.exists(fjson):
        return read_json(fjson)


def main():
    # mus_from_bartel2019_npj()
    # ssub()
    # mus_from_mp_no_corrections()
    mus_at_0K()
    # mus_at_T()


if __name__ == "__main__":
    main()
