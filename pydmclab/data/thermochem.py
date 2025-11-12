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
        'Ac': -4.07814994,
        'Ag': -2.717566636666667,
        'Al': -3.72747094,
        'Ar': -0.04024922,
        'As': -4.682053975,
        'Au': -3.22543615,
        'B': -6.7049832025,
        'Ba': -1.909760855,
        'Be': -3.7599265875,
        'Bi': -3.86884349,
        'Br': -1.44212653,
        'C': -9.199231915,
        'Ca': -1.93153898,
        'Cd': -0.74560783,
        'Ce': -5.921598105,
        'Cl': -1.8458800525,
        'Co': -7.03719571,
        'Cr': -9.515461885,
        'Cs': -0.8951742186206897,
        'Cu': -3.74513504,
        'Dy': -4.529309136666667,
        'Er': -4.495170506666667,
        'Eu': -10.29636708,
        'F': -1.80965965,
        'Fe': -8.27161473,
        'Ga': -2.9132388475,
        'Gd': -13.981797415,
        'Ge': -4.507745255,
        'H': -3.394378795,
        'He': -0.02165726,
        'Hf': -9.923830525,
        'Hg': -0.183946075,
        'Ho': -4.515334846666667,
        'I': -1.359441135,
        'In': -2.561892916666667,
        'Ir': -8.8565415,
        'K': -1.060647223,
        'Kr': -0.04490603,
        'La': -4.894495445,
        'Li': -1.9062828033333332,
        'Lu': -4.449524705,
        'Mg': -1.49982141,
        'Mn': -8.992716942413793,
        'Mo': -10.92071152,
        'N': -8.32644572,
        'Na': -1.2761079241379312,
        'Nb': -10.10334713,
        'Nd': -4.723421585,
        'Ne': -0.02128077,
        'Ni': -5.48422812,
        'Np': -12.73119343375,
        'O': -4.93913603,
        'Os': -11.223396495,
        'P': -5.409239460952381,
        'Pa': -9.41125813,
        'Pb': -3.57680932,
        'Pd': -5.21837678,
        'Pm': -4.6949224075,
        'Pr': -4.7434366675,
        'Pt': -6.08256756,
        'Pu': -13.9884109,
        'Rb': -0.94818047875,
        'Re': -12.39400358,
        'Rh': -7.27165212,
        'Ru': -9.236562915,
        'S': -4.1303699215625,
        'Sb': -4.146554885,
        'Sc': -6.245474435,
        'Se': -3.5072754221875,
        'Si': -5.42070396,
        'Sm': -4.6557698025,
        'Sn': -3.83029594,
        'Sr': -1.6272253,
        'Ta': -11.81537868,
        'Tb': -4.56844973,
        'Tc': -10.344412205,
        'Te': -3.1402593133333334,
        'Th': -7.44188808,
        'Ti': -7.806284746666667,
        'Tl': -2.2203468566666666,
        'Tm': -4.469228606666666,
        'U': -11.13607304,
        'V': -8.96201137,
        'W': -12.95412021,
        'Xe': -0.04027387,
        'Y': -6.42697752,
        'Yb': -4.4680880499999995,
        'Zn': -1.1150354,
        'Zr': -8.518544045,
    }

    return write_json(mus_omat, fjson)

def mus_from_matpes_pbe():
    fjson = os.path.join(DATA_PATH, "mus_from_matpes_pbe.json")
    if os.path.exists(fjson):
        return read_json(fjson)

    mus_matpes_pbe = {
        'Ac': -4.0587065525,
        'Ag': -2.712658993333333,
        'Al': -3.74216679,
        'Ar': -0.042668825,
        'As': -4.66724489,
        'Au': -3.21117785,
        'B': -6.7062477,
        'Ba': -3.24176334,
        'Be': -3.7660998475,
        'Bi': -3.87640265,
        'Br': -1.6421302925,
        'C': -9.23232136,
        'Ca': -1.92635007,
        'Cd': -0.743553695,
        'Ce': -5.93217632,
        'Cl': -1.8313096175,
        'Co': -7.031060445,
        'Cr': -9.46909636,
        'Cs': -0.85719267,
        'Cu': -3.73914613,
        'Dy': -7.8132298425,
        'Er': -5.162969693333333,
        'Eu': -11.457918285,
        'F': -1.8303724125,
        'Fe': -8.260235505,
        'Ga': -2.892797815,
        'Gd': -11.70878079,
        'Ge': -4.50614858,
        'H': -3.39528257,
        'He': -0.01132933,
        'Hf': -9.922628655,
        'Hg': -0.18664053,
        'Ho': -6.5099600933333335,
        'I': -1.52328802,
        'In': -2.55932268,
        'Ir': -8.83881215,
        'K': -1.05008959,
        'Kr': -0.03904487,
        'La': -4.8853782375,
        'Li': -1.90638152,
        'Lu': -4.45560569,
        'Mg': -1.513484145,
        'Mn': -8.94467369,
        'Mo': -10.92398723,
        'N': -8.33976345,
        'Na': -1.31872484,
        'Nb': -10.09922062,
        'Nd': -6.4526366625,
        'Ne': -0.01870333,
        'Ni': -5.48294956,
        'Np': -12.72067328,
        'O': -4.9518358425,
        'Os': -11.226704565,
        'P': -5.403957041428571,
        'Pa': -9.40683889,
        'Pb': -3.55599251,
        'Pd': -5.21918858,
        'Pm': -7.66163035,
        'Pr': -5.464460628333334,
        'Pt': -6.08371852,
        'Pu': -13.996956520625,
        'Rb': -0.94353145,
        'Re': -12.3930327725,
        'Rh': -7.25600058,
        'Ru': -9.227753265,
        'S': -4.12370117375,
        'Sb': -4.138483165,
        'Sc': -6.24440527,
        'Se': -3.50088016671875,
        'Si': -5.4236135,
        'Sm': -9.163031595,
        'Sn': -3.83589641,
        'Sr': -1.635995145,
        'Ta': -11.812164331666667,
        'Tb': -9.52411453,
        'Tc': -10.345558845,
        'Te': -3.1390379,
        'Th': -7.44360899,
        'Ti': -7.808735856666666,
        'Tl': -2.223056975,
        'Tm': -4.00267056,
        'U': -11.133507695,
        'V': -8.96179728,
        'W': -12.96540428,
        'Xe': -0.03791398,
        'Y': -6.425541805,
        'Yb': -3.54873555,
        'Zn': -1.104468055,
        'Zr': -8.52111572,
    }

    return write_json(mus_matpes_pbe, fjson)

def mus_from_matpes_r2scan():
    fjson = os.path.join(DATA_PATH, "mus_from_matpes_r2scan.json")
    if os.path.exists(fjson):
        return read_json(fjson)

    mus_matpes_r2scan = {
        'Ac': -68.6353431675,
        'Ag': -21.3549977225,
        'Al': -6.72858831,
        'Ar': -4.8569829,
        'As': -14.7073791075,
        'Au': -50.582982755,
        'B': -7.346208425,
        'Ba': -27.8354637,
        'Be': -4.33768086,
        'Bi': -58.502132445,
        'Br': -13.2379773125,
        'C': -9.946179045,
        'Ca': -7.03299513,
        'Cd': -20.10496472,
        'Ce': -30.83927465,
        'Cl': -6.1521282125,
        'Co': -13.23617975,
        'Cr': -15.0716839,
        'Cs': -25.02547969,
        'Cu': -10.84677467,
        'Dy': -37.7248231525,
        'Er': -36.027139995,
        'Eu': -39.7073089,
        'F': -3.1201512325,
        'Fe': -14.412679855,
        'Ga': -11.4427978225,
        'Gd': -41.100368075,
        'Ge': -13.8716078,
        'H': -3.45518371125,
        'He': -0.31474001,
        'Hf': -45.15188157,
        'Hg': -49.3665303,
        'Ho': -37.284987275,
        'I': -24.535261165,
        'In': -22.57821872,
        'Ir': -52.35942139,
        'K': -6.01628447,
        'Kr': -12.57787177,
        'La': -29.741186105,
        'Li': -2.38444874,
        'Lu': -38.07978784,
        'Mg': -4.168801315,
        'Mn': -14.68573582551724,
        'Mo': -26.07072021,
        'N': -9.1278508575,
        'Na': -3.56345051,
        'Nb': -24.73190072,
        'Nd': -33.041469535,
        'Ne': -1.91212637,
        'Ni': -11.98035986,
        'Np': -83.0244023575,
        'O': -5.95767759,
        'Os': -52.98464406,
        'P': -9.001581174880952,
        'Pa': -76.85801947,
        'Pb': -56.27614582,
        'Pd': -22.95958473,
        'Pm': -34.84920939,
        'Pr': -31.260144136666668,
        'Pt': -51.49517425,
        'Pu': -86.357756555,
        'Rb': -13.92726379,
        'Re': -52.410308085,
        'Rh': -24.13777898,
        'Ru': -25.422695435,
        'S': -8.0341289034375,
        'Sb': -25.69891105,
        'Sc': -11.3974349,
        'Se': -14.31169046359375,
        'Si': -8.774763405,
        'Sm': -36.560696415,
        'Sn': -24.7454617,
        'Sr': -15.143240576666665,
        'Ta': -48.59174402466667,
        'Tb': -38.91241293,
        'Tc': -25.9642671,
        'Te': -25.416242023333336,
        'Th': -73.47523629,
        'Ti': -12.93772981,
        'Tl': -53.179147585,
        'Tm': -35.378087365,
        'U': -79.94230473,
        'V': -14.08804361,
        'W': -51.36672133,
        'Xe': -23.861382795,
        'Y': -20.3114169,
        'Yb': -35.50338827,
        'Zn': -8.91051781,
        'Zr': -22.74827471,
    }

    return write_json(mus_matpes_r2scan, fjson)

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
