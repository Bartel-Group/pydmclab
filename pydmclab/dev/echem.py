import os
import numpy as np

from pydmclab.core.comp import CompTools
from pydmclab.core.energies import ChemPots
from pydmclab.utils.handy import read_json, write_json

from pydmclab.core.hulls import MixingHull


class VoltageCurve(object):
    def __init__(
        self,
        input_energies,
        mu_Li="dmc-r2scan",
        energy_key="E_per_at",
        intercalating_ion="Li",
        charged_composition="FeP2S6",
        discharged_composition="Li2FeP2S6",
        basis_ion=("Fe", 1),
    ):
        self.input_energies = input_energies
        if isinstance(mu_Li, str):
            mu_Li = ChemPots(
                functional=mu_Li.split("-")[1], standard=mu_Li.split("-")[0]
            ).chempots["Li"]
        self.mu_Li = mu_Li

        self.energy_key = energy_key
        self.intercalating_ion = intercalating_ion
        self.charged_composition = CompTools(charged_composition).clean
        self.discharged_composition = CompTools(discharged_composition).clean
        self.basis_ion = basis_ion

    @property
    def z(self):
        ion = self.intercalating_ion
        if ion in ["Li", "Na", "K"]:
            return 1
        elif ion in ["Mg", "Ca", "Zn"]:
            return 2
        elif ion in ["Al"]:
            return 3
        else:
            raise ValueError("ion not recognized")

    @property
    def mixing_results(self):
        hull = MixingHull(
            input_energies=self.input_energies,
            energy_key=self.energy_key,
            varying_element=self.intercalating_ion,
            end_members=[self.charged_composition, self.discharged_composition],
            shared_element_basis=self.basis_ion[0],
        )

        d = hull.results
        return d

    @property
    def input_to_voltage_curve(self):
        d = self.mixing_results
        out = {}
        for original_formula in d:
            stability = d[original_formula]["stability"]
            E_per_at = d[original_formula]["E"]
            n_basis_el = CompTools(original_formula).stoich(self.basis_ion[0])
            if n_basis_el == 0:
                continue
            factor = self.basis_ion[1] / n_basis_el
            E_per_basis_fu = E_per_at * CompTools(original_formula).n_atoms * factor

            n_ion = CompTools(original_formula).stoich(self.intercalating_ion)
            n_basis_ion = n_ion * factor

            out[original_formula] = {
                "on_mixing_hull": stability,
                "E_per_at": E_per_at,
                "factor": factor,
                "n_intercalating": n_basis_ion,
                "E": E_per_basis_fu,
            }
        return out

    @property
    def breaks_in_voltage_curve(self):
        d = self.input_to_voltage_curve
        stable_ns = sorted(
            [d[k]["n_intercalating"] for k in d if d[k]["on_mixing_hull"]]
        )
        return stable_ns

    @property
    def plateaus(self):
        breaks = self.breaks_in_voltage_curve
        p = []
        for i, b in enumerate(breaks):
            if i == 0:
                continue
            p.append((breaks[i - 1], b))
        return p

    @property
    def voltages(self):
        z = self.z
        mu_Li = self.mu_Li
        d = self.input_to_voltage_curve
        plateaus = self.plateaus
        voltages = []
        for p in plateaus:
            charged_n, discharged_n = p
            charged_comp = [c for c in d if d[c]["n_intercalating"] == charged_n][0]
            discarged_comp = [c for c in d if d[c]["n_intercalating"] == discharged_n][
                0
            ]
            charged_E = d[charged_comp]["E"]
            discharged_E = d[discarged_comp]["E"]
            delta_n = discharged_n - charged_n

            V = (charged_E + delta_n * mu_Li - discharged_E) / z / delta_n
            voltages.append(V)
        out = {}
        for i, p in enumerate(plateaus):
            out["_".join([str(v) for v in list(p)])] = voltages[i]
        return out


def make_mixing_json():
    fjson = os.path.join("/users/cbartel", "Downloads", "Li2MP2S6_gga_gs_E_per_at.json")
    d = read_json(fjson)

    out = {}
    for M in ["Mn", "Fe", "Co", "Ni"]:
        hull = MixingHull(
            d,
            energy_key="E_per_at",
            varying_element="Li",
            end_members=["Li2%sP2S6" % M, "%sP2S6" % M],
            shared_element_basis=M,
        )
        results = hull.results
        for c in results:
            out[c] = results[c]

    for c in out:
        print("\n")
        print(c)
        print("Emix : %.5f" % out[c]["E_mix"])
        print("x : %.2f" % out[c]["x"])
        print("stability : %s" % out[c]["stability"])

    write_json(out, fjson.replace(".json", "_cjb.json"))
    return out


def make_voltages():
    fjson = os.path.join("/users/cbartel", "Downloads", "Li2MP2S6_gga_gs_E_per_at.json")
    d = read_json(fjson)
    out = {}
    for M in ["Mn", "Fe", "Co", "Ni"]:
        vc = VoltageCurve(
            input_energies=d,
            energy_key="E_per_at",
            intercalating_ion="Li",
            charged_composition="%sP2S6" % M,
            discharged_composition="Li2%sP2S6" % M,
            basis_ion=(M, 1),
        )
        out[M] = vc.voltages

    for M in out:
        print("\n%s" % M)
        print(out[M])
    write_json(out, fjson.replace(".json", "_voltages_cjb.json"))
    return out


def chrisc_main():
    fjson = os.path.join("/users/cbartel", "Downloads", "Li2MP2S6_gga_gs_E_per_at.json")
    energies = read_json(fjson)
    mixing = make_mixing_json()
    voltages = make_voltages()
    return energies, mixing, voltages


def mm_stuff():
    print(os.getcwd())

    fjson = "../data/input_energies.json"
    energies = read_json(fjson)

    energies["Ba1S3Zr1"] = {"E_per_at": -100}
    out = {}
    for M in ["Nb", "Ta"]:
        mixing = MixingHull(
            energies,
            energy_key="E_per_at",
            varying_element=M,
            end_members=["BaZrS3", "Ba%sS3" % M],
            shared_element_basis="Ba",
        )

        print(mixing.relevant_compounds)
        out[M] = mixing.results

    return energies, out


def main():
    fjson = os.path.join("/Users/cbartel/Downloads/gs_ABX3-subs_b.json")
    gs = read_json(fjson)
    input_energies = gs["dmc"]["gga"]
    input_energies["Ba1S3Zr1"] = {"E": -6.391578691}

    input_energies = {
        c: {"E": input_energies[c]["E"]}
        for c in input_energies
        if input_energies[c]["E"]
    }
    out = {}
    for M in ["Nb", "Ta"]:
        mixing = MixingHull(
            input_energies,
            energy_key="E",
            varying_element=M,
            end_members=["BaZrS3", "Ba%sS3" % M],
            shared_element_basis="Ba",
        )

        print(mixing.relevant_compounds)
        out[M] = mixing.results

    results = out["Nb"]
    x = 0.0625
    compound_at_x = [c for c in results if results[c]["x"] == x][0]
    E_mix = results[compound_at_x]["E_mix"]
    print(E_mix)
    return out


if __name__ == "__main__":
    out = main()
