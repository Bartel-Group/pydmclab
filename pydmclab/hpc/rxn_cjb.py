from pydmclab.utils.handy import read_json, write_json
from pydmclab.core.query import MPQuery
from pydmclab.core.energies import (
    ChemPots,
    FormationEnthalpy,
    FormationEnergy,
    ReactionEnergy,
)
from pydmclab.core.comp import CompTools
import os

data_dir = os.path.join("..", "data", "rxns")
if not os.path.exists(data_dir):
    os.mkdir(data_dir)


def get_compounds_from_mp(savename="Ef_mp.json", remake=False):
    fjson = os.path.join(data_dir, savename)
    if os.path.exists(fjson) and remake:
        return read_json(fjson)

    metals = ["Mn", "Fe", "Co", "Ni"]
    chlorides = ["%sCl2" % m for m in metals]
    iodides = ["%sI2" % m for m in metals]
    other = ["Li2S", "LiCl", "P2S5"]

    compounds = chlorides + iodides + other
    print(compounds)
    mpq = MPQuery(api_key="***REMOVED***")
    query = mpq.get_data_for_comp(
        comp=compounds, include_structure=True, only_gs=True, max_Ehull=None
    )

    out = {}
    for ID in query:
        out[query[ID]["cmpd"]] = {
            "Ef": query[ID]["Ef_mp"],
            "structure": query[ID]["structure"],
            "E": query[ID]["E_mp"],
            "mpid": ID,
        }

    write_json(out, fjson)
    return read_json(fjson)


def get_compounds_from_chrisc(Ef_mp, savename="Ef_chrisc.json", remake=False):
    fjson = os.path.join(data_dir, savename)
    if not remake and os.path.exists(fjson):
        return read_json(fjson)

    Efs = Ef_mp.copy()

    metals = ["Mn", "Fe", "Co", "Ni"]
    compounds = [CompTools("Li2%sP2S6" % m).clean for m in metals]

    ####### REWRITE THIS TO RETRIEVE CHRISC DATA #######

    for c in compounds:
        Efs[c] = {"structure": None, "E": None, "mpid": None}

    ####### REWRITE THIS TO RETRIEVE CHRISC DATA #######

    mus = ChemPots(functional="pbe", standard="dmc").chempots

    for c in compounds:
        fe = FormationEnthalpy(formula=c, E_DFT=Efs[c]["E"], chempots=mus)
        Ef = fe.Ef
        Efs[c]["Ef"] = Ef

    write_json(Efs, fjson)
    return read_json(fjson)


def get_dGf(Efs, savename="dGf.json", remake=False):
    fjson = os.path.join(data_dir, savename)
    if not remake and os.path.exists(fjson):
        return read_json(fjson)

    dGfs = Efs.copy()
    Ts = [0, 300, 600, 900, 1200]
    for c in dGfs:
        structure = dGfs[c]["structure"]
        Ef = dGfs[c]["Ef"]
        dGfs[c]["dGfs"] = {}
        for T in Ts:
            mus = ChemPots(temperature=T).chempots
            fe = FormationEnergy(
                formula=c,
                structure=structure,
                Ef=Ef,
                chempots=mus,
                include_Svib=True,
                include_Sconfig=False,
            )
            dGf = fe.dGf(T)
            dGfs[c]["dGfs"][T] = dGf

    elements = ["Li", "P", "S", "Cl", "I", "Mn", "Fe", "Co", "Ni"]
    for e in elements:
        dGfs[e] = {"dGfs": {T: 0 for T in Ts}}
        for k in ["Ef", "E", "mpid"]:
            dGfs[e][k] = None

    write_json(dGfs, fjson)
    return read_json(fjson)


def get_rxn_energies(dGfs, savename="rxn_energies.json", remake=False):
    fjson = os.path.join(data_dir, savename)
    if not remake and os.path.exists(fjson):
        return read_json(fjson)
    reaction_names = ["traditional", "chloride", "iodide"]
    # traditional = Li2S + M + P2S5 --> Li2MP2S6
    # chloride = 2 Li2S + MCl2 + P2S5 --> Li2MP2S6 + 2LiCl + S
    # iodide = Li2S + MI2 + P2S5 --> Li2MP2S6 + I2

    metals = ["Mn", "Fe", "Co", "Ni"]

    rxns = {}
    for M in metals:
        rxns[M] = {}
        for rxn in reaction_names:
            rxns[M][rxn] = {"label": rxn}
            reactants = ["Li2S", "P2S5"]
            products = ["Li2%sP2S6" % M]
            if rxn == "chloride":
                reactants += ["%sCl2" % M]
                products += ["LiCl", "S"]
            elif rxn == "iodide":
                reactants += ["%sI2" % M]
                products += ["I2"]
            elif rxn == "traditional":
                reactants += [M]
            rxns[M][rxn]["reactants"] = reactants
            rxns[M][rxn]["products"] = products
            for T in dGfs[M]["dGfs"]:
                input_energies = {c: {"E": dGfs[c]["dGfs"][T]} for c in dGfs}
                re = ReactionEnergy(
                    input_energies=input_energies,
                    reactants=reactants,
                    products=products,
                    energy_key="E",
                    norm="rxn",
                )
                rxn_string = re.rxn_string
                dE_rxn_per_mole = re.dE_rxn
                re = ReactionEnergy(
                    input_energies=input_energies,
                    reactants=reactants,
                    products=products,
                    energy_key="E",
                    norm="atom",
                )
                dE_rxn_per_atom = re.dE_rxn

                rxns[M][rxn]["rxn_string"] = rxn_string
                rxns[M][rxn][T] = {
                    "dE_rxn_per_mole": dE_rxn_per_mole,
                    "dE_rxn_per_atom": dE_rxn_per_atom,
                }

    write_json(rxns, fjson)
    return read_json(fjson)


def main():
    Ef_mp = get_compounds_from_mp()
    Efs = get_compounds_from_chrisc(Ef_mp)
    dGfs = get_dGf(Efs)
    rxns = get_rxn_energies(dGfs)

    return Ef_mp, Efs, dGfs, rxns


if __name__ == "__main__":
    main()
