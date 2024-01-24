from pydmclab.utils.plotting import set_rc_params
from pydmclab.core.comp import CompTools
from pydmclab.core.hulls import MixingHull, GetHullInputData, AnalyzeHull

set_rc_params()

import matplotlib.pyplot as plt

d_BaTiO3 = 0.06
data = {
    "Ba1O1": {"x": 1, "E_per_at": -2.831},
    "O2Ti1": {"x": 0, "E_per_at": -3.512},
    "Ba1O3Ti1": {"x": 1 / 2, "E_per_at": -3.502 + d_BaTiO3},
    "Ba2O4Ti1": {"x": 2 / 3, "E_per_at": -3.408},
}

for cmpd in data:
    n_atoms = CompTools(cmpd).n_atoms
    n_basis = 3 - data[cmpd]["x"]
    data[cmpd]["n_atoms"] = n_atoms
    data[cmpd]["n_basis"] = n_basis

    data[cmpd]["E_per_fu"] = data[cmpd]["E_per_at"] * n_atoms
    data[cmpd]["E_per_basis"] = data[cmpd]["E_per_at"] * n_basis


def rxn_energy_basis(target, data):
    x = data[target]["x"]
    E_target = data[target]["E_per_basis"]
    E_BaO = data["Ba1O1"]["E_per_basis"]
    E_TiO2 = data["O2Ti1"]["E_per_basis"]
    return E_target - x * E_BaO - (1 - x) * E_TiO2


def rxn_energy_per_at(target, data):
    E_per_basis = rxn_energy_basis(target, data)
    return E_per_basis / (3 - data[target]["x"])


cmpds = ["O2Ti1", "Ba1O3Ti1", "Ba2O4Ti1", "Ba1O1"]
x = [data[c]["x"] for c in cmpds]
y_basis = [rxn_energy_basis(c, data) for c in cmpds]
y_per_at = [rxn_energy_per_at(c, data) for c in cmpds]

fig = plt.figure()
ax = plt.subplot(111)
ax = plt.plot(x, y_per_at, marker="o", markerfacecolor="white", label="eV/at")
ax = plt.plot(x, y_basis, marker="o", markerfacecolor="white", label="eV/3-x")

for c in data:
    ax = plt.text(data[c]["x"], 0.08, c, horizontalalignment="center", fontsize=12)

ax = plt.xlabel(r"$x\/in(BaO)_{x}(TiO_2)_{1-x}$")
ax = plt.ylabel(r"$E_{rxn}\/(eV)$")
ax = plt.legend()
ax = plt.ylim()
plt.show()

print(y_basis)
print(y_per_at)

ghid = GetHullInputData(data, "E_per_at")
hullin = ghid.hullin_data()
ah = AnalyzeHull(hullin, "Ba_O_Ti")
hullout = ah.hull_output_data

print(hullout)

"""
from pydmclab.core.hulls import MixingHull
d={'TiO2': -10.507/3, 'BaO': -5.649/2, 'Ba2TiO4': -23.799/7, 'BaTiO3' : -17.465/5 }
input_energies = {c : {'E' : d[c]} for c in d}

mh = MixingHull(input_energies=input_energies,
                left_end_member='TiO2',
                right_end_member='BaO')
out = mh.results
for k in out:
    out[k]['E_mix'] = out[k]['E_mix_per_fu']

for k in out:
    print(k)
    print(out[k]['E_mix'])

from pydmclab.plotting.pd import BinaryPD

fig2 = plt.figure()
ax2 = plt.subplot(111)
bpd = BinaryPD(out, ['BaO', 'TiO2'], stability_source='MixingHull')
ax2 = bpd.ax_pd()
plt.show()
"""
