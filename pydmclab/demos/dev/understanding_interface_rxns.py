from pydmclab.utils.plotting import set_rc_params

set_rc_params()

import matplotlib.pyplot as plt

BaO = (1, -2.831 * 2)
TiO2 = (0, -3.512 * 3)
BaTiO3 = (1 / 2, (3 - 1 / 2) * -3.502)
Ba2TiO4 = (2 / 3, (3 - 2 / 3) * -3.408)
cmpds = [TiO2, BaTiO3, Ba2TiO4, BaO]
data = dict(zip(["TiO2", "BaTiO3", "Ba2TiO4", "BaO"], cmpds))
print(data)
x = [c[0] for c in cmpds]


def rxn_energy(target):
    return target[1] - target[0] * BaO[1] - (1 - target[0]) * TiO2[1]


y = [rxn_energy(c) for c in cmpds]
print(y)

# y = [c[1] for c in cmpds]

fig = plt.figure()
ax = plt.subplot(111)
ax = plt.plot(x, y, marker="o", markerfacecolor="white")
for c in data:
    ax = plt.text(data[c][0], 0.08, c, horizontalalignment="center", fontsize=12)

ax = plt.xlabel(r"$x\/in(BaO)_{x}(TiO_2)_{1-x}$")
ax = plt.ylabel(r"$E_{rxn}\/(\frac{eV}{basis})$")
ax = plt.ylim()
plt.show()

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
