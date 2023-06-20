from pydmclab.utils.handy import read_json
import matplotlib.pyplot as plt
import os
from pydmclab.plotting.dos import ax_tdos
from pydmclab.plotting.utils import get_colors


def main():
    test_data_dir = "../data/test_data/vasp/AlN"
    tdos = read_json(os.path.join(test_data_dir, "tdos.json"))

    fig = plt.figure()
    ax1 = plt.subplot(121)
    ax1 = ax_tdos(
        tdos,
        population_sources=["total", "Al"],
        colors={"total": "black", "Al": "green"},
        color_palette=get_colors("set2"),
        params={"line_alpha": 0.9, "fill_alpha": 0.2, "lw": 1},
        special_labels=None,
        spins="summed",
        normalization=1,
        smearing=1,
        Efermi=0.0,
        xlim=(0, 10),
        ylim=(-2, 2),
        xticks=(True, [0, 5, 10]),
        yticks=(True, [-2, -1, 0, 1, 2]),
        xlabel="DOS",
        ylabel=r"$E-E_F\/(eV)$",
        legend=True,
        title=None,
        savename=None,
        show=False,
    )

    ax2 = plt.subplot(122)
    ax2 = ax_tdos(
        tdos,
        population_sources=["total", "Al"],
        colors={"total": "black", "Al": "orange"},
        color_palette=get_colors("set2"),
        params={"line_alpha": 0.9, "fill_alpha": 0.2, "lw": 1},
        special_labels=None,
        spins="summed",
        normalization=1,
        smearing=2,
        Efermi=0.0,
        xlim=(0, 10),
        ylim=(-2, 2),
        xticks=(True, [0, 5, 10]),
        yticks=(False, [-2, -1, 0, 1, 2]),
        xlabel="DOS",
        ylabel="",
        legend=True,
        title=None,
        savename=None,
        show=False,
    )

    return tdos


if __name__ == "__main__":
    tdos = main()
