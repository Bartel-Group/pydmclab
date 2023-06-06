import os

from pydmclab.data.plotting_configs import get_color_palettes

from scipy.ndimage import gaussian_filter1d

from pydmclab.core.comp import CompTools
from pydmclab.core.struc import StrucTools
from pydmclab.utils.plotting import get_colors, set_rc_params
from pydmclab.utils.handy import read_json, write_json

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import random


def get_colors(palette):
    """

    returns rgb colors that are nicer than matplotlibs defaults

    Args:
        palette (str):
            'tab10' : tableau 10 colors
            'paired' : "paired" light and dark colors
            'set2' : pastel-y colors
            'dark2' : dark pastel-y colors

        For reference, see: https://matplotlib.org/stable/_images/sphx_glr_colormaps_006.png


    Returns:
        {color (str) : rgb (tuple)}

        so, to use this you could do:
            from pydmc.utils.plotting import get_colors
            my_colors = get_colors('tab10')
            ax = plt.scatter(x, y, color=my_colors['blue'])
    """
    colors = get_color_palettes()[palette]
    colors["black"] = (0, 0, 0)
    colors["white"] = (1, 1, 1)
    return colors


def set_rc_params():
    """
    Args:

    Returns:
        dictionary of settings for mpl.rcParams
    """
    params = {
        "axes.linewidth": 1.5,
        "axes.unicode_minus": False,
        "figure.dpi": 300,
        "font.size": 20,
        "legend.frameon": False,
        "legend.handletextpad": 0.4,
        "legend.handlelength": 1,
        "legend.fontsize": 12,
        "mathtext.default": "regular",
        "savefig.bbox": "tight",
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.top": True,
        "ytick.right": True,
        "axes.edgecolor": "black",
        "figure.figsize": [6, 4],
    }
    for p in params:
        mpl.rcParams[p] = params[p]
    return params


# Please note that this function only deals with the case of total spin
def ax_tdos(
    tdos,
    population_sources="all",
    colors={"total": "black"},
    color_palette=get_colors("set2"),
    params={"line_alpha": 0.9, "fill_alpha": 0.2, "lw": 1},
    special_labels=None,
    spins="summed",
    normalization=None,
    smearing=0.2,
    Efermi=0.0,
    xlim=(0, 10),
    ylim=(-2, 2),
    xticks=(True, [0, 5, 10]),
    yticks=(True, [-2, -1, 0, 1, 2]),
    xlabel="DOS",
    ylabel=r"$E-E_F\/(eV)$",
    legend=True,
    title=None,
    savename=False,
    show=True,
):
    """
    Args:
        tdos (dict)
            result of pydmclab.hpc.analyze.tdos()
                list(d.keys()) = ['E', 'total', 'up', 'down'] + [list of elements (str)]
                    d['E'] = 1d array of energies corresponding with DOS
                    d[el] = 1d array of DOS for that element (sums all sites, orbitals, and spins)
                    d['total'] = 1d array of DOS for structure (sums all elements, sites, orbitals, spins)
                    d['up'] or d['down']:
                        keys are ['total'] + [list of elements (str)]
                        d['up']['total'] = 1d array of spin-up DOS for structure
                        d['down'][el] = 1d array of spin-down DOS for that element
                        etc
                so if I wanted to plot the total DOS for my structure and separate spin up (+ DOS) and spin down (- DOS)
                    energies = d['E']
                    dos_up = d['up']['total']
                    dos_down = d['down']['total']
                    plt.plot(dos_up, energies)
                    plt.plot(-1*dos_down, energies)

        population_sources (str or list)
            'all' : plot all elements and the total
            ['total'] : plots just the total
            ['Al', 'N'] : plots just Al and N
            ['Al'] : plots just Al

        colors (dict)
            {element or 'total' (str) : color (str)}

        color_palette (dict)
            {color (str) : rgb (tuple)}
                usually loaded from pydmc.utils.plotting.get_colors()

        params (dict)
            {'fill_alpha' : transparency for occ populations,
             'line_alpha' : transparency for DOS line,
             'lw' : DOS linewidth}

        special_labels (dict)
            {element or 'total' (str) : label (str)}

        spins (str)
            'summed' : plot + and - DOS together as +
            'separate' : plot + and - DOS separately as + and -
                @TO-DO: implement separate spins

        normalization (float)
            divide DOS by this number
                common normalizations:
                    1 : no normalization (same as None)
                    CompTools(formula).n_atoms (per formula unit)
                    results['meta']['all_input_parameters']['NELECT'] (per electron)

        smearing (float or False)
            std. dev. for Gaussian smearing of DOS or False for no smearing

        Efermi (float)
            Fermi level (eV)

        xlim (tuple)
            (xmin (float), xmax (float))

        ylim (tuple)
            (ymin (float), ymax (float))

        xticks (tuple)
            (bool to show label or not, (xtick0, xtick1, ...))

        xlabel (str)
            x-axis label

        ylabel (str)
            y-axis label

        legend (bool)
            include legend

        title (str)
            title of plot

        savename (str)
            if False: don't save (just return ax)
            if str, save fig object at this location

        show (bool)
            show plot

    Returns:
        matplotlib axis object
    """
    set_rc_params()
    random.seed(42)

    if spins != "summed":
        raise NotImplementedError("Sorry, only summed spins are implemented right now.")

    if savename:
        fig = plt.figure(figsize=(5, 8))

    Efermi = 0.0
    occupied_up_to = Efermi
    print("Fermi level = ", occupied_up_to)

    if not normalization:
        normalization = 1

    if population_sources == "all":
        tdos_keys = list(tdos.keys())
        non_sources = ["E", "up", "down"]
        population_sources = [k for k in tdos_keys if k not in non_sources]

    for src in population_sources:
        orig = tdos[src]
        to_plot = np.array(orig) / normalization
        tdos[src] = to_plot

    for el in population_sources:
        if el in colors:
            color = color_palette[colors[el]]
        else:
            color = color_palette[random.choice(list(color_palette.keys()))]

        label = el
        if special_labels:
            if el in special_labels:
                label = special_labels[el]

        energies = tdos["E"]
        populations = tdos[el]

        occ_energies = []
        occ_populations = []
        unocc_energies = []
        unocc_populations = []

        for idx, E in enumerate(energies):
            if E == occupied_up_to:
                occ_energies.append(energies[idx])
                occ_populations.append(populations[idx])
                unocc_energies.append(energies[idx])
                unocc_populations.append(populations[idx])
            if E < occupied_up_to:
                occ_energies.append(energies[idx])
                occ_populations.append(populations[idx])
            elif E > occupied_up_to:
                unocc_energies.append(energies[idx])
                unocc_populations.append(populations[idx])

        # smearing with Gaussian filter
        if smearing:
            occ_populations = gaussian_filter1d(occ_populations, smearing)
            unocc_populations = gaussian_filter1d(unocc_populations, smearing)

        ax = plt.plot(
            occ_populations,
            occ_energies,
            color=color,
            label=label,
            alpha=params["line_alpha"],
            lw=params["lw"],
        )
        ax = plt.plot(
            unocc_populations,
            unocc_energies,
            color=color,
            label="__nolegend__",
            alpha=params["line_alpha"],
            lw=params["lw"],
        )
        ax = plt.fill_betweenx(
            occ_energies, occ_populations, color=color, alpha=params["fill_alpha"], lw=0
        )

    ax = plt.axhline(y=Efermi, color="black", linestyle="--")

    ax = plt.xticks(xticks[1])
    ax = plt.yticks(yticks[1])
    if not xticks[0]:
        ax = plt.gca().xaxis.set_ticklabels([])
    if not yticks[0]:
        ax = plt.gca().yaxis.set_ticklabels([])
    ax = plt.xlabel(xlabel)
    ax = plt.ylabel(ylabel)
    ax = plt.title(title)
    ax = plt.xlim(xlim)
    ax = plt.ylim(ylim)

    if legend:
        if isinstance(legend, str):
            ax = plt.legend(loc=legend)
        else:
            ax = plt.legend(loc="best")

    if show:
        plt.show()

    if savename:
        plt.savefig(savename)

    return ax


def get_label(cmpd, els):
    """
    Args:
        cmpd (str) - chemical formula
        els (list) - ordered list of elements (str) as you want them to appear in label

    Returns:
        neatly formatted chemical formula label
    """
    label = r"$"
    for el in els:
        amt = CompTools(cmpd).stoich(el)
        if amt == 0:
            continue
        label += el
        if amt == 1:
            continue
        label += "_{%s}" % amt
    label += "$"
    return label


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
