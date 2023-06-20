import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

from pydmclab.plotting.utils import set_rc_params, get_colors
from pydmclab.data.plotting_configs import get_color_palettes
from scipy.ndimage import gaussian_filter1d

import random


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
