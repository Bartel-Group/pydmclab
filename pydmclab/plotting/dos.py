import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

from pydmclab.plotting.utils import set_rc_params, get_colors
from pydmclab.data.plotting_configs import get_color_palettes
from scipy.ndimage import gaussian_filter1d

import random


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
    NOTE: This function currently only deals with the case of total spin

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
                to plot the total DOS of a structure, separating spin up (+ DOS) and spin down (- DOS)
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
            'separate' : plot + and - DOS separately as + and - (note: not yet implemented)

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


def ax_tcohp(
    tcohp,
    colors,
    population_sources="all",
    color_palette=get_colors("set2"),
    params={"line_alpha": 0.9, "fill_alpha": 0.3, "lw": 1},
    special_labels=None,
    normalization=None,
    flip=True,
    bond_type="cohp",
    smearing=2.0,
    Efermi=0.0,
    xlim=(-6, 6),
    ylim=(-2, 2),
    xticks=(True, [-6, -4, -2, 0, 2, 4, 6]),
    yticks=(True, [-2, -1, 0, 1, 2]),
    xlabel=r"$-COHP$",
    ylabel=r"$E-E_F\/(eV)$",
    legend=True,
    title=None,
    savename=False,
    show=True,
):
    """
    Args:
        tcohp (dict)
            result of pydmclab.hpc.analyze.tcohp()
                list(d.keys()) = ['E'] + [el1-el2 (str) for all pairs of elements in structure that might be bonding]
                    note: el1-el2 is sorted alphabetically
                    e.g., the keys for Ca2Ru2O7 might be: ['Ca-Ca', 'Ca-Ru', 'Ca-O', 'Ru-Ru', 'O-Ru', 'O-O', 'E']
                d['E'] = 1d array of energies corresponding with DOS
                for each "bond type" (el1-el2), we have two keys: ['cohp', 'icohp']
                    cohp --> absolute populations
                    icohp --> integrated populations
                    d[el1-el2]['cohp' or 'icohp'] will return a 1d array of populations summed over all interactions of el1-el2 in the structure, summed over all spins
                to plot the total COHP for the interaction between Ru and O throughout the structure:
                    energies = d['E']
                    cohp = d['O-Ru']['cohp']
                    plt.plot(cohp, energies)

        colors (dict)
            {bond : color (str)}

        population_sources (str or list)
            'all' : plot all bonds
            ['S-S'] : plots just the S-S bonds
            ['S-S', 'P-S'] : plots just S-S and P-S bonds


        color_palette (dict)
            {color (str) : rgb (tuple)}
                usually loaded from pydmc.utils.plotting.get_colors()

        params (dict)
            {'fill_alpha' : transparency for occ populations,
             'line_alpha' : transparency for COHP line,
             'lw' : COHP linewidth}

        special_labels (dict)
            {bond : label (str)}

        normalization (float)
            divide COHP by this number
                common normalizations:
                    1 : no normalization (same as None)
                    CompTools(formula).n_atoms (per formula unit)
                    results['meta']['all_input_parameters']['NELECT'] (per electron)

        flip = (bool)
            flip the COHP value to -COHP

        bond_type (str)
            "cohp" or "icohp"

        smearing (float or False)
            std. dev. for Gaussian smearing of COHP or False for no smearing

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

    if savename:
        fig = plt.figure(figsize=(5, 8))

    Efermi = 0.0
    occupied_up_to = Efermi
    print("Fermi level = ", occupied_up_to)

    if not normalization:
        normalization = 1

    if population_sources == "all":
        tcohp_keys = list(tcohp.keys())
        non_sources = ["E"]
        population_sources = [k for k in tcohp_keys if k not in non_sources]

    for src in population_sources:
        orig = tcohp[src][bond_type]
        if not flip:
            to_plot = np.array(orig) / normalization
        else:
            to_plot = -np.array(orig) / normalization
        tcohp[src] = to_plot

    for bond in population_sources:
        if bond in colors:
            color = color_palette[colors[bond]]
        else:
            color = color_palette[random.choice(list(color_palette.keys()))]

        label = bond
        if special_labels:
            if bond in special_labels:
                label = special_labels[bond]

        energies = tcohp["E"]
        populations = tcohp[bond]

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

    ax = plt.axvline(x=0, color="black", linestyle="--")
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
