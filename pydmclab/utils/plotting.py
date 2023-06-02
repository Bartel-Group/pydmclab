import os

from pydmclab.data.plotting_configs import get_color_palettes

from scipy.ndimage import gaussian_filter1d

from pydmclab.core.comp import CompTools
from pydmclab.core.struc import StrucTools
from pydmclab.utils.plotting import get_colors, set_rc_params
from pydmclab.utils.handy import read_json, write_json

import matplotlib as mpl
import matplotlib.pyplot as plt

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


# make slim tdos results 
def make_smaller_results(xc=None, remake=False):
    # data directory
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    
    fjson = os.path.join(DATA_DIR, "tdos.json")
    if not remake and os.path.exists(fjson):
        return read_json(fjson)

    d = read_json(os.path.join(DATA_DIR, "results.json"))
    out = {}
    
    # only keep the converged results and the xc you specified
    for k in d:
        if not d[k]["results"]["convergence"]:
            continue
        if xc != None and not d[k]["meta"]["setup"]["xc"] == xc:
            continue
        out[k] = {}

    # keep whatever keys you need to plot tdos
    # Note: we need "structure" only if we want to normalize by formula unit
    keep_keys = ["results", "meta", "structure", "tdos"]
    
    for k in out:        
        for keep_key in keep_keys:
            out[k][keep_key] = d[k][keep_key].copy()

    write_json(out, fjson)
    return read_json(fjson)


# Please note that this function only deals with the case of total spin
def plot_tdos(di, 
        formula,
        what_to_plot,
        colors_and_labels,
        title,
        saveas,
        normalization,
        xlabel,
        colors = get_colors('set2'),
        xlim=(0, 10), ylim=(-2, 2), 
        xticks=(True, [0, 5, 10]), yticks=(True, [-2, -1, 0, 1, 2]), 
        ylabel=r'$E-E_F\/(eV)$',
        legend=True,
        smearing=0.2,
        show=False,
        ):
    """
    Args:
        calc_dir (str) - path to calculation with DOSCAR
        formula - formula of compound
        what_to_plot (dict) - {element or 'total' (str)}
        colors_and_labels (dict) - {element-spin-orbital (str) : {'color' : color (str),
                                                                  'label' : label (str)}}
        title (str) - title of plot
        saveas (str) - filename to save plot as
        colors - please select the color you like
        xlim (tuple) - (xmin (float), xmax (float))
        ylim (tuple) - (ymin (float), ymax (float))
        xticks (tuple) - (bool to show label or not, (xtick0, xtick1, ...))
        yticks (tuple) - (bool to show label or not, (ytick0, ytick1, ...))
        xlabel (str) - x-axis label
        ylabel (str) - y-axis label
        legend (bool) - include legend or not
        smearing (float or False) - std. dev. for Gaussian smearing of DOS or False for no smearing         
        normalization ('electron', 'formula') - divide populations by number of electrons or formula unit
        show (bool) - if True, show figure; else just return ax
                   
    Returns:
        matplotlib axes object
    """
    set_rc_params()

    fig = plt.figure(figsize=(5,8))
    ax = plt.subplot(111)

    print("formula = ", formula)
    
    Efermi = 0.
    occupied_up_to = Efermi
    print('Fermi level = ',occupied_up_to)

    population = {}
    
    #normalized dos w/ number of electrons or formula units
    if normalization == 'electron':
        normalization = di['meta']['all_input_settings']['NELECT']
        print('electron normalization = ', normalization)
    elif normalization == 'formula':
        normalization = len(StrucTools(di['structure']).structure)/ CompTools(di['results']['formula']).n_atoms
        print('formula normalization = ', normalization)

    for ele in di['tdos']:
        print(ele)
        if ((ele == 'E') or (ele == 'up') or (ele == 'down')):
            continue

        population[ele] = [val/normalization for val in di['tdos'][ele]]


    dos_lw = 1    

    for element in what_to_plot:
        if str(colors_and_labels[element]['color']) == 'black':
            color = colors_and_labels[element]['color']
        else:
            color = colors[colors_and_labels[element]['color']]
        label = colors_and_labels[element]['label']
                  
        energies = di['tdos']['E']
        populations = population[element]
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
            elif E < occupied_up_to:
                occ_energies.append(energies[idx])
                occ_populations.append(populations[idx])
            elif E > occupied_up_to:
                unocc_energies.append(energies[idx])
                unocc_populations.append(populations[idx])
                  
        # smearing with Gaussian filter
        if smearing:
            occ_populations = gaussian_filter1d(occ_populations, smearing)
            unocc_populations = gaussian_filter1d(unocc_populations, smearing)
        ax = plt.plot(occ_populations, occ_energies, color=color, label=label, alpha=0.9, lw=dos_lw)
        ax = plt.plot(unocc_populations, unocc_energies, color=color, label='__nolegend__', alpha=0.9, lw=dos_lw)                    
        ax = plt.fill_betweenx(occ_energies, occ_populations, color=color, alpha=0.2, lw=0)                               
         
    ax = plt.axhline(y = 0, color = 'black', linestyle = '--')


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
        ax = plt.legend(loc = 'upper right')
    if show:
        plt.show()
    if saveas:
        plt.savefig(saveas)
    
    return ax


def main():

    plot_tdos_switch = True
    
    d = make_smaller_results(xc = "metagga", remake=True)
    dcp = d.copy()
    
    for key in dcp:
        formula = dcp[key]["results"]["formula"]
        eles = CompTools(formula).els
        di = dcp[key]
        
        # please change the following line to the title you want
        title = formula
        normalization = 'formula'
        
        if normalization == 'formula':
            xlabel = r'$DOS\/(per\/f.u.)$'
        elif normalization == 'electron':
            xlabel = r'$DOS\/(per\/e^{-})$'
        
        
        if plot_tdos_switch:
            plot_tdos(di = di,
                formula = formula,
                what_to_plot={ele for ele in eles},
                colors_and_labels = {'Li' : {'color' : 'black', 'label' : 'Li'},
                                    'P' : {'color' : 'gray', 'label' : 'P'}, 
                                    'Mn' : {'color' : 'green', 'label' : "Mn"}, 
                                    'S' : {'color' : 'pink', 'label' : 'S'}}, 
                title = title,
                saveas = formula + '.png',
                normalization = normalization,
                xlabel = xlabel)

    return


if __name__ == "__main__":
    main()