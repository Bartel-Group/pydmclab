import os

from pydmclab.data.plotting_configs import get_color_palettes


from pydmclab.core.comp import CompTools
from pydmclab.core.struc import StrucTools
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
