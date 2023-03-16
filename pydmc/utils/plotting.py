import matplotlib as mpl
from pydmc.data.plotting_configs import get_color_palettes


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
