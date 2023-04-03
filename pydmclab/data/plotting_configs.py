import matplotlib as mpl
import os
import numpy as np
from pydmc.utils.handy import read_json, write_json

this_dir, this_filename = os.path.split(__file__)
DATA_PATH = os.path.join(this_dir, "data")


def get_colors_dict(palette, names):

    palette = mpl.cm.get_cmap(palette)
    # Create an empty dictionary to store the mappings
    color_dict = {}

    # Loop through each color in the colormap

    for i in range(palette.N):
        # Convert the color index to a RGB tuple
        rgb = tuple(np.array(palette(i)[:3]))
        # Add the RGB tuple to the dictionary, using the color name as the key
        color_dict[names[i]] = rgb

    return color_dict


def get_color_palettes(remake=False):
    fjson = os.path.join(DATA_PATH, "colors.json")
    if not remake and os.path.exists(fjson):
        return read_json(fjson)
    tab10 = mpl.cm.get_cmap("tab10")
    tab10_names = [
        "blue",
        "orange",
        "green",
        "red",
        "purple",
        "brown",
        "pink",
        "gray",
        "yellow",
        "cyan",
    ]

    paired = mpl.cm.get_cmap("Paired")
    paired_names = [
        "light blue",
        "blue",
        "light green",
        "green",
        "light red",
        "red",
        "light orange",
        "orange",
        "light purple",
        "purple",
        "light brown",
        "brown",
    ]

    set2 = mpl.cm.get_cmap("Set2")
    set2_names = [
        "green",
        "orange",
        "blue",
        "pink",
        "light green",
        "yellow",
        "brown",
        "gray",
    ]

    dark2 = mpl.cm.get_cmap("Dark2")
    dark2_names = [
        "green",
        "orange",
        "purple",
        "pink",
        "light green",
        "yellow",
        "brown",
        "gray",
    ]

    palettes = [tab10, paired, set2, dark2]
    palette_names = ["tab10", "paired", "set2", "dark2"]
    color_names = [tab10_names, paired_names, set2_names, dark2_names]

    data = {}
    for i, palette in enumerate(palettes):
        data[palette_names[i]] = get_colors_dict(palette, color_names[i])

    write_json(data, fjson)
    return read_json(fjson)


def main():
    colors = get_color_palettes()
    return colors


if __name__ == "__main__":
    d = main()
