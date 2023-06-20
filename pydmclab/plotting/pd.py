from pydmclab.core.comp import CompTools
from pydmclab.core.hulls import GetHullInputData, AnalyzeHull, MixingHull
from pydmclab.plotting.utils import set_rc_params, get_colors
from pydmclab.utils.handy import read_json, write_json

import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

import itertools

data_dir = "../data/test_data/ternarypd"


def get_data():
    return read_json(os.path.join(data_dir, "Mg-Cr-S-Na-Cl_query.json"))


def get_small_data(remake=False):
    fjson = os.path.join(data_dir, "Efs.json")
    if not remake and os.path.exists(fjson):
        return read_json(fjson)

    d = get_data()
    out = {k: {"Ef": d[k]["Ef_mp"]} for k in d}

    write_json(out, fjson)
    return read_json(fjson)


def get_hullout(Efs, remake=False):
    fjson = os.path.join(data_dir, "hullout.json")
    if not remake and os.path.exists(fjson):
        return read_json(fjson)

    ghid = GetHullInputData(Efs, "Ef")
    hullin = ghid.hullin_data()
    out = {}
    for space in hullin:
        hullout = AnalyzeHull(hullin, space).hull_output_data
        out.update(hullout)

    write_json(out, fjson)
    return read_json(fjson)


def get_mixing(hullout, remake=False):
    fjson = os.path.join(data_dir, "mixing.json")
    if not remake and os.path.exists(fjson):
        return read_json(fjson)

    hullout = {k: hullout[k] for k in ["Cl1Na1", "Cl2Mg1", "Cl4Mg1Na2", "Cl8Mg1Na6"]}
    print(hullout.keys())
    mh = MixingHull(
        hullout,
        varying_element="Na",
        end_members=["NaCl", "MgCl2"],
        shared_element_basis="Cl",
        energy_key="Ef",
    )

    print(mh.mixing_energies)
    out = mh.results

    write_json(out, fjson)
    return read_json(fjson)


class BinaryPD(object):
    def __init__(
        self,
        stability_data,
        end_members,
        polymorph_data={},
        stability_source="AnalyzeHull",
    ):
        """

        Args:
            stability_data (dict):
                AnalyzeHull(...).hull_output_data
                    {formula (str) :
                        {'Ef' : formation energy (eV/atom),
                        'stability' : True if on the hull else False}
                        }

                OR

                MixingHull(...).results
                    {formula (str) :
                        {'E_mix' : mixing energy (eV/atom),
                         'stability' : True if on the mixing hull else False,
                         'x' : mole fraction of right end member}}}



            end_members (list): [left end, right end]
                e.g., ['Mg', 'S'] for a typical Mg-S phase diagram (AnalyzeHull)
                e.g., ['MgCl2', 'NaCl'] for a mixing hull laong NaCl-MgCl2 (MixingHull)

            polymorph_data (dict):
                {formula (str) :
                    {ID (str) :
                        {'dE_gs' : energy above ground state (eV/atom)}}}

            stability_source (str): 'AnalyzeHull' or 'MixingHull'

        Returns:
            data (dict):
                {formula (str) :
                    {'Ef' : formation energy (eV/atom),
                     'x' : mole fraction of right end member,
                     'stability' : True if on the hull else False,
                     'polys' : [{'ID' : ID (str), 'Ef' : Ef_gs + dE_gs (eV/atom)}]}}


            left_end (str): left end member
            right_end (str): right end member
        """
        data = stability_data.copy()
        left_end, right_end = end_members

        if stability_source == "AnalyzeHull":
            if (CompTools(end_members[0]).n_els != 1) or (
                CompTools(end_members[1]).n_els != 1
            ):
                raise ValueError(
                    "end_members must be elements if source is AnalyzeHull; use MixingHull for other situations"
                )

            els = [CompTools(end_members[0]).els[0], CompTools(end_members[1]).els[0]]
            data = {
                k: data[k] for k in data if set(CompTools(k).els).issubset(set(els))
            }

            for formula in data:
                x = CompTools(formula).mol_frac(right_end)
                data[formula]["x"] = x

        elif stability_source == "MixingHull":
            for formula in data:
                data[formula]["Ef"] = data[formula]["E_mix"]

        for formula in stability_data:
            data[formula]["polys"] = []
            if formula in polymorph_data:
                for ID in polymorph_data[formula]:
                    dE_gs = polymorph_data[formula][ID]["dE_gs"]
                    data[formula]["polys"].append(
                        {"ID": ID, "Ef": data[formula]["Ef"] + dE_gs}
                    )

        self.data = data
        self.left_end, self.right_end = left_end, right_end

    @property
    def sorted_xs(self):
        """

        Returns:
            list of x values sorted from 0 to 1
        """
        data = self.data
        return sorted([data[k]["x"] for k in data])

    @property
    def stable(self):
        """


        Returns:
            list of dicts
                {'x' : mole fraction of right end member,
                 'Ef' : formation energy (eV/atom),
                 'label' : formula}
            sorted by x (ascending)
            only for stable compounds
        """
        data = self.data
        data = {k: data[k] for k in data if data[k]["stability"]}
        xs = self.sorted_xs
        to_plot = []
        for x in xs:
            formula = [k for k in data if data[k]["x"] == x]
            if formula:
                formula = formula[0]
            else:
                continue
            to_plot.append({"x": x, "Ef": data[formula]["Ef"], "label": formula})

        return to_plot

    @property
    def unstable(self):
        """
        Returns:
            list of dicts
                {'x' : mole fraction of right end member,
                 'Ef' : formation energy (eV/atom),
                 'label' : formula (for gs) or ID (for non-gs)}
            sorted by x (ascending)
            only for unstable compounds
        """
        data = self.data
        to_plot = []
        for formula in data:
            polys = data[formula]["polys"]
            x = data[formula]["x"]
            for p in polys:
                to_plot.append({"x": x, "Ef": p["Ef"], "label": p["ID"]})
            if not data[formula]["stability"]:
                to_plot.append({"x": x, "Ef": data[formula]["Ef"], "label": formula})

        return to_plot

    def ax_pd(
        self,
        color_palette=get_colors("tab10"),
        stable_params={
            "markerfacecolor": "white",
            "color": "blue",
            "marker": "o",
            "markersize": 8,
            "lw": 2,
        },
        unstable_params={
            "color": "white",
            "edgecolor": "red",
            "marker": "^",
            "s": 80,
        },
        label_ends=True,
        el_order=None,
        xlim=(0, 1),
        ylim=(-5, 1),
        xticks=(True, [0, 0.2, 0.4, 0.6, 0.8, 1]),
        yticks=(True, [-5, -4, -3, -2, -1, 0, 1]),
        xlabel="auto",
        ylabel=r"$\Delta$" + r"$\it{E}$" + r"$_f$" + r"$\/(\frac{eV}{atom})$",
        legend=True,
        title=None,
        savename=None,
        show=True,
    ):
        """_summary_

        Args:
            color_palette (_type_, optional): _description_. Defaults to get_colors("tab10").
            stable_params (dict, optional): _description_. Defaults to { "markerfacecolor": "white", "color": "blue", "marker": "o", "markersize": 8, "lw": 2, }.
            unstable_params (dict, optional): _description_. Defaults to { "color": "white", "edgecolor": "red", "marker": "^", "s": 80, }.
            label_ends (bool, optional): _description_. Defaults to True.
            el_order (_type_, optional): _description_. Defaults to None.
            xlim (tuple, optional): _description_. Defaults to (0, 1).
            ylim (tuple, optional): _description_. Defaults to (-5, 1).
            xticks (tuple, optional): _description_. Defaults to (True, [0, 0.2, 0.4, 0.6, 0.8, 1]).
            yticks (tuple, optional): _description_. Defaults to (True, [-5, -4, -3, -2, -1, 0, 1]).
            xlabel (str, optional): _description_. Defaults to "auto".
            ylabel (regexp, optional): _description_. Defaults to r"$\Delta$"+r"$\it{E}$"+r"$"+r"$\/(\frac{eV}{atom})$".
            legend (bool, optional): _description_. Defaults to True.
            title (_type_, optional): _description_. Defaults to None.
            savename (_type_, optional): _description_. Defaults to None.
            show (bool, optional): _description_. Defaults to True.
        """
        stable_params["color"] = color_palette[stable_params["color"]]
        unstable_params["edgecolor"] = color_palette[unstable_params["edgecolor"]]

        stable_to_plot = self.stable
        unstable_to_plot = self.unstable

        left_end, right_end = self.left_end, self.right_end

        x, y, labels = [], [], []
        for d in stable_to_plot:
            x.append(d["x"])
            y.append(d["Ef"])
            labels.append(d["label"])

        ax = plt.plot(x, y, label="stable", **stable_params, zorder=1)

        x, y, labels = [], [], []
        for d in unstable_to_plot:
            x.append(d["x"])
            y.append(d["Ef"])
            labels.append(d["label"])

        ax = plt.scatter(x, y, label="unstable", **unstable_params, zorder=0)

        ax = plt.xticks(visible=xticks[0], ticks=xticks[1])
        ax = plt.yticks(visible=yticks[0], ticks=yticks[1])
        ax = plt.xlim(xlim)
        ax = plt.ylim(ylim)

        if label_ends:
            xpos, ypos = 0, 1.1 * max(yticks[1])
            ax = plt.text(
                xpos,
                ypos,
                get_label(left_end, el_order),
                fontsize=20,
                ha="center",
                va="bottom",
            )
            xpos, ypos = 1, 1.1 * max(yticks[1])
            ax = plt.text(
                xpos,
                ypos,
                get_label(right_end, el_order),
                fontsize=20,
                ha="center",
                va="bottom",
            )
        if xlabel == "auto":
            xlabel = (
                r"$x\/in\/$"
                + "(%s)" % get_label(left_end, el_order)
                + r"$_{1-x}$"
                + "(%s)" % get_label(right_end, el_order)
                + r"$_x$"
            )
        ax = plt.xlabel(xlabel)
        ax = plt.ylabel(ylabel)

        if legend:
            ax = plt.legend(loc=legend if isinstance(legend, str) else "best")


class TernaryPD(object):
    def __init__(self, stability_data, end_members):
        """

        Args:
            stability_data (dict):
                AnalyzeHull(...).hull_output_data
                    {formula (str) :
                        {'Ef' : formation energy (eV/atom),
                        'stability' : True if on the hull else False}
                        }

                NOTE: MixingHull results not supported (yet) for ternary diagrams

            end_members (list):
                list of end members (str) in the ternary system [right, top, left]
                NOTE: only elemental end members supported (for now)

        Returns:
            data (dict):
                {formula (str) :
                    {'Ef' : formation energy (eV/atom),
                     'stability' : True if on the hull else False,
                     'a' : mol fraction of right end member,
                     'b' : mol fraction of top end member,
                     'c' : mol fraction of left end member,
                     'x' : x coordinate of formula in ternary diagram,
                     'y' : y coordinate of formula in ternary diagram,}

            left_end (str):
                left end member (str) in the ternary system

            right_end (str):
                right end member (str) in the ternary system

            top_end (str):
                top end member (str) in the ternary system
        """
        stability_data = stability_data.copy()

        for e in end_members:
            if CompTools(e).n_els != 1:
                raise ValueError("Only elemental end members supported (for now)")
        right_end, top_end, left_end = [CompTools(e).els[0] for e in end_members]
        els_in_end_members = [right_end, top_end, left_end]

        data = {
            k: stability_data[k]
            for k in stability_data
            if set(CompTools(k).els).issubset(set(els_in_end_members))
        }
        for formula in data:
            a, b, c = [CompTools(formula).mol_frac(el) for el in els_in_end_members]
            data[formula].update({"a": a, "b": b, "c": c})
            x, y = triangle_to_square((a, b, c))
            data[formula].update({"x": x, "y": y})

        self.data = data
        self.right_end, self.top_end, self.left_end = els_in_end_members

    @property
    def points_to_plot(self):
        """

        Returns:
            list of dicts
                {'x' : x coordinate of formula in ternary diagram,
                 'y' : y coordinate of formula in ternary diagram,
                 'Ef' : formation energy (eV/atom),
                 'stability' : True if on the hull else False,
                 'formula' : formula (str)}
        """
        data = self.data
        to_plot = []
        for formula in data:
            to_plot.append(
                {
                    "x": data[formula]["x"],
                    "y": data[formula]["y"],
                    "Ef": data[formula]["Ef"],
                    "stability": data[formula]["stability"],
                    "formula": formula,
                }
            )
        return to_plot

    @property
    def space(self):
        """_summary_

        Returns:
            space (str) of the ternary diagram right-el_top-el_left-el
        """
        return "_".join(sorted([self.right_end, self.top_end, self.left_end]))

    @property
    def hullin(self):
        """_summary_

        Returns:
            {formula (str) :
                {'E' : formation energy (eV/atom),
                 'amts' : {el (str) : mol fraction of el in space (float)}}
        """
        data = self.data
        space = self.space
        out = {
            space: {
                c: {
                    "E": data[c]["Ef"],
                    "amts": {el: CompTools(c).mol_frac(el) for el in space.split("_")},
                }
                for c in data
            }
        }
        return out

    @property
    def lines_to_plot(self):
        """_summary_

        Returns:
            list of dicts
                {'x' : x endpoints of line in ternary diagram,
                 'y' : y endpoints of line in ternary diagram}
        """
        data = self.data
        hullin = self.hullin
        ah = AnalyzeHull(hullin, self.space)
        simplices = ah.hull_simplices
        lines = uniquelines(simplices)
        sorted_compounds = ah.sorted_compounds
        to_plot = []
        for l in lines:
            compounds = (sorted_compounds[l[0]], sorted_compounds[l[1]])

            a0, b0, c0 = [data[compounds[0]][v] for v in ["a", "b", "c"]]
            a1, b1, c1 = [data[compounds[1]][v] for v in ["a", "b", "c"]]
            x0, y0 = triangle_to_square((a0, b0, c0))
            x1, y1 = triangle_to_square((a1, b1, c1))
            to_plot.append({"x": [x0, x1], "y": [y0, y1]})

        return to_plot

    def ax_pd(
        self,
        color_palette=get_colors("tab10"),
        stable_params={
            "color": "white",
            "edgecolor": "blue",
            "marker": "o",
            "s": 50,
        },
        unstable_params={
            "color": "white",
            "edgecolor": "red",
            "marker": "^",
            "s": 50,
        },
        label_els=True,
        label_compounds=[],
        legend=True,
        title=None,
        savename=None,
        show=True,
    ):
        """_summary_

        Args:
            color_palette (_type_, optional): _description_. Defaults to get_colors("tab10").
            stable_params (dict, optional): _description_. Defaults to { "color": "white", "edgecolor": "blue", "marker": "o", "s": 50, }.
            unstable_params (dict, optional): _description_. Defaults to { "color": "white", "edgecolor": "red", "marker": "^", "s": 50, }.
            label_els (bool, optional): _description_. Defaults to True.
            label_compounds (list, optional): _description_. Defaults to [].
            legend (bool, optional): _description_. Defaults to True.
            title (_type_, optional): _description_. Defaults to None.
            savename (_type_, optional): _description_. Defaults to None.
            show (bool, optional): _description_. Defaults to True.
        """
        data = self.data

        stable_params["edgecolor"] = color_palette[stable_params["edgecolor"]]
        unstable_params["edgecolor"] = color_palette[unstable_params["edgecolor"]]

        points_to_plot = self.points_to_plot
        stable_to_plot = [d for d in points_to_plot if d["stability"]]
        unstable_to_plot = [d for d in points_to_plot if not d["stability"]]

        x, y, labels = [], [], []
        for d in stable_to_plot:
            x.append(d["x"])
            y.append(d["y"])
            labels.append(d["formula"])

        ax = plt.scatter(x, y, label="stable", **stable_params, zorder=2)

        x, y, labels = [], [], []
        for d in unstable_to_plot:
            x.append(d["x"])
            y.append(d["y"])
            labels.append(d["formula"])

        ax = plt.scatter(x, y, label="unstable", **unstable_params, zorder=1)

        lines_to_plot = self.lines_to_plot
        for l in lines_to_plot:
            ax = plt.plot(l["x"], l["y"], color="black", zorder=0)

        for spine in ["bottom", "left", "top", "right"]:
            ax = plt.gca().spines[spine].set_visible(False)
        ax = plt.gca().xaxis.set_ticklabels([])
        ax = plt.gca().yaxis.set_ticklabels([])
        ax = plt.gca().tick_params(bottom=False, top=False, left=False, right=False)

        if label_els:
            el_label_size = 18
            left_el_pos = (-0.02, -0.05)
            right_el_pos = (1.02, -0.05)
            top_el_pos = (0.5, 0.89)
            ax = plt.text(
                right_el_pos[0],
                right_el_pos[1],
                self.right_end,
                fontsize=el_label_size,
                horizontalalignment="left",
            )
            ax = plt.text(
                top_el_pos[0],
                top_el_pos[1],
                self.top_end,
                fontsize=el_label_size,
                horizontalalignment="center",
            )
            ax = plt.text(
                left_el_pos[0],
                left_el_pos[1],
                self.left_end,
                fontsize=el_label_size,
                horizontalalignment="right",
            )

        if legend:
            ax = plt.legend(loc=legend if isinstance(legend, str) else "best")


def triangle_to_square(pt):
    """
    Args:
        pt (tuple) - (left, top, right) triangle point
                e.g., (1, 0, 0) is left corner
                e.g., (0, 1, 0) is top corner

    Returns:
        pt (tuple) - (x, y) same point in 2D space
    """
    a, b, c = pt
    converter = np.array([[1, 0], [0.5, np.sqrt(3) / 2]])
    new_pt = np.dot(np.array([a, b]), converter)
    return new_pt.transpose()


def square_to_triangle(pt):
    """
    Args:
        pt (tuple) - (x, y) planar point in 2D space

    Returns:
        pt (tuple) - (left, top, right) triangle point
                e.g., (1, 0, 0) is left corner
                e.g., (0, 1, 0) is top corner
    """
    x, y = pt
    b = y * 2 / np.sqrt(3)
    a = x - 0.5 * b
    c = 1 - a - b
    return (a, b, c)


def uniquelines(q):
    """
    Given all the facets, convert it into a set of unique lines.  Specifically
    used for converting convex hull facets into line pairs of coordinates.

    Args:
        q: A 2-dim sequence, where each row represents a facet. E.g.,
            [[1,2,3],[3,6,7],...]

    Returns:
        setoflines:
            A set of tuple of lines.  E.g., ((1,2), (1,3), (2,3), ....)
    """
    setoflines = set()
    for facets in q:
        for line in itertools.combinations(facets, 2):
            setoflines.add(tuple(sorted(line)))
    return setoflines


def get_label(cmpd, els=None):
    """
    Args:
        cmpd (str) - chemical formula
        els (list) - ordered list of elements (str) as you want them to appear in label

    Returns:
        neatly formatted chemical formula label
    """
    if not els:
        els = CompTools(cmpd).els
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
    set_rc_params()
    query = get_small_data()
    hullout = get_hullout(query)
    mixing = get_mixing(hullout, remake=False)
    # bpd_ah = BinaryPD(hullout, ["Mg", "Cl"], stability_source="AnalyzeHull")
    # bpd_ah.ax_pd()
    bpd_mh = BinaryPD(
        mixing,
        ["NaCl", "MgCl2"],
        polymorph_data={
            "Cl4Mg1Na2": {"mp-1234": {"dE_gs": 0.05}},
            "Cl8Mg1Na6": {
                "mp-4321": {"dE_gs": 0.1},
            },
        },
        stability_source="MixingHull",
    )

    tpd = None

    bpd_mh.ax_pd(
        yticks=(True, [-0.1, 0.1]), ylim=(-0.1, 0.1), el_order=("Na", "Mg", "Cl")
    )

    # tpd = TernaryPD(hullout, ["S", "Cr", "Mg"])
    # tpd.ax_pd()
    return query, hullout, mixing, bpd, tpd


if __name__ == "__main__":
    query, hullout, mixing, bpd, tpd = main()
