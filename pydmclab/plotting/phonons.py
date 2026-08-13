import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import physical_constants
from pydmclab.utils.handy import read_json, write_json, convert_numpy_to_native
from pydmclab.core.struc import StrucTools

from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from scipy.spatial.distance import cosine

# Conversion factors: eV/atom -> J/mol or kJ/mol
EV_TO_J = physical_constants['electron volt-joule relationship'][0]
AVOGADRO = physical_constants['Avogadro constant'][0]
EV_TO_J_PER_MOL = EV_TO_J * AVOGADRO
EV_TO_KJ_PER_MOL = EV_TO_J_PER_MOL / 1000.0


#This code needs a lot of cleanup and refactoring, but I want to get some quick plotting in before I spend time on that. So apologies for the messiness here.

def plot_phonon_bandstructure(bs_dict, 
                              labels=None, 
                              ylabel="Energy (eV)", 
                              title="", 
                              figsize=(8, 6)):
    """
    Plot a phonon band structure from Phonopy-style qpoints and frequencies.

    Parameters
    ----------
    bs_dict : dict
        {
        'qpoints': list of np.ndarray (npaths, npoints, 3),
        'frequencies': list of np.ndarray (npaths, npoints, nbranches)
         }
    labels : list of str or None
        High symmetry point labels. Must have length = npaths + 1.
        Example: ["Γ", "X", "K", "Γ", "L"] for 4 paths
        Note: The middle labels connect paths (end of one = start of next)
    ylabel : str
        Label for y-axis
    figsize : tuple
        Figure size
    """
    qpoints = np.array(bs_dict['qpoints'])
    frequencies = np.array(bs_dict['frequencies'])

    npaths, npoints, nbranches = frequencies.shape

    if labels is not None and len(labels) != npaths + 1:
        raise ValueError(f"labels must have length npaths + 1 ({npaths+1}), got {len(labels)}")
    
    def compute_path_distances(qpath):
        """Given qpath of shape (npoints, 3), return cumulative distance array."""
        dq = np.diff(qpath, axis=0)
        dist = np.linalg.norm(dq, axis=1)
        return np.concatenate(([0], np.cumsum(dist)))

    plt.figure(figsize=figsize)

    # Track cumulative distance and label positions
    x_offset = 0
    tick_positions = [0]  # Start with the first point

    for i in range(npaths):
        qpath = qpoints[i]       # (npoints, 3)
        freqs = frequencies[i]   # (npoints, nbranches)

        # Compute cumulative distance for this path
        dist = compute_path_distances(qpath)
        dist = dist + x_offset   # Shift to continue from previous path

        # Plot each phonon branch
        for b in range(nbranches):
            plt.plot(dist, freqs[:, b], linewidth=0.7, color='black')

        # Update offset for next path (end of current path)
        x_offset = dist[-1]
        
        # Add the end position of this path (which is start of next path)
        tick_positions.append(x_offset)

    # Draw vertical lines at high-symmetry points
    for pos in tick_positions:
        plt.axvline(pos, color="gray", linewidth=0.6, alpha=0.5)

    # Apply x-axis labels
    if labels is not None:
        plt.xticks(tick_positions, labels)
    else:
        plt.xticks([])

    # Labels and formatting
    plt.ylabel(ylabel)
    plt.xlabel("Wave Vector")
    plt.grid(alpha=0.2, axis='y')
    plt.xlim(0, tick_positions[-1])
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_phonon_dos(dos_dict, 
                    plot_in_thz=False, 
                    title="", 
                    figsize=(6,4), 
                    ylims=None):
    """
    Plot phonon density of states.

    Parameters
    ----------
    dos_dict : dict
            {
                'total_dos': [
                    {'E': -0.004, 'total_dos': 0.0},
                    {'E': 0.0, 'total_dos': 1.2},
                    ...
                ]
            }
    ylabel : str
        Label for y-axis
    figsize : tuple
        Figure size
    """
    frequencies = np.array([r['E'] for r in dos_dict['total_dos']])
    dos = np.array([r['total_dos'] for r in dos_dict['total_dos']])

    if plot_in_thz:
        h = physical_constants['Planck constant in eV/Hz'][0]
        frequencies = frequencies / (h * 1e12)  # Convert eV to THz
        dos = dos * (h*1e12)  # Adjust DOS units accordingly
        ylabel = "DOS (states/THz)"
        xlabel = "Frequency (THz)"
    else:
        ylabel = "DOS (states/eV)"
        xlabel = "Energy (eV)"
    plt.figure(figsize=figsize)
    plt.plot(frequencies, dos, color='blue', linewidth=1.2)
    plt.fill_between(frequencies, dos, color='lightblue', alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ylims is not None:
        plt.ylim(ylims)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_thermal_properties(thermal_props, 
                            plot_props=["F", "S"], 
                            title="", figsize=(8,4), 
                            plot_in_j_mol=False, 
                            atoms_per_formula_units=None):
    """
    Plot thermal properties (Helmholtz free energy, entropy, heat capacity).

    Properties with the same units share the same axis.
    Properties with different units get separate axes (e.g., F on left, S/Cv on right).

    Args:
        thermal_props (list[dict]):
           [{'T': float (K), 
            'F': float (eV/atom), 
            'S': float (eV/atom/K), 
            'Cv': float (eV/atom/K)}, ...]}
        plot_props (list or str):
            Which properties to plot. Options: "F" for Helmholtz free energy, "S" for entropy, "Cv" for heat capacity. Default is ["F", "S"] to plot both.
        title (str):
            Title for the plot.
        figsize : tuple
            Figure size
        plot_in_j_mol (bool):
            If True, convert F → kJ/mol and S/Cv → J/K/mol.
        atoms_per_formula_units : int or None
            Number of atoms per formula unit. If provided, scales per-atom to per-formula-unit.
    """
    if isinstance(plot_props, str):
        plot_props = [plot_props]

    # Define unit groups: which properties share units
    # "F" is in eV/atom or kJ/mol, "S" and "Cv" are in J/K/mol
    prop_to_group = {
        "F": "energy",
        "S": "thermal",
        "Cv": "thermal"
    }

    # Collect data
    temperatures = [point['T'] for point in thermal_props]
    props = {prop: [point[prop] for point in thermal_props] for prop in plot_props}

    # Create figure and axes
    fig, ax1 = plt.subplots(figsize=figsize)
    axes = [ax1]
    ax = ax1
    current_side = "left"

    # Plot each property, creating new axes for different unit groups
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    color_idx = 0

    
    # Check which groups are needed
    needed_groups = []

    for prop in plot_props:
        if prop not in thermal_props[0]:
            raise ValueError(f"Property '{prop}' not found in thermal_props. Available properties: {list(thermal_props[0].keys())}")
        group = prop_to_group.get(prop)
        if group and group not in needed_groups:
            needed_groups.append(group)

        # Scale by atoms_per_formula_units if provided
        if atoms_per_formula_units is not None:
                props[prop] = [val * atoms_per_formula_units for val in props[prop]]

        # Convert units if requested
        if plot_in_j_mol:
                if prop == "F":
                    props[prop] = [val * EV_TO_KJ_PER_MOL for val in props[prop]]
                elif prop == "S" or prop == "Cv":
                    props[prop] = [val * EV_TO_J_PER_MOL for val in props[prop]]
        
        # Determine which axis to use
        group_idx = needed_groups.index(group) if group in needed_groups else 0
        if group_idx == 0:
            ax = axes[0]  # Use left axis for first group
        else:
            # Create twin axis for additional unit groups
            if len(axes) <= group_idx:
                ax = ax1.twinx()
                axes.append(ax)
                current_side = "right" if current_side == "left" else "left"
            else:
                ax = axes[group_idx]

        if plot_in_j_mol:
            ylabel = f"{prop} ({'kJ/mol' if prop == 'F' else 'J/K/mol'}{'-fu' if formula_units else ''})"
        else:
            ylabel = f"{prop} (eV/atom)" if prop == "F" else f"{prop} (eV/atom/K)"
            if atoms_per_formula_units is not None:
                ylabel = ylabel.replace("atom", "fu")

        ax.set_ylabel(ylabel, fontsize=15)
        ax.tick_params(axis='y', labelsize=15)
        ax.plot(temperatures, props[prop], color=colors[color_idx], linewidth=1.5, label=prop)
        color_idx += 1

    # Add legend - collect handles and labels from all axes
    handles_labels = []
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            handles_labels.append((handles[0], labels[0]))
    
    if handles_labels:
        handles, labels = zip(*handles_labels)
        ax1.legend(handles, labels, loc='best', fontsize=12)

    # Set x-axis on the primary axis
    ax1.set_xlabel("Temperature (K)", fontsize=15)
    ax1.tick_params(axis='x', labelsize=15)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_relative_prop(
    thermal_props_dict,
    prop="F",
    align_at_0K=False,
    figsize=(6, 4),
    ylims=None,
    xlims=None,
    plot_in_j_mol=False,
    atoms_per_formula_units=None,
    colors=["green", "blue", "orange", "purple", "red"],
    title = "",
    xlabel_kwargs=None,
    ylabel_kwargs=None,
    legend_kwargs=None,
):
    
    """
    Plot the difference in a thermal property between two structures, for one or
    more datasets (e.g. DFT vs matcalc).

    The difference is always (second structure − first structure) within each dataset.

    Parameters
    ----------
    thermal_props_dict :
        Outer keys  → dataset label (shown in legend if more than one, e.g. "DFT", "matcalc").
        Inner keys  → structure label (e.g. "needle", "perovskite").
        Each leaf   → list of dicts with keys 'T', 'F', 'S', 'Cv', …

        Example:
            {
                'DFT':     {'needle': [...], 'perovskite': [...]},
                'matcalc': {'needle': [...], 'perovskite': [...]},
            }

        if only one dataset is provided, outer keys don't matter and will be ignored in the legend (e.g. just "ΔF: perovskite - needle" instead of "ΔF: perovskite - needle [DFT]").
        Can say:
            {'data': {'needle': [...], 'perovskite': [...]},
            
        Exactly two inner keys are required per dataset (the difference is
        computed as second_structure - first_structure).

    prop : str
        Property to take the difference of ('F', 'S', 'Cv', …).
    align_at_0K : bool
        If True, shift every curve so that its 0 K value matches the lowest
        0 K value across all datasets. Useful for comparing relative changes
        with temperature without an absolute offset.
    figsize : tuple
        Figure size.
    ylims : tuple or None
        y-axis limits.
    xlims : tuple or None
        x-axis limits.
    plot_in_j_mol : bool
        If True, convert F → kJ/mol and S/Cv → J/K/mol.
    atoms_per_formula_units : int or None
        If provided, scales per-atom values to per-formula-unit.
    colors : list[str]
        One color per dataset (cycles if needed).
    """

    DS_MARKERS = ["o", "^", "D", "s", "v", "P", "X", "h"]
    MARKEVERY  = 10
    MARKERSIZE = 5

    def ds_style(ds_idx):
        if ds_idx == 0:
            return "-", "None", None
        marker = DS_MARKERS[(ds_idx - 1) % len(DS_MARKERS)]
        return "--", marker, MARKEVERY

    deltas   = {}
    temps_by = {}

    for ds_label, structures in thermal_props_dict.items():
        struct_labels = list(structures.keys())
        if len(struct_labels) != 2:
            raise ValueError(
                f"Dataset '{ds_label}' has {len(struct_labels)} structure(s); "
                "exactly 2 are required to compute a difference."
            )
        lbl_a, lbl_b = struct_labels
        data_a, data_b = structures[lbl_a], structures[lbl_b]

        for d, lbl in [(data_a, lbl_a), (data_b, lbl_b)]:
            if prop not in d[0]:
                raise ValueError(
                    f"Property '{prop}' not found in dataset '{ds_label}' / "
                    f"structure '{lbl}'. Available: {list(d[0].keys())}"
                )

        temps  = np.array([pt["T"]  for pt in data_a])
        vals_a = np.array([pt[prop] for pt in data_a])
        vals_b = np.array([pt[prop] for pt in data_b])
        delta  = vals_b - vals_a

        deltas[ds_label]   = delta
        temps_by[ds_label] = temps

    if align_at_0K:
        lowest_0K = min(d[0] for d in deltas.values())
        for ds_label in deltas:
            offset = deltas[ds_label][0] - lowest_0K
            deltas[ds_label] = deltas[ds_label] - offset

    if atoms_per_formula_units is not None:
        for ds_label in deltas:
            deltas[ds_label] = deltas[ds_label] * atoms_per_formula_units

    if plot_in_j_mol:
        factor = EV_TO_KJ_PER_MOL if prop == "F" else EV_TO_J_PER_MOL
        for ds_label in deltas:
            deltas[ds_label] = deltas[ds_label] * factor
        unit = "kJ/mol" if prop == "F" else "J/K/mol"
    else:
        unit = "eV/fu" if atoms_per_formula_units else "eV/atom"

    ylabel = f"Δ{prop} ({unit}{'-fu' if (atoms_per_formula_units and plot_in_j_mol) else ''})"

    plt.figure(figsize=figsize)

    for i, (ds_label, structures) in enumerate(thermal_props_dict.items()):
        struct_labels = list(structures.keys())
        lbl_a, lbl_b  = struct_labels
        legend_label  = (
            f"Δ{prop}: {lbl_b} - {lbl_a}"
            if len(thermal_props_dict) == 1
            else f"{ds_label}"
        )
        color             = colors[i % len(colors)]
        ls, marker, markevery = ds_style(i)

        plt.plot(
            temps_by[ds_label],
            deltas[ds_label],
            color=color,
            linewidth=1.5,
            linestyle=ls,
            marker=marker,
            markevery=markevery,
            markersize=MARKERSIZE,
            label=legend_label,
        )

    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)

    first_label = next(iter(thermal_props_dict))
    delta_ref   = deltas[first_label]
    temps_ref   = temps_by[first_label]

    trans_T = None
    if prop == "F":
        zero_idxs = np.where(np.isclose(delta_ref, 0.0, atol=1e-12))[0]
        if zero_idxs.size > 0:
            trans_T = float(temps_ref[zero_idxs[0]])
        else:
            signs      = np.sign(delta_ref)
            cross_idxs = np.where(np.diff(signs) != 0)[0]
            if cross_idxs.size > 0:
                i      = cross_idxs[0]
                t1, t2 = float(temps_ref[i]), float(temps_ref[i + 1])
                f1, f2 = float(delta_ref[i]), float(delta_ref[i + 1])
                trans_T = t1 - f1 * (t2 - t1) / (f2 - f1) if (f2 - f1) != 0 else (t1 + t2) / 2

    if trans_T is not None:
        ax = plt.gca()
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        plt.axvline(trans_T, color="red", linestyle=":", linewidth=1.2)
        x_off   = 0.03 * (xmax - xmin)
        label_x = trans_T + x_off if (trans_T + x_off) < xmax else trans_T - x_off
        label_y = max(ymin + 0.05 * (ymax - ymin), ymin + 0.25 * (ymax - ymin))
        ha      = "left" if label_x > trans_T else "right"
        plt.text(
            label_x, label_y, f"T_trans = {trans_T:.1f} K",
            color="red", ha=ha, va="center", fontsize=14, backgroundcolor="white",
        )

    plt.title(title)
    plt.xlabel("Temperature (K)")
    plt.ylabel(ylabel)
    if ylims is not None:
        plt.ylim(ylims)
    if xlims is not None:
        plt.xlim(xlims)
    plt.legend(**(legend_kwargs or {'loc': 'best', 'fontsize': 12}))
    plt.tight_layout()
    plt.show()

def plot_phonon_dos_comparison(
    dos_dict,
    plot_in_thz=False,
    title="",
    figsize=(6, 4),
    ylabel_kwargs=None,
    xlabel_kwargs=None,
    legend_kwargs=None,
    ylims=None,
    fill_between=False,
    normalize=True,
):
    """
    Plot phonon density of states for one or more datasets.

    Parameters
    ----------
    dos_dict : dict
        Keys are dataset labels (str), values are lists of dicts with keys 'E' and 'total_dos'.
        Example:
            {
                'DFT':     [{'E': -0.004, 'total_dos': 0.0}, ...],
                'matcalc': [{'E': -0.004, 'total_dos': 0.0}, ...],
            }
    plot_in_thz : bool
        If True, convert frequencies from eV to THz and adjust DOS units accordingly.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    ylims : tuple or None
        y-axis limits.
    fill_between : bool
        If True and more than one dataset is provided, fill the area between each
        dataset and the first dataset to visually highlight differences.
    normalize : bool
        If True and more than one dataset is provided, normalize each DOS to unit area
        before plotting.
    """
    if not dos_dict:
        raise ValueError("dos_dict is empty.")

    h = physical_constants['Planck constant in eV/Hz'][0]
    THz_factor = h * 1e12  # eV → THz conversion factor

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    linestyles = ['-', '--', '--', '--', '--', '--']  # first solid, rest dashed

    multi = len(dos_dict) > 1

    if plot_in_thz:
        ylabel = "DOS (states/THz)" if not (multi and normalize) else "Normalized DOS (arb. units)"
        xlabel = "Frequency (THz)"
    else:
        ylabel = "DOS (states/eV)" if not (multi and normalize) else "Normalized DOS (arb. units)"
        xlabel = "Energy (eV)"

    # Parse all datasets up front
    parsed = {}
    for label, records in dos_dict.items():
        freqs = np.array([r['E'] for r in records])
        dos   = np.array([r['total_dos'] for r in records])
        if plot_in_thz:
            freqs = freqs / THz_factor
            dos   = dos   * THz_factor
        parsed[label] = (freqs, dos)

    plt.figure(figsize=figsize)

    # Reference dataset (first key) for fill_between and shape metrics
    ref_label = next(iter(parsed))
    ref_freqs, ref_dos = parsed[ref_label]
    ref_norm = trapezoid(ref_dos, ref_freqs) if multi and normalize else 1.0
    ref_dos_plot = ref_dos / ref_norm

    shape_metrics = {}

    for i, (label, (freqs, dos)) in enumerate(parsed.items()):
        color = colors[i % len(colors)]
        ls    = linestyles[min(i, len(linestyles) - 1)]

        if multi and normalize:
            norm = trapezoid(dos, freqs)
            dos_plot = dos / norm
            ylabel = "Normalized DOS (arb. units)"
        else:
            dos_plot = dos

        plt.plot(freqs, dos_plot, color=color, linewidth=1.2, linestyle=ls, label=label)

        # Fill between this dataset and the reference (skip the reference itself)
        if fill_between and multi and i > 0:
            common_freq = np.linspace(
                max(ref_freqs[0], freqs[0]),
                min(ref_freqs[-1], freqs[-1]),
                max(len(ref_freqs), len(freqs)),
            )
            interp_ref  = interp1d(ref_freqs, ref_dos_plot, bounds_error=False, fill_value=0.0)
            interp_this = interp1d(freqs, dos_plot,         bounds_error=False, fill_value=0.0)
            dos_ref_c   = interp_ref(common_freq)
            dos_this_c  = interp_this(common_freq)
            plt.fill_between(common_freq, dos_ref_c, dos_this_c,
                             alpha=0.15, color=color, label=f'{ref_label}↔{label} diff')

        # Shape metrics vs reference
        if multi and i > 0:
            common_freq = np.linspace(
                max(ref_freqs[0], freqs[0]),
                min(ref_freqs[-1], freqs[-1]),
                max(len(ref_freqs), len(freqs)),
            )
            interp_ref  = interp1d(ref_freqs, ref_dos_plot, bounds_error=False, fill_value=0.0)
            interp_this = interp1d(freqs, dos_plot,         bounds_error=False, fill_value=0.0)
            dos_ref_c   = interp_ref(common_freq)
            dos_this_c  = interp_this(common_freq)
            shape_metrics[label] = {
                'shape_diff': trapezoid(np.abs(dos_ref_c - dos_this_c), common_freq),
                'cosine_sim': 1 - cosine(dos_ref_c, dos_this_c),
            }

    # Single-dataset shading
    if not multi:
        _, (freqs, dos) = next(iter(parsed.items())), (ref_freqs, ref_dos)
        plt.fill_between(ref_freqs, ref_dos, color='lightblue', alpha=0.5)

    if multi:
        plt.legend(**(legend_kwargs or {}))
        for label, metrics in shape_metrics.items():
            print(f"[{ref_label} vs {label}]  "
                  f"Shape diff (∫|ΔDOS|): {metrics['shape_diff']:.4f}  |  "
                  f"Cosine similarity: {metrics['cosine_sim']:.4f}")

    plt.xlabel(xlabel, **(xlabel_kwargs or {}))
    plt.ylabel(ylabel, **(ylabel_kwargs or {}))
    if ylims is not None:
        plt.ylim(ylims)
    plt.title(title)
    # plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_thermal_properties_comparison(
    thermal_props_dict,
    plot_props=["F", "S"],
    title="",
    figsize=(8, 4),
    plot_in_j_mol=False,
    atoms_per_formula_units=None,
    colors=["blue", "red", "green", "orange", "purple"],
    align_F_to_reference=False,
):
    """
    thermal_props_dict : dict
        Outer keys  → dataset label (shown in legend if more than one, e.g. "DFT", "matcalc").
        Inner keys  → list of dicts with keys 'T', 'F', 'S', 'Cv', …
        Example:
            {
                'DFT':     [{'T': 0, 'F': -1.2, 'S': 0.0, 'Cv': 0.0}, ...],
                'matcalc': [{'T': 0, 'F': -1.0, 'S': 0.0, 'Cv': 0.0}, ...],
            }
    plot_props : list or str
        Which properties to plot. Options: "F" for Helmholtz free energy, "S" for entropy, "Cv" for heat capacity. Default is ["F", "S"].
    title : str
        Title for the plot.
    figsize : tuple
        Figure size.
    plot_in_j_mol : bool
        If True, convert F → kJ/mol and S/Cv → J/K/mol.
    atoms_per_formula_units : int or None
        Number of atoms per formula unit. If provided, scales per-atom to per-formula-unit.
    colors : list[str]
        One color per dataset (cycles if needed).
    align_F_to_reference : bool
        If True, shift F for every non-reference dataset so that its value at
        the first temperature matches the reference (first) dataset. 
    """
    if isinstance(plot_props, str):
        plot_props = [plot_props]
    if not thermal_props_dict:
        raise ValueError("thermal_props_dict is empty.")

    prop_to_group    = {"F": "energy", "S": "thermal", "Cv": "thermal"}
    prop_linestyles  = ["-", "--", "-.", ":"]
    labels           = list(thermal_props_dict.keys())
    datasets         = list(thermal_props_dict.values())
    multi            = len(labels) > 1

    for prop in plot_props:
        for label, data in thermal_props_dict.items():
            if prop not in data[0]:
                raise ValueError(
                    f"Property '{prop}' not found in dataset '{label}'. "
                    f"Available: {list(data[0].keys())}"
                )

    temperatures = [point["T"] for point in datasets[0]]

    def prepare(values, prop):
        vals = list(values)
        if atoms_per_formula_units is not None:
            vals = [v * atoms_per_formula_units for v in vals]
        if plot_in_j_mol:
            factor = EV_TO_KJ_PER_MOL if prop == "F" else EV_TO_J_PER_MOL
            vals = [v * factor for v in vals]
        return vals

    parsed = {
        label: {
            prop: prepare([pt[prop] for pt in data], prop)
            for prop in plot_props
        }
        for label, data in thermal_props_dict.items()
    }

    # ── Align F to reference dataset's starting point ─────────────────────────
    if align_F_to_reference and "F" in plot_props:
        ref_label   = labels[0]
        ref_F_start = parsed[ref_label]["F"][0]
        for label in labels[1:]:
            offset = parsed[label]["F"][0] - ref_F_start
            parsed[label]["F"] = [v - offset for v in parsed[label]["F"]]

    # Axis setup
    fig, ax1 = plt.subplots(figsize=figsize)
    axes = [ax1]

    needed_groups = []
    for prop in plot_props:
        group = prop_to_group.get(prop)
        if group and group not in needed_groups:
            needed_groups.append(group)
    for _ in needed_groups[1:]:
        axes.append(ax1.twinx())

    def ylabel_for(prop, aligned=False):
        if plot_in_j_mol:
            unit = "kJ/mol" if prop == "F" else "J/K/mol"
        else:
            unit = "eV/atom" if prop == "F" else "eV/atom/K"
        if atoms_per_formula_units is not None:
            unit = unit.replace("atom", "fu")
        suffix = " [aligned]" if (aligned and prop == "F") else ""
        return rf"$\mathit{{{prop}}}$ ({unit}){suffix}"

    for prop in plot_props:
        group     = prop_to_group.get(prop, "energy")
        group_idx = needed_groups.index(group) if group in needed_groups else 0
        ax        = axes[group_idx]
        props_on_axis = [p for p in plot_props if prop_to_group.get(p) == group]
        combined_ylabel = " / ".join(
            ylabel_for(p, aligned=align_F_to_reference) for p in props_on_axis
        )
        ax.set_ylabel(combined_ylabel, fontsize=15)
        ax.tick_params(axis="y", labelsize=15)

    DS_MARKERS = ["o", "^", "D", "s", "v", "P", "X", "h"]
    MARKEVERY  = 10
    MARKERSIZE = 5

    def ds_style(ds_idx, single_prop):
        """Return (linestyle, marker, markevery) for a dataset index."""
        if not single_prop or ds_idx == 0:
            return "-", "None", None
        marker = DS_MARKERS[(ds_idx - 1) % len(DS_MARKERS)]
        return "--", marker, MARKEVERY

    single_prop = len(plot_props) == 1

    # Plot
    for prop_idx, prop in enumerate(plot_props):
        group     = prop_to_group.get(prop, "energy")
        group_idx = needed_groups.index(group) if group in needed_groups else 0
        ax        = axes[group_idx]
        ls_prop   = prop_linestyles[prop_idx % len(prop_linestyles)]

        for ds_idx, label in enumerate(labels):
            color        = colors[ds_idx % len(colors)]
            legend_label = (
                rf"$\mathit{{{prop}}}$ ({label})" if multi
                else rf"$\mathit{{{prop}}}$"
            )

            ls, marker, markevery = ds_style(ds_idx, single_prop)
            # When multiple props, fall back to prop-based linestyle
            if not single_prop:
                ls = ls_prop

            ax.plot(
                temperatures,
                parsed[label][prop],
                color=color,
                linewidth=1.5,
                linestyle=ls,
                marker=marker,
                markevery=markevery,
                markersize=MARKERSIZE,
                label=legend_label,
            )

    all_handles, all_labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        all_handles.extend(h)
        all_labels.extend(l)
    if all_handles:
        ax1.legend(all_handles, all_labels, loc="best", fontsize=12)

    ax1.set_xlabel("Temperature (K)", fontsize=15)
    ax1.tick_params(axis="x", labelsize=15)
    plt.title(title)
    plt.tight_layout()
    plt.show()
    


    # def plot_phonon_dos(self, volume=None, remove_imaginary=False):
    #     """
    #     Plot phonon density of states for a specific volume.
    #     Args:
    #         formula (str): Chemical formula of the material.
    #         mpid (str): Materials Project ID.
    #         volume (float): Volume of the structure. If none, will plot phonon dos for all the volumes.
    #         remove_imaginary (bool): Whether to remove imaginary frequencies. Default is False.
    #     """
    #     phonon_dos_dict = self.phonon_dos(remove_imaginary=remove_imaginary)
        
    #     if not volume:
    #         plt.figure(figsize=(10, 6))
    #         for volume in phonon_dos_dict:
    #             phonon_dos = phonon_dos_dict[str(volume)]

    #             frequency_points = np.array([d['E'] for d in phonon_dos['dos']])
    #             total_dos = np.array([d['dos'] for d in phonon_dos['dos']])
    #             label = f"{float(volume):.2f} A^3"
    #             plt.plot(frequency_points, total_dos, label=label)

    #         plt.title(f"Phonon Density of States for {mpid}", fontsize=14)
    #         plt.legend(title="Volumes", loc="best", fontsize=10)

    #     else:
    #         phonon_dos = phonon_dos_dict[str(volume)]

    #         frequency_points = np.array([d['E'] for d in phonon_dos['dos']])
    #         total_dos = np.array([d['dos'] for d in phonon_dos['dos']])

    #         plt.figure(figsize=(10, 6))
    #         plt.plot(frequency_points, total_dos, label=f"{volume:.2f} A^3")
    #         plt.title(f"Phonon Density of States for {volume:.2f} A^3", fontsize=14)
            
    #     plt.xlabel("Energy (eV)", fontsize=12)
    #     plt.ylabel("Phonon DOS (1/eV)", fontsize=12)

    
   
    # def plot_helmholtz_free_energy(self, temp_cutoff=None):
    #     """
    #     Plot Temperature vs Helmholtz Free Energy at Different Volumes.

    #     Args:
    #         F (dict): Dictionary containing Helmholtz free energy data for different volumes.
    #         temp_cutoff (tuple): Optional temperature range (min_temp, max_temp) for filtering.
    #     """

    #     F = self.helmholtz()

    #     plt.figure(figsize=(10, 6))  

    #     volumes = list(F.keys())  

    #     for vol in volumes:
    #         # Extract Helmholtz free energies and temperatures
    #         data = F[vol]['data']
    #         if temp_cutoff:
    #             data = [d for d in data if temp_cutoff[0] <= d['T'] <= temp_cutoff[1]]

    #         Fs = [i['F'] for i in data]
    #         Ts = [i['T'] for i in data]

    #         plt.plot(Ts, Fs, label=f"{float(vol):.2f}")  # Format volume label to 2 decimals

    #     # Add title and axis labels
    #     # plt.title("Temperature vs Helmholtz Free Energy at Different Volumes", fontsize=14)
    #     plt.xlabel("Temperature (K)")
    #     plt.ylabel("Helmholtz Free Energy (eV/atom)" if self.natoms else "Helmholtz Free Energy (eV/cell)")

    #     # Add legend and grid
    #     plt.legend(title="Volumes", loc="best", fontsize=10)


    #     # Display the plot
    #     plt.show()

    # def plot_gibbs_free_energy(self, temp_cutoff=None):
    #     """
    #     Plot Temperature vs Gibbs Free Energy at Different Volumes.

    #     Args:
    #         G (dict): Dictionary containing Gibbs free energy data.
    #         volumes (list): List of volume values.
    #         temp_cutoff (tuple): Optional temperature range (min_temp, max_temp) for filtering.
    #     """
    #     G = self.gibbs()

    #     plt.figure(figsize=(10, 6))  # Create a new figure with a specified size

    #     data = G['data']
    #     if temp_cutoff:
    #         data = [d for d in data if temp_cutoff[0] <= d['T'] <= temp_cutoff[1]]

    #     Gs = [i['G'] for i in data]
    #     Ts = [i['T'] for i in data]

    #     plt.plot(Ts, Gs) 

    #     # Add title and axis labels
    #     # plt.title("Temperature vs Gibbs Free Energy at Different Volumes", fontsize=14)
    #     plt.xlabel("Temperature (K)")
    #     plt.ylabel("Gibbs Free Energy (eV/atom)" if self.natoms else "Gibbs Free Energy (eV/cell)")

    #     # Add legend and grid
    #     # plt.legend(title="Volumes", loc="best", fontsize=10)
    #     # plt.grid(True)

    #     # Display the plot
    #     plt.show()
    
    # def plot_volumes_vs_helmholtz(self, skip=1, temp_cutoff=None, normalize_298K=False):
    #     """
    #     Plot Helmholtz Free Energy vs Volume at different temperatures.
    #     Optionally subtract the energy at 298K for normalization.
    #     """
    #     F = self.helmholtz()
    #     G = self.gibbs()

    #     # Extract all temperature points (assuming consistent across volumes)
    #     temperatures = [entry['T'] for entry in next(iter(F.values()))['data']]
    #     if temp_cutoff:
    #         temperatures = [T for T in temperatures if temp_cutoff[0] <= T <= temp_cutoff[1]]

    #     plt.figure(figsize=(3, 6))  # Prepare the figure

    #     equil_vols = []  # List to store equilibrium volumes
    #     equil_F_values = []  # List to store F value (Gibbs) at equilibrium volume

    #     # Loop over temperatures with the specified skip step
    #     for i, T in enumerate(temperatures[1::skip]): 
    #         vols = []  # List to store volumes
    #         Fs = []  # List to store free energy values
    #         equil_vol = next(item['V'] for item in G['data'] if item['T'] == T)  # Get the equilibrium volume
    #         equil_vol_F = next(item['G'] for item in G['data'] if item['T'] == T)  # Get the Gibbs free energy at the equilibrium volume

    #         equil_vols.append(equil_vol)  # Add equilibrium volumes
    #         equil_F_values.append(equil_vol_F)  # Add equilibrium Gibbs free energies

    #         for vol, data in F.items():
    #             for entry in data['data']:
    #                 if entry['T'] == T:
    #                     F_value = entry['F']
    #                     if normalize_298K:
    #                         G_at_T300 = next(item['G'] for item in G['data'] if item['T'] == 300)
    #                         F_value -= G_at_T300
    #                     vols.append(float(vol))
    #                     Fs.append(F_value)
    #                     break

    #         # Plot Helmholtz Free Energy vs Volume for the current temperature
    #         plt.scatter(vols, Fs, marker='o', color='black')
    #                 # Plot the smooth fitted line for this temperature
    #         if T in G['fitted_F_values']:
    #             plt.plot(G['volumes_for_fitting'], G['fitted_F_values'][T], color='black')


    #     print("Equilibrium Volumes: ", equil_vols)
    #     print("Equilibrium Free Energy Values: ", equil_F_values)

    #     # Now plot the equilibrium volumes and Gibbs free energies at each temperature
    #     plt.plot(equil_vols, equil_F_values, color='red', marker='x', label="Equilibrium Volume")

    #     # # Add legend for first and last temperature
    #     # plt.text(484, -30.8, f"T = {temperatures[1::skip][0]} K", fontsize=14,
    #     #          color='black', ha='center', va='center')
    #     # plt.text(448, -37.8, f"T = {temperatures[1::skip][-1]} K", fontsize=14,
    #     #          color='black', ha='center', va='center')

    #     plt.xlabel("Volume ($\mathrm{Å}^3$)", fontsize=18)
    #     plt.ylabel("F (eV/f.u.)", fontsize=18)
    #     plt.yticks(fontsize=14)
    #     plt.xticks(fontsize=14)
    #     # plt.title(
    #     #     "Helmholtz Free Energy vs Volume at Different Temperatures" +
    #     #     (" (Normalized to 298K)" if normalize_298K else ""),
    #     #     fontsize=14
    #     # )

    #     # Show the plot
    #     plt.show()


    # def plot_equilibrium_volume_vs_temperature(self, temp_cutoff=None):
    #     """
    #     Plot Equilibrium Volume vs Temperature.

    #     Args:
    #         G (dict): Dictionary containing Gibbs free energy data.
    #         volumes (list): List of volume values.
    #         temp_cutoff (tuple): Optional temperature range (min_temp, max_temp) for filtering.
    #     """
    #     G = self.gibbs() 

    #     plt.figure(figsize=(2, 6))

    #     data = G['data']
    #     if temp_cutoff:
    #         data = [d for d in data if temp_cutoff[0] <= d['T'] <= temp_cutoff[1]]
        
    #     Ts = [i['T'] for i in data]
    #     Vs = [i['V'] for i in data]

    #     plt.plot(Ts, Vs, label=mpid)
    #     # plt.title("Equilibrium Volume vs Temperature", fontsize=14)
    #     plt.xlabel("Temperature (K)", fontsize=12)
    #     plt.ylabel("Equilibrium Volume ($\mathrm{Å}^3$)", fontsize=12)
    #     plt.xticks(fontsize=12)
    #     plt.yticks(fontsize=12)
    #     # plt.legend(loc="best", fontsize=10)

