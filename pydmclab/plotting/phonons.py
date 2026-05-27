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
    qpoints = bs_dict['qpoints']
    frequencies = bs_dict['frequencies']

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

def plot_phonon_dos(dos_dict, plot_in_thz=False, title="", figsize=(6,4), ylims=None):
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


def plot_thermal_properties(thermal_props, plot_props=["F", "S"], title="", figsize=(8,4), plot_in_j_mol=False, atoms_per_formula_units=None):
    """
    Plot thermal properties (Helmholtz free energy, entropy, heat capacity).

    Properties with the same units share the same axis.
    Properties with different units get separate axes (e.g., F on left, S/Cv on right).

    Args:
        thermal_props (dict):
            {'data': 
                [{'T': float, 
                'F': float, 
                'S': float}, ...]}
        plot_props (list or str):
            Which properties to plot. Options: "F" for Helmholtz free energy, "S" for entropy, "Cv" for heat capacity. Default is ["F", "S"] to plot both.
        title (str):
            Title for the plot.
        figsize : tuple
            Figure size
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
            if formula_units is not None:
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
    if not thermal_props_dict:
        raise ValueError("thermal_props_dict is empty.")

    linestyles = ["-", "--", "-.", ":"]  # first dataset solid, rest dashed

    # ── Parse & compute Δprop for every dataset ──────────────────────────────
    deltas   = {}   # label → np.ndarray of Δprop (raw eV/atom)
    temps_by = {}   # label → np.ndarray of temperatures

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

        temps   = np.array([pt["T"]    for pt in data_a])
        vals_a  = np.array([pt[prop]   for pt in data_a])
        vals_b  = np.array([pt[prop]   for pt in data_b])
        delta   = vals_b - vals_a          # second − first

        deltas[ds_label]   = delta
        temps_by[ds_label] = temps

    # ── Optional 0 K alignment ───────────────────────────────────────────────
    if align_at_0K:
        lowest_0K = min(d[0] for d in deltas.values())
        for ds_label in deltas:
            offset = deltas[ds_label][0] - lowest_0K
            deltas[ds_label] = deltas[ds_label] - offset

    # ── Unit scaling ─────────────────────────────────────────────────────────
    if atoms_per_formula_units is not None:
        for ds_label in deltas:
            deltas[ds_label] = deltas[ds_label] * atoms_per_formula_units

    if plot_in_j_mol:
        factor = EV_TO_KJ_PER_MOL if prop == "F" else EV_TO_J_PER_MOL
        for ds_label in deltas:
            deltas[ds_label] = deltas[ds_label] * factor
        unit   = "kJ/mol" if prop == "F" else "J/K/mol"
    else:
        unit   = "eV/fu" if atoms_per_formula_units else "eV/atom"

    ylabel = f"Δ{prop} ({unit}{'-fu' if (atoms_per_formula_units and plot_in_j_mol) else ''})"

    # ── Plot ─────────────────────────────────────────────────────────────────
    plt.figure(figsize=figsize)

    for i, (ds_label, structures) in enumerate(thermal_props_dict.items()):
        struct_labels  = list(structures.keys())
        lbl_a, lbl_b   = struct_labels
        legend_label = (
                        f"Δ{prop}: {lbl_b} - {lbl_a}"
                        if len(thermal_props_dict) == 1
                        else f"Δ{prop}: {lbl_b} - {lbl_a}  [{ds_label}]"
                    )
        color          = colors[i % len(colors)]
        ls             = linestyles[i % len(linestyles)]

        plt.plot(
            temps_by[ds_label],
            deltas[ds_label],
            color=color,
            linewidth=1.5,
            linestyle=ls,
            label=legend_label,
        )

    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)

    # ── Transition temperature: detect for the FIRST dataset only ────────────
    first_label = next(iter(thermal_props_dict))
    delta_ref   = deltas[first_label]
    temps_ref   = temps_by[first_label]

    trans_T = None
    zero_idxs = np.where(np.isclose(delta_ref, 0.0, atol=1e-12))[0]
    if zero_idxs.size > 0:
        trans_T = float(temps_ref[zero_idxs[0]])
    else:
        signs      = np.sign(delta_ref)
        cross_idxs = np.where(np.diff(signs) != 0)[0]
        if cross_idxs.size > 0:
            i       = cross_idxs[0]
            t1, t2  = float(temps_ref[i]), float(temps_ref[i + 1])
            f1, f2  = float(delta_ref[i]), float(delta_ref[i + 1])
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
            color="red", ha=ha, va="center", fontsize=9, backgroundcolor="white",
        )

    plt.xlabel("Temperature (K)")
    plt.ylabel(ylabel)
    if ylims is not None:
        plt.ylim(ylims)
    if xlims is not None:
        plt.xlim(xlims)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_phonon_dos_comparison(
    dos_dict,
    plot_in_thz=False,
    title="",
    figsize=(6, 4),
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
        plt.legend()
        for label, metrics in shape_metrics.items():
            print(f"[{ref_label} vs {label}]  "
                  f"Shape diff (∫|ΔDOS|): {metrics['shape_diff']:.4f}  |  "
                  f"Cosine similarity: {metrics['cosine_sim']:.4f}")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ylims is not None:
        plt.ylim(ylims)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def compare_thermal_properties(
    thermal_props_dict,
    plot_props=["F", "S"],
    title="",
    figsize=(8, 4),
    plot_in_j_mol=False,
    atoms_per_formula_units=None,
    colors=["blue", "red", "green", "orange", "purple"],
):
    """
    Plot thermal properties for one or more datasets.

    Parameters
    ----------
    thermal_props_dict : dict
        Keys are dataset labels (str); values are lists of dicts with keys
        'T', 'F', 'S', and/or 'Cv'.
        Example:
            {
                'DFT':     [{'T': 0, 'F': -1.2, 'S': 0.0}, ...],
                'matcalc': [{'T': 0, 'F': -1.1, 'S': 0.0}, ...],
            }
    plot_props : list[str] or str
        Which properties to plot. Options: "F", "S", "Cv". Default ["F", "S"].
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    plot_in_j_mol : bool
        If True, convert F → kJ/mol and S/Cv → J/K/mol.
    atoms_per_formula_units : int or None
        If provided, scales all values from per-atom to per-formula-unit.
    colors : list[str]
        One color per dataset (cycles if more datasets than colors).
    """
    if isinstance(plot_props, str):
        plot_props = [plot_props]
    if not thermal_props_dict:
        raise ValueError("thermal_props_dict is empty.")

    # Unit-group logic: properties that share axes
    prop_to_group = {"F": "energy", "S": "thermal", "Cv": "thermal"}

    labels   = list(thermal_props_dict.keys())
    datasets = list(thermal_props_dict.values())
    linestyles = ["-"] + ["--"] * (len(labels) - 1)   # first solid, rest dashed

    # Validate that requested props exist in every dataset
    for prop in plot_props:
        for label, data in thermal_props_dict.items():
            if prop not in data[0]:
                raise ValueError(
                    f"Property '{prop}' not found in dataset '{label}'. "
                    f"Available: {list(data[0].keys())}"
                )

    # Use temperatures from the first dataset for x-axis
    temperatures = [point["T"] for point in datasets[0]]

    # Build per-dataset, per-property value arrays (scaled + converted)
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

    # Axis creation
    fig, ax1 = plt.subplots(figsize=figsize)
    axes = [ax1]

    needed_groups = []
    for prop in plot_props:
        group = prop_to_group.get(prop)
        if group and group not in needed_groups:
            needed_groups.append(group)

    # Create twin axes for additional unit groups
    for _ in needed_groups[1:]:
        axes.append(ax1.twinx())

    def ylabel_for(prop):
        if plot_in_j_mol:
            unit = "kJ/mol" if prop == "F" else "J/K/mol"
        else:
            unit = "eV/atom" if prop == "F" else "eV/atom/K"
        if atoms_per_formula_units is not None:
            unit = unit.replace("atom", "fu")
        return rf"$\mathit{{{prop}}}$ ({unit})"

    # Plot
    prop_color_idx = 0   # one color per property across all datasets

    for prop in plot_props:
        group     = prop_to_group.get(prop, "energy")
        group_idx = needed_groups.index(group) if group in needed_groups else 0
        ax        = axes[group_idx]
        ax.set_ylabel(ylabel_for(prop), fontsize=15)
        ax.tick_params(axis="y", labelsize=15)

        prop_color = colors[prop_color_idx % len(colors)]
        prop_color_idx += 1

        for i, label in enumerate(labels):
            ls = linestyles[i]
            ax.plot(
                temperatures,
                parsed[label][prop],
                color=prop_color,
                linewidth=1.5,
                linestyle=ls,
                label=rf"$\mathit{{{prop}}}$ ({label})",
            )

    # Combined legend on primary axis
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

def main():
    plot_in_j_mol = False
    # atoms_per_formula_units = 5 # number of atoms per formula unit
    atoms_per_formula_units = None
    xlims = (0, 2200)

    r =read_json(os.path.join(DATA_DIR, 'relaxation_results.json'))
    phonons = read_json(os.path.join(DATA_DIR, 'phonons.json'))

    high_symm_points = [['Γ', 'X'],
                        ['X', 'S'],
                        ['S', 'Y'],
                        ['Y', 'Γ'],
                        ['Γ', 'Z'],
                        ['Z', 'U'],
                        ['U', 'R'],
                        ['R', 'T'],
                        ['T', 'Z']]

    dft_phonons = read_json(os.path.join(DATA_DIR, '../251124/phonons.json'))
    dft_dos = {}
    dft_tprops = {}
    for key in dft_phonons:
        calc_type = key.split('--')[-1]
        if calc_type == 'metagga-static':
            continue

        mpid = key.split('--')[1]
        dos_data = dft_phonons[key]['phonons']['total_dos']['total_dos']
        energies = [point['E'] for point in dos_data]
        dft_dos[mpid] = {'dos': np.array([point['total_dos'] for point in dos_data]), 'energies': np.array(energies)}
        dft_tprops[mpid] = dft_phonons[key]['phonons']['helmholtz']['data']

    tprops = {}
    for key in phonons:
        mpid = key
        # bs_qpoints = phonons[key]['band_structure']['qpoints']
        # bs_frequencies = phonons[key]['band_structure']['frequencies']
        # # high_symm_points = phonons[key]['band_structure']['path']
        # labels = [i[0] for i in high_symm_points] + [high_symm_points[-1][-1]]
        # plot_phonon_bandstructure(np.array(bs_qpoints), np.array(bs_frequencies), labels=labels, title=mpid)
        # # forces = phonons[key]['forces']
        # # plot_forces_distribution(forces, mpid)
        # dos_data = phonons[key]['total_dos']['total_dos']
        # natoms = len(r[key]['final_structure']['sites'])
        # print(f"Plotting DOS for {mpid} with {natoms} atoms in the cell")
        # tdos = np.array([point['total_dos'] for point in dos_data])
        # energies = [point['E'] for point in dos_data]
        # ylims = (0, 50000) if 'needle' in mpid else (0, 50000)
        # mpid = mpid.replace('dist_perovskite', 'perovskite')
        # plot_phonon_dos(np.array(energies), np.array(tdos), title=mpid, ylims=ylims)
        # ylims = (0, 80) if 'needle' in mpid else (0, 80)
        # dos2 = dft_dos[key]['dos']
        # energies2 = dft_dos[key]['energies']
        # plot_phonon_dos_comparison(np.array(energies), np.array(tdos), frequencies2=np.array(energies2), dos2=np.array(dos2), label1="TensorNet", label2="DFT", title=mpid, ylims=ylims)
        # # ylims = (0, 300) if 'needle' in mpid else (0, 90)
        # # plot_phonon_dos(np.array(energies), np.array(tdos), plot_in_thz=True, title=mpid, ylims=ylims)
        #Need to fix plot phonon_dos
        F = phonons[key]['thermal_properties']
        # plot_thermal_properties(F, plot_props=["F", "S"], title=mpid, plot_in_j_mol=plot_in_j_mol, formula_units=formula_units)
        tprops[mpid] = F

    key_order = ['S3Sr1Zr1_needle', 'S3Sr1Zr1_dist_perovskite']
    tprops = {k: tprops[k] for k in key_order}
    dft_tprops = {k: dft_tprops[k] for k in key_order}

    props_to_plot = ['F', 'S', 'Cv']
    for prop in props_to_plot:
        plot_relative_prop(dft_tprops,
                           tprops,
                        data_set_labels=["DFT", "TensorNet"],
                        prop=prop, 
                        align_Fs=True,
                        plot_in_j_mol=plot_in_j_mol, 
                        formula_units=formula_units, 
                        xlims=xlims)
        
    # dft_phonons = read_json(os.path.join(DATA_DIR, '../251124/phonons.json'))
    # dft_tprops = {}
    # for key in dft_phonons:
    #     calc_type = key.split('--')[-1]
    #     if calc_type == 'metagga-static':
    #         continue
    #     mpid = key.split('--')[1]
    #     F = dft_phonons[key]['phonons']['helmholtz']['data']
    #     dft_tprops[mpid] = F

    # for prop in props_to_plot:
    #     plot_relative_prop(dft_tprops['S3Sr1Zr1_needle'], 
    #                     prop_needle, 
    #                     prop=prop, 
    #                     label1=rf'$\mathit{{{prop}}}_\mathrm{{needle}}$' + ' (DFT)', 
    #                     label2=rf'$\mathit{{{prop}}}_\mathrm{{needle}}$' + ' (TensorNet)', 
    #                     plot_in_j_mol=plot_in_j_mol, 
    #                     formula_units=formula_units, 
    #                     xlims=xlims)
        
    #     plot_relative_prop(dft_tprops['S3Sr1Zr1_dist_perovskite'], 
    #                     prop_perovskite, 
    #                     prop=prop, 
    #                     label1=rf'$\mathit{{{prop}}}_\mathrm{{perovskite}}$' + ' (DFT)', 
    #                     label2=rf'$\mathit{{{prop}}}_\mathrm{{perovskite}}$' + ' (TensorNet)', 
    #                     plot_in_j_mol=plot_in_j_mol, 
    #                     formula_units=formula_units, 
    #                     xlims=xlims)

    # for prop in ['F', 'S']:
    #     compare_thermal_properties(dft_tprops['S3Sr1Zr1_needle'], 
    #                             prop_needle, 
    #                             plot_props=prop, 
    #                             title='Needle', 
    #                             plot_in_j_mol=plot_in_j_mol, 
    #                             formula_units=formula_units,
    #                             figsize=(6,4),
    #                             colors=['blue'] if prop == 'F' else ['red'])
        
    #     compare_thermal_properties(dft_tprops['S3Sr1Zr1_dist_perovskite'], 
    #                             prop_perovskite, 
    #                             plot_props=prop, 
    #                             title='Perovskite', 
    #                             plot_in_j_mol=plot_in_j_mol, 
    #                             formula_units=formula_units,
    #                             figsize=(6,4),
    #                             colors=['blue'] if prop == 'F' else ['red'])