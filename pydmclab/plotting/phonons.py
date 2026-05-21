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

def plot_phonon_bandstructure(qpoints, frequencies, labels=None, 
                              ylabel="Energy (eV)", title="", figsize=(8, 6)):
    """
    Plot a phonon band structure from Phonopy-style qpoints and frequencies.

    Parameters
    ----------
    qpoints : ndarray, shape (npaths, npoints, 3)
        Q-points for each path in reciprocal space
    frequencies : ndarray, shape (npaths, npoints, nbranches)
        Phonon frequencies for each q-point and branch
    labels : list of str or None
        High symmetry point labels. Must have length = npaths + 1.
        Example: ["Γ", "X", "K", "Γ", "L"] for 4 paths
        Note: The middle labels connect paths (end of one = start of next)
    ylabel : str
        Label for y-axis
    figsize : tuple
        Figure size
    """
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

def plot_phonon_dos(frequencies, dos, plot_in_thz=False, title="", figsize=(6,4), ylims=None):
    """
    Plot phonon density of states.

    Parameters
    ----------
    frequencies : ndarray, shape (npoints,)
        Frequency values in eV
    dos : ndarray, shape (npoints,)
        Density of states values in states/eV
    ylabel : str
        Label for y-axis
    figsize : tuple
        Figure size
    """
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


def plot_relative_prop(thermal_props_dict, 
                       prop="F", align_Fs=False, 
                       figsize=(6,4), 
                       ylims=None, 
                       xlims=None, plot_in_j_mol=False, 
                       atoms_per_formula_units=None,
                       thermal_props_dict2=None, 
                       data_set_labels=["DFT", "TensorNet"]):
    """
    Plot relative property between two structures. The property is plotted as the difference (structure 2 - structure 1) vs temperature.
    If second set of thermal properties is provided, it will be plotted on the same graph for comparison. (e.g., comparing DFT vs TensorNet for both needle and dist_perovskite structures)

    Parameters
    ----------
    thermal_props_dict : dict
        Dictionary containing thermal properties for both structures
        {"label1": [{'T': float, 'F': float, 'S': float, ...}, ...],
         "label2": [{'T': float, 'F': float, 'S': float, ...}, ...]}
    thermal_props_dict2 : dict or None
        Dictionary containing thermal properties for a second set of structures (optional)
        Must be the same size array over the same temperatures as thermal_props_dict
    prop : str
        Property to plot
    figsize : tuple
        Figure size
    plot_in_j_mol : bool
        If True, plot in kJ/mol; if False, plot in eV/atom
    atoms_per_formula_units : int or None
        Number of atoms per formula unit. If provided, scales per-atom to per-formula-unit.
    """
    # Ensure numpy arrays for numeric operations
    label1, label2 = list(thermal_props_dict.keys())
    thermal_props1 = thermal_props_dict[label1]
    thermal_props2 = thermal_props_dict[label2]
    temps = np.array([point['T'] for point in thermal_props1])
    prop_struc1_vals = np.array([point[prop] for point in thermal_props1])
    prop_struc2_vals = np.array([point[prop] for point in thermal_props2])
    delta_prop = prop_struc2_vals - prop_struc1_vals

    if thermal_props_dict2 is not None:
        label3, label4 = list(thermal_props_dict2.keys())
        thermal_props3 = thermal_props_dict2[label3]
        thermal_props4 = thermal_props_dict2[label4]
        prop_struc3_vals = np.array([point[prop] for point in thermal_props3])
        prop_struc4_vals = np.array([point[prop] for point in thermal_props4])
        delta_prop2 = prop_struc4_vals - prop_struc3_vals
        if align_Fs:
            # Align the F values of the two datasets by substacting the difference at 0K (or lowest T) from the entire curve. This way we can compare relative changes with temperature without an absolute offset.
            data_set1_0K_val = delta_prop[0]
            data_set2_0K_val = delta_prop2[0]
            lower_0K = min(data_set1_0K_val, data_set2_0K_val)
            #shift the one that has a larger value at 0K down to match the lower one
            if data_set1_0K_val > data_set2_0K_val:
                delta_prop = delta_prop - (data_set1_0K_val - lower_0K)
            elif data_set2_0K_val > data_set1_0K_val:
                delta_prop2 = delta_prop2 - (data_set2_0K_val - lower_0K)

    if atoms_per_formula_units is not None:
        delta_prop = delta_prop * atoms_per_formula_units

    if plot_in_j_mol:
        delta_prop_plot = delta_prop * EV_TO_KJ_PER_MOL if prop == "F" else delta_prop * EV_TO_J_PER_MOL
        ylabel = f"Δ{prop} (kJ/mol{'-fu' if atoms_per_formula_units else ''})"
    else:
        delta_prop_plot = delta_prop
        ylabel = f"Δ{prop} (eV/{'fu' if atoms_per_formula_units else 'atom'})"

    plt.figure(figsize=figsize)
    plt.plot(temps, delta_prop_plot, color='green', linewidth=1.5, label=f"Δ{prop}: {label2} - {label1} - {data_set_labels[0]}")
    if thermal_props_dict2 is not None:
        plt.plot(temps, delta_prop2, color='green', linewidth=1.5, linestyle='--', label=f"Δ{prop}: {label4} - {label3} - {data_set_labels[1]}")
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)

    # Detect first zero-crossing (sign change) and mark it
    trans_T = None
    # Look for sign changes between adjacent points
    signs = np.sign(delta_prop)
    # Consider small values as zero for robustness
    zero_idxs = np.where(np.isclose(delta_prop, 0.0, atol=1e-12))[0]
    if zero_idxs.size > 0:
        # exact zero exists — take first occurrence
        trans_T = float(temps[zero_idxs[0]])
    else:
        cross_idxs = np.where(np.diff(signs) != 0)[0]
        if cross_idxs.size > 0:
            i = cross_idxs[0]
            # linear interpolation between (temps[i], delta_prop[i]) and (temps[i+1], delta_prop[i+1])
            t1, t2 = float(temps[i]), float(temps[i+1])
            f1, f2 = float(delta_prop[i]), float(delta_prop[i+1])
            if (f2 - f1) != 0:
                trans_T = t1 - f1 * (t2 - t1) / (f2 - f1)
            else:
                trans_T = float((t1 + t2) / 2.0)

    if trans_T is not None:
        ax = plt.gca()
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        # draw vertical red dotted line at transition temperature
        plt.axvline(trans_T, color='red', linestyle=':', linewidth=1.2)
        # label the transition temperature to the side of the line
        label_text = f"T_trans = {trans_T:.1f} K"
        # horizontal offset as fraction of axis width
        x_off = 0.03 * (xmax - xmin)
        # prefer placing label to the right; if too close to right edge, place to left
        label_x = trans_T + x_off if (trans_T + x_off) < xmax else trans_T - x_off
        # vertical position: scooch label down (about 25% above bottom)
        label_y = ymin + 0.25 * (ymax - ymin)
        # clamp so label isn't too close to the axis bottom
        min_y = ymin + 0.05 * (ymax - ymin)
        if label_y < min_y:
            label_y = min_y
        ha = 'left' if label_x > trans_T else 'right'
        plt.text(label_x, label_y, label_text, color='red', ha=ha, va='center', fontsize=9, backgroundcolor='white')

    plt.xlabel("Temperature (K)")
    plt.ylabel(ylabel)
    # plt.title("Relative Helmholtz Free Energy")
    # plt.grid(alpha=0.3)
    if ylims is not None:
        plt.ylim(ylims)
    if xlims is not None:
        plt.xlim(xlims)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_phonon_dos_comparison(frequencies, dos, frequencies2=None, dos2=None, label1="Dataset 1", label2="Dataset 2",
                   plot_in_thz=False, title="", figsize=(6,4), ylims=None, fill_between=False):
    """
    Plot phonon density of states. If dos2 is provided, both are normalized to unit area and plotted together for comparison. (Like if you want to compare the DOS from two different calculation methods)
    Parameters    
    ----------
    frequencies : ndarray, shape (npoints,)
        Frequency values for dos in eV
    dos : ndarray, shape (npoints,)
        Density of states values for dos in states/eV
    frequencies2 : ndarray or None
        Frequency values for dos2 in eV. Must be the same length as dos2 if dos2 is provided.
    dos2 : ndarray or None
        Density of states values for dos2 in states/eV. Must be the same length as frequencies2 if provided.
    label1, label2 : str
        Labels for the two datasets (used when dos2 is provided)
    plot_in_thz : bool
        If True, convert frequencies from eV to THz and adjust DOS units accordingly
    title : str
        Plot title
    figsize : tuple
        Figure size
    ylims : tuple or None
        y-axis limits
    fill_between : bool
        If True and dos2 is provided, fill the area between the two DOS curves to visually highlight differences.
    """

    if plot_in_thz:
        h = physical_constants['Planck constant in eV/Hz'][0]
        frequencies = frequencies / (h * 1e12)
        dos = dos * (h * 1e12)
        if dos2 is not None:
            frequencies2 = frequencies2 / (h * 1e12)
            dos2 = dos2 * (h * 1e12)
        ylabel = "DOS (states/THz)"
        xlabel = "Frequency (THz)"
    else:
        ylabel = "DOS (states/eV)"
        xlabel = "Energy (eV)"

    plt.figure(figsize=figsize)

    if dos2 is not None and frequencies2 is not None:
        # Normalize both to unit area on their own grids
        norm1 = trapezoid(dos, frequencies)
        norm2 = trapezoid(dos2, frequencies2)
        dos1_norm = dos / norm1
        dos2_norm = dos2 / norm2

        # Build a common grid spanning both frequency ranges
        common_freq = np.linspace(
            max(frequencies[0], frequencies2[0]),
            min(frequencies[-1], frequencies2[-1]),
            max(len(frequencies), len(frequencies2))
        )

        # Interpolate both onto the common grid for fill_between
        interp1 = interp1d(frequencies, dos1_norm, bounds_error=False, fill_value=0.0)
        interp2 = interp1d(frequencies2, dos2_norm, bounds_error=False, fill_value=0.0)
        dos1_common = interp1(common_freq)
        dos2_common = interp2(common_freq)

        # Plot each on its own original grid
        plt.plot(frequencies, dos1_norm, color='blue', linewidth=1.2, label=label1)
        plt.plot(frequencies2, dos2_norm, color='red', linewidth=1.2, linestyle='--', label=label2)

        if fill_between:
            # Fill on the common grid
            plt.fill_between(common_freq, dos1_common, dos2_common, alpha=0.2, color='gray', label='difference')
        plt.legend()
        ylabel = "Normalized DOS (arb. units)"

        # Shape metrics on common grid
        shape_diff = trapezoid(np.abs(dos1_common - dos2_common), common_freq)
        cos_sim = 1 - cosine(dos1_common, dos2_common)
        print(f"Shape difference (integral of |ΔDOS|): {shape_diff:.4f}")
        print(f"Cosine similarity: {cos_sim:.4f}")

    else:
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

def compare_thermal_properties(thermal_props1, 
                               thermal_props2, 
                               plot_props=["F", "S"], 
                               title="", figsize=(8,4), 
                               plot_in_j_mol=False, 
                               atoms_per_formula_units=None, 
                               colors=['blue', 'red', 'green', 'orange', 'purple']):
    """
    Plot thermal properties (Helmholtz free energy, entropy, heat capacity).

    Properties with the same units share the same axis.
    Properties with different units get separate axes (e.g., F on left, S/Cv on right).

    Args:
        thermal_props1 (dict):
            OJO: Right now this is hard coded to assume dataset 1 is DFT and dataset 2 is Foundation potential.
            {'data': 
                [{'temperature': float, 
                'helmholtz_free_energy': float, 
                'entropy': float}, ...]}
        thermal_props2 (dict):
            {'data': 
                [{'temperature': float, 
                'helmholtz_free_energy': float, 
                'entropy': float}, ...]}
        plot_props (list or str):
            Which properties to plot. Options: "F" for Helmholtz free energy, "S" for entropy, "Cv" for heat capacity. Default is ["F", "S"] to plot both.
        title (str):
            Title for the plot.
        figsize : tuple
            Figure size
        formula_units : int or None
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
    temperatures = [point['T'] for point in thermal_props1]
    props1 = {prop: [point[prop] for point in thermal_props1] for prop in plot_props}
    props2 = {prop: [point[prop] for point in thermal_props2] for prop in plot_props}

    # Create figure and axes
    fig, ax1 = plt.subplots(figsize=figsize)
    axes = [ax1]
    ax = ax1
    current_side = "left"

    # Plot each property, creating new axes for different unit groups
    color_idx = 0

    # Check which groups are needed
    needed_groups = []

    for prop in plot_props:
        if prop not in thermal_props1[0] or prop not in thermal_props2[0]:
            raise ValueError(f"Property '{prop}' not found in one or both thermal_props. Available properties: {list(thermal_props1[0].keys())}")
        group = prop_to_group.get(prop)
        if group and group not in needed_groups:
            needed_groups.append(group)

        # Scale by formula_units if provided
        if atoms_per_formula_units is not None:
                props1[prop] = [val * atoms_per_formula_units for val in props1[prop]]
                props2[prop] = [val * atoms_per_formula_units for val in props2[prop]]

        # Convert units if requested
        if plot_in_j_mol:
                if prop == "F":
                    props1[prop] = [val * EV_TO_KJ_PER_MOL for val in props1[prop]]
                    props2[prop] = [val * EV_TO_KJ_PER_MOL for val in props2[prop]]
                elif prop == "S" or prop == "Cv":
                    props1[prop] = [val * EV_TO_J_PER_MOL for val in props1[prop]]
                    props2[prop] = [val * EV_TO_J_PER_MOL for val in props2[prop]]

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
            ylabel = f"{prop} ({'kJ/mol' if prop == 'F' else 'J/K/mol'}{'-fu' if atoms_per_formula_units else ''})"
        else:
            ylabel = f"{prop} (eV/atom)" if prop == "F" else f"{prop} (eV/atom/K)"
            if atoms_per_formula_units is not None:
                ylabel = ylabel.replace("atom", "fu")

        ax.set_ylabel(ylabel, fontsize=15)
        ax.tick_params(axis='y', labelsize=15)
        ax.plot(temperatures, props1[prop], color=colors[color_idx], linewidth=1.5, label=rf'$\mathit{{{prop}}}$ (DFT)') 
        ax.plot(temperatures, props2[prop], color=colors[color_idx], linewidth=1.5, linestyle='--', label=rf'$\mathit{{{prop}}}$ (TensorNet)')
        color_idx += 1

    handles_labels = []
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for h, l in zip(handles, labels):
            handles_labels.append((h, l))

    if handles_labels:
        handles, labels = zip(*handles_labels)
        ax1.legend(handles, labels, loc='best', fontsize=12)

    # Set x-axis on the primary axis
    ax1.set_xlabel("Temperature (K)", fontsize=15)
    ax1.tick_params(axis='x', labelsize=15)
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