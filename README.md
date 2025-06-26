# PyDMCLab

A Python package for computational materials science research, developed and utilized by the DMC Lab.

## Overview

PyDMCLab provides tools for materials design, analysis, and high-throughput calculations. It relies on various existing computational materials science tools (see `Components` below) and provides an interface for:

- Structure manipulation and analysis
- Phase stability and convex hull construction
- Electronic structure analysis
- Defect modeling and analysis
- Machine learning potentials for molecular dynamics
- High-performance computing workflow management

## Installation

```bash
# Clone the repository
git clone https://github.com/bartel-group/pydmclab.git

# Change to the repository directory
cd pydmclab

# Install the package
pip install .
```

## Components

### Core Modules

- **[`comp`](pydmclab/core/comp.py)**: Composition analysis tools including formula normalization, element stoichiometry calculation, molecular fraction calculation, chemical system handling, and oxidation state prediction
- **[`struc`](pydmclab/core/struc.py)**: Structure manipulation including structure creation, transformation, symmetry analysis, site manipulation, solid solution generation, and slab generation for surface studies
- **[`energies`](pydmclab/core/energies.py)**: Energy calculation tools for formation energies, reaction energies, chemical potentials, thermodynamic analysis, and defect formation energy calculations
- **[`hulls`](pydmclab/core/hulls.py)**: Phase stability analysis including convex hull construction, decomposition energy calculation, stability prediction, mixing energy calculation, and parallel hull analysis for large systems
- **[`mag`](pydmclab/core/mag.py)**: Magnetic property analysis tools for generating and analyzing different magnetic configurations and predicting magnetic ordering
- **[`query`](pydmclab/core/query.py)**: Materials database interfaces, primarily for the Materials Project API, supporting various query types (composition, chemical systems, structure, and properties)
- **[`defects`](pydmclab/core/defects.py)**: Tools for generating and analyzing defect structures including substitutions, vacancies, and interstitials, with support for structure distortion
- **[`alloys`](pydmclab/core/alloys.py)**: Alloy modeling with regular solution models for isomorphic and heterogeneous alloys, phase diagram construction, and thermodynamic property calculation

@todo: Add citation information for core modules

### HPC Tools

*Disclaimer*: Such tools are primarily designed to interface with high-performance computing resources at The University of Minnesota and may not be suitable for all environments.

- **[`analyze.py`](pydmclab/hpc/analyze.py)**: Analyze VASP output files to extract energies, structures, DOS, bonding data, and convergence results
- **[`collector.py`](pydmclab/hpc/collector.py)**: Collect and aggregate results from multiple calculations for comparative analysis
- **[`helpers.py`](pydmclab/hpc/helpers.py)**: Utility functions for configuration, submission, and results processing
- **[`launch.py`](pydmclab/hpc/launch.py)**: Tools for launching batches of VASP calculations across multiple compositions or structures
- **[`passer.py`](pydmclab/hpc/passer.py)**: Utilities for passing data between different calculation steps
- **[`phonons.py`](pydmclab/hpc/phonons.py)**: Setup and analysis of phonon calculations for vibrational properties
- **[`sets.py`](pydmclab/hpc/sets.py)**: VASP input set creation and management with customizable parameters
- **[`submit.py`](pydmclab/hpc/submit.py)**: Tools for preparing and submitting VASP calculations to HPC clusters
- **[`vasp.py`](pydmclab/hpc/vasp.py)**: VASP input file generation, error handling, and calculation restart management

@todo: Add citation information for HPC tools

### Machine Learning Potentials

- **[`mace`](pydmclab/mlp/mace)**: MACE (Many-body Atomistic Conformal Embedding) neural network potential integration
  - **[`dynamics.py`](pydmclab/mlp/mace/dynamics.py)**: Structure relaxation, energy/force predictions, committee models for uncertainty quantification (untested), stress tensor calculation, and dispersion corrections
  - **[`utils.py`](pydmclab/mlp/mace/utils.py)**: Helper functions for model type determination and data conversion
  - Supports multiple optimization algorithms (FIRE, BFGS, LBFGS)
  - Provides access to pretrained models via `MACELoader` with various model sizes

- **[`chgnet`](pydmclab/mlp/chgnet)**: CHGNet (Crystal Hamiltonian Graph Neural Network) potential integration
  - **[`dynamics.py`](pydmclab/mlp/chgnet/dynamics.py)**: Molecular dynamics simulations with various ensembles (NVT, NPT), structure relaxation, and trajectory analysis tools
  - **[`enums.py`](pydmclab/mlp/chgnet/enums.py)**: Enumeration classes for optimizers and learning rate schedulers
  - **[`trainer.py`](pydmclab/mlp/chgnet/trainer.py)**: Tools for training custom CHGNet models with various loss functions and optimization strategies

- **[`utils.py`](pydmclab/mlp/utils.py)**: Common utilities for machine learning potentials including file handling and data preparation

@todo: Add citation information for machine learning potentials

### Plotting Tools

- **[`xrd.py`](pydmclab/plotting/xrd.py)**: X-ray diffraction pattern visualization with support for simulated patterns, peak broadening, overlay comparison between multiple structures, and animated transitions between different XRD patterns
- **[`dos.py`](pydmclab/plotting/dos.py)**: Tools for density of states (DOS) and crystal orbital Hamilton population (COHP) plotting with element-specific projections, customizable appearance, Gaussian smearing, and normalization options
- **[`rdf.py`](pydmclab/plotting/rdf.py)**: Radial distribution function visualization for analyzing atomic distances and coordination environments, with support for species-specific RDFs and structure averaging
- **[`pd.py`](pydmclab/plotting/pd.py)**: Phase diagram visualization tools:
  - `BinaryPD`: Binary phase diagrams with convex hull construction, stable/unstable phase marking, and customizable appearance
  - `TernaryPD`: Ternary phase diagrams with barycentric coordinate plotting, tie-lines, and stability visualization
- **[`bs.py`](pydmclab/plotting/bs.py)**: Electronic band structure visualization with element-projected fatband plotting for understanding orbital contributions
- **[`utils.py`](pydmclab/plotting/utils.py)**: Common utilities for plotting including custom color palettes, publication-quality figure parameters, and formula labeling helpers

@todo: Add citation information for plotting tools

### Data and Utilities

- **[`utils`](pydmclab/utils)**: General utility functions and helpers
  - **[`handy.py`](pydmclab/utils/handy.py)**: Collection of convenience functions:
    - File I/O utilities: `read_json`, `write_json`, `read_yaml`, `write_yaml` for seamless data persistence
    - Energy unit conversion between eV/atom and kJ/mol with `eVat_to_kJmol` and `kJmol_to_eVat`
    - Data serialization helper `convert_numpy_to_native` for JSON-serializable data conversion

@todo: Add citation information for data and utilities (probably none but double-check)

## Usage

Check the demos directory for example scripts demonstrating different functionalities:

- Structure analysis and manipulation: `demo_struc.py`
- Composition tools: `demo_comp.py`
- Phase stability: `demo_hulls.py`
- Materials Project queries: `demo_query.py`
- CHGNet molecular dynamics: `demo_chgnet_md.py`
- CHGNet structure relaxation: `demo_chgnet_relaxation.py`
- MACE structure relaxation: `demo_mace_relaxation.py`

## License

See the LICENSE.txt file for details.
