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

- **[`fairchem`](pydmclab/mlp/fairchem/)**: UMA (Universal Models for Atoms) neutral network potential integration
  - **[`dynamics`](pydmclab/mlp/fairchem/dynamics.py)**: Molecular dynamics simulations, structure relaxation, energy/force predictions
  - **[`utils.py`](pydmclab/mlp/fairchem/utils.py)**: Helper functions for error handling
  - Consider citing
    ```bibtex
    @misc{wood_2025_uma,
        title={UMA: A Family of Universal Models for Atoms},
        DOI={10.48550/arXiv.2506.23971},
        author={Wood, Brandon M. and Dzamba, Misko and Fu, Xiang and Gao, Meng and Shuaibi, Muhammed and Barroso-Luque, Luis and Abdelmaqsoud, Kareem and Gharakhanyan, Vahe and Kitchin, John R. and Levine, Daniel S. and Michel, Kyle and Sriram, Anuroop and Cohen, Taco and Das, Abhishek and Rizvi, Ammar and Sahoo, Sushree Jagriti and Ulissi, Zachary W. and Zitnick, C. Lawrence},
        year={2025},
        eprint={2506.23971},
        archiveprefix={arXiv}
    }
    ```

- **[`mace`](pydmclab/mlp/mace)**: MACE (Many-body Atomistic Conformal Embedding) neural network potential integration
  - **[`dynamics.py`](pydmclab/mlp/mace/dynamics.py)**: Structure relaxation, energy/force predictions, committee models for uncertainty quantification (untested), stress tensor calculation, and dispersion corrections
  - **[`utils.py`](pydmclab/mlp/mace/utils.py)**: Helper functions for model type determination and data conversion
  - Supports multiple optimization algorithms (FIRE, BFGS, LBFGS)
  - Provides access to pretrained models via `MACELoader` with various model sizes
  - Consider citing
    ```bibtex
    @inproceedings{batatia_2022_mace,
        title={MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields},
        booktitle={Advances in Neural Information Processing Systems},
        author={Batatia, Ilyes and Kovacs, David Peter and Simm, Gregor N. C. and Ortner, Christoph and Csanyi, Gabor},
        editor={Oh, Alice H. and Agarwal, Alekh and Belgrave, Danielle and Cho, Kyunghyun},
        year={2022},
        url={https://openreview.net/forum?id=YPpSngE-ZU}
    }

    @misc{batatia_2022_design_space,
        title={The Design Space of E(3)-Equivariant Atom-Centered Interatomic Potentials},
        DOI={10.48550/arXiv.2205.06643},
        author={Batatia, Ilyes and Batzner, Simon and Kov{\'a}cs, D{\'a}vid P{\'e}ter and Musaelian, Albert and Simm, Gregor N. C. and Drautz, Ralf and Ortner, Christoph and Kozinsky, Boris and Cs{\'a}nyi, G{\'a}bor},
        year={2022},
        eprint={2205.06643},
        archiveprefix={arXiv}
    }
    ```

- **[`chgnet`](pydmclab/mlp/chgnet)**: CHGNet (Crystal Hamiltonian Graph Neural Network) potential integration
  - **[`dynamics.py`](pydmclab/mlp/chgnet/dynamics.py)**: Molecular dynamics simulations with various ensembles (NVT, NPT), structure relaxation, and trajectory analysis tools
  - **[`enums.py`](pydmclab/mlp/chgnet/enums.py)**: Enumeration classes for optimizers and learning rate schedulers
  - **[`trainer.py`](pydmclab/mlp/chgnet/trainer.py)**: Tools for training custom CHGNet models with various loss functions and optimization strategies
  - Consider citing
    ```bibtex
    @article{deng_2023_chgnet,
        title={CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling},
        DOI={10.1038/s42256-023-00716-3},
        journal={Nature Machine Intelligence},
        author={Deng, Bowen and Zhong, Peichen and Jun, KyuJung and Riebesell, Janosh and Han, Kevin and Bartel, Christopher J. and Ceder, Gerbrand},
        year={2023},
        pages={1–11}
    }
    ```

- **[`analyze.py`](pydmclab/mlp/analyze.py)**: Common analysis tools for ASE molecular dynamic simulations with machine learning potential calculators.
  - Consider citing
    ```bibtex 
    @article{deng_2016_alkali_superionic,
        title={Data-Driven First-Principles Methods for the Study and Design of Alkali Superionic Conductors},
        DOI={10.1021/acs.chemmater.6b02648},
        journal={Chemistry of Materials},
        author={Deng, Zhi and Zhu, Zhenbin and Chu, I.-Hung and Ong, Shyue Ping},
        year={2016}
    }
    ```

- **[`utils.py`](pydmclab/mlp/utils.py)**: Common utilities for machine learning potentials including file handling and data preparation

- For general use of the MLP functionality, consider citing
  ```bibtex
  @article{ong_2013_pymatgen,
      title={Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis},
      DOI={10.1016/j.commatsci.2012.10.028},
      journal={Computational Materials Science},
      volume={68},
      pages={314--319},
      author={Ong, Shyue Ping and Richards, William Davidson and Jain, Anubhav and Hautier, Geoffroy and Kocher, Michael and Cholia, Shreyas and Gunter, Dan and Chevrier, Vincent L. and Persson, Kristin A. and Ceder, Gerbrand},
      year={2013}
  }

  @article{larsen_2017_ase,
      title={The atomic simulation environment—a Python library for working with atoms},
      DOI={10.1088/1361-648X/aa680e},
      journal={Journal of Physics: Condensed Matter},
      volume={29},
      number={27},
      pages={273002},
      author={Ask Hjorth Larsen and Jens Jørgen Mortensen and Jakob Blomqvist and Ivano E Castelli and Rune Christensen and Marcin Dułak and Jesper Friis and Michael N Groves and Bjørk Hammer and Cory Hargus and Eric D Hermes and Paul C Jennings and Peter Bjerre Jensen and James Kermode and John R Kitchin and Esben Leonhard Kolsbjerg and Joseph Kubal and Kristen Kaasbjerg and Steen Lysgaard and Jón Bergmann Maronsson and Tristan Maxson and Thomas Olsen and Lars Pastewka and Andrew Peterson and Carsten Rostgaard and Jakob Schiøtz and Ole Schütt and Mikkel Strange and Kristian S Thygesen and Tejs Vegge and Lasse Vilhelmsen and Michael Walter and Zhenhua Zeng and Karsten W Jacobsen},
      year={2017}
  }
  ```

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
