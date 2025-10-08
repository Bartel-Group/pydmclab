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

## Extensions

PyDMCLab also contains optional extensions for additional functionality:

```bash
pip install -e .["mlp"]  # Machine learning potentialsm excludes FAIRChem due to dependencies
pip install -e .["fairchem"]  # FAIRChem extension
pip install -e .["defects"] # Defect generation tools
pip install -e .["phonons"] # Phonon calculation tools
```

## Components

### Core Modules

The PyDMCLab core modules rely heavily on [pymatgen](https://github.com/materialsproject/pymatgen). Please consider citing it if you use PyDMCLab in your work, along with any specific references found under each submodule.

  ```bibtex
  @article{ong_2013_python_materials_genomics,
      title={Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis},
      DOI={10.1016/j.commatsci.2012.10.028},
      journal={Computational Materials Science},
      volume={68},
      pages={314--319},
      author={Ong, Shyue Ping and Richards, William Davidson and Jain, Anubhav and Hautier, Geoffroy and Kocher, Michael and Cholia, Shreyas and Gunter, Dan and Chevrier, Vincent L. and Persson, Kristin A. and Ceder, Gerbrand},
      year={2013}
  }
  ```

- **[`comp`](pydmclab/core/comp.py)**: Composition analysis tools including formula normalization, element stoichiometry calculation, molecular fraction calculation, chemical system handling, and oxidation state prediction
- **[`struc`](pydmclab/core/struc.py)**: Structure manipulation including structure creation, transformation, symmetry analysis, site manipulation, solid solution generation, and slab generation for surface studies

  - If you use the `core.SolidSolutionGenerator` class, please consider citing:

    ```bibtex
    @article{sqsgen,
        doi = {10.1016/j.cpc.2023.108664},
        url = {https://doi.org/10.1016/j.cpc.2023.108664},
        year = {2023},
        month = jan,
        publisher = {Elsevier {BV}},
        pages = {108664},
        author = {Dominik Gehringer and Martin Fri{\'{a}}k and David Holec},
        title = {Models of configurationally-complex alloys made simple},
        journal = {Computer Physics Communications}
    }
    ```

- **[`energies`](pydmclab/core/energies.py)**: Energy calculation tools for formation energies, reaction energies, chemical potentials, thermodynamic analysis, and defect formation energy calculations

  - If you use the `energies.FormationEnergy` class, please consider citing:

    ```bibtex
    @article{bartel_2018_physical_descriptor,
      title={Physical descriptor for the Gibbs energy of inorganic crystalline solids and temperature-dependent materials chemistry},
      DOI={},
      journal={Nature Communications},
      volume={9},
      issue={1},
      pages={4168},
      author={Bartel, Christopher J. and Millican, Samantha L. and Deml, Ann M. and Rumptz, John R. and Tumas, William and Weimer, Alan W. and Lany, Stephan and Stevanovi{\'c}, Vladan and Musgrave, Charles B. and Holder, Aaron M.},
      year={2018}
    }
    ```

- **[`hulls`](pydmclab/core/hulls.py)**: Phase stability analysis including convex hull construction, decomposition energy calculation, stability prediction, mixing energy calculation, and parallel hull analysis for large systems

  - If you use the `hulls` module, please consider citing:

    ```bibtex
    @article{bartel_2022_computational_approaches,
      title={Review of computational approaches to predict the thermodynamic stability of inorganic solids},
      DOI={10.1007/s10853-022-06915-4},
      journal={Journal of Materials Science},
      volume={57},
      issue={23},
      pages={10475--10498},
      author={Bartel, Christopher J.},
      year={2022}
    }
    ```

- **[`mag`](pydmclab/core/mag.py)**: Magnetic property analysis tools for generating and analyzing different magnetic configurations and predicting magnetic ordering
- **[`query`](pydmclab/core/query.py)**: Materials database interfaces, primarily for the Materials Project API, supporting various query types (composition, chemical systems, structure, and properties)

- If you use the `query` module, please consider citing:

    ```bibtex
    @article{Horton2025,
      title = {Accelerated data-driven materials science with the Materials Project},
      ISSN = {1476-4660},
      url = {http://dx.doi.org/10.1038/s41563-025-02272-0},
      DOI = {10.1038/s41563-025-02272-0},
      journal = {Nature Materials},
      publisher = {Springer Science and Business Media LLC},
      author = {Horton,  Matthew K. and Huck,  Patrick and Yang,  Ruo Xi and Munro,  Jason M. and Dwaraknath,  Shyam and Ganose,  Alex M. and Kingsbury,  Ryan S. and Wen,  Mingjian and Shen,  Jimmy X. and Mathis,  Tyler S. and Kaplan,  Aaron D. and Berket,  Karlo and Riebesell,  Janosh and George,  Janine and Rosen,  Andrew S. and Spotte-Smith,  Evan W. C. and McDermott,  Matthew J. and Cohen,  Orion A. and Dunn,  Alex and Kuner,  Matthew C. and Rignanese,  Gian-Marco and Petretto,  Guido and Waroquiers,  David and Griffin,  Sinead M. and Neaton,  Jeffrey B. and Chrzan,  Daryl C. and Asta,  Mark and Hautier,  Geoffroy and Cholia,  Shreyas and Ceder,  Gerbrand and Ong,  Shyue Ping and Jain,  Anubhav and Persson,  Kristin A.},
      year = {2025},
      month = jul 
    }

    @article{Jain2013,
      title = {Commentary: The Materials Project: A materials genome approach to accelerating materials innovation},
      volume = {1},
      ISSN = {2166-532X},
      url = {http://dx.doi.org/10.1063/1.4812323},
      DOI = {10.1063/1.4812323},
      number = {1},
      journal = {APL Materials},
      publisher = {AIP Publishing},
      author = {Jain,  Anubhav and Ong,  Shyue Ping and Hautier,  Geoffroy and Chen,  Wei and Richards,  William Davidson and Dacek,  Stephen and Cholia,  Shreyas and Gunter,  Dan and Skinner,  David and Ceder,  Gerbrand and Persson,  Kristin A.},
      year = {2013},
      month = jul 
    }
    ```

- **[`defects`](pydmclab/core/defects.py)**: Tools for generating and analyzing defect structures including substitutions, vacancies, and interstitials, with support for structure distortion

  - If you use the `defects` module, please consider citing:

    ```bibtex
    @article{Kavanagh2024,
      title = {doped: Python toolkit for robust and repeatable charged
    defect supercell calculations},
      volume = {9},
      ISSN = {2475-9066},
      url = {http://dx.doi.org/10.21105/joss.06433},
      DOI = {10.21105/joss.06433},
      number = {96},
      journal = {Journal of Open Source Software},
      publisher = {The Open Journal},
      author = {Kavanagh,  Seán R. and Squires,  Alexander G. and Nicolson,  Adair and Mosquera-Lois,  Irea and Ganose,  Alex M. and Zhu,  Bonan and Brlec,  Katarina and Walsh,  Aron and Scanlon,  David O.},
      year = {2024},
      month = apr,
      pages = {6433}
    }

    @article{MosqueraLois2022,
      title = {ShakeNBreak: Navigating the defect configurational
    landscape},
      volume = {7},
      ISSN = {2475-9066},
      url = {http://dx.doi.org/10.21105/joss.04817},
      DOI = {10.21105/joss.04817},
      number = {80},
      journal = {Journal of Open Source Software},
      publisher = {The Open Journal},
      author = {Mosquera-Lois,  Irea and Kavanagh,  Seán R. and Walsh,  Aron and Scanlon,  David O.},
      year = {2022},
      month = dec,
      pages = {4817}
    }
    ```

- **[`alloys`](pydmclab/core/alloys.py)**: Alloy modeling with regular solution models for isomorphic and heterogeneous alloys, phase diagram construction, and thermodynamic property calculation

### HPC Tools

*Disclaimer*: Such tools are primarily designed to interface with high-performance computing resources at The University of Minnesota and may not be suitable for all environments.

If you use the HPC tools to run or analyze `LOBSTER` calculations, please consider citing:

    ```bibtex
    @article{maintz_2016_lobster,
        title={LOBSTER: A tool to extract chemical bonding from plane-wave based DFT},
        DOI={10.1002/jcc.24300},
        journal={Journal of Computational Chemistry},
        volume={37},
        issue={11},
        pages={1030-1035},
        author={Maintz, Stefan and Deringer, Volker L. and Tchougr{\'e}eff, Andriy L. and Dronskowski, Richard},
        year={2016}
    }

    @article{naik_2024_lobsterpy,
        title={LobsterPy: A package to automatically analyze LOBSTER runs},
        DOI={10.21105/joss.06286},
        journal={Journal of Open Source Software},
        volume={9},
        issue={94},
        pages={6286},
        author={Naik, Aakash Ashok and Ueltzen, Katharina and Ertural, Christina and Jackson, Adam J. and George, Janine},
        year={2024}
    }

    @article{george_2022_bonding_analysis,
        title={Automated bonding analysis with Crystal Orbital Hamilton Populations},
        DOI={10.1002/cplu.202200123},
        journal={ChemPlusChem},
        volume={87},
        issue={11},
        author={George, Janine and Petretto, Guido and Naik, Aakash and Esters, Marco and Jackson, Adam J. and Nelson, Ryky and Dronskowski, Richard and Rignanese, Gian-Marco and Hautier, Geoffroy},
        year={2022}
    }
    ```

- **[`analyze.py`](pydmclab/hpc/analyze.py)**: Analyze VASP output files to extract energies, structures, DOS, bonding data, and convergence results
- **[`collector.py`](pydmclab/hpc/collector.py)**: Collect and aggregate results from multiple calculations for comparative analysis
- **[`helpers.py`](pydmclab/hpc/helpers.py)**: Utility functions for configuration, submission, and results processing
- **[`launch.py`](pydmclab/hpc/launch.py)**: Tools for launching batches of VASP calculations across multiple compositions or structures
- **[`passer.py`](pydmclab/hpc/passer.py)**: Utilities for passing data between different calculation steps
- **[`phonons.py`](pydmclab/hpc/phonons.py)**: Setup and analysis of phonon calculations for vibrational properties
- **[`sets.py`](pydmclab/hpc/sets.py)**: VASP input set creation and management with customizable parameters
- **[`submit.py`](pydmclab/hpc/submit.py)**: Tools for preparing and submitting VASP calculations to HPC clusters
- **[`vasp.py`](pydmclab/hpc/vasp.py)**: VASP input file generation, error handling, and calculation restart management

### Machine Learning Potentials

If you use the machine learning potential modules, please consider [ASE](https://ase-lib.org/) as well as the respective potential implementations found under each submodule:

  ```bibtex
  @article{larsen_2017_ase,
      title={The atomic simulation environment—a Python library for working with atoms},
      DOI={10.1088/1361-648X/aa680e},
      journal={Journal of Physics: Condensed Matter},
      volume={29},
      number={27},
      pages={273002},
      author={Larsen, Ask Hjorth and Mortensen, Jens Jørgen and Blomqvist, Jakob and Castelli, Ivano E and Christensen, Rune and Dułak, Marcin and Friis, Jesper and Groves, Michael N and Hammer, Bjørk and Hargus, Cory and Hermes, Eric D and Jennings, Paul C and Jensen, Peter Bjerre and Kermode, James and Kitchin, John R and Kolsbjerg, Esben Leonhard and Kubal, Joseph and Kaasbjerg, Kristen and Lysgaard, Steen and Maronsson, Jón Bergmann and Maxson, Tristan and Olsen, Thomas and Pastewka, Lars and Peterson, Andrew and Rostgaard, Carsten and Schiøtz, Jakob and Schütt, Ole and Strange, Mikkel and Thygesen, Kristian S and Vegge, Tejs and Vilhelmsen, Lasse and Walter, Michael and Zeng, Zhenhua and Jacobsen, Karsten W},
      year={2017}
  }
  ```

- **[`chgnet`](pydmclab/mlp/chgnet)**: CHGNet (Crystal Hamiltonian Graph Neural Network) potential integration
  - **[`dynamics.py`](pydmclab/mlp/chgnet/dynamics.py)**: Molecular dynamics simulations with various ensembles (NVT, NPT), structure relaxation, and trajectory analysis tools
  - **[`enums.py`](pydmclab/mlp/chgnet/enums.py)**: Enumeration classes for optimizers and learning rate schedulers
  - **[`trainer.py`](pydmclab/mlp/chgnet/trainer.py)**: Tools for training custom CHGNet models with various loss functions and optimization strategies

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

- **[`fairchem`](pydmclab/mlp/fairchem/)**: UMA (Universal Models for Atoms) neutral network potential integration
  - **[`dynamics`](pydmclab/mlp/fairchem/dynamics.py)**: Molecular dynamics simulations, structure relaxation, energy/force predictions
  - **[`utils.py`](pydmclab/mlp/fairchem/utils.py)**: Helper functions for error handling

    ```bibtex
    @article{wood_2025_uma,
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

    ```bibtex
    @article{batatia_2022_mace,
        title={MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields},
        booktitle={Advances in Neural Information Processing Systems},
        author={Batatia, Ilyes and Kovacs, David Peter and Simm, Gregor N. C. and Ortner, Christoph and Csanyi, Gabor},
        editor={Oh, Alice H. and Agarwal, Alekh and Belgrave, Danielle and Cho, Kyunghyun},
        year={2022},
        url={https://openreview.net/forum?id=YPpSngE-ZU}
    }

    @article{batatia_2022_design_space,
        title={The Design Space of E(3)-Equivariant Atom-Centered Interatomic Potentials},
        DOI={10.48550/arXiv.2205.06643},
        author={Batatia, Ilyes and Batzner, Simon and Kov{\'a}cs, D{\'a}vid P{\'e}ter and Musaelian, Albert and Simm, Gregor N. C. and Drautz, Ralf and Ortner, Christoph and Kozinsky, Boris and Cs{\'a}nyi, G{\'a}bor},
        year={2022},
        eprint={2205.06643},
        archiveprefix={arXiv}
    }
    ```

- **[`analyze.py`](pydmclab/mlp/analyze.py)**: Common analysis tools for ASE molecular dynamic simulations with machine learning potential calculators.

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
