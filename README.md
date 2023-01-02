# pydmc
- common framework to rely upon for typical calculations and analysis done in the Bartel research group

## [core](pydmc/core)
- core utilities common to work in the group

### [comp](pydmc/core/comp.py)
- for manipulating and parsing chemical compositions

### [hulls](pydmc/core/hulls.py)
- for performing the convex hull analysis to compute phase stability

### [mag](pydmc/core/mag.py)
- for generating MAGMOMs for spin-polarized DFT calculations

### [query](pydmc/core/query.py)
- for retrieving and processing data from Materials Project

### [struc](pydmc/core/struc.py)
- for manipulating and analyzing crystal structures

## [hpc](pydmc/hpc)
- utilities related to setting up, running, and processing DFT calculations

### [analyze](pydmc/hpc/analyze.py)
- for processing VASP outputs into compact .json
- *@TODO*: 
    - Bader/Mulliken/Lowdin charge analysis
    - DOS analysis (test and polish)
    - COHP/COOP/DOE/COBI analysis

### [launch](pydmc/hpc/launch.py)
- for managing the set up of high-throughput DFT calculations

### [submit](pydmc/hpc/submit.py)
- for preparing HPC submission scripts to run chains of DFT calculations

### [vasp](pydmc/hpc/vasp.py)
- for making and editing VASP input files

## [data](pydmc/data)
- for loading datasets and configuration files

### [configs](pydmc/data/configs.py)
- baseline configurations for high-throughput executiong and analysis of DFT calculations

### [thermochem](pydmc/data/thermochem.py)
- elemental reference energies
- experimental thermochem data
- *@TODO*:
    - incorporate r2SCAN mus calculated with DMC standards
    - incorporate more Materials Project correction business

### [features](pydmc/data/features.py)
- for loading elemental property data (called by Gibbs energy calculator)

## [utils](pydmc/utils)
- standalone, simple helper functions

### [handy](pydmc/utils/handy.py)
- handy little functions like reading/writing .json

### [plotting](pydmc/utils/plotting.py)
- stores colors and matplotlib settings
- *@TODO*:
    - consider migrating to loadable matplotlib "style" and include in data/configs

## [dev](pydmc/dev)
- work in progress

### [energies](pydmc/dev/energies.py)
- to compute formation energies
- to be moved to pydmc/core
- *@TODO*:
    - test + polish
    - decide on whether to strictly follow MP reference energy rules

### [entries](pydmc/dev/entries.py)
- to play with MP's ComputedEntry functionality
- might be needed for phase diagram stuff beyond standard hulls
- *@TODO*:
    - set up basic functionality

### [grand](pydmc/dev/grand.py)
- for computing grand potential phase diagrams
- *@TODO*:
    - set up basic functionality

### [reactions](pydmc/dev/reactions.py)
- for computing reaction energies
- *@TODO*:
    - set up basic functionality

## [demos](pydmc/demos)
- demonstrations of how to perform typical actions with pydmc

## Future developments
- unit tests
- NEB set up and analysis
- AIMD set up and analysis
- ML IP set up and analysis
- more sophisticated plotting tools/standards
