# Contents

- [Installation](#installation)
- [General best practices](#general-best-practices)
- [VASP](#for-vasp)
  - [general setup](#general-setup)
  - [the hpc set of modules](#the-hpc-set-of-modules)
  - [the launcher script](#the-launcher-script)
  - [a typical flow](#a-typical-flow)
- [Thermodynamics](#for-thermodynamics)
  - [formation energies](#formation-energies)
  - [decomposition energies](#decomposition-energies)
- [Supporting modules](#supporting-modules)
- [Development](#development)
- [Maintenance](#maintenance)
- [FAQ](#faq)

# Installation

## Installation instructions

- there will always be two versions of `pydmclab` available:
  - the "release" version
    - each new release will have a new version number (e.g., 0.0.2)
    - this is always a stable version of the code
    - this version will only be updated periodically (i.e., not after each git commit)
  - the "development" version
    - this is the latest version of the code
    - this version will be updated after each git commit
    - this version may include features not present in the "release" version
      - it may also include bugs not present in the "release" version

### Installing the "release" version

```bash
pip install pydmclab
```

- this can be executed from anywhere on your computer (or cluster)
- note:
  - you can specify which python version you want to install this under with:

```bash
pip install pydmclab --prefix=<path to your python version>
```

- e.g.,

```bash
pip install pydmclab --prefix=/home/cbartel/cbartel/bin/anaconda3
```

### Installing the "development" version

- clone the repository if you have not already:

```bash
git clone https://github.umn.edu/bartel-group/pydmclab.git
```

- navigate to the repository:

```bash
cd pydmclab
```

- pull the repository:

```bash
git pull
```

- install the repository:

```bash
pip install .
```

- note:
  - you can specify which python version you want to install this under with:

```bash
pip install . --prefix=<path to your python version>
```

- e.g.,

```bash
pip install . --prefix=/home/cbartel/cbartel/bin/anaconda3
```

### Configuring your pseudopotentials with pymatgen

- if you are getting lots of POTCAR errors after installing, do this

```bash
pmg config --add PMG_VASP_PSP_DIR /home/cbartel/shared/bin/pymatgen_pot
pmg config --add PMG_DEFAULT_FUNCTIONAL PBE_54
```

- you should only have to do this one time, not for each successive installation of `pydmclab`

# General best practices

- see our [group wiki](https://github.umn.edu/bartel-group/dmc/wiki/research_output#code) for coding best practices

# For VASP

## general setup

- VASP should be run on MSI
- all calculations that follow a common structure (e.g., a certain phase diagram) should be executed using a single python script (called `launcher.py`)
- `launcher.py` should live in a folder called: `...../<some relevant header>/scripts`
- `pydmclab` will create two directories at the same level of scripts when `launcher.py` is executed (using `python launcher.py`)
  - `.../data` - this is where .json files will be stored during job creation and analysis
- `.../calcs/` - this is where VASP calculations will be executed

## the hpc set of modules

- there are four main modules of `pydmclab` for running VASP. you can think of these like layers of an onion
  - `pydmclab.hpc.vasp`
    - this is for configuring specific VASP calculations
      - INCAR, POSCAR, KPOINTS, POTCAR
    - this module operates on a single crystal structure, with a single initial magnetic configuration, with a single set of inputs
    - this is the core of the onion
  - `pydmclab.hpc.submit`
    - this is for configuring a submission script to launch a chain of VASP calculations
      - e.g., if we want to run a geometry optimization then a static calculation, we can configure our submission script to first run VASP to optimize the geometry then run VASP to perform a static (electronic) optimizatino
    - this module operates on a single crystal structure with a single initial magnetic configuration
      - this particular input will be funneled through a serial sequence of VASP calculations to get a desired output
    - this module will also inspect running or previously run chains of calculations to update the submission script and VASP input files to avoid duplicate calculations and/or to fix errors
    - now, we're one layer further from the core (specific VASP jobs) as this module handles multiple VASP jobs (connected through chains)
  - `pydmclab.hpc.launch`
    - this is for configuring a set of submission scripts to launch a set of similar VASP calculation chains for different materials (crystal structures and/or magnetic configurations)
    - this module operates on a collection of crystal structures and magnetic configurations
    - now we're in the outermost layer because this layer handles multiple chains of VASP jobs
  - `pydmclab.hpc.analyze`
    - this is for analyzing the results of many VASP calculations launched in a similar fashion
    - this module operates on a collection of crystal structures and magnetic configurations
    - also in the outermost layer of the onion because this layer handles every single VASP job in every single chain

## the launcher script

- the launcher script is a single python file that uses the above modules to launch a desired set of similar chains of VASP calculations
  - i.e., a set of materials that will be calculated in a common way
    - e.g., all materials in a chemical system for a phase diagram calculation
    - e.g., a Li-containing material with varying fractions of Li for a voltage curve calculation
    - etc.

- see `pydmclab/pydmclab/demos/demo_launcher_template.py` for a good starting template

- the launcher script has a natural sequence of steps

### high level settings

- there are a number of variables we set at the top, but I will discuss these after discussing each function because it will make more sense that way

### `get_query`

- a generic version is available at `pydmclab.hpc.helpers.get_query`
- you will often have to modify arguments in this function (and potentially modify the function itself!)
- you need one or more structures to start with
- these structures could come from Materials Project, experimental collaborators, your own previous calculations, someone else's calculations, etc.
- these structures can be fully occupied or partially occupied
- these structures could have the composition you want to calculate or they could not
- these could be unit cells or supercells
- in the next step, we will manipulate these structures
- for now, we just need a starting point
- this function should return something like:

```python
query = {<globally unique ID> : 
            'structure' : <structure object>,
            'formula' : <formula string>}
```

- it's OK if there's other stuff, but you at least need to map an ID to a formula and a structure

### `get_strucs`

- a generic version is available at `pydmclab.hpc.helpers.get_strucs`
- you will have to modify this unless you are calculating your structures exactly as they leave `get_query`
- starts with the results of `get_query`
- this is where we manipulate our starting structures (or not)
  - e.g., by generating many ordered versions of partially occupied structures
  - e.g., by replacing some element with another element
  - e.g., by removing some fraction of some element
  - etc.
- the structures that leave this step must be ready for DFT calculations
  - they must be ordered
  - they must have the composition we want
  - they must be the cell sizes we want
- the result of this function should look like

```python
strucs = {<composition : 
            {<unique ID for that composition> 
                : <structure object>}}
```

- `<composition>`
  - could be a chemical formula (e.g., one generated using `pydmclab.core.comp.CompTools(<formula>).clean`)
  - could be some unique identifier you used to transform a structure (e.g., `Li2MP2S6_3` indicating I took 3 Li out per cell)
- `<unique ID for that composition>`
  - could be a Materials Project ID indicating a certain polymorph for that composition (e.g., `mp-1234`)
  - could be your own ID (e.g., `my-12345`)
  - could be an index from an ordered structure enumeration procedure (e.g., `0`, `1`, `2`, etc.)

### `get_magmoms`

- a generic version is available at `pydmclab.hpc.helpers.get_magmoms`
- you will rarely need to modify this
- starts with the results of `get_strucs`
- if we are running AFM calculations, we need to associate various initial magnetic configurations with each unique crystal structure we want to calculate
- if we are not running AFM calculations, then `pydmclab.core.mag` takes care of this for us and we do not need to execute this function
- for AFM configuration generation, this function will generate a finite set of randomly and symmetrically distinct magnetic orderings (spin up, spin down) for each unique crystal structure
- the result of this function should look like

```python
magmoms = {<composition> : 
            {<unique ID for that composition> 
                : {<unique ID for that magnetic configuration> 
                    : <list of magnetic moments>}}}
```

- `<unique ID for that magnetic configuration>`
  - this will be `0`, `1`, `2`, etc. for randomly generated magnetic configurations

### `get_launch_dirs`

- a generic version is available at `pydmclab.hpc.helpers.get_launch_dirs`
- you will rarely have to modify this
- now that we have structures and magnetic configurations, we can enter our onion (described [above](#the-hpc-set-of-modules))
- this function will generate all the directories and files we need to launch VASP jobs for every crystal structure + magnetic configuration we prepared
- it will create the directory structure and write the VASP input files
- we use a common directory structure here. assuming our `launcher.py` file (this script) is in `*/scripts`, then our "launch_dirs" will spawn in `*/calcs/<composition>/<unique ID for that composition>/<standard>/<unique ID for that magnetic configuration>`
- the result of this function should look like

```python
launch_dirs = {<composition>/<unique ID for that composition>/<standard>/<unique ID for that magnetic configuration> : {'xcs' : [list of (final) exchange correlation methods to run for this launch directory],
'magmoms' : [list of magmoms associated with calculations in that directory]}}
```

- `<standard>` could be `dmc` for our general group standard or `mp` for Materials Project consistency (I'll get into configurations shortly)

### `submit_calcs`

- a generic version is available at `pydmclab.hpc.helpers.submit_calcs`
- you will rarely have to modify this
- now we are going one level deeper into our onion to create/modify every submission script that launches VASP calculation chains from every launch directory we just created
- this function is slow because it needs to check convergence and sort out errors in every one of our individual VASP calculations to figure out the best course of action to take
  - because of this, there is an option to parallelize the execution of this function
- this function returns nothing, but it will go into every launch directory, inspect all submission scripts, inspect all VASP calculations, and assess whether or not to make changes to VASP input files and/or submission scripts
- in general, you'll want to execute your launcher script multiple times until your calculations finish
  - essentially, any time you think there is a possibility that `pydmclab` will find calculations that might have run into errors or hit the walltime

### `get_results`

- a generic version is available at `pydmclab.hpc.helpers.get_results`
- you will rarely have to modify this
- now we are going to collect the results of every VASP calculation within every chain within every launch directory
- this function will update a dictionary/.json of a (detailed) summary of the results from all successful calculations executed in all your launch directories
- the result of this function looks like:

```python
{<key> : 
    {'meta' : {<meta data>},
      'results' : {<summary of results>},
      'structure' : {<relaxed structure>},
      etc.}}
```

- `<key>` has the form of each VASP calculation with folders joined by '--'
  - e.g., `LiMn2O4--mp-1234--dmc--fm--metagga-static`

### `get_gs`

- a generic version is available at `pydmclab.hpc.helpers.get_gs`
- you will rarely have to modify this
- this function collects data only for the lowest energy structure you calculated for each unique composition
- the result of this function looks like:

```python
{<standard>:
  {<xc> : 
    {<formula> : 
      {DATA}}}}
```

### `get_thermo_results`

- a generic version is available at `pydmclab.hpc.helpers.get_thermo_results`
- you will rarely have to modify this
- this function grabs thermodynamic data for all your calculations, including ground-states and non-ground-states
- the result of this function looks like:

```python
{<standard>:
  {<xc> : 
    {<formula> : 
      {DATA}}}}
```

### `<custom functions>`

- feel free to add whatever functions you'd like after this
- e.g., a common one might be `get_slim_results` which parses the output of `get_results` and grabs only what you need (e.g., ground-states, meta-gga only, etc.)
- the goal will be to perform any analysis that requires VASP output files on MSI --> pack that into a nice .json file --> transfer that .json file to your local computer --> do more analysis that does not require VASP output files (e.g., plotting DOS, making phase diagrams etc)

### `main`

- you will have to add your `<custom functions>` to this if you have any
- you will likely want to toggle `remake` switches as you develop
- this is the code that actually executes when you execute `python launcher.py`
- the basic format is to include switches for each of the functions above
  - usually in the form of `remake_*` where if `remake_*` is True, then the function, `get_*` is guaranteed to run
  - if `remake_*` is False, then the code will try to read the json file containing the relevant output of `get_*` and use that instead
- in general:
  - `remake_query`, `remake_strucs`, `remake_magmoms`, and `remake_launch_dirs` can usually be set to False
    - unless you previously ran this launcher script, then changed something and want to overwrite one of these .json's
  - `remake_subs` should generally be set to True
    - this will allow `pydmclab` to inspect your calculations and modify them as needed (e.g., to fix errors or re-launch prematurely terminated calculations)
    - you might set this to False if you don't care about executing any more calculations and just want to analyze what you have
  - `remake_results` should generally be set to True
    - this will scrape your calculations and update with the latest results
    - you might set this to False if you are working on some functions that work after `get_results` and don't care about refreshing `results.json` to get the latest results
- when you are setting up your launcher for the first time:
  - you might add a `return` statement after each function
  - this will kill the execution of the launcher script after that function runs, allowing you to inspect the .json file (or launch directories or submission scripts etc) that were just created

### specify `CONFIGS`

- at the top of `launcher.py`
- you will generally have to modify these things!
- these dictionaries allow a user to use non-default settings (configurations) in each of the above `pydmclab.hpc.*` modules
- the default configs can be found in `pydmclab/pydmclab/data/data/*.yaml`
  - `_vasp_configs.yaml`
    - settings pertaining to `pydmclab.hpc.vasp`
    - how individual VASP calculations can be modified
  - `_sub_configs.yaml`
    - settings pertaining to `pydmclab.hpc.sub`
    - how submission script generation can be modified
    - note: there is one config here called `n_procs` that will allow submission scripts to be executed in parallel (if > 1). this has no effect on VASP itself, just the generation of these submission scripts
  - `_slurm_configs.yaml`
    - also pertains to `pydmclab.hpc.sub`
    - spefically affects the `#SBATCH` parameters in each submission script
      - e.g., how many nodes, tasks, etc will be used for each VASP job
  - `_launch_configs.yaml`
    - settings pertaining to `pydmclab.hpc.launch`
    - how the generation of launch directories gets modified
      - e.g., to perturb each POSCAR at the start of all chains beginning with that structure
      - e.g., whether we're going to compare our results to MP
      - e.g., how many AFM calcs we'll run
  - `_batch_vasp_analyze_configs.yaml`
    - settings pertaining to `pydmclab.hpc.analyze`
    - how the analysis of VASP calculations gets modified
      - e.g., what things you'd like to collect from each calculation
      - e.g., which types of calculations you want to collect data for
      - also has an `n_procs` config to allow for VASP calculations to be analyzed in parallel (similar to parallelization of the `sub` step)

### other high level settings

- `COMPOSITIONS`
  - if you are using MP
  - an input to a Materials Project query (if appropriate)
  - could be a formula (like `Li2O`), a list of formulas, a chemical system (like `Ba-Zr-S`), or a list of chemical systems
  - used by `get_query`
- `API_KEY`
  - if you are using MP

## a typical flow

1. I decide I want to run some calculations
2. I create a folder on MSI: `/home/cbartel/<username>/<specific name>` (e.g., `/home/cbartel/cbartel/LiMn2O4`)
3. I create a folder called `scripts` (e.g., `/home/cbartel/cbartel/LiMn2O4/scripts`)
4. I copy a demo launcher script into this folder (e.g., `cp ~/bin/pydmclab/pydmclab/demos/demo_launcher_template.py /home/cbartel/cbartel/LiMn2O4/scripts/launcher.py`)
5. I start editing `launcher.py`

- start at the top
- what compositions do I want to query for (if any)?
- do I want to run AFM or not?
- what kind of standard/functional do I want?
- what kind of configs do I want to use?
  - while developing the launcher script, you should leave `...parallel=False` in both the `SUB_CONFIGS` and the `ANALYSIS_CONFIGS` so that you can execute `python launcher.py` from the login node to quickly inspect what is printed
- how do I need to customize `get_query`?
- how do I need to customize `get_strucs`?

6. I'm happy with my `get_query` function, so I'll execute it and add a `return` statement in `main` after `query.json` is generated
7. I'll go to `.../data` in `ipython` to inspect the query results and make sure they look like I expect
8. I'm happy with my `get_strucs` function, so I'll execute it and add a `return` statement in `main` after `strucs.json` is generated
9. I'll go to `../data` in `ipython` to inspect the structures I generated and make sure they look like I expect
10. I'll set `ready_to_launch` to False and move the `return` statement to the bottom of `main`. Now all my functions will execute but no VASP jobs will be submitted
11. I'll go take a look at the (POSCAR, INCAR, KPOINTS, POTCAR, sub_*.sh) files in 1-3 calculations in `../calcs/*/*/*/*/*/` to make sure they look like I expect
12. OK, now we're ready to roll. I want to execute my launcher script in parallel so that it goes fast, so I first need to make a submission script (`.sh`) file for it (this is just a file with SLURM tags and one execution line: `python launcher.py`)

- `ipython`
- `from pydmclab.utils.handy import make_sub_for_launcher`
- `make_sub_for_launcher()`

13. All right, now I'll make sure `...parallel` in `SUB_CONFIGS` is the same as `...parallel` in `ANALYSIS_CONFIGS` and that they're both the same as `--ntasks` in `sub_launcher.sh`

- usually ~8-32 is a good number

14. I'll submit `sub_launcher.sh` to the queue

- `sbatch sub_launcher.sh`

15. Now I'll tail the log files to make sure things don't go awry

- `tail -f *log*`
- if something prints to `*log*e`, then my launcher script exited with an error..
- if things go according to plan, `*log*o` will get populated with what usually prints to the screen when you manually execute `python launcher.py`

16. I could also tail the OSZICAR's of all my calculations to see that those are running properly

- `tail -f ../calcs/*/*/*/*/*/OSZICAR`

17. If all looks good, I'll go do something else for 4-24 hours

18. Now, I'll come back and re-submit my launcher

- `sbatch sub_launcher.sh`
- this will have `pydmclab` crawl through all my calculations, fix the ones that fizzled, re-launch any that aren't done, and analyze any that are done

19. Now, I might look at my results dictionary to see if I'm getting the results I suspect

20. Repeat 17-19 until calcs are done or you want to move data locally to start more analysis

21. Create the smallest .json file you can that contains all the data you need for your analysis and transfer that to your local computer for more analysis (use `scp` or `Globus` for the file transfer)

# For thermodynamics

- two main modules:
  - `pydmclab.core.energies` for computing formation energies
  - `pydmclab.core.hulls` for computing decomposition energies (stabilities)

- [Chris B's review](https://bartel.cems.umn.edu/sites/bartel.cems.umn.edu/files/2022-07/bartel.bartel_2022-j.mater_.sci_.pdf) should be a useful reference to understand how/why this works

## formation energies

- the mapping from total (internal) energies computed with DFT to formation energies requires a comparison to elemental reference states (chemical potentials)

```math
\Delta E_{\mathrm{f}} = E_{\mathrm{DFT}} - \sum_{i} \mu_i n_i
```

- where the sum goes over all elements in the compound and is weighted by the stoichiometry (composition) of the formula of interest

- the basic idea:
  - start with a DFT total energy
  - decide the conditions you want to compute the formation energy at (e.g., 0 K or finite T)
  - generate the appropriate chemical potentials
  - modify the DFT total energy if needed (eg to account for finite T)

### zero temperature

- at zero temperature, the total (internal) energies of compounds and elements come from DFT
- these energies must be computed in a compatible manner
- note that 0 K DFT formation energies are often compared to experimental formation enthalpies at ambient conditions (298 K)
- at 0 K, we'll make use of `pydmclab.core.energies.FormationEnthalpy`, which takes as input:
  - the chemical formula
  - the DFT total energy
  - the elemental reference states (chemical potentials)

- chemical potentials can be obtained using `pydmclab.core.energies.ChemPots`
  - need to tell it specifics of your DFT calculations
  - don't worry about temperature, partial pressures, etc. b/c T = 0

### finite temperature

- at finite temperature and a closed system, the Gibbs energy is what we care about:

```math
\Delta G_{\mathrm{f}} = G_{\mathrm{DFT}} - \sum_{i} G_i n_i
```

- entropy contributes to the energy of compounds as well as elements. there are four contributors to the entropy

- for solids, there are four contributors to the entropy:
  - vibrational
    - related to phonon dispersion
  - configurational
    - related to (quasi)random occupation of certain sites by >1 atom
  - electronic
    - related to (quasi)random occupation of certain sites by the same atom having different valence states (sort of..)
  - magnetic
    - related to (quasi)random occupation of certain sites by the same atom having different spin states (sort of..)
- for non-solids, we also have rotational and translational entropy

#### computing Gibbs energies

- just like with 0 K formation energies, we'll need to:
  - determine our reference energies using `pydmclab.core.energies.ChemPots`
  - compute the formation energy, this time using `pydmclab.core.energies.FormationEnergy` (instead of "...Enthalpy")

- for elements, we fortuntely have nice experimental data for Gibbs energies as a function of temperature (retrieve as dictionary with `pydmclab.data.thermochem.mus_at_T`)
  - we can also modify these reference energies by accounting for the activity of gaseous species (i.e., when p_i < 1 atm)

  ```math
  G_i(T, p_i) = G_i(T) + k_B T ln(p_i)
  ```

  - this essentially says that the free energy for an element to be a gas becomes more negative (more favorable) as the concentration of that element in the gase phase decreases
  - note that most gases are diatomic, so we need to use a 1/2 somewhere

- for solid compounds, we need to modify the DFT energies on our own:
  - vibrational entropy: [Bartel 2018](https://www.nature.com/articles/s41467-018-06682-4)
  - configurational entropy: ideal mixing is a decent first approximation

  ```math
  S_{\mathrm{mix}}/f.u.= -k_B[xlnx - (1-x)ln(1-x)]
  ```

  - for binary partial occupation of some site with concentrations (x, 1-x)

  - electronic entropy: usually ignore (though important for [LiFePO4](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.97.155704) and [CeO2](https://www.nature.com/articles/s41467-017-00381-2)! among other materials..)
  - magnetic entropy: usually ignore

- finite temperature also allows us to consider the effects of partial pressures of gaseous species (e.g., elements like O2) in the system

- so the procedure looks like:
  - get the elemental reference energies (chemical potentials) at the conditions of interest using `pydmclab.core.energies.ChemPots`
  - modify the DFT energies to account for finite temperature and compute the formation energy with respect to these chemical potentials using `pydmclab.core.energies.FormationEnergy`
  - a common approach would be to loop through a range of temperatures and compute the formation energy at each temperature

## decomposition energies

- now we'll take the formation energies we computed for all compounds in a given chemical space (or many chemical spaces) and perform the convex hull analysis to obtain the thermodynamic stability at the conditions where we computed the formation energies

- for this, we'll use `pydmclab.core.hulls` which has two basic functions:
  - assemble a dictionary of formation energies for a given chemical space (create the input to the convex hull analysis)
    - `pydmclab.core.hulls.GetHullInputData`
  - perform the convex hull analysis for each chemical space that was prepared
    - `pydmclab.core.hulls.AnalyzeHull`
- the result is a dictionary including stability info for every compound of interest that looks like:

```python
{<formula> : {'Ef' : formation energy (the one you inputted, hopefully),
              'Ed' : decomposition energy (as calculated from the hull analysis),
              'stability' : True if on the hull else False,
              'rxn' : the decomposition reaction that defines stability }}
```

- this can also be done in parallel (i.e., parallelized over chemical (sub)spaces) using `pydmclab.core.hulls.ParallelHulls`
  
## demos

- decomposition energies: `pydmclab/pydmclab/demos/demo_hulls.py`
- formation energies: **@TODO**

# Supporting modules

## core

- core `pydmclab` module used by `pydmclab.energies` and `pydmclab.hpc`

### Chemical compositions

- `pydmclab.core.comp.CompTools` will allow you to systematically generate and analyze chemical formulas
  - make a systematic string format for any chemical formula
  - determine what elements are in a compound
  - determine how many of each element are in a compound
  - etc.

### Query Materials Project

- `pydmclab.core.query.MPQuery` will allow you to systematically query Materials Project and parse the data in typical ways (e.g., to collect only ground-states for some chemical space or chemical formula of interest)

### Crystal structures

- `pydmclab.core.struc.StrucTools` will allow you to parse and manipulate crystal structures
  - figure out how many sites
  - make a supercell
  - replace species
  - order disordered structures
  - etc.

- `pydmclab.core.struc.SiteTools` will allow you to parse and manipulate individual sites within a crystal structure
  - e.g., get the occupation

### Magnetic sampling

- `pydmclab.core.mag.MagTools` will prepare `MAGMOM` strings for crystal structures
  - for AFM orderings (for which there are often very many options), this module will enumerate symmetrically distinct orderings

## utils

- "utilities" that are convenient but not worthy of their own modules

### handy functions

- `pydmclab.utils.handy` provides several standalone functions that help you do stuff
  - e.g., read and write a .json file in one line
  - e.g., convert kJ/mol to eV/atom

### plotting functions

- `pydmclab.utils.plotting` provides several functions that help with consistent plotting
  - e.g., retrieve color palettes
  - e.g., set our `maplotlib` "rc parameters"
- typical use case. let's say you are creating a `.py` file to plot stuff. at the top of this script, do the following:

  ```python
  from pydmclab.utils.plotting import set_rc_params, get_colors

  set_rc_params()
  my_palette = <one of the strings accepted by `get_colors`>
  colors = get_colors(my_palette)
  ```

## data

- data files are stored in `pydmclab.data.data` and loaded in `pydmclab.data.*`

### `pydmclab.data.configs`

- this mainly pertains to loading `.yaml` files for the `pydmclab.hpc.*` modules (related to running VASP)
- here, you'll find the default configurations associated with running VASP, as described in the [VASP section](#for-vasp)

### `pydmclab.data.thermochem`

- this loads mainly elemental reference energies (chemical potentials) for use by `pydmclab.core.energies.ChemPots` as described in the [thermodynamics section](#for-thermodynamics)

### `pydmclab.data.plotting_configs`

- this holds data used by `pydmclab.utils.plotting` as described in the [plotting section](#plotting-functions)

### `pydmclab.data.features`

- this holds the atomic masses of the elements, which is needed for applying the Bartel 2018 vibrational entropy model used in `pydmclab.core.energies.FormationEnergy` as described in the [thermodynamics section](#for-thermodynamics)

## demos

- this holds various demo scripts to illustrate how different modules work

## dev

- this is where we can work on developing new things before integrating with existing modules (or creating new ones)

## old

- this is where we (temporarily) save stuff that might be useful but is not currently integrated

## hpc

- these modules are covered in detail in the [VASP section](#for-vasp)

## energies

- these modules are covered in detail in the [thermodynamics section](#for-thermodynamics)

# Development

## Top priorities

- (pseudo-)ternary phase diagrams
- slabs
- defects
- universal potentials
- tests

## Small things

- `pydmclab.core.energies`
  - formation energy demo
- create `requirements.txt` file
- create `pydmclab` conda environment

## Medium things

- `pydmclab.core.query`
  - move to new API
- `pydmclab.utils.plotting`
  - plot DOS-like things
    - tdos, pdos, tcohp, pcohp
  - plot phase diagrams
    - binary
    - ternary
- `pydmclab.core.struc`
  - more analysis capabilities
    - e.g., using `pymatgen.analysis.local_env` and `pymatgen.analysis.chemenv`
  - capabilities for surfaces

## Big things

- `pydmclab.hpc.*`
  - handling surface calculations
  - handling optimization with ML potentials
  - handling NEB calculations
  - handling (charged) defect calculations and analysis
- `pydmclab.ml.*`
  - modules for using/tuning ML potentials
- `pydmclab.md.*`
  - modules for running MD with ASE
- `pydmclab.tests`
  - thorough unit tests
  
# Maintenance

- `core.comp` documented, tested, demoed (6/23/23)
- `core.struc` documented, tested, demoed (6/23/23)
- `core.mag` documented, tested, demoed (6/23/23)
- `core.query` tested (6/1/23)
  - **@TODO** document, demo
- `core.energies` tested (6/1/23)
  - **@TODO** document, demo
- `core.hulls` tested (6/1/23)
  - **@TODO** document, demo

- `hpc.*`
  - **@TODO** document, test, demo
- `plotting.*`
  - **@TODO** document, test, demo

# FAQ

- please add questions!
