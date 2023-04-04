# General best practices

- if you are going to do something twice, write a function
- if you are going to perform some analysis more than once, save the output to a `.json` file
- write very clear documentation
  - and read the documentation within `pydmclab` ! This should not be a black box
- python files should be a collection of functions/classes and a `main()` function that allows you to execute any sequence of them
- if you are going to use parallel processing on a supercomputer, you must submit whatever is using multiple cores to a compute node, not a login node
  - note: this happens automatically for VASP jobs, but it is often useful to parallelize the execution of the launcher script which spawns many VASP jobs

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
  - `pydmclab.hpc.sub`
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

- see `pydmclab/pydmclab/demos/demo_*vasp*.py` for examples

- the launcher script has a natural sequence of steps

### high level settings

- there are a number of variables we set at the top, but I will discuss these after discussing each function because it will make more sense that way

### `get_query`

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

- <composition>
    - could be a chemical formula (e.g., one generated using `pydmclab.core.comp.CompTools(<formula>).clean`)
    - could be some unique identifier you used to transform a structure (e.g., `Li2MP2S6_3` indicating I took 3 Li out per cell)
- <unique ID for that composition>
    - could be a Materials Project ID indicating a certain polymorph for that composition (e.g., `mp-1234`)
    - could be your own ID (e.g., `my-12345`)
    - could be an index from an ordered structure enumeration procedure (e.g., `0`, `1`, `2`, etc.)

### `get_magmoms`

- you will rarely need to modify this
- starts with the results of `get_strucs`
- if we are running AFM calculations, we need to associate various initial magnetic configurations with each unique crystal structure we want to calculate
- if we are not running AFM calculations, then `pydmc.core.mag` takes care of this for us and we do not need to execute this function
- for AFM configuration generation, this function will generate a finite set of randomly and symmetrically distinct magnetic orderings (spin up, spin down) for each unique crystal structure
- the result of this function should look like

    ```python
    magmoms = {<composition> : 
                {<unique ID for that composition> 
                    : {<unique ID for that magnetic configuration> 
                        : <list of magnetic moments>}}}
    ```

- <unique ID for that magnetic configuration>
    - this will be `0`, `1`, `2`, etc. for randomly generated magnetic configurations

### `get_launch_dirs`

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

- <standard> could be `dmc` for our general group standard or `mp` for Materials Project consistency (I'll get into configurations shortly)

### `submit_calcs`

- you will rarely have to modify this
- now we are going one level deeper into our onion to create/modify every submission script that launches VASP calculation chains from every launch directory we just created
- this function is slow because it needs to check convergence and sort out errors in every one of our individual VASP calculations to figure out the best course of action to take
  - because of this, there is an option to parallelize the execution of this function
- this function returns nothing, but it will go into every launch directory, inspect all submission scripts, inspect all VASP calculations, and assess whether or not to make changes to VASP input files and/or submission scripts
- in general, you'll want to execute your launcher script multiple times until your calculations finish
  - essentially, any time you think there is a possibility that `pydmclab` will find calculations that might have run into errors or hit the walltime

### `get_results`

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

  - <key> has the form of each VASP calculation with folders joined by '.'
    - e.g., `LiMn2O4.mp-1234.dmc.fm.metagga-static`

### `<custom functions>`

- feel free to add whatever functions you'd like after this
- e.g., a common one might be `get_slim_results` which parses the output of `get_results` and grabs only what you need (e.g., ground-states, meta-gga only, etc.)
- the goal will be to perform any analysis that requires VASP output files on MSI --> pack that into a nice .json file --> transfer that .json file to your local computer --> do more analysis that does not require VASP output files (e.g., plotting DOS, making phase diagrams etc)

### `main`

- you will have to add you `<custom functions>` to this if you have any
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

- you will generally have to modify these!
- `COMPOSITIONS`
  - an input to a Materials Project query (if appropriate)
  - could be a formula (like `Li2O`), a list of formulas, a chemical system (like `Ba-Zr-S`), or a list of chemical systems
  - used by `get_query`
- `TRANSFORM_STRUCS`
  - need to specify this if you want to systematically modify structures in `get_strucs`
  - e.g., you might make this `[1, 2, 3, 4]` if looping through that will be meaningful during `get_strucs` (e.g., to insert 1, 2, 3, 4 Li)
  - used by `get_strucs`
- `GEN_MAGMOMS`
  - set to True if running AFM calcs
  - otherwise, not used
  - if True, `get_magmoms` is executed
- `TO_LAUNCH`
  - tells `pydmclab` what kinds of calculations (i.e., standards and exchange-correlation methods) you're interested in.
  - a common setting would be `TO_LAUNCH = {'dmc' : ['metagga']}` to run r2SCAN (meta-GGA) at DMC standards

## a typical flow

1. I decide I want to run some calculations
2. I create a folder on MSI: `/home/cbartel/<username>/<specific name>` (e.g., `/home/cbartel/cbartel/LiMn2O4`)
3. I create a folder called `scripts` (e.g., `/home/cbartel/cbartel/LiMn2O4/scripts`)
4. I copy a demo launcher script into this folder (e.g., `cp ~/bin/pydmclab/pydmclab/demos/demo_vasp_template.py /home/cbartel/cbartel/LiMn2O4/scripts/launcher.py`)
5. I start editing `launcher.py`

- start at the top
- what compositions do I want to query for (if any)?
- do I want to run AFM or not?
- am I going to transform my structures (e.g., replace species)?
- what kind of standard/functional do I want?
- what kind of configs do I want to use?
  - while developing the launcher script, you should leave `n_procs = 1` in both the `SUB_CONFIGS` and the `ANALYSIS_CONFIGS` so that you can execute `python launcher.py` from the login node to quickly inspect what is printed
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

13. All right, now I'll make sure `n_procs` in `SUB_CONFIGS` is the same as `n_procs` in `ANALYSIS_CONFIGS` and that they're both the same as `--ntasks` in `sub_launcher.sh`

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
