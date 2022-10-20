# pydmc
- writing a common framework the group can use to do routine things related to running and processing DFT calculations
- a more organized version of compmatscipy
- meant to leverage pymatgen w/ more ease of use due to the limited scope


## CompTools

#### Summary
- for handling and manipulating chemical formulas
- built around the base convention that we usually want formulas that are:
    - sorted by element
    - have "1"s 
    - don't have decimals
    - are reduced

#### TO DO
- experiment + add capabilities


## StrucTools

#### Summary:
- for manipulating and analyzing crystal structures

#### TO DO:
- getting structures from MP
- making supercells
- making slabs
- assigning oxidation states
- ordering disordered structures
- creating defects
- unit tests

## QueryMP

#### Summary:
- for retrieving typical data from Materials Project

#### TO DO:
- unit tests

## VASPTools
#### Summary:
- for setting up and analyzing typical DFT calculations (e.g., geometry optimizations)
### VASPSetUp
#### Summary:
- for setting up DFT calculations

#### TO DO:
- clean up compmatscipy version
- leverage pymatgen a bit more
- unit tests

### VASPAnalysis
#### Summary:
- for analyzing DFT calculations (mainly convergence, energies)

#### TO DO:
- clean up compmatscipy version
- leverage pymatgen a bit more
- unit tests

## JobSubmission
#### Summary:
- for figuring out which jobs to run and running them

#### TO DO:
- clean up compmatscipy version
- leverage pymatgen a bit more

## ThermoTools
- for computing thermodynamic properties

## DOSTools
- for analyzing DOS-like objects
### DOSAnalysis
- for analyzing density of states
### LOBSTERAnalysis
- for analyzing COOP, COHP, etc

## DiffusionTools
### NEBSetUp
- for setting up NEB calculations
### NEBAnalysis
- for analyzing NEB calculations
### AIMDSetUp
- for setting up AIMD calculations
### AIMDAnalysis
- for analyzing AIMD calculations

## utils
### handy
### plotting