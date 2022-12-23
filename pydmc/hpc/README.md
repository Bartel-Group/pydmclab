# pydmc/hpc
This group of modules are meant to be run either on HPC

## analyze
- analyze VASP output files in a directory where VASP was executed

## launch
- launch a series of VASP calculations
    - for instance, if you were running all compounds in an A-B-C phase diagram using the same protocol, you could use this to launch all of the calculations

## submit
- submit a single chain of VASP calculations
    - i.e., for a given structure, this would be used to prepare and execute a submission file that would launch the correct sequence of VASP calculations (e.g., loose --> relax --> static)

## vasp
- prepare VASP input files for a specified calculation
    - also handles errors for re-launching