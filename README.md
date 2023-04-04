# pydmclab

- common framework to rely upon for typical calculations and analysis done in the Bartel research group

## Installation instructions

- there will always be two versions of pydmclab available:
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

- you should only have to do this one time, not for each successive installation of pydmclab
