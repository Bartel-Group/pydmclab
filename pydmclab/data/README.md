elemental_reference_energies_0K
    - created by Chris B on MSI at ~/projects/mus
    - MP and DMC standard calculations for all elements using PBE and R2SCAN

elemental_gibbs_energies_T
    - Gels.json from [here](https://github.com/CJBartel/predict-gibbs-energies)
    - from Chris B's [paper](https://www.nature.com/articles/s41467-018-06682-4)

omat24_mp2020-style_compatibility.yaml
    - OMat24Compatibility.yaml from [here](https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/data/omat/entries/OMat24Compatibility.yaml)
    - MP2020-style anion + GGA/GGA+U mixing corrections fit for OMat24-style calculations
    - loaded w/ thermochem.omat24_mp2020_style_compatibility_yaml(); used by core.energies.UMAFormationEnergy

uma-s1p2p1-mp_elemental_references.json
    - created by AJ Lannerd on MSI
    - UMA total energies (eV/atom) for the elemental references of 89 elements
    - UMA settings: model = uma-s-1p2p1, task = omat, atoms + cell relaxed, fmax < 0.03 eV/A
    - loaded w/ thermochem.uma_s1p2p1_mp_elemental_references(); used by core.energies.UMAFormationEnergy

uma-s1p2p1-mp_formation_energies.json
    - created by AJ Lannerd on MSI
    - UMA formation energies (eV/atom) for 81,308 compounds from MP structures, one per formula
        - only cmpds within 200 meV/atom of hull were considered
        - corrected w/ omat24_mp2020-style_compatibility.yaml, referenced to uma-s1p2p1-mp_elemental_references.json
    - UMA settings: model = uma-s-1p2p1, task = omat, atoms + cell relaxed, fmax < 0.03 eV/A
    - loaded w/ thermochem.uma_s1p2p1_mp_formation_energies(); used by core.hulls.UMAHull