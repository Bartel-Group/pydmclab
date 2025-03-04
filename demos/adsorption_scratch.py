from pymatgen.core import Structure, Lattice, Molecule
from pymatgen.core.surface import SlabGenerator
from pymatgen.analysis.adsorption import *
from pymatgen.core.surface import generate_all_slabs
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.structure import Structure
from pymatgen.core.surface import SlabGenerator
from matplotlib import pyplot as plt
from pydmclab.utils.handy import read_json, write_json
import os


ru_001 = Structure.from_file("/Users/christopherakiki/scratch/Ru_001.cif")

miller_index = (0, 0, 1)
min_slab_size = 10.0
min_vacuum_size = 3.0

slabgen = SlabGenerator(
    initial_structure=ru_001, 
    miller_index=miller_index,
    min_slab_size=min_slab_size,
    min_vacuum_size=min_vacuum_size,
    center_slab=True,
)

ru_001_slab = slabgen.get_slabs()[0]  # Returns a list, usually just one slab

asf_ru_001_slab = AdsorbateSiteFinder(ru_001_slab)
ads_sites = asf_ru_001_slab.find_adsorption_sites()
assert len(ads_sites) == 4
print(ads_sites)
fig = plt.figure()
ax = fig.add_subplot(111)
plot_slab(ru_001_slab, ax, adsorption_sites=True)


print(ads_sites['all'])

# Adding adsorbate
site = ads_sites["all"][0]

adsorbate = Molecule("O", [[0, 0, 0]])

# Add the adsorbate to the slab at the chosen site
adsorbed_structure = asf_ru_001_slab.add_adsorbate(adsorbate, site,repeat = [2,2,1])

adsorbed_structure.to(fmt="cif", filename="/users/christopherakiki/scratch/adsorbed_structure.cif")