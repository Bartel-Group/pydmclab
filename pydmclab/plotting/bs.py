from pymatgen.electronic_structure.plotter import BSDOSPlotter
from pymatgen.io.lobster import Fatband, Doscar
import os


def fig_bs(bs_dir, fig_dir, savename):
    """
    After a bandstructure calculation, plot the fatband element-projected bandstructure.
    """
    fpng = os.path.join(fig_dir, savename)
    fvasprun = os.path.join(bs_dir, "vasprun.xml")
    fkpoints = os.path.join(bs_dir, "KPOINTS")
    bssymline = Fatband(
        filenames=bs_dir, vasprun=fvasprun, Kpointsfile=fkpoints
    ).get_bandstructure()
    BSDOSPlotter(bs_projection="elements").get_plot(bs=bssymline).savefig(fpng)
