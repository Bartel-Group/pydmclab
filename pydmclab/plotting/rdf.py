from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from pymatgen.analysis.diffusion.aimd.rdf import RadialDistributionFunctionFast
from pydmclab.core.struc import StrucTools
from pydmclab.plotting.utils import set_rc_params, get_colors

set_rc_params()


class PlotRDF(object):
    """
    Wrapper for pymatgen.analysis.diffusion.aimd.rdf.RadialDistributionFunctionFast

    Used to plot the radial distribution function (RDF) of a structure.
    """

    def __init__(
        self,
        structures: (
            Structure | tuple[Structure] | dict | tuple[dict] | str | tuple[str]
        ),
        min_radius: float = 0.0,
        max_radius: float = 10.0,
        n_grid: int = 101,
        sigma: float = 0.05,
    ) -> None:
        """
        Args:
            structures: pymatgen Structure objects, Structure.as_dict(), or paths to structure file
            min_radius: minimum radius
            max_radius: maximum radius
            n_grid: number of grid points
            sigma: smoothing factor (Gaussian smoothing)
        """

        self.strucs = [StrucTools(struc).structure for struc in structures]
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.n_grid = n_grid
        self.sigma = sigma

        self.rdf = RadialDistributionFunctionFast(
            structures=self.strucs,
            rmin=self.min_radius,
            rmax=self.max_radius,
            sigma=self.sigma,
            ngrid=self.n_grid,
        )

    def get_rdf(
        self,
        ref_species: str | list[str],
        species: str | list[str],
        average_over_strucs: bool = False,
    ) -> tuple[np.ndarray, np.ndarray[np.ndarray]] | tuple[np.ndarray, np.ndarray]:
        """
        Args:
            ref_species: RDFs are calculated with respect to this species (they are "centered")
            species: species to calculate RDFs for
            average_over_strucs: if True, RDFs are averaged over all structures
        Returns:
            (radii, [rdfs]) if average_over_strucs is False, else (radii, rdf)
        """
        return self.rdf.get_rdf(
            ref_species=ref_species, species=species, is_average=average_over_strucs
        )

    def plot_rdf(
        self,
        ref_species: str | list[str],
        species: str | list[str],
        rdfs_to_plot: str | int | tuple[int, int] = "all",
        average_over_strucs: bool = False,
        savename: str | None = None,
        show: bool = False,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Args:
            rdf_to_plot: "all" or index of (or starting and ending indices of) RDF to plot
            ref_species: RDFs are calculated with respect to this species (they are "centered")
            species: species to calculate RDFs for
            average_over_strucs: if True, yields single RDF averaged over all structures
            savename: save the plot to specified file location if not None
            show: whether tp display the plot
            **kwargs: additional keyword arguments for customizing the plot
        Returns:
            fig, ax: figure and axis objects of the plot
        """

        radii, all_rdfs = self.get_rdf(
            ref_species=ref_species,
            species=species,
            average_over_strucs=average_over_strucs,
        )

        if average_over_strucs:
            rdfs = [all_rdfs]
        elif rdfs_to_plot == "all":
            rdfs = all_rdfs
        elif isinstance(rdfs_to_plot, int):
            rdfs = [all_rdfs[rdfs_to_plot]]
        else:
            rdfs = all_rdfs[rdfs_to_plot[0] : rdfs_to_plot[1] + 1]

        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 6)))

        colors = list(get_colors("tab10").values())
        lw = kwargs.get("lw", 2)

        for idx, rdf in enumerate(rdfs):
            color = colors[idx % len(colors)]
            if average_over_strucs:
                label = f"({ref_species}) - ({species}) (Averaged)"
            elif len(rdfs) == 1:
                label = f"({ref_species}) - ({species})"
            else:
                label = f"({ref_species}) - ({species}) (Struc {idx})"
            ax.plot(radii, rdf, lw=lw, color=color, label=label)

        ax.set_xlim(kwargs.get("xlim", (self.min_radius, 1.1 * self.max_radius)))
        ax.set_ylim(kwargs.get("ylim", (0, 1.1 * np.max(rdfs))))
        xticks = kwargs.get(
            "xticks", np.arange(self.min_radius, 1.1 * self.max_radius, 10)
        )
        if xticks is not None:
            ax.set_xticks(xticks)
        yticks = kwargs.get("yticks", np.arange(0, 1.1 * np.max(rdfs), 10))
        if yticks is not None:
            ax.set_yticks(yticks)

        ax.set_label(kwargs.get("xlabel", "Distance (Å)"))
        ax.set_ylabel(kwargs.get("ylabel", "g(r)"))

        ax.legend()

        if kwargs.get("grid", True):
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if savename is not None:
            fig.savefig(savename)

        if show:
            plt.show()

        plt.close(fig)

        return fig, ax
