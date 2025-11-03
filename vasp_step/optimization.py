# -*- coding: utf-8 -*-

"""Non-graphical part of the Optimization step in a VASP flowchart"""

import logging
from pathlib import Path
import pkg_resources
import textwrap

from tabulate import tabulate

import vasp_step
import molsystem
import seamm
import seamm_util.printing as printing
from seamm_util.printing import FormattedText as __

# In addition to the normal logger, two logger-like printing facilities are
# defined: "job" and "printer". "job" send output to the main job.out file for
# the job, and should be used very sparingly, typically to echo what this step
# will do in the initial summary of the job.
#
# "printer" sends output to the file "step.out" in this steps working
# directory, and is used for all normal output from this step.

logger = logging.getLogger(__name__)
job = printing.getPrinter()
printer = printing.getPrinter("VASP")

# Add this module's properties to the standard properties
path = Path(pkg_resources.resource_filename(__name__, "data/"))
csv_file = path / "properties.csv"
if path.exists():
    molsystem.add_properties_from_file(csv_file)


class Optimization(vasp_step.Energy):
    """
    The non-graphical part of a Optimization step in a flowchart.

    Attributes
    ----------
    parser : configargparse.ArgParser
        The parser object.

    options : tuple
        It contains a two item tuple containing the populated namespace and the
        list of remaining argument strings.

    subflowchart : seamm.Flowchart
        A SEAMM Flowchart object that represents a subflowchart, if needed.

    parameters : OptimizationParameters
        The control parameters for Optimization.

    See Also
    --------
    TkOptimization,
    Optimization, OptimizationParameters
    """

    def __init__(
        self, flowchart=None, title="Optimization", extension=None, logger=logger
    ):
        """A substep for Optimization in a subflowchart for VASP.

        You may wish to change the title above, which is the string displayed
        in the box representing the step in the flowchart.

        Parameters
        ----------
        flowchart: seamm.Flowchart
            The non-graphical flowchart that contains this step.

        title: str
            The name displayed in the flowchart.
        extension: None
            Not yet implemented
        logger : Logger = logger
            The logger to use and pass to parent classes

        Returns
        -------
        None
        """
        logger.debug(f"Creating Optimization {self}")

        super().__init__(
            flowchart=flowchart,
            title=title,
            extension=extension,
            logger=logger,
        )

        self._calculation = "Optimization"
        self._model = None
        self._metadata = vasp_step.metadata
        self.parameters = vasp_step.OptimizationParameters()

    @property
    def header(self):
        """A printable header for this section of output"""
        return "Step {}: {}".format(".".join(str(e) for e in self._id), self.title)

    @property
    def version(self):
        """The semantic version of this module."""
        return vasp_step.__version__

    @property
    def git_revision(self):
        """The git version of this module."""
        return vasp_step.__git_revision__

    def description_text(self, P=None):
        """Create the text description of what this step will do.
        The dictionary of control values is passed in as P so that
        the code can test values, etc.

        Parameters
        ----------
        P: dict
            An optional dictionary of the current values of the control
            parameters.
        Returns
        -------
        str
            A description of the current step.
        """
        if not P:
            P = self.parameters.values_to_dict()

        result = ""
        if self._calculation == "Optimization":
            result = self.header + "\n"

        method = P["optimization method"]
        text = "The structure will be optimized using the {optimization method}"
        text += " until the RMS force is below about {convergence cutoff} with a"
        text += " limit of {number of steps} steps."
        tmp = P["optimize atom positions"]
        atoms = tmp == "yes" if isinstance(tmp, str) else tmp
        tmp = P["optimize cell shape"]
        shape = tmp == "yes" if isinstance(tmp, str) else tmp
        tmp = P["optimize cell volume"]
        volume = tmp == "yes" if isinstance(tmp, str) else tmp

        if self.is_expr(atoms) or self.is_expr(shape) or self.is_expr(volume):
            if self.is_expr(atoms):
                text += " '{optimize atom positions}' will determine whether to"
                text += " optimize the atoms positions."
            elif atoms:
                text += " The atom positions will be relaxed."
            if self.is_expr(shape):
                text += " '{optimize cell shape}' will determine whether to"
                text += " optimize the shape of the cell."
            elif shape:
                text += " The shape of the cell will be relaxed."
            if self.is_expr(volume):
                text += " '{optimize cell volume}' will determine whether to"
                text += " optimize the volume of the cell."
            elif volume:
                text += " The volume of the cell will be relaxed."
        else:
            if atoms and shape and volume:
                text += " The cell and atom positions will be fully relaxed."
            elif atoms:
                text += " The atom positions"
                if shape:
                    text += " and cell shape"
                elif volume:
                    text += " and cell volume"
                text += " will be optimized."
            else:
                if shape:
                    text += " The cell shape"
                    if volume:
                        text += " and volume"
                elif volume:
                    text += " The cell volume"
                text += " will be optimized, but the fractional coordinates of the "
                text += "atoms will be fixed."

        text += "\n"
        if self.is_expr(method):
            text += "The details of the algorithm used will be established at runtime."
        elif method == "RMM-DIIS":
            text += "The RMM-DIIS method will scale the steps by {step scale factor}"
            history = P["iteration history length"]
            if self.is_expr(history):
                text += " with {iteration history length} controlling how many "
                text += "previous steps will be used in determining the next step."
            elif history == "default":
                text += " and VASP will decide how many previous steps to use, based on"
                text += " the eigenvalues of the inverse approximate Hessian matrix."
            else:
                text += " and use {iteration history length} previous steps. "
                text += "For large systems you may find that values of 10-20 work well."
        elif method == "Conjugate Gradients":
            text += "The conjugate gradients method will scale the steps "
            text += "by {step scale factor}."
        elif method == "Damped MD":
            text += "The damped MD will "
            approach = P["damped MD approach"]
            if self.is_expr(approach):
                text += f"damp or quench the velocity based on '{approach}'."
            elif approach == "damping":
                text += "scale the previous velocity by a factor of "
                text += "{velocity scale factor} and the force by a factor of"
                text += "{force scale factor}"
            else:
                text += "scale the force by a factor of {velocity quenching factor}"
                text += " and zeroing the previous velocity unless it points in the "
                text += "same direction as the force."
        else:
            text += f"Warning: the optimization method '{method}' was not recognized."

        text += " The details of the model and electronic calculation are as follows."
        result += __(text, **P, indent=4 * " ").__str__()

        result += "\n\n"
        result += super().description_text(P)

        return result

    def get_keywords(self, P=None):
        """Get the keywords and values for the calculation."""
        # Get the values of the parameters, dereferencing any variables
        if P is None:
            P = self.parameters.current_values_to_dict(
                context=seamm.flowchart_variables._data
            )

        # Get the keywords from the energy class
        keywords, descriptions = super().get_keywords(P=P)

        # The definition of the type of calculation: optimization
        keywords["ISIF"] = 3 if P["calculate stress"] else 0
        keywords["NSW"] = P["number of steps"]
        ediffg = P["convergence cutoff"].m_as("eV/Å")
        keywords["EDIFFG"] = f"-{ediffg**2:.1E}"

        method = P["optimization method"]
        if "DIIS" in method:
            keywords["IBRION"] = 1
            keywords["POTIM"] = P["step scale factor"]
            if P["iteration history length"] != "default":
                keywords["NFREE"] = P["iteration history length"]
        elif "MD" in method:
            keywords["IBRION"] = 3
            if P["damped MD approach"] == "damping":
                keywords["SMASS"] = P["force scale factor"]
                keywords["POTIM"] = 2 * (1 - float(P["velocity scale factor"]))
            else:
                keywords["POTIM"] = -float(P["velocity quenching factor"])
        else:  # Conjugate gradients
            keywords["IBRION"] = 2
            keywords["POTIM"] = P["step scale factor"]

        # Work out ISIF
        xyz = P["optimize atom positions"]
        shape = P["optimize cell shape"]
        volume = P["optimize cell volume"]
        if xyz:
            if shape:
                if volume:
                    isif = 3
                else:
                    isif = 4
            else:
                if volume:
                    isif = 8
                else:
                    match P["calculate stress"]:
                        case "no":
                            isif = 0
                        case "only pressure":
                            isif = 1
                        case _:
                            isif = 2
        else:
            if shape:
                if volume:
                    isif = 6
                else:
                    isif = 5
            else:
                if volume:
                    isif = 7
                else:
                    raise RuntimeError("Not optimizing any degrees of freedom!")
        keywords["ISIF"] = isif

        # Check for default for NELMIN and set if needed
        if P["nelmin"] == "default":
            keywords["NELMIN"] = 6

        return keywords, descriptions

    def analyze(
        self,
        P=None,
        configuration=None,
        indent="",
        text="",
        table=None,
        results={},
        **kwargs,
    ):
        """Do any analysis of the output from this step.

        Also print important results to the local step.out file using
        "printer".

        Parameters
        ----------
        indent: str
            An extra indentation for the output
        """
        if table is None:
            table = {
                "Property": [],
                "Value": [],
                "Units": [],
            }

        # Extract the data we need from the output files.
        hdf5_file = self.wd / "vaspout.h5"
        xml_file = self.wd / "vasprun.xml"
        if hdf5_file.exists():
            results = self.parse_hdf5(hdf5_file)
        elif xml_file.exists():
            results = self.parse_xml(hdf5_file)
        else:
            results = {}
            text = f"Something is very wrong! Cannot find either {hdf5_file} or "
            text += f"{xml_file}. Did VASP fail?"
            printer.normal(textwrap.indent(text, self.indent + 4 * " "))
            printer.normal("")
            return results

        table["Property"].append("Number of steps")
        table["Value"].append(results["nOptimizationSteps"])
        table["Units"].append("")

        # Update the configuration with the final cell and fractionals
        configuration.cell.parameters = results["cell,iter"][-1]
        # Reorder the atoms back to SEAMM order
        tmp = results["fractionals,iter"][-1]
        configuration.atoms.coordinates = [tmp[i] for i in self.to_VASP_order]

        super().analyze(
            P=P,
            configuration=configuration,
            indent=indent,
            text=text,
            table=table,
            results=results,
            **kwargs,
        )

        # Print the change in the cell
        a0, b0, c0, alpha0, beta0, gamma0 = results["cell,iter"][0]
        V0 = results["V,iter"][0]
        a, b, c, alpha, beta, gamma = results["cell,iter"][-1]
        V = results["V,iter"][-1]
        ctable = {
            "": ("𝗮", "𝗯", "𝗰", "𝞪", "𝞫", "𝞬", "V"),
            "Initial": (
                f"{a0:.3f}",
                f"{b0:.3f}",
                f"{c0:.3f}",
                f"{alpha0:.1f}",
                f"{beta0:.1f}",
                f"{gamma0:.1f}",
                f"{V0:.1f}",
            ),
            "Final": (
                f"{a:.3f}",
                f"{b:.3f}",
                f"{c:.3f}",
                f"{alpha:.1f}",
                f"{beta:.1f}",
                f"{gamma:.1f}",
                f"{V:.1f}",
            ),
            "Change": (
                f"{a - a0:.3f}",
                f"{b - b0:.3f}",
                f"{c - c0:.3f}",
                f"{alpha - alpha0:.1f}",
                f"{beta - beta0:.1f}",
                f"{gamma - gamma0:.1f}",
                f"{V - V0:.1f}",
            ),
            "Units": ("Å", "Å", "Å", "°", "°", "°", "Å\N{SUPERSCRIPT THREE}"),
        }

        tmp = tabulate(
            ctable,
            headers="keys",
            tablefmt="rounded_outline",
            colalign=("center", "decimal", "decimal", "decimal", "center"),
            disable_numparse=True,
        )
        length = len(tmp.splitlines()[0])
        text_lines = []
        text_lines.append("Cell Parameters".center(length))
        text_lines.append(tmp)

        printer.normal(textwrap.indent("\n".join(text_lines), self.indent + 7 * " "))
        printer.normal("")

        # Print the change in the stress
        (xx0, yy0, zz0, yz0, xz0, xy0) = results["stress,iter"][0]
        (xx, yy, zz, yz, xz, xy) = results["stress,iter"][-1]
        ctable = {
            "": ("xx", "yy", "zz", "yz", "xz", "xy"),
            "Initial": (
                f"{xx0:.4f}",
                f"{yy0:.4f}",
                f"{zz0:.4f}",
                f"{yz0:.4f}",
                f"{xz0:.4f}",
                f"{xy0:.4f}",
            ),
            "Final": (
                f"{xx:.4f}",
                f"{yy:.4f}",
                f"{zz:.4f}",
                f"{yz:.4f}",
                f"{xz:.4f}",
                f"{xy:.4f}",
            ),
            "Change": (
                f"{xx - xx0:.4f}",
                f"{yy - yy0:.4f}",
                f"{zz - zz0:.4f}",
                f"{yz - yz0:.4f}",
                f"{xz - xz0:.4f}",
                f"{xy - xy0:.4f}",
            ),
            "Units": ("GPa",) * 6,
        }

        tmp = tabulate(
            ctable,
            headers="keys",
            tablefmt="rounded_outline",
            colalign=("center", "decimal", "decimal", "decimal", "center"),
            disable_numparse=True,
        )
        length = len(tmp.splitlines()[0])
        text_lines = []
        text_lines.append("Stress".center(length))
        text_lines.append(tmp)

        printer.normal(textwrap.indent("\n".join(text_lines), self.indent + 7 * " "))
        printer.normal("")
