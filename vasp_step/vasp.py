# -*- coding: utf-8 -*-

"""Non-graphical part of the VASP step in a SEAMM flowchart"""

import json
import logging
from pathlib import Path
import pkg_resources
import pprint  # noqa: F401
import sys
import time

import vasp_step
import molsystem
import seamm
from seamm_util import ureg, Q_, getParser, CompactJSONEncoder  # noqa: F401
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


class VASP(seamm.Node):
    """
    The non-graphical part of a VASP step in a flowchart.

    Attributes
    ----------
    parser : configargparse.ArgParser
        The parser object.

    options : tuple
        It contains a two item tuple containing the populated namespace and the
        list of remaining argument strings.

    subflowchart : seamm.Flowchart
        A SEAMM Flowchart object that represents a subflowchart, if needed.

    parameters : VASPParameters
        The control parameters for VASP.

    See Also
    --------
    TkVASP,
    VASP, VASPParameters
    """

    def __init__(
        self,
        flowchart=None,
        title="VASP",
        namespace="org.molssi.seamm.vasp",
        extension=None,
        logger=logger,
    ):
        """A step for VASP in a SEAMM flowchart.

        You may wish to change the title above, which is the string displayed
        in the box representing the step in the flowchart.

        Parameters
        ----------
        flowchart: seamm.Flowchart
            The non-graphical flowchart that contains this step.

        title: str
            The name displayed in the flowchart.
        namespace : str
            The namespace for the plug-ins of the subflowchart
        extension: None
            Not yet implemented
        logger : Logger = logger
            The logger to use and pass to parent classes

        Returns
        -------
        None
        """
        logger.debug(f"Creating VASP {self}")
        self.subflowchart = seamm.Flowchart(
            parent=self, name="VASP", namespace=namespace
        )  # yapf: disable

        super().__init__(
            flowchart=flowchart,
            title="VASP",
            extension=extension,
            module=__name__,
            logger=logger,
        )  # yapf: disable

        self._potential_metadata = None

    @property
    def version(self):
        """The semantic version of this module."""
        return vasp_step.__version__

    @property
    def git_revision(self):
        """The git version of this module."""
        return vasp_step.__git_revision__

    @property
    def potential_metadata(self):
        """The metadata data for the potentials."""
        # Make sure we have the information about the PAW potentials
        if self._potential_metadata is None:
            directory = Path("~/SEAMM/Parameters/VASP").expanduser()
            index = directory / "index.json"
            if index.exists():
                with index.open() as fd:
                    self._potential_metadata = json.load(fd)
            else:
                self._potential_metadata = self.catalog_potentials()
                with index.open("w") as fd:
                    json.dump(
                        self.potential_metadata,
                        fd,
                        indent=4,
                        sort_keys=True,
                        cls=CompactJSONEncoder,
                    )
                print(f"Wrote the index to {index}")
        return self._potential_metadata

    def set_id(self, node_id):
        """Set the id for node to a given tuple"""
        self._id = node_id

        # and set our subnodes
        self.subflowchart.set_ids(self._id)

        return self.next()

    def create_parser(self):
        """Setup the command-line / config file parser"""
        parser_name = self.step_type
        parser = getParser()

        # Remember if the parser exists ... this type of step may have been
        # found before
        parser_exists = parser.exists(parser_name)

        # Create the standard options, e.g. log-level
        result = super().create_parser(name=parser_name)

        if parser_exists:
            return result

        # VASP specific options
        parser.add_argument(
            parser_name,
            "--ncores",
            default="available",
            help=(
                "The maximum number of cores to use for VASP. "
                "Default: all available cores."
            ),
        )
        parser.add_argument(
            parser_name,
            "--graph-formats",
            default=tuple(),
            choices=("html", "png", "jpeg", "webp", "svg", "pdf"),
            nargs="+",
            help="extra formats to write for graphs",
        )
        parser.add_argument(
            parser_name,
            "--graph-fontsize",
            default=15,
            help="Font size in graphs, defaults to 15 pixels",
        )
        parser.add_argument(
            parser_name,
            "--graph-width",
            default=1024,
            help="Width of graphs in formats that support it, defaults to 1024",
        )
        parser.add_argument(
            parser_name,
            "--graph-height",
            default=1024,
            help="Height of graphs in formats that support it, defaults to 1024",
        )

        return result

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
        self.subflowchart.root_directory = self.flowchart.root_directory

        # Get the first real node
        node = self.subflowchart.get_node("1").next()

        text = self.header + "\n\n"
        while node is not None:
            try:
                text += __(node.description_text(), indent=3 * " ").__str__()
            except Exception as e:
                print(f"Error describing vasp flowchart: {e} in {node}")
                logger.critical(f"Error describing vasp flowchart: {e} in {node}")
                raise
            except:  # noqa: E722
                print(
                    "Unexpected error describing vasp flowchart: {} in {}".format(
                        sys.exc_info()[0], str(node)
                    )
                )
                logger.critical(
                    "Unexpected error describing vasp flowchart: {} in {}".format(
                        sys.exc_info()[0], str(node)
                    )
                )
                raise
            text += "\n"
            node = node.next()

        return text

    def run(self):
        """Run a VASP step.

        Parameters
        ----------
        None

        Returns
        -------
        seamm.Node
            The next node object in the flowchart.
        """
        next_node = super().run(printer)

        # Print our header to the main output
        printer.normal(self.header)
        printer.normal("")

        # Get the first real node
        node = self.subflowchart.get_node("1").next()

        while node is not None:
            node.run()
            # Loop to next node
            node = node.next()

        return next_node

    def analyze(self, indent="", **kwargs):
        """Do any analysis of the output from this step.

        Also print important results to the local step.out file using
        "printer".

        Parameters
        ----------
        indent: str
            An extra indentation for the output
        """
        # Get the first real node
        node = self.subflowchart.get_node("1").next()

        # Loop over the subnodes, asking them to do their analysis
        while node is not None:
            for value in node.description:
                printer.important(value)
                printer.important(" ")

            node.analyze()

            node = node.next()

    def catalog_potentials(self):
        """Create a catalog of the PAW potentials."""
        t0 = time.time_ns()
        directory = Path("~/SEAMM/Parameters/VASP").expanduser()
        print(f"Potentials directory is {directory}")
        paths = directory.glob("**/POTCAR")
        data = {}
        for n, path in enumerate(paths, start=1):
            _type, potential = path.parts[-3:-1]
            # print(f"{_type} -- {potential} {path}")
            result = self.parse_potcar(path)
            result["element"] = potential.split("_")[0]
            result["file"] = str(path)
            # pprint.pprint(result)
            if _type not in data:
                data[_type] = {}
            if potential in data[_type]:
                raise ValueError(f"Potential {_type} / {potential} is duplicated!")
            data[_type][potential] = result
        t = (time.time_ns() - t0) / 1.0e9
        print(f"Took {t:.3f} seconds to process {n} POTCAR files")

        return data

    def parse_potcar(self, path):
        """Read a POTCAR file and extract pertinent information.

        Parameters
        ----------
        path : pathlib.Path
            The path to the POTCAR file

        Returns
        -------
        results : dict(str, str)
            A dictionary of the data from the POTCAR file
        """
        result = {}
        lines = iter(path.read_text().splitlines())
        line1 = next(lines).strip()
        for line in lines:
            line = line.strip()
            if line.startswith("TITEL"):
                result["title"] = line.split("=")[1].strip()
                if result["title"] != line1:
                    print(
                        f"Line 1 and the title differ:\n\t{line1}\n\t{result['title']}"
                    )
            elif line.startswith("EATOM"):
                result["Eatom"] = line.split("=")[1].split()[0].strip()
            elif line.startswith("POMASS"):
                tmp = line.split(";")
                result["mass"] = tmp[0].split("=")[1].strip()
                result["zval"] = tmp[1].split("=")[1].split()[0].strip()
            elif line.startswith("ENMAX"):
                tmp = line.split(";")
                result["Emax"] = tmp[0].split("=")[1].strip()
                result["Emin"] = tmp[1].split("=")[1].split()[0].strip()
            elif line == "Atomic configuration":
                line = next(lines).strip()
                n_orbitals = int(line.split()[0])
                next(lines)
                orbitals = []
                for i in range(n_orbitals):
                    line = next(lines).strip()
                    n, l, j, E, occupation = line.split()
                    n = int(n)
                    l = int(l)  # noqa=E741
                    E = float(E)
                    occupation = occupation.rstrip("0.")
                    if occupation == "":
                        occupation = "0"
                    orbitals.append((n, l, j, E, occupation))
                result["orbitals"] = orbitals

                # Find the occupied valence orbitals
                electrons = 0
                symbols = []
                for n, l, j, E, occupation in orbitals[::-1]:
                    Z = float(result["zval"])
                    occ = float(occupation)
                    if occ != 0:
                        electrons += occ
                        symbol = f"{n}{('s', 'p', 'd', 'f')[l]}{occupation}"
                        symbols.append(symbol)
                        if electrons >= Z:
                            break
                result["configuration"] = " ".join(symbols[::-1])
                break
        return result
