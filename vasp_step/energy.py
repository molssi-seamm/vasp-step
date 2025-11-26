# -*- coding: utf-8 -*-

"""Non-graphical part of the Energy step in a VASP flowchart"""

from collections import Counter
import configparser
import csv
from datetime import datetime, timezone
import importlib
import logging
from math import isnan, ceil as ceiling
from pathlib import Path
import pkg_resources
import platform
import pprint  # noqa: F401
import shutil
import textwrap
import time

from cpuinfo import get_cpu_info
import h5py
from lxml import etree
import numpy as np
from numpy import linalg as LA
import pandas
from tabulate import tabulate

import vasp_step  # noqa: E999
import molsystem
import seamm
import seamm_exec
from seamm_util import ureg, Q_, CompactJSONEncoder, Configuration  # noqa: F401
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


def humanize(memory, suffix="B", kilo=1024):
    """
    Scale memory to its proper format e.g:

        1253656 => '1.20 MiB'
        1253656678 => '1.17 GiB'
    """
    if kilo == 1000:
        units = ["", "k", "M", "G", "T", "P"]
    elif kilo == 1024:
        units = ["", "Ki", "Mi", "Gi", "Ti", "Pi"]
    else:
        raise ValueError("kilo must be 1000 or 1024!")

    for unit in units:
        if memory < 10 * kilo:
            return f"{int(memory)}{unit}{suffix}"
        memory /= kilo


def dehumanize(memory, suffix="B"):
    """
    Unscale memory from its human readable form e.g:

        '1.20 MB' => 1200000
        '1.17 GB' => 1170000000
    """
    units = {
        "": 1,
        "k": 1000,
        "M": 1000**2,
        "G": 1000**3,
        "P": 1000**4,
        "Ki": 1024,
        "Mi": 1024**2,
        "Gi": 1024**3,
        "Pi": 1024**4,
    }

    tmp = memory.split()
    if len(tmp) == 1:
        return memory
    elif len(tmp) > 2:
        raise ValueError("Memory must be <number> <units>, e.g. 1.23 GB")

    amount, unit = tmp
    amount = float(amount)

    for prefix in units:
        if prefix + suffix == unit:
            return int(amount * units[prefix])

    raise ValueError(f"Don't recognize the units on '{memory}'")


_subscript = {
    "0": "\N{SUBSCRIPT ZERO}",
    "1": "\N{SUBSCRIPT ONE}",
    "2": "\N{SUBSCRIPT TWO}",
    "3": "\N{SUBSCRIPT THREE}",
    "4": "\N{SUBSCRIPT FOUR}",
    "5": "\N{SUBSCRIPT FIVE}",
    "6": "\N{SUBSCRIPT SIX}",
    "7": "\N{SUBSCRIPT SEVEN}",
    "8": "\N{SUBSCRIPT EIGHT}",
    "9": "\N{SUBSCRIPT NINE}",
}


def subscript(n):
    """Return the number using Unicode subscript characters."""
    return "".join([_subscript[c] for c in str(n)])


middot = "\N{MIDDLE DOT}"
lDelta = "\N{GREEK CAPITAL LETTER DELTA}"
one_half = "\N{VULGAR FRACTION ONE HALF}"
degree_sign = "\N{DEGREE SIGN}"
standard_state = {
    "H": f"{one_half}H{subscript(2)}(g)",
    "He": "He(g)",
    "Li": "Li(s)",
    "Be": "Be(s)",
    "B": "B(s)",
    "C": "C(s,gr)",
    "N": f"{one_half}N{subscript(2)}(g)",
    "O": f"{one_half}O{subscript(2)}(g)",
    "F": f"{one_half}F{subscript(2)}(g)",
    "Ne": "Ne(g)",
    "Na": "Na(s)",
    "Mg": "Mg(s)",
    "Al": "Al(s)",
    "Si": "Si(s)",
    "P": "P(s)",
    "S": "S(s)",
    "Cl": f"{one_half}Cl{subscript(2)}(g)",
    "Ar": "Ar(g)",
    "K": "K(s)",
    "Ca": "Ca(s)",
    "Sc": "Sc(s)",
    "Ti": "Ti(s)",
    "V": "V(s)",
    "Cr": "Cr(s)",
    "Mn": "Mn(s)",
    "Fe": "Fe(s)",
    "Co": "Co(s)",
    "Ni": "Ni(s)",
    "Cu": "Cu(s)",
    "Zn": "Zn(s)",
    "Ga": "Ga(s)",
    "Ge": "Ge(s)",
    "As": "As(s)",
    "Se": "Se(s)",
    "Br": f"{one_half}Br{subscript(2)}(l)",
    "Kr": "(g)",
}


class Energy(seamm.Node):
    """
    The non-graphical part of a Energy step in a flowchart.

    Attributes
    ----------
    parser : configargparse.ArgParser
        The parser object.

    options : tuple
        It contains a two item tuple containing the populated namespace and the
        list of remaining argument strings.

    subflowchart : seamm.Flowchart
        A SEAMM Flowchart object that represents a subflowchart, if needed.

    parameters : EnergyParameters
        The control parameters for Energy.

    See Also
    --------
    TkEnergy,
    Energy, EnergyParameters
    """

    def __init__(self, flowchart=None, title="Energy", extension=None, logger=logger):
        """A substep for Energy in a subflowchart for VASP.

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
        logger.debug(f"Creating Energy {self}")

        super().__init__(
            flowchart=flowchart,
            title=title,
            extension=extension,
            module=__name__,
            logger=logger,
        )  # yapf: disable

        self._calculation = "Energy"
        self._model = None
        self._metadata = vasp_step.metadata
        self.parameters = vasp_step.EnergyParameters()
        self._element_count = {}  # Number of atoms of each element (atomic number)
        self._to_VASP_order = []  # translation from SEAMM order to VASP
        self._to_SEAMM_order = []  # translation from VASP order to SEAMM

        self._gamma_point_only = False

        self._timing_data = []
        self._timing_path = Path("~/.seamm.d/timing/vasp.csv").expanduser()

        # Set up the timing information
        self._timing_header = [
            "node",  # 0
            "cpu",  # 1
            "cpu_version",  # 2
            "cpu_count",  # 3
            "cpu_speed",  # 4
            "date",  # 5
            "POSCAR",  # 6
            "INCAR",  # 7
            "KPOINTS",  # 8
            "potentials",  # 9
            "formula",  # 10
            "model",  # 11
            "nproc",  # 12
            "time",  # 13
        ]
        try:
            self._timing_path.parent.mkdir(parents=True, exist_ok=True)

            self._timing_data = 14 * [""]
            self._timing_data[0] = platform.node()
            tmp = get_cpu_info()
            if "arch" in tmp:
                self._timing_data[1] = tmp["arch"]
            if "cpuinfo_version_string" in tmp:
                self._timing_data[2] = tmp["cpuinfo_version_string"]
            if "count" in tmp:
                self._timing_data[3] = str(tmp["count"])
            if "hz_advertized_friendly" in tmp:
                self._timing_data[4] = tmp["hz_advertized_friendly"]

            if not self._timing_path.exists():
                with self._timing_path.open("w", newline="") as fd:
                    writer = csv.writer(fd)
                    writer.writerow(self._timing_header)
        except Exception:
            self._timing_data = None

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

    @property
    def to_VASP_order(self):
        """Translation of atoms from SEAMM to VASP order."""
        if len(self._to_VASP_order) == 0:
            self.atom_order()
        return self._to_VASP_order

    @property
    def to_SEAMM_order(self):
        """Translation of atoms from VASP to SEAMM order."""
        if len(self._to_SEAMM_order) == 0:
            self.atom_order()
        return self._to_SEAMM_order

    @property
    def element_count(self):
        """Numbers of atoms of each element."""
        if len(self._element_count) == 0:
            self.atom_order()
        return self._element_count

    def atom_order(self):
        """Get the coordinate information for VASP (POSCAR file)."""
        system, configuration = self.get_system_configuration()

        # Prepare to reorder the atoms into descending atomic number
        atnos = configuration.atoms.atomic_numbers
        unique_atnos = sorted(list(set(atnos)), reverse=True)

        # The dictionary to translate to/from the VASP order
        self._to_VASP_order = []
        self._to_SEAMM_order = []
        self._element_count = {atno: 0 for atno in atnos}
        for atno in atnos:
            self._element_count[atno] += 1
        n = 0
        offset = {}
        for atno in unique_atnos:
            offset[atno] = n
            n += self._element_count[atno]
        self._to_SEAMM_order = [-1] * len(atnos)
        for original, atno in enumerate(atnos):
            new = offset[atno]
            self._to_VASP_order.append(new)
            self._to_SEAMM_order[new] = original
            offset[atno] += 1

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

        if P["spin polarization"] == "collinear":
            text = "A non-spin-polarized"
        elif P["spin polarization"] == "noncollinear":
            text = "A spin-polarized"
        else:
            text = "A non-collinear magnetic"
        text += " calculation using {model} / {submodel}."

        lasph = P["nonspherical PAW"]
        if isinstance(lasph, str):
            if self.is_expr(lasph):
                text += " Whether to include the contribution of the nonspherical terms"
                text += " within the PAW spheres will be determined by"
                text += " {nonspherical PAW}."
            elif lasph == "yes":
                text += " The contribution of the nonspherical terms within the PAW"
                text += " spheres will be included."
        elif isinstance(lasph, bool):
            text += " The contribution of the nonspherical terms within the PAW"
            text += " spheres will be included."

        text += " The plane-wave basis will be cutoff at {plane-wave cutoff}."

        _type = P["occupation type"]
        text += " The orbital occupancies will determined using "
        if self.is_expr(_type):
            text += "the method given by {occupation type}. If the Methfessel-Paxton"
            text += " method is chosen, it will be of order {Methfessel-Paxton order},"
            text += " and if the method uses smearing, the width will be"
            text += " {smearing width}."
        elif "Methfessel" in _type:
            text += "the order={Methfessel-Paxton order} Methfessel-Paxton method"
            text += " with a smearing width of {smearing width}."
        else:
            text += "{occupation type}"
            if "without smearing" not in P["occupation type"]:
                text += " with a smearing width of {smearing width}."
            else:
                text += "."

        text += "\n\n"

        method = P["k-grid method"]
        odd = P["odd grid"]
        if isinstance(odd, str) and "yes" in odd:
            odd = True
        centering = P["centering"]

        text += "The numerical k-mesh for integration in reciprocal space"
        text += " will be"
        if "point" in method:
            text += " just the 𝚪-point."
        else:
            if self.is_expr(centering):
                pass
            elif "Monkhorst" in centering:
                text += " a Monkhorst-Pack grid"
            else:
                text += " a {centering} grid"
            if self.is_expr(method):
                text += " determined at run time by {k-grid method}."
                text += " If the grid is given explicitly it will be {na} x {nb}"
                text += " x {nc}. Otherwise it will be determined using a spacing"
                text += " of {k-spacing}"
                if isinstance(odd, bool) and odd:
                    text += " with the dimensions forced to odd numbers."
            elif "spacing" in method:
                text += " determined using a spacing of {k-spacing}"
                if self.is_expr(odd):
                    text += ". {odd grid} will determine if the grid dimensions are"
                    text += " forced to be odd numbers."
                elif isinstance(odd, bool) and odd:
                    text += " with the dimensions forced to odd numbers."
            else:
                text += " given explicitly as {na} x {nb} x {nc}."

        if self._calculation == "Energy":
            return self.header + "\n" + __(text, **P, indent=4 * " ").__str__()
        else:
            return __(text, **P, indent=4 * " ").__str__()

    def run(self):
        """Run a Energy step.

        Parameters
        ----------
        None

        Returns
        -------
        seamm.Node
            The next node object in the flowchart.
        """
        next_node = super().run(printer)

        # Get the values of the parameters, dereferencing any variables
        P = self.parameters.current_values_to_dict(
            context=seamm.flowchart_variables._data
        )
        input_only = P["input only"]

        # Print what we are doing
        printer.important(__(self.description_text(P), indent=self.indent))

        # Create the directory
        directory = self.wd
        directory.mkdir(parents=True, exist_ok=True)

        # Get the system & configuration
        system, starting_configuration = self.get_system_configuration(None)

        # And the model
        self.model = P["submodel"]

        # Check for successful run, don't rerun
        success_file = directory / "success.dat"
        if not success_file.exists():
            # Access the options
            options = self.parent.options
            seamm_options = self.parent.global_options

            # Get the computational environment and set limits
            ce = seamm_exec.computational_environment()

            # How many threads to use
            n_cores = ce["NTASKS"]
            self.logger.debug("The number of cores available is {}".format(n_cores))

            if options["ncores"] == "available":
                n_threads = n_cores
            else:
                n_threads = int(options["ncores"])
            if n_threads > n_cores:
                n_threads = n_cores
            if n_threads < 1:
                n_threads = 1
            if seamm_options["ncores"] != "available":
                n_threads = min(n_threads, int(seamm_options["ncores"]))

            np = P["np"]
            if np != "available" and np < n_threads:
                printer.important(
                    self.indent + f"    There are {n_threads} cores available; however,"
                    f" VASP will use {np} MPI processes as requested."
                )
                n_threads = np
            else:
                printer.important(
                    self.indent + f"    VASP will use {n_threads} MPI processes."
                )
            printer.important("")
            ce["NTASKS"] = n_threads
            self.logger.debug(f"VASP will use {n_threads} threads.")

            files = self.get_input(P)

            input_only = P["input only"]
            if input_only:
                # Just write the input files and stop
                for filename in files:
                    path = directory / filename
                    path.write_text(files[filename])
            else:
                executor = self.parent.flowchart.executor

                # Read configuration file for VASP if it exists
                executor_type = executor.name
                full_config = configparser.ConfigParser()
                ini_dir = Path(seamm_options["root"]).expanduser()
                path = ini_dir / "vasp.ini"

                # If the config file doesn't exist, get the default
                if not path.exists():
                    resources = importlib.resources.files("vasp_step") / "data"
                    ini_text = (resources / "vasp.ini").read_text()
                    txt_config = Configuration(path)
                    txt_config.from_string(ini_text)
                    txt_config.save()

                full_config.read(ini_dir / "vasp.ini")

                # Getting desperate! Look for an executable in the path
                if executor_type not in full_config:
                    exe_path = shutil.which("vasp_std")
                    if exe_path is None:
                        raise RuntimeError(
                            f"No section for '{executor_type}' in VASP ini file"
                            f" ({ini_dir / 'vasp.ini'}), nor in the defaults, "
                            "nor in the path!"
                        )

                    txt_config = Configuration(path)

                    if not txt_config.section_exists(executor_type):
                        txt_config.add_section(executor_type)

                    txt_config.set_value(executor_type, "installation", "local")
                    txt_config.set_value(
                        executor_type, "code", "mpiexec -np {NTASKS} vasp_std"
                    )
                    txt_config.set_value(
                        executor_type, "gamma_code", "mpiexec -np {NTASKS} vasp_gam"
                    )
                    txt_config.set_value(
                        executor_type,
                        "noncollinear_code",
                        "mpiexec -np {NTASKS} vasp_ncl",
                    )
                    txt_config.save()
                    full_config.read(ini_dir / "vasp.ini")

                config = dict(full_config.items(executor_type))
                # Use the matching version of the seamm-vasp image by default.
                config["version"] = self.version

                # Setup the calculation environment definition,
                # seeing which excutable to use
                if P["spin polarization"] == "noncollinear":
                    cmd = config["noncollinear_code"]
                elif self._gamma_point_only:
                    cmd = config["gamma_code"]
                else:
                    cmd = config["code"]
                cmd += " > output.txt"

                return_files = [
                    "*.h5",
                    "*CAR",
                    "output.txt",
                    "vasprun.xml",
                ]

                self.logger.debug(f"{cmd=}")

                if self._timing_data is not None:
                    self._timing_data[5] = datetime.now(timezone.utc).isoformat()
                    self._timing_data[6] = files["POSCAR"]
                    self._timing_data[7] = files["INCAR"]
                    self._timing_data[8] = files["KPOINTS"]
                t0 = time.time_ns()
                result = executor.run(
                    ce=ce,
                    cmd=[cmd],
                    config=config,
                    directory=self.directory,
                    files=files,
                    return_files=return_files,
                    in_situ=True,
                    shell=True,
                )

                t = (time.time_ns() - t0) / 1.0e9
                if self._timing_data is not None:
                    self._timing_data[12] = str(n_threads)
                    self._timing_data[13] = f"{t:.3f}"

                if not result:
                    self.logger.error("There was an error running VASP")
                    return None

                if self._timing_data is not None:
                    try:
                        with self._timing_path.open("a", newline="") as fd:
                            writer = csv.writer(fd)
                            writer.writerow(self._timing_data)
                    except Exception:
                        # Don't want an error with timing to be fatal
                        pass

        if not input_only:
            # Checkout that the main output exists
            data_file = directory / "vaspout.h5"
            if not data_file.exists():
                raise RuntimeError("VASP appears to have failed Cannot find vaspout.h5")

            # Follow instructions for where to put the coordinates,
            system, configuration = self.get_system_configuration(
                P=P, same_as=starting_configuration, model=self.model
            )

            # And analyze the results
            self.analyze(P=P, configuration=configuration)

            # Did it! Write the success file, so don't rerun VASP again
            success_file.write_text("success")

        # Add other citations here or in the appropriate place in the code.
        # Add the bibtex to data/references.bib, and add a self.reference.cite
        # similar to the above to actually add the citation to the references.

        return next_node

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
        if self.calculation == "Energy":
            # Extract the data we need from the output files.
            hdf5_file = self.wd / "vaspout.h5"
            xml_file = self.wd / "vasprun.xml"
            if hdf5_file.exists():
                results = self.parse_hdf5(hdf5_file)
            elif xml_file.exists():
                results = self.parse_xml(xml_file)
            else:
                results = {}
                text = f"Something is very wrong! Cannot find either {hdf5_file} or "
                text += f"{xml_file}. Did VASP fail?"
                printer.normal(textwrap.indent(text, self.indent + 4 * " "))
                printer.normal("")
                return results

        # Get the last of each property by itself
        new = {}
        for key, item in results.items():
            if key.endswith(",iter") and len(item) > 0:
                newkey = key[0:-5]
                new[newkey] = item[-1]
        results.update(new)

        # Get the norm and max of forces on the atoms for each iteration
        results["RMS atom force,iter"] = []
        results["maximum atom force,iter"] = []
        for dE in results["gradients,iter"]:
            dE = np.array(dE)
            norm = LA.norm(dE, axis=1)
            rms = np.sqrt(np.sum(norm**2) / len(norm))
            maximum = max(norm)

            results["RMS atom force,iter"].append(rms)
            results["maximum atom force,iter"].append(maximum)
        results["RMS atom force"] = results["RMS atom force,iter"][-1]
        results["maximum atom force"] = results["maximum atom force,iter"][-1]

        # Calculate the enthalpy of formation, if possible
        tmp_text = self.calculate_enthalpy_of_formation(P, results)
        if tmp_text != "":
            path = self.wd / "Thermochemistry.txt"
            path.write_text(tmp_text)

        if table is None:
            table = {
                "Property": [],
                "Value": [],
                "Units": [],
            }

        metadata = vasp_step.metadata["results"]
        for key, title in (
            ("DfE0", f"{lDelta}fE{degree_sign}"),
            ("energy", "E"),
            ("Gelec", "Gelec"),
            ("Ecoh", "Ecoh"),
            ("Ecoh/atom", "Ecoh/atom"),
            ("RMS atom force", "RMS force"),
            ("maximum atom force", "Maximum force"),
            ("P", "P"),
            ("V", "V"),
            ("a", "a"),
            ("b", "b"),
            ("c", "c"),
            ("alpha", "\N{GREEK SMALL LETTER ALPHA}"),
            ("beta", "\N{GREEK SMALL LETTER BETA}"),
            ("gamma", "\N{GREEK SMALL LETTER GAMMA}"),
        ):
            if key in results:
                tmp = metadata[key]
                if "format" in tmp:
                    fmt = tmp["format"]
                else:
                    fmt = "s"
                units = tmp["units"]
                table["Property"].append(title)
                table["Value"].append(f"{results[key]:{fmt}}")
                table["Units"].append(units.replace("^3", "\N{SUPERSCRIPT THREE}"))

        tmp = tabulate(
            table,
            headers="keys",
            tablefmt="rounded_outline",
            colalign=("center", "decimal", "center"),
            disable_numparse=True,
        )
        length = len(tmp.splitlines()[0])
        text_lines = []
        header = "Results"
        text_lines.append(header.center(length))
        text_lines.append(self.model.center(length))
        text_lines.append(tmp)

        if text != "":
            text = str(__(text, indent=self.indent + 4 * " "))
            text += "\n\n"
        text += textwrap.indent("\n".join(text_lines), self.indent + 7 * " ")
        printer.normal(text)
        printer.normal("")

        # Store the results as requested
        self.store_results(
            configuration=configuration,
            data=results,
        )

        if P["save gradients"]:
            # Store the gradients in the database, reordering back to SEAMM order
            factor = Q_("eV/Å").m_as("kJ/mol/Å")
            tmp = factor * np.array(results["gradients,iter"][-1])
            tmp = tmp.tolist()
            configuration.atoms.gradients = [tmp[i] for i in self.to_VASP_order]

    def calculate_enthalpy_of_formation(self, P, data):
        """Calculate the enthalpy of formation from the results of a calculation.

        This uses tabulated values of the enthalpy of formation of the atoms for
        the elements and tabulated energies calculated for atoms with the current
        method.

        Parameters
        ----------
        data : dict
            The results of the calculation.
        """
        # Get the atomic numbers and counts
        _, configuration = self.get_system_configuration(None)
        counts = Counter(configuration.atoms.atomic_numbers)
        symbols = sorted(molsystem.elements.to_symbols(counts.keys()))

        # Which set of potentials are we using?
        # potential_set = P["set of potentials"]
        # potentials = P["potentials"]
        # name = potentials[element]  # The potential used for element

        # Cutoff
        encut = P["plane-wave cutoff"].m_as("eV")
        if encut.is_integer():
            encut = int(encut)

        # Read the tabulated values from either user or data directory
        personal_file = Path("~/.seamm.d/data/element_energies.csv").expanduser()
        if personal_file.exists():
            personal_table = pandas.read_csv(personal_file, index_col=False)
        else:
            personal_table = None

        path = Path(pkg_resources.resource_filename(__name__, "data/"))
        csv_file = path / "element_energies.csv"
        table = pandas.read_csv(csv_file, index_col=False)

        self.logger.debug(f"self.model = {self.model}")

        # Check if have the data
        atom_formation_energy = None
        atom_energy = None
        column = self.model + "@" + str(encut)

        self.logger.debug(f"Looking for '{column}'")

        # atom_formation_energy is the energy of formation of the standard state,
        # per atom.
        # atom_energy is the calculated energy of the atom, which defaults to zero
        column2 = column + " atom energy"
        if personal_table is not None and column in personal_table.columns:
            atom_formation_energy = personal_table[column].to_list()
            if column2 in personal_table.columns:
                atom_energy = personal_table[column2].to_list()
        elif column in table.columns:
            atom_formation_energy = table[column].to_list()
            if column2 in table.columns:
                atom_energy = table[column2].to_list()

        if atom_formation_energy is None:
            # Not found!
            return f"There are no tabulated atom energies for {column}"

        # Assume an offset energy -- the energy of an isolated atom -- is zero if not
        # tabulated
        if atom_energy is None:
            atom_energy = [0.0] * len(atom_formation_energy)

        DfH0gas = None
        references = None
        term_symbols = None
        if personal_table is not None and "ΔfH°gas" in personal_table.columns:
            DfH0gas = personal_table["ΔfH°gas"].to_list()
            if "Reference" in personal_table.columns:
                references = personal_table["Reference"].to_list()
            if "Term Symbol" in personal_table.columns:
                term_symbols = personal_table["Term Symbol"].to_list()
        elif "ΔfH°gas" in table.columns:
            DfH0gas = table["ΔfH°gas"].to_list()
            if "Reference" in table.columns:
                references = table["Reference"].to_list()
            if "Term Symbol" in table.columns:
                term_symbols = table["Term Symbol"].to_list()

        # Get the Hill formula as a list
        composition = []
        if "C" in symbols:
            composition.append((6, "C", counts[6]))
            symbols.remove("C")
            if "H" in symbols:
                composition.append((1, "H", counts[1]))
                symbols.remove("H")

        for symbol in symbols:
            atno = molsystem.elements.symbol_to_atno[symbol]
            composition.append((atno, symbol, counts[atno]))

        # And the reactions. First, for atomization energy
        formula = ""
        tmp = []
        for atno, symbol, count in composition:
            if count == 1:
                formula += symbol
                tmp.append(f"{symbol}(g)")
            else:
                formula += f"{symbol}{subscript(count)}"
                tmp.append(f"{count}{middot}{symbol}(g)")
        gas_atoms = " + ".join(tmp)
        tmp = []
        for atno, symbol, count in composition:
            if count == 1:
                tmp.append(standard_state[symbol])
            else:
                tmp.append(f"{count}{middot}{standard_state[symbol]}")
        standard_elements = " + ".join(tmp)

        # The energy - any offsets is the negative of the atomization energy
        name = "Formula: " + formula
        try:
            name = configuration.PC_iupac_name(fallback=name)
        except Exception:
            # If there is an error, just use the name so far.
            name = name
            pass

        if name is None:
            name = "Formula: " + formula

        text = f"Thermochemistry of {name} with {column}\n\n"
        text += "Cohesive Energy\n"
        text += "------------------\n"
        text += textwrap.fill(
            f"The cohesive energy,  {lDelta}atE{degree_sign}, is the energy to break"
            " all the bonds in the system, separating the atoms from each other."
        )
        text += f"\n\n    {formula} --> {gas_atoms}\n\n"
        text += textwrap.fill(
            "The following table shows in detail the calculation. The first line is "
            "the system and its calculated energy. The next lines are the energies "
            "of each type of atom in the system. These have been tabulated by running "
            "calculations on each atom, and are included in the SEAMM release. "
            "The line give the formation energy from atoms in kJ/mol.",
        )
        text += "\n\n"
        table = {
            "System": [],
            "Term": [],
            "Value": [],
            "Units": [],
        }

        if "Epe" in data:
            E = data["Epe"]
        elif "energy" in data:
            E = Q_(data["energy"], "eV").m_as("kJ/mol")
        else:
            return "The energy is not in results from the calculation!"

        to_eV = Q_("kJ/mol").m_as("eV")

        Eatoms = 0.0
        Ef0 = 0.0
        for atno, symbol, count in composition:
            Eatom = atom_energy[atno - 1]
            if isnan(Eatom):
                # Don't have the data for this element
                return f"Do not have tabulated atom energies for {symbol} in {column}"
            Eatoms += count * Eatom
            table["System"].append(f"{symbol}(g)")
            table["Term"].append(f"{count} * {to_eV * Eatom:.2f}")
            table["Value"].append(f"{count * to_eV * Eatom:.2f}")
            table["Units"].append("")

            Ef0 += count * atom_formation_energy[atno - 1]

        data["DfE0"] = E - Ef0

        table["Units"][0] = "eV"

        table["System"].append("^")
        table["Term"].append("-")
        table["Value"].append("-")
        table["Units"].append("")

        table["System"].append(formula)
        table["Term"].append(f"{to_eV * E:.2f}")
        table["Value"].append(f"{to_eV * E:.2f}")
        table["Units"].append("eV")

        data["Ecoh"] = to_eV * (Eatoms - E)
        data["Ecoh/atom"] = to_eV * (Eatoms - E) / configuration.n_atoms

        table["System"].append("")
        table["Term"].append("")
        table["Value"].append("=")
        table["Units"].append("")

        table["System"].append("")
        table["Term"].append("")
        table["Value"].append(f'{data["Ecoh"]:.2f}')
        table["Units"].append("eV")

        tmp = tabulate(
            table,
            headers="keys",
            tablefmt="rounded_outline",
            colalign=("center", "center", "decimal", "center"),
            disable_numparse=True,
        )
        length = len(tmp.splitlines()[0])
        text_lines = []
        text_lines.append(f"Cohesive Energy for {formula}".center(length))
        text_lines.append(tmp)
        text += textwrap.indent("\n".join(text_lines), 4 * " ")

        if "H" not in data:
            text += "\n\n"
            text += "Cannot calculate enthalpy of formation without the enthalpy"
            return text
        if DfH0gas is None:
            text += "\n\n"
            text += "Cannot calculate enthalpy of formation without the tabulated\n"
            text += "atomization enthalpies of the elements."
            return text

        # Atomization enthalpy of the elements, experimental
        table = {
            "System": [],
            "Term": [],
            "Value": [],
            "Units": [],
            "Reference": [],
        }

        E = data["energy"]

        DfH_at = 0.0
        refno = 1
        for atno, symbol, count in composition:
            DfH_atom = DfH0gas[atno - 1]
            DfH_at += count * DfH_atom
            tmp = Q_(DfH_atom, "kJ/mol").m_as("E_h")
            table["System"].append(f"{symbol}(g)")
            if count == 1:
                table["Term"].append(f"{tmp:.6f}")
            else:
                table["Term"].append(f"{count} * {tmp:.6f}")
            table["Value"].append(f"{count * tmp:.6f}")
            table["Units"].append("")
            refno += 1
            table["Reference"].append(refno)

        table["Units"][0] = "E_h"

        table["System"].append("^")
        table["Term"].append("-")
        table["Value"].append("-")
        table["Units"].append("")
        table["Reference"].append("")

        table["System"].append(standard_elements)
        table["Term"].append("")
        table["Value"].append("0.0")
        table["Units"].append("E_h")
        table["Reference"].append("")

        table["System"].append("")
        table["Term"].append("")
        table["Value"].append("=")
        table["Units"].append("")
        table["Reference"].append("")

        result = f'{Q_(DfH_at, "kJ/mol").m_as("E_h"):.6f}'
        table["System"].append(f"{lDelta}atH{degree_sign}")
        table["Term"].append("")
        table["Value"].append(result)
        table["Units"].append("E_h")
        table["Reference"].append("")

        table["System"].append("")
        table["Term"].append("")
        table["Value"].append(f"{DfH_at:.2f}")
        table["Units"].append("kJ/mol")
        table["Reference"].append("")

        tmp = tabulate(
            table,
            headers="keys",
            tablefmt="rounded_outline",
            colalign=("center", "center", "decimal", "center", "center"),
            disable_numparse=True,
        )
        length = len(tmp.splitlines()[0])
        text_lines = []
        text_lines.append(
            "Atomization enthalpy of the elements (experimental)".center(length)
        )
        text_lines.append(tmp)

        text += "\n\n"
        text += "Enthalpy of Formation\n"
        text += "---------------------\n"
        text += textwrap.fill(
            f"The enthalpy of formation, {lDelta}fHº, is the enthalpy of creating the "
            "molecule from the elements in their standard state:"
        )
        text += f"\n\n   {standard_elements} --> {formula} (1)\n\n"
        text += textwrap.fill(
            "The standard state of the element, denoted by the superscript º,"
            " is its form at 298.15 K and 1 atm pressure, e.g. graphite for carbon, "
            "H2 gas for hydrogen, etc."
        )
        text += "\n\n"
        text += textwrap.fill(
            "Since it is not easy to calculate the enthalpy of e.g. graphite we will "
            "use two sequential reactions that are equivalent. First, we will create "
            "gas phase atoms from the elements:"
        )
        text += f"\n\n    {standard_elements} --> {gas_atoms} (2)\n\n"
        text += textwrap.fill(
            "This will use the experimental values of the enthalpy of formation of the "
            "atoms in the gas phase to calculate the enthalpy of this reaction. "
            "Then we react the atoms to get the desired system:"
        )
        text += f"\n\n    {gas_atoms} --> {formula} (3)\n\n"
        text += textwrap.fill(
            "Note that this is reverse of the atomization reaction, so "
            f"{lDelta}H = -{lDelta}atH."
        )
        text += "\n\n"
        text += textwrap.fill(
            "First we calculate the enthalpy of the atomization of the elements in "
            "their standard state, using tabulated experimental values:"
        )
        text += "\n\n"
        text += textwrap.indent("\n".join(text_lines), 4 * " ")

        # And the calculated atomization enthalpy
        table = {
            "System": [],
            "Term": [],
            "Value": [],
            "Units": [],
        }

        Hatoms = 0.0
        dH = Q_(6.197, "kJ/mol").m_as("E_h")
        for atno, symbol, count in composition:
            Eatom = atom_formation_energy[atno - 1]
            # 6.197 is the H298-H0 for an atom
            Hatoms += count * (Eatom + 6.197)

            table["System"].append(f"{symbol}(g)")
            if count == 1:
                table["Term"].append(f"{-Eatom:.2f} + {dH:.2f}")
            else:
                table["Term"].append(f"{count} * ({-Eatom:.2f} + {dH:.2f})")
            table["Value"].append(f"{-count * (Eatom + dH):.2f}")
            table["Units"].append("")

        table["System"].append("^")
        table["Term"].append("-")
        table["Value"].append("-")
        table["Units"].append("")

        H = data["H"]

        table["System"].append(formula)
        table["Term"].append(f"{H:.2f}")
        table["Value"].append("")
        table["Units"].append("kJ/mol")

        data["H atomization"] = Hatoms - Q_(H, "E_h").m_as("kJ/mol")
        data["DfH0"] = DfH_at - data["H atomization"]
        table["System"].append("")
        table["Term"].append("")
        table["Value"].append("=")
        table["Units"].append("")

        table["System"].append("")
        table["Term"].append("")
        table["Value"].append(f'{data["H atomization"]:.2f}')
        table["Units"].append("kJ/mol")

        tmp = tabulate(
            table,
            headers="keys",
            tablefmt="rounded_outline",
            colalign=("center", "center", "decimal", "center"),
            disable_numparse=True,
        )
        length = len(tmp.splitlines()[0])
        text_lines = []
        text_lines.append("Atomization Enthalpy (calculated)".center(length))
        text_lines.append(tmp)
        text += "\n\n"

        text += textwrap.fill(
            "Next we calculate the atomization enthalpy of the system. We have the "
            "calculated enthalpy of the system, but need the enthalpy of gas phase "
            f"atoms at the standard state (25{degree_sign}C, 1 atm). The tabulated "
            "energies for the atoms, used above, are identical to H0 for an atom. "
            "We will add H298 - H0 to each atom, which [1] is 5/2RT = 0.002360 E_h"
        )
        text += "\n\n"
        text += textwrap.indent("\n".join(text_lines), 4 * " ")
        text += "\n\n"
        text += textwrap.fill(
            "The enthalpy change for reaction (3) is the negative of this atomization"
            " enthalpy. Putting the two reactions together with the negative for Rxn 3:"
        )
        text += "\n\n"
        text += f"{lDelta}fH{degree_sign} = {lDelta}H(rxn 2) - {lDelta}H(rxn 3)\n"
        text += f"     = {DfH_at:.2f} - {data['H atomization']:.2f}\n"
        text += f"     = {DfH_at - data['H atomization']:.2f} kJ/mol\n"

        text += "\n\n"
        text += "References\n"
        text += "----------\n"
        text += "1. https://en.wikipedia.org/wiki/Monatomic_gas\n"
        refno = 1
        for atno, symbol, count in composition:
            refno += 1
            text += f"{refno}. {lDelta}fH{degree_sign} = {DfH0gas[atno - 1]} kJ/mol"
            if term_symbols is not None:
                text += f" for {term_symbols[atno - 1]} {symbol}"
            else:
                text += f" for {symbol}"
            if references is not None:
                text += f" from {references[atno-1]}\n"

        return text

    def get_input(self, P=None):
        """Get all the input for VASP"""

        # Get the values of the parameters, dereferencing any variables
        if P is None:
            P = self.parameters.current_values_to_dict(
                context=seamm.flowchart_variables._data
            )

        # Need to reset the element count for subsequent runs
        self._element_count = {}
        self._to_VASP_order = []
        self._to_SEAMM_order = []

        files = {}
        files["INCAR"] = self.get_INCAR(P)
        files["POTCAR"] = self.get_POTCAR(P)
        files["KPOINTS"] = self.get_KPOINTS(P)
        files["POSCAR"] = self.get_POSCAR(P)

        return files

    def get_INCAR(self, P=None):
        """Get the control input (INCAR) for this calculation."""
        keywords, descriptions = self.get_keywords(P)

        lines = []
        keydata = self.metadata["keywords"]
        for key, value in keywords.items():
            if key in descriptions:
                lines.append(f"{key:>20s} = {value:<20}  # {descriptions[key]}")
            elif key in keydata and "description" in keydata[key]:
                lines.append(
                    f"{key:>20s} = {value:<20}  # {keydata[key]['description']}"
                )
            else:
                lines.append(f"{key:>20s} = {value}")

        return "\n".join(lines)

    def get_keywords(self, P=None):
        """Get the keywords and values for the calculation."""
        # Get the values of the parameters, dereferencing any variables
        if P is None:
            P = self.parameters.current_values_to_dict(
                context=seamm.flowchart_variables._data
            )

        descriptions = {}
        keywords = {}

        # The DFT functional
        model = P["model"]
        submodel = P["submodel"]

        model_data = self.metadata["computational models"][
            "Density Functional Theory (DFT)"
        ]["models"][model]["parameterizations"]

        submodel_data = model_data[submodel]
        if self._timing_data is not None:
            self._timing_data[11] = f"{model} / {submodel}"

        tmp = submodel_data["keywords"]
        keywords.update(tmp)
        descriptions[list(tmp)[0]] = submodel_data["description"]

        # Spin polarization
        if P["spin polarization"] == "collinear":
            keywords["ISPIN"] = 2
        elif P["spin polarization"] == "noncollinear":
            keywords["LNONCOLLINEAR"] = ".True."
        else:
            keywords["ISPIN"] = 1

        # Non-spherical contributions in PAWs
        keywords["LASPH"] = ".True." if P["nonspherical PAW"] else ".False."

        # The energy cutoff, which may be an expression of ENMAX
        encut = P["plane-wave cutoff"]
        if isinstance(encut, str):
            global_dict = {**seamm.flowchart_variables._data}
            global_dict["ENMAX"] = P["enmax"]
            global_dict["enmax"] = P["enmax"]
            encut = eval(encut, global_dict)
        else:
            encut = encut.m_as("eV")
        keywords["ENCUT"] = f"{encut:.2f}"

        # Electronic optimization algorithm
        keywords["ALGO"] = P["electronic method"].title().replace(" ", "")
        keywords["NELM"] = P["nelm"]
        keywords["NELMIN"] = 2 if P["nelmin"] == "default" else P["nelmin"]
        keywords["EDIFF"] = f'{P["ediff"]:.2E}'
        keywords["PREC"] = P["precision"]

        # Smearing
        _type = P["occupation type"].lower()
        if "gaussian" in _type:
            ismear = 0
        elif "methfessel" in _type:
            ismear = P["Methfessel-Paxton order"]
        elif "tetrahedron" in _type:
            if "corrections" in _type:
                if "fermi" in _type:
                    ismear = -15
                else:
                    ismear = -5
            else:
                if "fermi" in _type:
                    ismear = -14
                else:
                    ismear = -4
        elif "fermi" in _type:
            ismear = -1
        else:
            raise ValueError(f"Occupation type (ISMEAR) '{_type} not recognized.")
        keywords["ISMEAR"] = ismear
        descriptions["ISMEAR"] = _type

        if ismear >= -1 or ismear in (-15, -14):
            sigma = P["smearing width"].m_as("eV")
            keywords["SIGMA"] = f"{sigma:.2f}"

        # The definition of the type of calculation: SPE
        keywords["IBRION"] = -1
        match P["calculate stress"]:
            case "no":
                isif = 0
            case "only pressure":
                isif = 1
            case _:
                isif = 2
        keywords["ISIF"] = isif
        keywords["NSW"] = 0
        efermi = P["efermi"]
        if "middle" in efermi:
            keywords["EFERMI"] = "MIDGAP"
        elif efermi == "legacy":
            keywords["EFERMI"] = "Legacy"
        else:
            keywords["EFERMI"] = efermi.m_as("eV")

        # Use the HDF5 output files
        keywords["LH5"] = ".True." if P["use hdf5 files"] else ".False."

        # Calculate on-site density and spin
        if P["lorbit"]:
            keywords["LORBIT"] = 11

        # Parameters controlling the performance
        keywords["NCORE"] = P["ncore"]
        keywords["KPAR"] = P["kpar"]
        keywords["LPLANE"] = ".True." if P["lplane"] else ".False."
        keywords["LREAL"] = "Auto" if P["lreal"] else ".False."
        keywords["NSIM"] = P["nsim"]
        keywords["LSCALAPACK"] = ".True." if P["lscalapack"] else ".False."
        if P["lscalapack"]:
            keywords["LSCALU"] = ".True." if P["lscalu"] else ".False."

        # Replace and add any extra keywords the user has specified
        # The values look like 'key=value'
        keyword_data = self.metadata["keywords"]
        for tmp in P["extra keywords"]:
            key, value = tmp.split("=", 1)
            keywords[key] = value
            if key in keyword_data:
                descriptions[key] = keyword_data[key]["description"]

        return keywords, descriptions

    def get_POTCAR(self, P=None):
        """Get the potential input (POTCAR) for this calculation.

        The elements are ordered by descending atomic number.
        """
        _, configuration = self.get_system_configuration()
        atnos = sorted(list(set(configuration.atoms.atomic_numbers)), reverse=True)
        elements = molsystem.elements.to_symbols(atnos)

        # Which set of potentials are we using?
        potential_set = P["set of potentials"]
        potential_data = self.parent.potential_metadata[potential_set]
        potentials = P["potentials"]

        text = ""
        names = []
        for element in elements:
            name = potentials[element]
            names.append(name)
            path = Path(potential_data[name]["file"])
            text += path.read_text()

        if self._timing_data is not None:
            self._timing_data[9] = " ".join(names)

        return text

    def get_KPOINTS(self, P=None):
        """Get the k-point grid, KPOINTS file."""
        _, configuration = self.get_system_configuration()

        lines = []
        if "point" in P["k-grid method"]:
            lines.append("𝚪-point only")
            na = nb = nc = 1
        elif "explicit" in P["k-grid method"]:
            lines.append("Explicit k-point mesh")
            na = P["na"]
            nb = P["nb"]
            nc = P["nc"]
        else:
            lengths = configuration.cell.reciprocal_lengths()
            spacing = P["k-spacing"].to("1/Å").magnitude
            lines.append(f"k-point mesh with spacing {spacing}")
            na = max(1, ceiling(lengths[0] / spacing))
            nb = max(1, ceiling(lengths[1] / spacing))
            nc = max(1, ceiling(lengths[2] / spacing))
            if P["odd grid"]:
                na = na + 1 if na % 2 == 0 else na
                nb = nb + 1 if nb % 2 == 0 else nb
                nc = nc + 1 if nc % 2 == 0 else nc

        self._gamma_point_only = na == 1 and nb == 1 and nc == 1

        lines.append("0")
        centering = P["centering"]
        if "Monkhorst" in centering and "point" not in P["k-grid method"]:
            lines.append("Monkhorst-Pack")
        else:
            lines.append("Gamma")
        lines.append(f"{na} {nb} {nc}")
        lines.append("0 0 0")

        return "\n".join(lines)

    def get_POSCAR(self, P=None):
        """Get the coordinate information for VASP (POSCAR file)."""
        system, configuration = self.get_system_configuration()

        # And finally make the POSCAR file contents
        lines = []
        sysname = system.name
        confname = configuration.name
        if sysname == "" and confname == "":
            formula, empirical, Z = configuration.formula
            if Z == 1:
                title = formula
            else:
                title = f"({empirical}) * {Z}"
        else:
            title = sysname + "/" + confname
            if len(title) > 100:
                if len(confname) <= 100:
                    title = confname
                else:
                    formula, empirical, Z = configuration.formula
                    if Z == 1:
                        title = formula
                    else:
                        title = f"({empirical}) * {Z}"
        lines.append(title)
        lines.append("1.0")  # The scale factor. SEAMM always uses 1

        # Cell vectors
        vectors = configuration.cell.vectors()
        for vector in vectors:
            a, b, c = vector
            lines.append(f"{a:12.6f} {b:12.6f} {c:12.6f}")

        # Species and number of each
        atnos = configuration.atoms.atomic_numbers
        unique_atnos = sorted(list(set(atnos)), reverse=True)
        unique_elements = molsystem.elements.to_symbols(unique_atnos)

        tmp = [f"{el:>3s}" for el in unique_elements]
        lines.append(" ".join(tmp))

        tmp = [f"{self.element_count[atno]:3d}" for atno in unique_atnos]
        lines.append(" ".join(tmp))

        # Coordinates
        lines.append("Direct")
        fractionals = configuration.atoms.get_coordinates(
            fractionals=True, in_cell=False
        )
        n = len(self._to_SEAMM_order)
        for i_vasp in range(n):
            i_seamm = self.to_SEAMM_order[i_vasp]
            xyz = [f"{x:12.6f}" for x in fractionals[i_seamm]]
            lines.append(" ".join(xyz))

        return "\n".join(lines)

    def parse_xml(self, data_file):
        """Get the data from the vasprun.xml file."""
        results = {}

        tree = etree.parse(data_file)
        root = tree.getroot()

        results["Gelec,iter"] = []
        results["energy,iter"] = []
        results["gradients,iter"] = []
        results["stress,iter"] = []
        results["P,iter"] = []
        results["cell,iter"] = []
        results["a,iter"] = []
        results["b,iter"] = []
        results["c,iter"] = []
        results["alpha,iter"] = []
        results["beta,iter"] = []
        results["gamma,iter"] = []
        results["V,iter"] = []
        results["fractionals,iter"] = []
        results["nElectronicSteps,iter"] = []

        tmpcell = molsystem.Cell(1, 1, 1, 90, 90, 90)

        # The optimization steps, or...
        steps = [child for child in root.iterchildren() if child.tag == "calculation"]
        results["nOptimizationSteps"] = len(steps)

        for step in steps:
            # The number of electronic iterations per optimization step
            results["nElectronicSteps,iter"].append(
                len([c for c in step.iterchildren() if c.tag == "scstep"])
            )

            # The forces and stresses are in calculation/
            arrays = {
                v.get("name"): v
                for v in step.iterchildren()
                if v.tag == "varray" and "name" in v.keys()
            }

            # gradients = -forces
            if "forces" in arrays:
                g = []
                for row in arrays["forces"].iterchildren():
                    g.append([-float(f) for f in row.text.split()])
                results["gradients,iter"].append(g)

            # stresses = -stress (VASP has a different sign) in kbar = 0.1 GPa
            if "stress" in arrays:
                tmp = []
                for row in arrays["stress"].iterchildren():
                    tmp.append([-0.1 * float(f) for f in row.text.split()])
                S = [
                    tmp[0][0],
                    tmp[1][1],
                    tmp[2][2],
                    (tmp[1][2] + tmp[2][1]) / 2,
                    (tmp[0][2] + tmp[2][0]) / 2,
                    (tmp[0][1] + tmp[1][0]) / 2,
                ]
                results["stress,iter"].append(S)

                results["P,iter"].append(-(S[0] + S[1] + S[2]) / 3)

            # The fractional coordinates and cell, which are under calculation/structure
            structure = [c for c in step.iterchildren() if c.tag == "structure"][0]

            # The fractional coordinates are in calculation/structure/positions
            arrays = {
                v.get("name"): v
                for v in structure.iterchildren()
                if v.tag == "varray" and "name" in v.keys()
            }
            if "positions" in arrays:
                xyz = []
                for row in arrays["positions"].iterchildren():
                    xyz.append([float(f) for f in row.text.split()])
                results["fractionals,iter"].append(xyz)
            else:
                print("Cannot find the fractionals ('positions') in this step")

            # The cell, which is in calculation/structure/crystal/basis
            crystal = [c for c in structure.iterchildren() if c.tag == "crystal"][0]
            arrays = {
                v.get("name"): v
                for v in crystal.iterchildren()
                if (v.tag == "varray" and "name" in v.keys())
            }
            if "basis" in arrays:
                vectors = []
                for row in arrays["basis"].iterchildren():
                    vectors.append([float(f) for f in row.text.split()])
                tmpcell.from_vectors(vectors)
                results["cell,iter"].append(tmpcell.parameters)
                a, b, c, alpha, beta, gamma = tmpcell.parameters
                results["a,iter"].append(a)
                results["b,iter"].append(b)
                results["c,iter"].append(c)
                results["alpha,iter"].append(alpha)
                results["beta,iter"].append(beta)
                results["gamma,iter"].append(gamma)
                results["V,iter"].append(tmpcell.volume)
            else:
                print("Cannot find the cell vectors ('basis') in this step")

            # The energies are in calculation/energy
            energy = [c for c in step.iterchildren() if c.tag == "energy"][0]
            arrays = {
                v.get("name"): v
                for v in energy.iterchildren()
                if v.tag == "i" and "name" in v.keys()
            }
            if "e_fr_energy" in arrays:
                results["Gelec,iter"].append(float(arrays["e_fr_energy"].text.strip()))
            if "e_0_energy" in arrays:
                results["energy,iter"].append(float(arrays["e_0_energy"].text.strip()))

        return results

    def parse_hdf5(self, data_file):
        """Get the data from the vaspout.h5 file."""
        results = {}

        results["Gelec,iter"] = []
        results["energy,iter"] = []
        results["gradients,iter"] = []
        results["stress,iter"] = []
        results["P,iter"] = []
        results["cell,iter"] = []
        results["a,iter"] = []
        results["b,iter"] = []
        results["c,iter"] = []
        results["alpha,iter"] = []
        results["beta,iter"] = []
        results["gamma,iter"] = []
        results["V,iter"] = []
        results["fractionals,iter"] = []
        # results["nElectronicSteps,iter"] = []

        tmpcell = molsystem.Cell(1, 1, 1, 90, 90, 90)

        with h5py.File(data_file, "r") as hdf5:
            results["model"] = self.model

            # Get the energies. Not yet sure why they have an initial dimension of 1
            section = hdf5["intermediate"]["ion_dynamics"]

            tmp = section["energies"][...].tolist()
            results["nOptimizationSteps"] = len(tmp)
            for Efree, E0, E in tmp:
                results["Gelec,iter"].append(Efree)
                results["energy,iter"].append(E)

            # Gradients are negative of forces
            results["gradients,iter"] = (-section["forces"][...]).tolist()

            # VASP gives force on cell = -stress in kbar = 0.1 GPa
            S = [
                [
                    tmp[0][0],
                    tmp[1][1],
                    tmp[2][2],
                    (tmp[1][2] + tmp[2][1]) / 2,
                    (tmp[0][2] + tmp[2][0]) / 2,
                    (tmp[0][1] + tmp[1][0]) / 2,
                ]
                for tmp in (-0.1 * section["stress"][...]).tolist()
            ]
            results["stress,iter"] = S

            results["P,iter"] = [
                -(S[0] + S[1] + S[2]) / 3 for S in results["stress,iter"]
            ]

            results["fractionals,iter"] = section["position_ions"][...].tolist()

            for vectors in section["lattice_vectors"][...].tolist():
                tmpcell.from_vectors(vectors)
                results["cell,iter"].append(tmpcell.parameters)
                a, b, c, alpha, beta, gamma = tmpcell.parameters
                results["a,iter"].append(a)
                results["b,iter"].append(b)
                results["c,iter"].append(c)
                results["alpha,iter"].append(alpha)
                results["beta,iter"].append(beta)
                results["gamma,iter"].append(gamma)
                results["V,iter"].append(tmpcell.volume)

        return results

    def plot(self, E_units="", F_units=""):
        """Generate a plot of the convergence of the geometry optimization."""
        figure = self.create_figure(
            module_path=("seamm",),
            template="line.graph_template",
            title="Geometry optimization convergence",
        )
        plot = figure.add_plot("convergence")

        x_axis = plot.add_axis("x", label="Step", start=0, stop=0.8)
        y_axis = plot.add_axis("y", label=f"Energy ({E_units})")
        y2_axis = plot.add_axis(
            "y",
            anchor=x_axis,
            label=f"Force ({F_units})",
            overlaying="y",
            side="right",
            tickmode="sync",
        )
        y3_axis = plot.add_axis(
            "y",
            anchor=None,
            label="Distance (Å)",
            overlaying="y",
            position=0.9,
            side="right",
            tickmode="sync",
        )
        x_axis.anchor = y_axis

        plot.add_trace(
            color="red",
            name="Energy",
            width=3,
            x=self._data["step"],
            x_axis=x_axis,
            xlabel="step",
            y=self._data["energy"],
            y_axis=y_axis,
            ylabel="Energy",
            yunits=E_units,
        )

        plot.add_trace(
            color="black",
            name="Max Force",
            width=3,
            x=self._data["step"],
            x_axis=x_axis,
            xlabel="step",
            y=self._data["max_force"],
            y_axis=y2_axis,
            ylabel="Max Force",
            yunits=F_units,
        )

        plot.add_trace(
            color="green",
            name="RMS Force",
            width=3,
            x=self._data["step"],
            x_axis=x_axis,
            xlabel="step",
            y=self._data["rms_force"],
            y_axis=y2_axis,
            ylabel="RMS Force",
            yunits=F_units,
        )

        plot.add_trace(
            color="blue",
            name="Max Step",
            width=3,
            x=self._data["step"],
            x_axis=x_axis,
            xlabel="step",
            y=self._data["max_step"],
            y_axis=y3_axis,
            ylabel="Max Step",
            yunits="Å",
        )

        figure.grid_plots("convergence")

        # Write to disk
        path = Path(self.directory) / "Convergence.graph"
        figure.dump(path)

        if "html" in self.options and self.options["html"]:
            path = Path(self.directory) / "Convergence.html"
            figure.template = "line.html_template"
            figure.dump(path)
