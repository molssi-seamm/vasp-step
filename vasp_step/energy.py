# -*- coding: utf-8 -*-

"""Non-graphical part of the Energy step in a VASP flowchart"""

import configparser
import csv
from datetime import datetime, timezone
import h5py
import importlib
import logging
from math import ceil as ceiling
from pathlib import Path
import pkg_resources
import platform
import pprint  # noqa: F401
import shutil
import textwrap
import time

from cpuinfo import get_cpu_info
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
            title="Energy",
            extension=extension,
            module=__name__,
            logger=logger,
        )  # yapf: disable

        self._calculation = "Energy"
        self._model = None
        self._metadata = vasp_step.metadata
        self.parameters = vasp_step.EnergyParameters()
        self._to_VASP_order = {}  # translation from SEAMM order to VASP
        self._to_SEAMM_order = {}  # translation from VASP order to SEAMM

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

        return self.header + "\n" + __(text, **P, indent=4 * " ").__str__()

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
            ce["NTASKS"] = n_threads
            self.logger.debug(f"VASP will use {n_threads} threads.")

            files = self.get_input(P)

            printer.important(
                self.indent + f"    VASP will use {n_threads} MPI processes."
            )
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

    def analyze(self, P=None, configuration=None, indent="", table=None, **kwargs):
        """Do any analysis of the output from this step.

        Also print important results to the local step.out file using
        "printer".

        Parameters
        ----------
        indent: str
            An extra indentation for the output
        """
        text = ""
        results = {}
        data_file = self.wd / "vaspout.h5"
        with h5py.File(data_file, "r") as hdf5:
            results["model"] = self.model

            # Get the energies. Not yet sure why they have an initial dimension of 1
            section = hdf5["intermediate"]["ion_dynamics"]

            tmp = section["energies"][...]
            Efree, E0, E = tmp[-1].tolist()
            results["Gelec"] = Efree
            results["energy"] = E

            tmp = section["forces"][...]
            dE = (-tmp[-1]).tolist()
            results["gradients"] = dE

            # VASP gives force on cell = -stress
            tmp = section["stress"]
            S = (-tmp[-1]).tolist()
            results["stress"] = S

            P = -(S[0][0] + S[1][1] + S[2][2]) / 3
            results["P"] = P

            tmp = section["lattice_vectors"]
            lattice = tmp[-1].tolist()

        cell = molsystem.Cell(1, 1, 1, 90, 90, 90)
        cell.from_vectors(lattice)
        a, b, c, alpha, beta, gamma = cell.parameters
        results["a"] = a
        results["b"] = b
        results["c"] = c
        results["alpha"] = alpha
        results["beta"] = beta
        results["gamma"] = gamma

        metadata = vasp_step.metadata["results"]
        if table is None:
            table = {
                "Property": [],
                "Value": [],
                "Units": [],
            }

        for key, title in (
            ("model", "model"),
            ("energy", "E"),
            ("Gelec", "Gelec"),
            ("P", "P"),
            ("a", "a"),
            ("b", "b"),
            ("c", "c"),
            ("alpha", "\N{GREEK SMALL LETTER ALPHA}"),
            ("beta", "\N{GREEK SMALL LETTER BETA}"),
            ("gamma", "\N{GREEK SMALL LETTER GAMMA}"),
        ):
            tmp = metadata[key]
            if "format" in tmp:
                fmt = tmp["format"]
            else:
                fmt = "s"
            units = tmp["units"]
            table["Property"].append(title)
            table["Value"].append(f"{results[key]:{fmt}}")
            table["Units"].append(units)

        tmp = tabulate(
            table,
            headers="keys",
            tablefmt="rounded_outline",
            colalign=("center", "decimal", "left"),
            disable_numparse=True,
        )
        length = len(tmp.splitlines()[0])
        text_lines = []
        header = "Results"
        text_lines.append(header.center(length))
        text_lines.append(tmp)

        if text != "":
            text = str(__(text, indent=self.indent + 4 * " "))
            text += "\n\n"
        text += textwrap.indent("\n".join(text_lines), self.indent + 7 * " ")
        printer.normal(text)

        # Store the results as requested
        self.store_results(
            configuration=configuration,
            data=results,
        )

    def get_input(self, P=None):
        """Get all the input for VASP"""

        # Get the values of the parameters, dereferencing any variables
        if P is None:
            P = self.parameters.current_values_to_dict(
                context=seamm.flowchart_variables._data
            )

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
        keywords["NCORE"] = P["ncore"]
        keywords["LPLANE"] = ".True." if P["lplane"] else ".False."
        keywords["LREAL"] = "Auto" if P["lreal"] else ".False."
        keywords["LH5"] = ".True."
        keywords["IBRION"] = -1
        keywords["NSW"] = 0
        keywords["NELM"] = P["nelm"]
        keywords["EDIFF"] = f'{P["ediff"]:.2E}'
        efermi = P["efermi"]
        if "middle" in efermi:
            keywords["EFERMI"] = "MIDGAP"
        elif efermi == "legacy":
            keywords["EFERMI"] = "Legacy"
        else:
            keywords["EFERMI"] = efermi.m_as("eV")

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

        # The energy cuttoff, which may be an expression of ENMAX
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

        # Calculate on-site density and spin
        if P["lorbit"]:
            keywords["LORBIT"] = 11

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
        if "explicit" in P["k-grid method"]:
            lines.append("Explicit k-point mesh")
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
        if "Monkhorst" in centering:
            lines.append("Monkhorst-Pack")
        else:
            lines.append("Gamma")
        lines.append(f"{na} {nb} {nc}")
        lines.append("0 0 0")

        return "\n".join(lines)

    def get_POSCAR(self, P=None):
        """Get the coordinate information for VASP (POSCAR file)."""
        system, configuration = self.get_system_configuration()

        # Prepare to reorder the atoms into descending atomic number
        atnos = configuration.atoms.atomic_numbers
        unique_atnos = sorted(list(set(atnos)), reverse=True)
        unique_elements = molsystem.elements.to_symbols(unique_atnos)

        # The dictionarie to translate to/from the VASP order
        self._to_VASP_order = {}
        self._to_SEAMM_order = {}
        count = {atno: 0 for atno in atnos}
        for atno in atnos:
            count[atno] += 1
        n = 0
        offset = {}
        for atno in unique_atnos:
            offset[atno] = n
            n += count[atno]
        for original, atno in enumerate(atnos):
            new = offset[atno]
            self._to_VASP_order[original] = new
            self._to_SEAMM_order[new] = original
            offset[atno] += 1

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
        tmp = [f"{el:>3s}" for el in unique_elements]
        lines.append(" ".join(tmp))

        tmp = [f"{count[atno]:3d}" for atno in unique_atnos]
        lines.append(" ".join(tmp))

        # Coordinates
        lines.append("Direct")
        fractionals = configuration.atoms.get_coordinates(
            fractionals=True, in_cell=False
        )
        n = len(self._to_SEAMM_order)
        for i_vasp in range(n):
            i_seamm = self._to_SEAMM_order[i_vasp]
            xyz = [f"{x:12.6f}" for x in fractionals[i_seamm]]
            lines.append(" ".join(xyz))

        return "\n".join(lines)
