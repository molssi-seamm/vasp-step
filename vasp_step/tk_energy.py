# -*- coding: utf-8 -*-

"""The graphical part of a Energy step"""

import json
import pprint  # noqa: F401
import tkinter as tk
import tkinter.ttk as ttk

import vasp_step  # noqa: F401, E999
import seamm
from seamm_util import installation_path
from seamm_util import ureg, Q_, units_class, CompactJSONEncoder  # noqa: F401, E999
import seamm_widgets as sw

from .potentials import default_potentials  # noqa: F401, E402

element_layout = [
    [
        "H",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "He",
    ],  # noqa: E501, E201
    [
        "Li",
        "Be",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
    ],  # noqa: E501
    [
        "Na",
        "Mg",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
    ],  # noqa: E501
    [
        "K",
        "Ca",
        "",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Ga",
        "Ge",
        "As",
        "Se",
        "Br",
        "Kr",
    ],  # noqa: E501, E201
    [
        "Rb",
        "Sr",
        "",
        "Y",
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "In",
        "Sn",
        "Sb",
        "Te",
        "I",
        "Xe",
    ],  # noqa: E501
    [
        "Cs",
        "Ba",
        "",
        "Lu",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
        "Tl",
        "Pb",
        "Bi",
        "Po",
        "At",
        "Rn",
    ],  # noqa: E501
    [
        "Fr",
        "Ra",
        "",
        "Lr",
        "Rf",
        "Db",
        "Sg",
        "Bh",
        "Hs",
        "Mt",
        "Ds",
        "Rg",
        "Cn",
        "Nh",
        "Fl",
        "Mc",
        "Lv",
        "Ts",
        "Og",
    ],  # noqa: E501
    [],
    [
        "",
        "",
        "",
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
    ],  # noqa: E501, E201
    [
        "",
        "",
        "",
        "Ac",
        "Th",
        "Pa",
        "U",
        "Np",
        "Pu",
        "Am",
        "Cm",
        "Bk",
        "Cf",
        "Es",
        "Fm",
        "Md",
        "No",
    ],  # noqa: E501, E201
]  # yapf: disable


class TkEnergy(seamm.TkNode):
    """
    The graphical part of a Energy step in a flowchart.

    Attributes
    ----------
    tk_flowchart : TkFlowchart = None
        The flowchart that we belong to.
    node : Node = None
        The corresponding node of the non-graphical flowchart
    canvas: tkCanvas = None
        The Tk Canvas to draw on
    dialog : Dialog
        The Pmw dialog object
    x : int = None
        The x-coordinate of the center of the picture of the node
    y : int = None
        The y-coordinate of the center of the picture of the node
    w : int = 200
        The width in pixels of the picture of the node
    h : int = 50
        The height in pixels of the picture of the node
    self[widget] : dict
        A dictionary of tk widgets built using the information
        contained in Energy_parameters.py

    See Also
    --------
    Energy, TkEnergy,
    EnergyParameters,
    """

    def __init__(
        self,
        tk_flowchart=None,
        node=None,
        canvas=None,
        x=None,
        y=None,
        w=200,
        h=50,
    ):
        """
        Initialize a graphical node.

        Parameters
        ----------
        tk_flowchart: Tk_Flowchart
            The graphical flowchart that we are in.
        node: Node
            The non-graphical node for this step.
        namespace: str
            The stevedore namespace for finding sub-nodes.
        canvas: Canvas
           The Tk canvas to draw on.
        x: float
            The x position of the nodes center on the canvas.
        y: float
            The y position of the nodes cetner on the canvas.
        w: float
            The nodes graphical width, in pixels.
        h: float
            The nodes graphical height, in pixels.

        Returns
        -------
        None
        """
        self.dialog = None
        self.available = {}  # The available potentials for the current set
        self.current_potential_set = None
        self.default_potential = {}

        super().__init__(
            tk_flowchart=tk_flowchart,
            node=node,
            canvas=canvas,
            x=x,
            y=y,
            w=w,
            h=h,
        )

        # Make sure we have the information about the PAW potentials
        directory = installation_path("Parameters", "VASP")
        index = directory / "index.json"
        if index.exists():
            with index.open() as fd:
                self.potential_metadata = json.load(fd)
        else:
            self.potential_metadata = self.node.parent.catalog_potentials()
            with index.open("w") as fd:
                json.dump(
                    self.potential_metadata,
                    fd,
                    indent=4,
                    sort_keys=True,
                    cls=CompactJSONEncoder,
                )
            print(f"Wrote the index to {index}")

    def create_dialog(self, title="Energy"):
        """
        Create the dialog. A set of widgets will be chosen by default
        based on what is specified in the Energy_parameters
        module.

        Parameters
        ----------
        None

        Returns
        -------
        None

        See Also
        --------
        TkEnergy.reset_dialog
        """

        # Shortcut for parameters
        P = self.node.parameters

        frame = super().create_dialog(title=title, default_number_values=1)

        # Make scrollable in case contents too large
        self["scrolled frame"] = sw.ScrolledFrame(frame)
        self["scrolled frame"].grid(row=0, column=0, sticky=tk.NSEW)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        main_frame = self["main frame"] = self["scrolled frame"].interior()

        # Then create the widgets
        for key in (
            "input only",
            "use hdf5 files",
            "calculate stress",
            "save gradients",
        ):
            self[key] = P[key].widget(main_frame)

        # Frame to isolate widgets for the calculation definition
        calculation_frame = self["calculation frame"] = ttk.LabelFrame(
            main_frame,
            borderwidth=4,
            relief="sunken",
            text="Single-Point Energy Control",
            labelanchor="n",
            padding=10,
        )
        # Then create the widgets and place them
        row = 0
        widgets = []
        for key in ("calculate stress", "save gradients"):
            w = self[key] = P[key].widget(calculation_frame)
            w.grid(row=row, column=0, sticky="ew")
            row += 1
            widgets.append(w)
        sw.align_labels(widgets, sticky="e")

        # Frame to isolate widgets
        e_frame = self["model frame"] = ttk.LabelFrame(
            main_frame,
            borderwidth=4,
            relief="sunken",
            text="Electronic Structure Definition",
            labelanchor="n",
            padding=10,
        )

        # Then create the widgets
        for key in ("model", "submodel", "spin polarization", "nonspherical PAW"):
            self[key] = P[key].widget(e_frame)
            try:
                self[key].configure(width=45)
            except Exception:
                pass

        for key in ("model", "submodel"):
            self[key].bind("<<ComboboxSelected>>", self.reset_model_frame)
            self[key].bind("<Return>", self.reset_model_frame)
            self[key].bind("<FocusOut>", self.reset_model_frame)

        # Frame for the control of the electronic minimization
        scf_frame = self["scf frame"] = ttk.LabelFrame(
            main_frame,
            borderwidth=4,
            relief="sunken",
            text="Electronic Minimization",
            labelanchor="n",
            padding=10,
        )

        # Then create the widgets and place them
        row = 0
        widgets = []
        for key in (
            "electronic method",
            "nelm",
            "nelmin",
            "ediff",
            "precision",
            "initial wavefunction",
        ):
            w = self[key] = P[key].widget(scf_frame)
            w.grid(row=row, column=0, sticky="ew")
            row += 1
            widgets.append(w)
        sw.align_labels(widgets, sticky="e")

        for key in ("electronic method",):
            self[key].bind("<<ComboboxSelected>>", self.reset_performance_frame)
            self[key].bind("<Return>", self.reset_performance_frame)
            self[key].bind("<FocusOut>", self.reset_performance_frame)

        # The k-space integration
        k_frame = self["k-space frame"] = ttk.LabelFrame(
            main_frame,
            borderwidth=4,
            relief="sunken",
            text="K-space Integration",
            labelanchor="n",
            padding=10,
        )
        for key in vasp_step.EnergyParameters.kspace_parameters:
            self[key] = P[key].widget(k_frame)

        for key in ("na", "nb", "nc"):
            self[key].entry.configure(width=4)

        for key in ("k-grid method", "occupation type"):
            self[key].bind("<<ComboboxSelected>>", self.reset_kspace_frame)
            self[key].bind("<Return>", self.reset_kspace_frame)
            self[key].bind("<FocusOut>", self.reset_kspace_frame)

        # Frame for the performance parameters
        performance_frame = self["performance frame"] = ttk.LabelFrame(
            main_frame,
            borderwidth=4,
            relief="sunken",
            text="Performance Parameters",
            labelanchor="n",
            padding=10,
        )

        # Then create the widgets
        for key in (
            "np",
            "ncore",
            "kpar",
            "lplane",
            "lreal",
            "lscalapack",
            "lscalu",
            "nsim",
        ):
            self[key] = P[key].widget(performance_frame)

        s_frame = self["structure frame"] = ttk.LabelFrame(
            main_frame,
            borderwidth=4,
            relief="sunken",
            text="Configuration Handling",
            labelanchor="n",
            padding=10,
        )
        row = 0
        widgets = []
        for key in ("structure handling", "system name", "configuration name"):
            self[key] = P[key].widget(s_frame)
            self[key].grid(row=row, column=0, sticky=tk.EW)
            widgets.append(self[key])
            row += 1
        sw.align_labels(widgets, sticky=tk.E)

        # Top level needs to call reset_dialog to layout the dialog
        if self.node.calculation == "energy":
            self.reset_dialog()

        self.create_potentials_tab()

    def create_potentials_tab(self):
        """Add a tab with a periodic table for setting the potentials."""
        # Shortcut for parameters
        P = self.node.parameters

        notebook = self["notebook"]
        potentials_frame = ttk.Frame(notebook)
        self["potentials frame"] = potentials_frame
        notebook.insert(
            self["results frame"],
            potentials_frame,
            text="Potentials",
            sticky="nsew",
        )

        row = 0

        widgets = []
        for key in ("set of potentials", "plane-wave cutoff"):
            self[key] = P[key].widget(potentials_frame)
            self[key].grid(row=row, column=0, sticky="ew")
            row += 1
            widgets.append(self[key])
        # Align the labels
        sw.align_labels(widgets, sticky="e")
        for key in ("set of potentials",):
            self[key].bind("<<ComboboxSelected>>", self._change_set)
            self[key].bind("<Return>", self._change_set)
            self[key].bind("<FocusOut>", self._change_set)

        # Display the max enmax next to the plane-wave cutoff
        for key in ("enmax",):
            w = self[key] = P[key].widget(potentials_frame)
            w.grid(row=1, column=1, sticky="w")
            w.config(state="readonly")

        # Put the possible potential sets into the widget
        metadata = self.potential_metadata
        sets = [*metadata.keys()]
        self.current_potential_set = self["set of potentials"].value
        self["set of potentials"].config(values=sets)
        if "GW" in P["subset"]:
            self.default_potential = default_potentials[
                self.current_potential_set + "_GW"
            ]
        else:
            self.default_potential = default_potentials[self.current_potential_set]

        # This is used for the radiobutton to control what subset is show/used
        var = self.tk_var["subset"] = tk.StringVar()
        var.set(P["subset"])

        # And make a scrollable periodic table area for the potentials
        self["scrolled periodic table"] = sw.ScrolledFrame(potentials_frame)
        self["scrolled periodic table"].grid(
            row=row, column=0, columnspan=2, sticky=tk.NSEW
        )
        potentials_frame.rowconfigure(row, weight=1)
        potentials_frame.columnconfigure(1, weight=1)
        row += 1

        frame = self["potentials"] = self["scrolled periodic table"].interior()

        # Create the widgets in the periodic table layout
        current = self.node.parameters["potentials"].value
        self.find_available_potentials()

        for _row, elements in enumerate(element_layout):
            for _col, element in enumerate(elements):
                if element != "":
                    if element in self.available:
                        w = self._widget[element] = ttk.Combobox(
                            frame,
                            values=[f"_{element}_", *self.available[element]],
                            width=6,
                        )
                        if element in current:
                            w.set(current[element])
                        else:
                            w.set(self.default_potential[element])
                    else:
                        w = self._widget[element] = ttk.Combobox(
                            frame,
                            width=5,
                        )
                        w.set(f"_{element}_")
                        w.config(state="disabled")
                    w.grid(row=_row, column=_col, sticky=tk.EW)
                    w.bind("<<ComboboxSelected>>", self.update_Emax)
                    w.bind("<Return>", self.update_Emax)
                    w.bind("<FocusOut>", self.update_Emax)

        self.update_Emax()

        # Radio button to allow selecting standard and GW potentials.
        for text in (
            "Show all potentials",
            "Show standard potentials",
            "Show GW potentials",
        ):
            w = self._widget[text] = ttk.Radiobutton(
                potentials_frame,
                text=text,
                value=text,
                variable=self.tk_var["subset"],
                command=self._handle_show_potentials,
            )
            w.grid(row=row, column=0, sticky="w")
            row += 1

        w = self._widget["default potentials"] = ttk.Button(
            potentials_frame,
            text="Reset to default potentials",
            command=self._set_to_default_potentials,
        )
        w.grid(row=row, column=0, sticky="w")
        row += 1

        w = self._widget["clear all potentials"] = ttk.Button(
            potentials_frame,
            text="Clear all potentials",
            command=self._clear_potentials,
        )
        w.grid(row=row, column=0, sticky="w")
        row += 1

        frame.grid_columnconfigure(2, minsize=30)
        frame.grid_rowconfigure(7, minsize=30)

    def _change_set(self, *args):
        """Handle changing the set of potentials to use."""
        self.current_potential_set = self["set of potentials"].get()
        self._handle_show_potentials()

    def _handle_show_potentials(self, current=None):
        """Show or hide the standard or GW potentials."""
        subset = self.tk_var["subset"].get()
        self.find_available_potentials()
        if "GW" in subset:
            self.default_potential = default_potentials[
                self.current_potential_set + "_GW"
            ]
        else:
            self.default_potential = default_potentials[self.current_potential_set]
        if current is None:
            current = self.get_current_potentials()
        for elements in element_layout:
            for element in elements:
                if element != "":
                    if element in self.available:
                        self._widget[element].config(
                            values=[f"_{element}_", *self.available[element]],
                            state="!disabled",
                        )
                        if element in current:
                            if current[element] in self.available[element]:
                                self._widget[element].set(current[element])
                            else:
                                self._widget[element].set(
                                    self.default_potential[element]
                                )
                        else:
                            self._widget[element].set(f"_{element}_")
                    else:
                        self._widget[element].config(state="!disabled")
                        self._widget[element].set(f"_{element}_")
                        self._widget[element].config(
                            values=[f"_{element}_"], state="disabled"
                        )
        self.update_Emax()

    def _set_to_default_potentials(self):
        """Reset to the default potentials."""
        self.find_available_potentials()
        subset = self.tk_var["subset"].get()
        if "GW" in subset:
            self.default_potential = default_potentials[
                self.current_potential_set + "_GW"
            ]
        else:
            self.default_potential = default_potentials[self.current_potential_set]
        for elements in element_layout:
            for element in elements:
                if element != "":
                    if element in self.available:
                        self._widget[element].set(self.default_potential[element])
                    else:
                        self._widget[element].config(state="!disabled")
                        self._widget[element].set(f"_{element}_")
                        self._widget[element].config(state="disabled")
        self.update_Emax()

    def _clear_potentials(self):
        """Reset to the no potentials."""
        for elements in element_layout:
            for element in elements:
                if element != "":
                    self._widget[element].set(f"_{element}_")
        self.update_Emax()

    def find_available_potentials(self):
        potential_data = self.potential_metadata[self.current_potential_set]
        self.available = {}
        subset = self.tk_var["subset"].get()
        for potential, data in potential_data.items():
            if "GW" in subset:
                if "_GW" not in potential:
                    continue
            elif "all" not in subset:
                if "_GW" in potential:
                    continue
            el = potential.split("_")[0]
            if el in self.available:
                self.available[el].append(potential)
            else:
                self.available[el] = [potential]
        for el, tmp in self.available.items():
            self.available[el] = sorted(tmp)

    def get_current_potentials(self):
        """Get the currently selected potentials from the widgets."""
        result = {}
        for elements in element_layout:
            for element in elements:
                if element != "":
                    if element in self.available:
                        tmp = self._widget[element].get()
                        if tmp != "" and not tmp.startswith("_"):
                            result[element] = tmp
        return result

    def get_Emax(self):
        """Get the maximum ENMAX for the currently slsected potentials"""
        potential_data = self.potential_metadata[self.current_potential_set]
        Emax = 0
        for elements in element_layout:
            for element in elements:
                if element != "":
                    if element in self.available:
                        potential = self._widget[element].get()
                        if potential.startswith("_") or potential == "":
                            pass
                        else:
                            Emax = max(Emax, float(potential_data[potential]["Emax"]))
        return Emax

    def update_Emax(self, event=None):
        """Update the value of Emax in the display."""
        self["enmax"].config(state="!readonly")
        self["enmax"].set(self.get_Emax(), unit_string="eV")
        self["enmax"].config(state="readonly")

    def reset_dialog(self, widget=None, row=0):
        """Layout the widgets in the dialog.

        The widgets are chosen by default from the information in
        Energy_parameter.

        This function simply lays them out row by row with
        aligned labels. You may wish a more complicated layout that
        is controlled by values of some of the control parameters.
        If so, edit or override this method

        Parameters
        ----------
        widget : Tk Widget = None

        Returns
        -------
        None

        See Also
        --------
        TkEnergy.create_dialog
        """

        # Remove any widgets previously packed
        frame = self["main frame"]
        for slave in frame.grid_slaves():
            slave.grid_forget()

        # Shortcut for parameters
        P = self.node.parameters

        # Whether to just write input
        widgets = []
        for key in ("input only", "use hdf5 files"):
            self[key].grid(row=row, column=0, sticky=tk.W)
            widgets.append(self[key])
            row += 1
        sw.align_labels(widgets, sticky=tk.E)

        # The calculation and handling of the structure
        self["calculation frame"].grid(
            row=row, column=0, columnspan=2, pady=5, sticky="new"
        )
        self["structure frame"].grid(
            row=row, column=2, columnspan=2, pady=5, sticky="new"
        )
        row += 1

        # The model for the calculation
        self["model frame"].grid(row=row, column=0, columnspan=2, pady=5, sticky="new")
        self["scf frame"].grid(row=row, column=2, columnspan=2, pady=5, sticky="new")
        row += 1
        self["k-space frame"].grid(
            row=row, column=0, columnspan=2, pady=5, sticky="new"
        )
        self["performance frame"].grid(
            row=row, column=2, columnspan=2, pady=5, sticky="new"
        )
        row += 1

        # Layout the energy widgets
        self.reset_calculation_frame()
        self.reset_model_frame()
        self.reset_kspace_frame()
        self.reset_performance_frame()

        # Setup the results if there are any
        have_results = (
            "results" in self.node.metadata and len(self.node.metadata["results"]) > 0
        )
        if have_results and "results" in P:
            self.setup_results()

        return row

    def reset_calculation_frame(self, widget=None):
        """Layout the widgets for the calculation control.

        Note
        ----
        The single point energy has nothing that changes, so do nothing.
        """
        pass

    def reset_model_frame(self, widget=None):
        """Layout the widgets in the dialog.

        The widgets are chosen by default from the information in
        Energy_parameter.

        This function simply lays them out row by row with
        aligned labels. You may wish a more complicated layout that
        is controlled by values of some of the control parameters.
        If so, edit or override this method

        Parameters
        ----------
        widget : Tk Widget = None

        Returns
        -------
        None

        See Also
        --------
        TkEnergy.create_dialog
        """

        # Remove any widgets previously packed
        frame = self["model frame"]
        for slave in frame.grid_slaves():
            slave.grid_forget()

        # Shortcut for parameters
        P = self.node.parameters

        # keep track of the row in a variable, so that the layout is flexible
        # if e.g. rows are skipped to control such as "method" here
        row = 0

        # The model for the calculation
        keys = ["model", "submodel", "spin polarization", "nonspherical PAW"]
        model = self["model"].get()

        # And what models have available sublevels?
        model_data = self.node.metadata["computational models"][
            "Density Functional Theory (DFT)"
        ]["models"]
        models = []
        for tmp in model_data.keys():
            models.append(tmp)
        w = self["model"]
        if w.combobox.cget("values") != models:
            w.config(values=models)
            if not self.is_expr(model) and model not in models and len(models) > 0:
                w.set(models[0])
            else:
                w.set(model)

        # Fill out the possible submodels
        submodel = self["submodel"].get()
        choices = model_data[model]["parameterizations"]
        submodels = [*choices.keys()]

        w = self["submodel"]
        if w.combobox.cget("values") != submodels:
            w.config(values=submodels)
            if (
                not self.is_expr(submodel)
                and submodel not in submodels
                and len(submodels) > 0
            ):
                w.set(submodels[0])
            else:
                w.set(submodel)

        # And the basis
        widgets = []

        for key in keys:
            self[key].grid(row=row, column=0, sticky=tk.EW)
            widgets.append(self[key])
            row += 1

        # Align the labels
        sw.align_labels(widgets, sticky=tk.E)
        frame.columnconfigure(0, weight=1)

        # Setup the results if there are any
        have_results = (
            "results" in self.node.metadata and len(self.node.metadata["results"]) > 0
        )
        if have_results and "results" in P:
            self.setup_results()

    def reset_kspace_frame(self, widget=None):
        """Layout the widgets in the dialog.

        The widgets are chosen by default from the information in
        Energy_parameter.

        This function simply lays them out row by row with
        aligned labels. You may wish a more complicated layout that
        is controlled by values of some of the control parameters.
        If so, edit or override this method

        Parameters
        ----------
        widget : Tk Widget = None

        Returns
        -------
        None

        See Also
        --------
        TkEnergy.create_dialog
        """

        # Remove any widgets previously packed
        frame = self["k-space frame"]
        for slave in frame.grid_slaves():
            slave.grid_forget()

        # keep track of the row in a variable, so that the layout is flexible
        # if e.g. rows are skipped to control such as "method" here
        row = 0
        widgets = []
        widgets1 = []

        # The model for the calculation
        key = "k-grid method"
        method = self[key].get()
        self[key].grid(row=row, column=0, columnspan=4, sticky=tk.W)
        widgets.append(self[key])
        row += 1

        if "explicit" in method:
            for col, key in enumerate(("na", "nb", "nc"), start=1):
                self[key].grid(row=row, column=col, sticky=tk.EW)
            row += 1
        elif "point" in method:
            pass
        else:
            for key in ("k-spacing", "odd grid", "centering"):
                self[key].grid(row=row, column=1, columnspan=3, sticky=tk.EW)
                widgets1.append(self[key])
                row += 1

        key = "occupation type"
        smearing = self[key].get()
        self[key].grid(row=row, column=0, columnspan=4, sticky=tk.W)
        widgets.append(self[key])
        row += 1

        if "Methfessel-Paxton" in smearing:
            key = "Methfessel-Paxton order"
            self[key].grid(row=row, column=1, columnspan=3, sticky=tk.EW)
            widgets1.append(self[key])
            row += 1

        if "without smearing" not in smearing:
            key = "smearing width"
            self[key].grid(row=row, column=1, columnspan=3, sticky=tk.EW)
            widgets1.append(self[key])
            row += 1

        w0 = sw.align_labels(widgets, sticky=tk.E)
        w1 = sw.align_labels(widgets1, sticky=tk.E)
        minwidth = w0 - w1 + 30
        frame.columnconfigure(0, minsize=minwidth)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)
        frame.columnconfigure(3, weight=1)

    def reset_performance_frame(self, widget=None):
        """Layout the widgets in the performance frame.

        Parameters
        ----------
        widget : Tk Widget = None

        Returns
        -------
        None
        """

        # Remove any widgets previously packed
        frame = self["performance frame"]
        for slave in frame.grid_slaves():
            slave.grid_forget()

        electronic_method = self["electronic method"].get()

        keys = ["np", "ncore", "kpar", "lplane", "lreal", "lscalapack", "lscalu"]
        if (
            self.is_expr(electronic_method)
            or "fast" in electronic_method
            or "normal" in electronic_method
        ):
            keys.append("nsim")

        row = 0
        widgets = []
        for key in keys:
            self[key].grid(row=row, column=0, sticky=tk.EW)
            widgets.append(self[key])
            row += 1

        # Align the labels
        sw.align_labels(widgets, sticky=tk.E)

    def right_click(self, event):
        """
        Handles the right click event on the node.

        Parameters
        ----------
        event : Tk Event

        Returns
        -------
        None

        See Also
        --------
        TkEnergy.edit
        """

        super().right_click(event)
        self.popup_menu.add_command(label="Edit..", command=self.edit)

        self.popup_menu.tk_popup(event.x_root, event.y_root, 0)

    def fit_dialog(self):
        """Make the dialog fill the window."""
        screen_w = self.dialog.winfo_screenwidth()
        screen_h = self.dialog.winfo_screenheight()
        w = int(1.0 * screen_w)
        h = int(1.0 * screen_h)
        x = 0
        y = int(0.0 * screen_h / 2)

        self.dialog.geometry(f"{w}x{h}+{x}+{y}")

    def handle_dialog(self, result):
        """Handle closing the dialog.

        Parameters
        ----------
        result : str
            The button that was pressed to close the dialog, or None if the x dialog
            close button was pressed.
        """
        if result is None or result == "Cancel":
            # Reset the potentials to their original values
            P = self.node.parameters
            subset = P["subset"]
            self.tk_var["subset"].set(subset)
            for key in ("set of potentials", "plane-wave cutoff"):
                self[key].set(P[key])
            self.current_potential_set = P["set of potentials"].value
            if "GW" in subset:
                self.default_potential = default_potentials[
                    self.current_potential_set + "_GW"
                ]
            else:
                self.default_potential = default_potentials[self.current_potential_set]
            current = self.node.parameters["potentials"].value
            self._handle_show_potentials(current)
        elif result == "OK":
            self.node.parameters["potentials"].value = self.get_current_potentials()

        # And let the superclass do the rest
        super().handle_dialog(result)
