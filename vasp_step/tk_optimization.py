# -*- coding: utf-8 -*-

"""The graphical part of a Optimization step"""

import tkinter as tk

from .tk_energy import TkEnergy
import seamm_widgets as sw


class TkOptimization(TkEnergy):
    """
    The graphical part of a Optimization step in a flowchart.

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
        contained in Optimization_parameters.py

    See Also
    --------
    Optimization, TkOptimization,
    OptimizationParameters,
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

        super().__init__(
            tk_flowchart=tk_flowchart,
            node=node,
            canvas=canvas,
            x=x,
            y=y,
            w=w,
            h=h,
        )

    def create_dialog(self, title="Optimization"):
        """
        Create the dialog. A set of widgets will be chosen by default
        based on what is specified in the Optimization_parameters
        module.

        Parameters
        ----------
        None

        Returns
        -------
        None

        See Also
        --------
        TkOptimization.reset_dialog
        """

        super().create_dialog(title=title)

        # Shortcut for parameters
        P = self.node.parameters

        calculation_frame = self["calculation frame"]
        calculation_frame.config(text="Optimization Control")

        # Then create the widgets
        for key in (
            "optimize atom positions",
            "optimize cell shape",
            "optimize cell volume",
            "optimization method",
            "number of steps",
            "convergence cutoff",
            "step scale factor",
            "iteration history length",
            "damped MD approach",
            "force scale factor",
            "velocity scale factor",
            "velocity quenching factor",
        ):
            self[key] = P[key].widget(calculation_frame)

        for key in (
            "optimize cell shape",
            "optimize cell volume",
            "optimization method",
            "damped MD approach",
        ):
            self[key].bind("<<ComboboxSelected>>", self.reset_calculation_frame)
            self[key].bind("<Return>", self.reset_calculation_frame)
            self[key].bind("<FocusOut>", self.reset_calculation_frame)

        # Top level needs to call reset_dialog to layout the dialog
        if self.node.calculation == "energy":
            self.reset_dialog()

    def reset_calculation_frame(self, widget=None):
        """Layout the widgets in the calculation frame.

        Parameters
        ----------
        widget : Tk Widget = None

        Returns
        -------
        None
        """

        # Remove any widgets previously packed
        frame = self["calculation frame"]
        for slave in frame.grid_slaves():
            slave.grid_forget()

        # Place the control in the left column
        method = self["optimization method"].get()

        row = 0
        widgets = []
        keys = []
        keys.append("optimization method")
        keys.append("number of steps")
        keys.append("convergence cutoff")

        if self.is_expr(method):
            keys.append("step scale factor")
            keys.append("iteration history length")
            keys.append("damped MD approach")

        elif method == "Damped MD":
            pass
        elif method == "RMM-DIIS":
            keys.append("step scale factor")
            keys.append("iteration history length")
        else:
            keys.append("step scale factor")

        if self.is_expr(method) or method == "Damped MD":
            damping_approach = self["damped MD approach"].get()
            keys.append("damped MD approach")
            if self.is_expr(damping_approach):
                keys.append("force scale factor")
                keys.append("velocity scale factor")
                keys.append("velocity quenching factor")
            elif damping_approach == "damping":
                keys.append("force scale factor")
                keys.append("velocity scale factor")
            else:
                keys.append("velocity quenching factor")

        row = 0
        widgets = []
        for key in keys:
            self[key].grid(row=row, column=0, sticky=tk.EW)
            widgets.append(self[key])
            row += 1

        # Align the labels
        sw.align_labels(widgets, sticky=tk.E)

        # Add the cell constraints in the second column
        shape = self["optimize cell shape"].get() == "yes"
        volume = self["optimize cell volume"].get() == "yes"

        keys = [
            "optimize atom positions",
            "optimize cell shape",
            "optimize cell volume",
        ]
        if not shape and not volume:
            keys.append("calculate stress")
        keys.append("save gradients")

        # Place in the second column
        row = 0
        widgets = []
        for key in keys:
            self[key].grid(row=row, column=1, sticky=tk.EW)
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
        TkOptimization.edit
        """

        super().right_click(event)
        self.popup_menu.add_command(label="Edit..", command=self.edit)

        self.popup_menu.tk_popup(event.x_root, event.y_root, 0)
