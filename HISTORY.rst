=======
History
=======
2025.11.26: Added cohesive and formation energies, and saving gradients
    * Added an option to save the gradients in the configuration.
    * Calculate the cohesive energy and energy of formation if tabulated atom and
      element energies are available for the DFT functional and planewave cutoff.

2025.11.2: Small improvements and better output.
    * Added missing precision control
    * Added control over using HDF5 files
    * Improved the output, particularly for optimization
    * Standardized pressure and stress units to GPa
    * Changed some of the defaults to make standard calculations work better.
      
2025.10.31: Added optimization and improved the output.

2025.10.22: First working version
    * Handles single point energies reasonably well.

2025.10.12: Plug-in created using the SEAMM plug-in cookiecutter.
