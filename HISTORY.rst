=======
History
=======
2026.2.1: Added -D3BJ functionals and reference energies for r2SCAN-D3BJ
    * Added the natively supported functionals with D3(BJ) dispersion corrections,
      PBE-D3BJ, PBEsol-D3BJ, RPBE-D3BJ, revPBE-D3BJ, M06-l-D3BJ, TPSS-D3BJ, and
      SCAN-D3BJ. Also added r2SCAN-D3BJ using the D3 parameters from ORCA.
    * Added reference energies for r2SCAN-D3BJ with a 700 ev cutoff for elements He-Ca
      except P and S.

2026.1.28: Bugfix: protected against missing data calculating energies of formation
    * Fixed issues with missing data when trying to calculate the energies of formation,
      etc. Now the code notes the error and continues safely.

2026.1.1: Added more reference energies for r2SCAN
    * Added reference energies for r2SCAN for H-S and Ar

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
