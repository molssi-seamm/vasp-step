=======
History
=======
2026.10.7 -- The step declares what its cost model is made of
    * ``energy.TIMING_SPEC`` -- valence electrons and the grid volume (the cell
      volume scaled by (ENCUT/500 eV)^1.5, now a descriptor of the record) as size
      variables, the functional as the method class, the task, ionic steps as the
      unit, k-points as a multiplier -- is passed when a run is recorded (seamm-exec
      2026.10.6.1 writes it beside the records), so the cost model is fitted from the
      step's own description rather than a table in seamm-exec.
    * Removed the unused Docker option from the vasp.ini template.

2026.10.6 -- Timing records that a cost model can be fitted to
    * Each VASP run -- the Energy step's, and a model-chemistry task when
      ``analyze_task`` is given its task -- appends a record to
      ``~/.seamm.d/timing/vasp.csv`` through ``seamm_exec.timing``: the machine
      class, ranks, wall time and outcome, and the descriptors of the calculation
      -- the variables of the cost model (valence electrons, cell volume, ENCUT,
      k-points), ALGO, EDIFF, PREC, ISPIN, IBRION/NSW, the functional and
      potentials, the atoms, and from the OUTCAR the electronic and ionic steps,
      ranks and VASP's own times. This replaces the step's own CSV, which carried
      the POSCAR, INCAR and KPOINTS text and grew without bound (#18). See
      seamm_exec's campaign of 2026-10-05.
    * Requires seamm-exec 2026.10.6.
2026.10.3.1 -- Realistic time estimates for VASP tasks
    * The estimated time of a VASP calculation, which decides how calculations are
      bundled into batch jobs, is now fitted to about 400,000 VASP runs on ARC's
      TinkerCliffs and checked against the MBE prototype's 64,656: it was up to five times
      too low for large cells.
    * A periodic cell's calculation gets a time limit of three times its estimate, at
      least an hour.
2026.10.3 -- VASP as a model chemistry, for the MBE step
    * VASP can be used as a model chemistry, ``VASP:DFT@<functional>/<potentials>@<ENCUT>``
      (e.g. ``VASP:DFT@r2SCAN-D4/PAW-hard@1200``), so steps that evaluate many structures
      -- first the MBE step's periodic low level -- run VASP as separate calculations,
      locally or bundled on a cluster, and reuse finished ones on a rerun.
    * A molecule can be placed in a box registered on a parent cell's FFT grid (whole grid
      steps, the same atom-to-grid offsets as in the cell), so that many-body increments of
      plane-wave calculations cancel the "egg-box" error; the cell's grid is set
      explicitly.
    * ``-D4`` functionals add the D4 dispersion with the dftd4 program after VASP (set
      ``dftd4`` in vasp.ini), since many VASP builds lack D4: periodic for a cell, for the
      isolated molecule for a fragment.
    * ``PAW-hard`` uses the hard potentials (H, B, C, N, O, F, P, S, Cl); an ENCUT below
      the potentials' ENMAX, and small cells without k-points, are refused.
    * Without the graphical interface, elements without a chosen potential now get the
      set's default instead of an error. The inputs of the VASP step itself are unchanged.
2026.9.29 -- Bugfix: a vasp.ini template to edit, and an installer that writes it
    * The step expected to create ``~/SEAMM/vasp.ini`` from a template, but the template
      was never included, so running VASP without the file failed. The template is now
      included, with comments explaining each option for a local build or environment
      modules.
    * Added ``vasp-step-installer``, which the SEAMM Manager runs when the step is
      installed, to write the template to ``~/SEAMM/vasp.ini`` for you to edit.
    * If ``vasp.ini`` gives no command line, ``vasp_std`` (and ``vasp_gam`` and
      ``vasp_ncl``) are looked for on the PATH; otherwise the error says what to set.
      Gamma-point calculations use ``code`` when there is no Gamma-only build, and
      non-collinear calculations without ``noncollinear_code`` say that it is needed.
    * The documentation describes installing with the SEAMM Manager and configuring
      ``vasp.ini``.

2026.9.27 -- The VASP potentials can belong to the installation
    * The PAW potentials were always taken from ``~/SEAMM/Parameters/VASP``. They are now
      taken from the ``Parameters/VASP`` directory of the SEAMM installation in use,
      falling back to ``~/SEAMM/Parameters/VASP`` if that installation has none. A
      second installation such as ``~/SEAMM_DEV`` therefore works without copying the
      potentials, but can have its own. Requires seamm-util 2026.9.27.1.

2026.7.28: Initial wavefunction can reference another job

    * **Initial wavefunction** can now reference another job's WAVECAR, via
      ``job://<job number>/<name>`` (SEAMM's ``Node.file_path`` gained
      read-only cross-job references) -- useful for seeding from a
      wavefunction computed in a different job.

2026.3.1: Internal: switching from deprecated library pkg_resources to importlib

2026.2.5: Corrected r2SCAN and added functionality
    * Corrected an error with r2SCAN which resulted in running rSCAN instead!
    * Added atomic and elemental reference energies for PBE, PBE-D3BJ, r2SCAN, and
      r2SCAN-D3BJ with a 700 eV cutoff. Have all atoms through Pu and elements through
      Ar except for S which is still running.
    * Added changes in the cell parameters, density and volume to the available results,
      as well as the elapsed time for the VASP calculation and the number of processors
      used and the energy per atom.
    * Add the standard system and configuration handling to create new configurations
      and systems to store the results of calculations.
    * Added the ability to start from a previous WAVECAR file, be default that from the
      previous step of the VASP calculation.
    * Switched the default to using the new line search (ISEARCH=1) in optimizations.
    * Corrected the geometry convergence parameter to be the maximum force on an atom or
      cell coordinate, not the square.

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
