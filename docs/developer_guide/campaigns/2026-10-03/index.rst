==========================================================
2026-10-03: the batch path for the MBE step (MBE phase 3)
==========================================================

Part of the MBE campaign: the design is ``~/Work/SEAMM/MBE_correction_step_design.rst``,
and the step is ``mbe_step``. vasp_step gains the Model Chemistry batch contract
(``get_model_chemistry_options``, ``get_task``, ``analyze_task``, ``can_run_task``)
with the "fragment registered to a parent cell" mode the MBE step's periodic low
level needs. Reviewed by the "design" session; approved by Paul.

.. contents::
   :local:

Step 1: the input builder, with no behaviour change
====================================================

``vasp_step/inputs.py`` builds INCAR, POTCAR, KPOINTS and POSCAR from explicit
settings, and the Energy substep now calls it. Values that need the flowchart
(the initial WAVECAR, an ENCUT expression, extra keywords with variables) are
resolved by the substep and passed in.

**Identity test.**

- ``tests/data/inputs`` holds the substep's four input files for four
  representative settings, captured from the *unchanged* code:

  - water at Gamma with stress;
  - Si by k-spacing (odd, Monkhorst-Pack, Methfessel-Paxton, pressure only);
  - spin-polarized LiF (Li_sv, explicit grid, tetrahedron + Blöchl + Fermi-Dirac,
    extra keywords);
  - a water Optimization.

- The test requires the files to stay byte-identical. The POTCAR library is fake,
  since licensed files can't be shipped.
- To regenerate them from the code *before* a change:

  - ``git worktree add <tmp> <old commit>``, then copy the installed
    ``_version.py`` into it;
  - run ``tests/input_harness.py``'s ``capture()`` with that worktree first on
    ``PYTHONPATH``.

  This is how the fixtures were made and re-verified (vasp_step 1690fe2).

**Headless fixes.**

- The default potentials moved from ``tk_energy.py``, which imports tkinter, to
  ``potentials.py``. A calculation set up without the dialog has an empty
  "potentials" parameter: it now gets the set's defaults instead of a KeyError.
- An ENCUT expression of ENMAX takes ENMAX from the potentials when the dialog has
  not set it.

Step 2: the batch path
======================

- **Model chemistries:** ``VASP:DFT@<functional>/<PAW|PAW-hard|PAW-LDA>@<ENCUT
  eV>`` in the existing grammar.

  - Declared: periodic, not MDI, ``prefers_batch``,
    ``stress_convention = "stress"``.
  - vasp_step already turns VASP's kB pressure into a GPa stress (−0.1 × kB), so
    nothing is flipped twice.

- **Grid** (``grid.py``) reproduces the prototype's ``gen_registered.py``:

  - NG is the smallest even 2·3·5·7-smooth n with h ≤ max_spacing (150 for the
    12.4297 Å pilot cell);
  - the box is that n·h with n·h ≥ max(12 Å, extent + padding);
  - the shift is a whole number of grid steps, k = round((L/2 − centre)/h).
  - The tests compare NG, box and coordinates with the prototype's own POSCARs and
    INCARs (cell, m00, d00_18) to 1e-9 Å. The outer pair d00_17, at image
    (0, 0, −1), is checked against a port of ``register``.
  - Only orthorhombic parents are supported.

- **get_task:**

  - INCAR, POTCAR, KPOINTS through the same builder as the substep. A test checks
    that the batch path and the substep write identical INCAR, KPOINTS and POTCAR
    for the same settings. That test caught string values ("no") being truthy, so
    the batch settings now go through ``EnergyParameters`` exactly like the
    substep's.
  - POSCAR in Cartesian with 10 decimals.
  - For a charged structure, NELECT from the POTCARs' ZVAL; for an open shell,
    NUPDOWN.
  - Fragments get the dipole correction (IDIPOL = 4, DIPOL = 0.5 0.5 0.5: a
    registered fragment is centred to within h/2).
  - ``return_files``: INCAR, KPOINTS, POSCAR, OUTCAR, OSZICAR, vasprun.xml,
    vasp.out, and the dftd4 files.
  - ``success_text``: OUTCAR "General timing", and the presence of dftd4.json.
  - ``estimated_seconds`` is calibrated on the prototype: 330 s for a water
    fragment on a 150³ grid at 8 ranks, scaled by grid points, √atoms and ranks.

- **D4:**

  - Neither TinkerCliffs VASP build (6.6.1-intel2025b, 6.5.1-intel2023a) is
    compiled with DFTD4, so IVDW = 13 is out. The dftd4 CLI runs inside the task
    after VASP: ``--func <f> --grad --json dftd4.json --charge Q``.
  - It runs on POSCAR for a cell (periodic) and on ``fragment.xyz`` for a fragment
    (isolated).
  - The rule and its evidence: on monomer m00, dftd4 on the fragment's POSCAR (in
    its 12.43 Å box) gives −1.3399e−3 eV, while the xyz gives −1.3103e−3 eV. The
    prototype's fragment D4 (−1.31026e−3, the library without a lattice) matches
    the xyz exactly. Image dispersion would break the regression and the
    consistency with the molecular level.
  - TinkerCliffs: conda-forge dftd4 4.3.0 at
    ``/projects/seamm/conda-envs/dftd4/bin/dftd4`` (installed with Paul's OK). It
    reproduces the prototype's cell D4 (−2.655539152468714 eV,
    P_D4 −2322.500241 atm) and m00.

- **analyze_task:**

  - Reads vasprun.xml, taking the *calculation's* own ``<energy>`` block: the
    first ``e_0_energy`` inside the last ``<calculation>`` is an SCF step's
    (+7052.9 eV for the pilot cell, against the real −1007.455).
  - Forces go back to the structure's atom order; the stress is −0.1 × the kB
    tensor (vasprun.xml's stress has OUTCAR's "in kB" sign); dftd4's energy,
    gradient and virial are added (σ_D4 = virial/V).
  - An SCF without "aborting loop because EDIFF is reached" in OUTCAR raises.
  - The vaspout.h5 requirement is the Energy substep's, untouched; the batch path
    doesn't need HDF5.
  - Tested on the prototype's own pilot runs. Energies match to 1e-6 kJ/mol;
    forces to 1e-4 kJ/mol/Å, because vasprun.xml has 8 decimals and OUTCAR 6.
  - The stress of the cell is σ = −(P_VASP + P_D4), and gives the prototype's
    −61,519.55 − 2,322.50 atm. vasprun.xml's tensor is asymmetric by 2e-6 GPa.

- **resolver** (entry point ``vasp``): puts vasp.ini's ``gamma_code`` (or
  ``code``) and ``dftd4`` into the command, before the executor fills in
  ``{NTASKS}``.

Noticed, not changed
====================

- metadata.py gives plain ``revPBE`` IVDW = 12, the same as ``revPBE-D3BJ``.
- TinkerCliffs' ``/projects/seamm/SEAMM/vasp.ini`` uses
  ``gamma_code = mpiexec -np {NTASKS} vasp_std``. vasp_gam would be faster for
  Gamma-only runs; the prototype used it.
