VASP as a model chemistry
=========================

Steps that evaluate a *model chemistry* at many structures can use VASP through a
**Model Chemistry**, with no settings in the VASP step itself. The **MBE** step does
this for the periodic low level of its many-body corrections. Each structure runs as
a separate VASP calculation (a *task*), wherever the job's target sends tasks: this
machine, or bundled batch jobs on a cluster. A rerun reuses the ones that finished.

Level strings
-------------

A VASP model chemistry is written

    ``VASP:DFT@<functional>/<potentials>@<ENCUT>``

for example ``VASP:DFT@r2SCAN-D4/PAW-hard@1200``.

- **Functional**: the name the VASP step's dialog shows before " : ", e.g. ``PBE``,
  ``PBE-D3BJ``, ``SCAN``, ``r2SCAN``. A ``-D4`` functional (``PBE-D4``,
  ``r2SCAN-D4``, ...) is the functional plus the D4 dispersion correction from the
  dftd4 program, run in the same task after VASP. Many VASP builds are compiled
  without D4, so VASP's own ``IVDW = 13`` is not used.
- **Potentials**:

  - ``PAW`` is potpaw_PBE.64 with VASP's recommended potentials;
  - ``PAW-hard`` is the same with the hard potentials where they exist (H, B, C, N,
    O, F, P, S, Cl);
  - ``PAW-LDA`` is potpaw_LDA.64.

- **ENCUT** is the plane-wave cutoff in eV. If it is missing, 1.3 × the largest
  ENMAX of the potentials is used. A cutoff below the largest ENMAX is refused.

The functionals with VASP's own D4 (``-D4BJ``, IVDW = 13) are not offered: they
need a VASP compiled with DFTD4, which many builds are not. Use ``-D4`` instead.

The calculations use the settings for accurate energies and forces: PREC =
Accurate, EDIFF = 1e-7, ALGO = All, no symmetry, and no WAVECAR or CHGCAR. The stress
is computed for a periodic structure when it is asked for. The POTCAR (a licensed
file) is deleted when the calculation finishes.

**k-points.** A cell is sampled at the Gamma point alone, which suits the large
cells of liquids and fragment boxes. A cell narrower than 10 Å is refused unless a
k-point spacing is given (the ``k_spacing`` option, in 1/Å), which then sets a
Gamma-centred mesh.

Molecules in boxes, registered on a cell
-----------------------------------------

VASP is periodic, so a molecule (a fragment of a cell, for the MBE step) runs in a
box. To make many-body increments of plane-wave calculations meaningful, the box
must be *registered* on its parent cell's FFT grid:

- the cell's grid is set explicitly, with a spacing of at most 0.0829 Å;
- each fragment's box is a whole number of those grid steps, also with an explicit
  grid;
- the fragment is the cell's coordinates shifted by whole grid steps.

Every atom then sits at the same offset from the grid in every calculation, and the
"egg-box" error (2–3 meV/Å per calculation) cancels.

**Only compact fragments belong in a periodic code.** An extended fragment interacts
with its own periodic images in a box of affordable size. Padding the box by
15 Å instead of 7.5 Å roughly halves that error at about six times the cost, and the
dipole correction (used for every fragment) does not cure it. The MBE step therefore
uses the periodic level only for monomers and close pairs, and a molecular code for
the rest.

**Dispersion of fragments is that of the isolated fragment.** The D4 correction of a
fragment is computed for the isolated molecule, without the dispersion with its
images in the box: about 3 × 10⁻⁵ eV per water monomer in a 12.4 Å box, and more
for larger fragments. This matches the molecular codes' fragments, so the many-body
increments stay consistent. A cell gets the periodic D4.

Setting up the programs
-----------------------

On each machine that runs the tasks, ``vasp.ini`` gives the commands:

- ``gamma_code``, the Gamma-point build, e.g. ``mpiexec -np {NTASKS} vasp_gam``;
- ``dftd4``, the dftd4 program (conda-forge ``dftd4``, version 4) for the ``-D4``
  functionals.

The POTCARs are read where the tasks are made, from the VASP potential library
(``<SEAMM>/Parameters/VASP``), so that machine needs the library.

Cost and time limits
--------------------

Each calculation carries an estimate of its wall time. It is fitted to the VASP step's
own timing records on ARC's TinkerCliffs (about 400,000 Gamma-point single points on
8 ranks), from the valence electrons and the plane-wave grid (volume × (ENCUT/500 eV)^1.5).
It is within a factor of 1.3 for two thirds of those runs, and within that of the
medians of the MBE prototype's 64,656 runs. The estimate decides how many calculations
share a batch job. A cell, which runs alone, also gets a time limit of three times its
estimate, at least an hour.

Results
-------

Each structure returns its energy (kJ/mol, σ → 0) and gradients (kJ/mol/Å), and for
a periodic structure the stress: a 3×3 tensor in GPa, as a stress (σ = −P, positive
when tensile), the same sign as the VASP step stores. Every model chemistry declares
this as ``stress_convention = "stress"``. A calculation whose SCF did not reach EDIFF
is reported as failed, never used.
