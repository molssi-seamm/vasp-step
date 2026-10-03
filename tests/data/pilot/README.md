# The MBE prototype's VASP runs (pilot frame), for the batch-path tests

From TinkerCliffs `/projects/seamm/psaxe/periodic/frames/pilot_opls0997/vasp.tar.gz`,
VASP 6.6.1 gamma-only, r2SCAN, O_h/H_h at 1200 eV, written by the prototype's
`gen_stage2_frame.py` / `gen_registered.py` (2026-09-29):

- `r2_cell`: the 64-water cell (12.4297 Å, NGX 150).
- `r2_m00`: monomer m00, in a registered box.
- `r2_d00_18`: a close pair (O-O < 3.5 Å), in a registered box.

Each has INCAR, POSCAR and order.txt as written, and vasprun.xml and OUTCAR (gzipped)
as VASP wrote them. `dftd4.json` is the dftd4 CLI (conda-forge 4.2.0,
`--func r2scan --grad --json`) on the cell's POSCAR (periodic) and on each fragment
as an isolated molecule. `expected.json` holds the prototype's own numbers from
seamm_mbe's pilot fixture: VASP energy/forces in molecule order, D4 energy/forces
and the cell's stress ("in kB") and dftd4 virial. `frags_subset.json` has the
fragments' coordinates from the prototype's frags.json (m00, d00_18, and the outer
pair d00_17, which sits at a non-zero image).
