# Staggered mesh MP3, CCD, and G0W0

FSEC imports the restricted periodic implementations by Xin Xing from
`staggered_mesh_methods`. The package includes both two-mesh and three-mesh
MP3/CCD and staggered G0W0 with linearized or nonlinear quasiparticle equations.
The nonlinear quasiparticle solver still uses fixed G0 and W0; it does not
perform self-consistent GW.

The initial numerical validation targets insulating, three-dimensional cells
with integer occupations. MP3 and CCD correlation energies are in hartree per
primitive cell. GW orbital energies are in hartree.

## Solvers and examples

| Import from | Classes | Runnable example |
| --- | --- | --- |
| `fsec.staggered_mesh.mp` | `KMP3`, `KMP3_2mesh`, `KMP3_3mesh` | [MP3](../examples/staggered_mp3.py) |
| `fsec.staggered_mesh.cc` | `KRCCD_2mesh`, `KRCCD_3mesh` | [CCD](../examples/staggered_ccd.py) |
| `fsec.staggered_mesh.gw` | `KRGWAC_stagger_submesh`, `KRGWAC_stagger_extend` | [G0W0](../examples/staggered_gw.py) |

Construct an ordinary converged PySCF mean-field reference first. The MP3 and
CCD examples use KRHF; the GW example uses PBE KRKS. With
`flag_submesh=True`, MP3/CCD extract the staggered meshes from the reference
mesh: sampled directions must be divisible by two or three, respectively.
With `flag_submesh=False`, they generate additional orbitals with `get_bands`.
For GW, the two public constructors select those workflows explicitly and
take a three-entry `nks_target` mesh size.

These are dense reference implementations. Start with the small examples;
integral and amplitude storage grows rapidly with the mesh and orbital count.
GW runs without MPI or the archived `pyrpa` package. MP3/CCD integral
transformations can use Python thread workers, in addition to PySCF threading.

## Migration from the archive

### MP3

`kernel()` returns `(e_corr, t2)`. The energy attributes are:

- `e_corr_mp2`: second-order correlation energy, E2.
- `e_corr_mp3`: third-order increment, E3.
- `e_corr`: E2 + E3.
- `t2`: first-order amplitudes, following the newer PySCF KMP3 convention.

The archive instead returned the sum of first- and second-order amplitudes in
`t2`. The port retains the combined amplitudes internally for evaluating E3.
Consequently, an energy match does not imply that old and new `t2` arrays have
the same meaning. `with_t2=False` suppresses the public stored amplitudes.

For ordinary `KMP3`, the default `madelung=None` selects the Ewald counterterm
when the reference uses `exxdiv='ewald'`. An explicit boolean overrides that
choice. The staggered formulations retain their archive defaults:
`KMP3_2mesh(..., madelung=True)` and
`KMP3_3mesh(..., madelung=False)`. These flags describe the respective
formulations' corrections; they are not interchangeable with a GW correction.

### CCD

The public solvers return `(e_corr, t1, t2)` and maintain zero singles. Both
CCD variants default to `madelung=False`, as in the imported source. Inspect
`converged` before using a result. The archived three-mesh implementation
without symmetry is a validation reference, not an additional public solver.

### GW

| Archived interface | FSEC interface |
| --- | --- |
| `linearized` | `qpe_linearized` |
| `kernel(nw=100, iw_cutoff=5.0, ...)` | Set `nw` and `ac_iw_cutoff` on the object, then call `kernel(...)` |
| Returned selected QP array / `quasi_mo_energy` | `kernel()` returns `None`; read target-grid QP energies from `mo_energy` |
| Mean-field energies in `mo_energy` | Reference energies are kept separately from QP results |
| Procedural `pyrpa` continuation helpers | PySCF `PadeAC` and `TwoPoleAC` |
| Mandatory `mpi4py` and rank scratch files | Serial execution without those dependencies |

`kptlist` indexes the target mesh, not the combined reference/shifted mesh.
`orbs` indexes the reference bands. `mo_energy` has shape
`(nkpts_target, nmo)`; unrequested or failed entries are `NaN`.
`qpe_converged` is a boolean array of the same shape, and `converged` reports
whether every requested state converged. `mo_energy_ref` holds target-grid
reference energies. Set `qpe_linearized_range=None` to disable the current
PySCF-style renormalization-factor clipping when comparing with the archive.
Frozen orbitals and low-dimensional GW are unsupported in this port.

The self-energy is sampled at `ef + i*omega`. Fitting and evaluating the current
PySCF continuation objects consistently uses absolute frequencies, preserving
the archive's Fermi-energy offset.

`fc=False` disables both staggered finite-size corrections. With `fc=True`,
`fc_ex_madelung` and `fc_corr_madelung` independently select the archived
exchange and correlation Madelung corrections. All three flags default to
true. The standard PySCF GW `fc` calculation uses a different correction
formula and is not an equivalent numerical reference for these terms.

### Compatibility changes

The port replaces factory-based `isinstance` checks with concrete PySCF
classes and checks KRKS before KRHF because KRKS inherits from KRHF. It also
uses the appropriate `dm`/`dm_kpts` argument, corrects the archived `vcut_sp`
spelling to `vcut_sph`, replaces removed NumPy scalar aliases, and avoids the
removed private `_cut_mesh_for_ewald` helper using explicit reciprocal grids.

Current PySCF's fast momentum-conservation lookup assumes canonical
`make_kpts` ordering. Concatenated staggered meshes can violate that assumption
even when their union is a uniform grid. MP3 and CCD therefore build their
momentum and integral-symmetry maps from the actual fractional coordinates.
Overlapping submeshes are rejected.

The archived low-memory integral paths also depend on obsolete PySCF internals.
The port raises an explicit error when dense integral storage exceeds the
memory budget; it does not provide a new out-of-core implementation.

For a standard PySCF GW comparison following PBE, build complete density-fitting
integrals with `df.GDF(cell, kpts).build(j_only=False)`. A Coulomb-only PBE
integral file lacks the off-diagonal k-point pairs required by GW.

## Numerical comparisons

### Ordinary KMP3

The archived `MBPT_stagger/kmp3.py` and the newer private PySCF `KRMP3` were
run on the same converged H2 reference:

- Lattice: 6-bohr cube; H positions `(3, 3, 2.1)` and `(3, 3, 3.9)` bohr.
- GTH-SZV basis, GTH-PADE pseudopotential, GDF, `ke_cutoff=100` hartree.
- `cell.precision=1e-9`, k mesh `(1, 1, 3)`.
- KRHF `conv_tol=1e-12`, `conv_tol_grad=1e-9`, one numerical thread.

| Reference / archive correction | Archive E2 + E3 | Newer PySCF E2 + E3 | Absolute difference (Ha) |
| --- | ---: | ---: | ---: |
| `exxdiv=None`, `madelung=False` | -0.028615637505912 | -0.028615637505924 | 1.24e-14 |
| `exxdiv='ewald'`, `madelung=False` | -0.023499032694120 | -0.026472224964795 | 2.973192270674e-3 |
| `exxdiv='ewald'`, `madelung=True` | -0.026472224964794 | -0.026472224964795 | 1.10e-15 |

For `exxdiv=None`, the newer implementation gives
E2 = -0.022909492354553 and E3 = -0.005706145151372 Ha. For Ewald it gives
E2 = -0.019393277812456 and E3 = -0.007078947152339 Ha. Thus the energies
agree when the correction conventions match; the archived default alone is
not an equivalent comparison with an Ewald reference.

After importing the code, a further live comparison used the same H2 setup
with GTH-DZVP, multiple virtual orbitals, complex Bloch coefficients, and
`exxdiv=None`:

| Component (Ha) | FSEC KMP3 | Newer PySCF KRMP3 |
| --- | ---: | ---: |
| E2 | -0.03664781803901504 | -0.03664781803901505 |
| E3 increment | -0.00436112722740852 | -0.00436112722734623 |
| E2 + E3 | -0.04100894526642356 | -0.04100894526636128 |

The largest difference is 6.23e-14 Ha. The tests also check that orbital-phase
changes leave the energy invariant and transform the returned first-order
amplitudes consistently.

### Three-mesh CCD: inherited reference discrepancy

The port reproduces the archived symmetry-reduced CCD solver's converged
energy. The archive's separate `nosymm` implementation gives a slightly
different converged result on the small H2 test:

| Solver | Correlation energy (Ha) |
| --- | ---: |
| FSEC three-mesh CCD | -0.015443694230431754 |
| Archived symmetry-reduced CCD | -0.015443694230431754 |
| Archived `nosymm` CCD | -0.015441771700058228 |

This test uses the 6-bohr H2 cell, GTH-SZV/GTH-PADE, a `(3, 1, 1)` reference
mesh, submesh mode, and no CCD Madelung correction. Its deliberately small
real-space grid is `(11, 11, 11)`, with `cell.precision=1e-8`.
The integral tensors and MP2 starting energies agree; the converged CCD
energies differ by 1.92e-6 Ha. The imported equations retain the archived
symmetry-reduced behavior. Equality with `nosymm` is recorded as an explicit
expected failure in the optional archive tests, rather than relaxing the
comparison tolerance. The `nosymm` equations include an additional averaged
contraction route for mesh configurations `(x, y, y, x)` that is absent from
the symmetry-reduced equations. Choosing between those archived formulations
requires further scientific validation; this import does not change them.

### Historical H2 GW reference: not reproduced

[result_staggerGW_Nov9.pdf](result_staggerGW_Nov9.pdf) contains plotted results
rather than tabulated reference energies. A bounded test used the H2 geometry
above and the PDF's GTH-SZV, 100-hartree cutoff, 100 frequency points,
1-hartree imaginary-frequency cutoff, and linearized G0W0. Additional inputs
were taken from the archive: GTH-PADE, PBE KRKS, a Gamma-centered `(3, 3, 3)`
mesh, `cell.precision=1e-10`, and SCF `conv_tol=1e-12`.

Current **standard** PySCF GW with `qpe_linearized_range=None` gave:

| Standard GW option | Gamma VBM (Ha) | Gamma CBM (Ha) | Gamma gap (Ha) |
| --- | ---: | ---: | ---: |
| `fc=False` | -0.410813273049 | 0.466478547047 | 0.877291820096 |
| `fc=True` | -0.541688998414 | 0.466477458096 | 1.008166456510 |

The PDF's standard uncorrected gap at that mesh is approximately 0.861 Ha,
read from page 3. The 0.877292 Ha result does not reproduce it. The reference
SCF converged; its Gamma orbital energies were -0.365926163353 and
0.213995383776 Ha. The only retry changed the input to build complete GDF
integrals after the initial Coulomb-only file caused a missing-pair error.

The archived staggered GW also failed its initial import because `mpi4py` was
absent. This historical reproduction attempt was stopped as requested.
The import's small numerical tests validate the port; they do not establish
agreement with the PDF. No convergence sweep or tuning to the plotted values
is included.

## Environment, tests, and provenance

The tested local environment is `fsec-312` (Python 3.12, PySCF 2.14.0). The
new GW continuation interface must be available. Other PySCF versions have
not been validated by this import.

From the repository root, after activating the environment:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m pytest fsec/staggered_mesh
python -m pytest -m slow fsec/staggered_mesh
```

Slow calculations are excluded by default; an explicit `-m slow` selects
them. In this workspace the compiled PySCF libraries also require
`LD_LIBRARY_PATH=/home/sjquiton/software/anaconda3/envs/fsec-312/lib` when
invoking the environment's Python directly.

The newer KMP3 comparison reference is supplied by the local private extension:

```bash
export PYSCF_EXT_PATH=/home/sjquiton/sjq_dev/sjq-private-repos/pyscf-forge-priv
python -m pytest -m slow fsec/staggered_mesh/mp
```

This extension is used only for comparison, not by the installed solvers.
CCD reference comparisons are opt-in with
`FSEC_CCD_REFERENCE_DIR=/path/to/CC_stagger`. This explicitly supplies the
comparison files; the tests do not assume any neighboring checkout, and
ordinary package imports do not need them.

With both comparison references enabled, the combined staggered-mesh suite
completed with **14 passed and 1 expected failure**. The expected failure is
the three-mesh CCD `nosymm` discrepancy described above. The final multi-target
two-pole GW test and the configurable CCD reference loader were also checked
individually after their last edits.

The GW tests compare the imaginary-frequency self-energy with explicit
contractions and exercise nonlinear Padé and linearized two-pole QP equations,
selected-state indexing, and exchange-correction toggles. Separate tiny-cell
checks also completed with both finite-size corrections enabled. These checks
test the port's equations and interfaces; the historical PDF comparison above
remains unsuccessful.

| Source | Revision inspected | Role |
| --- | --- | --- |
| `../staggered_mesh_methods` | `bf4c9d0` | Xin Xing's staggered MP3, CCD, and GW implementations |
| `../pyscf-sjq` | `ecc97b19d` | PySCF 2.14.0 used by `fsec-312` |
| `../sjq-private-repos/pyscf-forge-priv` | `1eca2c5` | Newer ordinary KMP3 comparison |

The accompanying method notes are [staggered MP3](Staggered_MP3_Sep15.pdf),
[staggered GW](Staggered_GW_Sep26.pdf), and
[three-mesh CCD](staggered_ccd_3mesh.pdf). Original author and license notices
are retained in the imported source files.
