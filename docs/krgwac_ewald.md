# Scalar Ewald corrections for periodic G0W0-AC

`fsec.standard.krgwac_ewald.KRGWAC` subclasses the PySCF 2.14 `KRGWAC`
implementation and adds a scalar Ewald correction to exchange and the
correlation self-energy. The ordinary density-fitted correlation self-energy,
analytic continuation, and quasiparticle solvers remain PySCF's routines. The
Ewald correction is added to the imaginary-axis self-energy before
continuation.

```python
from fsec.standard import krgwac_ewald

gw = krgwac_ewald.KRGWAC(mf, frozen=None)
gw.fc_method = 'ewald'       # default; use 'spherical' for PySCF's correction
gw.kernel(orbs=[0, 1], kptlist=[0])
```

Set `gw.fc = False` to disable both exchange and correlation finite-size
corrections. `fc_method='spherical'` delegates directly to upstream PySCF,
including its existing `fc_grid` treatment. Ewald mode also preserves those
`fc_grid` directions when it averages the long-wavelength dielectric head.

## Correction and conventions

The complete uniform Monkhorst-Pack mesh is passed once to
`pyscf.pbc.tools.madelung(cell, kpts=gw.kpts, omega=0)`. Its scalar result
`chi` has units of inverse Bohr. Ewald exchange adds `-chi` to the diagonal
matrix elements of active occupied orbitals. The exchange matrix itself is
computed from the uncorrected upstream DF exchange routine, so the SCF object
is not changed and no spherical exchange term is combined with `-chi`.

For a cubic integration supercell of side `L` Bohr,
`chi = 2.837297479480619 / L`, so this shifts occupied exchange downwards.
The full supercell geometry already contains the mesh normalization: there
is no extra factor of `1/nkpts` in either Ewald correction. Virtual exchange
and all off-diagonal exchange entries are unchanged.

For each imaginary-axis self-energy sample `z`, the diagonal correlation
correction is

\[
\Delta\Sigma_c^{nk}(z) = -\frac{\chi}{\pi}
\sum_l w_l\,[h_l-1]\,
\frac{z-\epsilon_{nk}}
     {(z-\epsilon_{nk})^2+\nu_l^2}.
\]

Here `nu_l` and `w_l` are PySCF's scaled Gauss-Legendre quadrature points and
weights. `h_l` is the average of the inverse dielectric head at `i nu_l` over
the directions selected by `fc_grid`. Each direction uses PySCF's
long-wavelength head and wing response;
the head is inverted with the Schur complement of the dielectric body, so
dielectric-wing coupling is included. The scalar direction average is a
screening approximation: it does not retain the angular dependence of the
screened Coulomb singularity in the Ewald sum.

`fc_grid=False` samples the first reciprocal-lattice direction;
`fc_grid=True` averages PySCF's 26 positive-octant directions. These are
long-wavelength k.p responses, not finite-q extrapolations. The 26 transition
densities are constructed from three basis directions using their exact
linearity in q. This scalar treatment is not a tensor-screened Ewald sum and
may leave substantial errors in anisotropic systems. When `h_l=1` the
correlation correction is zero. For positive `chi`, screening below unity,
and a real `z` inside the gap, the occupied correlation correction is
positive and the virtual correction is negative.

The leading self-energy wing has the form
`A(q) = exp(-alpha*q**2) * (b dot q) / q**2`, with reciprocal-lattice
periodization. It is odd under `q -> -q`. The supported complete uniform
meshes have symmetric transfer meshes, so its integral and paired discrete
sum vanish; Ewald mode therefore adds no separate leading-wing term and does
not reuse PySCF's spherical wing correction. Higher-order wing contributions
are left uncorrected.

Explicitly, periodization means summing `A(q+G)` over reciprocal-lattice
vectors `G`. The odd terms cancel on pairs of transfer points, including for
uniformly shifted orbital meshes. This cancellation concerns the separate
self-energy wing; it does not remove the dielectric wings in the Schur
complement. The remaining regular wing contribution discussed by Freysoldt
et al. scales with the omitted reciprocal-space cell volume and must still
be checked by mesh convergence.

## Supported scope

Ewald mode supports 3D restricted, gapped systems with integer occupations,
a complete uniform Monkhorst-Pack mesh (including a uniform shift), and GDF
integrals, including RSDF, with the full Coulomb interaction. Mixed DF,
smearing, spinor/unrestricted references, and symmetry-reduced meshes are
outside this scope. `kptlist` can select any output subset; it does not
change `chi`, which is evaluated from the full mean-field mesh. Corrected
out-of-core calculations and unsupported Ewald setups raise an explicit
exception. The uncorrected and spherical paths keep upstream PySCF behavior.

Ewald mode builds any missing exchange DF blocks in a private cache without
changing the mean-field DF object. Noncontiguous output bands are evaluated
through their containing contiguous range and selected afterwards to avoid
PySCF 2.14's advanced-indexing limitations. Frozen orbital labels and output
k-point labels otherwise follow upstream conventions. With `fullsigma=True`
only correlation diagonals receive the correction; stored `sigmaI` (when
`rdm=True`) and `writefile` output contain the corrected samples.

## Running the comparison and checks

Run `python examples/krgwac_ewald_comparison.py` for all three modes on one
small periodic H2 reference. The bond has a component along the default
dielectric sampling direction, so the head correction is nonzero. The
example explicitly builds full DF blocks because upstream spherical and
uncorrected GW require them even after a pure-DFT SCF calculation.

The focused tests are in `fsec/standard/tests/test_krgwac_ewald.py`.
Periodic SCF/GW tests use `-m slow`. The additional mesh comparison is opt-in:

```bash
FSEC_RUN_KRGWAC_EWALD_MESH_STUDY=1 python -m pytest -s -m slow \
    fsec/standard/tests/test_krgwac_ewald.py -k mesh_study
```

It prints the residual difference between `1x1x1` and `2x1x1` QP gaps for
all three modes. This small anisotropic mesh study is a diagnostic, not a
claim of bulk convergence or universally monotonic improvement. For the
local `fsec-312` environment, include its `lib` directory in
`LD_LIBRARY_PATH`; one BLAS/OpenMP thread is sufficient for these small cells.

The opt-in study with PySCF 2.14.0, PBE/GTH-SZV, 16 integration points,
six Pade points, and linearized QP energies gave:

| Mode | Gap(1x1x1) minus gap(2x1x1), Ha |
| --- | ---: |
| Ewald | 0.06895421 |
| Spherical | 0.00834521 |
| Uncorrected | -0.10960141 |

The two meshes remain far from a bulk convergence study. In particular,
Ewald is not the smallest residual in this example; scalar screening and
regular wing/body errors remain relevant.

## References

- C.-N. Yeh et al., *Phys. Rev. B* **106**, 085121 (2022), Eqs. 49–50,
  [accepted manuscript](https://link.aps.org/accepted/10.1103/PhysRevB.106.085121).
- C. Freysoldt et al., “Dielectric anisotropy in the GW space-time method,”
  Sec. 3.3, [article PDF](https://juser.fz-juelich.de/record/55618/files/aniso-gwst.pdf?version=1).

The first reference motivates the Madelung head correction to dynamic GW and
the static exchange correction. The second discusses why the leading wing is
odd and its isotropic average vanishes.
