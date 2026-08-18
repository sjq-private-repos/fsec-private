# CCD singularity subtraction

`KRCCD_SS` implements iterative singularity subtraction for the six
Coulomb-singular ERI contractions in restricted periodic coupled-cluster
doubles. It builds on `fsec.staggered_mesh.cc.KRCCD`, which in turn subclasses
PySCF's periodic restricted `kccsd_rhf.RCCSD` implementation.

## Public interface

```python
from fsec.singularity_subtraction import CCDSSOptions, KRCCD_SS

cc = KRCCD_SS(
    kmf,
    frozen=None,
    mo_coeff=None,
    mo_occ=None,
    options=None,
    use_constraint_2=True,
    line_points=3,
    pair_density_becke_grid_level=0,
    fit_with_coul=True,
    fixed_sigma=None,
    amplitude_fit_tol=1e-12,
    occupied_orbital_shift=None,
)
e_corr, t1, t2 = cc.kernel()
```

The mean-field object must satisfy the same requirements as `KRCCD`: a
closed-shell, three-dimensional `KRHF` calculation on a uniform k-point mesh.
Set `kmf.exxdiv = None` and `cc.keep_exxdiv = False` when using the routine's
explicit occupied-orbital correction.

`occupied_orbital_shift` optionally replaces that correction. It accepts a
finite real scalar or an array with shape `(nkpts, nocc)`, in Hartree, and is
added directly to the active occupied Fock diagonals and orbital energies. A
scalar is broadcast to every occupied state. Array columns use PySCF's
correlated, padded occupied-orbital layout; values corresponding to padded
states are ignored. Virtual energies and off-diagonal Fock elements are not
changed.

When a custom shift is present, `KRCCD_SS` builds the uncorrected one-body
baseline regardless of `keep_exxdiv`: it bypasses the explicit Madelung shift
when `keep_exxdiv=False` and replaces the retained SCF exchange-divergence
correction when `keep_exxdiv=True`. The public value of `keep_exxdiv` is not
modified.

The current `ExxSS` implementation produces one uniform correction per
occupied orbital. Its sign convention can be passed to CCDSS as follows:

```python
from fsec.singularity_subtraction import ExxSS, KRCCD_SS

exxss = ExxSS(kmf)
exxss.compute_correction()
cc = KRCCD_SS(kmf, occupied_orbital_shift=-exxss.chi)
```

`CCDSSOptions` contains the same six method-specific controls. An options
object can be passed instead of individual overrides, but the two forms may
not be mixed ambiguously.

## Amplitude and momentum layout

PySCF stores doubles as

```text
t2[ki, kj, ka, i, j, a, b]
```

with `kb = kconserv[ki, ka, kj]`. Occupied indices are `i,j`; virtual indices
are `a,b`. Frozen-orbital masks and padded bands follow PySCF's KMP2/KCCSD
conventions. Singles are replaced by zero during initialization, every
amplitude update, and energy evaluation, so the iterative equations remain
CCD rather than CCSD.

## Runtime structure

The calculation has four main stages.

1. `KRCCD_SS.__init__` validates the options and initializes `KRCCD` with the
   occupied-orbital Madelung shift enabled unless a custom occupied-orbital
   shift was supplied. The exact ERI Madelung correction is disabled because
   it is replaced by the fitted singularity-subtraction residual.
2. `_build_pair_factors` creates a periodic Becke grid, evaluates the active
   padded molecular orbitals, and precomputes normalized transition pair
   densities. The origin is normalized to exactly one.
3. `_prepare_ss` constructs the six normalized structure-factor channels,
   fits a unit-coefficient isotropic Gaussian to each, and evaluates its
   analytic-integral-minus-finite-quadrature correction `xi_n`.
4. `_inject_ss_residual` combines the channels as

   ```text
   xi1 + xi2 - xi3 - xi4 - xi5 - xi6
   ```

   multiplies this coefficient by the current T2 amplitude, and injects the
   result into the doubles numerator before the orbital-energy denominator is
   applied.

The six unconstrained amplitude samples are:

| Channel | Sampled amplitude |
| --- | --- |
| 1 | `t2[ki-q, kj+q, ka]` |
| 2 | `t2[ki, kj, ka+q]` |
| 3 | `t2[ki+q, kj, ka+q]` |
| 4 | the pair-transposed `t2[kj+q, ki, kb+q]` |
| 5 | the pair-transposed `t2[kj+q, ki, kb]` |
| 6 | `t2[ki+q, kj, ka]` |

All additions and subtractions of momenta use the discrete mesh maps; the
fourth and fifth channels also transpose `(j,i,b,a)` back to `(i,j,a,b)`.

## Sampling and Gaussian fitting

The sample grid contains the origin and `line_points` positive samples along
each reciprocal-lattice direction. Along direction `i`, adjacent samples are
separated by

```text
b_i / n_i,
```

where `n_i` is the Monkhorst-Pack mesh size. The default therefore contains
the origin plus nine nonzero samples.

Each normalized channel is fitted to

```text
h(Q) = exp(-|Q|^2 / (2 sigma^2)).
```

The coefficient is fixed to one, enforcing the exact origin normalization.
With `fit_with_coul=True`, the least-squares residual is weighted by
`1/|Q|^2`. The resulting channel correction is the analytic Coulomb integral
of `h` minus its finite `(q+G)` quadrature.

## Constraints and update timing

Constraint (1), which retains only orbital indices contributing at the
singular point, is always active.

With the default `use_constraint_2=True`, the amplitude is fixed at its
`q=0` value when forming each normalized structure factor. The pair densities,
Gaussian widths, and `xi_n` tensors are prepared once from the initial T2 and
reused throughout CCD iteration.

With `use_constraint_2=False`, amplitudes are sampled at their shifted
momenta. The six Gaussian fits are consequently rebuilt from the current T2
on every amplitude update. Entries whose external amplitude is smaller than
`amplitude_fit_tol` fall back to a unit amplitude ratio to avoid unstable
division.

The counters `ss_prepare_count` and `ss_fit_count` expose how often these
operations occurred. The most recent fitted widths and corrections are
available as `ss_sigmas` and `ss_xi`.

## Fixed-sigma limits

`fixed_sigma` bypasses fitting while retaining the six-channel machinery:

- `fixed_sigma=0` gives no ERI singularity-subtraction residual. The result is
  orbital-corrected CCD.
- A large finite width approaches the exact ERI Madelung correction.
- `fixed_sigma=np.inf` uses the exact Madelung residual limit directly.
- `fixed_sigma=None` performs the Gaussian fits described above.

The H2 example in `examples/h2_ccdss.py` demonstrates the fitted calculation
and both exact limiting cases.
