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
    use_constraint_1=True,
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

`CCDSSOptions` contains the method-specific controls, including both
constraint switches. Both switches default to `True`. Set
`use_constraint_1=False` to sum over full occupied and virtual transition-
density matrices. T2 is factored out of the fitted curves and only the current
T2 multiplies the cached residual. Constraint (2) relaxation
(`use_constraint_2=False`) remains unsupported and raises `NotImplementedError`.
An options object can be passed instead of individual overrides, but the two
forms may not be mixed ambiguously.

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
2. `_build_pair_factors` or `_build_pair_densities` creates a periodic Becke
   grid, evaluates the active padded molecular orbitals, and precomputes
   normalized transition pair densities. Constraint (1) selects diagonal
   factors or full occupied/virtual transition-density matrices. Per-k
   occupied and virtual masks from `padding_k_idx(kind="split")` remove padded
   external and internal states from the aggregate.
3. `_prepare_ss` evaluates six scalar density-only structure factors, normalizes
   them by the exact physical q=0 entry count, fits a unit-coefficient
   isotropic Gaussian to each, and evaluates its
   analytic-integral-minus-finite-quadrature correction `xi_n`.
4. `_inject_ss_residual` combines the channels as

   ```text
   xi1 + xi2 - xi3 - xi4 - xi5 - xi6
   ```

   multiplies this coefficient by the current T2 amplitude, and injects the
   result into the doubles numerator before the orbital-energy denominator is
   applied.

## Contracted structure factors

Let `Q = q + G`. The diagonal transition factor linking the same orbital at
`k` and `k+q` is

\[
F_n(k,Q) =
\frac{\int_\Omega u^*_{n,k}(r)u_{n,k+q}(r)e^{-iG\cdot r}\,dr}
{\sqrt{\int_\Omega |u_{n,k}(r)|^2dr
             \int_\Omega |u_{n,k+q}(r)|^2dr}}.
\]

The implementation handles reciprocal wrapping when `k+q` leaves the first
Brillouin zone. Define

\[
\sum_{\mathcal I} =
\sum_{k_i k_j k_a}\sum_{ijab}^{\rm active},
\qquad k_b = k_i-k_a+k_j
\]

with all momenta understood modulo the sampled mesh. The six practical,
contracted structure factors are evaluated explicitly as

\[
\begin{aligned}
\widetilde S_1(Q) &= \sum_{\mathcal I}
 F_i(k_i-q,Q)F_j^*(k_j,Q), \\
\widetilde S_2(Q) &= \sum_{\mathcal I}
 F_a(k_a,Q)F_b^*(k_b-q,Q), \\
\widetilde S_3(Q) &= \sum_{\mathcal I}
 F_a(k_a,Q)F_i^*(k_i,Q), \\
\widetilde S_4(Q) &= \sum_{\mathcal I}
 F_b(k_b,Q)F_j^*(k_j,Q), \\
\widetilde S_5(Q) &= \sum_{\mathcal I}
 F_a(k_a,Q)F_j^*(k_j,Q), \\
\widetilde S_6(Q) &= \sum_{\mathcal I}
 F_b(k_b,Q)F_i^*(k_i,Q).
\end{aligned}
\]

Thus each sample is one number per channel and `Q`, stored in an array with
shape `(6, n_samples)`. This is the distinction meant by *contracted*: every
orbital, band, and external k-point label has been summed, with no T2 weight.

When constraint (1) is disabled, the diagonal factors are replaced by full
normalized transition-density matrices. For each external momentum block, the
internal orbital sums are the following scalar contractions (the second
density is conjugated in every channel):

```text
L1: ki,jl -> ij
L2: ac,db -> ab
L3: ac,ik -> ia
L4: bc,jk -> jb
L5: ac,jk -> ja
L6: bc,ik -> ib
```

The density momentum locations remain those in the diagonal equations above:
L1 uses `ki-q` and `kj`, L2 uses `ka` and `kb-q`, L3 uses `ka` and `ki`,
L4 uses `kb` and `kj`, L5 uses `ka` and `kj`, and L6 uses `kb` and `ki`.
For a block `(ki,kj,ka)`, the full-density partials are respectively
`ki,jl->ij`, `ac,db->ab`, `ac,ik->ia`, `bc,jk->jb`, `ac,jk->ja`, and
`bc,ik->ib`. Internal density indices are masked at their actual source and
target momenta as well as the external indices. The scalar contractions are
accumulated directly with `pyscf.lib.einsum`.

At the origin, `F_n(k,0) = 1`, so all six equations reduce to the exact number
of physical active entries

\[
\widetilde S_n(0) = N_{\rm active}
 = \sum_{k_i k_j k_a}n_i n_j n_a n_b,
\qquad k_b = k_i-k_a+k_j.
\]

The curves passed to the fitter are therefore

\[
\widehat S_n(Q) =
\frac{\widetilde S_n(Q)}{N_{\rm active}},
\qquad \widehat S_n(0)=1.
\]

All momentum translations use the cached discrete q maps. T2 is absent from
the fitted curves, including channels L4 and L5; no T2 transpose is formed.

## Sampling and Gaussian fitting

The sample grid contains the origin and `line_points` positive samples along
each reciprocal-lattice direction. Along direction `i`, adjacent samples are
separated by

```text
b_i / n_i,
```

where `n_i` is the Monkhorst-Pack mesh size. The default therefore contains
the origin plus nine nonzero samples.

Each normalized aggregate channel is fitted independently to

```text
h(Q) = exp(-|Q|^2 / (2 sigma^2)).
```

The coefficient is fixed to one, enforcing the exact origin normalization.
With `fit_with_coul=True`, the least-squares residual is weighted by
`1/|Q|^2`. The resulting channel correction is the analytic Coulomb integral
of `h` minus its finite `(q+G)` quadrature.

## Constraints and update timing

Constraint (1) retains only diagonal orbital pair factors when enabled. When it
is disabled, all internal occupied and virtual orbital pairs are summed using
the full transition-density matrices. Constraint (2) factors the amplitude
momentum at q=0; its relaxation is unsupported and is rejected before a
calculation starts.

Fitted mode prepares the six scalar density-only curves during `init_amps` and
performs exactly six independent fits. The resulting widths and corrections
are reused for every CCD amplitude update; a direct update path that bypasses
`init_amps` prepares them lazily on its first update. Thus a normal calculation
has `ss_prepare_count == 1` and `ss_fit_count == 6`. The normalization value is
the exact physical active-entry count, independent of T2, and every normalized
origin is assigned exactly `1`. `amplitude_fit_tol` remains accepted and
validated for compatibility, but is unused by the supported constraint modes.

The most recent fitted state is stored in `ss_sigmas` and `ss_xi`, each with
shape `(6,)`. Fixed-sigma modes are amplitude-independent: zero, infinity,
and finite fixed widths prepare once (at initialization or lazily), do not
perform numerical fits, and retain the same six-element state. The signed correction coefficient is
`xi1 + xi2 - xi3 - xi4 - xi5 - xi6`; that scalar multiplies the current T2
residual before denominator division.

The counters `ss_prepare_count` and `ss_fit_count` expose how often these
operations occurred. The most recent fitted widths and corrections are
available as `ss_sigmas` and `ss_xi`.
At `verbose >= logger.INFO`, each preparation also reports its CPU and wall
time. Both constraint-(1) modes use the same one-time preparation policy.

Set `solver.verbose = logger.DEBUG2` to emit one `CCDSS_SF_NORM` text row for
each channel and preparation before the sample rows. It reports the raw
complex `tilde-S_n(0)` used to normalize that channel as `norm_real`,
`norm_imag`, and `norm_abs`; all six values are the exact physical active-entry
count. The six rows retain independent channel labels for both
constraint-1 modes. The same setting emits one `CCDSS_SF` text row
for each channel, zero-based `q_index`, and preparation. These rows contain
the Cartesian `qx`, `qy`, and `qz` components of the sampled q vector (in the
same reciprocal-vector units as `cell.reciprocal_vectors()`), the complex
normalized structure-factor sample as `raw_real`/`raw_imag`, its fitted
`sigma` and `xi`, the fitted Gaussian value `fit`, and
`residual = fit - raw` as `residual_real`/`residual_imag`. The L3-L6 signs are
applied only when the channel corrections are combined. Rows are emitted in
channel and q-sample order and carry the one-based preparation number in
`prep`. The six-curve set is emitted once for a normal initialized calculation.
Fixed-sigma preparations do not construct samples and emit no rows.

## Fixed-sigma limits

`fixed_sigma` bypasses fitting while retaining the six-channel machinery:

- `fixed_sigma=0` gives no ERI singularity-subtraction residual. The result is
  orbital-corrected CCD.
- A large finite width approaches the exact ERI Madelung correction.
- `fixed_sigma=np.inf` uses the exact Madelung residual limit directly.
- `fixed_sigma=None` performs the six Gaussian fits described above once during
  amplitude initialization.

The H2 example in `examples/h2_ccdss.py` demonstrates the fitted calculation
and both exact limiting cases. `examples/h2_ccdss_structure_factors.py` runs
the same 1x1x2 GTH-SZV system at `logger.DEBUG2` and prints the
`CCDSS_SF_NORM` and `CCDSS_SF` rows from the fitted calculation.
