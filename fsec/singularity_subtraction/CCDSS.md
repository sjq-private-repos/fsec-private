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
constraint switches. Both switches default to `True` and are required by the
current aggregate implementation. Passing either switch as `False` to
`KRCCD_SS` raises `NotImplementedError`; the fields are retained for a future
relaxed contraction implementation. An options object can be passed instead of
individual overrides, but the two forms may not be mixed ambiguously.

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
   normalized transition pair densities. The origin is normalized to exactly
   one.
3. `_prepare_ss` evaluates the six contracted practical structure factors
   defined below, normalizes the resulting complex curves by their common
   origin, fits a unit-coefficient isotropic Gaussian to each, and evaluates
   its analytic-integral-minus-finite-quadrature correction `xi_n`.
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
\sum_{k_i k_j k_a}\sum_{ijab},
\qquad k_b = k_i-k_a+k_j
\]

with all momenta understood modulo the sampled mesh. The six practical,
contracted structure factors are evaluated explicitly as

\[
\begin{aligned}
\widetilde S_1(Q) &= \sum_{\mathcal I}
 F_i(k_i-q,Q)F_j^*(k_j,Q)t_{ij}^{ab}(k_i,k_j,k_a), \\
\widetilde S_2(Q) &= \sum_{\mathcal I}
 F_a(k_a,Q)F_b^*(k_b-q,Q)t_{ij}^{ab}(k_i,k_j,k_a), \\
\widetilde S_3(Q) &= \sum_{\mathcal I}
 F_a(k_a,Q)F_i^*(k_i,Q)t_{ij}^{ab}(k_i,k_j,k_a), \\
\widetilde S_4(Q) &= \sum_{\mathcal I}
 F_b(k_b,Q)F_j^*(k_j,Q)t_{ji}^{ba}(k_j,k_i,k_b), \\
\widetilde S_5(Q) &= \sum_{\mathcal I}
 F_a(k_a,Q)F_j^*(k_j,Q)t_{ji}^{ba}(k_j,k_i,k_b), \\
\widetilde S_6(Q) &= \sum_{\mathcal I}
 F_b(k_b,Q)F_i^*(k_i,Q)t_{ij}^{ab}(k_i,k_j,k_a).
\end{aligned}
\]

Thus each sample is one number per channel and `Q`, stored in an array with
shape `(6, n_samples)`. This is the distinction meant by *contracted*: every
orbital, band, and external k-point label has been summed. The inactive
uncontracted helpers instead leave `(ki,kj,ka,i,j,a,b)` open and produce one
structure-factor curve for every T2 entry.

At the origin, `F_n(k,0) = 1`, so all six equations reduce to

\[
\widetilde S_n(0) = \sum_{\mathcal I}t_{ij}^{ab}(k_i,k_j,k_a)
                   = \operatorname{sum}(T2).
\]

The curves passed to the fitter are therefore

\[
\widehat S_n(Q) =
\frac{\widetilde S_n(Q)}{\widetilde S_n(0)},
\qquad \widehat S_n(0)=1.
\]

All momentum translations use the cached discrete q maps. Channels L4 and L5
read the pair-transposed PySCF entry `t2[kj,ki,kb,j,i,b,a]` directly rather
than allocating a transposed T2 tensor.

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

Constraint (1) retains only the diagonal orbital pair factors that contribute
at the singular point. Constraint (2) evaluates the amplitudes at the
unshifted external indices in the contracted practical equations above. Both
are required for the current aggregate path; either disabled option is
rejected clearly before a calculation starts.

The uncontracted per-entry builders remain private for method comparisons, but
no public option dispatches to them.

In fitted mode, `_prepare_ss` is called for every CCD amplitude update. It
rebuilds the six aggregate curves and performs exactly six independent fits,
so `ss_prepare_count` increases by one and `ss_fit_count` by six per update.
`init_amps` does not perform an extra fitted preparation. The common
normalization value is the complex `sum(t2)`. If its magnitude is below
`amplitude_fit_tol`, preparation stops with a clear error instead of dividing
by a small amplitude. The normalized origin is assigned exactly `1` for all
six channels.

The most recent fitted state is stored in `ss_sigmas` and `ss_xi`, each with
shape `(6,)`. Fixed-sigma modes are amplitude-independent: zero, infinity,
and finite fixed widths prepare lazily once, do not perform numerical fits,
and retain the same six-element state. The signed correction coefficient is
`xi1 + xi2 - xi3 - xi4 - xi5 - xi6`; that scalar multiplies the current T2
residual before denominator division.

The counters `ss_prepare_count` and `ss_fit_count` expose how often these
operations occurred. The most recent fitted widths and corrections are
available as `ss_sigmas` and `ss_xi`.
At `verbose >= logger.INFO`, each preparation also reports its CPU and wall
time.

Set `solver.verbose = logger.DEBUG2` to emit one `CCDSS_SF_NORM` text row for
each channel and preparation before the sample rows. It reports the raw
complex `tilde-S_n(0)` used to normalize that channel as `norm_real`,
`norm_imag`, and `norm_abs`; in the current approximation, all six values are
the common `sum(T2)`. The six rows retain independent channel labels for
future relaxed contractions. The same setting emits one `CCDSS_SF` text row
for each channel, zero-based `q_index`, and preparation. These rows contain
the Cartesian `qx`, `qy`, and `qz` components of the sampled q vector (in the
same reciprocal-vector units as `cell.reciprocal_vectors()`), the complex
normalized structure-factor sample as `raw_real`/`raw_imag`, its fitted
`sigma` and `xi`, the fitted Gaussian value `fit`, and
`residual = fit - raw` as `residual_real`/`residual_imag`. The L3-L6 signs are
applied only when the channel corrections are combined. Rows are emitted in
channel and q-sample order and carry the one-based preparation number in
`prep`. Fitted mode emits a new six-curve set on every amplitude update.
Fixed-sigma preparations do not construct samples and emit no rows.

## Fixed-sigma limits

`fixed_sigma` bypasses fitting while retaining the six-channel machinery:

- `fixed_sigma=0` gives no ERI singularity-subtraction residual. The result is
  orbital-corrected CCD.
- A large finite width approaches the exact ERI Madelung correction.
- `fixed_sigma=np.inf` uses the exact Madelung residual limit directly.
- `fixed_sigma=None` performs the six Gaussian fits described above on every
  amplitude update.

The H2 example in `examples/h2_ccdss.py` demonstrates the fitted calculation
and both exact limiting cases. `examples/h2_ccdss_structure_factors.py` runs
the same 1x1x2 GTH-SZV system at `logger.DEBUG2` and prints the
`CCDSS_SF_NORM` and `CCDSS_SF` rows from the fitted calculation.
