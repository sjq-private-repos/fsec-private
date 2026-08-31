# Review of the CCD singularity-subtraction implementation

> Status update: The issue identified in this historical review has been
> resolved by the approved density-only implementation. Both constraint-(1)
> modes now build six scalar curves independent of T2, normalize by the exact
> physical q=0 active-entry count, fit six `xi` values once during amplitude
> initialization, and reuse them for later residual updates. See `CCDSS.md`
> and `CCD_SS_CONSTRAINT_1_PLAN.md` for the current behavior; the analysis
> below describes the superseded implementation.

The six-channel signs, momentum shifts, pair transposes, and denominator
injection in `ccdss.py` appear consistent with the explicit uncontracted
structure-factor definitions discussed below. I did not find an obvious
global-sign or momentum-conservation error. I did find one definite correctness
problem and several strong reasons why the current strategy can underperform
ordinary converged CCD.

## Highest-confidence implementation issue

### Independent relaxation of constraint (1) is incorrect

The combination

```python
use_constraint_1=False
use_constraint_2=True
```

does not correctly represent constraint (1) being relaxed while constraint (2)
is retained.

The current `_density_only_sums` implementation forms expressions such as

```text
sum_kl rho_ki rho_jl*
```

and `_density_only_channel_samples` then implicitly treats the external
amplitude as a common factor. For L1, however, relaxing constraint (1) while
imposing constraint (2) should give, schematically,

\[
\frac{
  \sum_{kl}
  \rho_{ki}(q)\rho_{jl}^{*}(q)
  t_{kl}^{ab}(k_i,k_j,k_a)
}{
  t_{ij}^{ab}(k_i,k_j,k_a)
}.
\]

Constraint (2) sets the momentum dependence of the amplitude to its value at
`q=0`; it does not replace every internal-orbital amplitude
\(t_{kl}^{ab}\) by the external \(t_{ij}^{ab}\). Analogous unshifted but
internal-index amplitudes are required for L2 through L6.

This erroneous interpretation is also stated in
`CCD_SS_CONSTRAINT_1_PLAN.md` and is explicitly encoded in
`test_density_only_sums_match_nested_loop_references_and_ignore_t2`, which
requires the samples to be independent of T2.

Consequences include:

- This option combination is a new uncontrolled approximation rather than a
  relaxation toward the complete structure factor.
- It can naturally become worse when constraint (1) is supposedly relaxed.
- Once the internal T2 amplitudes are retained, the fitted correction generally
  cannot remain iteration-independent. It could be precomputed only by
  explicitly freezing those amplitudes, for example at MP2, which would be a
  separate approximation.
- Padded orbital diagonals are set to one in the transition-density matrices.
  Without the corresponding internal T2 amplitudes to zero padded terms, padded
  states may contribute spuriously in this path.

Results from constraint (1) off and constraint (2) on should not be used to
assess whether relaxing constraint (1) helps until this is corrected.

## Why ordinary converged CCD is difficult to beat

`KRCCD_SS` enables the occupied-orbital Madelung shift by default while
replacing the ERI Madelung correction with a fitted, tensor-valued correction.

For converged CCD, applying no correction and applying the Madelung correction
to both orbital energies and ERI contractions are algebraically equivalent:

\[
(\epsilon+2\xi)T=A(T)+2\xi T
\quad\Longleftrightarrow\quad
\epsilon T=A(T).
\]

The fitted SS correction is entry- and channel-dependent, so it breaks this
exact cancellation while the orbital side remains ordinary Madelung. Therefore,
a result slightly worse than either uncorrected CCD or Madelung-both CCD is not
by itself evidence of a sign error. The fitted correction has to improve the
underlying quadrature by more than the cancellation error it introduces.

A systematic improvement likely requires matched SS treatments for both:

- occupied orbital energies; and
- the combined six-channel ERI residual.

Using ordinary Madelung for one side and a noisy fitted model for the other is
intrinsically fragile. The optional ExxSS-derived orbital shift is a step in
this direction, but the implementation does not impose any consistency between
that shift and the six fitted ERI corrections.

## Other correctness and robustness risks

### Gauge dependence of the constrained approximation

The default constrained samples multiply diagonal overlaps between Bloch
orbitals at neighboring k-points. There is no phase alignment or band tracking.
With constraint (2) enabled, the shifted T2 amplitude that could cancel the
orbital phases is omitted. Consequently, arbitrary orbital phase changes,
degenerate-subspace rotations, or band reordering can change the fitted sigmas
and corrections.

This is particularly concerning because the sampled values are complex but
are fitted to a real positive Gaussian. In `_fit_unit_gaussian`, the imaginary
part of the least-squares residual is independent of sigma, so it has no effect
on the optimized width. A gauge phase can rotate information between the fitted
real component and the ignored imaginary component.

A useful correctness test is to randomly rephase every MO at every k-point,
transform T2 consistently, and verify that the SS residual and final energy are
unchanged. For degenerate spaces, the analogous test should use random unitary
rotations.

Possible remedies include a parallel-transport gauge, explicit band tracking,
or using the complete density-amplitude contractions whose phases cancel.

### Sampling and model flexibility

The sample set contains only positive points along the three reciprocal-lattice
axes. It has no negative partners and no off-axis points. On a small k-mesh, the
first nonzero point may not be in the local `q -> 0` regime; in directions with
one k-point it is already a reciprocal-lattice vector.

The model is a single isotropic, real, monotone Gaussian with coefficient fixed
to one. Individual CCD tensor elements can instead be anisotropic, oscillatory,
or complex. A single width fitted to the three crystallographic directions may
therefore have little connection to the local even curvature that controls the
singularity correction.

At minimum, sampling should include `+q` and `-q` pairs and fit their even
component. Off-axis samples and an anisotropic positive-definite Gaussian would
be natural next steps. The correction should also be checked for convergence
with respect to `line_points`.

### A second singular feature from the amplitude

The exact relaxed structure factors can have an additional non-smooth feature
near

\[
q = k_a-k_i.
\]

The analysis by Xing and Lin notes that this amplitude singularity can approach
or overlap the ERI singularity, with the worst case occurring when
\(k_a=k_i\). In that case, even the Madelung correction does not uniformly
improve every ERI-contraction entry.

When constraint (2) is disabled, line samples that hit or approach this second
feature can dominate the Gaussian fit even though the model is centered only
at `q=0`. Possible strategies are:

- restrict the fit to a genuinely local neighborhood of `q=0`;
- exclude samples near the amplitude singularity;
- use a multi-center or cusp-aware model;
- retain the ordinary Madelung correction for problematic momentum blocks; or
- optimize an aggregate residual or energy correction instead of fitting every
  tensor element independently.

### Independent channel fits amplify cancellation errors

Six independently fitted corrections are combined as

```text
xi1 + xi2 - xi3 - xi4 - xi5 - xi6.
```

This is a cancellation among relatively large quantities. Small independent
fit errors can dominate the final coefficient. The resulting tensor is not
explicitly symmetrized under CCD pair exchange or time reversal, so numerical
or sampling asymmetries can also break exact T2 symmetries.

A more stable approach would fit the signed combined channel structure factor,
or perform a coupled fit with shared symmetry constraints. At minimum, the
final coefficient should be explicitly checked and symmetrized under

```text
(ki, kj, ka, i, j, a, b)
    <->
(kj, ki, kb, j, i, b, a).
```

### Fits lack quality controls

The optimizer currently accepts any finite sigma returned with a success flag.
There is no warning or rejection for:

- a sigma at either optimization bound;
- a large fitting residual;
- a large imaginary component;
- negative or strongly nonmonotone target values;
- a fitted correction far outside the range between zero and the Madelung
  correction; or
- violation of pair and time-reversal symmetries.

Useful lightweight diagnostics would include:

- fit RMS and maximum error;
- imaginary-to-real norm ratio;
- number of lower- and upper-bound hits;
- percentiles of sigma, each channel xi, and the combined coefficient;
- the combined coefficient relative to `2*madelung_constant`; and
- the norm of the pair-symmetry violation.

### Pair-density grid accuracy

Pair densities use a level-0 periodic Becke grid by default and are individually
renormalized using their grid norms. This may compensate for coarse-grid norm
errors, but it also changes the physical transition matrix elements when the
grid is not converged. The higher-Q line samples are especially sensitive to
oscillatory quadrature error.

The fitted correction should be checked at several Becke grid levels. A uniform
FFT-compatible grid may be more natural for Fourier pair densities.

### Silent broad-Gaussian fallback

`_gaussian_xi` returns the infinite-width Madelung value whenever the estimated
reciprocal sum would exceed two million points. This is silent and makes the
correction discontinuous as a function of the fitted sigma. The axis-by-axis
shell estimate is also not a rigorous cutoff for skew reciprocal cells, where
large integer combinations can partially cancel.

The code should at least report this fallback. Preferably, the sum should use a
norm-based reciprocal cutoff derived from the full reciprocal metric or a
proper Ewald evaluation.

## Parts that appear consistent

The following pieces agree with the uncontracted per-entry equations:

- The residual sign pattern is `+ + - - - -`.
- The six shifted momentum mappings appear correct.
- Channels 4 and 5 apply the required pair transpose.
- The complete constraint-off contractions have the intended internal indices
  and momentum blocks.
- The SS residual is added to the numerator before denominator division.
- The analytic Gaussian Coulomb integral and the infinite-width sign convention
  appear consistent with the local Madelung convention.

These observations make a global sign or denominator-placement error less
likely than the fitting, gauge, and consistency problems described above.

## Test coverage gaps

The slow H2 test verifies that calculations converge, that values remain
finite, and that fixed-sigma limits reproduce the orbital-only and
Madelung-both endpoints. It does not verify that fitted SS improves finite-size
error against a larger-mesh reference.

Several algebra tests reproduce the implementation formulas using random
arrays, but they are not independent physical checks against explicitly
modified ERIs or directly evaluated structure factors. In particular, the
constraint-(1)-off test encodes the erroneous assumption that its samples
should ignore T2.

Important missing regressions are:

- a mesh series compared with a large-mesh reference;
- explicit equality of converged uncorrected and Madelung-both CCD;
- gauge and degenerate-subspace invariance;
- CCD pair and time-reversal symmetry of the SS residual;
- fit-quality and bound-hit checks;
- grid-level and sampling convergence; and
- an independent direct evaluation of at least one channel correction.

## Recommended order of investigation

1. Correct or disable the constraint-(1)-off/constraint-(2)-on path.
2. Add gauge-invariance and pair-symmetry diagnostics.
3. Inspect fit residuals, imaginary norms, bound hits, and combined correction
   distributions for the existing calculations.
4. Add `+q`/`-q` sampling and fit the even component.
5. Compare a matched orbital-SS plus ERI-SS calculation rather than fitted ERI
   SS plus ordinary orbital Madelung.
6. Treat samples near the amplitude singularity separately.
7. Replace the six independent isotropic fits with a symmetry-coupled or
   combined-channel model.

## Review verification

No implementation files were changed as part of this review. Static Python
compilation and `git diff --check` passed. Runtime tests could not be executed
in the current environment because importing the local PySCF installation
fails due to the missing shared library `libmkl_intel_lp64.so.3`.

## References

- `fsec/singularity_subtraction/CCDSS.md`.
- `CCD_SS_IMPLEMENTATION_PLAN.md`.
- `CCD_SS_CONSTRAINT_1_PLAN.md`.
- X. Xing and L. Lin, “Inverse Volume Scaling of Finite-Size Error in Periodic
  Coupled Cluster Theory,” *Physical Review X* **14**, 011059 (2024),
  <https://doi.org/10.1103/PhysRevX.14.011059>.
