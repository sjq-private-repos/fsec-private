# CCD singularity subtraction

`KRCCD_SS` applies iterative singularity subtraction to the six linear
Madelung-invoking contractions in restricted periodic CCD.  The implementation
uses the complete, k-point-summed structure factors from Eqs. (2.1) and
(2.3)--(2.7) of [`ccd-ss.pdf`](../../ccd-ss.pdf).

## Interface

```python
from fsec.singularity_subtraction import CCDSSOptions, KRCCD_SS

options = CCDSSOptions(
    auxfunc="Gauss",
    line_points=3,
    pair_density_becke_grid_level=2,
    fit_with_coul=True,
    fixed_sigma=None,
    amplitude_fit_tol=1e-12,
    pair_density_identity_tol=1e-3,
    structure_factor_mode="separate",
)
cc = KRCCD_SS(kmf, options=options)
```

The options may instead be passed directly to `KRCCD_SS`.  An options object
cannot be combined with non-default direct overrides.  `line_points` gives the
number of positive samples along each reciprocal direction.  `fixed_sigma`
may be a non-negative number, infinity, or `None` for dynamic Gaussian fitting.
The default `auxfunc="Gauss"` preserves the existing Gaussian behavior.
The default `structure_factor_mode="separate"` fits the six channels
independently.  Set `structure_factor_mode="combined"` to fit their signed
sum as one curve.

`auxfunc` accepts `"Gauss"` or the opt-in `"QuarticExponential"`.  For
example:

```python
options = CCDSSOptions(auxfunc="QuarticExponential", line_points=3)
cc = KRCCD_SS(kmf, options=options)
```

The quartic model has unit coefficient and is

```text
f(Q) = exp[-alpha * (sqrt(1 + beta/alpha * |Q|^2
                          + kappa/alpha * |Q|^4) - 1)]
```

where `alpha`, `beta`, and `kappa` are positive.  In separate mode, each of
the six normalized structure-factor channels is fit independently; in
combined mode, their signed raw sum is normalized once and fit once.  Fits use
the real projection and the same optional Coulomb weights as the Gaussian fit.
The parameters are optimized in log coordinates.  The initial values are
`alpha=1`, `beta=s^-2`, and `kappa=s^-4`, where `s` is the smallest nonzero
sampled momentum norm; each parameter has bounds spanning `1e-8` to `1e8`
relative to its initial value.  `fixed_sigma` is restricted to the Gaussian
model and cannot be combined with `auxfunc="QuarticExponential"`.

By default, the occupied one-particle energies include the orbital Madelung
shift.  A scalar or `(nkpts, nocc)` `occupied_orbital_shift` replaces that
default as well as any inherited `keep_exxdiv` correction.

## Chapter-2 structure factors

For each sample `q`, `plus[k]` and `minus[k]` identify `k+q` and `k-q` on the
finite mesh.  A normalized transition density

```text
rho[k, p, s] = <u_p,k | exp(-i G.r) | u_s,k+q>
               / sqrt(<u_p,k|u_p,k><u_s,k+q|u_s,k+q>)
```

is built on a periodic Becke grid.  Only the occupied--occupied and
virtual--virtual matrices are cached after all origin checks succeed.  With
`kb = kconserv[ki, ka, kj]`, the six scalar sums use:

```text
S1: ki,jl,klab -> rho_oo[minus[ki]], rho_oo[kj]*,
                   t2[minus[ki], plus[kj], ka]
S2: ac,db,ijcd -> rho_vv[ka], rho_vv[minus[kb]]*,
                   t2[ki, kj, plus[ka]]
S3: ac,ik,kjcb -> rho_vv[ka], rho_oo[ki]*,
                   t2[plus[ki], kj, plus[ka]]
S4: bc,jk,kica -> rho_vv[kb], rho_oo[kj]*,
                   t2[plus[kj], ki, plus[kb]]
S5: ac,jk,kibc -> rho_vv[ka], rho_oo[kj]*,
                   t2[plus[kj], ki, kb]
S6: bc,ik,kjac -> rho_vv[kb], rho_oo[ki]*,
                   t2[plus[ki], kj, ka]
```

Every external and internal orbital is summed.  Padded states are removed at
the actual momentum of each density endpoint and amplitude index.  Each block
is contracted directly to a scalar; the implementation does not allocate a
second tensor with the size of T2.

The six raw contractions are retained in both modes.  In combined mode they
are reduced before normalization with

```text
S_combined(Q) = S1(Q) + S2(Q) - S3(Q) - S4(Q) - S5(Q) - S6(Q).
```

The origin is evaluated by the same overlap, reciprocal-wrap, norm, and
normalization path as every other sample, rather than being set to an
identity.  Each separate channel is normalized by its own `S_n(0)`; the
combined curve is normalized by `S_combined(0)`.  These give an exact unit
origin after validation.  An applicable origin smaller than
`amplitude_fit_tol` is rejected.  At `q=0`, each active occupied and virtual
block is checked against identity with `max(abs(rho-I))`; padded rows and
columns are excluded.  A deviation above `pair_density_identity_tol` names
the sector and k-point and aborts density caching.  All six normalized curves
and the combined normalized curve may remain complex; fitting uses only their
real projections.

## Fitting and CCD updates

In Gaussian mode, the normalized real curves are fitted to
`exp(-|Q|**2 / (2 sigma**2))` with unit coefficient.  Coulomb-weighted
least-squares is the default.  In quartic mode, the model and fitting
definitions are those above.  Since the chapter-2 curves contain T2, dynamic
separate mode rebuilds all six curves and performs six fits every time the CCD
residual is updated.  Dynamic combined mode reduces those curves first and
performs one fit per update.  No dynamic fitted preparation occurs during
amplitude initialization.

`ss_model_parameters` holds the latest model parameters.  Separate mode uses
shape `(6, 2)` with rows `[1, sigma]` in Gaussian mode and shape `(6, 4)` with
rows `[1, alpha, beta, kappa]` in quartic mode.  Combined Gaussian mode stores
`[1, sigma]` with shape `(2,)`; combined quartic mode stores
`[1, alpha, beta, kappa]` with shape `(4,)`.  Similarly, `ss_sigmas` and
`ss_xi` are six-element arrays in separate mode and scalar values in combined
mode; `ss_sigmas` is `None` for quartic models.
`ss_prepare_count` counts successful preparations, and `ss_fit_count` advances
by six or one after each successful fitted preparation in separate or
combined mode, respectively.  The correction inserted in the numerator
before denominator division is

```text
(xi1 + xi2 - xi3 - xi4 - xi5 - xi6) * t2
```

In combined mode the signed origin is `-2A` when the six channel origins are
the common value `A`, so the correction is `(-2 * ss_xi) * t2`.

With `fixed_sigma=0`, the ERI correction is zero.  With
`fixed_sigma=np.inf`, it is the exact ERI Madelung correction.  Fixed widths
are amplitude-independent, cached after one preparation, and do not perform
numerical fits.  For the quartic model, each `xi` is the continuous Coulomb
integral from `QuarticExponentialModel` minus the reciprocal-lattice
quadrature.  The reciprocal sum is truncated where the model exponent reaches
32 and retains the existing lattice-size safety cap.

## Diagnostics

At `logger.DEBUG2`, every fitted preparation emits one `CCDSS_SF` row for each
channel and q sample in separate mode.  Combined mode emits exactly one row
per q sample with `channel=combined`.  Rows contain `prep`, `channel`,
`q_index`, `auxfunc`, `qx`, `qy`, `qz`, `raw_real`, `raw_imag`,
`normalized_real`, `normalized_imag`, `xi`, `fit`, `residual_real`, and
`residual_imag`, plus `sigma` in Gaussian mode or `alpha`, `beta`, and `kappa`
in quartic mode.  These rows report the latest committed diagnostic state only
after all channel fits succeed.  The
periodic H2 diagnostic in
[`examples/h2_ccdss_structure_factors.py`](../../examples/h2_ccdss_structure_factors.py)
continues to show the Gaussian output.
