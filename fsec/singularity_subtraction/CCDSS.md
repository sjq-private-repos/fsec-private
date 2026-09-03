# CCD singularity subtraction

`KRCCD_SS` applies iterative singularity subtraction to the six linear
Madelung-invoking contractions in restricted periodic CCD.  The implementation
uses the complete, k-point-summed structure factors from Eqs. (2.1) and
(2.3)--(2.7) of [`ccd-ss.pdf`](../../ccd-ss.pdf).

## Interface

```python
from fsec.singularity_subtraction import CCDSSOptions, KRCCD_SS

options = CCDSSOptions(
    line_points=3,
    pair_density_becke_grid_level=2,
    fit_with_coul=True,
    fixed_sigma=None,
    amplitude_fit_tol=1e-12,
    pair_density_identity_tol=1e-3,
)
cc = KRCCD_SS(kmf, options=options)
```

The options may instead be passed directly to `KRCCD_SS`.  An options object
cannot be combined with non-default direct overrides.  `line_points` gives the
number of positive samples along each reciprocal direction.  `fixed_sigma`
may be a non-negative number, infinity, or `None` for dynamic fitting.

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

The origin is evaluated by the same overlap, reciprocal-wrap, norm, and
normalization path as every other sample, rather than being set to an
identity.  Each channel is normalized by its own `S_n(0)`, which gives an
exact unit origin after validation.  An origin smaller than
`amplitude_fit_tol` is rejected.  At `q=0`, each active occupied and virtual
block is checked against identity with `max(abs(rho-I))`; padded rows and
columns are excluded.  A deviation above `pair_density_identity_tol` names
the sector and k-point and aborts density caching.  All six normalized curves
may remain complex; fitting uses their real projections.

## Fitting and CCD updates

The normalized real curves are fitted to
`exp(-|Q|**2 / (2 sigma**2))` with unit coefficient.  Coulomb-weighted least
squares is the default.  Since the chapter-2 curves contain T2, fitted mode
rebuilds all six curves and performs six fits every time the CCD residual is
updated.  No fitted preparation occurs during amplitude initialization.

`ss_sigmas` and `ss_xi` hold the latest six-element fitted state.
`ss_prepare_count` counts successful preparations, and `ss_fit_count` advances
by six after each successful fitted preparation.  The correction inserted in
the numerator before denominator division is

```text
(xi1 + xi2 - xi3 - xi4 - xi5 - xi6) * t2
```

With `fixed_sigma=0`, the ERI correction is zero.  With
`fixed_sigma=np.inf`, it is the exact ERI Madelung correction.  Fixed widths
are amplitude-independent, cached after one preparation, and do not perform
numerical fits.

## Diagnostics

At `logger.DEBUG2`, every fitted preparation emits one `CCDSS_SF` row for each
channel and q sample.  Rows contain `prep`, `channel`, `q_index`, `qx`, `qy`,
`qz`, `raw_real`, `raw_imag`, `normalized_real`, `normalized_imag`, `sigma`,
`xi`, `fit`, `residual_real`, and `residual_imag`.  The periodic H2 diagnostic
in [`examples/h2_ccdss_structure_factors.py`](../../examples/h2_ccdss_structure_factors.py)
shows this output.
