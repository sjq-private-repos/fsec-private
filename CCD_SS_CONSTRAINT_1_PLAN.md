# CCDSS density-only one-time fit

## Goal

Support `use_constraint_1=False` while retaining `use_constraint_2=True`.
Constraint (1) selects diagonal pair factors or full transition-density
contractions. Constraint (2) fixes the amplitude momentum at its q=0 block;
`use_constraint_2=False` remains unsupported and raises `NotImplementedError`.
Both switches default to `True`, and the public `KRCCD_SS` interface is
unchanged.

## Aggregate implementation

The six fitted curves are scalar and independent of T2. At each sampled `Q`,
the implementation traverses every physical active `(ki,kj,ka,i,j,a,b)` entry.
For `kb = kconserv[ki,ka,kj]`, the diagonal mode uses the existing products

```text
L1: F_i(ki-q) F_j(kj)*
L2: F_a(ka) F_b(kb-q)*
L3: F_a(ka) F_i(ki)*
L4: F_b(kb) F_j(kj)*
L5: F_a(ka) F_j(kj)*
L6: F_b(kb) F_i(ki)*
```

The relaxed mode replaces each product by the corresponding full-density
partial:

```text
L1: ki,jl -> ij
L2: ac,db -> ab
L3: ac,ik -> ia
L4: bc,jk -> jb
L5: ac,jk -> ja
L6: bc,ik -> ib
```

The density momentum locations are unchanged: `ki-q,kj`; `ka,kb-q`; `ka,ki`;
`kb,kj`; `ka,kj`; and `kb,ki` for L1 through L6. Every second density is
complex-conjugated. Full-density internal indices are masked at their actual
minus/plus momentum endpoints in addition to the external indices, so padded
identity safeguards never enter a partial.

The shared driver uses `padding_k_idx(kind="split")` to build per-k occupied
and virtual masks. For a block, let `ni`, `nj`, `na`, and `nb` be the active
counts at `ki`, `kj`, `ka`, and `kb`. Its common physical-entry count is
`ni*nj*na*nb`. The partial channel sums are weighted by the unaffected active
counts:

```text
L1: active ij * nva*nvb
L2: active ab * noi*noj
L3: active ia * noj*nvb
L4: active jb * noi*nva
L5: active ja * noi*nvb
L6: active ib * noj*nva
```

The q=0 raw value of every channel is the exact total
`N_active = sum(ni*nj*na*nb)`. Curves are divided by `N_active`, and their
origins are explicitly assigned to exactly `1`. No T2 value, `sum(t2)`, or
near-zero-amplitude safeguard participates in fitting. The driver accumulates
scalar partial results directly and never creates a T2-sized intermediate.

## Preparation and residual update

`init_amps` prepares the six widths and six scalar `xi` values once, before CCD
iterations. A direct update path that bypasses initialization lazily performs
the same preparation on its first residual update. Both constraint-(1) modes
therefore fit exactly six curves in total, and later updates reuse the cached
state. Fixed-sigma mode also prepares once, but evaluates no numerical fits.

The residual correction remains

```text
(xi1 + xi2 - xi3 - xi4 - xi5 - xi6) * current_t2
```

and is injected into the CCD numerator before denominator division. Fitting has
no L4/L5 T2 transpose because T2 is absent from all six curves. `ss_sigmas`
and `ss_xi` remain arrays with shape `(6,)`. `amplitude_fit_tol` remains
accepted and validated for compatibility and future constraint-(2) work, but
is unused in the supported modes.

## Tests and documentation

Feature tests compare all six diagonal and relaxed aggregates with explicit
complex nested-loop references, including momentum maps, off-diagonal density
terms, active-index multiplicities, and padded external/internal states. They
also verify exact origins, independence from changed or zero T2, one-time six-
fit caching over multiple updates, fixed-sigma caching, diagonal-density
equivalence, residual signs, and residual-before-denominator injection. The
finite slow H2 calculation retains coverage for fitted and relaxed behavior.

`CCDSS.md`, the examples, and the historical `CCDSS_REVIEW.md` status note
describe this density-only one-time-fit behavior.
