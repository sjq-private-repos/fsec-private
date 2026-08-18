# Independently Relax CCDSS Constraint (1)

## Summary

Add a separate switch for constraint (1). When only constraint (1) is disabled,
sum over all internal orbital pairs in the pair densities while keeping the T2
amplitude fixed at the external $t_{ij}^{ab}(k_i,k_j,k_a)$. Thus, $\xi$ remains
precomputable whenever constraint (2) is enabled.

## Implementation

- Add `use_constraint_1: bool = True` to `CCDSSOptions` and `KRCCD_SS`, with a
  property and `dump_flags` entry. Preserve both constraints as enabled by
  default.
- Build normalized transition-density matrices $\rho_{p k,r(k+q)}$ when
  constraint (1) is disabled. Set their $q=0$ value to the exact identity.
- With constraint (2) enabled, factor out the unchanged external T2 amplitude
  and form density-only sums:
  - L1: `ki,jl->ij`
  - L2: `ac,db->ab`
  - L3: `ac,ik->ia`
  - L4: `bc,jk->jb`
  - L5: `ac,jk->ja`
  - L6: `bc,ik->ib`
  Broadcast each result over its unaffected external indices, fit the normalized
  channel, and continue scaling the resulting $\xi_n$ by the current external
  T2 during every CCD iteration.
- With constraint (2) disabled:
  - Keep the existing shifted-amplitude behavior when constraint (1) is
    enabled.
  - When both constraints are disabled, evaluate the complete
    density-amplitude contractions from Eqs. (13) and (21)-(25):
    - L1: `ki,jl,klab->ijab`
    - L2: `ac,db,ijcd->ijab`
    - L3: `ac,ik,kjcb->ijab`
    - L4: `bc,jk,kica->ijab`
    - L5: `ac,jk,kibc->ijab`
    - L6: `bc,ik,kjac->ijab`
- Precompute and reuse `ss_xi` whenever constraint (2) is enabled, regardless
  of constraint (1). Refit every update only when constraint (2) is disabled.
- Preserve the existing small-external-amplitude safeguard for normalized
  channels involving shifted T2 amplitudes.
- Use PySCF-style `lib.einsum` and separate helpers for pair-density
  construction, density-only sums, and full density-amplitude contractions.

| Constraint 1 | Constraint 2 | Pair densities | T2 amplitude | Fitting |
| --- | --- | --- | --- | --- |
| on | on | matching indices | external T2 | once |
| off | on | full orbital sums | external T2, outside sum | once |
| on | off | matching indices | shifted T2 | every update |
| off | off | full orbital sums | shifted internal-index T2 | every update |

## Tests

- Extend option merging, properties, logging, and defaults for
  `use_constraint_1`.
- Verify all six density-only sums against explicit nested-loop references and
  confirm the external T2 does not enter their construction.
- Verify all six full contractions and shifted momentum mappings when both
  constraints are disabled.
- Test all four switch combinations, including that both constraint-(2)-enabled
  cases precompute exactly once.
- Verify exact origin normalization, complex pair densities, broadcasting over
  unaffected indices, padded orbitals, and negligible external amplitudes.
- Extend the small H2 integration test to cover constraint (1) disabled with
  constraint (2) both enabled and disabled.
- Run `PYTHONPATH=. pytest -q fsec/singularity_subtraction/tests/test_ccdss.py`,
  including the slow test when feasible.

## Assumptions

- Disabling only constraint (1) changes the pair-density orbital sums, not the
  indices or momentum of the external T2 amplitude.
- Existing results remain unchanged with both default switches enabled.
- Fixed-sigma behavior is independent of either structure-factor sampling
  constraint.
