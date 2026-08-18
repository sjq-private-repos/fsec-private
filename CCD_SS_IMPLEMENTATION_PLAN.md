# KRCCD Singularity Subtraction Implementation

## Summary

First save this complete plan as `CCD_SS_IMPLEMENTATION_PLAN.md`. Then launch Luna max-thinking to implement iterative CCD singularity subtraction, review its work, return substantial issues to Luna, and directly fix only smaller issues.

## Implementation

- Add `fsec/singularity_subtraction/ccdss.py` with `KRCCD_SS`, subclassing PySCF's periodic restricted `kccsd_rhf.RCCSD`.
- Keep T1 identically zero through amplitude initialization, updates, and energy evaluation.
- Override `update_amps` to apply the six corrections from `ccd-ss.pdf` with the residual coefficient
  \[
  \xi_1+\xi_2-\xi_3-\xi_4-\xi_5-\xi_6.
  \]
- Preserve PySCF momentum conservation, padding, frozen-orbital handling, denominators, and occupied-orbital Madelung treatment.
- Sample the origin and three positive points along each reciprocal-lattice vector by default.
- Precompute required pair densities on weighted periodic Becke grids. Port the relevant strategy from sibling repository `../fsec` without creating a runtime dependency on it.
- Fit normalized isotropic Gaussian structure factors with fixed unit coefficient using Coulomb-weighted least squares.
- Calculate each correction as the analytic Gaussian integral minus the finite \((q+G)\) quadrature.
- With `use_constraint_2=True`, prepare \(\xi_n\) before CCD iterations using the \(q=0\) T2 value. When false, refit from current T2 during each update. Constraint (1) remains enabled in both modes.
- Define `fixed_sigma=0` as zero ERI SS correction and `fixed_sigma=np.inf` as the exact Madelung ERI correction.

## Public API

Export `CCDSSOptions` and:

```python
KRCCD_SS(
    mf,
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
)
```

Document array conventions, constraints, fitting behavior, and six-term mappings. Reject invalid sampling counts, grid levels, sigma values, and incompatible amplitude shapes with clear errors.

## Tests

- Test line sampling, Gaussian fitting, Becke weights, origin normalization, padding, frozen orbitals, and momentum-index handling.
- Independently verify every channel mapping and sign.
- Verify residual injection occurs before denominator division.
- Confirm constrained corrections are computed once and reused.
- Confirm unconstrained corrections are refitted after T2 changes.
- Cover invalid inputs, negligible amplitudes, padded bands, and non-finite fit results.
- For H2/GTH-SZV on a \(1\times1\times2\) mesh, verify:
  - near-zero sigma matches orbital-correction-only CCD;
  - large sigma approaches orbital-plus-ERI Madelung CCD;
  - infinite sigma reproduces the exact Madelung residual limit;
  - the default fitted calculation stays finite and converges.

## Luna Orchestration and Review

1. Write this plan to `CCD_SS_IMPLEMENTATION_PLAN.md` before editing implementation files.
2. Launch Luna with the exact requested command, replacing `PLAN` with this plan and all necessary repository context:

   ```bash
   codex exec \
     -m gpt-5.6-luna \
     -c 'model_reasoning_effort="max"' \
     --ephemeral \
     -s workspace-write \
     -a never \
     'PLAN'
   ```

3. Tell Luna to preserve existing user changes, inspect all three PDFs, use `../fsec` as the Becke-grid reference, implement the feature, and run feasible tests.
4. Review the complete diff for mathematical correctness, simple abstractions, PySCF conventions, useful documentation, performance, and meaningful failure-path tests.
5. Run targeted tests and static checks. If the existing MKL loading error prevents PySCF execution, record the exact blocked tests.
6. Send substantial or interconnected findings back to another Luna max-thinking run with the full plan, current state, failures, and issue list. Make small, localized corrections directly when that is materially cheaper.
7. Repeat review and testing until the implementation is coherent and every feasible check passes.

## Cost Accounting

Maintain a usage ledger for every Luna invocation and the orchestrator's work.

- Luna:
  \[
  C_L = 0.20(I_L/10^6)+1.20(O_L/10^6)
  \]
- SOL xhigh orchestrator:
  \[
  C_O = 0.50(I_O/10^6)+30.00(O_O/10^6)
  \]

Aggregate all Luna retries into the Luna total. Report input tokens, output tokens, input cost, output cost, and total cost separately for Luna and the orchestrator.

Use CLI telemetry where available. If exact orchestrator telemetry is unavailable, provide a clearly labeled estimate and explain the counting basis rather than presenting it as exact.

Also report a SOL-xhigh-only counterfactual:

- Estimate the tokens SOL would have needed to perform implementation, review, fixes, and testing alone.
- Base the estimate on observed technical work from the actual run, excluding delegation prompts, repeated context sent only to Luna, and orchestration-only reporting.
- Apply $0.50/M input and $30.00/M output.
- Label this as a counterfactual estimate, state its assumptions, and compare it with the actual combined Luna-plus-orchestrator cost.

## Assumptions

- Existing repository modifications and the attached PDFs remain untouched.
- Becke-grid reference code comes from `../fsec`, not `../fsec-private`.
- PySCF 2.14's periodic restricted `RCCSD` is the available KRCCD base.
- Defaults are constraint (2) enabled, three positive line samples per reciprocal direction, Becke grid level 0, and Coulomb-weighted fitting.
- The supplied prices are user-provided rates and contain no cached-input discount.
