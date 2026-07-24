# TRS- and AO-Symmetry-Aware Selective Lov GDF

## Summary

PySCF 2.13.1 does not use its fastest symmetry path when a custom
`kptij_lst` is supplied: it forces `aosym="s1"` and disables conjugate-\(q\)
reuse. A future optimization can retain both AO-pair symmetry and
time-reversal symmetry while constructing only the GDF blocks needed by the
small-\(q\) Lov tensors.

Measured on the ammonia \(1\times1\times2\) benchmark:

- Full optimized GDF, `s2`: 16 ordered pairs, 25.08 s.
- Stock sparse GDF, `s1`: 4 pairs, 21.24 s.
- Sparse transpose-closed prototype, `s2`: 8 half-AO blocks, 13.54 s.
- Sparse TR-reduced prototype, `s2`: 4 half-AO blocks, 11.19 s.

The occupied grid has TR map `[0, 1]`, the shifted grid `[1, 0]`, and the
`ia` and `jb` pairs form exact TR-partner sets.

## Implementation

- Compute `ka_map` and `kb_map` before constructing correlation GDF. Validate
  both grids with `conj_mapping` and require
  `trs_vir[ka[i]] == kb[trs_occ[i]]`.
- Form the requested `ia` and `jb` pairs, add their transposes for `s2` AO
  reconstruction, then quotient this closure by physical TR:
  \((k_p,k_q)\mapsto(-k_p,-k_q)\). For ammonia this gives four requested
  pairs, eight transpose-closed pairs, and four pairs requiring integral
  evaluation.
- Add a contained selective-GDF adapter that follows PySCF's normal
  `_RSGDFBuilder`/`_CCGDFBuilder` selection but permits `aosym="s2"` after
  validating transpose closure. Reuse one conjugate `j2c` factorization for
  each \(q/-q\) pair.
- Keep only the TR representatives in HDF5. Wrap `CDERIArray` so a missing
  TR-partner block is obtained by conjugating its stored representative; its
  transpose is present for `s2` reconstruction.
- Transform requested Lov blocks using their actual MO coefficients. Do not
  infer conjugate MO coefficients across TR because independently
  diagonalized orbitals can differ by phases or degenerate-subspace
  rotations.
- Log requested, transpose-closed, TR-evaluated, and full pair counts plus
  `aosym` and timing.
- Leave the separate shifted-band GDF construction unchanged.

No public API or benchmark-option changes are required.

## Test Plan

- Test TR maps and `ka`/`kb` equivariance for \(1^3\),
  \(1\times1\times2\), and \(2^3\) meshes, including self-TR boundary points.
- On a small periodic cell, compare full GDF against selective TR/`s2` CDERI
  and Lov blocks to \(10^{-10}\), covering stored, reconstructed, transposed,
  and self-TR cases.
- Verify the selective file uses `s2`, contains only evaluated
  representatives, and never requests an unavailable transpose.
- Run all existing small-\(q\) and structure-factor tests.
- Rerun the ammonia GDF/Ewald/Laplace benchmark into a profiling log. Require
  the same correction within \(10^{-10}\) Ha and compare the correlation-GDF
  time with the 25.08 s baseline; approximately 11--13 s is expected.

## Assumptions

- The small-\(q\) calculation is nonmagnetic with a real Hamiltonian, so
  physical TR applies.
- PySCF 2.13.1 private GDF-builder interfaces are the implementation target.
- If TR validation fails, retain the correct transpose-closed `s2` sparse
  build but disable TR reconstruction and emit a clear diagnostic.

