# TRS- and AO-Symmetry-Aware Selective Lov GDF

## Summary

PySCF 2.14 does not use its fastest symmetry path when a custom
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

## Pair Counting

A \(1\times1\times2\) mesh does contain only two k-points in the original
grid. Small-\(q\), however, introduces a second, half-shifted grid containing
two virtual k-points. The correlation GDF object is built on the union of
these grids, so its internal k-point list has four entries:

| Quantity | Count |
| --- | ---: |
| Original/occupied k-points | 2 |
| Half-shifted/virtual k-points | 2 |
| K-points in the combined correlation GDF | 4 |
| All ordered pairs in a generic full GDF, \(4^2\) | 16 |
| Physical Lov pairs requested by small-\(q\) | 4 |
| Orientations after Hermitian `s2` closure | 8 |
| Orientations evaluated after TR reduction | 4 |

Thus, four is indeed the number of physical Lov pairs. For each of the two
\(k_i\) values, fixed \(q'\) determines exactly one \(k_a\), giving two `ia`
pairs. Likewise, each of the two \(k_j\) values determines exactly one
\(k_b\), giving two `jb` pairs. In the combined-grid indexing used by the
ammonia calculation, these four requested ordered pairs are
`(0,2)`, `(1,3)`, `(0,3)`, and `(1,2)`.

The count of 16 is only the full-GDF baseline: a generic build on the four
combined k-points constructs every ordered pair, including
original--original and shifted--shifted pairs that Lov never uses.

The count temporarily rises from four to eight because retaining Hermitian
AO-pair symmetry (`aosym="s2"`) requires the reverse orientation of every
requested pair. Those reverse orientations are bookkeeping/integral blocks,
not four additional physical Lov contractions. Time-reversal symmetry groups
the eight orientations into four conjugate pairs, so only four half-AO
orientations require explicit integral evaluation; the other four are
reconstructed by conjugation.

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
- PySCF 2.14 private GDF-builder interfaces are the implementation target.
- If TR validation fails, retain the correct transpose-closed `s2` sparse
  build but disable TR reconstruction and emit a clear diagnostic.

## Implementation Results

Implemented and profiled on the ammonia \(1\times1\times2\) benchmark:

- Four requested pairs produce eight transpose-closed `s2` orientations and
  four explicitly evaluated TR representatives.
- Stored ammonia CDERI blocks are byte-identical to the full build; blocks
  reconstructed through TR agree within \(9.5\times10^{-14}\).
- The isolated CDERI file is 231 MiB versus 893 MiB for the full build.
- Correlation GDF construction is 12.09 s versus 22.26 s for a full 16-pair
  control through the same code path, a 1.84x speedup.
- The complete small-\(q\) kernel is 44.80 s versus 53.87 s, a 16.8% reduction.
- The end-to-end benchmark is 1:58.85 versus 2:07.69.
- All 31 structure-factor tests pass, including full-versus-selective CDERI
  checks for stored and reconstructed pairs.

The independently executed heavy runs show approximately \(2.6\times10^{-6}\)
Ha variation in the fitted \(q_2\) correction. The integral-level comparison,
which holds the physical inputs fixed, agrees to the tolerances above; the
independent-run variation is therefore reported separately from CDERI
correctness.
