# MP3 density fitting and RSDF-direct integration

This document reviews FSEC at `d5a34c0` on 2026-10-28, together with the
local PySCF fork (`pyscf-sjq`, `dfddc7c5`). It covers ordinary `KMP3` and the
four staggered MP3 schemes (2a, 2b, 3a, 3b) in both submesh and extension
modes. The last sections note what carries over to staggered CCD and GW.
None of the proposal is implemented yet. The numerical checks in
[§2.2](#22-numerical-checks-on-staggered-unions) were run as standalone probes
and are not part of the test suite.

## Summary

- **Every MP3 solver uses density fitting only to generate integrals.**
  Each solver calls a periodic `ao2mo` backend once per k-point quartet and
  stores dense four-index MO blocks (`oooo`, `oovv`, `ovov`, `voov`,
  `vvvv`). No contraction uses three-index factors.
- **GDF is the only backend that is exercised.** In extension mode the
  constructor always builds a new GDF on the union mesh. Stored-integral
  RSDF fits the same interface but is untested. RSDF-direct cannot run
  through the current `ao2mo` path at all.
- **`vvvv` dominates memory**, at 73–91% of integral storage in the diamond
  example below. All integrals are held in memory; there is no out-of-core
  path.
- **The fork already produces the needed factors.** Its
  `rsdf_direct_ao2mo.ao2mo_e2` produces metric-fitted pair factors on
  staggered k-point unions. In the probes, `Σ_L L[kp,kr] L[kq,ks]` matches
  stored-RSDF `ao2mo` to about 5e-11, with BvK on or off.
- **Proposal:** a single pair-factor provider, keyed by (pair type, left
  mesh, right mesh), serves all five solvers. It lands first as a drop-in
  replacement for the dense `_ERIS_*` builders (Stage A). Stage B then
  replaces stored ERIs with tiles generated from factors. Loops are grouped
  so that each tile is built once.

## 1. Current capabilities

### 1.1 Where each solver gets its integrals

| Solver | Scheme | ERI builder | Backend, submesh mode | Backend, extension mode |
| --- | --- | --- | --- | --- |
| `KMP3` | ordinary | `_ERIS_MP3` | `mp._scf.with_df` | not applicable |
| `KMP3_2mesh` | 2a | `_ERIS_2mesh` | `mp.with_df = mf.with_df` | `df.GDF(cell, union).build()` in `__init__` |
| `KMP3_2mesh_mixed` | 2b | `_ERIS_3mesh` | as above | as above |
| `KMP3_3mesh(scheme='occ')` | 3a | `_ERIS_3mesh` | as above | as above |
| `KMP3_3mesh(scheme='vir')` | 3b | `_ERIS_3mesh` | as above | as above |

- Every builder calls `with_df.ao2mo(mo_coeffs, kpts, compact=False)` for
  each quartet. It divides by the number of k-points on one mesh: `nkpts`,
  `nkpts_ov`, or `nkpts_base`.
- `KMP3` reads `_scf.with_df` directly, so assigning `mp.with_df` has no
  effect. The staggered solvers read `mp.with_df`, which can be replaced
  before `ao2mo()` or `kernel()`. In extension mode, however, the
  constructor has already paid for a GDF build by that point.
- Extension mode first evaluates band orbitals with a temporary FFTDF; KRHF
  also temporarily sets `exxdiv='vcut_sph'`. It then builds GDF on the
  union. An RSDF reference therefore does not carry over to the correlation
  integrals. Extension mode requires `cell.dimension == 3`.

### 1.2 Backend compatibility

| Backend | Status |
| --- | --- |
| GDF | Used by [the example](../examples/staggered_mp3.py) and the slow regressions in [test_kmp3.py](../fsec/staggered_mesh/mp/tests/test_kmp3.py), with absolute tolerances of 1e-9 to 1e-10 |
| Stored RSDF (`direct=False`) | Same `ao2mo` interface, so it should work; no MP3 regression covers it |
| FFTDF, MDF | No type restriction and no validation |
| RSDF-direct (`direct=True`) | **Not usable.** `RSGDF` inherits `GDF.ao2mo`, which goes through `df_ao2mo.general` to `sr_loop`. `sr_loop` calls `build()`, which skips the three-center build when `direct=True`, so no `_cderi` exists to read |
| Contractions over three-index factors | None in MP3 |
| Out-of-core integrals | None. A `MemoryError` is raised unless `cell.incore_anyway` is set |

RSDF raises `NotImplementedError` for `cell.dimension < 3`. Low-dimensional
cells, which only submesh mode supports, must therefore stay on GDF.

### 1.3 Integral and amplitude storage

Each mesh assignment of a block is one dense array of shape
`(K, K, K, n_p, n_q, n_r, n_s)`, where `K` is the per-mesh k-point count.
`mesh_blocks_mp3` gives the number of assignments per block:

| Scheme | `oovv` | `oooo` | `ovov` | `voov` | `vvvv` | Auxiliary MP2 amplitudes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ordinary, 2a | 1 | 1 | 1 | 1 | 1 | 1 |
| 2b | 5 | 1 | 1 | 2 | 1 | 5 |
| 3a | 5 | 1 | 2 | 2 | 1 | 5 |
| 3b | 7 | 1 | 2 | 2 | 1 | 7 |

The generalized kernel forms one auxiliary MP2 amplitude array for each
`oovv` assignment, plus `t_mp3`. Custom `set_mesh_config` assignments
change these counts.

The example system is two-atom diamond with gth-dzvp/gth-pade (`nao=26`,
`nocc=4`, `nvir=22`) and the default RSDF auxiliary basis (`naux=168`).
Sizes are complex128 in GB (1e9 bytes), excluding transformation buffers:

| Scheme | K | Dense ERIs | of which `vvvv` | Amplitudes | Pair factors needed (§3.1) | All union pairs × `nmo²` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ordinary | 27 | 81 | 74 | 4.9 | 1.2 | 1.3 |
| 2a | 27 | 81 | 74 | 4.9 | 1.2 | 5.3 |
| 2b | 27 | 93 | 74 | 14.6 | 1.7 | 5.3 |
| 3a | 27 | 96 | 74 | 14.6 | 1.7 | 11.9 |
| 3b | 27 | 101 | 74 | 19.5 | 2.6 | 11.9 |
| 2a | 64 | 1081 | 983 | 65 | 6.5 | 30 |
| 3b | 64 | 1341 | 983 | 260 | 14.7 | 67 |

Removing `vvvv` is the first-order memory win. For 2b, 3a, and 3b, the
stored auxiliary amplitudes and the remaining `oovv` assignments come next.
`with_t2=False` frees amplitudes after the energy is computed but does not
lower the peak. `mp.eris` keeps every integral alive after `kernel()`.

## 2. RSDF-direct in the local PySCF fork

### 2.1 Available API

- **`RSDF.direct = True`.** `build()` performs only the range-separation
  setup. `get_jk` has a direct path; `ao2mo` does not.
- **`pyscf.pbc.df.rsdf_direct_ao2mo.ao2mo_e2`.** The call is
  `ao2mo_e2(mydf, (C1, C2), kpts, j3c_order='Lij', kernel=1, out=None)`.
  It returns an `(nk, nk)` object array, or fills an `h5py.Group`, whose
  `[ki, kj]` entry is the metric-fitted factor `L[aux, i, j]`. It covers
  every ordered pair of the k-points supplied. Orbital widths may differ
  between k-points, but a width of zero raises a `ValueError` from a
  reshape. Unwanted k-points therefore cannot be masked out this way.
- **Selective generation hook.** AO pairs are enumerated by
  `get_kptij_lst(kpts)` with `ksym='s2'` and generated by `loop_j3c`.
  `loop_j3c` accepts an explicit `kptij_lst`, which is the hook for
  generating only selected pairs later.
- **BvK supercell.** `ao2mo_e2` calls `kpts_to_kmesh`, which for a staggered
  union returns the fine grid that contains it. For example, a 16-point
  two-mesh union of 2³ gives `(4, 4, 4)`, and a 24-point three-mesh union
  gives `(6, 6, 6)`. In extension mode the BvK supercell is therefore
  `nmesh^(d-1)` times larger than the union, for `d` sampled axes. In
  submesh mode the union is the complete SCF mesh, so there is no overhead.
  Results are correct (§2.2); the cost has not been measured.
- **`kmp2_direct._init_mp_df_eris(mp, ..., oslice=...)`.** This wraps
  `ao2mo_e2` for occupied–virtual factors only and reads `mp._scf.with_df`.
  The FSEC
  [MP2 structure-factor helpers](../fsec/singularity_subtraction/structure_factor/mp2_rsdf_direct_helpers.py)
  use it. It is not enough for MP3, which also needs `oo` and `vv` factors.
- **Memory estimate caveat.** The `Lij` kernels set `size_Lij0` to the full
  table, `Σ_ki Σ_kj n1·n2·naux`, and then multiply it by `2 + nkpts²`. For
  realistic tables `mem_avail` becomes negative and the AO block size
  collapses to its minimum. Check this before benchmarking.

The fork API is experimental. FSEC's `pyscf` dependency is unpinned, so
record the fork commit, as `DIRECT_RSDF_TARGET_COMMIT` does for MP2.

### 2.2 Numerical checks on staggered unions

The probes used He₂ in a 2.6 Å cube with gth-dzv, `precision=1e-10`, and
random complex unitary MO coefficients at each k-point. Stored RSDF used the
same cell and auxiliary basis.

| Union | Check | Max abs error |
| --- | --- | ---: |
| 2-mesh, base 1×1×2 (4 points, unsorted) | `Σ_L L[kp,kr] L[kq,ks]` vs stored-RSDF `ao2mo`; all 192 `ovov`-, `oovv`-, and `voov`-type quartets; BvK on and off | 4.7e-11 |
| same | `L_vo[ka,ki] == L_ov[ki,ka].conj().transpose(0,2,1)` | 1.7e-13 |
| 2-mesh, base 1×2×2 (8 of a 1×4×4 grid) | BvK vs no-BvK `L_ov`; BvK vs stored RSDF on 25 random quartets | 1.6e-13; 1.8e-11 |
| 3-mesh, base 1×2×2 (12 of a 1×6×6 grid) | same | 2.3e-13; 2.3e-11 |

These checks support three conclusions:

- The factor product reconstructs `(pr|qs)` without any extra conjugation.
- Reversed pairs are conjugate transposes, so they need not be generated
  separately.
- BvK is safe on incomplete unions.

**Pitfall.** `pyscf.pbc.lib.kpts_helper.get_kconserv` takes a fast path
whenever `kpts_to_kmesh` finds a complete mesh, and it then assumes
`make_kpts` ordering. On the concatenated 1×1×4 union it returned wrong
partners, for example `kconserv[0,1,0] = 3` instead of `1`. FSEC's
[`make_kpts_helper`](../fsec/staggered_mesh/_kpoints.py) exists for this
reason. Any factor code must use `mp.khelper.kconserv`, never a fresh
`get_kconserv`.

## 3. Proposal

### 3.1 Factor families per scheme

Because `<pq|rs> = (pr|qs)`, each stored block is a product of two ordered
pair factors:

| Block | `(p,r)` factor | `(q,s)` factor |
| --- | --- | --- |
| `oovv` | `ov` | `ov` |
| `oooo` | `oo` | `oo` |
| `ovov` | `oo` | `vv` |
| `voov` | `vo` | `ov` |
| `vvvv` | `vv` | `vv` |

Define a **family** `F(type; ml, mr)` as the factors `L[kl, kr]` for all
`kl` on mesh `ml` and all `kr` on mesh `mr`. It is a `(K, K)` table of
`(naux, nl, nr)` arrays. The families for each scheme follow from applying
this table to every `(mp, mq, mr, ms)` returned by `mesh_blocks_mp3`. For
2a, use the generalized configuration `(1,1,0,0)`,
`[(1,1),(0,0)] + [(1,0)]*6`, which
`test_nmesh_engine_reproduces_existing_staggered_schemes` already shows to
be equivalent. Finally, drop every family that is the conjugate transpose of
another: `vo(a,b)` of `ov(b,a)`, `oo(a,b)` of `oo(b,a)`, and `vv(a,b)` of
`vv(b,a)`.

| Scheme | Independent families | Count | Same-mesh families |
| --- | --- | ---: | --- |
| ordinary | `oo`, `ov`, `vv` | 3 | all |
| 2a | `oo(1,1)`, `ov(1,0)`, `vv(0,0)` | 3 | `oo`, `vv` |
| 2b | `oo(0,1)`, `vv(0,1)`, `ov(0,0)`, `ov(0,1)`, `ov(1,0)`, `ov(1,1)` | 6 | `ov(0,0)`, `ov(1,1)` |
| 3a | `oo(1,0)`, `oo(2,0)`, `vv(1,2)`, `ov(0,1)`, `ov(0,2)`, `ov(1,2)`, `ov(2,1)` | 7 | none |
| 3b | `oo(1,2)`, `vv(0,1)`, `vv(0,2)`, `ov(1,0)`, `ov(1,2)`, `ov(2,0)`, `ov(2,1)` | 7 | none |

Only same-mesh families have transfers that include `q = 0`. This is the
zero-transfer property of each scheme seen from the factor side:

- 3a and 3b never need same-mesh families.
- 2b needs them only for `ov`, where the omitted `q = 0` term vanishes by
  orthogonality.
- 2a keeps the singular same-mesh `oo` and `vv` pairs and relies on its
  Madelung correction.

This has two consequences:

- **One provider serves every scheme.** Only the family list differs. Derive
  it from `_mesh_eri` rather than hand-coding it, so that custom
  `set_mesh_config` assignments work automatically.
- **3a and 3b can skip AO pairs.** They never need AO pairs within one mesh,
  so a selective `loop_j3c` driver could skip about a third of the union's
  unordered k-pairs.

### 3.2 Pair-factor provider

Place the provider under `fsec/staggered_mesh/`, for example as
`_pair_factors.py`, so that CCD and GW can reuse it:

```python
class PairFactors:
    """Metric-fitted pair factors L[aux, p, r] on staggered k-point meshes.

    ``get(kind, ml, mr)`` returns a (K, K) object array whose [ikl, ikr]
    entry has shape (naux, n_l, n_r), with k_l the ikl-th point of mesh ml
    and k_r the ikr-th point of mesh mr.  ``kind`` is 'oo', 'ov', 'vo' or
    'vv'.  Reversed families are returned as conjugate transposes.
    """
    def __init__(self, mp, mydf, families, mo_coeff=None): ...
    def get(self, kind, ml, mr): ...
```

The mesh maps are:

- ordinary: `{0: arange(nkpts)}`
- 2a: `{0: kpts_idx_vir, 1: kpts_idx_occ}`
- 2b, 3a, 3b: `kpts_base_idx`

Coefficients are the padded `mo_coeff` used by the builders now, including
band orbitals in extension mode.

The provider can be generated in two stages:

1. **Initial version.** Make one `ao2mo_e2` call on the union with full
   padded coefficients as both `C1` and `C2`. Slice the families out and
   discard the rest. This is a single AO three-center pass, but its transient
   size is the last column of the §1.3 table: 11.9 GB for 3a/3b at K=27,
   against 1.7–2.6 GB retained. Pass `out=` an HDF5 group when the
   transient does not fit.
2. **Later version.** Write a driver around `loop_j3c` that takes the
   selected `kptij_lst` and occupied/virtual slices for each pair. This
   avoids both the transient and the unused AO pairs.

### 3.3 Backend selection

Use PySCF-style attribute assignment instead of new constructor arguments:

- **Lazy default.** In extension mode, leave `with_df` unset in `__init__`
  and build the default GDF on first use in `ao2mo()`. `KMP3` should read
  `self.with_df` and fall back to `_scf.with_df`.
- **Dispatch.** `ao2mo()` uses the provider when `with_df` is an `RSDF`
  with `direct=True`, and the current dense path otherwise. Initially,
  validate `cell.dimension == 3`, `semidirect=False`, and `ksym='s2'`.

With these changes, the user writes:

```python
mp = KMP3_3mesh(kmf, scheme='vir')      # no GDF build in extension mode
mydf = df.RSDF(mp.cell, mp.kpts)
mydf.direct = True
mp.with_df = mydf.build()               # range-separation setup only
mp.kernel()
```

In submesh mode the union is the SCF mesh, so an SCF that already used
direct RSDF can be reused as is.

### 3.4 Stage A: dense ERIs from factors

Replace each `fao2mo` call in `_ERIS_MP3`, `_ERIS_2mesh`, and `_ERIS_3mesh`
with

```text
eri[ikp, ikq, ikr] = einsum('Lpr,Lqs->pqrs', F_pr[ikp, ikr], F_qs[ikq, iks]) / K
```

Here `K` is `nkpts` for ordinary MP3, `nkpts_ov` for 2a, and `nkpts_base`
for 2b, 3a, and 3b. These are the divisors the builders use now, and the
energy routines divide by `K` again. Normalizing by the union size would
give wrong energies.

The kernels stay unchanged. Stage A removes the union GDF build and its
`_cderi` file, and its DF error matches stored RSDF, but memory is
unchanged. To exit Stage A, E2, E3, and `t2` must match the stored-RSDF
dense path to 1e-9 for all five solvers in both modes.

### 3.5 Stage B: tiles instead of stored ERIs

Every ERI tile carries three k-labels, and the fourth amplitude label is
free, so each tile is used by `K` outer amplitude blocks. If the loops are
grouped by tile, building tiles from factors costs only a fraction of the
contraction that consumes them:

| Tile | Build cost | Use cost (`K` blocks) | Overhead ratio | Diamond, K=27 |
| --- | --- | --- | --- | ---: |
| `vvvv` (2h4p) | `naux·v⁴` | `K·o²·v⁴` | `naux/(K·o²)` | 0.39 |
| `oooo` (4h2p) | `naux·o⁴` | `K·o⁴·v²` | `naux/(K·v²)` | 0.013 |
| `voov`, `ovov` (3h3p) | `naux·o²·v²` | `K·o³·v³` | `naux/(K·o·v)` | 0.07 |

The ladder should build `vvvv` tiles rather than contract through the
auxiliary index. A factorized ladder that never forms a `v⁴` tile costs
`2·naux/v` times the dense contraction, which is about 15× here because
`naux > v`.

The loops need restructuring:

- Split the fused loop in `mp3_amplitude_3mesh_efficient` into one pass per
  term. Each pass runs over the term's tile labels and accumulates into
  `t_mp3` before the denominators are applied. The ordinary and 2a kernels
  already accumulate this way.
- Bound tile batches by `max_memory`. For example, `vvvv` for fixed
  `(ika, ikb)` and all `ikc` takes `K·v⁴`, which is 0.1 GB at K=27.

After Stage B, the remaining storage is:

- the factors,
- `t_mp3`,
- the auxiliary MP2 amplitudes,
- the two `oovv` assignments the energy reads, which can also be
  regenerated.

The auxiliary amplitudes then dominate for 2b, 3a, and 3b: 14.6–19.5 GB at
K=27 and 195–260 GB at K=64. Regenerating an amplitude tile at every use
costs about `naux/(o·v) ≈ 2` times the contraction that consumes it. Store
them by default, and regenerate under a memory cap only. A fully streaming,
energy-only path would also need amplitude blocking; leave that for a later
change.

### 3.6 Per-variant notes

| Variant | Notes |
| --- | --- |
| ordinary | Read `self.with_df`, falling back to `_scf.with_df`. One mesh; `vo` comes from `ov` |
| 2a | Map mesh 1 to the occupied mesh and mesh 0 to the virtual mesh. Keep the unshifted Madelung default (`madelung=True`) |
| 2b | Same-mesh `ov` families include `q = 0`, which is harmless by orthogonality. Keep the shifted Madelung default |
| 3a, 3b | Cross-mesh families only; keep `madelung=False` as the default |
| Submesh mode | The union is the complete SCF mesh, so the BvK mesh equals the SCF mesh |
| Extension mode | Keep the FFTDF band step. Build direct RSDF on `mp.kpts`. The BvK supercell is the fine grid (§2.1) |

The Madelung constants depend only on the meshes, so finite-size corrections
are unaffected by how DF is done.

## 4. Validation

1. **Factor tests (lightweight).** Turn the §2.2 probe into a test.
   Reconstruct all five block types against stored RSDF on a two-mesh and a
   three-mesh union, with complex k-points, reversed pairs, and BvK on and
   off. Momentum maps come from `make_kpts_helper`.
2. **Stage A (slow).** Compare E2, E3, and `t2` for all five solvers in both
   modes against the stored-RSDF dense path, with matching auxiliary basis
   and precision, to 1e-9. Compare against GDF only loosely, with a measured
   tolerance, because the DF construction differs.
3. **Existing invariances.** Check that 2a matches its generalized
   configuration, that 3b is invariant under occupied-mesh relabeling, that
   the energy is invariant under orbital phases, and that a frozen-core
   active space works.
4. **Stage B.** Check that the energy does not depend on tile batch size
   and that no global `vvvv` array is allocated.
5. **Direct path.** Assert that `with_df._cderi is None` after `kernel()`.

## 5. Reuse in staggered CCD and GW

- **Two-mesh CCD.** `KRCCD_2mesh` sets `direct = True`, and
  `add_vvvv_stagger` already has an `eris.Lvv` branch that builds `<ab|cd>`
  tiles from `Lvv`, which is the Stage B pattern. However, `_ERIS_2mesh`
  never sets `Lvv` and always allocates `vvvv`. Setting
  `eris.Lvv = factors.get('vv', 0, 0)` would activate the branch, in the
  `(nkpts_ov, nkpts_ov, naux, nvir, nvir)` layout its comment expects.
  Check its `1/nkpts_ov` normalization and its diagonal Madelung shift.
- **Three-mesh CCD.** It transforms full-`nmo` quartets through GDF
  `fao2mo`, so the Stage A route applies directly.
- **Staggered GW.** `_load_lmn` reads AO factors through `sr_loop` and
  transforms them for each transfer, which is exactly what the provider
  returns. Swap in the provider while keeping `right_index`, the
  per-transfer auxiliary ordering, and the positive-metric check. GW
  already contracts three-index factors, so it needs no Stage B. Setting
  `direct=True` while keeping `sr_loop` fails, for the reason given in §1.2.

Sources: [two-mesh CCD](../fsec/staggered_mesh/cc/kccd_rhf_2mesh.py),
[three-mesh CCD](../fsec/staggered_mesh/cc/kccd_rhf_3mesh.py), and
[staggered GW](../fsec/staggered_mesh/gw/krgw_ac_stagger.py).

## 6. Open items

- **BvK cost.** Benchmark the cost of BvK on incomplete extension unions,
  where the supercell is `nmesh^(d-1)` times the union, against
  `use_bvk=False`.
- **Fork memory estimate.** Fix or work around it (§2.1) before measuring
  `ao2mo_e2` performance.
- **Extensions.** Decide whether `semidirect` and `ksym='s1'` are worth
  supporting after Stage A.
- **Fork stability.** The fork API may change. Pin a commit in a helper
  constant and fail with a clear `ImportError` otherwise.

The PySCF sources inspected were `pyscf/pbc/df/rsdf.py`, `df.py`,
`df_ao2mo.py`, `rsdf_direct_ao2mo.py`, and `rsdf_direct_helper.py`,
`pyscf/pbc/mp/kmp2_direct.py`, and `pyscf/pbc/lib/kpts_helper.py`.
