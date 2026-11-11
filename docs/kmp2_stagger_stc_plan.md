# fsec `KMP2_stagger` with GDF+sTC band orbitals

## Context

PySCF's `KMP2_stagger(flag_submesh=False)` always gets the orbitals and energies on
the shifted mesh from a fixed-density `get_bands` call with a temporary
`FFTDF` and hard `vcut_sph`, even when the SCF used GDF. That makes all-electron
cells impractical and ties the band Fock to FFT. It also rebuilds a default-aux
GDF for correlation and ignores the reference `auxbasis`.

The fork (`pyscf-sjq`) already has stored sTC exchange in
`pyscf/pbc/df/rsdf_stc.py` (`RSGDF_STC`, `density_fit`). Goal: a copy of
`kmp2_stagger` in fsec that behaves identically by default, and can also build
the band Fock with GDF+sTC.

The band Fock at each band point $\mathbf{k}$, with the density $D$ fixed on the
SCF mesh, becomes

$$
F^{\mathbf{k}}_{\mu\nu}[D] = h^{\mathbf{k}}_{\mu\nu} + J^{\mathbf{k}}_{\mu\nu}[D]
  - \tfrac{1}{2} K^{\mathrm{sTC},\mathbf{k}}_{\mu\nu}[D]
$$

where $J$ is ordinary GDF and $K^{\mathrm{sTC}}$ comes from `RSGDF_STC`. The
correlation integrals stay bare (ordinary GDF).

## Deviation found during implementation

The fsec copy reads (L|pq) with `sr_loop` instead of `CDERIArray`, so the FFTDF
mode is identical to PySCF only without linear dependency. `CDERIArray` uses the
first k-pair's `naux` for every pair, which corrupts the integrals when linear
dependency makes `naux` depend on q. See `docs/staggered_mesh.md`.

## Decisions (agreed)

- **Reference:** both supported. An SCF from `rsdf_stc.density_fit` is
  auto-detected and its `eta`/`exxdiv` reused; an ordinary GDF/RSGDF SCF opts in.
- **Default cutoff for a non-sTC reference:** `vcut_ws`, `eta=4.0`.
- **Aux basis in sTC mode:** inherited from the reference by sTC $K$, band $J$,
  and correlation. The FFTDF mode stays identical to PySCF.

## Feasibility already confirmed (H2/gth-szv, 2x2x2, `pyscf-sjq` `dfddc7c56`)

- `RSGDF_STC` builds on the half-shifted union and keeps the cutoff on the SCF
  mesh (`_stc_kmesh=[2,2,2]`).
- Against `FFTDF_STC` at shifted points, with a 122-function aux at `eta=4`:
  $|\Delta K|$ = 3.4e-6 (`vcut_ws`), 3.8e-6 (`vcut_sph`); $|\Delta J|$ = 2.6e-7.
  That matches bare GDF vs FFTDF on the SCF mesh.
- With the default 12-function aux, $|\Delta J|$ is 4.8e-3, which is why the aux
  must be inherited.
- `FFTDF_STC` tends to hard truncation with `eta`: $|\Delta K|$ 2.2e-3 (eta 4) to
  2.2e-5 (eta 40) for `vcut_ws`. Off-mesh WS q-vectors go through `vq_cache`.
- PySCF `KptsHelper` agrees with fsec `make_kpts_helper` on all union meshes
  tried, so the copy needs no momentum-map fix.
- `KMP3_2mesh.e_corr_mp2` equals `kmp2_stagger` to 10 decimals in both modes.

## Implementation

### New: `fsec/staggered_mesh/mp/kmp2_stagger.py`

Copy `pyscf/pbc/mp/kmp2_stagger.py` (keep the PySCF licence header and Xin Xing
attribution, as `fsec/vcut/kmp2_stc.py` does; drop the `__main__` block). Keep
`kernel`, `_init_mp_df_eris_stagger`, mesh construction, and index maps as they
are, including the all-axes half shift and PySCF's even-mesh check. Use
`make_kpts_helper` from `fsec/staggered_mesh/_kpoints.py`.

Constructor (first three arguments unchanged):

```python
KMP2_stagger(mf, frozen=None, flag_submesh=False, band_df=None,
             eta=None, exxdiv=None, auxbasis=None)
```

- `band_df`: `None` auto-detects (`'stc'` for an `RSGDF_STC` reference, otherwise
  `'fftdf'`). `'fftdf'` is the unchanged PySCF path. `'stc'` is the new path.
- `eta`, `exxdiv`: explicit value, else the sTC reference's, else `4.0` and
  `'vcut_ws'`. `rc_type` is derived from `exxdiv` with `_validate_cutoff`, since
  the pairing is one-to-one.
- `auxbasis`: explicit value, else `mf.with_df.auxbasis`. Unused by `'fftdf'`.

Import `rsdf_stc` lazily inside the sTC path, so `fsec.staggered_mesh.mp` still
imports on a stock PySCF.

New helper `_get_bands_stc(mp, mf, kpts_occ, kpts)`, used when
`band_df='stc'` and `flag_submesh=False`:

1. Build one ordinary `df.GDF(cell, kpts)` on the union with the resolved aux,
   and store it as `self.with_df`.
2. Build a fresh `RSGDF_STC(cell, mf.kpts, eta, exxdiv, rc_type)` with the same
   aux. `kpts` must be `mf.kpts`: `_scf_kpts` defines the cutoff, and the union
   is not a regular mesh.
3. Set `stc_df.with_df_j = self.with_df`, so J reuses the union factors and no
   second bare GDF is built.
4. Call `stc_df.build(j_only=False, kpts_band=kpts_occ)` once up front, so
   `get_k_kpts` does not trigger a second build.
5. Run `mf.get_bands(kpts)` inside
   `lib.temporary_env(mf, exxdiv=None, with_df=stc_df)`, then `get_occ`.

The reference `with_df` is never touched, so its `_cderi` is not rewritten.

Change `_init_mp_df_eris_stagger`: if `mp.with_df` is set, use it; otherwise run
the existing PySCF logic. Take `naux` in the memory check from `mp.with_df` when
it is set.

`flag_submesh=True` with an sTC reference: orbitals come from `mf`, and
`self.with_df` is a fresh ordinary `df.GDF(cell, mf.kpts)` with the inherited
aux. This is required. A straight copy would read the sTC `_cderi` through
`CDERIArray` and silently give sTC correlation, and the reference's `with_df_j`
cannot be reused because `get_j_kpts` builds it with `j_only=True` (diagonal
k-pairs only).

Errors, only on the new path:

- `band_df='stc'` with `flag_submesh=True` and a non-sTC reference: `ValueError`,
  since there is no band step.
- `band_df='stc'` with a non-GDF reference or a KRKS reference: `TypeError`.
- `cell.dimension != 3`: `NotImplementedError` (already raised by the band path).

`dump_flags` logs `band_df`, and in sTC mode the cutoff, `eta`, and `auxbasis`.

### Update

- `fsec/staggered_mesh/mp/__init__.py`: export `KMP2_stagger`.
- `fsec/staggered_mesh/__init__.py`: mention MP2 in the docstring.
- `docs/staggered_mesh.md`: short section on the options and the fork dependency.
  This file already has uncommitted edits; append only.

### New example: `examples/staggered_mp2_stc.py`

Runs the same cell with ordinary GDF + FFTDF bands, ordinary GDF + sTC bands, and
an `rsdf_stc.density_fit` reference, and prints the energies.

### New tests: `fsec/staggered_mesh/mp/tests/test_kmp2_stagger.py`

`pytest.mark.slow`, H2/gth-szv at 2x2x2, brief with docstrings, following
`test_kmp3.py`.

1. Default path matches PySCF `KMP2_stagger` for both `flag_submesh` values
   (`atol=1e-10`).
2. sTC bands match `get_bands` with `FFTDF_STC` for `vcut_ws` and `vcut_sph`,
   using a large even-tempered aux (`atol` about 1e-4), plus one pinned sTC energy
   as a regression value.
3. An sTC reference is auto-detected: `eta`/`exxdiv` are reused, `auxbasis` is
   inherited, and `mf.with_df._cderi` is unchanged.
4. Submesh with an sTC reference uses bare integrals: the energy matches PySCF
   `KMP2_stagger` run with the same orbitals on an ordinary GDF of the same aux.

## Out of scope

- Other solvers (`KMP3_*`, KCCD, GW), which share the same FFTDF band pattern and
  could adopt this helper later.
- Changes to `pyscf-sjq` or to PySCF's own `kmp2_stagger`.
- `fsec/vcut` (unmerged `mp2-exchange-vcut` branch); only the fork's `RSGDF_STC`
  is needed.
- Existing behaviour outside sTC mode, including MDF in submesh mode, stays the
  same as PySCF.
- sTC correlation integrals, RSDF-direct, and restricting the union build to
  occupied–virtual pairs. The sTC build stores all union k-pairs, so disk use
  scales like the existing extension GDF.

## Verification

Use `fsec-312` with `PYTHONPATH` including `pyscf-sjq` (non-interactive shells
do not source `.bashrc`, so it must be set explicitly).

```bash
python -m pytest -m slow fsec/staggered_mesh/mp/tests/test_kmp2_stagger.py
```

```bash
python -m pytest fsec/staggered_mesh
```

```bash
python examples/staggered_mp2_stc.py
```

```bash
python -m pytest
```

Also check `KMP3_2mesh.e_corr_mp2` still equals the new class on the default path,
and that `stc_df` is built exactly once. As a science check, run diamond/gth-szv
2x2x2 and report the sTC energy against the FFTDF+`vcut_sph` value
(−0.01268 Ha from PySCF's test, `frozen=[0,1,2]`).
