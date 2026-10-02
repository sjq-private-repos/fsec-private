
## sTC and hybrid MP2

`fsec.vcut.KMP2_STC` computes the exchange contribution (`exi`) to restricted
periodic MP2 by default. Set `kernel(with_direct=True)` to return the full sTC
correlation energy, including the direct contribution. Each component
uses its interaction in both ERI factors, including the amplitude numerator.
Both modes retain the supplied HF orbitals and orbital energies, and leave the
input SCF object unchanged. This requires the custom PySCF
`pyscf.pbc.df.rsdf_stc` implementation available in the `fsec-312` environment.

```python
from fsec.vcut import KMP2_STC

# kmf is an already converged KRHF calculation on a full regular k-point mesh.
stc_mp = KMP2_STC(kmf, eta=4.0, exxdiv="vcut_ws", rc_type="ws")
e_exchange, _ = stc_mp.kernel()  # hartree per primitive cell
print(stc_mp.e_corr_exchange)
e_stc_both, _ = stc_mp.kernel(with_direct=True)
print(stc_mp.e_corr_direct, stc_mp.e_corr_exchange)
```

`fsec.vcut.KMP2_HYBRID` combines a bare RSDF direct term with an sTC exchange
term using independent builders and the same orbitals and denominators:

```python
from fsec.vcut import KMP2_HYBRID

hybrid = KMP2_HYBRID(kmf, eta=4.0, exxdiv="vcut_ws", rc_type="ws")
e_hybrid, t2 = hybrid.kernel(with_t2=True)
print(hybrid.e_corr_direct, hybrid.e_corr_exchange, e_hybrid)
print(t2["direct"].shape, t2["exchange"].shape)
```

`KMP2_STC` stores its latest returned energy in `e_corr`; `e_corr_direct` is
`None` whenever the latest call does not request the direct term. The hybrid
always computes and stores both components. Amplitudes are retained only with
`with_t2=True`; the STC class
returns one ndarray, while the hybrid returns a dictionary with `direct` and
`exchange` arrays. The hybrid's bare builder is `with_df_direct`; it and the
inherited sTC `with_df` builder can be configured independently before their
first build.

## On-demand direct sTC MP2

For lower memory use, `KMP2_STC_DIRECT` and `KMP2_HYBRID_DIRECT` transform
occupied orbitals in blocks and generate sTC factors on demand. They use the
same constructors, kernel options, component energies, and amplitude shapes as
their stored counterparts. They do not write AO three-center integrals to
CDERI storage:

```python
from fsec.vcut import KMP2_HYBRID_DIRECT, KMP2_STC_DIRECT

stc_direct = KMP2_STC_DIRECT(kmf, eta=4.0)
stc_direct.max_memory = 2000  # MB; used for automatic block selection
e_exchange, _ = stc_direct.kernel()
e_full_stc, _ = stc_direct.kernel(with_direct=True)

hybrid_direct = KMP2_HYBRID_DIRECT(
    kmf, eta=4.0, rsdf_occ_block_size=2
)
e_hybrid, _ = hybrid_direct.kernel()
```

The new module also exposes `KMP2_STC`, `KMP2_HYBRID`, and `KMP2` (the STC
alias) through `fsec.vcut.kmp2_stc_direct`. Existing package-level imports
continue to select the stored calculators. The direct route requires the
custom fork's `pyscf.pbc.mp.kmp2_direct` and direct RSDF helpers.

The `_DIRECT` suffix selects on-demand integral generation. Independently,
`kernel(with_direct=True)` includes the physical sTC MP2 direct contribution
in addition to exchange. `KMP2_HYBRID_DIRECT` always combines bare RSDF direct with sTC
exchange, matching `KMP2_HYBRID`. The optional `rsdf_occ_block_size` sets the
padded occupied block size and is capped at `nocc`. Its default, `None`, picks
the largest block estimated to fit within 80% of available memory. An explicit
block that does not fit, or a calculation for which even block size one does
not fit, raises `MemoryError`. The estimate includes native transform and
metric workspaces; it is not a strict process memory limit.

Keep `with_t2=False` to avoid retaining the full padded amplitudes. With
`with_t2=True`, their storage still scales as
`nkpts**3 * nocc**2 * nvir**2`; the hybrid retains one array for each
interaction. Blocking lowers factor and contraction workspace use at the cost
of repeating transforms for occupied blocks. The direct implementation
requires Cholesky-decomposable RSDF metrics and does not support eigendecomposed
metrics or semidirect mode. Set numerical builder options before the first
kernel call, and create a new calculator after changing those settings.

The default uses Wigner–Seitz truncation. For spherical truncation, set both
`exxdiv="vcut_sph"` and `rc_type="sph"`. The smoothing parameter is
`eta = omega_stc * R_in`, with the cutoff length determined by the cell and
k-point mesh. No extra Ewald correction is added to the STC integrals.
The reference's existing orbital-energy convention is retained.

All calculators support 3D closed-shell KRHF references, shifted regular
meshes, and frozen orbitals. Symmetry-reduced meshes, unrestricted references,
and semidirect DF are unsupported. The stored calculators require stored
RSDF factors; the direct calculators require Cholesky-decomposable metrics.
`auxbasis` defaults to the reference DF basis when available. Numerical
builder settings such as `stc_mp.with_df.mesh_compact` and `mesh_j2c` may be
set before the first call to `kernel`. Create a new calculator when changing
the cutoff, `eta`, or DF settings after a calculation, to avoid reusing old
integrals.

Use `kernel(with_t2=True)` to retain the padded amplitudes with shape
`(nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir)` and indexing
`(ki, kj, ka, i, j, a, b)`. Amplitudes are not stored by default.
The default exchange-only result is neither the full MP2 correlation energy
nor its full same-spin component. For ordinary PySCF KMP2, the comparable
exchange term is `e_corr_ss - e_corr_os`.

[examples/mp2_stc_exchange.py](examples/mp2_stc_exchange.py) runs an ordinary
RSDF reference and compares ordinary, sTC, and hybrid correlation components
on the same orbitals.
For the local `fsec-312` installation, after activating the environment and
installing FSEC, run:

```bash
LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python examples/mp2_stc_exchange.py
```
