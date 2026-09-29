
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

The default uses Wigner–Seitz truncation. For spherical truncation, set both
`exxdiv="vcut_sph"` and `rc_type="sph"`. The smoothing parameter is
`eta = omega_stc * R_in`, with the cutoff length determined by the cell and
k-point mesh. No extra Ewald correction is added to the STC integrals.
The reference's existing orbital-energy convention is retained.

Both calculators support 3D closed-shell KRHF references, shifted regular
meshes, frozen orbitals, and stored RSDF factors. Symmetry-reduced meshes,
unrestricted references, and integral-direct/semidirect DF modes are unsupported.
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