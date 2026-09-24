# FSEC

FSEC provides finite-size error corrections for periodic electronic-structure
calculations. It includes singularity-subtraction methods for exact exchange
(ExxSS) and MP2 (MP2SS) correlation energies and is designed to work with periodic
[PySCF](https://pyscf.org/) calculations.

## Installation

FSEC requires Python 3.9 or newer. Clone the repository and install it with
`pip`:

```bash
git clone https://github.com/stephen-quiton/fsec.git
cd fsec
python -m pip install .
```

For development, use an editable installation instead:

```bash
python -m pip install -e .
```

## ExxSS quick start

The following example calculates periodic Hartree–Fock exchange for a helium
atom in a cubic 4 Å cell and applies the default ExxSS correction. It uses the
GTH-DZVP basis, a 100 hartree kinetic-energy cutoff, and a 2×2×2 k-point
mesh.

```python
import numpy as np
from pyscf.pbc import df, gto, scf

from fsec.singularity_subtraction import ExxSS


cell = gto.Cell()
cell.unit = "Angstrom"
cell.atom = "He 0.0 0.0 0.0"
cell.a = np.eye(3) * 4.0
cell.basis = "gth-dzvp"
cell.ke_cutoff = 100  # hartree
cell.precision = 1e-8
cell.build()

kpts = cell.make_kpts(
    [2, 2, 2],
    wrap_around=True,
    with_gamma_point=True,
)

kmf = scf.KRHF(cell, kpts)
kmf.exxdiv = "ewald"
kmf.with_df = df.GDF(cell, kpts).build()
kmf.kernel()

if not kmf.converged:
    raise RuntimeError("Hartree–Fock calculation did not converge")

exxss = ExxSS(kmf)
exxss.compute_correction()

print("Uncorrected exact exchange (hartree):", exxss.Ek_uncorr)
print("ExxSS correction (hartree):", exxss.correction)
print("Corrected exact exchange (hartree):", exxss.Ek_ss)
```

`compute_correction()` fits an auxiliary function to the exchange structure factor near the Coulomb singularity and stores the uncorrected exchange energy, correction, and corrected exchange energy in `Ek_uncorr`, `correction`, and `Ek_ss`, respectively.

## STC MP2 exchange

`fsec.vcut.KMP2_STC` computes only the exchange contribution (`exi`) to
restricted periodic MP2. It uses the smoothed truncated Coulomb (STC) kernel
in both ERI factors, including the amplitude numerator, while retaining the
supplied HF orbitals and orbital energies. The input SCF object is unchanged.
This requires the custom PySCF `pyscf.pbc.df.rsdf_stc` implementation available
in the `fsec-312` environment.

```python
from fsec.vcut import KMP2_STC

# kmf is an already converged KRHF calculation on a full regular k-point mesh.
stc_mp = KMP2_STC(kmf, eta=4.0, exxdiv="vcut_ws", rc_type="ws")
e_exchange, _ = stc_mp.kernel()  # hartree per primitive cell
print(stc_mp.e_corr_exchange)
```

The default uses Wigner–Seitz truncation. For spherical truncation, set both
`exxdiv="vcut_sph"` and `rc_type="sph"`. The smoothing parameter is
`eta = omega_stc * R_in`, with the cutoff length determined by the cell and
k-point mesh. No extra Ewald correction is added to the STC integrals.
The reference's existing orbital-energy convention is retained.

The calculator supports 3D closed-shell KRHF references, shifted regular
meshes, frozen orbitals, and stored RSDF factors. Symmetry-reduced meshes,
unrestricted references, and direct/semidirect STC are unsupported.
`auxbasis` defaults to the reference DF basis when available. Numerical
builder settings such as `stc_mp.with_df.mesh_compact` and `mesh_j2c` may be
set before the first call to `kernel`. Create a new calculator when changing
the cutoff, `eta`, or DF settings after a calculation, to avoid reusing old
integrals.

Use `kernel(with_t2=True)` to retain the padded amplitudes with shape
`(nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir)` and indexing
`(ki, kj, ka, i, j, a, b)`. Amplitudes are not stored by default.
This exchange energy is neither the full MP2 correlation energy nor its full
same-spin component. For ordinary PySCF KMP2, the comparable exchange term
is `e_corr_ss - e_corr_os`.

[examples/mp2_stc_exchange.py](examples/mp2_stc_exchange.py) runs an ordinary
RSDF reference and compares ordinary and STC MP2 exchange on the same orbitals.
For the local `fsec-312` installation, after activating the environment and
installing FSEC, run:

```bash
LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python examples/mp2_stc_exchange.py
```

## References

- S. J. Quiton, J. D. F. Pottecher, X. Xing, M. Head-Gordon, and L. Lin,
  “Optimized Auxiliary Functions for Robust Mitigation of Finite-Size Errors
  in Periodic Hybrid Density Functional Theory,” *Journal of Chemical Theory
  and Computation* **21**, 8863–8875 (2025).
  [doi:10.1021/acs.jctc.5c01066](https://doi.org/10.1021/acs.jctc.5c01066)

- S. J. Quiton, J. D. F. Pottecher, M. Head-Gordon, and L. Lin, “Reduction of
  finite-size effects for second-order Møller–Plesset perturbation theory with
  singularity subtraction,” arXiv:2605.12727 (2026).
  [doi:10.48550/arXiv.2605.12727](https://doi.org/10.48550/arXiv.2605.12727)
