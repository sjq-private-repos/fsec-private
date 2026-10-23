# FSEC

FSEC provides finite-size error corrections for periodic electronic-structure
calculations. It includes singularity-subtraction methods for exact exchange
(ExxSS) and MP2 (MP2SS) correlation energies and is designed to work with periodic
[PySCF](https://pyscf.org/) calculations.

Staggered mesh methods for restricted periodic MP3, CCD, and G0W0 are available
under `fsec.staggered_mesh.mp`, `fsec.staggered_mesh.cc`, and
`fsec.staggered_mesh.gw`. See the [staggered mesh guide](docs/staggered_mesh.md)
for examples, API migration notes, and numerical validation, including the
unresolved historical H₂ GW reference discrepancy.

Standard periodic G0W0-AC with lattice-aware q → 0 corrections is available as
`fsec.standard.KRGWAC_ewald`: Gaussian auxiliary functions (FHI-gap, the
default), the Madelung constant, or PySCF's spherical integration, with an
optional Lebedev average of the k·p dielectric head and wings. See the
[method guide](docs/krgwac_ewald.md) for the supported systems, correction
conventions, and comparisons with uncorrected calculations.

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
