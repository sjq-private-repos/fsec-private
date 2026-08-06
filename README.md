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

## MP2SS correlation variants

`MP2SSOptions` keeps ordinary MP2 as the default and accepts case-insensitive
aliases for SOS-MP2, SCS-MP2, SCS-MI-MP2, kappa-MP2, sigma-MP2, and
sigma-squared-MP2. Regularized methods use the defaults
`kappa=1.1 Eh^-1`, `sigma=0.7 Eh^-1`, and `sigma2=0.4 Eh^-2`; pass a
non-negative `regularization_strength` to override one.

```python
from fsec.singularity_subtraction import MP2SS, MP2SSOptions

options = MP2SSOptions(
    correlation_method="kappa-MP2",
    regularization_strength=1.1,
)
mp2ss = MP2SS(kmf, kmp, t2=t2, options=options)
mp2ss.compute_correction()
print(mp2ss.e_corr_uncorrected, mp2ss.e_corr_corrected)
```

The spin-only presets use `(cOS, cSS) = (1.3, 0)`, `(1.2, 1/3)`, and
`(0.40, 1.29)` for SOS, SCS, and SCS-MI MP2, respectively. These options
remain restricted to gapped closed-shell references.

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
