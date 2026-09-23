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

## MP2 small-q fitting point

For a converged restricted `kmf` and its ordinary-Coulomb `kmp`, enable one
additional direct fitting point with:

```python
from fsec.singularity_subtraction import MP2SS, MP2SSOptions
from fsec.singularity_subtraction.structure_factor import (
    MP2SmallQ, MP2SmallQOptions,
)

smallq = MP2SmallQOptions(relative_shift=(0.0, 0.0, 0.1), eta=4.0, cutoff="ws")
_, t2 = kmp.kernel(with_t2=True)
calculation = MP2SS(kmf, kmp, t2=t2, options=MP2SSOptions(smallq=smallq))
correction = calculation.compute_correction()

# The extra point can also be calculated independently of MP2SS.
point = MP2SmallQ(kmf, kmp, options=smallq).kernel()
print(point.qprime, point.sq_direct, point.sq_q4)
```

See [the runnable H2 example](examples/h2_mp2_smallq.py). The PySCF installation
must provide `pyscf.pbc.df.rsdf_stc`, as the local `fsec-312` environment does.
This extension is checked when generating the small-q bands; ordinary MP2SS
remains usable without it.

The shift is a fraction of the original k-mesh spacing:
`qprime = cell.get_abs_kpts(relative_shift / nks)`, in inverse Bohr. It must be
nonzero, with each component in `[-0.5, 0.5]`. The default shift is
`(0.5, 0.5, 0.5)`. The calculation supports three-dimensional restricted
systems on complete regular k meshes. `cutoff="ws"` selects Wigner–Seitz
truncation; `"sph"` selects spherical truncation. The dimensionless smoothing
parameter defaults to `eta=4.0`; PySCF derives the range-separation parameter
from `eta` and the cutoff radius of the original SCF mesh.

Only the shifted virtual bands are recomputed, using the original SCF density,
smoothed truncated Coulomb exchange, and ordinary-Coulomb Hartree. Occupied
coefficients and energies come from KMP2, including supplied energy shifts.
These occupied and virtual spaces can therefore come from different Fock
operators, and their overlap need not vanish as q approaches zero.

The standalone module uses separate ordinary GDF correlation integrals and
explicit energy denominators. It retains Becke/uniform pair-density quadrature
and accepts the existing grid controls. MP2SS forwards its grid settings and
uses the extra point only for fitting. Correlation integral storage uses the
full ordinary GDF build in this first version; the main MP2SS Laplace and
amplitude-storage settings do not control the small-q calculation.

`MP2SSOptions.smallq=None` disables the feature. The experimental
`smallq_band_df` and `smallq_band_exxdiv` options have been replaced by `smallq`.

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
