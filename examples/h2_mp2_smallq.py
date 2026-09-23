"""Compute one MP2 small-q point using RSDF-sTC virtual bands.

Requires the PySCF ``rsdf_stc`` extension. The original occupied orbitals and
energies are retained; the extra point uses ordinary GDF correlation ERIs.
"""

import numpy as np
from pyscf.pbc import gto, mp, scf

from fsec.singularity_subtraction.structure_factor import (
    MP2SmallQ,
    MP2SmallQOptions,
)


cell = gto.M(
    unit="Bohr",
    a=np.eye(3) * 6.0,
    atom="H 0 0 0; H 0 0 1.8",
    basis="gth-szv",
    pseudo="gth-hf",
    mesh=np.array([17, 17, 17]),
    precision=1e-8,
    verbose=0,
)
kpts = cell.make_kpts([1, 1, 2], wrap_around=True)
kmf = scf.KRHF(cell, kpts, exxdiv="ewald").density_fit()
kmf.conv_tol = 1e-10
kmf.kernel()
if not kmf.converged:
    raise RuntimeError("Reference KRHF calculation did not converge")

# KMP2 supplies the occupied data and active space; a full t2 is unnecessary.
kmp = mp.KMP2(kmf)
options = MP2SmallQOptions(relative_shift=(0.0, 0.0, 0.1))
result = MP2SmallQ(kmf, kmp, options=options).kernel()

print("q (inverse Bohr):", result.qprime)
print("Direct structure factor:", result.sq_direct)
print("Fourth-order structure factor:", result.sq_q4)

# For a full MP2SS correction, run kmp.kernel(with_t2=True) and pass its t2
# together with MP2SSOptions(smallq=options) to MP2SS(kmf, kmp, t2=t2, ...).
