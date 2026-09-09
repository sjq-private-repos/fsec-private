#!/usr/bin/env python
"""Small Gaussian-corrected periodic G0W0 example."""

import numpy as np
from pyscf.pbc import df, gto, scf

from fsec.singularity_subtraction import G0W0SS


cell = gto.Cell()
cell.unit = "Bohr"
cell.atom = "H 0 0 0; H 1.8 0 0"
cell.a = np.eye(3) * 6.0
cell.basis = "gth-szv"
cell.pseudo = "gth-hf"
cell.ke_cutoff = 100
cell.precision = 1e-8
cell.build()

kpts = cell.make_kpts([1, 1, 2], wrap_around=True, with_gamma_point=True)
kmf = scf.KRHF(cell, kpts).density_fit()
kmf.with_df = df.GDF(cell, kpts).build()
kmf.kernel()
if not kmf.converged:
    raise RuntimeError("Hartree-Fock calculation did not converge")

gw = G0W0SS(kmf)
gw.kernel(orbs=[0, 1], kptlist=[0])
print("Gaussian width:", gw.gaussian_coefficients.sigma)
print("QP energies:", gw.mo_energy[0, [0, 1]])
