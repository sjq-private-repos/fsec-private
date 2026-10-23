"""Compare Ewald, spherical, and uncorrected periodic G0W0-AC results."""

import numpy as np
from pyscf.pbc import df, dft, gto

from fsec.standard import krgwac_ewald


cell = gto.Cell()
cell.build(
    unit='Bohr',
    a=np.eye(3) * 7.0,
    atom='H 2.1 3.0 3.0; H 4.9 3.0 3.0',
    dimension=3,
    pseudo='gth-pade',
    basis='gth-szv',
    precision=1e-8,
    verbose=4,
)

mf = dft.KRKS(cell, cell.make_kpts([2, 1, 1]))
mf.with_df = df.GDF(cell, mf.kpts)
mf.xc = 'pbe'
mf.conv_tol = 1e-9
mf.kernel()
if not mf.converged:
    raise RuntimeError('The reference SCF calculation did not converge')
# SCF builds diagonal J blocks; KRGWAC also needs off-diagonal k-point pairs.
mf.with_df.build(j_only=False)

settings = (
    ('ewald', 'ewald', True),
    ('spherical', 'spherical', True),
    ('uncorrected', 'ewald', False),
)
for label, method, enabled in settings:
    gw = krgwac_ewald.KRGWAC(mf, frozen=None)
    gw.fc_method = method
    gw.fc = enabled
    gw.nw = 40
    gw.ac_iw_cutoff = None
    gw.ac_pade_npts = 12
    gw.kernel(orbs=[0, 1], kptlist=[0])
    print(f'{label:12s} QP energies at k=0 (Ha): {gw.mo_energy[0]}')
