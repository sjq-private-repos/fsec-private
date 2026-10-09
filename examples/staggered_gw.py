"""Run restricted G0W0 on target and half-shifted k-point meshes."""

import numpy as np
from pyscf.pbc import dft, gto

from fsec.staggered_mesh.gw import (
    KRGWAC_stagger_extend,
    KRGWAC_stagger_submesh,
)


cell = gto.Cell()
cell.build(
    unit='Bohr',
    a=np.eye(3) * 6.0,
    atom='H 3 3 2.1; H 3 3 3.9',
    dimension=3,
    verbose=4,
    pseudo='gth-pade',
    basis='gth-szv',
    precision=1e-10,
)

# The 2x2x2 reference mesh contains both the gamma target point and its
# half-shifted partner for a 1x1x1 staggered mesh.
mf = dft.KRKS(cell, cell.make_kpts([2, 2, 2]))
mf.xc = 'pbe'
mf.conv_tol = 1e-10
mf.kernel()

for gw_type in (KRGWAC_stagger_submesh, KRGWAC_stagger_extend):
    gw = gw_type(mf, [1, 1, 1])
    gw.qpe_linearized = True
    gw.nw = 100
    gw.ac_iw_cutoff = 5.0
    gw.kernel(orbs=[0, 1], kptlist=[0])
    print(gw_type.__name__, gw.mo_energy[0])
