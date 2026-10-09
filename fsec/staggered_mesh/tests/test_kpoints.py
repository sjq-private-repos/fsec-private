"""Momentum conservation for reordered unions of staggered meshes."""

import numpy as np
from pyscf.pbc import gto

from fsec.staggered_mesh._kpoints import make_kpts_helper


def test_concatenated_mesh_momentum_conservation():
    """A mesh union must conserve physical momentum in its actual ordering."""
    cell = gto.Cell()
    cell.a = [[6., 0., 0.], [1., 6., 0.], [0., 1., 6.]]
    cell.atom = 'He 0 0 0'
    cell.basis = 'sto-3g'
    cell.verbose = 0
    cell.build()
    # Concatenate the even and odd submeshes of a six-point uniform grid.
    kpts = cell.make_kpts([1, 1, 6])[[0, 2, 4, 1, 3, 5]]
    helper = make_kpts_helper(cell, kpts)
    scaled = cell.get_scaled_kpts(kpts)
    residual = (scaled[:, None, None, :] - scaled[None, :, None, :]
                + scaled[None, None, :, :] - scaled[helper.kconserv])
    np.testing.assert_allclose(residual - np.rint(residual), 0, atol=1e-9)
