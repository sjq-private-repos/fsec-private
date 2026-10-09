"""K-point helpers for unions of shifted Monkhorst-Pack meshes."""

import numpy as np
from pyscf.pbc.lib import kpts_helper


def make_kpts_helper(cell, kpts):
    """Build a momentum map without assuming canonical Monkhorst-Pack order.

    PySCF's fast momentum map assumes a single mesh in ``make_kpts`` order.
    Staggered calculations concatenate several shifted meshes, so compute the
    fourth point from fractional coordinates and then build the usual symmetry
    map from that exact conservation table.
    """
    kpts = np.asarray(kpts)
    scaled = np.mod(cell.get_scaled_kpts(kpts), 1.0)
    scaled = np.mod(np.round(scaled, 10), 1.0)
    by_coordinate = {}
    for index, point in enumerate(scaled):
        key = tuple(point)
        if key in by_coordinate:
            raise ValueError('The staggered k-point meshes contain duplicate points')
        by_coordinate[key] = index

    nkpts = len(kpts)
    kconserv = np.empty((nkpts, nkpts, nkpts), dtype=int)
    for ki in range(nkpts):
        for kj in range(nkpts):
            for kk in range(nkpts):
                target = np.mod(scaled[ki] - scaled[kj] + scaled[kk], 1.0)
                key = tuple(np.mod(np.round(target, 10), 1.0))
                index = by_coordinate.get(key)
                if index is None:
                    delta = scaled - target
                    delta -= np.rint(delta)
                    matches = np.flatnonzero(np.max(np.abs(delta), axis=1) < 1e-8)
                    if len(matches) != 1:
                        raise ValueError(
                            'The supplied staggered k-points are not closed under '
                            'momentum conservation'
                        )
                    index = matches[0]
                kconserv[ki, kj, kk] = index

    helper = kpts_helper.KptsHelper(cell, kpts, init_symm_map=False)
    helper.kconserv = kconserv
    helper.build_symm_map()
    return helper
