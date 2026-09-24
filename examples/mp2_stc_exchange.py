#!/usr/bin/env python
"""Compare ordinary and STC MP2 exchange on fixed KRHF orbitals.

Requires the custom PySCF STC builder in the fsec-312 environment. Energies
printed below are exchange contributions in hartree per primitive cell.
"""

import numpy as np

from pyscf.pbc import gto, scf
from pyscf.pbc.mp import kmp2

from fsec.vcut import KMP2_STC


def main():
    cell = gto.M(
        a=np.eye(3) * 4.0,
        atom="He 0 0 0",
        basis="gth-dzvp",
        pseudo="gth-pade",
        precision=1e-8,
        mesh=[17] * 3,
        verbose=3,
    )
    kpts = cell.make_kpts([3, 1, 1], scaled_center=[0.17, 0.0, 0.0])
    kmf = scf.KRHF(cell, kpts, exxdiv=None).rs_density_fit(auxbasis="weigend")
    kmf.conv_tol = 1e-10
    kmf.kernel()
    if not kmf.converged:
        raise RuntimeError("KRHF did not converge")

    ordinary = kmp2.KMP2(kmf)
    ordinary.kernel(with_t2=False)
    e_ordinary = ordinary.e_corr_ss - ordinary.e_corr_os

    # Both STC ERI factors use the same fixed orbitals and energy denominators
    # as the ordinary calculation. STC creates its own stored RSDF builder.
    stc = KMP2_STC(kmf, eta=4.0, exxdiv="vcut_ws", rc_type="ws")
    e_stc, _ = stc.kernel()

    print("Ordinary MP2 exchange (hartree/cell): %.12f" % e_ordinary)
    print("STC MP2 exchange      (hartree/cell): %.12f" % e_stc)
    print("STC - ordinary        (hartree/cell): %.12f" % (e_stc - e_ordinary))


if __name__ == "__main__":
    main()
