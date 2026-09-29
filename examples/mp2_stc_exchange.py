#!/usr/bin/env python
"""Compare ordinary, sTC, and hybrid MP2 components on fixed KRHF orbitals.

Requires the custom PySCF STC builder in the fsec-312 environment. Energies
printed below are correlation energies and components in hartree per primitive
cell.
"""

import numpy as np

from pyscf.pbc import gto, scf
from pyscf.pbc.mp import kmp2

from fsec.vcut import KMP2_HYBRID, KMP2_STC


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

    # STC creates its own stored builder and uses the fixed HF orbitals and
    # energy denominators in both interaction factors.
    stc = KMP2_STC(kmf, eta=4.0, exxdiv="vcut_ws", rc_type="ws")
    e_stc, _ = stc.kernel(with_direct=True)

    # Hybrid uses an independent bare RSDF builder for the direct contribution
    # and the sTC builder for exchange.
    hybrid = KMP2_HYBRID(kmf, eta=4.0, exxdiv="vcut_ws", rc_type="ws")
    e_hybrid, t2 = hybrid.kernel(with_t2=True)

    print("Ordinary MP2 direct   (hartree/cell): %.12f" % (2 * ordinary.e_corr_os))
    print("Ordinary MP2 exchange (hartree/cell): %.12f" % e_ordinary)
    print("Ordinary correlation  (hartree/cell): %.12f" % ordinary.e_corr)
    print("STC MP2 direct        (hartree/cell): %.12f" % stc.e_corr_direct)
    print("STC MP2 exchange      (hartree/cell): %.12f" % stc.e_corr_exchange)
    print("STC MP2 correlation   (hartree/cell): %.12f" % e_stc)
    print("Hybrid bare direct    (hartree/cell): %.12f" % hybrid.e_corr_direct)
    print("Hybrid sTC exchange   (hartree/cell): %.12f" % hybrid.e_corr_exchange)
    print("Hybrid correlation    (hartree/cell): %.12f" % e_hybrid)
    print("Hybrid t2 shapes:", t2["direct"].shape, t2["exchange"].shape)


if __name__ == "__main__":
    main()
