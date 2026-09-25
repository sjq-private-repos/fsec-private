#!/usr/bin/env python
"""Generate orbital-resolved SS band energies at mesh and arbitrary k-points."""

import numpy as np
from pyscf.pbc import df, gto, scf

from fsec.singularity_subtraction import BandsSS


def main():
    cell = gto.Cell()
    cell.unit = "Bohr"
    cell.atom = """
        H 0.0 0.0 0.0
        H 0.0 0.0 1.8
        H 3.0 3.0 3.0
        H 3.0 3.0 5.2
    """
    cell.a = np.eye(3) * 8.0
    cell.basis = "gth-szv"
    cell.pseudo = "gth-hf"
    cell.ke_cutoff = 40
    cell.precision = 1e-8
    cell.verbose = 0
    cell.build()

    # A shifted full mesh avoids restricting the occupied manifold to Gamma.
    kpts = cell.make_kpts(
        (1, 1, 2), scaled_center=(0.11, 0.07, 0.13),
        wrap_around=True, with_gamma_point=False,
    )
    kmf = scf.KRHF(cell, kpts)
    kmf.exxdiv = "ewald"
    kmf.with_df = df.GDF(cell, kpts).build()
    kmf.kernel()
    if not kmf.converged:
        raise RuntimeError("The source KRHF calculation did not converge")

    kpts_band = np.asarray([
        kpts[0],                           # on the occupied sampling mesh
        kpts[0] + [1.0e-5, 0.0, 0.0],     # close to a mesh point
        [0.035, -0.020, 0.014],            # off the mesh
    ])
    bands_ss = BandsSS(kmf, kpts_band, q_zero_tol=1e-4)
    energies_ss, coefficients = bands_ss.get_bands()

    print("Band k-point (bohr^-1)             bare occupied      corrected occupied")
    for ik, kpt in enumerate(kpts_band):
        bare = bands_ss.mo_energy_band_baseline[ik, :bands_ss.nocc]
        corrected = energies_ss[ik, :bands_ss.nocc]
        print(f"{kpt!s:38s} {bare!s:18s} {corrected}")
    print("Per-orbital unscaled xi (hartree):")
    print(bands_ss.xi)


if __name__ == "__main__":
    main()
