"""Compare sTC and FFTDF bands for one H2 MP2 small-q fitting point.

The default sweep runs one reference KRHF/KMP2 calculation, then compares
finite-eta sTC bands with FFTDF bands on 25^3, 33^3, and 49^3 meshes. Each run
uses the same occupied data, spherical Coulomb cutoff, uniform pair-density
quadrature, and ordinary GDF correlation settings.
"""

import argparse

import numpy as np
from pyscf.pbc import gto, mp, scf

from fsec.singularity_subtraction.structure_factor import (
    MP2SmallQ,
    MP2SmallQOptions,
)


RELATIVE_SHIFT = (0.0, 0.0, 0.1)
PAIR_DENSITY_MESH = (17, 17, 17)


def _energy_vector(result):
    """Join active virtual energies in the reported plus/minus order."""
    blocks = (
        result.active_virtual_energies_plus
        + result.active_virtual_energies_minus
    )
    return np.concatenate([np.asarray(block).ravel() for block in blocks])


def _run_point(kmf, kmp, options):
    return MP2SmallQ(
        kmf,
        kmp,
        options=options,
        N_local=PAIR_DENSITY_MESH,
        pair_density_eval_grid="uniform",
        verbose=0,
    ).kernel()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fft-meshes",
        nargs="+",
        type=int,
        default=(25, 33, 49),
        help="cubic FFTDF meshes to compare (default: 25 33 49)",
    )
    parser.add_argument(
        "--etas",
        nargs="+",
        type=float,
        default=(4.0, 8.0),
        help="sTC eta values to compare (default: 4 8)",
    )
    args = parser.parse_args()
    if any(mesh < 1 for mesh in args.fft_meshes):
        parser.error("FFT mesh sizes must be positive")
    if any(eta <= 0 for eta in args.etas):
        parser.error("eta values must be positive")

    cell = gto.Cell()
    cell.unit = "Bohr"
    cell.atom = "H 0 0 0; H 0 0 1.8"
    cell.a = np.eye(3) * 6.0
    cell.basis = "gth-szv"
    cell.pseudo = "gth-hf"
    cell.mesh = np.asarray((17, 17, 17))
    cell.precision = 1e-8
    cell.verbose = 0
    cell.build()

    kpts = cell.make_kpts((1, 1, 2), wrap_around=True)
    kmf = scf.KRHF(cell, kpts, exxdiv="ewald").density_fit()
    kmf.conv_tol = 1e-10
    kmf.kernel()
    if not kmf.converged:
        raise RuntimeError("Reference H2 KRHF calculation did not converge")

    kmp = mp.KMP2(kmf)
    kmp.with_df_ints = True
    kmp.kernel(with_t2=False)

    fft_results = []
    all_results = []
    for mesh_size in args.fft_meshes:
        options = MP2SmallQOptions(
            relative_shift=RELATIVE_SHIFT,
            cutoff="sph",
            band_backend="fftdf",
            fft_mesh=(mesh_size,) * 3,
        )
        result = _run_point(kmf, kmp, options)
        fft_results.append(result)
        all_results.append((f"FFTDF {mesh_size}^3", result))

    for eta in args.etas:
        options = MP2SmallQOptions(
            relative_shift=RELATIVE_SHIFT,
            eta=eta,
            cutoff="sph",
            band_backend="rsdf_stc",
        )
        all_results.append((f"sTC eta={eta:g}", _run_point(kmf, kmp, options)))

    finest = max(fft_results, key=lambda result: np.prod(result.fft_mesh))
    finest_energies = _energy_vector(finest)
    print("q (inverse Bohr):", finest.qprime)
    print("Fixed pair-density mesh:", PAIR_DENSITY_MESH)
    print("Fixed cutoff: spherical")
    print()
    for label, result in all_results:
        energies = _energy_vector(result)
        print(label)
        print("  active virtual energies k+q (Hartree):")
        for values in result.active_virtual_energies_plus:
            print("   ", np.array2string(values, precision=10))
        print("  active virtual energies k-q (Hartree):")
        for values in result.active_virtual_energies_minus:
            print("   ", np.array2string(values, precision=10))
        print(f"  direct structure factor: {result.sq_direct:.12e}")
        print(f"  fourth-order structure factor: {result.sq_q4:.12e}")
        print(
            "  difference from finest FFTDF: "
            f"direct={result.sq_direct - finest.sq_direct:+.6e}, "
            f"q4={result.sq_q4 - finest.sq_q4:+.6e}, "
            f"max |active virtual energy delta|="
            f"{np.max(np.abs(energies - finest_energies)):.6e} Hartree"
        )
        print()


if __name__ == "__main__":
    main()
