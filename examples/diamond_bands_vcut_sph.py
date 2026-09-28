#!/usr/bin/env python
"""Overlay regular self-consistent FFTDF/vcut_sph bands on saved diamond SS bands.

Run ``python -m examples.diamond_bands_vcut_sph`` after diamond_bands_ss.
Use OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 to avoid oversubscription.
The saved SS calculation and its original figures are preserved.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from pyscf.pbc import df, scf
from pyscf.pbc.scf import chkfile

from examples.diamond_bands_ss import HARTREE_TO_EV, plot


def calculate(output, batch_size=8):
    """Converge vcut_sph HF and evaluate the exact saved band path."""
    with np.load(output / "bands.npz") as saved:
        reference = dict(saved)
    cell, source = chkfile.load_scf(str(output / "source.chk"))
    np.testing.assert_array_equal(cell.mesh, reference["exchange_mesh"])
    np.testing.assert_allclose(source["e_tot"], reference["source_energy"], atol=1e-10, rtol=0)
    cell.verbose = 4
    kmf = scf.KRHF(cell, reference["source_kpts"])
    kmf.with_df = df.FFTDF(cell, kmf.kpts)
    kmf.exxdiv = "vcut_sph"
    kmf.conv_tol = 1e-9
    kmf.chkfile = str(output / "vcut_sph_source.chk")
    for key in ("mo_coeff", "mo_occ", "mo_energy"):
        setattr(kmf, key, source[key])
    initial_density = kmf.make_rdm1()
    started = time.monotonic()
    kmf.kernel(dm0=initial_density)
    if not kmf.converged:
        raise RuntimeError("The vcut_sph source SCF did not converge")
    scf_seconds = time.monotonic() - started
    print(f"vcut_sph SCF complete in {scf_seconds:.1f} s", flush=True)

    kpts_band = reference["kpts_band"]
    energies, coefficients = [], []
    band_started = time.monotonic()
    helper = cell.get_abs_kpts([0.23, 0.17, 0.11])
    for start in range(0, len(kpts_band), batch_size):
        stop = min(start + batch_size, len(kpts_band))
        # Batch with an unused nonzero point to avoid PySCF's single-Gamma cast.
        eval_kpts = np.vstack((kpts_band[start:stop], helper))
        energy, coefficient = kmf.get_bands(eval_kpts)
        energies.extend(energy[:stop-start])
        coefficients.extend(coefficient[:stop-start])
        print(f"vcut_sph bands {stop}/{len(kpts_band)}; "
              f"{time.monotonic() - band_started:.1f} s", flush=True)
        np.savez_compressed(
            output / "vcut_sph_bands.npz", energies=np.asarray(energies),
            coefficients=np.asarray(coefficients), kpts_band=kpts_band[:stop],
            source_kpts=kmf.kpts, source_energy=kmf.e_tot,
            mesh=cell.mesh, exxdiv=kmf.exxdiv,
        )
    band_seconds = time.monotonic() - band_started
    energies = np.asarray(energies)
    assert energies.shape == reference["corrected"].shape
    assert np.all(np.isfinite(energies))
    gamma = np.flatnonzero(np.linalg.norm(kpts_band, axis=1) < 1e-10)
    gamma_error = float(np.max(np.abs(energies[gamma[0]] - energies[gamma[-1]])))
    np.testing.assert_allclose(energies[gamma[0]], energies[gamma[-1]], atol=1e-8, rtol=0)
    nocc = int(reference["nocc"])
    metadata = {
        "method": "KRHF, FFTDF, exxdiv='vcut_sph' in both SCF and bands",
        "source_mesh": [2, 2, 2], "fft_mesh": cell.mesh.tolist(),
        "number_of_band_points": len(kpts_band),
        "source_energy_hartree": float(kmf.e_tot),
        "energy_zero_hartree": float(np.max(reference["corrected"][:, :nocc])),
        "energy_alignment": "Common SS valence maximum; no independent shifts",
        "sampled_path_gap_eV": float((energies[:, nocc:].min() -
                                       energies[:, :nocc].max()) * HARTREE_TO_EV),
        "repeated_gamma_max_difference_hartree": gamma_error,
        "scf_seconds": scf_seconds, "band_seconds": band_seconds,
        "source_density_change_norm": float(np.linalg.norm(kmf.make_rdm1() - initial_density)),
    }
    (output / "vcut_sph_settings.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)
    return energies


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/diamond_bands_ss"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        with np.load(args.output / "vcut_sph_bands.npz") as data:
            energies = data["energies"]
    else:
        energies = calculate(args.output, args.batch_size)
    plot(args.output, vcut_sph=energies)
