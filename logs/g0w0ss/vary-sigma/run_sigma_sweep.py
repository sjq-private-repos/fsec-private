#!/usr/bin/env python
"""Sweep the G0W0SS Gaussian width for the small periodic H2 example."""

import csv
import json
import logging
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyscf.data import nist
from pyscf.pbc import df, gto, scf

from fsec.singularity_subtraction import G0W0SS


OUTPUT_DIR = Path(__file__).resolve().parent
HARTREE_TO_EV = nist.HARTREE2EV
KPT_MESH = np.asarray([1, 1, 2], dtype=int)


def build_mean_field():
    """Build and converge the same small periodic H2 system as the example."""
    cell = gto.Cell()
    cell.unit = "Bohr"
    cell.atom = "H 0 0 0; H 1.8 0 0"
    cell.a = np.eye(3) * 6.0
    cell.basis = "gth-szv"
    cell.pseudo = "gth-hf"
    cell.ke_cutoff = 100
    cell.precision = 1e-8
    cell.verbose = 4
    cell.output = str(OUTPUT_DIR / "scf.log")
    cell.build()

    kpts = cell.make_kpts(
        KPT_MESH, wrap_around=True, with_gamma_point=True
    )
    kmf = scf.KRHF(cell, kpts)
    kmf.with_df = df.GDF(cell, kpts).build()
    kmf.conv_tol = 1e-10
    kmf.kernel()
    cell.stdout.flush()
    if not kmf.converged:
        raise RuntimeError("Hartree-Fock calculation did not converge")
    return kmf


def run_gw(kmf, requested_sigma, default_sigma):
    """Run one independent edge-only G0W0 calculation and return diagnostics."""
    sigma = default_sigma if requested_sigma is None else requested_sigma
    tag = f"{sigma:.6f}".replace(".", "p")
    log_path = OUTPUT_DIR / f"sigma_{tag}.log"
    start = time.perf_counter()

    gw = G0W0SS(kmf)
    gw.gaussian_sigma = requested_sigma
    gw.nw = 32
    gw.ac_pade_npts = 10
    nocc = int(np.count_nonzero(np.asarray(kmf.mo_occ[0]) > 1.0))
    edge_orbitals = [nocc - 1, nocc]
    kptlist = list(range(gw.nkpts))

    with log_path.open("w", encoding="utf-8") as stream:
        gw.stdout = stream
        gw.verbose = 4
        gw.kernel(orbs=edge_orbitals, kptlist=kptlist)

        qp_edges = np.asarray(gw.mo_energy)[:, edge_orbitals] * HARTREE_TO_EV
        vbm_kpoint = int(np.argmax(qp_edges[:, 0]))
        cbm_kpoint = int(np.argmin(qp_edges[:, 1]))
        vbm = float(qp_edges[vbm_kpoint, 0])
        cbm = float(qp_edges[cbm_kpoint, 1])
        elapsed = time.perf_counter() - start
        print(f"selected sigma (1/Bohr) = {gw.gaussian_coefficients.sigma}", file=stream)
        print(f"VBM (eV) = {vbm}", file=stream)
        print(f"CBM (eV) = {cbm}", file=stream)
        print(f"indirect gap (eV) = {cbm - vbm}", file=stream)
        print(f"wall time (s) = {elapsed}", file=stream)

    result = {
        "requested_sigma_inverse_bohr": requested_sigma,
        "sigma_inverse_bohr": float(gw.gaussian_coefficients.sigma),
        "is_geometry_derived_default": requested_sigma is None,
        "qp_edge_energies_ev": qp_edges.tolist(),
        "vbm_ev": vbm,
        "vbm_kpoint_index": vbm_kpoint,
        "cbm_ev": cbm,
        "cbm_kpoint_index": cbm_kpoint,
        "indirect_gap_ev": cbm - vbm,
        "wall_time_seconds": elapsed,
        "gaussian_coefficients": gw.gaussian_coefficients.as_dict(),
        "log_file": log_path.name,
    }
    with (OUTPUT_DIR / f"sigma_{tag}.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    return result


def write_summary(results, kmf, default_sigma):
    """Write machine-readable summaries and the requested edge/gap plot."""
    results.sort(key=lambda item: item["sigma_inverse_bohr"])
    summary = {
        "system": "periodic H2",
        "cell_bohr": (np.eye(3) * 6.0).tolist(),
        "atom": "H 0 0 0; H 1.8 0 0",
        "basis": "gth-szv",
        "pseudopotential": "gth-hf",
        "kpoint_mesh": KPT_MESH.tolist(),
        "nw": 32,
        "ac_pade_npts": 10,
        "geometry_derived_sigma_inverse_bohr": default_sigma,
        "mean_field_edge_energies_ev": (
            np.asarray(kmf.mo_energy)[:, [kmf.cell.nelectron // 2 - 1,
                                         kmf.cell.nelectron // 2]]
            * HARTREE_TO_EV
        ).tolist(),
        "results": results,
    }
    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    with (OUTPUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "sigma_inverse_bohr",
                "is_geometry_derived_default",
                "vbm_ev",
                "cbm_ev",
                "indirect_gap_ev",
                "wall_time_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(
            {key: result[key] for key in writer.fieldnames} for result in results
        )

    sigmas = np.asarray([item["sigma_inverse_bohr"] for item in results])
    vbm = np.asarray([item["vbm_ev"] for item in results])
    cbm = np.asarray([item["cbm_ev"] for item in results])
    gap = np.asarray([item["indirect_gap_ev"] for item in results])
    figure, (edge_axis, gap_axis) = plt.subplots(2, 1, figsize=(6.4, 7.0), sharex=True)
    edge_axis.plot(sigmas, vbm, "o-", label="VBM")
    edge_axis.plot(sigmas, cbm, "s-", label="CBM")
    edge_axis.axvline(default_sigma, color="0.5", linestyle="--", label="default sigma")
    edge_axis.set_ylabel("Quasiparticle energy (eV)")
    edge_axis.legend()
    edge_axis.grid(alpha=0.25)

    gap_axis.plot(sigmas, gap, "o-", color="tab:green")
    gap_axis.axvline(default_sigma, color="0.5", linestyle="--")
    gap_axis.set_xlabel(r"Gaussian width $\sigma$ (Bohr$^{-1}$)")
    gap_axis.set_ylabel("Indirect band gap (eV)")
    gap_axis.grid(alpha=0.25)
    figure.suptitle(r"Periodic H$_2$ G0W0SS, $1\times1\times2$ k-mesh")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "vbm_cbm_gap_vs_sigma.png", dpi=200)
    plt.close(figure)


def main():
    """Run five widths spanning the geometry-selected value."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(OUTPUT_DIR / "sweep.log", mode="w"),
            logging.StreamHandler(),
        ],
    )
    start = time.perf_counter()
    logging.info("Building periodic H2 mean field")
    kmf = build_mean_field()
    default_sigma = float(
        (6.0 * np.pi**2 / (kmf.cell.vol * len(kmf.kpts))) ** (1.0 / 3.0)
    )
    requested_sigmas = [0.30, 0.40, None, 0.60, 0.70]
    results = []
    for requested_sigma in requested_sigmas:
        selected = default_sigma if requested_sigma is None else requested_sigma
        logging.info("Running G0W0SS with sigma=%.9f 1/Bohr", selected)
        result = run_gw(kmf, requested_sigma, default_sigma)
        results.append(result)
        logging.info(
            "sigma=%.9f: VBM=%.6f eV, CBM=%.6f eV, gap=%.6f eV",
            selected,
            result["vbm_ev"],
            result["cbm_ev"],
            result["indirect_gap_ev"],
        )
    write_summary(results, kmf, default_sigma)
    logging.info("Sweep completed in %.2f s", time.perf_counter() - start)


if __name__ == "__main__":
    main()
