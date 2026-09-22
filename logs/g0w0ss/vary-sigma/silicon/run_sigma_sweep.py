#!/usr/bin/env python
"""Sweep the G0W0SS Gaussian width for all-electron 2x2x2 silicon."""

import csv
import json
import logging
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyscf import lib, scf
from pyscf.data import nist
from pyscf.pbc import df, dft, gto
from pyscf.pbc.dft import gen_grid

from fsec.singularity_subtraction import G0W0SS


OUTPUT_DIR = Path(__file__).resolve().parent
REFERENCE_CHKFILE = Path("/tmp/si_222_ae_pbe.chk")
REFERENCE_CDERI = Path("/tmp/si_222_ae_ccpvtz_gdf.h5")
LOCAL_CHKFILE = OUTPUT_DIR / "si_222_ae_pbe.chk"
LOCAL_CDERI = OUTPUT_DIR / "si_222_ae_ccpvtz_gdf.h5"
HARTREE_TO_EV = nist.HARTREE2EV
LATTICE_CONSTANT_ANGSTROM = 5.430
KPT_MESH = np.asarray([2, 2, 2], dtype=int)
SIGMA_SCALES = (0.6, 0.8, 1.0, 1.2, 1.4)


def build_cell():
    """Build the all-electron primitive diamond-Si cell from the reference run."""
    a0 = LATTICE_CONSTANT_ANGSTROM
    cell = gto.Cell()
    cell.unit = "Angstrom"
    cell.a = np.asarray(
        [
            [0.0, a0 / 2, a0 / 2],
            [a0 / 2, 0.0, a0 / 2],
            [a0 / 2, a0 / 2, 0.0],
        ]
    )
    cell.atom = [
        ("Si", (0.0, 0.0, 0.0)),
        ("Si", (a0 / 4, a0 / 4, a0 / 4)),
    ]
    cell.basis = "cc-pvtz"
    cell.dimension = 3
    cell.precision = 1e-10
    cell.verbose = 5
    cell.max_memory = 50000
    cell.output = str(OUTPUT_DIR / "scf.log")
    cell.build()
    return cell


def make_mean_field(cell, kpts, gdf):
    """Construct the PBE mean-field object with the reference settings."""
    kmf = dft.KRKS(cell, kpts)
    kmf.xc = "pbe"
    kmf.exxdiv = "ewald"
    kmf.with_df = gdf
    # Tight all-electron primitives make the plane-wave-derived uniform grid
    # pathological, so the reference calculation uses atom-centered grids.
    kmf.grids = gen_grid.BeckeGrids(cell)
    kmf.grids.level = 3
    kmf.conv_tol = 1e-10
    kmf.max_cycle = 100
    return kmf


def load_or_build_mean_field():
    """Reuse the exact reference intermediates, or reproduce them if absent."""
    cell = build_cell()
    if REFERENCE_CHKFILE.is_file() and REFERENCE_CDERI.is_file():
        chkfile = REFERENCE_CHKFILE
        cderi = REFERENCE_CDERI
        logging.info("Reusing reference SCF checkpoint %s", chkfile)
        logging.info("Reusing reference GDF tensor %s", cderi)
    elif LOCAL_CHKFILE.is_file() and LOCAL_CDERI.is_file():
        chkfile = LOCAL_CHKFILE
        cderi = LOCAL_CDERI
        logging.info("Reusing locally generated SCF and GDF intermediates")
    else:
        kpts = cell.make_kpts(KPT_MESH, scaled_center=[0, 0, 0])
        gdf = df.GDF(cell, kpts)
        gdf.auxbasis = "cc-pvtz-ri"
        gdf._cderi_to_save = str(LOCAL_CDERI)
        logging.info("Building all-electron cc-pVTZ-RI GDF tensor")
        gdf.build()
        kmf = make_mean_field(cell, kpts, gdf)
        kmf.chkfile = str(LOCAL_CHKFILE)
        logging.info("Running all-electron PBE mean field")
        kmf.kernel()
        cell.stdout.flush()
        if not kmf.converged:
            raise RuntimeError("PBE calculation did not converge")
        return kmf

    _, scf_record = scf.chkfile.load_scf(str(chkfile))
    kpts = np.asarray(scf_record["kpts"])
    gdf = df.GDF(cell, kpts)
    gdf.auxbasis = "cc-pvtz-ri"
    gdf._cderi = str(cderi)
    kmf = make_mean_field(cell, kpts, gdf)
    kmf.__dict__.update(scf_record)
    # Restoring the checkpoint overwrites attributes, so reinstall the exact
    # density-fitting and numerical-grid objects used by the reference run.
    kmf.with_df = gdf
    kmf.grids = gen_grid.BeckeGrids(cell)
    kmf.grids.level = 3
    return kmf


def result_paths(sigma):
    """Return stable per-width output paths."""
    tag = f"{sigma:.6f}".replace(".", "p")
    return OUTPUT_DIR / f"sigma_{tag}.json", OUTPUT_DIR / f"sigma_{tag}.log"


def run_gw(kmf, requested_sigma, default_sigma):
    """Run one independent edge-only G0W0SS calculation."""
    sigma = default_sigma if requested_sigma is None else requested_sigma
    result_path, log_path = result_paths(sigma)
    if result_path.is_file():
        with result_path.open(encoding="utf-8") as stream:
            result = json.load(stream)
        if np.isclose(result["sigma_inverse_bohr"], sigma, rtol=0.0, atol=1e-12):
            logging.info("Reusing completed result %s", result_path.name)
            return result

    start = time.perf_counter()
    nocc = kmf.cell.nelectron // 2
    edge_orbitals = [nocc - 1, nocc]
    gw = G0W0SS(kmf)
    gw.gaussian_sigma = requested_sigma
    gw.nw = 100
    gw.ac = "pade"
    gw.ac_pade_npts = 18
    gw.qpe_linearized = False
    gw.fc = True
    gw.fc_grid = False
    gw.max_memory = 50000

    with log_path.open("w", encoding="utf-8") as stream:
        gw.stdout = stream
        gw.verbose = 5
        gw.kernel(orbs=edge_orbitals, kptlist=range(len(kmf.kpts)))

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
        "sigma_relative_to_default": sigma / default_sigma,
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
    with result_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    return result


def write_summary(results, kmf, default_sigma, elapsed):
    """Write machine-readable summaries and the requested edge/gap plot."""
    results.sort(key=lambda item: item["sigma_inverse_bohr"])
    nocc = kmf.cell.nelectron // 2
    edge_orbitals = [nocc - 1, nocc]
    scaled_kpts = kmf.cell.get_scaled_kpts(kmf.kpts)
    summary = {
        "system": "all-electron primitive diamond silicon",
        "source_configuration": "logs/g0w0/si_222_ae_g0w0_results.json",
        "basis": "cc-pVTZ",
        "auxbasis": "cc-pVTZ-RI",
        "lattice_constant_angstrom": LATTICE_CONSTANT_ANGSTROM,
        "kpoint_mesh": KPT_MESH.tolist(),
        "scaled_kpoints": scaled_kpts.tolist(),
        "nelectron": kmf.cell.nelectron,
        "nao": kmf.cell.nao_nr(),
        "edge_orbitals_zero_based": edge_orbitals,
        "pbe_total_energy_hartree": float(kmf.e_tot),
        "pbe_edge_energies_ev": (
            np.asarray(kmf.mo_energy)[:, edge_orbitals] * HARTREE_TO_EV
        ).tolist(),
        "nw": 100,
        "analytic_continuation": "pade",
        "ac_pade_npts": 18,
        "qpe_linearized": False,
        "geometry_derived_sigma_inverse_bohr": default_sigma,
        "sigma_scales": list(SIGMA_SCALES),
        "total_sweep_wall_time_seconds": elapsed,
        "results": results,
    }
    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    with (OUTPUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "sigma_inverse_bohr",
                "sigma_relative_to_default",
                "is_geometry_derived_default",
                "vbm_ev",
                "cbm_ev",
                "indirect_gap_ev",
                "vbm_kpoint_index",
                "cbm_kpoint_index",
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
    figure.suptitle(r"All-electron Si G0W0SS, $2\times2\times2$ k-mesh")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "vbm_cbm_gap_vs_sigma.png", dpi=200)
    plt.close(figure)


def main():
    """Run five widths from 0.6 to 1.4 times the geometry-selected value."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(OUTPUT_DIR / "sweep.log", mode="a"),
            logging.StreamHandler(),
        ],
    )
    start = time.perf_counter()
    lib.num_threads(32)
    kmf = load_or_build_mean_field()
    default_sigma = float(
        (6.0 * np.pi**2 / (kmf.cell.vol * len(kmf.kpts))) ** (1.0 / 3.0)
    )
    logging.info("Geometry-derived sigma is %.12f 1/Bohr", default_sigma)
    results = []
    for scale in SIGMA_SCALES:
        requested_sigma = None if scale == 1.0 else scale * default_sigma
        sigma = default_sigma if requested_sigma is None else requested_sigma
        logging.info("Running G0W0SS with sigma=%.12f 1/Bohr (%.1fx)", sigma, scale)
        try:
            result = run_gw(kmf, requested_sigma, default_sigma)
        except Exception:
            logging.exception("G0W0SS failed for sigma=%.12f", sigma)
            raise
        results.append(result)
        logging.info(
            "sigma=%.12f: VBM=%.6f eV, CBM=%.6f eV, gap=%.6f eV",
            sigma,
            result["vbm_ev"],
            result["cbm_ev"],
            result["indirect_gap_ev"],
        )
    elapsed = time.perf_counter() - start
    write_summary(results, kmf, default_sigma, elapsed)
    logging.info("Sweep completed in %.2f s", elapsed)


if __name__ == "__main__":
    main()
