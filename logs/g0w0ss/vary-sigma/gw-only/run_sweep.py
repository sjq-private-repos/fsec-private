#!/usr/bin/env python
"""Sweep only the G0W0SS correlation width for periodic H2 or diamond."""

import argparse
import csv
import json
import logging
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyscf import scf as molecular_scf
from pyscf.data import nist
from pyscf.pbc import df, dft, gto, scf, tools
from pyscf.pbc.dft import gen_grid

from fsec.singularity_subtraction import G0W0SS


ROOT = Path(__file__).resolve().parent
HARTREE_TO_EV = nist.HARTREE2EV
SMALL_SYSTEM_SIGMAS = (0.3, 0.4, None, 0.6, 0.7)
SILICON_SIGMA_SCALES = (0.6, 0.8, 1.0, 1.2, 1.4)
GW_SETTINGS = {
    "h2": {"nw": 32, "ac_pade_npts": 10},
    "diamond": {"nw": 32, "ac_pade_npts": 10},
    "silicon": {"nw": 100, "ac_pade_npts": 18},
    "silicon-no-exx": {"nw": 100, "ac_pade_npts": 18},
    "silicon-madelung": {"nw": 100, "ac_pade_npts": 18},
}


def build_h2(output_dir):
    """Build the same periodic H2 Hartree-Fock reference as the first sweep."""
    cell = gto.Cell()
    cell.unit = "Bohr"
    cell.atom = "H 0 0 0; H 1.8 0 0"
    cell.a = np.eye(3) * 6.0
    cell.basis = "gth-szv"
    cell.pseudo = "gth-hf"
    cell.ke_cutoff = 100
    cell.precision = 1e-8
    cell.verbose = 4
    cell.output = str(output_dir / "scf.log")
    cell.build()

    kmesh = np.asarray([1, 1, 2], dtype=int)
    kpts = cell.make_kpts(kmesh, wrap_around=True, with_gamma_point=True)
    kmf = scf.KRHF(cell, kpts)
    kmf.with_df = df.GDF(cell, kpts).build()
    kmf.conv_tol = 1e-10
    kmf.kernel()
    cell.stdout.flush()
    if not kmf.converged:
        raise RuntimeError("Periodic H2 Hartree-Fock calculation did not converge")
    metadata = {
        "system": "periodic H2",
        "cell_bohr": (np.eye(3) * 6.0).tolist(),
        "atom": "H 0 0 0; H 1.8 0 0",
        "basis": "gth-szv",
        "pseudopotential": "gth-hf",
        "mean_field": "KRHF",
        "kpoint_mesh": kmesh.tolist(),
    }
    return kmf, metadata


def build_diamond(output_dir):
    """Build the repository example's primitive diamond cell with PBE orbitals."""
    lattice_half = 3.370326545430162
    atom_quarter = 1.68516327271508
    cell = gto.Cell()
    cell.unit = "Bohr"
    cell.atom = [
        ("C", (0.0, 0.0, 0.0)),
        ("C", (atom_quarter, atom_quarter, atom_quarter)),
    ]
    cell.a = np.asarray(
        [
            [0.0, lattice_half, lattice_half],
            [lattice_half, 0.0, lattice_half],
            [lattice_half, lattice_half, 0.0],
        ]
    )
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pbe"
    cell.ke_cutoff = 100
    cell.precision = 1e-8
    cell.max_memory = 4000
    cell.verbose = 4
    cell.output = str(output_dir / "scf.log")
    cell.build()

    kmesh = np.asarray([2, 2, 2], dtype=int)
    kpts = cell.make_kpts(kmesh, wrap_around=True, with_gamma_point=True)
    kmf = dft.KRKS(cell, kpts)
    kmf.xc = "pbe"
    kmf.exxdiv = "ewald"
    kmf.with_df = df.GDF(cell, kpts).build()
    kmf.conv_tol = 1e-10
    kmf.kernel()
    cell.stdout.flush()
    if not kmf.converged:
        raise RuntimeError("Diamond PBE calculation did not converge")
    metadata = {
        "system": "primitive diamond",
        "conventional_lattice_constant_bohr": 2.0 * lattice_half,
        "basis": "gth-szv",
        "pseudopotential": "gth-pbe",
        "mean_field": "PBE",
        "kpoint_mesh": kmesh.tolist(),
    }
    return kmf, metadata


def build_silicon(output_dir):
    """Restore the all-electron silicon reference used by the saved G0W0 run."""
    a0 = 5.430
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
    cell.max_memory = 50000
    cell.verbose = 5
    cell.output = str(output_dir / "scf.log")
    cell.build()

    chkfile = Path("/tmp/si_222_ae_pbe.chk")
    cderi_file = Path("/tmp/si_222_ae_ccpvtz_gdf.h5")
    if not chkfile.is_file() or not cderi_file.is_file():
        raise FileNotFoundError(
            "The silicon sweep requires /tmp/si_222_ae_pbe.chk and "
            "/tmp/si_222_ae_ccpvtz_gdf.h5 from the reference calculation"
        )

    _, scf_record = molecular_scf.chkfile.load_scf(str(chkfile))
    kpts = np.asarray(scf_record["kpts"])
    gdf = df.GDF(cell, kpts)
    gdf.auxbasis = "cc-pvtz-ri"
    gdf._cderi = str(cderi_file)
    kmf = dft.KRKS(cell, kpts)
    kmf.xc = "pbe"
    kmf.exxdiv = "ewald"
    kmf.with_df = gdf
    kmf.grids = gen_grid.BeckeGrids(cell)
    kmf.grids.level = 3
    kmf.__dict__.update(scf_record)
    kmf.with_df = gdf
    kmf.grids = gen_grid.BeckeGrids(cell)
    kmf.grids.level = 3
    cell.stdout.flush()
    metadata = {
        "system": "all-electron primitive silicon",
        "source_configuration": "logs/g0w0/si_222_ae_g0w0_results.json",
        "conventional_lattice_constant_angstrom": a0,
        "basis": "cc-pVTZ",
        "auxbasis": "cc-pVTZ-RI",
        "mean_field": "PBE",
        "kpoint_mesh": [2, 2, 2],
    }
    return kmf, metadata


def hold_exchange_width_fixed(gw, correlation_sigma, exchange_sigma):
    """Keep the net occupied exchange correction at ``exchange_sigma``.

    The production kernel adds ``-2/pi * correlation_sigma`` to occupied
    exchange matrix elements.  Pre-adjusting the raw exchange by the
    difference makes the net correction exactly ``-2/pi * exchange_sigma``
    while leaving the correlation head and wing coefficients untouched.
    """
    get_raw_exchange = gw.get_sigma_exchange

    def get_fixed_exchange():
        exchange = get_raw_exchange()
        correction_difference = -2.0 / np.pi * (
            exchange_sigma - correlation_sigma
        )
        for kpoint in range(gw.nkpts):
            for orbital in range(gw.nocc):
                exchange[kpoint][orbital, orbital] += correction_difference
        return exchange

    gw.get_sigma_exchange = get_fixed_exchange


def result_paths(output_dir, sigma):
    """Return stable result and trace paths for one correlation width."""
    tag = f"{sigma:.6f}".replace(".", "p")
    return output_dir / f"sigma_{tag}.json", output_dir / f"sigma_{tag}.log"


def seed_silicon_default_result(output_dir, default_sigma):
    """Reuse the supplied reference for the unchanged default-width point."""
    result_path, log_path = result_paths(output_dir, default_sigma)
    if result_path.is_file():
        return
    source_path = Path("logs/g0w0/si_222_ae_g0w0_results.json")
    with source_path.open(encoding="utf-8") as stream:
        source = json.load(stream)
    source_sigma = source["gaussian_coefficients"]["sigma"]
    if not np.isclose(source_sigma, default_sigma, rtol=0.0, atol=1e-12):
        raise ValueError("The supplied silicon result has an unexpected Gaussian width")
    result = {
        "requested_correlation_sigma_inverse_bohr": None,
        "correlation_sigma_inverse_bohr": source_sigma,
        "exchange_sigma_inverse_bohr": default_sigma,
        "is_geometry_derived_default": True,
        "qp_edge_energies_ev": source["qp_edge_energies_ev"],
        "vbm_ev": source["vbm_ev"],
        "vbm_kpoint_index": source["vbm_kpoint_index"],
        "cbm_ev": source["cbm_ev"],
        "cbm_kpoint_index": source["cbm_kpoint_index"],
        "indirect_gap_ev": source["indirect_gap_ev"],
        "wall_time_seconds": source["wall_time_seconds"],
        "correlation_gaussian_coefficients": source["gaussian_coefficients"],
        "log_file": log_path.name,
        "reused_reference_result": str(source_path),
    }
    with result_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    with log_path.open("w", encoding="utf-8") as stream:
        print(f"Reused unchanged default-width result from {source_path}", file=stream)
        print(f"G0W0 correlation sigma (1/Bohr) = {source_sigma}", file=stream)
        print(f"fixed ExxSS sigma (1/Bohr) = {default_sigma}", file=stream)


def run_gw(
    kmf,
    output_dir,
    requested_sigma,
    default_sigma,
    exchange_sigma,
    exchange_mode,
    settings,
):
    """Run one GW point with selected exchange and correlation widths."""
    correlation_sigma = default_sigma if requested_sigma is None else requested_sigma
    result_path, log_path = result_paths(output_dir, correlation_sigma)
    if result_path.is_file():
        with result_path.open(encoding="utf-8") as stream:
            result = json.load(stream)
        stored_mode = result.get("exchange_correction")
        if stored_mode is None:
            stored_mode = (
                "none"
                if result.get("exchange_sigma_inverse_bohr") == 0.0
                else "fixed-gaussian"
            )
        sigma_matches = (
            exchange_mode != "fixed-gaussian"
            or np.isclose(
                result["exchange_sigma_inverse_bohr"],
                exchange_sigma,
                rtol=0.0,
                atol=1e-12,
            )
        )
        if (
            np.isclose(
                result["correlation_sigma_inverse_bohr"],
                correlation_sigma,
                rtol=0.0,
                atol=1e-12,
            )
            and stored_mode == exchange_mode
            and sigma_matches
        ):
            logging.info("Reusing completed result %s", result_path.name)
            return result

    start = time.perf_counter()
    nocc = kmf.cell.nelectron // 2
    edge_orbitals = [nocc - 1, nocc]
    gw = G0W0SS(kmf)
    gw.gaussian_sigma = requested_sigma
    gw.nw = settings["nw"]
    gw.ac = "pade"
    gw.ac_pade_npts = settings["ac_pade_npts"]
    gw.qpe_linearized = False
    gw.fc_grid = False
    gw.max_memory = kmf.cell.max_memory
    if exchange_mode == "madelung":
        gw.exchange_correction = "madelung"
    elif exchange_mode == "none":
        gw.exchange_correction = None
    else:
        hold_exchange_width_fixed(gw, correlation_sigma, exchange_sigma)

    with log_path.open("w", encoding="utf-8") as stream:
        gw.stdout = stream
        gw.verbose = 4
        gw.kernel(orbs=edge_orbitals, kptlist=range(gw.nkpts))

        qp_edges = np.asarray(gw.mo_energy)[:, edge_orbitals] * HARTREE_TO_EV
        vbm_kpoint = int(np.argmax(qp_edges[:, 0]))
        cbm_kpoint = int(np.argmin(qp_edges[:, 1]))
        vbm = float(qp_edges[vbm_kpoint, 0])
        cbm = float(qp_edges[cbm_kpoint, 1])
        elapsed = time.perf_counter() - start
        print(
            f"G0W0 correlation sigma (1/Bohr) = {gw.gaussian_coefficients.sigma}",
            file=stream,
        )
        if exchange_mode == "madelung":
            print(
                f"Madelung exchange correction (hartree) = "
                f"{-tools.madelung(gw.mol, gw.kpts)}",
                file=stream,
            )
        elif exchange_mode == "none":
            print("ExxSS correction = disabled", file=stream)
        else:
            print(f"fixed ExxSS sigma (1/Bohr) = {exchange_sigma}", file=stream)
        print(f"VBM (eV) = {vbm}", file=stream)
        print(f"CBM (eV) = {cbm}", file=stream)
        print(f"indirect gap (eV) = {cbm - vbm}", file=stream)
        print(f"wall time (s) = {elapsed}", file=stream)

    result = {
        "requested_correlation_sigma_inverse_bohr": requested_sigma,
        "correlation_sigma_inverse_bohr": float(gw.gaussian_coefficients.sigma),
        "exchange_sigma_inverse_bohr": exchange_sigma,
        "exchange_correction": exchange_mode,
        "madelung_exchange_correction_hartree": (
            -tools.madelung(gw.mol, gw.kpts)
            if exchange_mode == "madelung"
            else None
        ),
        "is_geometry_derived_default": requested_sigma is None,
        "qp_edge_energies_ev": qp_edges.tolist(),
        "vbm_ev": vbm,
        "vbm_kpoint_index": vbm_kpoint,
        "cbm_ev": cbm,
        "cbm_kpoint_index": cbm_kpoint,
        "indirect_gap_ev": cbm - vbm,
        "wall_time_seconds": elapsed,
        "correlation_gaussian_coefficients": gw.gaussian_coefficients.as_dict(),
        "log_file": log_path.name,
    }
    with result_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    return result


def write_summary(
    output_dir,
    metadata,
    results,
    kmf,
    default_sigma,
    elapsed,
    settings,
    requested_sigmas,
    exchange_sigma,
    exchange_mode,
):
    """Write JSON/CSV summaries and the edge/gap figure."""
    results.sort(key=lambda item: item["correlation_sigma_inverse_bohr"])
    nocc = kmf.cell.nelectron // 2
    edge_orbitals = [nocc - 1, nocc]
    summary = {
        **metadata,
        "sweep_scope": {
            "madelung": "G0W0SS correlation with conventional Madelung exchange correction",
            "none": "G0W0SS correlation only; exchange correction disabled",
            "fixed-gaussian": "G0W0SS correlation only; Gaussian exchange held fixed",
        }[exchange_mode],
        "exchange_correction": exchange_mode,
        "exchange_sigma_inverse_bohr": exchange_sigma,
        "madelung_exchange_correction_hartree": (
            -tools.madelung(kmf.cell, kmf.kpts)
            if exchange_mode == "madelung"
            else None
        ),
        "geometry_derived_sigma_inverse_bohr": default_sigma,
        "requested_correlation_sigmas_inverse_bohr": list(requested_sigmas),
        "edge_orbitals_zero_based": edge_orbitals,
        "mean_field_total_energy_hartree": float(kmf.e_tot),
        "mean_field_edge_energies_ev": (
            np.asarray(kmf.mo_energy)[:, edge_orbitals] * HARTREE_TO_EV
        ).tolist(),
        "nw": settings["nw"],
        "ac_pade_npts": settings["ac_pade_npts"],
        "qpe_linearized": False,
        "total_sweep_wall_time_seconds": elapsed,
        "results": results,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    fieldnames = [
        "correlation_sigma_inverse_bohr",
        "exchange_sigma_inverse_bohr",
        "is_geometry_derived_default",
        "vbm_ev",
        "cbm_ev",
        "indirect_gap_ev",
        "vbm_kpoint_index",
        "cbm_kpoint_index",
        "wall_time_seconds",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {key: result[key] for key in fieldnames} for result in results
        )

    sigmas = np.asarray(
        [item["correlation_sigma_inverse_bohr"] for item in results]
    )
    vbm = np.asarray([item["vbm_ev"] for item in results])
    cbm = np.asarray([item["cbm_ev"] for item in results])
    gap = np.asarray([item["indirect_gap_ev"] for item in results])
    figure, (vbm_axis, cbm_axis, gap_axis) = plt.subplots(
        1, 3, figsize=(15.0, 4.5), sharex=True
    )
    vbm_axis.plot(sigmas, vbm, "o-", color="tab:blue")
    vbm_axis.axvline(
        default_sigma, color="0.5", linestyle="--", label="default sigma"
    )
    vbm_axis.set_ylabel("VBM (eV)")
    vbm_axis.legend()
    vbm_axis.grid(alpha=0.25)

    cbm_axis.plot(sigmas, cbm, "s-", color="tab:orange")
    cbm_axis.axvline(default_sigma, color="0.5", linestyle="--")
    cbm_axis.set_ylabel("CBM (eV)")
    cbm_axis.grid(alpha=0.25)

    gap_axis.plot(sigmas, gap, "o-", color="tab:green")
    gap_axis.axvline(default_sigma, color="0.5", linestyle="--")
    gap_axis.set_ylabel("Indirect band gap (eV)")
    gap_axis.grid(alpha=0.25)
    if exchange_mode == "madelung":
        exchange_label = "Madelung exchange correction; no ExxSS"
    elif exchange_mode == "none":
        exchange_label = "Exchange correction disabled"
    else:
        exchange_label = (
            f"ExxSS fixed at $\\sigma={exchange_sigma:.6f}$ Bohr$^{{-1}}$"
        )
    figure.suptitle(
        f"{metadata['system']}: G0W0SS width sweep\n{exchange_label}"
    )
    figure.supxlabel(r"G0W0SS correlation width $\sigma$ (Bohr$^{-1}$)")
    figure.tight_layout()
    figure.savefig(output_dir / "vbm_cbm_gap_vs_g0w0ss_sigma.png", dpi=200)
    figure.savefig(output_dir / "vbm_cbm_gap_vs_g0w0ss_sigma_1x3.png", dpi=200)
    plt.close(figure)


def main():
    """Run a restartable correlation-only sweep for the selected system."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "system",
        choices=(
            "h2",
            "diamond",
            "silicon",
            "silicon-no-exx",
            "silicon-madelung",
        ),
    )
    args = parser.parse_args()
    output_dir = ROOT / args.system
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "sweep.log", mode="a"),
            logging.StreamHandler(),
        ],
    )
    start = time.perf_counter()
    builders = {
        "h2": build_h2,
        "diamond": build_diamond,
        "silicon": build_silicon,
        "silicon-no-exx": build_silicon,
        "silicon-madelung": build_silicon,
    }
    build = builders[args.system]
    settings = GW_SETTINGS[args.system]
    logging.info("Building %s mean field", args.system)
    kmf, metadata = build(output_dir)
    default_sigma = float(
        (6.0 * np.pi**2 / (kmf.cell.vol * len(kmf.kpts))) ** (1.0 / 3.0)
    )
    if args.system == "silicon-madelung":
        exchange_mode = "madelung"
        exchange_sigma = None
        logging.info(
            "Using Madelung exchange correction %.12f hartree",
            -tools.madelung(kmf.cell, kmf.kpts),
        )
    elif args.system == "silicon-no-exx":
        exchange_mode = "none"
        exchange_sigma = 0.0
        logging.info("Disabling the ExxSS correction")
    else:
        exchange_mode = "fixed-gaussian"
        exchange_sigma = default_sigma
        logging.info("Holding ExxSS at sigma=%.12f 1/Bohr", exchange_sigma)
    if args.system in ("silicon", "silicon-no-exx", "silicon-madelung"):
        requested_sigmas = tuple(
            None if scale == 1.0 else scale * default_sigma
            for scale in SILICON_SIGMA_SCALES
        )
        if args.system == "silicon":
            seed_silicon_default_result(output_dir, default_sigma)
    else:
        requested_sigmas = SMALL_SYSTEM_SIGMAS

    results = []
    for requested_sigma in requested_sigmas:
        sigma = default_sigma if requested_sigma is None else requested_sigma
        logging.info("Running G0W0SS correlation sigma=%.12f 1/Bohr", sigma)
        result = run_gw(
            kmf,
            output_dir,
            requested_sigma,
            default_sigma,
            exchange_sigma,
            exchange_mode,
            settings,
        )
        results.append(result)
        logging.info(
            "sigma=%.12f: VBM=%.6f eV, CBM=%.6f eV, gap=%.6f eV",
            sigma,
            result["vbm_ev"],
            result["cbm_ev"],
            result["indirect_gap_ev"],
        )
    elapsed = time.perf_counter() - start
    write_summary(
        output_dir,
        metadata,
        results,
        kmf,
        default_sigma,
        elapsed,
        settings,
        requested_sigmas,
        exchange_sigma,
        exchange_mode,
    )
    logging.info("Sweep completed in %.2f s", elapsed)


if __name__ == "__main__":
    main()
