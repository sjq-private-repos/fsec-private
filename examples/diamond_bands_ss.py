#!/usr/bin/env python
"""Plot diamond KRHF bands with orbital-resolved singularity subtraction.

Run from the repository root with ``python -m examples.diamond_bands_ss``.
Use ``--plot-only`` to redraw the saved numerical results without an SCF run.
Requires ASE and Matplotlib in addition to the usual fsec dependencies.
Use ``OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4`` to avoid BLAS oversubscription.
"""

import argparse
import copy
import json
from pathlib import Path
import time

from ase.cell import Cell as ASECell
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from pyscf.pbc import df, gto, scf

from fsec.singularity_subtraction import BandsSS


HARTREE_TO_EV = 27.211386245988


def calculate(output, npoints, batch_size):
    """Keep the converged source density fixed while evaluating band batches."""
    cell = gto.Cell()
    cell.unit = "Bohr"
    cell.atom = "C 0 0 0; C 1.68516327271508 1.68516327271508 1.68516327271508"
    half_a = 3.370326545430162
    cell.a = half_a * (np.ones((3, 3)) - np.eye(3))
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pbe"
    cell.ke_cutoff = 100
    cell.precision = 1e-8
    cell.max_memory = 4000
    cell.verbose = 4
    cell.build()
    # Resolve the real-space AO products and pseudopotential more tightly than
    # the nominal 100-Ha mesh (23^3 for this cell).
    cell.mesh = np.maximum(cell.mesh, 37)
    kpts = cell.make_kpts([2, 2, 2], wrap_around=True, with_gamma_point=True)
    kmf = scf.KRHF(cell, kpts)
    kmf.exxdiv = "ewald"
    kmf.conv_tol = 1e-9
    kmf.chkfile = str(output / "source.chk")
    if Path(kmf.chkfile).exists():
        kmf.init_guess = "chkfile"
    kmf.with_df = df.FFTDF(cell, kpts)
    started = time.monotonic()
    kmf.kernel()
    if not kmf.converged:
        raise RuntimeError("The diamond source KRHF calculation did not converge")
    print(f"Source SCF complete in {time.monotonic() - started:.1f} s", flush=True)

    path = ASECell(cell.lattice_vectors()).bandpath("LGXWKG", npoints=npoints)
    kpts_band = cell.get_abs_kpts(path.kpts)
    # Native FFTDF excludes exact zero with exxdiv=None. This path has no
    # nonzero transfer inside the BandsSS exclusion radius, so masks agree.
    scaled_q = path.kpts[:, None, :] - cell.get_scaled_kpts(kpts)[None, :, :]
    images = np.asarray(np.meshgrid(*([np.arange(-2, 3)] * 3), indexing="ij"))
    images = images.reshape(3, -1).T
    transfers = (scaled_q[:, :, None, :] + images) @ cell.reciprocal_vectors()
    norms = np.linalg.norm(transfers, axis=-1)
    assert not np.any((norms > 1e-9) & (norms <= 1e-4))
    bare_mf = copy.copy(kmf)
    bare_mf.exxdiv = None
    distance, nodes, labels = path.get_linear_kpoint_axis()
    bare, corrected, sigmas, xis, coefficients = [], [], [], [], []
    for start in range(0, len(kpts_band), batch_size):
        stop = min(start + batch_size, len(kpts_band))
        # A nonzero helper keeps PySCF's single-Gamma Fock allocation complex.
        # It is discarded immediately and does not change any requested point.
        helper = cell.get_abs_kpts([0.23, 0.17, 0.11])
        eval_kpts = np.vstack((kpts_band[start:stop], helper))
        baseline, baseline_coeff = bare_mf.get_bands(eval_kpts)
        baseline = baseline[:stop-start]
        baseline_coeff = baseline_coeff[:stop-start]
        ss = BandsSS(kmf, kpts_band[start:stop], baseline, baseline_coeff,
                     q_zero_tol=1e-4, block_size=4096)
        energies, coeff = ss.get_bands()
        bare.extend(ss.mo_energy_band_baseline)
        corrected.extend(energies)
        sigmas.extend(ss.sigma)
        xis.extend(ss.xi)
        coefficients.extend(coeff)
        print(f"Bands {stop}/{len(kpts_band)} complete; elapsed "
              f"{time.monotonic() - started:.1f} s", flush=True)
        np.savez_compressed(
            output / "bands.npz", distance=distance[:stop], nodes=nodes,
            labels=labels, kpts_band=kpts_band[:stop], scaled_kpts=path.kpts[:stop],
            bare=np.asarray(bare), corrected=np.asarray(corrected),
            sigma=np.asarray(sigmas), xi=np.asarray(xis),
            coefficients=np.asarray(coefficients), nocc=ss.nocc,
            source_kpts=kpts, source_energy=kmf.e_tot,
            q_zero_tol=ss.q_zero_tol, exchange_mesh=ss.exchange_mesh,
        )
    assert np.all(np.isfinite(corrected))
    assert np.array_equal(np.asarray(corrected)[:, ss.nocc:],
                          np.asarray(bare)[:, ss.nocc:])


def plot(output, vcut_sph=None):
    """Use one energy reference for bare and corrected bands."""
    with np.load(output / "bands.npz") as data:
        bare = data["bare"]
        corrected = data["corrected"]
        nocc = int(data["nocc"])
        distance, nodes, labels = data["distance"], data["nodes"], data["labels"]
        if not np.isclose(distance[-1], nodes[-1]):
            raise RuntimeError("Band sampling is incomplete")
        reference = float(np.max(corrected[:, :nocc]))
        ss_ev = (corrected - reference) * HARTREE_TO_EV
        bare_ev = (bare - reference) * HARTREE_TO_EV
        path_gap = float(np.min(ss_ev[:, nocc:]) - np.max(ss_ev[:, :nocc]))
        metadata = {
            "method": "KRHF; fixed source density; orbital-resolved BandsSS",
            "basis": "gth-szv", "pseudopotential": "gth-pbe",
            "source_mesh": [2, 2, 2], "source_exxdiv": "ewald",
            "density_fitting": "FFTDF on 37x37x37 grid; supplied bare bands",
            "band_exchange": "bare Coulomb with common near-zero exclusion",
            "q_zero_tol_bohr_inverse": float(data["q_zero_tol"]),
            "exchange_mesh": data["exchange_mesh"].tolist(),
            "nominal_plane_wave_cutoff_hartree": 100,
            "path": "L-Gamma-X-W-K-Gamma", "number_of_points": len(distance),
            "energy_zero_hartree": reference, "sampled_path_gap_eV": path_gap,
            "note": "Energy zero is the maximum corrected occupied energy on this path. "
                    "The sampled path gap is not a full-Brillouin-zone gap. "
                    "Virtual SS shifts are zero; orbitals are not rediagonalized.",
        }
        if vcut_sph is None:
            (output / "settings.json").write_text(json.dumps(metadata, indent=2) + "\n")

    if vcut_sph is not None:
        vcut_sph = np.asarray(vcut_sph)
        if vcut_sph.shape != corrected.shape or not np.all(np.isfinite(vcut_sph)):
            raise ValueError("The vcut_sph bands must match the complete SS band array")
        vcut_ev = (vcut_sph - reference) * HARTREE_TO_EV

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 6.3), sharex=True, sharey=True)
    occupied_color, virtual_color, bare_color = "#1762A0", "#52616F", "#C66C36"
    lower = np.floor(np.min(ss_ev[:, :nocc]) / 5) * 5 - 2
    upper = np.ceil(np.max(ss_ev[:, nocc:]) / 5) * 5 + 2
    for ax in axes:
        for node in nodes:
            ax.axvline(node, color="#D8DFE6", lw=0.8, zorder=0)
        ax.axhline(0, color="#8B99A5", lw=0.8, ls=":", zorder=0)
        ax.set_xticks(nodes, [r"$\Gamma$" if label == "G" else label for label in labels])
        ax.set_xlim(distance[0], distance[-1])
        ax.set_ylim(lower, upper)
        ax.set_xlabel("Crystal momentum")
    axes[0].plot(distance, ss_ev[:, :nocc], color=occupied_color, lw=1.7)
    axes[0].plot(distance, ss_ev[:, nocc:], color=virtual_color, lw=1.5)
    axes[0].set_title("Bands SS", loc="left", fontweight="bold", pad=12)
    axes[0].set_ylabel("Energy relative to corrected valence maximum (eV)")
    axes[1].plot(distance, bare_ev[:, :nocc], color=bare_color, lw=1.1,
                 ls="--", alpha=0.85)
    axes[1].plot(distance, ss_ev[:, :nocc], color=occupied_color, lw=1.7)
    axes[1].plot(distance, ss_ev[:, nocc:], color=virtual_color, lw=1.3, alpha=0.6)
    axes[1].set_title("Effect of singularity subtraction", loc="left",
                      fontweight="bold", pad=12)
    handles = [
        Line2D([], [], color=occupied_color, lw=1.7, label="SS occupied"),
        Line2D([], [], color=bare_color, lw=1.1, ls="--", label="Bare occupied"),
        Line2D([], [], color=virtual_color, lw=1.5,
               label=("Virtual (zero SS shift)" if vcut_sph is None
                      else "SS virtual (zero SS shift)")),
    ]
    if vcut_sph is not None:
        for ax in axes:
            ax.plot(distance, vcut_ev, color="#17845E", lw=1.25, ls="--")
        handles.append(Line2D([], [], color="#17845E", lw=1.25, ls="--",
                              label=r"FFTDF + $v_{\mathrm{cut,sph}}$"))
        axes[0].set_title("SS and spherical cutoff", loc="left", fontweight="bold", pad=12)
        axes[1].set_title("Bare, SS, and spherical cutoff", loc="left", fontweight="bold", pad=12)
    axes[1].legend(handles=handles, loc="center right", bbox_to_anchor=(0.99, 0.60),
       frameon=False, fontsize=9)
    fig.suptitle("Diamond  |  orbital-resolved singularity subtraction",
                 x=0.08, ha="left", fontsize=17, fontweight="bold", y=0.98)
    fig.text(0.08, 0.92, "Restricted Hartree–Fock · GTH-SZV / GTH-PBE · 2×2×2 source mesh · FFTDF (37³)",
             color="#52616F", fontsize=11)
    footer = ("Bare Coulomb kernel; common exclusion |q + G| ≤ 10⁻⁴ bohr⁻¹. "
              "Bare poles extend below the displayed energy range.")
    if vcut_sph is not None:
        footer = ("Green: self-consistent FFTDF + vcut_sph. All curves use the SS valence maximum as zero. "
                  "Bare poles extend below the displayed range.")
    fig.text(0.08, 0.035, footer, fontsize=9,
             color="#52616F")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.84, wspace=0.12)
    stem = "diamond_bands_ss" if vcut_sph is None else "diamond_bands_ss_vcut_sph_overlay"
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=220, facecolor="white")
    plt.close(fig)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/diamond_bands_ss"))
    parser.add_argument("--npoints", type=int, default=81)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        calculate(args.output, args.npoints, args.batch_size)
    plot(args.output)
