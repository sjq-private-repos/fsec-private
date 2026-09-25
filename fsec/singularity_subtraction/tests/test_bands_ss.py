"""Scientific checks for orbital-resolved band singularity subtraction."""

import copy

import numpy as np
import pytest
from pyscf.pbc import df, dft, gto, scf
from pyscf.pbc.df import ft_ao

from fsec.singularity_subtraction import (
    BandsSS,
    ExxSSGaussian,
    fourier_exchange_matrices,
    orbital_resolved_structure_factor,
)
from fsec.singularity_subtraction.structure_factor.exx_sf import ExxStructureFactor


@pytest.fixture(scope="module")
def source_mf():
    """A compact shifted, two-occupied-orbital periodic HF reference."""
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
    kpts = cell.make_kpts(
        (1, 1, 2), scaled_center=(0.11, 0.07, 0.13),
        wrap_around=True, with_gamma_point=False,
    )
    kmf = scf.KRHF(cell, kpts)
    kmf.exxdiv = "ewald"
    kmf.conv_tol = 1e-10
    kmf.kernel()
    if not kmf.converged:
        raise RuntimeError("The shifted H4 KRHF reference did not converge")
    return kmf


def _supplied_bands(kmf, kpts_band):
    """Give each test target the first source point's orbitals."""
    energies = np.repeat(np.asarray(kmf.mo_energy[0])[None, :], len(kpts_band), axis=0)
    coeff = np.repeat(np.asarray(kmf.mo_coeff[0])[None, :, :], len(kpts_band), axis=0)
    return energies, coeff


def _independent_gaussian_xi(cell, source_kpts, target_kpt, sigma, q_zero_tol):
    """Large-box reciprocal sum independent of BandsSS's image generator."""
    Gv = cell.get_Gv(mesh=(61, 61, 61))
    total = 0.0
    for source_kpt in source_kpts:
        p = target_kpt - source_kpt + Gv
        p2 = np.einsum("gi,gi->g", p, p)
        keep = (p2 > q_zero_tol**2) & (p2 <= (8.0 * sigma)**2 + 1e-12)
        total += np.sum(4.0 * np.pi * np.exp(-p2[keep] / (2.0 * sigma**2)) / p2[keep])
    quadrature = total / (cell.vol * len(source_kpts))
    return quadrature - np.sqrt(2.0 / np.pi) * sigma


def test_orbital_structure_factor_normalization_and_total_average(source_mf):
    """At p=0 each occupied S is one, and its mesh average recovers Exx S."""
    kmf = source_mf
    nocc = np.count_nonzero(kmf.mo_occ[0] == 2)
    mesh = kmf.cell.mesh
    Gv = kmf.cell.get_Gv(mesh=mesh)
    zero_g = int(np.argmin(np.linalg.norm(Gv, axis=1)))
    accumulated = {}

    for ik, kpt in enumerate(kmf.kpts):
        p, S, source_ids, g_ids = orbital_resolved_structure_factor(
            kmf, kpt, kmf.mo_coeff[ik], mesh=mesh,
        )
        at_same_k = (source_ids == ik) & (g_ids == zero_g)
        np.testing.assert_allclose(S[at_same_k, :], np.ones((1, nocc)), atol=2e-8)
        for transfer, orbital_values in zip(p, S):
            key = tuple(np.round(transfer, 9))
            accumulated[key] = accumulated.get(key, 0.0) + np.sum(orbital_values)

    total_sf = ExxStructureFactor(kmf, N_local=mesh, sq_inversion_symm=False)
    total_values = total_sf.build_structure_factor()
    nk = len(kmf.kpts)
    compared = 0
    # Each transfer collects all target/source orbital pairs; the total
    # structure factor applies one source-k average.
    for point, reference in zip(total_sf.grids.qG_grid_truncated, total_values):
        key = tuple(np.round(point, 9))
        if key in accumulated:
            np.testing.assert_allclose(accumulated[key] / nk, reference, atol=2e-6, rtol=2e-6)
            compared += 1
    assert compared == len(total_values)


def test_fourier_exchange_matches_pyscf_aft_off_mesh(source_mf):
    """The blocked AO Fourier contraction matches PySCF AFT at regular q."""
    kmf = source_mf
    target = np.asarray([[0.033, -0.020, 0.014]])
    dm = np.asarray(kmf.make_rdm1())
    actual = fourier_exchange_matrices(kmf, dm, target, q_zero_tol=1e-4)
    aft = df.AFTDF(kmf.cell, kmf.kpts)
    _, expected = aft.get_jk(
        dm,
        hermi=1,
        kpts=kmf.kpts,
        kpts_band=target,
        with_j=False,
        with_k=True,
        exxdiv=None,
    )
    np.testing.assert_allclose(actual, np.asarray(expected).reshape(actual.shape), atol=3e-7, rtol=2e-7)


def test_shared_gaussian_recovers_total_exxss_correction(source_mf):
    """A common Gaussian makes the band xi equal the normalized ExxSS shift."""
    kmf = source_mf
    sigma = 0.65
    bands = BandsSS(
        kmf,
        kmf.kpts,
        kmf.mo_energy,
        kmf.mo_coeff,
        sigma=sigma,
        N_local=kmf.cell.mesh,
        q_zero_tol=1e-4,
    )
    bands.compute_correction()
    exxss = ExxSSGaussian(
        kmf,
        sigma=sigma,
        fit_to_structure_factor=False,
        qG_norm_cutoff_sigma=None,
        N_local=kmf.cell.mesh,
        sq_inversion_symm=False,
    )
    exxss.compute_correction()
    nocc = bands.nocc
    np.testing.assert_allclose(bands.xi, bands.xi[0, 0], atol=2e-8, rtol=2e-8)
    np.testing.assert_allclose(np.mean(bands.xi), exxss.correction / nocc, atol=3e-6, rtol=2e-6)


def test_shifted_gaussian_quadrature_uses_same_cutoff_below_at_and_above(source_mf):
    """The actual shifted transfer grid and shared cutoff match an independent sum."""
    kmf = source_mf
    tau = 1e-4
    sigma = 0.65
    targets = np.asarray([
        kmf.kpts[0] + [0.5 * tau, 0.0, 0.0],
        kmf.kpts[0] + [tau, 0.0, 0.0],
        kmf.kpts[0] + [1.5 * tau, 0.0, 0.0],
    ])
    energies, coeff = _supplied_bands(kmf, targets)
    bands = BandsSS(
        kmf, targets, energies, coeff, sigma=sigma,
        q_zero_tol=tau, N_local=kmf.cell.mesh,
    )
    correction = bands.compute_correction()
    expected = np.asarray([
        _independent_gaussian_xi(kmf.cell, kmf.kpts, target, sigma, tau)
        for target in targets
    ])
    np.testing.assert_allclose(bands.xi[:, 0], expected, atol=2e-10, rtol=1e-12)
    np.testing.assert_allclose(correction[:, :bands.nocc], bands.xi, atol=1e-12)
    np.testing.assert_array_equal(correction[:, bands.nocc:], 0.0)

    dm = np.asarray(kmf.make_rdm1())
    volume = kmf.cell.vol
    for target in targets:
        q = target - kmf.kpts[0]
        K_unmasked = fourier_exchange_matrices(kmf, dm, [target], q_zero_tol=0.0)[0]
        K_masked = fourier_exchange_matrices(kmf, dm, [target], q_zero_tol=tau)[0]
        B0 = ft_ao.ft_aopair(
            kmf.cell,
            np.zeros((1, 3)),
            kpti_kptj=np.asarray([kmf.kpts[0], target]),
        )[0]
        missing = (4.0 * np.pi / (np.dot(q, q) * volume * len(kmf.kpts))) * (
            B0.conj().T @ dm[0] @ B0
        )
        missing = 0.5 * (missing + missing.conj().T)
        if np.linalg.norm(q) <= tau:
            np.testing.assert_allclose(K_unmasked - K_masked, missing, atol=2e-7, rtol=2e-8)
        else:
            np.testing.assert_allclose(K_unmasked, K_masked, atol=2e-7, rtol=2e-8)

    # For S=h, the negative bare quadrature plus xi is exactly the full-space
    # Gaussian integral, independent of whether a shifted grid samples zero.
    Q = expected + np.sqrt(2.0 / np.pi) * sigma
    np.testing.assert_allclose(
        -Q + bands.xi[:, 0], -np.sqrt(2.0 / np.pi) * sigma, atol=2e-8, rtol=0,
    )


def test_hybrid_scaling_is_applied_once_and_virtual_shifts_are_zero(source_mf):
    """PBE0 applies its quarter-exchange fraction to xi and preserves virtuals."""
    hf = source_mf
    kmf = dft.KRKS(hf.cell, hf.kpts)
    kmf.xc = "PBE0"
    kmf.mo_coeff = np.array(hf.mo_coeff, copy=True)
    kmf.mo_energy = np.array(hf.mo_energy, copy=True)
    kmf.mo_occ = np.array(hf.mo_occ, copy=True)
    targets = hf.kpts[:1]
    bands = BandsSS(
        kmf, targets, hf.mo_energy[:1], hf.mo_coeff[:1],
        sigma=0.7, N_local=hf.cell.mesh,
    )
    correction = bands.compute_correction()
    assert bands.hybrid_fraction == pytest.approx(0.25)
    np.testing.assert_allclose(correction[0, :bands.nocc], 0.25 * bands.xi[0], atol=1e-12)
    np.testing.assert_array_equal(correction[0, bands.nocc:], 0.0)


def test_generated_and_supplied_bands_agree_without_changing_source(source_mf):
    """Generated tiny-q bands can be replayed as supplied masked baselines."""
    kmf = source_mf
    targets = np.asarray([[1e-7, 0.0, 0.0], [0.033, -0.020, 0.014]])
    source_dm = np.array(kmf.make_rdm1(), copy=True)
    source_coeff = np.array(kmf.mo_coeff, copy=True)
    source_summary = copy.deepcopy(kmf.scf_summary)
    source_df = kmf.with_df
    source_exxdiv = kmf.exxdiv

    generated = BandsSS(kmf, targets, q_zero_tol=1e-4, N_local=kmf.cell.mesh)
    correction = generated.compute_correction()
    generated_energies, generated_coeff = generated.get_bands()
    replay = BandsSS(
        kmf,
        targets,
        generated.mo_energy_band_baseline,
        generated.mo_coeff_band,
        q_zero_tol=1e-4,
        N_local=kmf.cell.mesh,
    )
    replay_energies, replay_coeff = replay.get_bands()

    assert generated_energies.shape == correction.shape == (len(targets), source_coeff.shape[-1])
    assert np.all(np.isfinite(generated_energies))
    np.testing.assert_allclose(replay_energies, generated_energies, atol=2e-8, rtol=0)
    np.testing.assert_array_equal(replay_coeff, generated_coeff)
    np.testing.assert_array_equal(correction[:, generated.nocc:], 0.0)
    np.testing.assert_array_equal(kmf.make_rdm1(), source_dm)
    np.testing.assert_array_equal(kmf.mo_coeff, source_coeff)
    assert kmf.with_df is source_df
    assert kmf.exxdiv == source_exxdiv
    assert kmf.scf_summary == source_summary
    assert np.max(np.abs(generated.sigma[1, 0] - generated.sigma[1, 1])) > 1e-4


def test_reciprocal_image_and_source_order_do_not_change_fixed_fit(source_mf):
    """Canonical transfer images and source ordering leave xi unchanged."""
    kmf = source_mf
    target = np.asarray([[0.033, -0.020, 0.014]])
    energies, coeff = _supplied_bands(kmf, target)
    reference = BandsSS(kmf, target, energies, coeff, sigma=0.7)
    reference.compute_correction()

    reciprocal_shift = kmf.cell.reciprocal_vectors()[0]
    shifted = BandsSS(kmf, target + reciprocal_shift, energies, coeff, sigma=0.7)
    shifted.compute_correction()
    np.testing.assert_allclose(shifted.xi, reference.xi, atol=2e-7, rtol=0)

    permuted_mf = copy.deepcopy(kmf)
    order = np.arange(len(kmf.kpts))[::-1]
    permuted_mf.kpts = np.array(kmf.kpts[order], copy=True)
    permuted_mf.mo_coeff = np.array(kmf.mo_coeff[order], copy=True)
    permuted_mf.mo_energy = np.array(kmf.mo_energy[order], copy=True)
    permuted_mf.mo_occ = np.array(kmf.mo_occ[order], copy=True)
    permuted = BandsSS(permuted_mf, target, energies, coeff, sigma=0.7)
    permuted.compute_correction()
    np.testing.assert_allclose(permuted.xi, reference.xi, atol=2e-7, rtol=0)


def test_fractional_source_occupations_are_rejected(source_mf):
    """Fractional source occupations fall outside the restricted v1 method."""
    fractional = copy.deepcopy(source_mf)
    fractional.mo_occ = np.array(fractional.mo_occ, copy=True)
    fractional.mo_occ[0, 0] = 1.0
    with pytest.raises(NotImplementedError, match="Fractional occupations"):
        BandsSS(fractional, [[0.02, 0.01, 0.03]])


@pytest.mark.parametrize("mesh", [(2, 2, 2), (3, 3, 3)])
def test_skew_cell_mesh_validation_handles_periodic_boundary_roundoff(mesh):
    """Wrapped fractional zeros at skew-cell boundaries remain a full mesh."""
    cell = gto.Cell()
    cell.unit = "Bohr"
    cell.atom = "H 0.0 0.0 0.0; H 1.0 1.0 1.0"
    cell.a = np.asarray([[0.0, 3.37, 3.37], [3.37, 0.0, 3.37], [3.37, 3.37, 0.0]])
    cell.basis = "gth-szv"
    cell.pseudo = "gth-hf"
    cell.ke_cutoff = 20
    cell.verbose = 0
    cell.build()
    kpts = cell.make_kpts(mesh, wrap_around=True, with_gamma_point=True)
    kmf = scf.KRHF(cell, kpts)
    nocc = cell.tot_electrons() // 2
    kmf.mo_occ = np.tile(np.r_[np.full(nocc, 2.0), np.zeros(cell.nao_nr() - nocc)],
                         (len(kpts), 1))
    BandsSS(kmf, kpts[:1])
