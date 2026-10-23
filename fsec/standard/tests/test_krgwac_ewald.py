"""Focused checks for periodic G0W0-AC Ewald corrections."""

import itertools
import os
from copy import copy
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from pyscf.pbc import df, dft, gto, scf
from pyscf.pbc.gw import krgw_ac

from fsec.standard import krgwac_ewald


def _cell(length=6.0):
    """Build a small cubic cell for mesh and Madelung checks."""
    cell = gto.Cell()
    cell.build(
        unit='Bohr', a=np.eye(3) * length, atom='H 0 0 0; H 1 1 1',
        dimension=3, pseudo='gth-pade', basis='gth-szv',
        precision=1e-8, verbose=0, output='/dev/null',
    )
    return cell


def _mesh_kpts(cell, shape, shift=(0.0, 0.0, 0.0)):
    """Create a complete uniform mesh from fractional coordinates."""
    points = [
        (np.asarray(index, dtype=float) / shape + shift) % 1.0
        for index in itertools.product(*(range(n) for n in shape))
    ]
    return cell.get_abs_kpts(points)


@pytest.fixture(scope='module')
def h2_2k_mf():
    """Run a compact two-k-point restricted H2 SCF calculation."""
    cell = gto.Cell()
    cell.build(
        unit='Bohr', a=np.eye(3) * 7.0,
        atom='H 2.2 3 2.2; H 4.2 3 4.2', dimension=3,
        verbose=0, output='/dev/null', pseudo='gth-pade', basis='gth-szv',
        precision=1e-8,
    )
    mf = dft.KRKS(cell, cell.make_kpts([2, 1, 1]))
    mf.with_df = df.GDF(cell, mf.kpts)
    mf.xc = 'pbe'
    mf.conv_tol = 1e-9
    mf.kernel()
    assert mf.converged
    # SCF needs only diagonal J blocks; periodic GW needs every k-point pair.
    mf.with_df.build(j_only=False)
    return mf


@pytest.fixture(scope='module')
def lih_shifted_mf():
    """Provide occupied core/valence states on a shifted mesh with RSDF."""
    cell = gto.M(
        atom='Li 3 3 2; H 4.3 3.1 4.7', a=np.eye(3) * 8, unit='Bohr',
        basis='sto-3g', precision=1e-8, verbose=0,
    )
    kpts = cell.make_kpts([2, 1, 1]) + cell.get_abs_kpts([0.11, 0.07, -0.05])
    mf = scf.KRHF(cell, kpts)
    mf.with_df = df.RSDF(cell, kpts)
    mf.conv_tol = 1e-9
    mf.kernel()
    assert mf.converged
    return mf


def _configure_gw(gw):
    """Use a compact but stable integration and continuation setup."""
    gw.nw = 20
    gw.ac_iw_cutoff = None
    gw.ac_pade_npts = 8
    gw.rdm = True
    gw.qpe_linearized = True
    return gw


def test_uniform_mesh_madelung_scaling_shift_and_reordering():
    """Recognize shifted complete meshes and preserve inverse-length scaling."""
    cell = _cell(length=6.0)
    shape = (2, 2, 2)
    kpts = _mesh_kpts(cell, shape)
    shifted = _mesh_kpts(cell, shape, shift=(0.13, 0.21, 0.37))
    reordered = shifted[[5, 2, 7, 0, 3, 6, 1, 4]]

    assert tuple(krgwac_ewald._uniform_mesh_shape(cell, shifted)) == shape
    chi = krgwac_ewald._madelung_constant(cell, kpts)
    assert chi == pytest.approx(2.837297479480619 / 12.0, abs=1e-9)
    elongated = krgwac_ewald._madelung_constant(cell, _mesh_kpts(cell, (4, 2, 1)))
    assert abs(elongated - chi) > 1e-3  # Same volume, different supercell shape.
    assert krgwac_ewald._madelung_constant(cell, shifted) == pytest.approx(chi)
    assert krgwac_ewald._madelung_constant(cell, reordered) == pytest.approx(chi)

    doubled = _cell(length=12.0)
    chi_doubled = krgwac_ewald._madelung_constant(
        doubled, _mesh_kpts(doubled, shape)
    )
    assert chi_doubled == pytest.approx(chi / 2.0, rel=1e-7)

    incomplete = kpts.copy()
    incomplete[1] += cell.get_abs_kpts([[0.07, 0.0, 0.0]])[0]
    with pytest.raises(ValueError, match='uniform MP mesh'):
        krgwac_ewald._uniform_mesh_shape(cell, incomplete)


def test_schur_head_and_scalar_correlation_convolution():
    """Compare block inversion and the analytic one-pole screened convolution."""
    eps_00 = 1.8
    wings = np.array([0.2 + 0.1j, -0.08j])
    body_inverse = np.array([[0.7, 0.05j], [-0.05j, 0.8]])
    dielectric = np.block([
        [np.array([[eps_00]]), wings.conj()[None, :]],
        [wings[:, None], np.linalg.inv(body_inverse)],
    ])
    assert krgwac_ewald._schur_inverse_head(
        eps_00, wings, body_inverse
    ) == pytest.approx(np.linalg.inv(dielectric)[0, 0])

    chi = 0.4
    freqs, weights = krgw_ac._get_scaled_legendre_roots(80)
    omega = np.array([0.0, 0.1 + 0.2j, 0.1 + 0.9j])
    correction = krgwac_ewald._correlation_head_correction(
        chi, np.ones_like(freqs), freqs, weights, omega, energy=-0.5
    )
    np.testing.assert_allclose(correction, 0.0, atol=1e-14)

    # h(i nu) = 1 - A/(nu**2+B**2), with an analytic integral over [0, inf).
    amplitude, pole = 0.6, 1.2
    head = 1.0 - amplitude / (freqs**2 + pole**2)
    for energy in (-0.5, 0.7):
        sign = np.sign(-energy)
        expected = chi * amplitude * sign / (
            2 * pole * (pole + sign * (omega - energy))
        )
        np.testing.assert_allclose(
            krgwac_ewald._correlation_head_correction(
                chi, head, freqs, weights, omega, energy
            ), expected, atol=1e-12, rtol=0,
        )


def test_ewald_rejects_unsupported_physics():
    """Reject mixed DF, metals, spin orbitals, low dimensions, and out-of-core."""
    cell = _cell()
    mf = scf.KRHF(cell, cell.make_kpts([1, 1, 1])).density_fit()
    mf.mo_energy = np.array([[-0.5, 0.5]])
    mf.mo_occ = np.array([[2.0, 0.0]])
    mf.mo_coeff = np.eye(2)[None, :, :]
    gw = krgwac_ewald.KRGWAC(mf)
    gw.mo_energy, gw.mo_occ, gw.mo_coeff = mf.mo_energy, mf.mo_occ, mf.mo_coeff
    krgwac_ewald._validate_ewald_setup(gw)
    slab = cell.copy()
    slab.dimension = 2
    for attr, value, message in (
        ('outcore', True, 'in-core'),
        ('with_df', df.MDF(cell, mf.kpts), 'GDF or RSDF'),
        ('mo_occ', np.array([[1.8, 0.2]]), 'integer occupations'),
        ('mo_occ', np.array([[[1.0, 0.0]], [[1.0, 0.0]]]), 'restricted scalar'),
        ('mo_energy', np.zeros((1, 2)), 'gapped'),
        ('mol', slab, '3D periodic'),
    ):
        candidate = copy(gw)
        setattr(candidate, attr, value)
        with pytest.raises(NotImplementedError, match=message):
            krgwac_ewald._validate_ewald_setup(candidate)


def test_exchange_correction_targets_active_occupied_diagonal():
    """Apply minus chi to occupied active orbitals only, including frozen maps."""
    frozen_gw = SimpleNamespace(
        frozen=1,
        mo_occ=np.array([[2.0, 2.0, 0.0]]),
        mo_energy=np.array([[-1.0, -0.4, 0.7]]),
        nkpts=1,
        nmo=2,
    )
    active_occ = krgw_ac._mo_occ_frozen(frozen_gw, frozen_gw.mo_occ)
    assert active_occ.tolist() == [[2.0, 0.0]]

    vk = np.arange(2 * 4 * 4, dtype=float).reshape(2, 4, 4).astype(complex)
    original = vk.copy()
    corrected = krgwac_ewald._add_exchange_head(
        vk, chi=0.25, nocc=int(np.sum(active_occ[0]) // 2)
    )
    for kpt in range(2):
        for orb in range(4):
            expected = original[kpt, orb, orb] - (0.25 if orb == 0 else 0.0)
            assert corrected[kpt, orb, orb] == pytest.approx(expected)
    mask = ~np.eye(4, dtype=bool)
    np.testing.assert_array_equal(corrected[:, mask], original[:, mask])


@pytest.mark.slow
def test_pyscf_long_wavelength_head_even_and_wing_odd(h2_2k_mf):
    """Check the actual PySCF response has an even head and odd nonzero wing."""
    gw = krgwac_ewald.KRGWAC(h2_2k_mf)
    gw.mo_energy = np.array(h2_2k_mf.mo_energy, copy=True)
    gw.mo_coeff = np.array(h2_2k_mf.mo_coeff, copy=True)
    gw.mo_occ = np.array(h2_2k_mf.mo_occ, copy=True)
    energies = krgw_ac._mo_energy_frozen(gw, gw.mo_energy)
    coeff = krgw_ac._mo_frozen(gw, gw.mo_coeff)
    lia = krgwac_ewald._load_zero_transfer_lia(gw, coeff)
    q_scaled = np.array([[1e-3, 0.0, 0.0]])
    q = gw.mol.get_abs_kpts(q_scaled)[0]
    qij_plus = krgw_ac.get_qij(gw, q, energies, coeff)
    qij_minus = krgw_ac.get_qij(gw, -q, energies, coeff)
    nu = 0.3

    head_plus = krgw_ac.get_rho_response_head(nu, energies, qij_plus)
    head_minus = krgw_ac.get_rho_response_head(nu, energies, qij_minus)
    wing_plus = krgw_ac.get_rho_response_wing(nu, energies, lia, qij_plus)
    wing_minus = krgw_ac.get_rho_response_wing(nu, energies, lia, qij_minus)

    assert head_plus == pytest.approx(head_minus, abs=1e-12)
    np.testing.assert_allclose(wing_plus, -wing_minus, atol=1e-12)
    assert np.linalg.norm(wing_plus) > 1e-8
    body = np.eye(lia.shape[1]) - krgw_ac.get_rho_response(
        nu, energies, lia, np.arange(gw.nkpts)
    )
    eps_00 = 1.0 - 4 * np.pi / np.linalg.norm(q)**2 * head_plus
    eps_p0 = -np.sqrt(4 * np.pi) / np.linalg.norm(q) * wing_plus
    dielectric = np.block([
        [np.array([[eps_00]]), eps_p0.conj()[None, :]],
        [eps_p0[:, None], body],
    ])
    actual = krgwac_ewald._inverse_dielectric_head(gw, [nu], energies, coeff, lia)[0]
    assert actual == pytest.approx(np.linalg.inv(dielectric)[0, 0], abs=1e-12)
    assert abs(actual - 1.0 / eps_00) > 1e-8


@pytest.mark.parametrize('fullsigma', [False, True])
def test_correlation_head_preserves_selected_indexing(fullsigma):
    """Add only selected diagonal entries for reordered bands and k-points."""
    if fullsigma:
        sigma = np.zeros((1, 2, 2, 3), dtype=complex)
    else:
        sigma = np.zeros((1, 2, 3), dtype=complex)
    if fullsigma:
        sigma[..., 0] = 2.0
    else:
        sigma[...] = 2.0
    original = sigma.copy()
    energies = np.array([[-0.5, 0.7], [-0.4, 0.8]])
    freqs = np.array([0.2, 0.6])
    weights = np.array([0.4, 0.6])
    head = np.array([0.8, 0.9])
    omega = np.array([0.0, 0.0 + 0.2j, 0.0 + 0.6j])

    krgwac_ewald._add_correlation_head(
        sigma, fullsigma, 0.3, head, freqs, weights, omega,
        energies, orbs=[1, 0], kptlist=[1],
    )
    expected = np.stack([
        krgwac_ewald._correlation_head_correction(
            0.3, head, freqs, weights, omega, energies[1, orb]
        )
        for orb in [1, 0]
    ])
    if fullsigma:
        np.testing.assert_allclose(sigma[0, 0, 0], original[0, 0, 0] + expected[0])
        np.testing.assert_allclose(sigma[0, 1, 1], original[0, 1, 1] + expected[1])
        np.testing.assert_allclose(sigma[0, 0, 1], original[0, 0, 1])
    else:
        np.testing.assert_allclose(sigma[0], original[0] + expected)


@pytest.mark.slow
def test_spherical_and_uncorrected_modes_match_upstream(h2_2k_mf):
    """Keep PySCF sigma, exchange, and QP results identical on delegated paths."""
    for method, enabled in (('spherical', True), ('ewald', False)):
        upstream = _configure_gw(krgw_ac.KRGWAC(h2_2k_mf))
        candidate = _configure_gw(krgwac_ewald.KRGWAC(h2_2k_mf))
        upstream.fc = enabled
        candidate.fc_method = method
        candidate.fc = enabled
        upstream.kernel(orbs=[1, 0], kptlist=[1])
        candidate.kernel(orbs=[1, 0], kptlist=[1])

        np.testing.assert_allclose(candidate.vk, upstream.vk, atol=1e-12, rtol=0)
        np.testing.assert_allclose(candidate.sigmaI, upstream.sigmaI, atol=1e-12, rtol=0)
        np.testing.assert_allclose(candidate.mo_energy, upstream.mo_energy, atol=1e-9, rtol=0)


@pytest.mark.slow
def test_ewald_kernel_corrects_selected_fullsigma(h2_2k_mf, tmp_path, monkeypatch):
    """Check Ewald sigma files, selected indexing, exchange, and no SCF energy changes."""
    mf_energy = h2_2k_mf.mo_energy.copy()
    mf_coeff = h2_2k_mf.mo_coeff.copy()
    mf_occ = h2_2k_mf.mo_occ.copy()
    monkeypatch.chdir(tmp_path)
    gw = _configure_gw(krgwac_ewald.KRGWAC(h2_2k_mf))
    source_df = df.GDF(gw.mol, gw.kpts)
    gw.with_df = source_df
    gw.fullsigma = True
    gw.fc_grid = True
    gw.ac = 'twopole'
    gw.writefile = 1
    assert gw.kernel(orbs=[1, 0], kptlist=[1]) is None

    assert gw.sigmaI.shape[0:3] == (1, 2, 2)
    assert np.isfinite(gw.sigmaI).all()
    assert gw.with_df is not source_df and source_df._cderi is None
    np.testing.assert_array_equal(gw.mo_energy[0], np.zeros_like(gw.mo_energy[0]))
    np.testing.assert_array_equal(h2_2k_mf.mo_energy, mf_energy)
    np.testing.assert_array_equal(h2_2k_mf.mo_coeff, mf_coeff)
    np.testing.assert_array_equal(h2_2k_mf.mo_occ, mf_occ)
    with h5py.File('sigma_imag.h5', 'r') as sigma_file:
        np.testing.assert_allclose(sigma_file['sigmaI'], gw.sigmaI)
        np.testing.assert_allclose(sigma_file['sigmaI_full'], gw.sigmaI)
    with h5py.File('vxc.h5', 'r') as exchange_file:
        np.testing.assert_array_equal(exchange_file['vk'], gw.vk)
    raw_vk = gw.get_sigma_exchange()
    chi = krgwac_ewald._madelung_constant(gw.mol, gw.kpts)
    np.testing.assert_allclose(gw.vk[:, 0, 0] - raw_vk[:, 0, 0], -chi, atol=1e-10)
    np.testing.assert_allclose(gw.vk[:, 1, 1], raw_vk[:, 1, 1], atol=1e-10)


@pytest.mark.slow
def test_shifted_rsdf_frozen_noncontiguous_selection(lih_shifted_mf):
    """Compare frozen, reordered full sigma against the complete upstream result."""
    mf = lih_shifted_mf
    original_energy = mf.mo_energy.copy()
    reference = _configure_gw(krgw_ac.KRGWAC(mf, frozen=1))
    reference.fc = False
    reference.fullsigma = True
    reference.kernel()

    gw = _configure_gw(krgwac_ewald.KRGWAC(mf, frozen=1))
    gw.fullsigma = True
    gw.qpe_linearized = False
    bands, kpoints = [4, 1, 3], [1, 0]
    gw.kernel(orbs=bands, kptlist=kpoints)
    active = np.array(bands) - 1
    expected = reference.sigmaI[kpoints][:, active][:, :, active].copy()
    energies = original_energy[:, 1:]
    coeff = krgw_ac._mo_frozen(gw, gw.mo_coeff)
    lia = krgwac_ewald._load_zero_transfer_lia(gw, coeff)
    head = krgwac_ewald._inverse_dielectric_head(gw, gw.freqs, energies, coeff, lia)
    chi = krgwac_ewald._madelung_constant(gw.mol, gw.kpts)
    omega = np.r_[0j, gw.ef + 1j * gw.freqs]
    for ik, k in enumerate(kpoints):
        for ip, p in enumerate(active):
            expected[ik, ip, ip] += krgwac_ewald._correlation_head_correction(
                chi, head, gw.freqs, gw.wts, omega, energies[k, p]
            )
    assert np.max(np.abs(head - 1.0)) > 1e-4
    np.testing.assert_allclose(gw.sigmaI, expected, atol=1e-9, rtol=0)
    exchange_delta = np.zeros_like(reference.vk)
    exchange_delta[:, 0, 0] = -chi
    np.testing.assert_allclose(gw.vk - reference.vk, exchange_delta, atol=1e-9, rtol=0)
    np.testing.assert_array_equal(mf.mo_energy, original_energy)
    assert np.all(np.isfinite(gw.mo_energy[:, bands]))
    assert np.all(gw.mo_energy[:, bands] != 0.0)


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get('FSEC_RUN_KRGWAC_EWALD_MESH_STUDY') != '1',
    reason='set FSEC_RUN_KRGWAC_EWALD_MESH_STUDY=1 to run the mesh study',
)
def test_ewald_spherical_uncorrected_mesh_study():
    """Record residual mesh dependence for Ewald, spherical, and raw G0W0."""
    records = {}
    for nk in (1, 2):
        cell = gto.Cell()
        cell.build(
            unit='Bohr', a=np.eye(3) * 7.0,
            atom='H 2.2 3 2.2; H 4.2 3 4.2', dimension=3,
            verbose=0, output='/dev/null', pseudo='gth-pade', basis='gth-szv',
            precision=1e-8,
        )
        mf = dft.KRKS(cell, cell.make_kpts([nk, 1, 1]))
        mf.with_df = df.GDF(cell, mf.kpts)
        mf.xc = 'pbe'
        mf.conv_tol = 1e-8
        mf.kernel()
        assert mf.converged
        mf.with_df.build(j_only=False)
        for label, method, enabled in (
            ('ewald', 'ewald', True),
            ('spherical', 'spherical', True),
            ('uncorrected', 'ewald', False),
        ):
            gw = _configure_gw(krgwac_ewald.KRGWAC(mf))
            gw.nw = 16
            gw.ac_pade_npts = 6
            gw.fc_method = method
            gw.fc = enabled
            gw.kernel(orbs=[0, 1], kptlist=[0])
            records[(nk, label)] = float(gw.mo_energy[0, 1] - gw.mo_energy[0, 0])

    residuals = {
        label: records[(1, label)] - records[(2, label)]
        for label in ('ewald', 'spherical', 'uncorrected')
    }
    print('G0W0 gap mesh residuals (1x1x1 minus 2x1x1, Ha):', residuals)
    assert np.isfinite(list(records.values())).all()
