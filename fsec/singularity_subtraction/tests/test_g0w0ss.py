"""Focused numerical checks for Gaussian G0W0 finite-size helpers."""

import numpy as np
import pytest
from scipy.integrate import quad

from pyscf import lib
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.incore import _conc_mos
from pyscf.pbc import df, gto, scf
from pyscf.pbc.gw import krgw_ac as pyscf_krgw_ac

from fsec.singularity_subtraction import G0W0SS
from fsec.singularity_subtraction import g0w0ss
from fsec.singularity_subtraction.g0w0ss import (
    GaussianCoefficients,
    compute_gaussian_coefficients,
    get_sigma,
    inverse_dielectric_head_wing,
    monkhorst_pack_shape,
    reciprocal_supercell_lattice,
)
from fsec.singularity_subtraction.g0w0ss import _fit_two_pole_per_element, _set_frozen_orbs


def _cell(lattice=None):
    cell = gto.Cell()
    cell.unit = "Bohr"
    cell.atom = "He 0 0 0"
    cell.a = (np.array([[4.0, 0.2, 0.1], [0.1, 5.0, 0.3], [0.2, 0.4, 6.0]])
              if lattice is None else lattice)
    cell.basis = "sto-3g"
    cell.verbose = 0
    cell.build()
    return cell


@pytest.fixture(scope="module")
def reference_mf():
    """Build one shifted/reordered skew-cell insulating reference system."""
    cell = gto.Cell()
    cell.atom = "H 0 0 0; H 0.3 0.2 1.8"
    cell.unit = "Bohr"
    cell.a = np.array([[6.0, 0, 0], [1.7, 5.6, 0], [0.4, 1.3, 6.2]])
    cell.basis = "gth-szv"
    cell.pseudo = "gth-hf"
    cell.precision = 1e-8
    cell.verbose = 0
    cell.build()
    kpts = cell.make_kpts([2, 1, 1]) + cell.get_abs_kpts([0.13, 0.07, 0.11])
    kpts = kpts[::-1].copy()
    mf = scf.KRHF(cell, kpts)
    mf.exxdiv = "ewald"
    mf.conv_tol = 1e-10
    mf.with_df = df.GDF(cell, kpts).build()
    mf.kernel()
    assert mf.converged
    return mf


def _initialized_gw(cls, mf):
    gw = cls(mf)
    gw.mo_coeff = np.array(mf.mo_coeff)
    gw.mo_energy = np.array(mf.mo_energy)
    gw.mo_occ = np.array(mf.mo_occ)
    return gw


def _direct_dielectric_additions(gw, freqs, wts, omega):
    """Reconstruct additions by inverting the full dielectric matrix directly."""
    diagonal_integrals = []
    nao = gw.mo_coeff.shape[1]
    cderi = gw.with_df.cderi_array()
    for k, kpt in enumerate(gw.kpts):
        lpq = cderi.load(kpt, kpt)
        if lpq.shape[-1] == nao * (nao + 1) // 2:
            lpq = lib.unpack_tril(lpq)
        mos, slc = _conc_mos(gw.mo_coeff[k], gw.mo_coeff[k])[2:]
        transformed = _ao2mo.r_e2(lpq.reshape(-1, nao**2).astype(complex),
                                   mos, slc, tao=[], ao_loc=None)
        diagonal_integrals.append(transformed.reshape(-1, gw.nmo, gw.nmo))
    lij = np.asarray(diagonal_integrals)
    lia = lij[:, :, :gw.nocc, gw.nocc:]
    q = gw.mol.get_abs_kpts([1e-3, 0, 0])
    qnorm = np.linalg.norm(q)
    qij = pyscf_krgw_ac.get_qij(gw, q, gw.mo_energy, gw.mo_coeff)
    emo = omega[None, None, :] - gw.mo_energy[:, :, None]
    head = np.zeros_like(emo)
    wing = np.zeros_like(emo)
    for iw, frequency in enumerate(freqs):
        pi = pyscf_krgw_ac.get_rho_response(frequency, gw.mo_energy, lia, np.arange(gw.nkpts))
        p00 = pyscf_krgw_ac.get_rho_response_head(frequency, gw.mo_energy, qij)
        p0 = pyscf_krgw_ac.get_rho_response_wing(frequency, gw.mo_energy, lia, qij)
        e0 = -np.sqrt(4 * np.pi) / qnorm * p0
        dielectric = np.block([
            [np.array([[1 - 4 * np.pi * p00 / qnorm**2]]), e0.conj()[None, :]],
            [e0[:, None], np.eye(len(pi)) - pi],
        ])
        inverse = np.linalg.inv(dielectric)
        factor = -wts[iw] * emo / (emo**2 + frequency**2) / np.pi
        head += gw.gaussian_coefficients.head * (inverse[0, 0] - 1) * factor
        contraction = 2 * np.sum(np.diagonal(lij, axis1=2, axis2=3)
                                  * inverse[None, 1:, 0, None], axis=1).real
        wing += gw.gaussian_coefficients.wing * contraction[:, :, None] * factor
    return head, wing


def test_shifted_reordered_mesh_shape():
    """Mesh construction ignores a common shift and ordering."""
    cell = _cell()
    kpts = cell.make_kpts([2, 1, 2], wrap_around=True, with_gamma_point=True)
    shifted = kpts + cell.get_abs_kpts([0.13, 0.07, 0.11])
    assert np.array_equal(monkhorst_pack_shape(cell, shifted[::-1]), [2, 1, 2])
    assert G0W0SS.__name__ == "G0W0SS"


def test_gaussian_coefficients_cutoff_convergence_and_lattice_bound():
    """The reciprocal sum converges when the Gaussian cutoff is extended."""
    for cell in (_cell(), _cell(np.eye(3) * 6)):
        for mesh in ([1, 1, 1], [2, 1, 1]):
            c8 = compute_gaussian_coefficients(cell, mesh=mesh, cutoff_sigma=8)
            c10 = compute_gaussian_coefficients(cell, mesh=mesh, cutoff_sigma=10)
            np.testing.assert_allclose([c10.head, c10.wing], [c8.head, c8.wing], atol=1e-12)
            assert c8.head == c8.head_integral + c8.head_quadrature
            assert c8.wing == c8.wing_integral + c8.wing_quadrature
            assert np.array_equal(monkhorst_pack_shape(cell, cell.make_kpts(mesh)), mesh)


def test_gaussian_radial_integrals_match_closed_forms():
    """The continuum radial integrals retain the head and wing factors."""
    cell = _cell()
    for mesh in ([1, 1, 1], [2, 1, 3]):
        c = compute_gaussian_coefficients(cell, mesh=mesh)
        sigma = (6 * np.pi**2 / (cell.vol * np.prod(mesh))) ** (1 / 3)
        h = lambda q: np.exp(-q * q / (2 * sigma**2))
        # Angular measure 4*pi and Fourier measure (2*pi)^-3 are explicit.
        head = 4 * np.pi * 4 * np.pi / (2 * np.pi)**3 * quad(h, 0, np.inf)[0]
        wing = (np.sqrt(4 * np.pi * cell.vol) * 4 * np.pi / (2 * np.pi)**3
                * quad(lambda q: q * h(q), 0, np.inf)[0])
        np.testing.assert_allclose([c.sigma, c.head_integral, c.wing_integral],
                                   [sigma, head, wing], rtol=1e-12, atol=1e-12)


def test_skew_lattice_enumeration_contains_bruteforce_vectors():
    """Inverse-basis bounds include every short vector in a skew cell."""
    cell = _cell(np.array([[6., 0, 0], [5.6, 1.2, 0], [0.4, 0.3, 5.]]))
    coefficients = compute_gaussian_coefficients(cell, mesh=[2, 1, 1])
    cutoff = 8 * coefficients.sigma
    basis = cell.reciprocal_vectors() / np.asarray([2, 1, 1])[:, None]
    vectors = reciprocal_supercell_lattice(cell, [2, 1, 1], cutoff)
    # A singular-value bound gives an independent, deliberately larger box.
    bound = int(np.ceil(cutoff / np.linalg.svd(basis, compute_uv=False).min()))
    indices = np.indices((2 * bound + 1,) * 3).reshape(3, -1).T - bound
    brute = indices @ basis
    norms = np.linalg.norm(brute, axis=1)
    brute = brute[(norms > 0) & (norms <= cutoff * (1 + 1e-13))]
    assert {tuple(x) for x in np.round(vectors, 10)} == {tuple(x) for x in np.round(brute, 10)}
    q2 = np.einsum('gi,gi->g', brute, brute)
    h = np.exp(-q2 / (2 * coefficients.sigma**2))
    np.testing.assert_allclose(coefficients.head_quadrature,
                               -4 * np.pi / (cell.vol * 2) * np.sum(h / q2), rtol=1e-12)
    np.testing.assert_allclose(coefficients.wing_quadrature,
                               -np.sqrt(4 * np.pi * cell.vol) / (cell.vol * 2)
                               * np.sum(h / np.sqrt(q2)), rtol=1e-12)


def test_schur_inverse_matches_direct_dielectric_inverse():
    """The head and wing use the complete body Schur complement."""
    body = np.array([[1.1, 0.1 + 0.2j], [0.1 - 0.2j, 1.3]])
    eps_00 = 2.4 + 0.1j
    eps_p0 = np.array([0.2 + 0.1j, 0.1 - 0.3j])
    head, wing = inverse_dielectric_head_wing(eps_00, eps_p0, np.linalg.inv(body))
    dielectric = np.block([[np.array([[eps_00]]), eps_p0.conj()[None, :]], [eps_p0[:, None], body]])
    inverse = np.linalg.inv(dielectric)
    assert np.allclose(head, inverse[0, 0])
    assert np.allclose(wing, inverse[1:, 0])


def test_frozen_orbital_mapping_preserves_reordered_selection():
    """Full-space selections map explicitly into the active frozen space."""
    class Dummy:
        frozen = 1
        nkpts = 1
        mo_occ = np.asarray([[2.0, 2.0, 0.0]])
        orbs = [2, 1]

    dummy = Dummy()
    _set_frozen_orbs(dummy)
    assert dummy.orbs == [2, 1]
    assert dummy.orbs_frz == [1, 0]


def test_two_pole_fit_preserves_selected_orbital_axis():
    """Selected diagonal and full-matrix continuations recover known poles."""
    omega = 1j * np.linspace(0.1, 4.0, 10)
    data = np.empty((1, 2, len(omega)), dtype=complex)
    for index, pole in enumerate((2.0, 3.0)):
        data[0, index] = 0.1 + 0.2 / (omega + pole)
    full_data = data[:, :, None, :] * np.array([[1., .2], [.2, 1.]])[None, :, :, None]
    for values in (data, full_data):
        acobj = _fit_two_pole_per_element(values, omega, [1, 0], 1)
        assert acobj.coeff.shape == (10, *values.shape[:-1])
        np.testing.assert_allclose(acobj.ac_eval(omega), values, atol=1e-5, rtol=1e-5)


def test_periodic_spherical_coefficients_reproduce_pyscf(reference_mf):
    """Replacing Gaussian coefficients by Eq. 44/45 reproduces PySCF."""
    freqs, wts = pyscf_krgw_ac._get_scaled_legendre_roots(6)
    ef = pyscf_krgw_ac.get_ef(reference_mf, reference_mf.mo_energy)
    reference = _initialized_gw(pyscf_krgw_ac.KRGWAC, reference_mf)
    reference.fc_grid = False
    expected, _ = pyscf_krgw_ac.get_sigma(
        reference, freqs, wts, ef, reference.mo_energy, orbs=[0, 1], kptlist=[0]
    )

    gw = _initialized_gw(G0W0SS, reference_mf)
    sigma = compute_gaussian_coefficients(
        gw.mol, nkpts=gw.nkpts, mesh=monkhorst_pack_shape(gw.mol, gw.kpts)
    )
    spherical = GaussianCoefficients(
        sigma=sigma.sigma,
        head_integral=2.0 / np.pi * sigma.sigma,
        head_quadrature=0.0,
        head=2.0 / np.pi * sigma.sigma,
        wing_integral=np.sqrt(gw.mol.vol / (4.0 * np.pi**3)) * sigma.sigma**2,
        wing_quadrature=0.0,
        wing=np.sqrt(gw.mol.vol / (4.0 * np.pi**3)) * sigma.sigma**2,
    )
    gw.gaussian_coefficients = spherical
    actual, _ = get_sigma(gw, freqs, wts, ef, gw.mo_energy, orbs=[0, 1], kptlist=[0])
    assert np.allclose(actual, expected, atol=1e-12, rtol=1e-11)


def test_periodic_gaussian_head_wing_and_fullsigma(reference_mf):
    """Corrected head and wing additions are finite and diagonal-only."""
    freqs, wts = pyscf_krgw_ac._get_scaled_legendre_roots(5)
    ef = pyscf_krgw_ac.get_ef(reference_mf, reference_mf.mo_energy)
    gw = _initialized_gw(G0W0SS, reference_mf)
    gw.fullsigma = True
    sigma, omega = get_sigma(gw, freqs, wts, ef, gw.mo_energy, orbs=[1, 0], kptlist=[1], fullsigma=True)
    assert sigma.shape == (1, 2, 2, 6)
    assert np.isfinite(sigma).all()
    assert np.linalg.norm(gw.fc_sigma_head) > 0
    assert np.linalg.norm(gw.fc_sigma_wing) > 0
    body_gw = _initialized_gw(G0W0SS, reference_mf)
    body_gw.fc = False
    body, _ = pyscf_krgw_ac.get_sigma(
        body_gw, freqs, wts, ef, body_gw.mo_energy, fullsigma=True
    )
    body = body[1:2][:, [1, 0]][:, :, [1, 0]]
    head, wing = _direct_dielectric_additions(gw, freqs, wts, omega)
    head, wing = head[1:2, [1, 0]], wing[1:2, [1, 0]]
    np.testing.assert_allclose(gw.fc_sigma_head, head, atol=1e-12)
    np.testing.assert_allclose(gw.fc_sigma_wing, wing, atol=1e-12)
    expected_diagonal = body.copy()
    expected_diagonal[:, 0, 0] += head[:, 0] + wing[:, 0]
    expected_diagonal[:, 1, 1] += head[:, 1] + wing[:, 1]
    assert np.allclose(sigma, expected_diagonal, atol=1e-12, rtol=1e-10)
    # The finite-size additions are diagonal-only and therefore leave the
    # off-diagonal self-energy equal to the uncorrected value.
    assert np.allclose(sigma[:, 0, 1], body[:, 0, 1], atol=1e-12, rtol=1e-10)

    full_gw = _initialized_gw(G0W0SS, reference_mf)
    full, _ = get_sigma(full_gw, freqs, wts, ef, full_gw.mo_energy, kptlist=[1], fullsigma=True)
    assert np.allclose(sigma[:, 0, 0], full[:, 1, 1], atol=1e-12, rtol=1e-10)
    assert np.allclose(sigma[:, 1, 1], full[:, 0, 0], atol=1e-12, rtol=1e-10)
    diagonal, _ = get_sigma(full_gw, freqs, wts, ef, full_gw.mo_energy,
                            orbs=[1, 0], kptlist=[1])
    np.testing.assert_allclose(diagonal, sigma[:, [0, 1], [0, 1]], atol=1e-12)


def test_fc_grid_average_is_independent_of_direction_order(reference_mf, monkeypatch):
    """The Schur inverse is averaged per direction before contraction."""
    freqs, wts = pyscf_krgw_ac._get_scaled_legendre_roots(3)
    ef = pyscf_krgw_ac.get_ef(reference_mf, reference_mf.mo_energy)
    directions = np.asarray([[5e-4, 0, 0], [0, 5e-4, 0], [0, 0, 5e-4]])
    monkeypatch.setattr(g0w0ss, "_finite_q_points", lambda _: directions)
    first = _initialized_gw(G0W0SS, reference_mf)
    first.fc_grid = True
    sigma_first, _ = get_sigma(first, freqs, wts, ef, first.mo_energy, orbs=[0, 1], kptlist=[0])
    monkeypatch.setattr(g0w0ss, "_finite_q_points", lambda _: directions[::-1])
    second = _initialized_gw(G0W0SS, reference_mf)
    second.fc_grid = True
    sigma_second, _ = get_sigma(second, freqs, wts, ef, second.mo_energy, orbs=[0, 1], kptlist=[0])
    assert np.allclose(sigma_first, sigma_second, atol=1e-12, rtol=1e-10)
    assert np.allclose(first.fc_eps_inv_00, second.fc_eps_inv_00, atol=1e-12, rtol=1e-10)
    assert np.allclose(first.fc_eps_inv_p0, second.fc_eps_inv_p0, atol=1e-12, rtol=1e-10)


def test_frozen_occupied_response_uses_active_space_mapping():
    """A frozen occupied orbital leaves a finite corrected active response."""
    cell = gto.Cell()
    cell.unit = "Bohr"
    cell.atom = "H 0 0 0; H 1.5 0 0; H 3.0 0 0; H 4.5 0 0"
    cell.a = np.eye(3) * 8.0
    cell.basis = "gth-szv"
    cell.pseudo = "gth-hf"
    cell.ke_cutoff = 80
    cell.precision = 1e-7
    cell.verbose = 0
    cell.build()
    kpts = cell.make_kpts([1, 1, 1])
    mf = scf.KRHF(cell, kpts)
    mf.exxdiv = "ewald"
    mf.with_df = df.GDF(cell, kpts).build()
    mf.kernel()
    assert mf.converged

    gw = G0W0SS(mf, frozen=1)
    gw.mo_coeff = np.array(mf.mo_coeff)
    gw.mo_energy = np.array(mf.mo_energy)
    gw.mo_occ = np.array(mf.mo_occ)
    gw.orbs = [2, 1]
    gw.set_frozen_orbs()
    assert gw.nocc == 1
    assert gw.orbs_frz == [1, 0]
    freqs, wts = pyscf_krgw_ac._get_scaled_legendre_roots(3)
    ef = pyscf_krgw_ac.get_ef(mf, mf.mo_energy)
    active_energy = pyscf_krgw_ac._mo_energy_frozen(gw, gw.mo_energy)
    sigma, _ = get_sigma(gw, freqs, wts, ef, active_energy, orbs=gw.orbs_frz, kptlist=[0])
    assert sigma.shape == (1, 2, 4)
    assert np.isfinite(sigma).all()
    raw_exchange = gw.get_sigma_exchange()
    gw.nw = 32
    gw.ac_pade_npts = 10
    gw.fullsigma = True
    gw.kernel(orbs=[2, 1], kptlist=[0])
    expected_exchange = raw_exchange.copy()
    expected_exchange[:, 0, 0] -= 2 / np.pi * gw.gaussian_coefficients.sigma
    np.testing.assert_allclose(gw.vk, expected_exchange, atol=1e-12)
    assert np.isfinite(gw.mo_energy[0, [2, 1]]).all()
    assert np.all(gw.mo_energy[0, [2, 1]] != 0)
    list_gw = _initialized_gw(G0W0SS, mf)
    list_gw.frozen = [0]
    list_sigma, _ = get_sigma(list_gw, freqs, wts, ef, active_energy,
                              orbs=[1, 0], kptlist=[0], fullsigma=True)
    np.testing.assert_allclose(list_sigma[:, [0, 1], [0, 1]], sigma, atol=1e-12)


def test_uncorrected_kernel_matches_pyscf(reference_mf):
    """Disabling correction retains upstream QPs, exchange and continuation."""
    results = []
    for cls in (pyscf_krgw_ac.KRGWAC, G0W0SS):
        gw = cls(reference_mf)
        gw.fc = False
        gw.nw = 32
        gw.ac_pade_npts = 10
        gw.kernel(orbs=[0, 1], kptlist=[1])
        results.append(gw)
    np.testing.assert_allclose(results[0].mo_energy, results[1].mo_energy, atol=1e-12)
    np.testing.assert_allclose(results[0].vk, results[1].vk, atol=1e-12)
    np.testing.assert_allclose(results[0].acobj.coeff, results[1].acobj.coeff, atol=1e-12)


def test_unsupported_corrected_systems_are_rejected(reference_mf):
    """Unsupported physical and storage regimes fail before response work."""
    gw = G0W0SS(reference_mf)
    with lib.temporary_env(gw, outcore=True), pytest.raises(NotImplementedError, match='in-core'):
        gw.kernel()
    with lib.temporary_env(reference_mf, sigma=0.1), pytest.raises(NotImplementedError, match='smeared'):
        gw.kernel()
    with lib.temporary_env(gw.mol, dimension=2), pytest.raises(NotImplementedError, match='three-dimensional'):
        gw.kernel()
    with lib.temporary_env(gw, with_df=df.FFTDF(gw.mol)), pytest.raises(NotImplementedError, match='GDF'):
        gw.kernel()
    metallic_energies = np.array(reference_mf.mo_energy)
    metallic_energies[:, 1] = np.max(metallic_energies[:, 0]) - 0.01
    with lib.temporary_env(reference_mf, mo_energy=metallic_energies), pytest.raises(NotImplementedError, match='gap'):
        gw.kernel()
