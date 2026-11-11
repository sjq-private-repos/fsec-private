"""Reference checks for KRGWAC_ewald q -> 0 finite-size corrections."""

import itertools
from copy import copy

import h5py
import numpy as np
import pytest
from pyscf.pbc import df, dft, gto, scf, tools
from pyscf.pbc.gw import krgw_ac

from fsec.standard import KRGWAC_ewald
from fsec.standard.krgwac_ewald import (
    _uniform_mesh_shape,
    _validate_fc_setup,
    aux_constants,
    get_eps_inv_q0,
    get_lij_q0,
    madelung_inv_q,
)


def _mesh_kpts(cell, shape, shift=(0.0, 0.0, 0.0)):
    """Create a complete uniform mesh from fractional coordinates."""
    points = [
        (np.asarray(index, dtype=float) / shape + shift) % 1.0
        for index in itertools.product(*(range(n) for n in shape))
    ]
    return cell.get_abs_kpts(points)


def _configure_gw(gw):
    """Use a compact but stable integration and continuation setup."""
    gw.nw = 20
    gw.ac_iw_cutoff = None
    gw.ac_pade_npts = 8
    gw.rdm = True
    gw.qpe_linearized = True
    return gw


@pytest.fixture(scope='module')
def h2_4k_scf():
    """Four-point H2 chain; the z-oriented bond makes the q -> 0 head anisotropic."""
    cell = gto.Cell()
    cell.build(
        unit='Bohr', a=np.eye(3) * 6.0,
        atom='H 3 3 2.1; H 3 3 3.9', dimension=3,
        verbose=0, output='/dev/null', pseudo='gth-pade', basis='gth-szv',
        precision=1e-8,
    )
    kpts = cell.make_kpts([4, 1, 1])
    # build all k-pairs up front; a J-only GDF lacks the pairs GW needs
    gdf = df.GDF(cell, kpts)
    gdf.build()
    mf = dft.KRKS(cell, kpts).density_fit(with_df=gdf)
    mf.xc = 'pbe'
    mf.conv_tol = 1e-10
    mf.kernel()
    return mf


@pytest.fixture(scope='module')
def lih_shifted_mf():
    """Occupied core/valence states on a shifted mesh with RSDF."""
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


def test_sphere_reproduces_pyscf(h2_4k_scf):
    """Sphere coefficients along PySCF's q direction give PySCF's QP energies."""
    ref = krgw_ac.KRGWAC(h2_4k_scf)
    ref.kernel()
    gw = KRGWAC_ewald(h2_4k_scf)
    gw.fc_method = 'sphere'
    gw.kernel()
    assert np.allclose(gw.mo_energy, ref.mo_energy, atol=1e-8)


def test_aux_constants_limit_to_madelung():
    """Literal Eq. B.6 head is Madelung - 4 pi beta/V; the 1/k constant tends to its Ewald limit."""
    cell = gto.Cell()
    cell.build(
        unit='Bohr', a=np.array([[0., 3., 3.], [3., 0., 3.], [3., 3., 0.]]),
        atom='He 0 0 0', basis='gth-szv', pseudo='gth-pade', verbose=0,
    )
    nks = (2, 2, 2)
    vol = cell.vol * 8
    madelung = tools.madelung(cell, cell.make_kpts(nks))
    beta = (cell.vol / (6.0 * np.pi**2))**(1.0 / 3.0)
    c_head = aux_constants(cell, nks, beta)[0]
    assert abs(c_head - (madelung - 4.0 * np.pi * beta / vol)) < 1e-8

    c_inv_q = madelung_inv_q(cell, nks)
    assert abs(madelung_inv_q(cell, nks, tau=0.5) - c_inv_q) < 1e-10
    assert abs(aux_constants(cell, nks, 0.01)[1] / c_inv_q - 1.0) < 1e-3


def test_fc_rejects_unsupported_setups():
    """Accept shifted/reordered complete meshes; reject incomplete meshes and unsupported physics."""
    cell = gto.Cell()
    cell.build(
        unit='Bohr', a=np.eye(3) * 6.0, atom='H 0 0 0; H 1 1 1',
        dimension=3, pseudo='gth-pade', basis='gth-szv',
        precision=1e-8, verbose=0, output='/dev/null',
    )
    kpts = _mesh_kpts(cell, (2, 2, 2))
    shifted = _mesh_kpts(cell, (2, 2, 2), shift=(0.13, 0.21, 0.37))
    reordered = shifted[[5, 2, 7, 0, 3, 6, 1, 4]]
    assert tuple(_uniform_mesh_shape(cell, reordered)) == (2, 2, 2)
    assert tools.madelung(cell, reordered) == pytest.approx(tools.madelung(cell, kpts))
    incomplete = kpts.copy()
    incomplete[1] += cell.get_abs_kpts([[0.07, 0.0, 0.0]])[0]
    with pytest.raises(ValueError, match='uniform MP mesh'):
        _uniform_mesh_shape(cell, incomplete)

    mf = scf.KRHF(cell, cell.make_kpts([1, 1, 1])).density_fit()
    mf.mo_energy = np.array([[-0.5, 0.5]])
    mf.mo_occ = np.array([[2.0, 0.0]])
    mf.mo_coeff = np.eye(2)[None, :, :]
    gw = KRGWAC_ewald(mf)
    gw.mo_energy, gw.mo_occ, gw.mo_coeff = mf.mo_energy, mf.mo_occ, mf.mo_coeff
    _validate_fc_setup(gw)
    slab = cell.copy()
    slab.dimension = 2
    for attr, value, message in (
        ('with_df', df.MDF(cell, mf.kpts), 'GDF or RSDF'),
        ('mo_occ', np.array([[1.8, 0.2]]), 'integer occupations'),
        ('mo_occ', np.array([[[1.0, 0.0]], [[1.0, 0.0]]]), 'restricted scalar'),
        ('mo_energy', np.zeros((1, 2)), 'gapped'),
        ('mol', slab, '3D periodic'),
    ):
        candidate = copy(gw)
        setattr(candidate, attr, value)
        with pytest.raises(NotImplementedError, match=message):
            _validate_fc_setup(candidate)


def test_direction_average(h2_4k_scf):
    """Tensor head/wings match PySCF's scalar k.p route averaged over +-x, +-y, +-z."""
    mf = h2_4k_scf
    gw = KRGWAC_ewald(mf)
    gw.fc_qdirs = 6
    gw.mo_occ = np.asarray(mf.mo_occ)
    mo_energy = np.asarray(mf.mo_energy)
    mo_coeff = np.asarray(mf.mo_coeff)
    nocc, nkpts = gw.nocc, gw.nkpts
    Lia = np.ascontiguousarray(get_lij_q0(gw, mo_coeff)[:, :, :nocc, nocc:])
    freq = 0.5
    Pi = krgw_ac.get_rho_response(freq, mo_energy, Lia, np.arange(nkpts))
    Pi_inv = np.linalg.inv(np.eye(len(Pi)) - Pi)
    qij_cart = np.asarray([krgw_ac.get_qij(gw, e, mo_energy, mo_coeff) for e in np.eye(3)])
    dirs, weights = gw.get_qdirs()
    eps_inv_00, eps_inv_P0 = get_eps_inv_q0(freq, mo_energy, Lia, Pi_inv, qij_cart, dirs, weights)

    ref = []
    for q in dirs * 1e-3:
        qij = krgw_ac.get_qij(gw, q, mo_energy, mo_coeff)
        eps_00 = 1.0 - 4.0 * np.pi / np.dot(q, q) * krgw_ac.get_rho_response_head(freq, mo_energy, qij)
        eps_P0 = -np.sqrt(4.0 * np.pi / np.dot(q, q)) * krgw_ac.get_rho_response_wing(freq, mo_energy, Lia, qij)
        ref.append(1.0 / (eps_00 - eps_P0.conj() @ Pi_inv @ eps_P0))
    ref = np.asarray(ref)
    assert abs(eps_inv_00 - np.dot(weights, ref)) < 1e-10
    assert np.linalg.norm(eps_inv_P0) < 1e-10
    # x and z heads differ for the z-oriented bond
    assert abs(ref[0] - ref[4]) > 1e-4


def test_outcore_matches_incore(h2_4k_scf):
    """Aux corrections with direction averaging give the same QP energies outcore."""
    mo_energy = []
    for outcore in (False, True):
        gw = KRGWAC_ewald(h2_4k_scf)
        gw.fc_qdirs = 26
        gw.outcore = outcore
        gw.kernel()
        mo_energy.append(gw.mo_energy)
    assert np.allclose(mo_energy[0], mo_energy[1], atol=1e-8)


@pytest.mark.slow
def test_kernel_corrects_selected_fullsigma(h2_4k_scf, tmp_path, monkeypatch):
    """Reordered full sigma gets diagonal corrections only; files, exchange and SCF data stay consistent."""
    mf = h2_4k_scf
    mf_energy = np.array(mf.mo_energy, copy=True)
    mf_coeff = np.array(mf.mo_coeff, copy=True)
    monkeypatch.chdir(tmp_path)
    sigma = []
    for fc in (False, True):
        gw = _configure_gw(KRGWAC_ewald(mf))
        source_df = df.GDF(gw.mol, gw.kpts)
        gw.with_df = source_df
        gw.fc = fc
        gw.fullsigma = True
        gw.fc_grid = True
        gw.ac = 'twopole'
        gw.writefile = 1
        gw.kernel(orbs=[1, 0], kptlist=[1])
        sigma.append(gw.sigmaI)

    assert gw.sigmaI.shape[0:3] == (1, 2, 2)
    assert np.isfinite(gw.sigmaI).all()
    np.testing.assert_allclose(sigma[1][0, 0, 1], sigma[0][0, 0, 1], atol=1e-12)
    assert np.max(np.abs(sigma[1][0, 0, 0] - sigma[0][0, 0, 0])) > 1e-6
    assert gw.with_df is not source_df and source_df._cderi is None
    np.testing.assert_array_equal(gw.mo_energy[0], np.zeros_like(gw.mo_energy[0]))
    np.testing.assert_array_equal(np.asarray(mf.mo_energy), mf_energy)
    np.testing.assert_array_equal(np.asarray(mf.mo_coeff), mf_coeff)
    with h5py.File('sigma_imag.h5', 'r') as sigma_file:
        np.testing.assert_allclose(sigma_file['sigmaI'], gw.sigmaI)
        np.testing.assert_allclose(sigma_file['sigmaI_full'], gw.sigmaI)
    with h5py.File('vxc.h5', 'r') as exchange_file:
        np.testing.assert_array_equal(exchange_file['vk'], gw.vk)
    raw_vk = gw.get_sigma_exchange()
    c_head = gw.get_fc_coeffs()[0]
    np.testing.assert_allclose(gw.vk[:, 0, 0] - raw_vk[:, 0, 0], -c_head, atol=1e-10)
    np.testing.assert_allclose(gw.vk[:, 1, 1], raw_vk[:, 1, 1], atol=1e-10)


@pytest.mark.slow
def test_shifted_rsdf_frozen_noncontiguous_selection(lih_shifted_mf):
    """Frozen, reordered full sigma equals the upstream raw sigma plus the selected diagonal corrections."""
    mf = lih_shifted_mf
    original_energy = np.array(mf.mo_energy, copy=True)
    reference = _configure_gw(krgw_ac.KRGWAC(mf, frozen=1))
    reference.fc = False
    reference.fullsigma = True
    reference.kernel()

    gw = _configure_gw(KRGWAC_ewald(mf, frozen=1))
    gw.fullsigma = True
    gw.qpe_linearized = False
    gw.fc_qdirs = 26
    bands, kpoints = [4, 1, 3], [1, 0]
    gw.kernel(orbs=bands, kptlist=kpoints)
    active = np.array(bands) - 1
    expected = reference.sigmaI[kpoints][:, active][:, :, active].copy()
    omega = np.r_[0j, gw.ef + 1j * gw.freqs]
    # corrections for all active orbitals and k-points, then the selection
    correction = gw.get_sigma_fc(gw.freqs, gw.wts, omega, original_energy[:, 1:])
    for ik, k in enumerate(kpoints):
        for ip, p in enumerate(active):
            expected[ik, ip, ip] += correction[k, p]
    assert np.max(np.abs(correction)) > 1e-6
    np.testing.assert_allclose(gw.sigmaI, expected, atol=1e-9, rtol=0)
    exchange_delta = np.zeros_like(reference.vk)
    exchange_delta[:, 0, 0] = -gw.get_fc_coeffs()[0]
    np.testing.assert_allclose(gw.vk - reference.vk, exchange_delta, atol=1e-9, rtol=0)
    np.testing.assert_array_equal(np.asarray(mf.mo_energy), original_energy)
    assert np.all(np.isfinite(gw.mo_energy[:, bands]))
    assert np.all(gw.mo_energy[:, bands] != 0.0)
