"""Small reference checks for restricted staggered-mesh G0W0."""

import numpy as np
import pytest
from pyscf.pbc import dft, gto

from fsec.staggered_mesh.gw import (
    KRGWAC_stagger_extend,
    KRGWAC_stagger_submesh,
)
from fsec.staggered_mesh.gw.krgw_ac_stagger import (
    _get_scaled_legendre_roots,
    _load_lmn,
    _madelung_stagger,
)


@pytest.fixture(scope='module')
def h2_4k_scf():
    """Build a compact four-point H2 reference for both staggered factories."""
    cell = gto.Cell()
    cell.build(
        unit='Bohr', a=np.eye(3) * 6.0,
        atom='H 3 3 2.1; H 3 3 3.9', dimension=3,
        verbose=0, output='/dev/null', pseudo='gth-pade', basis='gth-szv',
        precision=1e-8,
    )
    mf = dft.KRKS(cell, cell.make_kpts([4, 1, 1]))
    mf.xc = 'pbe'
    mf.conv_tol = 1e-8
    mf.kernel()
    return mf


def _direct_sigma(gw, orbs, kptlist):
    """Re-evaluate the q, frequency, and MO contractions with explicit loops."""
    freqs, weights = _get_scaled_legendre_roots(gw.nw)
    if gw.ac_iw_cutoff is not None:
        nout = np.count_nonzero(freqs < gw.ac_iw_cutoff)
    else:
        nout = len(freqs)
    omega = gw.ef + 1j * np.r_[0.0, freqs[:nout]]
    shift_energy = gw._mo_energy[gw.kpts_shift_idx]
    target_energy = gw._mo_energy[gw.kpts_target_idx]
    expected = np.zeros_like(gw.sigma)
    qpts = gw.kpts_target - gw.kpts_shift[0]

    for qpt in qpts:
        lmn, right_index = _load_lmn(
            gw, gw.kpts_target, gw.kpts_shift,
            gw.mo_coeff[gw.kpts_target_idx], gw.mo_coeff[gw.kpts_shift_idx], qpt,
        )
        for iw, iw_freq in enumerate(freqs):
            pi = np.zeros((lmn.shape[1], lmn.shape[1]), dtype=np.complex128)
            for im, jn in enumerate(right_index):
                eia = target_energy[im, :gw.nocc, None] - shift_energy[jn, None, gw.nocc:]
                eia /= iw_freq**2 + eia**2
                pia = lmn[im, :, :gw.nocc, gw.nocc:] * eia[None, :, :]
                pi += 4.0 / gw.nkpts_target * np.einsum(
                    'Pia,Qia->PQ', pia, lmn[im, :, :gw.nocc, gw.nocc:].conj()
                )
            wc = np.linalg.inv(np.eye(len(pi)) - pi) - np.eye(len(pi))

            for out_k, target_k in enumerate(kptlist):
                shift_k = right_index[target_k]
                g0 = weights[iw] * (
                    omega[None, :] - shift_energy[shift_k, :, None]
                ) / (
                    (omega[None, :] - shift_energy[shift_k, :, None])**2
                    + iw_freq**2
                )
                for out_p, orb in enumerate(orbs):
                    for m in range(gw.nmo):
                        value = 0j
                        for p in range(lmn.shape[1]):
                            for q in range(lmn.shape[1]):
                                value += (
                                    lmn[target_k, p, orb, m].conj()
                                    * wc[p, q]
                                    * lmn[target_k, q, orb, m]
                                )
                        expected[out_k, out_p] -= (
                            value / gw.nkpts_target * g0[m] / np.pi
                        )
    return expected


@pytest.mark.slow
def test_submesh_nonlinear_qpe_and_direct_self_energy(h2_4k_scf):
    """Check selected-state indexing, full-mesh Fermi level, and contractions."""
    gw = KRGWAC_stagger_submesh(h2_4k_scf, [2, 1, 1])
    gw.fc = False
    gw.nw = 20
    gw.ac_iw_cutoff = None
    gw.ac_pade_npts = 8
    gw.qpe_linearized = False
    orbs = [1, 0]
    kptlist = [1]

    assert gw.kernel(orbs=orbs, kptlist=kptlist) is None
    assert gw.converged
    assert np.isfinite(gw.mo_energy[1]).all()
    assert gw.qpe_converged[1].all()
    assert np.isnan(gw.mo_energy[0]).all()
    expected_ef = 0.5 * (
        np.max(h2_4k_scf.mo_energy[:, :gw.nocc])
        + np.min(h2_4k_scf.mo_energy[:, gw.nocc:])
    )
    assert gw.ef == pytest.approx(expected_ef)
    assert gw.sigma == pytest.approx(_direct_sigma(gw, orbs, kptlist), abs=1e-10)

    gw.fc = False
    vk_uncorrected = gw._get_sigma_exchange(kptlist)
    gw.fc = True
    gw.fc_ex_madelung = True
    gw.fc_corr_madelung = False
    vk_corrected = gw._get_sigma_exchange(kptlist)
    correction = _madelung_stagger(gw.cell, gw.Nk_target, gw.mesh_shift)
    assert vk_corrected[0, 0, 0] - vk_uncorrected[0, 0, 0] == pytest.approx(-correction)
    assert vk_corrected[0, 1, 1] == pytest.approx(vk_uncorrected[0, 1, 1])


@pytest.mark.slow
def test_extension_two_pole_linearized_qpe(h2_4k_scf):
    """Exercise band extension, two-pole continuation, and reordered subsets."""
    gw = KRGWAC_stagger_extend(h2_4k_scf, [2, 1, 1])
    gw.ac = 'twopole'
    gw.fc = False
    gw.nw = 20
    gw.ac_iw_cutoff = None
    gw.qpe_linearized = True

    assert gw.kernel(orbs=[1, 0], kptlist=[1, 0]) is None
    assert gw.converged
    assert np.isfinite(gw.mo_energy[1]).all()
    assert np.isfinite(gw.mo_energy[0]).all()
