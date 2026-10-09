"""Small periodic regressions for restricted MP3 amplitudes and energies."""

import numpy as np
import pytest

from pyscf.pbc import df, gto, scf
from pyscf.pbc.mp.kmp2 import KMP2

from fsec.staggered_mesh.mp import KMP3, KMP3_2mesh, KMP3_3mesh
from fsec.staggered_mesh.mp.kmp3_2mesh import (
    mp3_amplitude_2mesh,
    mp3_amplitude_2mesh_efficient,
)
from fsec.staggered_mesh.mp.kmp3_3mesh import (
    mp3_amplitude_3mesh,
    mp3_amplitude_3mesh_efficient,
)


pytestmark = pytest.mark.slow


def make_h2(mesh, basis="gth-szv", exxdiv=None):
    """Build a small converged H2 KRHF reference with GDF integrals."""
    cell = gto.Cell()
    cell.atom = "H 3 3 2.1; H 3 3 3.9"
    cell.a = np.eye(3) * 6
    cell.unit = "B"
    cell.basis = basis
    cell.pseudo = "gth-pade"
    cell.ke_cutoff = 100
    cell.precision = 1e-9
    cell.verbose = 0
    cell.build()

    kpts = cell.make_kpts(mesh, with_gamma_point=True)
    mf = scf.KRHF(cell, kpts, exxdiv=exxdiv)
    mf.with_df = df.GDF(cell, kpts).build()
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.kernel()
    assert mf.converged
    return mf


@pytest.mark.parametrize(
    "exxdiv, expected_mp2, expected_mp3",
    [
        (None, -0.022909492354552606, -0.005706145151371517),
        ("ewald", -0.01939327781245577, -0.0070789471523389404),
    ],
)
def test_kmp3_energy_decomposition_and_first_order_amplitudes(
    exxdiv, expected_mp2, expected_mp3
):
    """Match the k-point reference energies and the KMP2 first-order t2."""
    mf = make_h2([1, 1, 3], exxdiv=exxdiv)
    mp = KMP3(mf)
    assert mp.madelung is (exxdiv == "ewald")
    assert KMP3(mf, madelung=not mp.madelung).madelung is not mp.madelung

    energy, t2 = mp.kernel(mo_energy=mf.mo_energy, mo_coeff=mf.mo_coeff)
    mp2_energy, mp2_t2 = KMP2(mf).kernel()

    np.testing.assert_allclose(mp.e_corr_mp2, expected_mp2, atol=1e-9, rtol=0)
    np.testing.assert_allclose(mp.e_corr_mp3, expected_mp3, atol=1e-9, rtol=0)
    np.testing.assert_allclose(energy, expected_mp2 + expected_mp3, atol=1e-9, rtol=0)
    np.testing.assert_allclose(mp2_energy, mp.e_corr_mp2, atol=1e-9, rtol=0)
    np.testing.assert_allclose(mp.emp2, mp.e_corr_mp2, atol=1e-12, rtol=0)
    np.testing.assert_allclose(t2, mp2_t2, atol=1e-9, rtol=0)
    assert mp.t2_mp3.shape == t2.shape
    assert not np.allclose(mp.t2_mp3, t2)

    energy_without_amplitudes, no_t2 = mp.kernel(eris=mp.eris, with_t2=False)
    np.testing.assert_allclose(energy_without_amplitudes, energy, atol=1e-12, rtol=0)
    assert no_t2 is None
    assert mp.t2_mp3 is None
    assert mp._t2_mp2 is None


def test_kmp3_complex_multikpoint_with_multiple_virtuals():
    """Compare complex Bloch orbitals, phase covariance, and E3 with KRMP3."""
    mf = make_h2([1, 1, 3], basis="gth-dzvp")
    assert mf.mo_coeff[0].shape[1] - mf.mo_occ[0].sum() // 2 > 1
    assert any(np.max(np.abs(coeff.imag)) > 1e-8 for coeff in mf.mo_coeff)

    mp = KMP3(mf)
    energy, t2 = mp.kernel(mo_energy=mf.mo_energy, mo_coeff=mf.mo_coeff)
    mp2_energy, mp2_t2 = KMP2(mf).kernel()

    assert np.isfinite(energy)
    np.testing.assert_allclose(mp.e_corr_mp2, mp2_energy, atol=1e-9, rtol=0)
    np.testing.assert_allclose(t2, mp2_t2, atol=1e-9, rtol=0)

    try:
        from pyscf.pbc.mp.kmp3 import KRMP3
    except ImportError:
        pytest.skip("the optional PySCF KRMP3 comparison module is unavailable")
    reference = KRMP3(mf)
    reference.kernel()
    np.testing.assert_allclose(mp.e_corr_mp3, reference.e_corr_mp3, atol=1e-9, rtol=0)
    np.testing.assert_allclose(mp.e_corr, reference.e_corr, atol=1e-9, rtol=0)

    nkpts, _, nmo = np.asarray(mf.mo_coeff).shape
    phase = np.exp(1j * np.arange(nkpts * nmo).reshape(nkpts, nmo) * 0.17)
    phased_coeff = np.asarray(mf.mo_coeff) * phase[:, None, :]
    phased = KMP3(mf)
    phased.kernel(mo_energy=mf.mo_energy, mo_coeff=phased_coeff)
    np.testing.assert_allclose(phased.e_corr, mp.e_corr, atol=1e-9, rtol=0)
    np.testing.assert_allclose(phased.e_corr_mp3, mp.e_corr_mp3, atol=1e-9, rtol=0)
    for ki, kj, ka in np.ndindex(phased.khelper.kconserv.shape):
        kb = phased.khelper.kconserv[ki, ka, kj]
        expected_t2 = t2[ki, kj, ka] * (
            phase[ki, :phased.nocc][:, None, None, None]
            * phase[kj, :phased.nocc][None, :, None, None]
            * phase[ka, phased.nocc:][None, None, :, None].conj()
            * phase[kb, phased.nocc:][None, None, None, :].conj()
        )
        np.testing.assert_allclose(
            phased.t2[ki, kj, ka], expected_t2, atol=1e-9, rtol=1e-9
        )


def test_staggered_modes_route_momenta_and_match_direct_contractions():
    """Check direct/optimized contractions on submesh and extension modes."""
    mf = make_h2([1, 1, 6])
    variants = (
        (KMP3_2mesh, mp3_amplitude_2mesh, mp3_amplitude_2mesh_efficient),
        (KMP3_3mesh, mp3_amplitude_3mesh, mp3_amplitude_3mesh_efficient),
    )

    # The two-mesh formulation enables its correction by default and leaves the
    # three-mesh adaptive correction disabled.
    for flag_submesh in (True, False):
        assert KMP3_2mesh(mf, flag_submesh=flag_submesh,
                          nks_mp=None if flag_submesh else [1, 1, 2]).madelung
        assert not KMP3_3mesh(
            mf, flag_submesh=flag_submesh,
            nks_mp=None if flag_submesh else [1, 1, 2]
        ).madelung

    for method_cls, direct, optimized in variants:
        for flag_submesh in (True, False):
            nks_mp = None if flag_submesh else [1, 1, 2]
            mp = method_cls(
                mf, flag_submesh=flag_submesh, nks_mp=nks_mp, madelung=False
            )
            eris = mp.ao2mo()

            for ki, ka, kj in np.ndindex(mp.khelper.kconserv.shape):
                kb = mp.khelper.kconserv[ki, ka, kj]
                scaled = mf.cell.get_scaled_kpts(
                    mp.kpts[[ki, ka, kj, kb]]
                )
                delta = scaled[0] - scaled[1] + scaled[2] - scaled[3]
                np.testing.assert_allclose(delta, np.rint(delta), atol=1e-9, rtol=0)

            direct_t2 = direct(mp, eris)
            optimized_t2 = optimized(mp, eris)
            np.testing.assert_allclose(optimized_t2, direct_t2, atol=1e-9, rtol=1e-9)

            energy, first_order_t2 = mp.kernel(eris=eris)
            assert np.isfinite(energy)
            assert first_order_t2.shape == optimized_t2.shape
            np.testing.assert_allclose(first_order_t2, mp._t2_mp2, atol=1e-12, rtol=0)
            np.testing.assert_allclose(
                mp.e_corr,
                mp.e_corr_mp2 + mp.e_corr_mp3,
                atol=1e-12,
                rtol=0,
            )
