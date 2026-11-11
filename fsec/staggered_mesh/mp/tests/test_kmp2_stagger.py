"""Staggered MP2 against PySCF, and its sTC band orbitals against FFT references."""

import numpy as np
import pytest

from pyscf import gto as mol_gto
from pyscf import lib
from pyscf.pbc import df, gto, scf
from pyscf.pbc.mp.kmp2_stagger import KMP2_stagger as PySCF_KMP2_stagger

from fsec.staggered_mesh.mp import KMP2_stagger
from fsec.staggered_mesh.mp.kmp2_stagger import _load_Lpq_ao

rsdf_stc = pytest.importorskip('pyscf.pbc.df.rsdf_stc')

pytestmark = pytest.mark.slow

#   Large enough that fitting error stays below the band tolerances used here.
AUXBASIS = {'H': mol_gto.expand_etbs([(0, 12, 0.1, 2.0), (1, 8, 0.2, 2.0),
                                      (2, 5, 0.3, 2.0)])}


@pytest.fixture(scope='module')
def cell():
    """Build a small H2 cell."""
    cell = gto.Cell()
    cell.atom = 'H 3 3 2.1; H 3 3 3.9'
    cell.a = np.eye(3) * 6
    cell.unit = 'B'
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.ke_cutoff = 100
    cell.precision = 1e-10
    cell.verbose = 0
    cell.build()
    return cell


@pytest.fixture(scope='module')
def gdf_reference(cell):
    """Converge an ordinary GDF KRHF reference on a 2x2x2 mesh."""
    kpts = cell.make_kpts([2, 2, 2], with_gamma_point=True)
    mf = scf.KRHF(cell, kpts, exxdiv='ewald')
    mf.with_df = df.GDF(cell, kpts)
    mf.with_df.auxbasis = AUXBASIS
    mf.conv_tol = 1e-11
    mf.kernel()
    assert mf.converged
    return mf


@pytest.fixture(scope='module')
def stc_reference(cell):
    """Converge a KRHF reference with stored sTC exchange on a 2x2x2 mesh."""
    kpts = cell.make_kpts([2, 2, 2], with_gamma_point=True)
    mf = rsdf_stc.density_fit(scf.KRHF(cell, kpts), auxbasis=AUXBASIS, eta=3.0,
                              exxdiv='vcut_sph', rc_type='sph')
    mf.conv_tol = 1e-11
    mf.kernel()
    assert mf.converged
    return mf


@pytest.mark.parametrize('flag_submesh', [True, False])
def test_default_path_matches_pyscf(gdf_reference, flag_submesh):
    """Without sTC options the energy is the PySCF staggered MP2 energy."""
    energy = KMP2_stagger(gdf_reference, flag_submesh=flag_submesh).kernel()
    reference = PySCF_KMP2_stagger(gdf_reference, flag_submesh=flag_submesh).kernel()
    np.testing.assert_allclose(energy, reference, atol=1e-10, rtol=0)


@pytest.mark.parametrize(
    'exxdiv, rc_type, expected',
    [
        ('vcut_ws', 'ws', -0.014055970345708338),
        ('vcut_sph', 'sph', -0.014076783777210692),
    ],
)
def test_stc_bands_match_fft_reference(gdf_reference, exxdiv, rc_type, expected):
    """GDF+sTC band energies at shifted points match the FFTDF_STC reference."""
    mf = gdf_reference
    mp = KMP2_stagger(mf, band_df='stc', exxdiv=exxdiv)

    fft_df = df.FFTDF_STC(mf.cell, mf.kpts, eta=mp.eta, exxdiv=exxdiv,
                          rc_type=rc_type)
    with lib.temporary_env(mf, exxdiv=None, with_df=fft_df):
        mo_energy = mf.get_bands(mp.kpts)[0]
    np.testing.assert_allclose(mp.mo_energy, mo_energy, atol=2e-5, rtol=0)
    np.testing.assert_allclose(mp.kernel(), expected, atol=1e-8, rtol=0)


def test_stc_reference_is_detected_and_left_unchanged(stc_reference):
    """An sTC reference sets the band options and keeps its own integrals."""
    cderi = stc_reference.with_df._cderi
    mp = KMP2_stagger(stc_reference)
    assert mp.band_df == 'stc'
    assert (mp.eta, mp.exxdiv) == (3.0, 'vcut_sph')
    assert mp.with_df.auxbasis is AUXBASIS
    assert stc_reference.with_df._cderi == cderi
    assert stc_reference.with_df.kpts_band is None


def test_submesh_stc_reference_uses_bare_integrals(stc_reference):
    """Submesh correlation with an sTC reference uses ordinary GDF integrals."""
    energy = KMP2_stagger(stc_reference, flag_submesh=True).kernel()

    bare = stc_reference.copy()
    bare.with_df = df.GDF(bare.cell, bare.kpts)
    bare.with_df.auxbasis = AUXBASIS
    bare.with_df.build()
    reference = PySCF_KMP2_stagger(bare, flag_submesh=True).kernel()
    np.testing.assert_allclose(energy, reference, atol=1e-10, rtol=0)


def test_linearly_dependent_aux_matches_fft_integrals():
    """A q-dependent auxiliary count still reproduces the FFTDF correlation energy."""
    cell = gto.Cell()
    cell.atom = 'H 1.5 1.5 0.6; H 1.5 1.5 2.0'
    cell.a = np.eye(3) * 4
    cell.unit = 'B'
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.ke_cutoff = 100
    cell.verbose = 0
    cell.build()
    kpts = cell.make_kpts([2, 2, 2], with_gamma_point=True)

    mf = scf.KRHF(cell, kpts, exxdiv='ewald')
    mf.with_df = df.GDF(cell, kpts)
    mf.with_df.auxbasis = {'H': mol_gto.expand_etbs([(0, 10, 0.06, 2.0),
                                                     (1, 6, 0.12, 2.0)])}
    mf.kernel()
    naux = {len(_load_Lpq_ao(mf.with_df, kpti, kptj))
            for kpti in kpts for kptj in kpts}
    assert len(naux) > 1

    energy = KMP2_stagger(mf, flag_submesh=True).kernel()
    fft_mf = mf.copy()
    fft_mf.with_df = df.FFTDF(cell, kpts)
    reference = KMP2_stagger(fft_mf, flag_submesh=True).kernel()
    np.testing.assert_allclose(energy, reference, atol=1e-5, rtol=0)
