"""Small-cell regression tests for staggered periodic restricted CCD."""

import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pytest
from pyscf.pbc import df, dft, gto, scf

from fsec.staggered_mesh.cc import KRCCD_2mesh, KRCCD_3mesh


def _make_cell():
    cell = gto.Cell()
    cell.atom = 'H 3 3 2.1; H 3 3 3.9'
    cell.a = '6 0 0; 0 6 0; 0 0 6'
    cell.unit = 'B'
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.mesh = [11, 11, 11]
    cell.precision = 1e-8
    cell.verbose = 0
    cell.build()
    return cell


def _run_reference(cell, mesh, ks=False):
    kpts = cell.make_kpts(mesh)
    mf_class = dft.KRKS if ks else scf.KRHF
    mf = mf_class(cell, kpts=kpts)
    if ks:
        mf.xc = 'pbe'
    else:
        mf.exxdiv = 'ewald'
    mf.with_df = df.GDF(cell, kpts)
    mf.conv_tol = 1e-9
    mf.conv_tol_grad = 1e-7
    mf.verbose = 0
    mf.kernel()
    return mf


def _assert_momentum_conservation(cell, cc, triples):
    scaled = cell.get_scaled_kpts(cc.kpts)
    for ki, kj, ka in triples:
        kb = cc.khelper.kconserv[ki, ka, kj]
        delta = scaled[ki] - scaled[ka] + scaled[kj] - scaled[kb]
        assert np.allclose(delta, np.rint(delta), atol=1e-10)


def _load_three_mesh_reference(monkeypatch, filename, module_name):
    """Load an explicitly configured CCD comparison implementation."""
    from fsec.staggered_mesh.cc import kintermediates_rhf_stagger

    reference_dir = os.environ.get('FSEC_CCD_REFERENCE_DIR')
    if not reference_dir:
        pytest.skip('Set FSEC_CCD_REFERENCE_DIR to enable reference comparisons')
    reference_path = Path(reference_dir) / filename
    monkeypatch.setitem(sys.modules, 'kintermediates_rhf_stagger',
                        kintermediates_rhf_stagger)
    spec = importlib.util.spec_from_file_location(module_name, reference_path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    monkeypatch.setattr(reference.KRCCD_3mesh, '__init__', KRCCD_3mesh.__init__)
    monkeypatch.setattr(np, 'float', float, raising=False)
    return reference


@pytest.mark.slow
def test_two_mesh_submesh_ccd_converges_with_zero_singles():
    """The two-mesh submesh path converges for a compact insulating cell."""
    cell = _make_cell()
    mf = _run_reference(cell, [2, 2, 2])
    cc = KRCCD_2mesh(mf, flag_submesh=True)
    cc.verbose = 0
    cc.conv_tol = 1e-9
    cc.conv_tol_normt = 1e-8

    e_corr, t1, t2 = cc.kernel()

    assert cc.converged
    assert e_corr == pytest.approx(-0.0160009060641, abs=3e-6)
    assert np.count_nonzero(t1) == 0
    assert np.isfinite(t2).all()
    assert not np.intersect1d(cc.kpts_idx_occ, cc.kpts_idx_vir).size


@pytest.mark.slow
def test_two_mesh_band_extension_conserves_momentum():
    """The two-mesh extension builds nonzero integrals on its requested mesh."""
    cell = _make_cell()
    mf = _run_reference(cell, [1, 1, 1])
    cc = KRCCD_2mesh(mf, flag_submesh=False, nks_ccd=[1, 1, 2])
    cc.verbose = 0
    cc.conv_tol = 1e-9
    cc.conv_tol_normt = 1e-8

    eris = cc.ao2mo()
    e_corr, t1, t2 = cc.kernel(eris=eris)

    triples = ((ki, kj, ka)
               for ki in cc.kpts_idx_occ
               for kj in cc.kpts_idx_occ
               for ka in cc.kpts_idx_vir)
    _assert_momentum_conservation(cell, cc, triples)
    assert abs(eris.oovv).max() > 1e-3
    assert cc.converged
    assert e_corr == pytest.approx(-0.0254100733791, abs=3e-6)
    assert np.count_nonzero(t1) == 0
    assert np.isfinite(t2).all()


@pytest.mark.slow
def test_three_mesh_submesh_ccd_converges_with_zero_singles():
    """The symmetric three-mesh amplitudes conserve momentum and converge."""
    cell = _make_cell()
    mf = _run_reference(cell, [3, 1, 1])
    cc = KRCCD_3mesh(mf, flag_submesh=True)
    cc.verbose = 0
    cc.conv_tol = 1e-9
    cc.conv_tol_normt = 1e-8

    e_corr, t1, t2 = cc.kernel()

    triples = ((ki, kj, ka)
               for ki in cc.kpts_base_idx[0]
               for kj in cc.kpts_base_idx[0]
               for ka in cc.kpts_base_idx[1])
    _assert_momentum_conservation(cell, cc, triples)
    assert cc.converged
    assert e_corr == pytest.approx(-0.0154436942304, abs=3e-6)
    assert np.count_nonzero(t1) == 0
    assert np.isfinite(t2).all()
    assert len(np.unique(np.concatenate(cc.kpts_base_idx))) == 3 * cc.nkpts_base


@pytest.mark.slow
def test_three_mesh_band_extension_converges():
    """The three-mesh extension generates shifted bands from a Gamma reference."""
    cell = _make_cell()
    mf = _run_reference(cell, [1, 1, 1])
    cc = KRCCD_3mesh(mf, flag_submesh=False)
    cc.verbose = 0
    cc.conv_tol = 1e-9
    cc.conv_tol_normt = 1e-8

    e_corr, t1, t2 = cc.kernel()

    assert cc.converged
    assert e_corr == pytest.approx(-0.0154318339157, abs=3e-6)
    assert np.count_nonzero(t1) == 0
    assert np.isfinite(t2).all()
    assert len(np.unique(np.concatenate(cc.kpts_base_idx))) == 3 * cc.nkpts_base


@pytest.mark.slow
def test_three_mesh_madelung_corrections_use_the_shifted_reciprocal_grid():
    """The three shifted meshes retain distinct Gaussian Madelung corrections."""
    cell = _make_cell()
    mf = _run_reference(cell, [3, 1, 1])
    cc = KRCCD_3mesh(mf, flag_submesh=True, madelung=True)

    assert np.allclose(
        cc.madelung_correction,
        [0.421402105473, -0.0467701597203, -0.0467701597203],
        atol=2e-6,
    )


@pytest.mark.slow
def test_krks_band_extension_builds_integrals_for_both_solvers():
    """KRKS factories use their DFT potential when generating shifted bands."""
    cell = _make_cell()
    mf = _run_reference(cell, [1, 1, 1], ks=True)

    cc2 = KRCCD_2mesh(mf, flag_submesh=False, nks_ccd=[1, 1, 2])
    eris2 = cc2.ao2mo()
    cc3 = KRCCD_3mesh(mf, flag_submesh=False)
    eris3 = cc3.ao2mo()

    assert np.isfinite(eris2.oovv).all()
    assert abs(eris2.oovv).max() > 1e-3
    assert np.isfinite(list(eris3.oovv.values())).all()
    assert max(abs(block).max() for block in eris3.oovv.values()) > 1e-3


@pytest.mark.slow
def test_three_mesh_matches_reference_symmetric_solver(monkeypatch):
    """The port reproduces converged CCD energy from the symmetric reference solver."""
    reference = _load_three_mesh_reference(
        monkeypatch, 'kccd_rhf_3mesh.py', '_fsec_reference_kccd_3mesh'
    )

    cell = _make_cell()
    mf = _run_reference(cell, [3, 1, 1])
    symmetric = KRCCD_3mesh(mf, flag_submesh=True)
    reference_solver = reference.KRCCD_3mesh(mf, flag_submesh=True)
    symmetric.verbose = reference_solver.verbose = 0
    symmetric.conv_tol = reference_solver.conv_tol = 1e-9
    symmetric.conv_tol_normt = reference_solver.conv_tol_normt = 1e-8
    reference_solver.e_hf = mf.e_tot

    e_symmetric, _, _ = symmetric.kernel()
    e_reference, _, _ = reference_solver.kernel()

    assert symmetric.converged
    assert reference_solver.converged
    assert e_symmetric == pytest.approx(e_reference, abs=1e-9)


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        'The reference variant averages an extra contraction route when the '
        'third amplitude dependency has length four; its converged energy '
        'differs by 1.92e-6 Ha from the symmetric solver.'
    ),
)
def test_three_mesh_symmetry_matches_nosymm_reference(monkeypatch):
    """Compare converged symmetry-reduced CCD against the independent variant."""
    reference = _load_three_mesh_reference(
        monkeypatch, 'kccd_rhf_3mesh_nosymm.py',
        '_fsec_reference_kccd_3mesh_nosymm',
    )
    cell = _make_cell()
    mf = _run_reference(cell, [3, 1, 1])
    symmetric = KRCCD_3mesh(mf, flag_submesh=True)
    reference_solver = reference.KRCCD_3mesh(mf, flag_submesh=True)
    reference_solver.kpts_ijab = [
        (0, 0, 1, 2), (0, 0, 2, 1), (0, 1, 1, 0), (1, 0, 0, 1),
        (0, 1, 2, 2), (1, 0, 2, 2), (0, 2, 1, 1), (2, 0, 1, 1),
        (0, 2, 2, 0), (2, 0, 0, 2), (1, 1, 0, 2), (1, 1, 2, 0),
        (1, 2, 0, 0), (2, 1, 0, 0), (1, 2, 2, 1), (2, 1, 1, 2),
        (2, 2, 0, 1), (2, 2, 1, 0),
    ]
    symmetric.verbose = reference_solver.verbose = 0
    symmetric.conv_tol = reference_solver.conv_tol = 1e-9
    symmetric.conv_tol_normt = reference_solver.conv_tol_normt = 1e-8
    # The reference wrapper predates the current CCD kernel's HF-energy setup.
    reference_solver.e_hf = mf.e_tot

    e_symmetric, _, _ = symmetric.kernel()
    e_reference, _, _ = reference_solver.kernel()

    assert symmetric.converged
    assert reference_solver.converged
    assert e_symmetric == pytest.approx(e_reference, abs=1e-9)
