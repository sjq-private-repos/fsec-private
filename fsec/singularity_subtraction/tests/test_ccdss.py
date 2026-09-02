"""Tests for unconstrained iterative CCD singularity subtraction."""

from types import SimpleNamespace
from unittest import mock
import io

import numpy as np
import pytest

from pyscf.lib import logger
from pyscf.pbc import gto, scf
from pyscf.pbc.cc import kccsd_rhf

from fsec.staggered_mesh.cc import KRCCD
from fsec.singularity_subtraction.ccdss import (
    CCDSSOptions,
    KRCCD_SS,
    _fit_unit_gaussian,
    _line_samples,
    _normalize_occupied_orbital_shift,
)


class FakeCell:
    """Minimal orthorhombic cell for reciprocal-line tests."""

    vol = 8.0

    def reciprocal_vectors(self):
        """Return distinct reciprocal lengths."""
        return np.diag([2.0, 3.0, 4.0])


def solver_shell(nkpts=1, nocc=1, nvir=1, fixed_sigma=0.0):
    """Build a lightweight solver without invoking the PySCF constructor."""
    solver = object.__new__(KRCCD_SS)
    solver.options = CCDSSOptions(fixed_sigma=fixed_sigma)
    solver.kpts = np.zeros((nkpts, 3))
    solver.mo_occ = np.asarray(
        [np.r_[np.full(nocc, 2.0), np.zeros(nvir)] for _ in range(nkpts)]
    )
    solver._nocc = nocc
    solver._nmo = nocc + nvir
    solver.get_nocc = lambda per_kpoint=False: (
        np.full(nkpts, nocc, dtype=int) if per_kpoint else nocc
    )
    solver.get_nmo = lambda per_kpoint=False: (
        np.full(nkpts, nocc + nvir, dtype=int)
        if per_kpoint
        else nocc + nvir
    )
    solver.frozen = None
    solver.level_shift = 0.0
    solver.madelung_constant = -0.75
    solver.madelung_orbital = True
    solver.madelung_eri = False
    solver.keep_exxdiv = False
    solver.occupied_orbital_shift = None
    solver.khelper = SimpleNamespace(
        kconserv=np.fromfunction(
            lambda ki, ka, kj: (ki - ka + kj) % nkpts,
            (nkpts, nkpts, nkpts),
            dtype=int,
        ).astype(int)
    )
    solver.ss_xi = None
    solver.ss_sigmas = None
    solver.ss_fit_count = 0
    solver.ss_prepare_count = 0
    solver.last_ss_residual_norm = 0.0
    solver._ss_active_masks = None
    solver._ss_pair_densities = None
    solver.verbose = logger.NOTE
    solver.stdout = io.StringIO()
    return solver


def aggregate_solver(imag_tol=1e6):
    """Return a three-k-point shell with complex densities and padding."""
    solver = solver_shell(nkpts=3, nocc=2, nvir=2, fixed_sigma=None)
    solver.options = CCDSSOptions(
        fixed_sigma=None, structure_factor_imag_tol=imag_tol
    )
    solver._ss_q_vectors = np.asarray(
        [[0.0, 0.0, 0.0], [0.7, -0.2, 0.1]]
    )
    solver._ss_plus = np.asarray([[0, 1, 2], [1, 2, 0]])
    solver._ss_minus = np.asarray([[0, 1, 2], [2, 0, 1]])
    occupied = np.asarray([[True, True], [True, False], [True, True]])
    virtual = np.asarray([[True, False], [True, True], [False, True]])
    solver._ss_active_masks = occupied, virtual
    rng = np.random.default_rng(18)
    rho_oo = rng.normal(size=(2, 3, 2, 2)) + 1j * rng.normal(
        size=(2, 3, 2, 2)
    )
    rho_vv = rng.normal(size=(2, 3, 2, 2)) + 1j * rng.normal(
        size=(2, 3, 2, 2)
    )
    rho_oo[0] = np.eye(2)
    rho_vv[0] = np.eye(2)
    solver._ss_pair_densities = rho_oo, rho_vv
    return solver


def explicit_structure_factors(solver, t2):
    """Evaluate all six equations with literal orbital-index loops."""
    occupied, virtual = solver._build_active_masks()
    rho_oo_all, rho_vv_all = solver._ss_pair_densities
    kconserv = solver.khelper.kconserv
    nocc = solver.nocc
    nvir = solver.nmo - nocc
    raw = np.zeros((6, len(solver._ss_q_vectors)), dtype=complex)

    def amp_active(k1, k2, ka, i, j, a, b):
        kb = kconserv[k1, ka, k2]
        return (
            occupied[k1, i]
            and occupied[k2, j]
            and virtual[ka, a]
            and virtual[kb, b]
        )

    for iq in range(len(solver._ss_q_vectors)):
        plus = solver._ss_plus[iq]
        minus = solver._ss_minus[iq]
        rho_oo = rho_oo_all[iq]
        rho_vv = rho_vv_all[iq]
        for ki in range(solver.nkpts):
            for kj in range(solver.nkpts):
                for ka in range(solver.nkpts):
                    kb = kconserv[ki, ka, kj]
                    for i in range(nocc):
                        for j in range(nocc):
                            for a in range(nvir):
                                for b in range(nvir):
                                    for k in range(nocc):
                                        for l in range(nocc):
                                            if (
                                                occupied[minus[ki], k]
                                                and occupied[ki, i]
                                                and occupied[kj, j]
                                                and occupied[plus[kj], l]
                                                and amp_active(
                                                    minus[ki], plus[kj], ka,
                                                    k, l, a, b,
                                                )
                                            ):
                                                raw[0, iq] += (
                                                    rho_oo[minus[ki], k, i]
                                                    * rho_oo[kj, j, l].conjugate()
                                                    * t2[
                                                        minus[ki], plus[kj], ka,
                                                        k, l, a, b,
                                                    ]
                                                )
                                    for c in range(nvir):
                                        for d in range(nvir):
                                            if (
                                                virtual[ka, a]
                                                and virtual[plus[ka], c]
                                                and virtual[minus[kb], d]
                                                and virtual[kb, b]
                                                and amp_active(
                                                    ki, kj, plus[ka], i, j, c, d
                                                )
                                            ):
                                                raw[1, iq] += (
                                                    rho_vv[ka, a, c]
                                                    * rho_vv[
                                                        minus[kb], d, b
                                                    ].conjugate()
                                                    * t2[
                                                        ki, kj, plus[ka], i, j, c, d
                                                    ]
                                                )
                                    for c in range(nvir):
                                        for k in range(nocc):
                                            if (
                                                virtual[ka, a]
                                                and virtual[plus[ka], c]
                                                and occupied[ki, i]
                                                and occupied[plus[ki], k]
                                                and amp_active(
                                                    plus[ki], kj, plus[ka],
                                                    k, j, c, b,
                                                )
                                            ):
                                                raw[2, iq] += (
                                                    rho_vv[ka, a, c]
                                                    * rho_oo[ki, i, k].conjugate()
                                                    * t2[
                                                        plus[ki], kj, plus[ka],
                                                        k, j, c, b,
                                                    ]
                                                )
                                            if (
                                                virtual[kb, b]
                                                and virtual[plus[kb], c]
                                                and occupied[kj, j]
                                                and occupied[plus[kj], k]
                                                and amp_active(
                                                    plus[kj], ki, plus[kb],
                                                    k, i, c, a,
                                                )
                                            ):
                                                raw[3, iq] += (
                                                    rho_vv[kb, b, c]
                                                    * rho_oo[kj, j, k].conjugate()
                                                    * t2[
                                                        plus[kj], ki, plus[kb],
                                                        k, i, c, a,
                                                    ]
                                                )
                                            if (
                                                virtual[ka, a]
                                                and virtual[plus[ka], c]
                                                and occupied[kj, j]
                                                and occupied[plus[kj], k]
                                                and amp_active(
                                                    plus[kj], ki, kb,
                                                    k, i, b, c,
                                                )
                                            ):
                                                raw[4, iq] += (
                                                    rho_vv[ka, a, c]
                                                    * rho_oo[kj, j, k].conjugate()
                                                    * t2[
                                                        plus[kj], ki, kb,
                                                        k, i, b, c,
                                                    ]
                                                )
                                            if (
                                                virtual[kb, b]
                                                and virtual[plus[kb], c]
                                                and occupied[ki, i]
                                                and occupied[plus[ki], k]
                                                and amp_active(
                                                    plus[ki], kj, ka,
                                                    k, j, a, c,
                                                )
                                            ):
                                                raw[5, iq] += (
                                                    rho_vv[kb, b, c]
                                                    * rho_oo[ki, i, k].conjugate()
                                                    * t2[
                                                        plus[ki], kj, ka,
                                                        k, j, a, c,
                                                    ]
                                                )
    return raw


def symmetric_t2(solver, seed=31):
    """Return padded amplitudes satisfying restricted-CCD pair symmetry."""
    rng = np.random.default_rng(seed)
    shape = (
        solver.nkpts,
        solver.nkpts,
        solver.nkpts,
        solver.nocc,
        solver.nocc,
        solver.nmo - solver.nocc,
        solver.nmo - solver.nocc,
    )
    source = rng.normal(size=shape)
    result = np.zeros(shape)
    for ki in range(solver.nkpts):
        for kj in range(solver.nkpts):
            for ka in range(solver.nkpts):
                kb = solver.khelper.kconserv[ki, ka, kj]
                block = source[ki, kj, ka]
                partner = source[kj, ki, kb].transpose(1, 0, 3, 2)
                result[ki, kj, ka] = 0.5 * (block + partner)
    return result


def test_options_and_removed_constraint_api():
    """Validate new defaults and reject obsolete keywords normally."""
    defaults = CCDSSOptions()
    assert defaults.pair_density_becke_grid_level == 2
    assert defaults.structure_factor_imag_tol == pytest.approx(1e-5)
    for value in (-1.0, np.inf, np.nan):
        with pytest.raises(ValueError, match="structure_factor_imag_tol"):
            CCDSSOptions(structure_factor_imag_tol=value)
    with pytest.raises(TypeError, match="unexpected keyword"):
        CCDSSOptions(use_constraint_1=False)
    with pytest.raises(TypeError, match="unexpected keyword"):
        KRCCD_SS(None, use_constraint_2=False)


def test_line_sampling_and_gaussian_fit():
    """Retain positive reciprocal lines and unit-Gaussian fitting."""
    kpts = np.zeros((2, 3))
    with mock.patch(
        "fsec.singularity_subtraction.ccdss.tools.get_monkhorst_pack_size",
        return_value=np.asarray([2, 3, 4]),
    ):
        points, metadata, mesh = _line_samples(FakeCell(), kpts, 2)
    np.testing.assert_array_equal(mesh, [2, 3, 4])
    np.testing.assert_allclose(
        points,
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [0, 2, 0],
         [0, 0, 1], [0, 0, 2]],
    )
    assert len(metadata) == 3
    sigma = 0.8
    values = np.exp(-np.linalg.norm(points, axis=1) ** 2 / (2 * sigma**2))
    assert _fit_unit_gaussian(points, values) == pytest.approx(sigma, rel=1e-6)


def test_pair_densities_cache_separate_occupied_and_virtual_blocks():
    """Store exact identity origins and no occupied--virtual blocks."""
    solver = solver_shell(nkpts=2, nocc=1, nvir=2, fixed_sigma=None)
    solver._ss_q_vectors = np.zeros((1, 3))
    solver._ss_plus = np.asarray([[0, 1]])
    solver._ss_minus = np.asarray([[0, 1]])
    coords = np.zeros((2, 3))
    weights = np.ones(2)
    u = np.ones((2, 3, 2), dtype=complex)
    norms = np.ones((2, 3))
    with mock.patch.object(
        solver,
        "_build_pair_orbitals",
        return_value=(coords, weights, u, norms, np.zeros((2, 3))),
    ):
        first = solver._build_pair_densities()
        second = solver._build_pair_densities()
    assert first is second
    rho_oo, rho_vv = first
    assert rho_oo.shape == (1, 2, 1, 1)
    assert rho_vv.shape == (1, 2, 2, 2)
    np.testing.assert_array_equal(rho_oo[0], np.broadcast_to(np.eye(1), (2, 1, 1)))
    np.testing.assert_array_equal(rho_vv[0], np.broadcast_to(np.eye(2), (2, 2, 2)))


def test_all_six_contractions_match_complex_shifted_loop_reference():
    """Match literal loops with complex densities, shifted T2, and padding."""
    solver = aggregate_solver()
    rng = np.random.default_rng(44)
    t2 = rng.normal(size=(3, 3, 3, 2, 2, 2, 2)) + 1j * rng.normal(
        size=(3, 3, 3, 2, 2, 2, 2)
    )
    expected = explicit_structure_factors(solver, t2)
    actual = solver._contract_aggregate_structure_factors(t2)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


def test_physical_origins_equal_aggregate_t2_and_normalize_to_one():
    """Recover the common aggregate origin for pair-symmetric amplitudes."""
    solver = aggregate_solver()
    t2 = symmetric_t2(solver)
    raw = solver._contract_aggregate_structure_factors(t2)
    occupied, virtual = solver._build_active_masks()
    aggregate = 0.0
    for ki in range(solver.nkpts):
        for kj in range(solver.nkpts):
            for ka in range(solver.nkpts):
                kb = solver.khelper.kconserv[ki, ka, kj]
                aggregate += t2[ki, kj, ka][
                    np.ix_(occupied[ki], occupied[kj], virtual[ka], virtual[kb])
                ].sum()
    np.testing.assert_allclose(raw[:, 0], aggregate, atol=1e-12)
    normalized = solver._normalize_aggregate_structure_factors(raw)
    np.testing.assert_array_equal(normalized[:, 0], np.ones(6))


def test_normalization_rejects_small_origins_and_large_l1_l2_imaginary_residues():
    """Guard L1/L2 complex residues while allowing complex L3--L6 curves."""
    solver = solver_shell(fixed_sigma=None)
    solver._ss_q_vectors = np.zeros((2, 3))
    raw = np.ones((6, 2), dtype=complex)
    raw[2, 0] = 1e-14
    with pytest.raises(ValueError, match=r"L3.*origin.*amplitude_fit_tol"):
        solver._normalize_aggregate_structure_factors(raw)

    solver.options = CCDSSOptions(
        fixed_sigma=None, structure_factor_imag_tol=1e-5
    )
    raw[:] = 2.0 + 1e-6j
    raw[:, 1] = 1.0 + 2e-6j
    normalized = solver._normalize_aggregate_structure_factors(raw)
    assert np.max(np.abs(normalized.imag)) < 1e-5

    raw[2:, 1] = np.asarray(
        [1.0 + 3e-5j, 1.0 + 4e-5j, 1.0 + 5e-5j, 1.0 + 6e-5j]
    )
    normalized = solver._normalize_aggregate_structure_factors(raw)
    assert np.all(np.abs(normalized[2:, 1].imag) > 1e-5)


def test_normalization_checks_imaginary_part_created_by_complex_origin():
    """Check normalized phase even when every raw imaginary part is small."""
    solver = solver_shell(fixed_sigma=None)
    solver._ss_q_vectors = np.zeros((2, 3))
    solver.options = CCDSSOptions(
        fixed_sigma=None, structure_factor_imag_tol=1e-5
    )
    raw = np.ones((6, 2), dtype=complex)
    raw[1, 0] = 1.0 + 9e-6j
    raw[1, 1] = 0.1 - 9.5e-6j
    with pytest.raises(ValueError, match=r"L2 sample 1 normalized imaginary"):
        solver._normalize_aggregate_structure_factors(raw)


def test_two_fitted_residual_updates_refit_changed_amplitudes():
    """Perform six fits per update and pass changed real curves to fitting."""
    solver = aggregate_solver(imag_tol=1e6)
    solver._ss_active_masks = (
        np.ones((3, 2), dtype=bool), np.ones((3, 2), dtype=bool)
    )
    first = symmetric_t2(solver, seed=4)
    changed = first.copy()
    changed[0, 0, 0, 0, 0, 0, 0] += 0.7
    fitted_curves = []

    def fake_fit(_q, values, fit_with_coul=True):
        fitted_curves.append(np.array(values, copy=True))
        return 0.5 + 0.01 * len(fitted_curves)

    with mock.patch(
        "fsec.singularity_subtraction.ccdss._fit_unit_gaussian",
        side_effect=fake_fit,
    ), mock.patch.object(solver, "_gaussian_xi", side_effect=lambda x: x):
        solver._ss_residual_coefficient(first)
        first_xi = solver.ss_xi.copy()
        solver._ss_residual_coefficient(changed)

    assert solver.ss_prepare_count == 2
    assert solver.ss_fit_count == 12
    assert len(fitted_curves) == 12
    assert any(
        not np.allclose(fitted_curves[channel], fitted_curves[channel + 6])
        for channel in range(6)
    )
    assert all(np.isrealobj(values) for values in fitted_curves)
    assert not np.array_equal(first_xi, solver.ss_xi)
    assert solver.ss_sigmas.shape == solver.ss_xi.shape == (6,)


def test_failed_fitted_preparation_does_not_advance_state_or_counters():
    """Commit counters and latest arrays only after all six fits succeed."""
    solver = aggregate_solver(imag_tol=1e6)
    t2 = symmetric_t2(solver)
    with mock.patch(
        "fsec.singularity_subtraction.ccdss._fit_unit_gaussian",
        side_effect=[1.0, 1.0, ValueError("fit failed")],
    ), mock.patch.object(
        solver, "_gaussian_xi", return_value=0.1
    ), pytest.raises(ValueError, match="fit failed"):
        solver._prepare_ss(t2)
    assert solver.ss_prepare_count == solver.ss_fit_count == 0
    assert solver.ss_sigmas is None and solver.ss_xi is None


def test_fixed_sigma_is_cached_without_sampling_or_fitting():
    """Prepare an amplitude-independent fixed width exactly once."""
    solver = solver_shell(fixed_sigma=np.inf)
    t2 = np.ones((1, 1, 1, 1, 1, 1, 1))
    with mock.patch.object(
        solver, "_aggregate_channel_samples"
    ) as sample, mock.patch(
        "fsec.singularity_subtraction.ccdss._fit_unit_gaussian"
    ) as fit:
        first = solver._prepare_ss(t2)
        second = solver._prepare_ss(3 * t2)
    assert first is second
    np.testing.assert_array_equal(first, np.full(6, 0.75))
    assert solver.ss_prepare_count == 1
    assert solver.ss_fit_count == 0
    sample.assert_not_called()
    fit.assert_not_called()


def test_debug2_reports_raw_and_normalized_rows_on_every_fit():
    """Emit one parseable row per preparation, channel, and q sample."""
    solver = solver_shell(fixed_sigma=None)
    solver.options = CCDSSOptions(
        fixed_sigma=None, structure_factor_imag_tol=1e-5
    )
    solver._ss_q_vectors = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=float)
    solver.verbose = logger.DEBUG2
    t2 = np.ones((1, 1, 1, 1, 1, 1, 1))
    raw = np.ones((6, 2), dtype=complex) * 2
    normalized = raw / 2
    with mock.patch.object(
        solver, "_aggregate_channel_samples", return_value=(raw, normalized)
    ), mock.patch(
        "fsec.singularity_subtraction.ccdss._fit_unit_gaussian", return_value=2.0
    ), mock.patch.object(solver, "_gaussian_xi", return_value=0.25):
        solver._prepare_ss(t2)
        solver._prepare_ss(t2)
    rows = [
        line for line in solver.stdout.getvalue().splitlines()
        if line.startswith("CCDSS_SF ")
    ]
    assert len(rows) == 24
    assert "prep=1 channel=L1 q_index=0" in rows[0]
    assert "prep=2 channel=L6 q_index=1" in rows[-1]
    for field in (
        "raw_real=", "raw_imag=", "normalized_real=", "normalized_imag=",
        "sigma=", "xi=", "fit=", "residual_real=", "residual_imag=",
    ):
        assert field in rows[0]


def test_fixed_sigma_limits_signs_and_residual_before_division():
    """Retain six signs, Madelung limits, and numerator-level injection."""
    solver = solver_shell()
    solver.ss_xi = np.arange(1.0, 7.0)
    assert solver._ss_residual_coefficient(np.ones((1,) * 7)) == -15.0

    solver.ss_xi = np.asarray([0.5, 0.5, 0, 0, 0, 0])
    t2 = np.full((1, 1, 1, 1, 1, 1, 1), 3.0)
    t2new = np.full_like(t2, 5.0)
    eris = SimpleNamespace(mo_energy=[np.asarray([-1.0, 2.0])])
    with mock.patch(
        "fsec.singularity_subtraction.ccdss.padding_k_idx",
        return_value=([np.asarray([0])], [np.asarray([0])]),
    ):
        result = solver._inject_ss_residual(t2new, t2, eris)
    denominator = -6.0
    assert result.item() == pytest.approx((5.0 * denominator + 3.0) / denominator)
    assert solver._gaussian_xi(0.0) == 0.0
    assert solver._gaussian_xi(np.inf) == 0.75


def test_occupied_orbital_shift_validation_and_application():
    """Broadcast custom shifts and apply them only to active occupied states."""
    scalar = _normalize_occupied_orbital_shift(-0.2, 2, 2)
    np.testing.assert_array_equal(scalar, np.full((2, 2), -0.2))
    for invalid in (np.zeros(2), np.inf, 1 + 0j, True):
        with pytest.raises(ValueError, match="occupied_orbital_shift"):
            _normalize_occupied_orbital_shift(invalid, 2, 2)

    solver = solver_shell(nkpts=2, nocc=2, nvir=1)
    solver.occupied_orbital_shift = np.asarray([[0.1, -0.2], [0.3, 9.0]])
    solver.keep_exxdiv = True
    eris = SimpleNamespace(
        fock=np.asarray([np.diag([1.0, 2.0, 3.0]), np.diag([4.0, 5.0, 6.0])]),
        mo_energy=[np.asarray([1.0, 2.0, 3.0]), np.asarray([4.0, 5.0, 6.0])],
    )
    with mock.patch.object(
        kccsd_rhf.RCCSD, "ao2mo", return_value=eris
    ), mock.patch(
        "fsec.singularity_subtraction.ccdss.padding_k_idx",
        return_value=(
            [np.asarray([0, 1]), np.asarray([0])],
            [np.asarray([0]), np.asarray([0])],
        ),
    ):
        result = solver.ao2mo()
    np.testing.assert_allclose(np.diagonal(result.fock, axis1=1, axis2=2),
                               [[1.1, 1.8, 3.0], [4.3, 5.0, 6.0]])
    assert solver.keep_exxdiv is True and solver.madelung_orbital is True


def test_init_defers_fitted_preparation_and_update_keeps_t1_zero():
    """Do no fitted work in initialization and keep CCD singles zero."""
    solver = solver_shell(fixed_sigma=None)
    t1 = np.ones((1, 1, 1))
    t2 = np.ones((1, 1, 1, 1, 1, 1, 1))
    with mock.patch.object(
        KRCCD, "init_amps", return_value=(1.0, t1, t2)
    ), mock.patch.object(solver, "_prepare_ss") as prepare:
        _, initial_t1, _ = solver.init_amps(SimpleNamespace())
    np.testing.assert_array_equal(initial_t1, 0)
    prepare.assert_not_called()

    with mock.patch.object(
        kccsd_rhf.RCCSD, "update_amps", return_value=(t1, 2 * t2)
    ), mock.patch.object(solver, "_inject_ss_residual", side_effect=lambda out, *_: out):
        updated_t1, updated_t2 = solver.update_amps(t1, t2, SimpleNamespace())
    np.testing.assert_array_equal(updated_t1, 0)
    np.testing.assert_array_equal(updated_t2, 2)


@pytest.mark.slow
def test_h2_1x1x2_limits_and_dynamic_fit():
    """Converge H2 with level-2 densities and six fits per CCD update."""
    cell = gto.Cell()
    cell.unit = "Bohr"
    cell.atom = "H 0 0 0; H 1.8 0 0"
    cell.a = np.eye(3) * 6.0
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pade"
    cell.verbose = 0
    cell.build()
    kpts = cell.make_kpts([1, 1, 2], with_gamma_point=True)
    mf = scf.KRHF(cell, kpts).density_fit()
    mf.exxdiv = None
    mf.conv_tol = 1e-9
    mf.kernel()
    assert mf.converged

    calculations = {
        "orbital": KRCCD(mf, madelung_orbital=True),
        "zero": KRCCD_SS(mf, fixed_sigma=0.0),
        "madelung": KRCCD(mf, madelung_orbital=True, madelung_eri=True),
        "infinite": KRCCD_SS(mf, fixed_sigma=np.inf),
        "fitted": KRCCD_SS(mf),
    }
    results = {}
    for name, cc in calculations.items():
        cc.keep_exxdiv = False
        cc.conv_tol = 1e-7
        cc.max_cycle = 30
        energy, _, t2 = cc.kernel()
        assert cc.converged and np.isfinite(energy)
        results[name] = energy, t2

    assert results["zero"][0] == pytest.approx(results["orbital"][0], abs=1e-11)
    assert results["infinite"][0] == pytest.approx(
        results["madelung"][0], abs=1e-11
    )
    fitted = calculations["fitted"]
    assert fitted.options.pair_density_becke_grid_level == 2
    assert fitted.ss_prepare_count == fitted.cycles
    assert fitted.ss_fit_count == 6 * fitted.ss_prepare_count
    assert np.all(np.isfinite(fitted.ss_sigmas))
    assert fitted.ss_sigmas.shape == fitted.ss_xi.shape == (6,)
    raw, normalized = fitted._aggregate_channel_samples(results["fitted"][1])
    scaled_raw_imag = np.abs(raw.imag) / np.abs(raw[:, :1])
    assert np.max(scaled_raw_imag) < 1e-5
    assert np.max(np.abs(normalized.imag)) < 1e-5
