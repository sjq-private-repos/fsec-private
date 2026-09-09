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
    _fit_unit_quartic_exponential,
    _line_samples,
    _normalize_occupied_orbital_shift,
)
from fsec.singularity_subtraction.model_function.exx_modfunc import (
    QuarticExponentialModel,
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
    solver.ss_model_parameters = None
    solver.ss_fit_count = 0
    solver.ss_prepare_count = 0
    solver.last_ss_residual_norm = 0.0
    solver._ss_active_masks = None
    solver._ss_pair_densities = None
    solver.verbose = logger.NOTE
    solver.stdout = io.StringIO()
    return solver


def aggregate_solver():
    """Return a three-k-point shell with complex densities and padding."""
    solver = solver_shell(nkpts=3, nocc=2, nvir=2, fixed_sigma=None)
    solver.options = CCDSSOptions(fixed_sigma=None)
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
    """Validate defaults and the pair-density identity tolerance."""
    defaults = CCDSSOptions()
    assert defaults.auxfunc == "Gauss"
    assert defaults.structure_factor_mode == "separate"
    assert defaults.pair_density_becke_grid_level == 2
    assert defaults.pair_density_identity_tol == pytest.approx(1e-3)
    for value in (-1.0, np.inf, np.nan):
        with pytest.raises(ValueError, match="pair_density_identity_tol"):
            CCDSSOptions(pair_density_identity_tol=value)
    with pytest.raises(TypeError, match="unexpected keyword"):
        CCDSSOptions(use_constraint_1=False)
    with pytest.raises(TypeError, match="unexpected keyword"):
        KRCCD_SS(None, use_constraint_2=False)
    with pytest.raises(ValueError, match="auxfunc"):
        CCDSSOptions(auxfunc="Lorentzian")
    with pytest.raises(ValueError, match="structure_factor_mode"):
        CCDSSOptions(structure_factor_mode="aggregate")
    with pytest.raises(ValueError, match="structure_factor_mode"):
        KRCCD_SS(None, structure_factor_mode="aggregate")
    with pytest.raises(ValueError, match="fixed_sigma"):
        CCDSSOptions(auxfunc="QuarticExponential", fixed_sigma=0.0)


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


def test_quartic_fit_recovers_positive_synthetic_parameters():
    """Recover a unit-coefficient quartic model from exact samples."""
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.8, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.5, 0.0],
            [0.0, 0.0, 0.6],
            [0.0, 0.0, 1.2],
            [0.0, 0.0, 1.8],
        ]
    )
    expected = np.asarray([1.7, 0.8, 0.35])
    values = QuarticExponentialModel(
        num_primitives=1,
        parameters=np.r_[1.0, expected].tolist(),
    ).eval_model(points)

    fitted = _fit_unit_quartic_exponential(points, values)

    assert np.all(fitted > 0)
    np.testing.assert_allclose(fitted, expected, rtol=1e-6, atol=1e-8)


def q0_pair_density_solver(active_masks=None, identity_tol=1e-3):
    """Return a two-k-point shell with a single sampled q=0 density."""
    solver = solver_shell(nkpts=2, nocc=2, nvir=2, fixed_sigma=None)
    # Make the second input k-point differ from its minimum-image
    # representative so the fast test exercises the reciprocal-wrap path.
    solver.kpts[1, 0] = np.pi
    solver.options = CCDSSOptions(
        fixed_sigma=None, pair_density_identity_tol=identity_tol
    )
    solver._ss_q_vectors = np.zeros((1, 3))
    solver._ss_plus = np.asarray([[0, 1]], dtype=int)
    solver._ss_minus = np.asarray([[0, 1]], dtype=int)
    if active_masks is None:
        active_masks = (
            np.ones((2, 2), dtype=bool),
            np.ones((2, 2), dtype=bool),
        )
    solver._ss_active_masks = active_masks
    return solver


def q0_pair_orbitals(solver, occupied_residue=0.0, virtual_residue=0.0):
    """Return normalized orbitals with controlled q=0 block residues."""
    coords = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    weights = np.ones(2)
    u = np.zeros((solver.nkpts, solver.nmo, 2), dtype=complex)
    u[:, 0] = [1.0, 0.0]
    u[:, 1] = [
        occupied_residue,
        np.sqrt(1.0 - occupied_residue**2),
    ]
    u[:, 2] = [1.0, 0.0]
    u[:, 3] = [
        virtual_residue,
        np.sqrt(1.0 - virtual_residue**2),
    ]
    norms = np.sum(u.conj() * u, axis=-1).real
    return coords, weights, u, norms, np.zeros((solver.nkpts, 3))


def test_pair_densities_compute_q0_residue_and_cache_after_success():
    """Compute q=0 through overlaps and cache validated active blocks."""
    solver = q0_pair_density_solver()
    data = q0_pair_orbitals(solver, virtual_residue=2e-4)
    with mock.patch.object(
        solver, "_build_pair_orbitals", return_value=data
    ) as build_pair_orbitals:
        first = solver._build_pair_densities()
        second = solver._build_pair_densities()

    assert first is second
    build_pair_orbitals.assert_called_once()
    rho_oo, rho_vv = first
    assert rho_oo.shape == (1, 2, 2, 2)
    assert rho_vv.shape == (1, 2, 2, 2)
    np.testing.assert_array_equal(
        rho_oo[0], np.broadcast_to(np.eye(2), (2, 2, 2))
    )
    assert rho_vv[0, 0, 0, 1] == pytest.approx(2e-4)
    assert rho_vv[0, 0, 0, 1] != 0


def test_pair_densities_reject_occupied_identity_residue_before_caching():
    """Reject an active occupied q=0 block above its tolerance."""
    solver = q0_pair_density_solver(identity_tol=1e-3)
    data = q0_pair_orbitals(solver, occupied_residue=2e-3)
    with mock.patch.object(
        solver, "_build_pair_orbitals", return_value=data
    ), pytest.raises(
        ValueError,
        match=r"sector=occupied.*k-point=0.*deviation=.*pair_density_identity_tol",
    ):
        solver._build_pair_densities()
    assert solver._ss_pair_densities is None


def test_pair_densities_reject_virtual_identity_residue_before_caching():
    """Reject an active virtual q=0 block above its tolerance."""
    solver = q0_pair_density_solver(identity_tol=1e-3)
    data = q0_pair_orbitals(solver, virtual_residue=2e-3)
    with mock.patch.object(
        solver, "_build_pair_orbitals", return_value=data
    ), pytest.raises(
        ValueError,
        match=r"sector=virtual.*k-point=0.*deviation=.*pair_density_identity_tol",
    ):
        solver._build_pair_densities()
    assert solver._ss_pair_densities is None


def test_pair_density_identity_check_ignores_padded_rows_and_columns():
    """Validate only active orbitals when padded q=0 rows are nonidentity."""
    active = np.asarray([[True, False], [True, False]])
    solver = q0_pair_density_solver((active, active))
    data = q0_pair_orbitals(solver, occupied_residue=1.0, virtual_residue=1.0)
    with mock.patch.object(
        solver, "_build_pair_orbitals", return_value=data
    ):
        densities = solver._build_pair_densities()

    rho_oo, rho_vv = densities
    assert abs(rho_oo[0, 0, 0, 1]) > 0.5
    assert abs(rho_vv[0, 0, 0, 1]) > 0.5
    np.testing.assert_array_equal(rho_oo[0, :, :1, :1], 1.0)
    np.testing.assert_array_equal(rho_vv[0, :, :1, :1], 1.0)


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


def test_normalization_rejects_small_origins_and_preserves_complex_curves():
    """Normalize each complex channel independently without residue guards."""
    solver = solver_shell(fixed_sigma=None)
    solver._ss_q_vectors = np.zeros((2, 3))
    raw = np.ones((6, 2), dtype=complex)
    raw[2, 0] = 1e-14
    with pytest.raises(ValueError, match=r"L3.*origin.*amplitude_fit_tol"):
        solver._normalize_aggregate_structure_factors(raw)

    raw = np.asarray(
        [
            [2.0 + 3.0j, 1.0 + 2.0j],
            [3.0 - 4.0j, -1.0 + 2.0j],
            [0.5 + 0.25j, 0.2 - 0.6j],
            [4.0 + 0.5j, 0.1 + 1.0j],
            [1.0 - 2.0j, -0.4 + 0.3j],
            [2.0 + 0.1j, 0.7 - 0.8j],
        ]
    )
    normalized = solver._normalize_aggregate_structure_factors(raw)
    expected = raw / raw[:, :1]
    expected[:, 0] = 1.0
    np.testing.assert_allclose(normalized, expected)
    np.testing.assert_array_equal(normalized[:, 0], np.ones(6))
    assert normalized[1, 1].imag != 0


def test_combined_signed_reduction_normalizes_once_and_preserves_complex_values():
    """Reduce the six raw curves before validating and normalizing one curve."""
    solver = solver_shell(fixed_sigma=None)
    solver.options = CCDSSOptions(
        fixed_sigma=None, structure_factor_mode="combined"
    )
    solver._ss_q_vectors = np.zeros((2, 3))
    raw = np.asarray(
        [
            [2.0 + 1.0j, 1.0 + 2.0j],
            [3.0 - 2.0j, -1.0 + 1.0j],
            [0.5 + 0.25j, 0.2 - 0.6j],
            [1.0 - 0.5j, 0.1 + 1.0j],
            [0.75 - 1.0j, -0.4 + 0.3j],
            [1.5 + 0.1j, 0.7 - 0.8j],
        ]
    )
    expected_raw = raw[0] + raw[1] - raw[2] - raw[3] - raw[4] - raw[5]
    with mock.patch.object(
        solver, "_contract_aggregate_structure_factors", return_value=raw
    ):
        combined_raw, normalized = solver._aggregate_channel_samples(None)

    np.testing.assert_allclose(combined_raw, expected_raw)
    np.testing.assert_allclose(normalized, expected_raw / expected_raw[0])
    np.testing.assert_array_equal(normalized[0], 1.0)
    assert normalized[1].imag != 0

    raw[:, 0] = 0.0
    raw[0, 0] = 1e-14
    with mock.patch.object(
        solver, "_contract_aggregate_structure_factors", return_value=raw
    ), pytest.raises(ValueError, match=r"combined.*origin.*amplitude_fit_tol"):
        solver._aggregate_channel_samples(None)


def test_quartic_xi_matches_independent_truncated_reciprocal_sum():
    """Use the model integral and exponent-32 reciprocal truncation."""
    solver = solver_shell(fixed_sigma=None)
    solver.options = CCDSSOptions(auxfunc="QuarticExponential")
    solver._scf = SimpleNamespace(cell=FakeCell())
    parameters = np.asarray([1.0, 1.2, 0.8, 0.3])

    model = QuarticExponentialModel(
        num_primitives=1, parameters=parameters.tolist()
    )
    rhs = 64.0 + 1024.0 / parameters[1]
    cutoff_squared = 2.0 * rhs / (
        parameters[2]
        + np.sqrt(parameters[2] ** 2 + 4.0 * parameters[3] * rhs)
    )
    reciprocal = FakeCell().reciprocal_vectors()
    shell = np.maximum(
        1, np.ceil(np.sqrt(cutoff_squared) / np.linalg.norm(reciprocal, axis=1))
    ).astype(int)
    integers = np.stack(
        np.meshgrid(
            *[np.arange(-n, n + 1) for n in shell], indexing="ij"
        ),
        axis=-1,
    ).reshape(-1, 3)
    qg = integers @ reciprocal
    q2 = np.sum(qg * qg, axis=1)
    values = model.eval_model(qg)
    active = (q2 > 1e-20) & (values >= np.exp(-32.0))
    quadrature = np.sum(4.0 * np.pi * values[active] / q2[active]) / FakeCell.vol
    expected = model.coulomb_integral() - quadrature

    assert solver._quartic_xi(parameters) == pytest.approx(expected, rel=1e-12)


def test_two_fitted_residual_updates_refit_changed_amplitudes():
    """Perform six fits per update and pass changed real curves to fitting."""
    solver = aggregate_solver()
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
    assert solver.ss_model_parameters.shape == (6, 2)
    np.testing.assert_array_equal(solver.ss_model_parameters[:, 0], 1.0)
    np.testing.assert_allclose(
        solver.ss_model_parameters[:, 1], solver.ss_sigmas
    )


def test_combined_gaussian_updates_fit_once_and_use_minus_two_xi():
    """Refit one signed curve per update and commit scalar combined state."""
    solver = aggregate_solver()
    solver.options = CCDSSOptions(
        fixed_sigma=None, structure_factor_mode="combined"
    )
    first = symmetric_t2(solver, seed=4)
    changed = first.copy()
    changed[0, 0, 0, 0, 0, 0, 0] += 0.7
    fitted_curves = []
    expected_curves = []
    for amplitudes in (first, changed):
        raw = solver._contract_aggregate_structure_factors(amplitudes)
        combined_raw = raw[0] + raw[1] - raw[2] - raw[3] - raw[4] - raw[5]
        normalized = combined_raw / combined_raw[0]
        normalized[0] = 1.0
        expected_curves.append(normalized.real)

    def fake_fit(_q, values, fit_with_coul=True):
        fitted_curves.append(np.array(values, copy=True))
        return 0.5 + 0.01 * len(fitted_curves)

    with mock.patch(
        "fsec.singularity_subtraction.ccdss._fit_unit_gaussian",
        side_effect=fake_fit,
    ), mock.patch.object(solver, "_gaussian_xi", side_effect=lambda x: x):
        first_coefficient = solver._ss_residual_coefficient(first)
        first_xi = solver.ss_xi
        second_coefficient = solver._ss_residual_coefficient(changed)

    assert len(fitted_curves) == 2
    np.testing.assert_allclose(fitted_curves, expected_curves)
    assert not np.allclose(fitted_curves[0], fitted_curves[1])
    assert first_coefficient == pytest.approx(-2.0 * first_xi)
    assert second_coefficient == pytest.approx(-2.0 * solver.ss_xi)
    assert solver.ss_prepare_count == 2
    assert solver.ss_fit_count == 2
    assert np.isscalar(solver.ss_xi)
    assert np.isscalar(solver.ss_sigmas)
    assert solver.ss_model_parameters.shape == (2,)
    assert solver.ss_model_parameters[0] == 1.0
    assert solver.ss_model_parameters[1] == solver.ss_sigmas


def test_failed_fitted_preparation_does_not_advance_state_or_counters():
    """Commit counters and latest arrays only after all six fits succeed."""
    solver = aggregate_solver()
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


def test_quartic_preparation_fits_six_channels_and_commits_atomically():
    """Fit exactly six quartic channels and retain state after a failed update."""
    solver = aggregate_solver()
    solver.options = CCDSSOptions(auxfunc="QuarticExponential")
    t2 = symmetric_t2(solver)
    fit_calls = []

    def fake_fit(_q, values, fit_with_coul=True):
        fit_calls.append(np.array(values, copy=True))
        return np.asarray([1.0, 2.0, 3.0])

    with mock.patch(
        "fsec.singularity_subtraction.ccdss._fit_unit_quartic_exponential",
        side_effect=fake_fit,
    ), mock.patch.object(solver, "_quartic_xi", return_value=0.1):
        solver._prepare_ss(t2)

    committed_parameters = solver.ss_model_parameters.copy()
    committed_xi = solver.ss_xi.copy()
    assert len(fit_calls) == 6
    assert solver.ss_prepare_count == 1
    assert solver.ss_fit_count == 6
    assert solver.ss_sigmas is None
    assert committed_parameters.shape == (6, 4)
    np.testing.assert_array_equal(
        committed_parameters,
        np.broadcast_to([1.0, 1.0, 2.0, 3.0], (6, 4)),
    )

    with mock.patch(
        "fsec.singularity_subtraction.ccdss._fit_unit_quartic_exponential",
        side_effect=[
            np.asarray([1.0, 2.0, 3.0]),
            np.asarray([1.0, 2.0, 3.0]),
            ValueError("quartic fit failed"),
        ],
    ) as quartic_fit, mock.patch.object(
        solver, "_quartic_xi", return_value=0.2
    ), pytest.raises(ValueError, match="quartic fit failed"):
        solver._prepare_ss(t2)

    assert quartic_fit.call_count == 3
    assert solver.ss_prepare_count == 1
    assert solver.ss_fit_count == 6
    np.testing.assert_array_equal(solver.ss_model_parameters, committed_parameters)
    np.testing.assert_array_equal(solver.ss_xi, committed_xi)


def test_combined_quartic_preparation_is_scalar_and_atomic():
    """Fit one quartic signed curve and retain state after a failed update."""
    solver = aggregate_solver()
    solver.options = CCDSSOptions(
        auxfunc="QuarticExponential", structure_factor_mode="combined"
    )
    t2 = symmetric_t2(solver)
    fit_calls = []

    def fake_fit(_q, values, fit_with_coul=True):
        fit_calls.append(np.array(values, copy=True))
        return np.asarray([1.0, 2.0, 3.0])

    with mock.patch(
        "fsec.singularity_subtraction.ccdss._fit_unit_quartic_exponential",
        side_effect=fake_fit,
    ), mock.patch.object(solver, "_quartic_xi", return_value=0.1):
        solver._prepare_ss(t2)

    committed_parameters = solver.ss_model_parameters.copy()
    committed_xi = solver.ss_xi
    assert len(fit_calls) == 1
    assert solver.ss_prepare_count == 1
    assert solver.ss_fit_count == 1
    assert solver.ss_sigmas is None
    assert committed_parameters.shape == (4,)
    np.testing.assert_array_equal(committed_parameters, [1.0, 1.0, 2.0, 3.0])

    with mock.patch(
        "fsec.singularity_subtraction.ccdss._fit_unit_quartic_exponential",
        side_effect=ValueError("combined quartic fit failed"),
    ), pytest.raises(ValueError, match="combined quartic fit failed"):
        solver._prepare_ss(t2)

    assert solver.ss_prepare_count == 1
    assert solver.ss_fit_count == 1
    np.testing.assert_array_equal(solver.ss_model_parameters, committed_parameters)
    assert solver.ss_xi == committed_xi


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
    np.testing.assert_array_equal(
        solver.ss_model_parameters,
        np.broadcast_to([1.0, np.inf], (6, 2)),
    )
    sample.assert_not_called()
    fit.assert_not_called()


@pytest.mark.parametrize(
    "fixed_sigma, expected_xi, expected_coefficient",
    [(0.0, 0.0, 0.0), (np.inf, 0.75, -1.5)],
)
def test_combined_fixed_sigma_is_scalar_cached_and_madelung_consistent(
    fixed_sigma, expected_xi, expected_coefficient
):
    """Handle zero and infinite fixed widths without sampling or fitting."""
    solver = solver_shell(fixed_sigma=fixed_sigma)
    solver.options = CCDSSOptions(
        fixed_sigma=fixed_sigma, structure_factor_mode="combined"
    )
    t2 = np.ones((1, 1, 1, 1, 1, 1, 1))
    with mock.patch.object(
        solver, "_aggregate_channel_samples"
    ) as sample, mock.patch(
        "fsec.singularity_subtraction.ccdss._fit_unit_gaussian"
    ) as fit:
        solver._prepare_ss(t2)
        solver._prepare_ss(3 * t2)

    assert solver.ss_prepare_count == 1
    assert solver.ss_fit_count == 0
    assert np.asarray(solver.ss_xi).shape == ()
    assert solver.ss_xi == pytest.approx(expected_xi)
    assert np.asarray(solver.ss_sigmas).shape == ()
    assert solver.ss_sigmas == fixed_sigma
    assert solver.ss_model_parameters.shape == (2,)
    assert solver._ss_residual_coefficient(t2) == pytest.approx(
        expected_coefficient
    )
    if np.isinf(fixed_sigma):
        assert solver._ss_residual_coefficient(t2) == pytest.approx(
            2.0 * solver.madelung_constant
        )
    sample.assert_not_called()
    fit.assert_not_called()


def test_debug2_reports_raw_and_normalized_rows_on_every_fit():
    """Emit one parseable row per preparation, channel, and q sample."""
    solver = solver_shell(fixed_sigma=None)
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
    assert "auxfunc=Gauss" in rows[0]
    for field in (
        "raw_real=", "raw_imag=", "normalized_real=", "normalized_imag=",
        "sigma=", "xi=", "fit=", "residual_real=", "residual_imag=",
    ):
        assert field in rows[0]


def test_debug2_combined_reports_one_row_per_q_sample():
    """Emit exactly one combined diagnostic row for each sampled momentum."""
    solver = solver_shell(fixed_sigma=None)
    solver.options = CCDSSOptions(
        fixed_sigma=None, structure_factor_mode="combined"
    )
    solver._ss_q_vectors = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=float)
    solver.verbose = logger.DEBUG2
    t2 = np.ones((1, 1, 1, 1, 1, 1, 1))
    raw = np.asarray([2.0 + 0.5j, 1.0 - 0.25j])
    normalized = raw / raw[0]
    with mock.patch.object(
        solver, "_aggregate_channel_samples", return_value=(raw, normalized)
    ), mock.patch(
        "fsec.singularity_subtraction.ccdss._fit_unit_gaussian", return_value=2.0
    ), mock.patch.object(solver, "_gaussian_xi", return_value=0.25):
        solver._prepare_ss(t2)

    rows = [
        line for line in solver.stdout.getvalue().splitlines()
        if line.startswith("CCDSS_SF ")
    ]
    assert len(rows) == 2
    assert all("channel=combined" in row for row in rows)
    assert all("channel=L" not in row for row in rows)
    for field in (
        "raw_real=", "raw_imag=", "normalized_real=", "normalized_imag=",
        "sigma=", "xi=", "fit=", "residual_real=", "residual_imag=",
    ):
        assert field in rows[0]


def test_debug2_reports_quartic_parameters():
    """Emit model-specific quartic parameters in every diagnostic row."""
    solver = solver_shell(fixed_sigma=None)
    solver.options = CCDSSOptions(auxfunc="QuarticExponential")
    solver._ss_q_vectors = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=float)
    solver.verbose = logger.DEBUG2
    t2 = np.ones((1, 1, 1, 1, 1, 1, 1))
    raw = np.ones((6, 2), dtype=complex) * 2
    normalized = raw / 2
    with mock.patch.object(
        solver, "_aggregate_channel_samples", return_value=(raw, normalized)
    ), mock.patch(
        "fsec.singularity_subtraction.ccdss._fit_unit_quartic_exponential",
        return_value=np.asarray([1.0, 2.0, 3.0]),
    ), mock.patch.object(solver, "_quartic_xi", return_value=0.25):
        solver._prepare_ss(t2)

    rows = [
        line for line in solver.stdout.getvalue().splitlines()
        if line.startswith("CCDSS_SF ")
    ]
    assert len(rows) == 12
    assert "auxfunc=QuarticExponential" in rows[0]
    assert "alpha=1.0000000000000000e+00" in rows[0]
    assert "beta=2.0000000000000000e+00" in rows[0]
    assert "kappa=3.0000000000000000e+00" in rows[0]
    assert "sigma=" not in rows[0]


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

    solver.options = CCDSSOptions(
        fixed_sigma=0.5, structure_factor_mode="combined"
    )
    solver.ss_xi = 0.5
    t2new.fill(5.0)
    with mock.patch(
        "fsec.singularity_subtraction.ccdss.padding_k_idx",
        return_value=([np.asarray([0])], [np.asarray([0])]),
    ):
        result = solver._inject_ss_residual(t2new, t2, eris)
    assert result.item() == pytest.approx((5.0 * denominator - 3.0) / denominator)
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
    """Converge H2 with level-2 densities and both dynamic fit modes."""
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
        "combined": KRCCD_SS(mf, structure_factor_mode="combined"),
        "quartic": KRCCD_SS(mf, auxfunc="QuarticExponential"),
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
    combined = calculations["combined"]
    assert combined.converged and np.isfinite(results["combined"][0])
    assert combined.ss_prepare_count == combined.cycles
    assert combined.ss_fit_count == combined.ss_prepare_count
    assert np.asarray(combined.ss_xi).shape == ()
    assert np.asarray(combined.ss_sigmas).shape == ()
    assert np.isfinite(combined.ss_xi)
    assert np.isfinite(combined.ss_sigmas)
    assert combined.ss_model_parameters.shape == (2,)
    quartic = calculations["quartic"]
    assert quartic.converged and np.isfinite(results["quartic"][0])
    assert quartic.ss_prepare_count == quartic.cycles
    assert quartic.ss_fit_count == 6 * quartic.ss_prepare_count
    assert quartic.ss_sigmas is None
    assert quartic.ss_model_parameters.shape == (6, 4)
    assert np.all(np.isfinite(quartic.ss_model_parameters))
    assert np.all(quartic.ss_model_parameters[:, 1:] > 0)
    assert quartic.ss_xi.shape == (6,)
    assert np.all(np.isfinite(quartic.ss_xi))
