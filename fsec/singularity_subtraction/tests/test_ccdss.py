"""Tests for iterative CCD singularity subtraction."""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from pyscf.pbc.cc import kccsd_rhf
from pyscf.pbc import gto, scf, tools

from fsec.staggered_mesh.cc import KRCCD
from fsec.singularity_subtraction.ccdss import (
    CCDSSOptions,
    KRCCD_SS,
    _fit_unit_gaussian,
    _line_samples,
    _normalize_occupied_orbital_shift,
)


class FakeCell:
    vol = 8.0

    def reciprocal_vectors(self):
        return np.diag([2.0, 3.0, 4.0])


def solver_shell(nkpts=1, nocc=1, nvir=1):
    solver = object.__new__(KRCCD_SS)
    solver.options = CCDSSOptions(fixed_sigma=0.0)
    solver.kpts = np.zeros((nkpts, 3))
    solver.mo_occ = np.asarray(
        [np.r_[np.full(nocc, 2.0), np.zeros(nvir)] for _ in range(nkpts)]
    )
    solver._nocc = nocc
    solver._nmo = nocc + nvir
    solver.frozen = None
    solver.level_shift = 0.0
    solver.madelung_constant = -0.75
    solver.madelung_orbital = True
    solver.madelung_eri = False
    solver.keep_exxdiv = False
    solver.occupied_orbital_shift = None
    solver.khelper = SimpleNamespace(
        kconserv=np.zeros((nkpts, nkpts, nkpts), dtype=int)
    )
    solver.ss_xi = None
    solver.ss_sigmas = None
    solver.ss_fit_count = 0
    solver.ss_prepare_count = 0
    solver.last_ss_residual_norm = 0.0
    return solver


def test_options_reject_invalid_inputs():
    for value in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="line_points"):
            CCDSSOptions(line_points=value)
    with pytest.raises(ValueError, match="grid_level"):
        CCDSSOptions(pair_density_becke_grid_level=-1)
    for value in (-1.0, np.nan):
        with pytest.raises(ValueError, match="fixed_sigma"):
            CCDSSOptions(fixed_sigma=value)
    with pytest.raises(ValueError, match="amplitude_fit_tol"):
        CCDSSOptions(amplitude_fit_tol=0.0)


def test_options_object_cannot_be_ambiguously_mixed_with_overrides():
    from fsec.singularity_subtraction.ccdss import _merge_options

    options = CCDSSOptions(line_points=5)
    assert _merge_options(
        options,
        use_constraint_2=True,
        line_points=3,
        pair_density_becke_grid_level=0,
        fit_with_coul=True,
        fixed_sigma=None,
        amplitude_fit_tol=1e-12,
    ) is options
    with pytest.raises(ValueError, match="line_points"):
        _merge_options(
            options,
            use_constraint_2=True,
            line_points=4,
            pair_density_becke_grid_level=0,
            fit_with_coul=True,
            fixed_sigma=None,
            amplitude_fit_tol=1e-12,
        )


def test_occupied_orbital_shift_validation_broadcasts_and_copies():
    scalar = _normalize_occupied_orbital_shift(-0.25, nkpts=2, nocc=3)
    np.testing.assert_array_equal(scalar, np.full((2, 3), -0.25))

    supplied = np.arange(6.0).reshape(2, 3)
    normalized = _normalize_occupied_orbital_shift(supplied, nkpts=2, nocc=3)
    supplied[0, 0] = 99.0
    assert normalized[0, 0] == 0.0

    for invalid in (
        np.zeros(3),
        np.zeros((3, 2)),
        np.asarray([[1.0, np.inf], [2.0, 3.0]]),
        1.0 + 0.0j,
        True,
        "0.1",
    ):
        with pytest.raises(ValueError, match="occupied_orbital_shift"):
            _normalize_occupied_orbital_shift(invalid, nkpts=2, nocc=2)


def test_custom_occupied_shift_replaces_exxdiv_and_respects_padding():
    def run(keep_exxdiv):
        solver = solver_shell(nkpts=2, nocc=2, nvir=1)
        # A custom shift wins even if either inherited correction switch is
        # subsequently enabled by user code.
        solver.madelung_orbital = True
        solver.keep_exxdiv = keep_exxdiv
        solver.occupied_orbital_shift = np.asarray(
            [[0.1, -0.2], [0.3, 8.0]]
        )
        observed_correction_flags = []

        def make_eris(_mo_coeff=None):
            observed_correction_flags.append(
                (solver.keep_exxdiv, solver.madelung_orbital)
            )
            return SimpleNamespace(
                fock=np.asarray(
                    [
                        [[1.0, 0.4, 0.0], [0.4, 2.0, 0.5], [0.0, 0.5, 6.0]],
                        [[3.0, 0.6, 0.0], [0.6, 4.0, 0.7], [0.0, 0.7, 7.0]],
                    ]
                ),
                mo_energy=[np.zeros(3), np.zeros(3)],
            )

        with mock.patch.object(
            kccsd_rhf.RCCSD, "ao2mo", side_effect=make_eris
        ), mock.patch(
            "fsec.singularity_subtraction.ccdss.padding_k_idx",
            return_value=(
                [np.asarray([0, 1]), np.asarray([0])],
                [np.asarray([0]), np.asarray([0, 1])],
            ),
        ), mock.patch("fsec.staggered_mesh.cc.krccd.logger.warn") as warn:
            eris = solver.ao2mo()

        assert observed_correction_flags == [(False, False)]
        assert solver.keep_exxdiv is keep_exxdiv
        assert solver.madelung_orbital is True
        warn.assert_not_called()
        np.testing.assert_allclose(
            eris.fock,
            [
                [[1.1, 0.4, 0.0], [0.4, 1.8, 0.5], [0.0, 0.5, 6.0]],
                [[3.3, 0.6, 0.0], [0.6, 4.0, 0.7], [0.0, 0.7, 7.0]],
            ],
        )
        np.testing.assert_allclose(
            eris.mo_energy,
            [[1.1, 1.8, 6.0], [3.3, 4.0, 7.0]],
        )
        return eris

    without_exxdiv = run(False)
    with_exxdiv = run(True)
    np.testing.assert_allclose(without_exxdiv.fock, with_exxdiv.fock)
    np.testing.assert_allclose(without_exxdiv.mo_energy, with_exxdiv.mo_energy)


def test_missing_occupied_shift_preserves_default_ao2mo_path():
    solver = solver_shell()
    solver.keep_exxdiv = True
    expected = SimpleNamespace()
    with mock.patch.object(KRCCD, "ao2mo", return_value=expected) as ao2mo:
        assert solver.ao2mo() is expected
    ao2mo.assert_called_once_with(None)
    assert solver.keep_exxdiv is True


def test_custom_occupied_shift_restores_flags_when_ao2mo_fails():
    solver = solver_shell()
    solver.occupied_orbital_shift = np.zeros((1, 1))
    solver.keep_exxdiv = True
    solver.madelung_orbital = True
    with mock.patch.object(
        kccsd_rhf.RCCSD, "ao2mo", side_effect=RuntimeError("failed ERI build")
    ), pytest.raises(RuntimeError, match="failed ERI build"):
        solver.ao2mo()
    assert solver.keep_exxdiv is True
    assert solver.madelung_orbital is True


def test_line_sampling_uses_positive_mesh_steps():
    with mock.patch(
        "fsec.singularity_subtraction.ccdss.tools.get_monkhorst_pack_size",
        return_value=np.asarray([1, 2, 4]),
    ):
        points, metadata, nks = _line_samples(
            FakeCell(), np.zeros((8, 3)), line_points=2
        )
    np.testing.assert_array_equal(nks, [1, 2, 4])
    np.testing.assert_allclose(
        points,
        [
            [0, 0, 0],
            [2, 0, 0],
            [4, 0, 0],
            [0, 1.5, 0],
            [0, 3, 0],
            [0, 0, 1],
            [0, 0, 2],
        ],
    )
    assert len(metadata) == 3


def test_k_shift_maps_treat_scaled_k_points_as_periodic():
    class IdentityScaledCell:
        @staticmethod
        def get_scaled_kpts(kpts):
            return np.asarray(kpts)

    solver = solver_shell()
    solver._scf = SimpleNamespace(cell=IdentityScaledCell())
    # The tiny negative component is numerically equivalent to zero modulo a
    # reciprocal lattice vector.  A non-periodic KD-tree sees it at 1 instead.
    q_vectors = np.asarray([[0.0, 0.0, -1e-18]])
    plus, minus = solver._build_k_shift_maps(q_vectors)
    np.testing.assert_array_equal(plus, [[0]])
    np.testing.assert_array_equal(minus, [[0]])


@pytest.mark.parametrize("weighted", [False, True])
def test_unit_gaussian_fit_recovers_sigma(weighted):
    q = np.asarray(
        [[0, 0, 0], [0.2, 0, 0], [0, 0.35, 0], [0, 0, 0.5]]
    )
    sigma = 0.63
    values = np.exp(-np.einsum("qi,qi->q", q, q) / (2 * sigma**2))
    fitted = _fit_unit_gaussian(q, values, fit_with_coul=weighted)
    assert fitted == pytest.approx(sigma, rel=1e-7)


def test_gaussian_fit_rejects_nonfinite_data():
    with pytest.raises(ValueError, match="finite"):
        _fit_unit_gaussian(np.asarray([[0, 0, 0], [1, 0, 0]]), [1, np.nan])


def test_gaussian_fit_rejects_nonfinite_optimizer_result():
    result = SimpleNamespace(success=True, x=np.asarray([np.nan]))
    with mock.patch(
        "fsec.singularity_subtraction.ccdss.least_squares", return_value=result
    ), pytest.raises(ValueError, match="non-finite sigma"):
        _fit_unit_gaussian(np.asarray([[0, 0, 0], [1, 0, 0]]), [1, 0.5])


def test_fixed_sigma_limits_and_six_channel_signs():
    solver = solver_shell()
    assert solver._gaussian_xi(0.0) == 0.0
    assert solver._gaussian_xi(np.inf) == pytest.approx(0.75)
    solver.ss_xi = np.asarray([1, 2, 3, 4, 5, 6], dtype=float).reshape(
        (6, 1, 1, 1, 1, 1, 1, 1)
    )
    t2 = np.ones((1, 1, 1, 1, 1, 1, 1))
    coefficient = solver._ss_residual_coefficient(t2)
    assert coefficient.item() == 1 + 2 - 3 - 4 - 5 - 6


def test_infinite_sigma_is_exact_madelung_residual():
    solver = solver_shell()
    solver.options = CCDSSOptions(fixed_sigma=np.inf)
    t2 = np.full((1, 1, 1, 1, 1, 1, 1), 2.0)
    coefficient = solver._ss_residual_coefficient(t2)
    assert coefficient.item() == pytest.approx(2.0 * solver.madelung_constant)


def test_residual_is_combined_with_numerator_before_division():
    solver = solver_shell()
    t2 = np.full((1, 1, 1, 1, 1, 1, 1), 3.0)
    t2new = np.full_like(t2, 5.0)
    eris = SimpleNamespace(mo_energy=[np.asarray([1.0, 4.0])])
    solver.ss_xi = np.asarray([0.5, 0.5, 0, 0, 0, 0]).reshape(
        (6, 1, 1, 1, 1, 1, 1, 1)
    )
    with mock.patch(
        "fsec.singularity_subtraction.ccdss.padding_k_idx",
        return_value=([np.asarray([0])], [np.asarray([0])]),
    ):
        solver._inject_ss_residual(t2new, t2, eris)
    # denominator = (1-4) + (1-4) = -6; (5*-6 + 1*3) / -6
    assert t2new.item() == pytest.approx(4.5)
    assert solver.last_ss_residual_norm == pytest.approx(3.0)


def test_residual_denominators_preserve_padded_band_masks():
    solver = solver_shell(nocc=2, nvir=2)
    t2 = np.ones((1, 1, 1, 2, 2, 2, 2))
    t2new = np.zeros_like(t2)
    eris = SimpleNamespace(mo_energy=[np.asarray([1.0, 2.0, 4.0, 5.0])])
    solver.ss_xi = np.asarray([1, 0, 0, 0, 0, 0], dtype=float).reshape(
        (6, 1, 1, 1, 1, 1, 1, 1)
    ) * np.ones((6,) + t2.shape)
    with mock.patch(
        "fsec.singularity_subtraction.ccdss.padding_k_idx",
        return_value=([np.asarray([0])], [np.asarray([1])]),
    ):
        solver._inject_ss_residual(t2new, t2, eris)
    assert t2new[0, 0, 0, 0, 0, 1, 1] == pytest.approx(1.0 / -8.0)
    assert abs(t2new[0, 0, 0, 1, 0, 1, 1]) < 1e-10


def test_constrained_corrections_are_reused_and_unconstrained_are_refit():
    t2 = np.ones((1, 1, 1, 1, 1, 1, 1))
    constrained = solver_shell()
    constrained.options = CCDSSOptions(use_constraint_2=True, fixed_sigma=0.0)
    constrained._ss_residual_coefficient(t2)
    constrained._ss_residual_coefficient(2 * t2)
    assert constrained.ss_prepare_count == 1

    unconstrained = solver_shell()
    unconstrained.options = CCDSSOptions(use_constraint_2=False, fixed_sigma=0.0)
    unconstrained._ss_residual_coefficient(t2)
    unconstrained._ss_residual_coefficient(2 * t2)
    assert unconstrained.ss_prepare_count == 2


def test_unconstrained_gaussians_are_actually_refitted_after_t2_changes():
    solver = solver_shell()
    solver.options = CCDSSOptions(use_constraint_2=False)
    t2 = np.ones((1, 1, 1, 1, 1, 1, 1))
    solver._ss_q_vectors = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=float)
    samples = np.ones((6, 2) + t2.shape, dtype=complex)
    with mock.patch.object(
        solver, "_channel_samples", return_value=samples
    ), mock.patch(
        "fsec.singularity_subtraction.ccdss._fit_unit_gaussian",
        return_value=0.5,
    ) as fit, mock.patch.object(solver, "_gaussian_xi", return_value=0.2):
        solver._ss_residual_coefficient(t2)
        solver._ss_residual_coefficient(2 * t2)
    assert fit.call_count == 12
    assert solver.ss_fit_count == 12


def test_each_unconstrained_channel_uses_the_documented_momentum_mapping():
    solver = solver_shell(nkpts=2, nocc=2, nvir=2)
    solver._ss_plus = np.asarray([[0, 1], [1, 0]])
    solver._ss_minus = np.asarray([[0, 1], [1, 0]])
    solver.khelper.kconserv = np.fromfunction(
        lambda ki, ka, kj: (ki - ka + kj) % 2, (2, 2, 2), dtype=int
    ).astype(int)
    t2 = np.arange(2 * 2 * 2 * 2 * 2 * 2 * 2).reshape(2, 2, 2, 2, 2, 2, 2)
    channels = solver._channel_amplitudes(t2, 1, ki=0, kj=1, ka=0)
    np.testing.assert_array_equal(channels[0], t2[1, 0, 0])
    np.testing.assert_array_equal(channels[1], t2[0, 1, 1])
    np.testing.assert_array_equal(channels[2], t2[1, 1, 1])
    np.testing.assert_array_equal(channels[3], t2[0, 0, 0].transpose(1, 0, 3, 2))
    np.testing.assert_array_equal(channels[4], t2[0, 0, 1].transpose(1, 0, 3, 2))
    np.testing.assert_array_equal(channels[5], t2[1, 1, 0])


def test_update_amps_keeps_t1_zero_and_validates_t2_shape():
    solver = solver_shell()
    t1 = np.ones((1, 1, 1))
    t2 = np.ones((1, 1, 1, 1, 1, 1, 1))
    with pytest.raises(ValueError, match="t2 must have shape"):
        solver.update_amps(t1, t2[..., 0], SimpleNamespace())

    with mock.patch.object(
        kccsd_rhf.RCCSD,
        "update_amps",
        return_value=(np.ones_like(t1), np.full_like(t2, 2.0)),
    ), mock.patch.object(KRCCD_SS, "_inject_ss_residual", lambda self, out, *_: out):
        out_t1, out_t2 = solver.update_amps(t1, t2, SimpleNamespace())
    np.testing.assert_array_equal(out_t1, np.zeros_like(t1))
    np.testing.assert_array_equal(out_t2, np.full_like(t2, 2.0))


@pytest.mark.slow
def test_h2_1x1x2_limits_and_default_fit():
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
        "custom_madelung": KRCCD_SS(
            mf,
            fixed_sigma=0.0,
            occupied_orbital_shift=-tools.madelung(cell, kpts),
        ),
        "near_zero": KRCCD_SS(mf, fixed_sigma=1e-10),
        "madelung": KRCCD(mf, madelung_orbital=True, madelung_eri=True),
        "sigma_0.5": KRCCD_SS(mf, fixed_sigma=0.5),
        "sigma_1.0": KRCCD_SS(mf, fixed_sigma=1.0),
        "sigma_2.0": KRCCD_SS(mf, fixed_sigma=2.0),
        "sigma_4.0": KRCCD_SS(mf, fixed_sigma=4.0),
        "large": KRCCD_SS(mf, fixed_sigma=1e6),
        "infinite": KRCCD_SS(mf, fixed_sigma=np.inf),
        "fitted": KRCCD_SS(mf),
        "unconstrained": KRCCD_SS(mf, use_constraint_2=False),
    }
    energies = {}
    for name, cc in calculations.items():
        cc.keep_exxdiv = False
        cc.conv_tol = 1e-7
        cc.max_cycle = 20
        energy, _, _ = cc.kernel()
        energies[name] = energy
        assert cc.converged
        assert np.isfinite(energy)

    assert energies["zero"] == pytest.approx(energies["orbital"], abs=1e-11)
    assert energies["custom_madelung"] == pytest.approx(
        energies["zero"], abs=1e-11
    )
    assert energies["near_zero"] == pytest.approx(energies["orbital"], abs=1e-9)
    assert energies["large"] == pytest.approx(energies["madelung"], abs=1e-11)
    assert energies["infinite"] == pytest.approx(energies["madelung"], abs=1e-11)

    # These widths remain below the reciprocal-shell shortcut.  They verify
    # the finite Gaussian sequence itself, rather than the explicit infinity
    # limit, approaches converged CCD with both Madelung corrections.
    finite_sigma_errors = [
        abs(energies[f"sigma_{sigma:.1f}"] - energies["madelung"])
        for sigma in (0.5, 1.0, 2.0, 4.0)
    ]
    assert all(
        later < earlier
        for earlier, later in zip(finite_sigma_errors, finite_sigma_errors[1:])
    )
    assert finite_sigma_errors[-1] < 3e-5

    fitted = calculations["fitted"]
    assert fitted.ss_prepare_count == 1
    assert fitted.ss_fit_count > 0
    np.testing.assert_array_equal(fitted._ss_pair_factors[0], 1.0)
    unconstrained = calculations["unconstrained"]
    assert unconstrained.ss_prepare_count > 1
    assert np.all(np.isfinite(unconstrained.ss_sigmas))
