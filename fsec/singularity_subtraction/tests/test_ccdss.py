"""Tests for iterative CCD singularity subtraction."""

from types import SimpleNamespace
from unittest import mock
import io

import numpy as np
import pytest

from pyscf.pbc.cc import kccsd_rhf
from pyscf.pbc import gto, scf, tools
from pyscf.lib import logger

from fsec.staggered_mesh.cc import KRCCD
from fsec.singularity_subtraction.ccdss import (
    CCDSSOptions,
    KRCCD_SS,
    _fit_unit_gaussian,
    _line_samples,
    _normalize_occupied_orbital_shift,
)


class FakeCell:
    """Minimal orthorhombic cell used to test reciprocal-line sampling."""

    vol = 8.0

    def reciprocal_vectors(self):
        """Return distinct reciprocal lengths for the three directions."""
        return np.diag([2.0, 3.0, 4.0])


def solver_shell(nkpts=1, nocc=1, nvir=1):
    """Build a lightweight solver without running the PySCF constructor."""
    solver = object.__new__(KRCCD_SS)
    solver.options = CCDSSOptions(fixed_sigma=0.0)
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
        kconserv=np.zeros((nkpts, nkpts, nkpts), dtype=int)
    )
    solver.ss_xi = None
    solver.ss_sigmas = None
    solver.ss_fit_count = 0
    solver.ss_prepare_count = 0
    solver.last_ss_residual_norm = 0.0
    solver._ss_active_masks = None
    solver._ss_active_count = None
    solver.verbose = logger.NOTE
    solver.stdout = io.StringIO()
    return solver


def _aggregate_solver_for_tests(nkpts=2, nocc=2, nvir=2):
    """Add a two-point q grid and deterministic pair factors to a solver."""
    solver = solver_shell(nkpts=nkpts, nocc=nocc, nvir=nvir)
    solver.options = CCDSSOptions(fixed_sigma=None)
    solver._ss_q_vectors = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    solver._ss_plus = np.asarray([[0, 1], [1, 0]])
    solver._ss_minus = np.asarray([[0, 1], [1, 0]])
    solver.khelper.kconserv = np.fromfunction(
        lambda ki, ka, kj: (ki - ka + kj) % 2,
        (nkpts, nkpts, nkpts),
        dtype=int,
    ).astype(int)
    rng = np.random.default_rng(29)
    factors = rng.normal(size=(2, nkpts, nocc + nvir)) + 1j * rng.normal(
        size=(2, nkpts, nocc + nvir)
    )
    factors[0] = 1.0
    solver._ss_pair_factors = factors
    return solver


def _relaxed_aggregate_solver_for_tests(nkpts=2, nocc=2, nvir=2):
    """Add full complex transition densities to an aggregate test solver."""
    solver = _aggregate_solver_for_tests(nkpts=nkpts, nocc=nocc, nvir=nvir)
    solver.options = CCDSSOptions(
        use_constraint_1=False,
        use_constraint_2=True,
        fixed_sigma=None,
    )
    rng = np.random.default_rng(37)
    pair_densities = rng.normal(
        size=(2, nkpts, nocc + nvir, nocc + nvir)
    ) + 1j * rng.normal(size=(2, nkpts, nocc + nvir, nocc + nvir))
    pair_densities[0] = np.eye(nocc + nvir, dtype=complex)
    solver._ss_pair_densities = pair_densities
    solver._ss_pair_factors = None
    return solver, pair_densities


def _explicit_diagonal_density_raw(solver, q_index):
    """Evaluate diagonal density-only aggregates with explicit loops."""
    factors = solver._ss_pair_factors[q_index]
    occupied, virtual = solver._build_active_masks()
    plus = solver._ss_plus[q_index]
    minus = solver._ss_minus[q_index]
    nocc = solver.nocc
    raw = np.zeros(6, dtype=complex)
    for ki in range(solver.nkpts):
        for kj in range(solver.nkpts):
            for ka in range(solver.nkpts):
                kb = solver.khelper.kconserv[ki, ka, kj]
                for i in range(nocc):
                    for j in range(nocc):
                        for a in range(solver.nmo - nocc):
                            for b in range(solver.nmo - nocc):
                                external = (
                                    occupied[ki, i]
                                    and occupied[kj, j]
                                    and virtual[ka, a]
                                    and virtual[kb, b]
                                )
                                if not external:
                                    continue
                                if occupied[minus[ki], i] and occupied[ki, i]:
                                    if occupied[kj, j] and occupied[plus[kj], j]:
                                        raw[0] += (
                                            factors[minus[ki], i]
                                            * factors[kj, j].conjugate()
                                        )
                                if virtual[ka, a] and virtual[plus[ka], a]:
                                    if virtual[minus[kb], b] and virtual[kb, b]:
                                        raw[1] += (
                                            factors[ka, nocc + a]
                                            * factors[minus[kb], nocc + b].conjugate()
                                        )
                                if virtual[ka, a] and virtual[plus[ka], a]:
                                    if occupied[ki, i] and occupied[plus[ki], i]:
                                        raw[2] += (
                                            factors[ka, nocc + a]
                                            * factors[ki, i].conjugate()
                                        )
                                if virtual[kb, b] and virtual[plus[kb], b]:
                                    if occupied[kj, j] and occupied[plus[kj], j]:
                                        raw[3] += (
                                            factors[kb, nocc + b]
                                            * factors[kj, j].conjugate()
                                        )
                                        raw[4] += (
                                            factors[ka, nocc + a]
                                            * factors[kj, j].conjugate()
                                        )
                                if virtual[kb, b] and virtual[plus[kb], b]:
                                    if occupied[ki, i] and occupied[plus[ki], i]:
                                        raw[5] += (
                                            factors[kb, nocc + b]
                                            * factors[ki, i].conjugate()
                                        )
    return raw


def _explicit_full_density_raw(solver, pair_densities, q_index):
    """Evaluate full-density aggregates with explicit nested-loop sums."""
    occupied, virtual = solver._build_active_masks()
    plus = solver._ss_plus[q_index]
    minus = solver._ss_minus[q_index]
    nocc = solver.nocc
    nvir = solver.nmo - nocc
    density = pair_densities[q_index]
    raw = np.zeros(6, dtype=complex)
    for ki in range(solver.nkpts):
        for kj in range(solver.nkpts):
            for ka in range(solver.nkpts):
                kb = solver.khelper.kconserv[ki, ka, kj]
                for i in range(nocc):
                    for j in range(nocc):
                        for a in range(nvir):
                            for b in range(nvir):
                                if not (
                                    occupied[ki, i]
                                    and occupied[kj, j]
                                    and virtual[ka, a]
                                    and virtual[kb, b]
                                ):
                                    continue
                                for k in range(nocc):
                                    for l in range(nocc):
                                        if (
                                            occupied[minus[ki], k]
                                            and occupied[plus[kj], l]
                                        ):
                                            raw[0] += (
                                                density[minus[ki], k, i]
                                                * density[kj, j, l].conjugate()
                                            )
                                for c in range(nvir):
                                    for d in range(nvir):
                                        if (
                                            virtual[plus[ka], c]
                                            and virtual[minus[kb], d]
                                        ):
                                            raw[1] += (
                                                density[ka, nocc + a, nocc + c]
                                                * density[
                                                    minus[kb], nocc + d, nocc + b
                                                ].conjugate()
                                            )
                                for c in range(nvir):
                                    for k in range(nocc):
                                        if (
                                            virtual[plus[ka], c]
                                            and occupied[plus[ki], k]
                                        ):
                                            raw[2] += (
                                                density[ka, nocc + a, nocc + c]
                                                * density[ki, i, k].conjugate()
                                            )
                                        if (
                                            virtual[plus[kb], c]
                                            and occupied[plus[kj], k]
                                        ):
                                            raw[3] += (
                                                density[kb, nocc + b, nocc + c]
                                                * density[kj, j, k].conjugate()
                                            )
                                        if (
                                            virtual[plus[ka], c]
                                            and occupied[plus[kj], k]
                                        ):
                                            raw[4] += (
                                                density[ka, nocc + a, nocc + c]
                                                * density[kj, j, k].conjugate()
                                            )
                                        if (
                                            virtual[plus[kb], c]
                                            and occupied[plus[ki], k]
                                        ):
                                            raw[5] += (
                                                density[kb, nocc + b, nocc + c]
                                                * density[ki, i, k].conjugate()
                                            )
    return raw


def test_options_reject_invalid_inputs():
    """Reject controls that would make sampling or fitting ill-defined."""
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
    """Accept an options object only when keyword overrides stay at defaults."""
    from fsec.singularity_subtraction.ccdss import _merge_options

    options = CCDSSOptions(line_points=5)
    assert _merge_options(
        options,
        use_constraint_1=True,
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
            use_constraint_1=True,
            use_constraint_2=True,
            line_points=4,
            pair_density_becke_grid_level=0,
            fit_with_coul=True,
            fixed_sigma=None,
            amplitude_fit_tol=1e-12,
        )


def test_constraint_1_relaxation_is_accepted_but_constraint_2_is_not():
    """Accept full density contractions and reject unsupported momentum shifts."""
    assert CCDSSOptions().use_constraint_1 is True
    assert CCDSSOptions().use_constraint_2 is True
    options = CCDSSOptions(use_constraint_1=False)
    assert options.use_constraint_1 is False

    def fake_init(self, *args, **kwargs):
        """Provide the parent attributes needed by the constructor test."""
        self.kpts = np.zeros((1, 3))
        self.mo_occ = np.asarray([[2.0, 0.0]])
        self._nocc = 1
        self._nmo = 2

    with mock.patch.object(KRCCD, "__init__", fake_init):
        solver = KRCCD_SS(None, use_constraint_1=False)
        assert solver.use_constraint_1 is False
        with pytest.raises(NotImplementedError, match="use_constraint_2"):
            KRCCD_SS(None, use_constraint_2=False)


def test_occupied_orbital_shift_validation_broadcasts_and_copies():
    """Normalize valid occupied shifts and reject malformed or unsafe values."""
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
    """Apply a custom shift only to active occupied Fock diagonal entries."""

    def run(keep_exxdiv):
        """Build ERIs with one inherited exchange-divergence setting."""
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
            """Record internal correction flags and return a small ERI shell."""
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
    """Delegate unchanged to KRCCD when no custom orbital shift is supplied."""
    solver = solver_shell()
    solver.keep_exxdiv = True
    expected = SimpleNamespace()
    with mock.patch.object(KRCCD, "ao2mo", return_value=expected) as ao2mo:
        assert solver.ao2mo() is expected
    ao2mo.assert_called_once_with(None)
    assert solver.keep_exxdiv is True


def test_custom_occupied_shift_restores_flags_when_ao2mo_fails():
    """Restore inherited correction flags if the custom-shift ERI build fails."""
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
    """Sample the origin and positive multiples of each reciprocal mesh step."""
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
    """Map roundoff near a Brillouin-zone boundary to the periodic image."""

    class IdentityScaledCell:
        """Treat Cartesian test points as already scaled k points."""

        @staticmethod
        def get_scaled_kpts(kpts):
            """Return the supplied points without a coordinate transform."""
            return np.asarray(kpts)

    solver = solver_shell()
    solver._scf = SimpleNamespace(cell=IdentityScaledCell())
    # The tiny negative component is numerically equivalent to zero modulo a
    # reciprocal lattice vector.  A non-periodic KD-tree sees it at 1 instead.
    q_vectors = np.asarray([[0.0, 0.0, -1e-18]])
    plus, minus = solver._build_k_shift_maps(q_vectors)
    np.testing.assert_array_equal(plus, [[0]])
    np.testing.assert_array_equal(minus, [[0]])


def test_pair_density_origin_is_exact_identity_and_complex_values_are_finite():
    """Set F(0) exactly to identity and keep padded complex factors finite."""
    solver = solver_shell(nkpts=2, nocc=2, nvir=2)
    solver._ss_q_vectors = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    solver._ss_plus = np.asarray([[0, 1], [1, 0]])
    solver._ss_minus = np.asarray([[0, 1], [1, 0]])
    solver._scf = SimpleNamespace(cell=SimpleNamespace())
    solver.kpts = np.zeros((2, 3))

    coords = np.asarray([[0.37, 0.0, 0.0]])
    weights = np.asarray([1.0])
    rng = np.random.default_rng(12)
    u = rng.normal(size=(2, 4, 1)) + 1j * rng.normal(size=(2, 4, 1))
    u[:, -1] = 0.0
    norms = np.ones((2, 4))
    norms[:, -1] = 0.0
    wrapped_kpts = np.zeros((2, 3))
    solver._ss_pair_orbitals = (coords, weights, u, norms, wrapped_kpts)

    densities = solver._build_pair_densities()
    np.testing.assert_array_equal(densities[0], np.broadcast_to(np.eye(4), (2, 4, 4)))
    assert np.iscomplexobj(densities)
    assert np.all(np.isfinite(densities))
    np.testing.assert_array_equal(densities[1, :, -1, -1], 1.0)

    solver.options = CCDSSOptions(
        use_constraint_1=False, use_constraint_2=True, fixed_sigma=None
    )
    solver._ss_pair_densities = densities
    raw = solver._contract_aggregate_structure_factors(
        np.ones((2, 2, 2, 2, 2, 2, 2), dtype=complex)
    )
    assert np.all(np.isfinite(raw))


@pytest.mark.parametrize("weighted", [False, True])
def test_unit_gaussian_fit_recovers_sigma(weighted):
    """Recover an exact Gaussian width with and without Coulomb weighting."""
    q = np.asarray(
        [[0, 0, 0], [0.2, 0, 0], [0, 0.35, 0], [0, 0, 0.5]]
    )
    sigma = 0.63
    values = np.exp(-np.einsum("qi,qi->q", q, q) / (2 * sigma**2))
    fitted = _fit_unit_gaussian(q, values, fit_with_coul=weighted)
    assert fitted == pytest.approx(sigma, rel=1e-7)


def test_gaussian_fit_rejects_nonfinite_data():
    """Reject nonfinite structure-factor samples before optimization."""
    with pytest.raises(ValueError, match="finite"):
        _fit_unit_gaussian(np.asarray([[0, 0, 0], [1, 0, 0]]), [1, np.nan])


def test_gaussian_fit_rejects_nonfinite_optimizer_result():
    """Reject an optimizer result whose fitted width is nonfinite."""
    result = SimpleNamespace(success=True, x=np.asarray([np.nan]))
    with mock.patch(
        "fsec.singularity_subtraction.ccdss.least_squares", return_value=result
    ), pytest.raises(ValueError, match="non-finite sigma"):
        _fit_unit_gaussian(np.asarray([[0, 0, 0], [1, 0, 0]]), [1, 0.5])


def test_fixed_sigma_limits_and_six_channel_signs():
    """Check zero/infinite widths and the +,+,-,-,-,- channel combination."""
    solver = solver_shell()
    assert solver._gaussian_xi(0.0) == 0.0
    assert solver._gaussian_xi(np.inf) == pytest.approx(0.75)
    solver.ss_xi = np.asarray([1, 2, 3, 4, 5, 6], dtype=float)
    t2 = np.ones((1, 1, 1, 1, 1, 1, 1))
    coefficient = solver._ss_residual_coefficient(t2)
    assert coefficient.item() == 1 + 2 - 3 - 4 - 5 - 6


def test_infinite_sigma_is_exact_madelung_residual():
    """Reduce the infinite-width six-channel coefficient to 2*xi_Madelung."""
    solver = solver_shell()
    solver.options = CCDSSOptions(fixed_sigma=np.inf)
    t2 = np.full((1, 1, 1, 1, 1, 1, 1), 2.0)
    coefficient = solver._ss_residual_coefficient(t2)
    assert coefficient.item() == pytest.approx(2.0 * solver.madelung_constant)


def test_residual_is_combined_with_numerator_before_division():
    """Add the fitted correction to the CCD numerator before division."""
    solver = solver_shell()
    t2 = np.full((1, 1, 1, 1, 1, 1, 1), 3.0)
    t2new = np.full_like(t2, 5.0)
    eris = SimpleNamespace(mo_energy=[np.asarray([1.0, 4.0])])
    solver.ss_xi = np.asarray([0.5, 0.5, 0, 0, 0, 0])
    with mock.patch(
        "fsec.singularity_subtraction.ccdss.padding_k_idx",
        return_value=([np.asarray([0])], [np.asarray([0])]),
    ):
        solver._inject_ss_residual(t2new, t2, eris)
    # denominator = (1-4) + (1-4) = -6; (5*-6 + 1*3) / -6
    assert t2new.item() == pytest.approx(4.5)
    assert solver.last_ss_residual_norm == pytest.approx(3.0)


def test_residual_denominators_preserve_padded_band_masks():
    """Update active amplitudes without reviving padded orbital entries."""
    solver = solver_shell(nocc=2, nvir=2)
    t2 = np.ones((1, 1, 1, 2, 2, 2, 2))
    t2new = np.zeros_like(t2)
    eris = SimpleNamespace(mo_energy=[np.asarray([1.0, 2.0, 4.0, 5.0])])
    solver.ss_xi = np.asarray([1, 0, 0, 0, 0, 0], dtype=float)
    with mock.patch(
        "fsec.singularity_subtraction.ccdss.padding_k_idx",
        return_value=([np.asarray([0])], [np.asarray([1])]),
    ):
        solver._inject_ss_residual(t2new, t2, eris)
    assert t2new[0, 0, 0, 0, 0, 1, 1] == pytest.approx(1.0 / -8.0)
    assert abs(t2new[0, 0, 0, 1, 0, 1, 1]) < 1e-10


@pytest.mark.parametrize("use_constraint_1", [True, False])
def test_fixed_sigma_is_cached_once_without_sampling_or_fitting(use_constraint_1):
    """Prepare a fixed-width correction once without curves, fits, or logs."""
    t2 = np.ones((1, 1, 1, 1, 1, 1, 1))
    solver = solver_shell()
    solver.options = CCDSSOptions(
        use_constraint_1=use_constraint_1, fixed_sigma=0.0
    )
    solver.verbose = logger.DEBUG2
    with mock.patch.object(
        solver, "_aggregate_channel_samples"
    ) as samples, mock.patch.object(
        solver, "_gaussian_xi", return_value=0.25
    ) as xi, mock.patch(
        "fsec.singularity_subtraction.ccdss._fit_unit_gaussian"
    ) as fit:
        solver._ss_residual_coefficient(t2)
        solver._ss_residual_coefficient(2 * t2)
    assert solver.ss_prepare_count == 1
    assert solver.ss_fit_count == 0
    assert solver.ss_sigmas.shape == (6,)
    assert solver.ss_xi.shape == (6,)
    samples.assert_not_called()
    fit.assert_not_called()
    xi.assert_called_once_with(0.0)
    assert "CCDSS_SF_NORM " not in solver.stdout.getvalue()


def test_xi_preparation_reports_cpu_and_wall_time_at_info():
    """Report preparation CPU and wall times at PySCF's INFO level."""
    solver = solver_shell()
    solver.verbose = logger.INFO
    t2 = np.ones((1, 1, 1, 1, 1, 1, 1))
    with mock.patch(
        "fsec.singularity_subtraction.ccdss.logger.process_clock",
        side_effect=[10.0, 12.0],
    ), mock.patch(
        "fsec.singularity_subtraction.ccdss.logger.perf_counter",
        side_effect=[20.0, 23.5],
    ):
        solver._prepare_ss(t2)

    assert (
        "KRCCDSS xi precomputation CPU 2.00 sec, wall 3.50 sec"
        in solver.stdout.getvalue()
    )


@pytest.mark.parametrize("use_constraint_1", [True, False])
def test_density_only_aggregates_match_complex_nested_loop_references(
    use_constraint_1,
):
    """Match all six density-only channels and ignore changed or zero T2."""
    if use_constraint_1:
        solver = _aggregate_solver_for_tests()
        reference = lambda: _explicit_diagonal_density_raw(solver, 1)
    else:
        solver, pair_densities = _relaxed_aggregate_solver_for_tests()
        reference = lambda: _explicit_full_density_raw(solver, pair_densities, 1)
    rng = np.random.default_rng(43)
    t2 = rng.normal(size=(2, 2, 2, 2, 2, 2, 2)) + 1j * rng.normal(
        size=(2, 2, 2, 2, 2, 2, 2)
    )
    raw = solver._contract_aggregate_structure_factors(t2)
    active_count = solver._ss_active_count
    expected = np.stack(
        (np.full(6, active_count), reference()), axis=1
    )
    np.testing.assert_allclose(raw, expected)
    np.testing.assert_allclose(
        solver._aggregate_channel_samples(t2)[:, 1],
        expected[:, 1] / active_count,
    )
    np.testing.assert_array_equal(
        solver._aggregate_channel_samples(np.zeros_like(t2))[:, 0],
        np.ones(6),
    )
    np.testing.assert_allclose(
        solver._aggregate_channel_samples(np.zeros_like(t2)),
        solver._aggregate_channel_samples(3.0 * t2),
    )


def test_full_density_masks_padded_external_and_internal_states():
    """Exclude padded endpoint states from external and internal sums."""
    solver, pair_densities = _relaxed_aggregate_solver_for_tests(
        nocc=2, nvir=2
    )
    occupied = np.asarray([[True, False], [True, True]])
    virtual = np.asarray([[True, True], [False, True]])
    solver._ss_active_masks = occupied, virtual
    solver._ss_active_count = None
    pair_densities[1] = 1.0 + 0.5j
    raw = solver._contract_aggregate_structure_factors(
        np.zeros((2, 2, 2, 2, 2, 2, 2), dtype=complex)
    )
    active_count = sum(
        occupied[ki].sum()
        * occupied[kj].sum()
        * virtual[ka].sum()
        * virtual[solver.khelper.kconserv[ki, ka, kj]].sum()
        for ki in range(2)
        for kj in range(2)
        for ka in range(2)
    )
    expected = np.stack(
        (
            np.full(6, active_count),
            _explicit_full_density_raw(solver, pair_densities, 1),
        ),
        axis=1,
    )
    np.testing.assert_allclose(raw, expected)
    assert raw[0, 1] != raw[0, 0]


def test_diagonal_transition_densities_reproduce_the_constrained_path():
    """Reduce full-density contractions to the unchanged diagonal factors."""
    constrained = _aggregate_solver_for_tests()
    relaxed = _aggregate_solver_for_tests()
    factors = constrained._ss_pair_factors
    pair_densities = np.zeros(
        (2, 2, constrained.nmo, constrained.nmo), dtype=complex
    )
    for q_index in range(2):
        for k in range(2):
            pair_densities[q_index, k] = np.diag(factors[q_index, k])
    relaxed.options = CCDSSOptions(
        use_constraint_1=False, use_constraint_2=True, fixed_sigma=None
    )
    relaxed._ss_pair_densities = pair_densities
    relaxed._ss_pair_factors = None
    rng = np.random.default_rng(47)
    t2 = rng.normal(size=(2, 2, 2, 2, 2, 2, 2)) + 1j * rng.normal(
        size=(2, 2, 2, 2, 2, 2, 2)
    )
    np.testing.assert_allclose(
        relaxed._contract_aggregate_structure_factors(t2),
        constrained._contract_aggregate_structure_factors(t2),
    )


def test_relaxed_off_diagonal_density_elements_change_the_aggregate():
    """Include off-diagonal transitions instead of silently using diagonals."""
    solver, pair_densities = _relaxed_aggregate_solver_for_tests()
    diagonal = np.diagonal(pair_densities, axis1=2, axis2=3).copy()
    diagonal_densities = np.zeros_like(pair_densities)
    for q_index in range(2):
        for k in range(2):
            diagonal_densities[q_index, k] = np.diag(diagonal[q_index, k])
    t2 = np.arange(2**7, dtype=float).reshape((2,) * 7) + 1j
    full = solver._contract_aggregate_structure_factors(t2)
    solver._ss_pair_densities = diagonal_densities
    diagonal_raw = solver._contract_aggregate_structure_factors(t2)
    assert np.max(np.abs(full[:, 1] - diagonal_raw[:, 1])) > 1e-8


@pytest.mark.parametrize("use_constraint_1", [True, False])
def test_density_only_fit_is_cached_once_for_both_constraint_1_modes(
    use_constraint_1,
):
    """Perform exactly six fits, then reuse them for later residual updates."""
    if use_constraint_1:
        solver = _aggregate_solver_for_tests(nocc=1, nvir=1)
    else:
        solver, _ = _relaxed_aggregate_solver_for_tests(nocc=1, nvir=1)
    t2 = np.ones((2, 2, 2, 1, 1, 1, 1), dtype=complex)
    changed = 3.0 * t2
    fit_values = []

    def fake_fit(q_vectors, values, fit_with_coul=True):
        """Capture one normalized curve and return a distinct width."""
        fit_values.append(np.array(values, copy=True))
        return float(len(fit_values))

    with mock.patch(
        "fsec.singularity_subtraction.ccdss._fit_unit_gaussian",
        side_effect=fake_fit,
    ), mock.patch.object(solver, "_gaussian_xi", side_effect=lambda sigma: sigma):
        solver._prepare_ss(t2)
        coefficient_a = solver._ss_residual_coefficient(t2)
        coefficient_b = solver._ss_residual_coefficient(changed)

    assert len(fit_values) == 6
    assert solver.ss_fit_count == 6
    assert solver.ss_prepare_count == 1
    assert solver.ss_sigmas.shape == (6,)
    assert solver.ss_xi.shape == (6,)
    assert coefficient_a == pytest.approx(1 + 2 - 3 - 4 - 5 - 6)
    assert coefficient_b == coefficient_a


def test_debug2_reports_one_row_per_aggregate_channel_and_q_sample():
    """Emit parseable fit diagnostics for every channel and sampled q point."""
    solver = _aggregate_solver_for_tests(nocc=1, nvir=1)
    solver.verbose = logger.DEBUG2
    t2 = np.ones((2, 2, 2, 1, 1, 1, 1), dtype=complex)
    samples = np.ones((6, 2), dtype=complex)
    samples[:, 1] = np.arange(1, 7) * (0.1 - 0.2j)
    with mock.patch.object(
        solver, "_aggregate_channel_samples", return_value=samples
    ), mock.patch(
        "fsec.singularity_subtraction.ccdss._fit_unit_gaussian",
        return_value=2.0,
    ), mock.patch.object(solver, "_gaussian_xi", return_value=0.25):
        solver._prepare_ss(t2)

    lines = [
        line
        for line in solver.stdout.getvalue().splitlines()
        if line.startswith("CCDSS_SF ")
    ]
    assert len(lines) == 12
    assert "prep=1 channel=L1 q_index=0" in lines[0]
    assert "prep=1 channel=L6 q_index=1" in lines[-1]
    assert "ki=" not in lines[0] and " i=" not in lines[0]
    raw = samples[2, 1]
    fit = np.exp(-1.0 / 8.0)
    residual = fit - raw
    row = lines[2 * 2 + 1]
    assert "channel=L3 q_index=1" in row
    assert f"raw_real={raw.real:.16e}" in row
    assert f"raw_imag={raw.imag:.16e}" in row
    assert "qx=1.0000000000000000e+00" in row
    assert "sigma=2.0000000000000000e+00" in row
    assert "xi=2.5000000000000000e-01" in row
    assert f"fit={fit:.16e}" in row
    assert f"residual_real={residual.real:.16e}" in row
    assert f"residual_imag={residual.imag:.16e}" in row


def test_debug2_reports_raw_aggregate_normalization_per_channel():
    """Emit the physical active-entry count for every channel origin."""
    solver = _aggregate_solver_for_tests(nocc=1, nvir=1)
    solver.verbose = logger.DEBUG2
    t2 = np.full(
        (2, 2, 2, 1, 1, 1, 1), 1.0 + 0.5j, dtype=complex
    )
    solver._build_active_masks()
    active_count = solver._ss_active_count
    with mock.patch(
        "fsec.singularity_subtraction.ccdss._fit_unit_gaussian",
        return_value=2.0,
    ), mock.patch.object(solver, "_gaussian_xi", return_value=0.25):
        solver._prepare_ss(t2)

    lines = [
        line
        for line in solver.stdout.getvalue().splitlines()
        if line.startswith("CCDSS_SF_NORM ")
    ]
    assert len(lines) == 6
    for channel, line in enumerate(lines, start=1):
        assert f"prep=1 channel=L{channel} " in line
        assert f"norm_real={active_count:.16e}" in line
        assert "norm_imag=0.0000000000000000e+00" in line
        assert f"norm_abs={active_count:.16e}" in line


def test_init_amps_prepares_fitted_state_once():
    """Prepare the cached six-channel state during amplitude initialization."""
    solver = solver_shell()
    solver.options = CCDSSOptions(fixed_sigma=None)
    eris = SimpleNamespace()
    solver.verbose = logger.DEBUG2
    with mock.patch.object(
        KRCCD_SS, "_prepare_ss"
    ) as prepare, mock.patch.object(
        KRCCD,
        "init_amps",
        return_value=(
            1.0,
            np.zeros((1, 1, 1)),
                np.ones((1, 1, 1, 1, 1, 1, 1)),
            ),
        ):
        result = solver.init_amps(eris)
    prepare.assert_called_once_with(result[2])


def test_update_amps_keeps_t1_zero_and_validates_t2_shape():
    """Enforce CCD singles and validate doubles before the parent update."""
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
    """Check fitted execution and exact fixed-width limits on a small cell."""
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
        "madelung": KRCCD(mf, madelung_orbital=True, madelung_eri=True),
        "infinite": KRCCD_SS(mf, fixed_sigma=np.inf),
        "fitted": KRCCD_SS(mf),
        "constraint_1_off": KRCCD_SS(mf, use_constraint_1=False),
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
    assert energies["infinite"] == pytest.approx(energies["madelung"], abs=1e-11)

    fitted = calculations["fitted"]
    assert fitted.ss_prepare_count == 1
    assert fitted.ss_fit_count == 6
    assert fitted.ss_sigmas.shape == (6,)
    assert fitted.ss_xi.shape == (6,)
    np.testing.assert_array_equal(fitted._ss_pair_factors[0], 1.0)
    relaxed = calculations["constraint_1_off"]
    assert relaxed.ss_prepare_count == 1
    assert relaxed.ss_fit_count == 6
    assert np.all(np.isfinite(relaxed.ss_sigmas))
    assert calculations["zero"].ss_prepare_count == 1
    assert calculations["infinite"].ss_prepare_count == 1
