import unittest

import numpy as np

from fsec.singularity_subtraction.function_fitting import (
    ETBLinearLeastSquares,
    ExxScipyLeastSquares,
    ExxScipyMinimize,
    MP2ScipyLeastSquares,
    MP2ScipyMinimize,
)
from fsec.singularity_subtraction.model_function.etb import (
    cutoff_for_min_shells,
    radial_shells,
    trim_etb_model_to_data,
)
from fsec.singularity_subtraction.model_function.exx_modfunc import (
    ContractedGaussianModel,
    EvenTemperedGaussianModel,
    ExponentialModel,
    QuarticExponentialModel,
)
from fsec.singularity_subtraction.model_function.mp2_direct_modfunc import (
    XNETB,
    XNETBLiftedStackedSingularityQMesh,
    XNETBStackedSingularity,
    XNETBStackedSingularityQMesh,
    XNGauss,
    XNGaussStackedSingularityQMesh,
)
from fsec.singularity_subtraction.model_function.mp2_exchange_modfunc import (
    XNETBStackedSingularityExchange,
    XNGaussStackedSingularityExchange,
)


class CubicCell:
    def __init__(self, lattice_constant):
        self.lattice_constant = lattice_constant
        self.reciprocal_lattice_constant = 2 * np.pi / lattice_constant

    def get_scaled_kpts(self, kpts):
        return np.asarray(kpts) / self.reciprocal_lattice_constant

    def get_abs_kpts(self, scaled_kpts):
        return np.asarray(scaled_kpts) * self.reciprocal_lattice_constant


def build_cubic_reciprocal_lattice_grid(mesh, reciprocal_lattice_constant):
    gx, gy, gz = np.meshgrid(
        *[np.fft.fftfreq(n, d=1 / n) for n in mesh],
        indexing="ij",
    )
    return (
        np.hstack((gx.reshape(-1, 1), gy.reshape(-1, 1), gz.reshape(-1, 1)))
        * reciprocal_lattice_constant
    )


class KnownValues(unittest.TestCase):
    QG_20 = np.array(
        [
            [0.0, 0.0, 0.0],
            [-1.0471975511965976, 0.0, 0.0],
            [0.0, -1.0471975511965976, 0.0],
            [0.0, 0.0, -1.0471975511965976],
            [0.0, 0.0, 1.0471975511965976],
            [0.0, 1.0471975511965976, 0.0],
            [1.0471975511965976, 0.0, 0.0],
            [-1.0471975511965976, -1.0471975511965976, 0.0],
            [-1.0471975511965976, 0.0, -1.0471975511965976],
            [-1.0471975511965976, 0.0, 1.0471975511965976],
            [-1.0471975511965976, 1.0471975511965976, 0.0],
            [0.0, -1.0471975511965976, -1.0471975511965976],
            [0.0, -1.0471975511965976, 1.0471975511965976],
            [0.0, 1.0471975511965976, -1.0471975511965976],
            [0.0, 1.0471975511965976, 1.0471975511965976],
            [1.0471975511965976, -1.0471975511965976, 0.0],
            [1.0471975511965976, 0.0, -1.0471975511965976],
            [1.0471975511965976, 0.0, 1.0471975511965976],
            [1.0471975511965976, 1.0471975511965976, 0.0],
            [-1.0471975511965976, -1.0471975511965976, -1.0471975511965976],
        ]
    )
    REFERENCE_EXX_20 = np.array(
        [
            1.0000000040758938,
            0.32744907530353146,
            0.3274490753036006,
            0.13484040769431635,
            0.13484040769431635,
            0.3274490753036006,
            0.32744907530353146,
            0.1329367270735302,
            0.055346487300111935,
            0.055346487300113774,
            0.1329367270735332,
            0.055346487300126236,
            0.055346487300125986,
            0.055346487300125986,
            0.055346487300126236,
            0.1329367270735332,
            0.055346487300113774,
            0.055346487300111935,
            0.1329367270735302,
            0.026606903933605867,
        ]
    )
    REFERENCE_MP2_DIRECT_20 = np.array(
        [
            -1.0893511193700196e-21,
            -1.054426172919673e-21,
            -1.0543974002128198e-21,
            -0.00026261641191463426,
            -0.00026261641191463426,
            -1.0543974002128198e-21,
            -1.054426172919673e-21,
            -1.0205123510313412e-21,
            -0.00011402023793530142,
            -0.00011402023793529546,
            -1.0204582352612464e-21,
            -0.00011402023793531614,
            -0.00011402023793530657,
            -0.00011402023793530657,
            -0.00011402023793531614,
            -1.0204582352612464e-21,
            -0.00011402023793529546,
            -0.00011402023793530142,
            -1.0205123510313412e-21,
            -5.774656738478592e-05,
        ]
    )
    REFERENCE_MP2_Q4_20 = np.array(
        [
            -7.717470607835843e-41,
            -7.230553951086863e-41,
            -7.23015934826672e-41,
            -4.485211663249214e-06,
            -4.485211663249214e-06,
            -7.23015934826672e-41,
            -7.230553951086863e-41,
            -6.772916892974211e-41,
            -8.454795391735684e-07,
            -8.454795391734801e-07,
            -6.77219860299934e-41,
            -8.454795391737869e-07,
            -8.45479539173645e-07,
            -8.45479539173645e-07,
            -8.454795391737869e-07,
            -6.77219860299934e-41,
            -8.454795391734801e-07,
            -8.454795391735684e-07,
            -6.772916892974211e-41,
            -2.168660470894743e-07,
        ]
    )
    REFERENCE_MP2_EXCHANGE_20 = np.array(
        [
            5.446755596850098e-22,
            5.272130864598365e-22,
            5.271987001064099e-22,
            0.00013130820595731713,
            0.00013130820595731713,
            5.271987001064099e-22,
            5.272130864598365e-22,
            5.102561755156706e-22,
            5.701011896765071e-05,
            5.701011896764773e-05,
            5.102291176306232e-22,
            5.701011896765807e-05,
            5.7010118967653286e-05,
            5.7010118967653286e-05,
            5.701011896765807e-05,
            5.102291176306232e-22,
            5.701011896764773e-05,
            5.701011896765071e-05,
            5.102561755156706e-22,
            2.887328369239296e-05,
        ]
    )

    @classmethod
    def setUpClass(cls):
        cls.cell = CubicCell(lattice_constant=6.0)
        cls.gamma_qGrid = np.array([[0.0, 0.0, 0.0]])
        cls.mp2_deltaGs = build_cubic_reciprocal_lattice_grid(
            mesh=(29, 29, 29),
            reciprocal_lattice_constant=cls.cell.reciprocal_lattice_constant,
        )
        cls.exx_qG = cls.QG_20
        cls.exx_SqG = cls.REFERENCE_EXX_20

        cls.mp2_qG = cls.QG_20
        cls.SqG_full_direct = cls.REFERENCE_MP2_DIRECT_20
        cls.SqG_full_q4 = cls.REFERENCE_MP2_Q4_20
        cls.SqG_full_exchange = cls.REFERENCE_MP2_EXCHANGE_20

    def _assert_fitted_parameters(self, fitted, expected):
        self.assertEqual(len(fitted), len(expected))
        for actual, reference in zip(fitted, expected):
            self.assertAlmostEqual(actual, reference, places=3)

    def _assert_parameters_with_coul(
        self, fit_cls, model, input_data, output_data, expected, **fit_kwargs
    ):
        fit_method = fit_cls(model, fit_with_coul=True)
        fitted = fit_method.fit_model(input_data, output_data, **fit_kwargs)
        self._assert_fitted_parameters(fitted, expected)

    def _assert_parameters(
        self,
        fit_cls,
        model,
        input_data,
        output_data,
        expected,
        expected_with_coul,
        **fit_kwargs,
    ):
        fit_method = fit_cls(model, fit_with_coul=False)
        fitted = fit_method.fit_model(input_data, output_data, **fit_kwargs)
        self._assert_fitted_parameters(fitted, expected)
        self._assert_parameters_with_coul(
            fit_cls,
            model,
            input_data,
            output_data,
            expected_with_coul,
            **fit_kwargs,
        )

    def test_exx_scipy_minimize_contracted_gaussian(self):
        model = ContractedGaussianModel(num_gaussians=1, isotropic=True)
        expected = [1.0, 1.0e-8]
        expected_with_coul = [1.0, 1.0e-8]
        self._assert_parameters(
            ExxScipyMinimize,
            model,
            self.exx_qG,
            self.exx_SqG,
            expected,
            expected_with_coul,
        )

    def test_exx_scipy_least_squares_contracted_gaussian(self):
        model = ContractedGaussianModel(num_gaussians=1, isotropic=True)
        expected = [1.0, 0.6485758873718301]
        expected_with_coul = [1.0, 0.64351473]
        self._assert_parameters(
            ExxScipyLeastSquares,
            model,
            self.exx_qG,
            self.exx_SqG,
            expected,
            expected_with_coul,
        )

    def test_exx_scipy_minimize_quartic_exponential(self):
        model = QuarticExponentialModel(num_primitives=1)
        expected = [1.0, 0.7388025253272993, 3.0887137273423044, 1.4083534745912776]
        expected_with_coul = [1.0, 0.81100787, 3.04164405, 1.272494]
        self._assert_parameters(
            ExxScipyMinimize,
            model,
            self.exx_qG,
            self.exx_SqG,
            expected,
            expected_with_coul,
            force_positive_params=True,
        )

    def test_exx_scipy_least_squares_quartic_exponential(self):
        model = QuarticExponentialModel(num_primitives=1)
        expected = [1.0, 2.1081797177657133, 2.7564722224352383, 0.40869039302204624]
        expected_with_coul = [1.0, 2.10855771, 2.75644022, 0.40859361]
        self._assert_parameters(
            ExxScipyLeastSquares,
            model,
            self.exx_qG,
            self.exx_SqG,
            expected,
            expected_with_coul,
            force_positive_params=True,
        )

    def test_mp2_scipy_minimize_direct_xngauss_stacked_singularity(self):
        model = XNGaussStackedSingularityQMesh(
            qGrid=self.gamma_qGrid,
            cell=self.cell,
            parameters=[1.0e-4, 1.0],
            deltaGs=self.mp2_deltaGs,
        )
        expected = [2.7391986894019372e-05, 0.8193662600962808]
        expected_with_coul = [2.86031846e-05, 0.813529376]
        self._assert_parameters(
            MP2ScipyMinimize,
            model,
            self.mp2_qG,
            self.SqG_full_direct,
            expected,
            expected_with_coul,
            fit_multipliers=[1.0e4, 1.0],
        )

    def test_mp2_scipy_least_squares_direct_xngauss_stacked_singularity(self):
        model = XNGaussStackedSingularityQMesh(
            qGrid=self.gamma_qGrid,
            cell=self.cell,
            parameters=[1.0e-4, 1.0],
            deltaGs=self.mp2_deltaGs,
        )
        expected = [2.7391967036276372e-05, 0.81936636144651]
        expected_with_coul = [2.86031466e-05, 0.813529570]
        self._assert_parameters(
            MP2ScipyLeastSquares,
            model,
            self.mp2_qG,
            self.SqG_full_direct,
            expected,
            expected_with_coul,
        )

    def test_mp2_scipy_minimize_dg0_xngauss(self):
        model = XNGauss(parameters=[1.0e-4, 1.0], negative=True, deg=4)
        expected = [0.5696568538141716, 0.20509585399570057]
        expected_with_coul = [0.60520757, 0.20462117]
        self._assert_parameters(
            MP2ScipyMinimize,
            model,
            self.mp2_qG,
            self.SqG_full_q4,
            expected,
            expected_with_coul,
            fit_multipliers=[1.0e4, 1.0],
        )

    def test_mp2_scipy_least_squares_dg0_xngauss(self):
        model = XNGauss(parameters=[1.0e-4, 1.0], negative=True, deg=4)
        expected = [1.3088397673385366e-05, 0.48253452987121476]
        expected_with_coul = [1.31522978e-05, 0.482114944]
        self._assert_parameters(
            MP2ScipyLeastSquares,
            model,
            self.mp2_qG,
            self.SqG_full_q4,
            expected,
            expected_with_coul,
        )

    def test_mp2_scipy_minimize_exchange_xngauss_stacked_singularity_exchange(self):
        model = XNGaussStackedSingularityExchange(
            parameters=[1.0e-4, 1.0], q2s=self.mp2_qG, dvol=1.0
        )
        expected = [1.7642180397497834e-05, 0.8193663363312604]
        expected_with_coul = [1.82902313e-05, 0.813529541]
        self._assert_parameters(
            MP2ScipyMinimize,
            model,
            self.mp2_qG,
            self.SqG_full_exchange,
            expected,
            expected_with_coul,
            fit_multipliers=[1.0e4, 1.0],
        )

    def test_mp2_scipy_least_squares_exchange_xngauss_stacked_singularity_exchange(self):
        model = XNGaussStackedSingularityExchange(
            parameters=[1.0e-4, 1.0], q2s=self.mp2_qG, dvol=1.0
        )
        expected = [1.7642177723423226e-05, 0.8193663658499301]
        expected_with_coul = [1.82902301e-05, 0.813529559]
        self._assert_parameters(
            MP2ScipyLeastSquares,
            model,
            self.mp2_qG,
            self.SqG_full_exchange,
            expected,
            expected_with_coul,
        )


class IsotropicInputShapeTests(unittest.TestCase):
    def setUp(self):
        self.model = QuarticExponentialModel(
            num_primitives=1, parameters=[1.0, 1.0, 1.0, 1.0]
        )
        self.fit_method = ExxScipyLeastSquares(self.model, fit_with_coul=False)
        self.params = np.array(self.model.parameters, dtype=float)

    def test_abs_diff_accepts_nx3_input_by_converting_to_r(self):
        qg = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 3.0, 4.0],
            ]
        )
        r = np.linalg.norm(qg, axis=1)
        output = self.model.eval_model_r(r)

        residual = self.fit_method.abs_diff(self.params, qg, output)
        self.assertTrue(np.allclose(residual, 0.0, atol=1e-12))

    def test_abs_diff_accepts_nx1_input(self):
        r_col = np.array([[0.0], [1.0], [2.0]])
        output = self.model.eval_model_r(r_col[:, 0])

        residual = self.fit_method.abs_diff(self.params, r_col, output)
        self.assertTrue(np.allclose(residual, 0.0, atol=1e-12))

    def test_abs_diff_rejects_invalid_isotropic_shape(self):
        bad_input = np.array([[1.0, 2.0], [3.0, 4.0]])
        output = np.array([0.0, 0.0])

        with self.assertRaises(ValueError):
            _ = self.fit_method.abs_diff(self.params, bad_input, output)



class EvenTemperedFittingTests(unittest.TestCase):
    spacing = 0.6
    coeffs = np.array([0.1, 0.2, 0.3, 0.25, 0.15])

    @classmethod
    def setUpClass(cls):
        g = np.arange(-4, 5) * cls.spacing
        grid = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
        cls.grid = grid
        cls.qG = grid[np.linalg.norm(grid, axis=1) < 2.5]

    def test_linear_fit_recovers_exx_coefficients(self):
        for coeffs in (self.coeffs, [0.0, 0.0, 1.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0, 0.5]):
            with self.subTest(coeffs=coeffs):
                SqG = EvenTemperedGaussianModel(parameters=coeffs).eval_model(self.qG)
                fit_method = ETBLinearLeastSquares(EvenTemperedGaussianModel(), fit_with_coul=True)
                fitted = fit_method.fit_model(self.qG, SqG)
                np.testing.assert_allclose(fitted, coeffs, atol=1e-10)
                self.assertAlmostEqual(np.sum(fitted), 1.0, places=14)

    def test_linear_fit_recovers_mp2_coefficients(self):
        q4_model = XNETB(parameters=1e-4 * self.coeffs, deg=4)
        fit_method = ETBLinearLeastSquares(XNETB(deg=4), fit_with_coul=True)
        fit_method.coul_deg = 4
        fitted = fit_method.fit_model(self.qG, q4_model.eval_model(self.qG))
        np.testing.assert_allclose(fitted, 1e-4 * self.coeffs, rtol=1e-8)

        exchange_model = XNETBStackedSingularityExchange(parameters=1e-3 * self.coeffs, q2s=self.grid, dvol=0.1)
        fit_method = ETBLinearLeastSquares(
            XNETBStackedSingularityExchange(q2s=self.grid, dvol=0.1), fit_with_coul=True)
        fitted = fit_method.fit_model(self.qG, exchange_model.eval_model(self.qG))
        np.testing.assert_allclose(fitted, 1e-3 * self.coeffs, rtol=1e-8)

    def test_linear_fit_matches_nonlinear_fit(self):
        """The linear problem has a unique minimizer, which the nonlinear fit methods must also find."""
        # Heavy-tailed target whose best ETB coefficients are all strictly positive (ExxScipyLeastSquares
        # eliminates c_0 without bounding it, so it only solves the same problem for interior optima)
        target = ExponentialModel(num_primitives=1, isotropic=True, parameters=[1.0, 2.0, 0.5])
        SqG = target.eval_model(self.qG)
        linear = ETBLinearLeastSquares(EvenTemperedGaussianModel(), fit_with_coul=True).fit_model(self.qG, SqG)
        self.assertTrue(np.all(linear > 1e-2))

        start = EvenTemperedGaussianModel().default_parameters()
        nonlinear = ExxScipyLeastSquares(
            EvenTemperedGaussianModel(), fit_with_coul=True, initial_guess=start,
        ).fit_model(self.qG, SqG, force_positive_params=True)
        np.testing.assert_allclose(nonlinear, linear, atol=1e-4)

        # MP2 q4: the linear fit recovers these exactly (test_linear_fit_recovers_mp2_coefficients); the nonlinear
        # optimizer converges to them within its tolerance
        q4_data = XNETB(parameters=1e-4 * self.coeffs, deg=4).eval_model(self.qG)
        q4_model = XNETB(deg=4)
        q4_nonlinear_fit = MP2ScipyLeastSquares(q4_model, fit_with_coul=True)
        q4_nonlinear_fit.coul_deg = 4
        q4_nonlinear_fit.initial_guess = q4_model.initial_parameters(1e-4)
        q4_nonlinear = q4_nonlinear_fit.fit_model(self.qG, q4_data, force_positive_params=True)
        np.testing.assert_allclose(q4_nonlinear, 1e-4 * self.coeffs, rtol=1e-3)

    def test_radial_shells(self):
        single_shell = np.vstack([np.eye(3), -np.eye(3)])
        self.assertEqual(len(radial_shells(single_shell)), 1)
        self.assertEqual(len(radial_shells(np.vstack([np.zeros((1, 3)), single_shell, 2 * single_shell]))), 2)
        self.assertAlmostEqual(cutoff_for_min_shells(self.grid, 2), np.sqrt(2) * self.spacing + 1e-8)
        self.assertIsNone(cutoff_for_min_shells(single_shell, 2))

    def test_too_few_shells(self):
        few = self.qG[np.linalg.norm(self.qG, axis=1) < 0.9]
        self.assertEqual(len(radial_shells(few)), 2)
        SqG = EvenTemperedGaussianModel(parameters=self.coeffs).eval_model(few)

        # Fitting all five coefficients from two |q| shells is ill-posed
        with self.assertRaisesRegex(ValueError, "not uniquely determined"):
            ETBLinearLeastSquares(EvenTemperedGaussianModel(), fit_with_coul=True).fit_model(few, SqG)

        # Trimming keeps the largest window around the reference Gaussian the data can determine
        trimmed = trim_etb_model_to_data(EvenTemperedGaussianModel(), few)
        self.assertLess(trimmed.num_gaussians, 5)
        self.assertIn(0.5, trimmed.etb_exponents)
        ETBLinearLeastSquares(trimmed, fit_with_coul=True).fit_model(few, SqG)

        enough = self.qG[np.linalg.norm(self.qG, axis=1) < 1.5]
        self.assertEqual(trim_etb_model_to_data(EvenTemperedGaussianModel(), enough).num_gaussians, 5)

        q4 = XNETB(deg=4)
        trimmed_q4 = trim_etb_model_to_data(q4, few)
        self.assertLess(trimmed_q4.num_gaussians, 5)
        self.assertEqual(trimmed_q4.deg, 4)

    def test_stacked_model_trimming_uses_proxy_design(self):
        cell = CubicCell(lattice_constant=6.0)
        deltaGs = build_cubic_reciprocal_lattice_grid((9, 9, 9), cell.reciprocal_lattice_constant)
        qGrid = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, cell.reciprocal_lattice_constant / 2]])
        model = XNETBStackedSingularityQMesh(qGrid=qGrid, cell=cell, deltaGs=deltaGs)
        model.set_parameters(model.initial_parameters(1e-4))
        coords = (qGrid[:, None, :] + deltaGs[None, :, :]).reshape(-1, 3)

        norms = np.linalg.norm(coords, axis=1)
        self.assertIs(trim_etb_model_to_data(model, coords[norms < 4.0]), model)

        trimmed = trim_etb_model_to_data(model, coords[norms < 1.1])  # two |q| shells
        self.assertLess(trimmed.num_gaussians, 5)
        self.assertIn(0.5, trimmed.etb_exponents)
        np.testing.assert_allclose(trimmed.qGrid, qGrid)

        with self.assertRaisesRegex(ValueError, "cannot determine"):
            trim_etb_model_to_data(model, coords[norms < 0.6])  # one |q| shell

        # A narrow reference Gaussian underflows at large |q|; the proxy columns must stay finite
        g = np.arange(-7, 8) * self.spacing
        wide_grid = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
        narrow = XNETBStackedSingularity(deltaGs=wide_grid, sigma=0.1)
        far = wide_grid[np.linalg.norm(wide_grid, axis=1) < 4.5]
        self.assertTrue(np.all(np.isfinite(narrow.linear_proxy_design_matrix(far))))
        trimmed = trim_etb_model_to_data(narrow, far)
        self.assertAlmostEqual(trimmed.etb_exponents[trimmed.etb_ref_index], 50.0)

    def _lifted_setup(self, remove_deltaG_zero=True, qcut=4.0):
        cell = CubicCell(lattice_constant=6.0)
        b = cell.reciprocal_lattice_constant
        deltaGs = build_cubic_reciprocal_lattice_grid((15, 15, 15), b)
        shifts = np.array([0.0, 0.5])
        qGrid = np.array(np.meshgrid(shifts, shifts, shifts, indexing="ij")).reshape(3, -1).T * b
        model = XNETBLiftedStackedSingularityQMesh(
            qGrid=qGrid, cell=cell, deltaGs=deltaGs, remove_deltaG_zero=remove_deltaG_zero)
        coords = (qGrid[:, None, :] + deltaGs[None, :, :]).reshape(-1, 3)
        return model, coords[np.linalg.norm(coords, axis=1) < qcut]

    def test_lifted_linear_fit_recovers_pair_matrix(self):
        """The lifted q2 model is linear in B, so a non-rank-1 B is recovered exactly by NNLS."""
        model, coords = self._lifted_setup()
        B = 1e-4 * (np.outer(self.coeffs, self.coeffs) + np.diag([0.0, 0.05, 0.0, 0.02, 0.0]))
        true = B[model.pair_rows, model.pair_cols]
        model.set_parameters(true)
        data = model.eval_model(coords)

        fit_model, _ = self._lifted_setup()
        self.assertIs(trim_etb_model_to_data(fit_model, coords), fit_model)
        fitted = ETBLinearLeastSquares(fit_model, fit_with_coul=True).fit_model(coords, data)
        np.testing.assert_allclose(fitted, true, rtol=1e-6, atol=1e-14)

    def test_lifted_model_reduced_until_parameters_are_determined(self):
        # Few |q| shells: N (N + 1) / 2 = 15 pair parameters cannot all be determined
        model, coords = self._lifted_setup(qcut=1.6)
        n_shells = len(radial_shells(coords))
        self.assertLess(n_shells, model.num_params + 1)
        trimmed = trim_etb_model_to_data(model, coords)
        self.assertLess(trimmed.num_gaussians, 5)
        self.assertGreaterEqual(n_shells, trimmed.num_params + 1)
        self.assertIn(0.5, trimmed.etb_exponents)
        self.assertEqual(trimmed.num_params, trimmed.num_gaussians * (trimmed.num_gaussians + 1) // 2)
        ETBLinearLeastSquares(trimmed, fit_with_coul=True).fit_model(coords, -np.ones(len(coords)))

        # Full direct (dG = 0 kept): H_l is lattice-periodic, so at most n_q * N pair columns are independent
        single_q, coords = self._lifted_setup(remove_deltaG_zero=False)
        single_q = XNETBLiftedStackedSingularityQMesh(
            qGrid=np.zeros((1, 3)), cell=single_q.cell, deltaGs=single_q.deltaGs)
        gamma_coords = coords[np.all(np.isclose(
            single_q.cell.get_scaled_kpts(coords) % 1.0, 0.0) | np.isclose(
            single_q.cell.get_scaled_kpts(coords) % 1.0, 1.0), axis=1)]
        trimmed = trim_etb_model_to_data(single_q, gamma_coords)
        self.assertLessEqual(trimmed.num_params, trimmed.num_gaussians)

    def test_linear_fit_rejects_nonlinear_models(self):
        cell = CubicCell(lattice_constant=6.0)
        deltaGs = build_cubic_reciprocal_lattice_grid((3, 3, 3), cell.reciprocal_lattice_constant)
        model = XNETBStackedSingularityQMesh(qGrid=np.zeros((1, 3)), cell=cell, deltaGs=deltaGs)
        with self.assertRaises(TypeError):
            ETBLinearLeastSquares(model)


if __name__ == "__main__":
    unittest.main()
