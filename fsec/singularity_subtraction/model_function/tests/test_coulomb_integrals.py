import unittest

import numpy as np
from scipy.integrate import dblquad

from fsec.singularity_subtraction.model_function import IsotropicModel
from fsec.singularity_subtraction.model_function.etb import even_tempered_exponents
from fsec.singularity_subtraction.model_function.exx_modfunc import (
    ContractedGaussianModel,
    EvenTemperedGaussianModel,
    ExpoLorentzianModel,
    ExponentialModel,
    QuarticExponentialModel,
)
from fsec.singularity_subtraction.model_function.mp2_direct_modfunc import (
    XNETB,
    XNETBStackedSingularity,
    XNETBStackedSingularityQMesh,
    XNExpAbs,
    XNExpAbs2,
    XNExpAbsStackedSingularity,
    XNExpAbsStackedSingularityQMesh,
    XNExponential,
    XNGauss,
    XNGaussStackedSingularity,
    XNGaussStackedSingularityQMesh,
    XNGeneral,
    XNQuarticExponential,
)
from fsec.singularity_subtraction.model_function.mp2_exchange_modfunc import (
    MP2StackedSingularityExchange,
    XNETBStackedSingularityExchange,
    XNExpAbs2StackedSingularityExchange,
    XNExpAbsStackedSingularityExchange,
    XNExponentialStackedSingularityExchange,
    XNGaussStackedSingularityExchange,
    XNQuarticExponentialStackedSingularityExchange,
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


class CoulombIntegralDefaultParameterTests(unittest.TestCase):
    cell = CubicCell(lattice_constant=6.0)
    deltaGs = build_cubic_reciprocal_lattice_grid(
        mesh=(29, 29, 29),
        reciprocal_lattice_constant=cell.reciprocal_lattice_constant,
    )
    qGrid = np.array([[0.0, 0.0, 0.0]])
    q2s = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ]
    )

    def _assert_coulomb_integral(self, model, expected):
        actual = model.coulomb_integral()
        self.assertTrue(np.isfinite(actual))
        self.assertAlmostEqual(actual, expected, places=10)

    def test_exx_model_coulomb_integrals_with_default_parameters(self):
        cases = [
            (
                "ContractedGaussianModel",
                ContractedGaussianModel(num_gaussians=1, isotropic=True),
                0.7978845608028654,
            ),
            (
                "ExponentialModel",
                ExponentialModel(num_primitives=1, isotropic=True),
                0.8633719706943167,
            ),
            (
                "ExpoLorentzianModel",
                ExpoLorentzianModel(num_primitives=1),
                0.7285878253497297,
            ),
            (
                "QuarticExponentialModel",
                QuarticExponentialModel(num_primitives=1),
                0.6456928457638528,
            ),
        ]

        for name, model, expected in cases:
            with self.subTest(model=name):
                self._assert_coulomb_integral(model, expected)

    def test_mp2_direct_model_coulomb_integrals_with_default_parameters(self):
        cases = [
            ("XNGauss", XNGauss(), -15.749609945722417),
            ("XNExpAbs", XNExpAbs(parameters=[1.0, 1.0]), -25.132741228718345),
            (
                "XNExpAbs2",
                XNExpAbs2(parameters=[1.0, 1.0, 1.0]),
                -2.0009560178328294,
            ),
            (
                "XNExponential",
                XNExponential(parameters=[1.0, 1.0, 1.0]),
                -55.50276939602421,
            ),
            (
                "XNQuarticExponential",
                XNQuarticExponential(parameters=[1.0, 1.0, 1.0, 1.0]),
                -7.358879567252773,
            ),
            (
                "XNGaussStackedSingularity",
                XNGaussStackedSingularity(deltaGs=self.deltaGs),
                -216.00000000000043,
            ),
            (
                "XNGaussStackedSingularityQMesh",
                XNGaussStackedSingularityQMesh(
                    qGrid=self.qGrid,
                    cell=self.cell,
                    deltaGs=self.deltaGs,
                ),
                -216.00000000000043,
            ),
            (
                "XNExpAbsStackedSingularity",
                XNExpAbsStackedSingularity(deltaGs=self.deltaGs),
                -549.9897072725538,
            ),
            (
                "XNExpAbsStackedSingularityQMesh",
                XNExpAbsStackedSingularityQMesh(
                    qGrid=self.qGrid,
                    cell=self.cell,
                    deltaGs=self.deltaGs,
                ),
                -549.9897072725538,
            ),
        ]

        for name, model, expected in cases:
            with self.subTest(model=name):
                self._assert_coulomb_integral(model, expected)

    def test_mp2_exchange_model_coulomb_integrals_with_default_parameters(self):
        cases = [
            (
                "XNGaussStackedSingularityExchange",
                XNGaussStackedSingularityExchange(
                    parameters=[1.0, 1.0],
                    q2s=self.q2s,
                    dvol=1.0,
                ),
                248.05021344239853,
            ),
            (
                "XNExponentialStackedSingularityExchange",
                XNExponentialStackedSingularityExchange(
                    parameters=[1.0, 1.0, 1.0],
                    q2s=self.q2s,
                    dvol=1.0,
                ),
                3080.557410628242,
            ),
            (
                "XNExpAbsStackedSingularityExchange",
                XNExpAbsStackedSingularityExchange(
                    parameters=[1.0, 1.0],
                    q2s=self.q2s,
                    dvol=1.0,
                ),
                631.6546816697189,
            ),
            (
                "XNExpAbs2StackedSingularityExchange",
                XNExpAbs2StackedSingularityExchange(
                    parameters=[1.0, 1.0, 1.0],
                    q2s=self.q2s,
                    dvol=1.0,
                ),
                4.0038249853014145,
            ),
            (
                "XNQuarticExponentialStackedSingularityExchange",
                XNQuarticExponentialStackedSingularityExchange(
                    parameters=[1.0, 1.0, 1.0, 1.0],
                    q2s=self.q2s,
                    dvol=1.0,
                ),
                54.15310848533036,
            ),
        ]

        for name, model, expected in cases:
            with self.subTest(model=name):
                self._assert_coulomb_integral(model, expected)



class EvenTemperedModelTests(unittest.TestCase):
    cell = CubicCell(lattice_constant=6.0)
    deltaGs = build_cubic_reciprocal_lattice_grid(
        mesh=(9, 9, 9),
        reciprocal_lattice_constant=cell.reciprocal_lattice_constant,
    )
    coeffs = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
    coords = np.random.default_rng(0).normal(size=(40, 3))
    q2s = np.vstack([np.zeros((1, 3)), np.random.default_rng(1).normal(size=(20, 3))])

    def test_even_tempered_exponents(self):
        np.testing.assert_allclose(even_tempered_exponents(), 0.5 * 2.0 ** np.arange(-2, 3))
        np.testing.assert_allclose(
            even_tempered_exponents(sigma=2.0), even_tempered_exponents() / 4.0)
        np.testing.assert_allclose(
            even_tempered_exponents(num_gaussians=4, ratio=3.0), 0.5 * 3.0 ** (np.arange(4) - 1.5))
        for kwargs in ({"sigma": 0.0}, {"num_gaussians": 0}, {"ratio": 1.0}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                even_tempered_exponents(**kwargs)

        model = EvenTemperedGaussianModel(sigma=0.5)
        self.assertEqual(model.etb_ref_index, 2)
        self.assertAlmostEqual(model.etb_exponents[model.etb_ref_index], 2.0)
        subset = model.with_exponents(model.etb_exponents[1:4])
        np.testing.assert_allclose(subset.etb_exponents, model.etb_exponents[1:4])
        self.assertEqual(subset.etb_ref_index, 1)

    def test_reference_coefficients_reduce_to_gaussian_models(self):
        """With only the Gaussian at sigma = 1 active, each ETB model equals its Gaussian counterpart."""
        exx = EvenTemperedGaussianModel()
        gauss = ContractedGaussianModel(num_gaussians=1, isotropic=True, parameters=[1.0, 1.0])
        np.testing.assert_allclose(exx.eval_model(self.coords), gauss.eval_model(self.coords), atol=1e-14)
        self.assertAlmostEqual(exx.coulomb_integral(), gauss.coulomb_integral(), places=12)

        for deg in (2, 4):
            for coul_deg in (2, 4):
                with self.subTest(deg=deg, coul_deg=coul_deg):
                    etb = XNETB(deg=deg)
                    gauss = XNGauss(parameters=[1.0, 1.0], deg=deg)
                    np.testing.assert_allclose(etb.eval_model(self.coords), gauss.eval_model(self.coords))
                    self.assertAlmostEqual(
                        etb.coulomb_integral(coul_deg), gauss.coulomb_integral(coul_deg), places=10)

        for remove_deltaG_zero in (False, True):
            with self.subTest(remove_deltaG_zero=remove_deltaG_zero):
                etb = XNETBStackedSingularity(deltaGs=self.deltaGs, remove_deltaG_zero=remove_deltaG_zero)
                gauss = XNGaussStackedSingularity(
                    parameters=[1.0, 1.0], deltaGs=self.deltaGs, remove_deltaG_zero=remove_deltaG_zero)
                np.testing.assert_allclose(etb.eval_model(self.coords), gauss.eval_model(self.coords))
                self.assertAlmostEqual(etb.coulomb_integral(), gauss.coulomb_integral(), places=10)

        gauss = XNGaussStackedSingularityExchange(parameters=[1.0, 1.0], q2s=self.q2s, dvol=0.1)
        etb = XNETBStackedSingularityExchange(q2s=self.q2s, dvol=0.1)
        etb.set_parameters(etb.one_hot_reference(gauss.uncorrected_q2_quad))
        self.assertAlmostEqual(etb.c0, 1.0, places=12)
        np.testing.assert_allclose(etb.eval_model(self.coords), gauss.eval_model(self.coords))
        self.assertAlmostEqual(etb.coulomb_integral(), gauss.coulomb_integral(), places=8)

    def test_analytic_coulomb_integrals_match_quadrature(self):
        exx = EvenTemperedGaussianModel(parameters=self.coeffs)
        self.assertAlmostEqual(exx.coulomb_integral(), IsotropicModel.coulomb_integral(exx), places=8)

        for deg, coul_deg in ((2, 2), (4, 4), (4, 2)):
            with self.subTest(deg=deg, coul_deg=coul_deg):
                etb = XNETB(parameters=self.coeffs, deg=deg)
                self.assertAlmostEqual(
                    etb.coulomb_integral(coul_deg), XNGeneral.coulomb_integral(etb, coul_deg), places=8)

        exchange = XNETBStackedSingularityExchange(parameters=self.coeffs, q2s=self.q2s, dvol=0.1)
        self.assertAlmostEqual(
            exchange.coulomb_integral() / MP2StackedSingularityExchange.coulomb_integral(exchange), 1.0, places=8)

    def test_stacked_pair_overlap_matches_quadrature(self):
        """-int h(q) h(q + dG) d^3q in cylindrical coordinates, for a single dG including cross terms."""
        dG = np.array([[0.0, 0.0, 0.7]])
        model = XNETBStackedSingularity(parameters=self.coeffs, deltaGs=dG)

        def integrand(z, rho):
            r1 = np.hypot(rho, z)
            r2 = np.hypot(rho, z + dG[0, 2])
            return 2 * np.pi * rho * model.decay_func(r1) * model.decay_func(r2)

        reference, _ = dblquad(integrand, 0, 20, -20, 20, epsabs=1e-12, epsrel=1e-12)
        self.assertAlmostEqual(model.coulomb_integral(), -reference, places=8)

    def test_linear_design_matrix_reproduces_model(self):
        models = [
            EvenTemperedGaussianModel(parameters=self.coeffs),
            XNETB(parameters=self.coeffs, deg=4),
            XNETBStackedSingularityExchange(parameters=self.coeffs, q2s=self.q2s, dvol=0.1),
        ]
        for model in models:
            with self.subTest(model=model.__class__.__name__):
                np.testing.assert_allclose(
                    model.linear_design_matrix(self.coords) @ self.coeffs, model.eval_model(self.coords),
                    rtol=1e-12, atol=1e-14)

    def test_qmesh_matches_direct_stacked_evaluation(self):
        # A large dG mesh so that shifting the stacked sum by a nearby G does not change its truncation
        deltaGs = build_cubic_reciprocal_lattice_grid(
            mesh=(29, 29, 29), reciprocal_lattice_constant=self.cell.reciprocal_lattice_constant)
        qGrid = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, self.cell.reciprocal_lattice_constant / 2]])
        nearby_Gs = deltaGs[np.linalg.norm(deltaGs, axis=1) < 1.1 * self.cell.reciprocal_lattice_constant]
        coords = (qGrid[:, None, :] + nearby_Gs[None, :, :]).reshape(-1, 3)
        for remove_deltaG_zero in (False, True):
            with self.subTest(remove_deltaG_zero=remove_deltaG_zero):
                stacked = XNETBStackedSingularity(
                    parameters=self.coeffs, deltaGs=deltaGs, remove_deltaG_zero=remove_deltaG_zero)
                qmesh = XNETBStackedSingularityQMesh(
                    qGrid=qGrid, cell=self.cell, deltaGs=deltaGs, remove_deltaG_zero=remove_deltaG_zero)
                qmesh.set_parameters(self.coeffs)
                np.testing.assert_allclose(
                    qmesh.eval_model(coords), stacked.eval_model(coords), rtol=1e-10, atol=1e-14)

    def test_qmesh_initial_parameters_match_gaussian(self):
        qGrid = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, self.cell.reciprocal_lattice_constant / 2]])
        gauss = XNGaussStackedSingularityQMesh(qGrid=qGrid, cell=self.cell, deltaGs=self.deltaGs)
        gauss.set_parameters(gauss.initial_parameters(1e-4))
        etb = XNETBStackedSingularityQMesh(qGrid=qGrid, cell=self.cell, deltaGs=self.deltaGs)
        etb.set_parameters(etb.one_hot_reference(etb.initial_parameters(1e-4)[etb.etb_ref_index]))
        np.testing.assert_allclose(etb.eval_model(self.coords), gauss.eval_model(self.coords), rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
