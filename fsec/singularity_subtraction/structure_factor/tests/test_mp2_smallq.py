import unittest
from unittest import mock

import numpy as np

from pyscf.pbc import df, gto, mp, scf

from fsec.singularity_subtraction.grids import minimum_image
from fsec.singularity_subtraction.mp2ss import MP2SS, MP2SSOptions
from fsec.singularity_subtraction.structure_factor.mp2_smallq import (
    MP2SmallQ,
)
from fsec.singularity_subtraction.structure_factor.mp2_sf import (
    MP2StructureFactor,
)


class MP2SmallQOptionsTests(unittest.TestCase):
    def test_band_df_normalization_and_exxdiv_validation(self):
        fft_options = MP2SSOptions(
            smallq_band_df=" fftdf ",
            smallq_band_exxdiv=" VCUT_SPH ",
        )
        self.assertEqual(fft_options.smallq_band_df, "FFTDF")
        self.assertEqual(fft_options.smallq_band_exxdiv, "vcut_sph")

        self.assertEqual(
            MP2SSOptions(smallq_band_df="gdf").smallq_band_df,
            "GDF",
        )
        self.assertIsNone(
            MP2SSOptions(
                smallq_band_df="GDF",
                smallq_band_exxdiv=None,
            ).smallq_band_exxdiv
        )
        with self.assertRaisesRegex(ValueError, "GDF bands only support"):
            MP2SSOptions(
                smallq_band_df="GDF",
                smallq_band_exxdiv="vcut_sph",
            )
        with self.assertRaisesRegex(ValueError, "smallq_band_df"):
            MP2SSOptions(smallq_band_df="MDF")


class MP2SmallQKnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cell = gto.Cell()
        cell.unit = "Bohr"
        cell.atom = """
            H 0.00 0.00 0.00
            H 0.00 0.00 1.80
        """
        cell.a = np.eye(3) * 6.0
        cell.basis = "gth-szv"
        cell.pseudo = "gth-hf"
        cell.mesh = [9, 9, 9]
        cell.precision = 1e-7
        cell.verbose = 0
        cell.build()

        kpts = cell.make_kpts(
            (1, 1, 1),
            wrap_around=True,
            with_gamma_point=True,
        )
        kmf = scf.KRHF(cell, kpts, exxdiv="ewald")
        kmf.with_df = df.GDF(cell, kpts).build()
        kmf.conv_tol = 1e-9
        kmf.kernel()
        if not kmf.converged:
            raise RuntimeError("Small-q H2 reference KRHF did not converge")

        kmp = mp.KMP2(kmf)
        kmp.with_df_ints = True
        kmp.kernel(with_t2=False)

        cls.cell = cell
        cls.kmf = kmf
        cls.kmp = kmp
        cls.N_local = np.array([9, 9, 9])

    def _smallq(self, band_df, band_exxdiv="ewald"):
        return MP2SmallQ(
            self.kmf,
            self.kmp,
            band_df=band_df,
            band_exxdiv=band_exxdiv,
            N_local=self.N_local,
            pair_density_eval_grid="uniform",
            check_trs=False,
        )

    def test_fftdf_and_gdf_reference_values(self):
        references = {
            "FFTDF": (-1.5455616988656258e-4, -8.910009779750352e-7),
            "GDF": (-1.439918272607873e-4, -8.302497872716938e-7),
        }
        for backend, (sq_direct, sq_q4) in references.items():
            with self.subTest(backend=backend):
                result = self._smallq(backend).kernel()
                self.assertAlmostEqual(
                    result.sq_direct, sq_direct, delta=1e-10
                )
                self.assertAlmostEqual(result.sq_q4, sq_q4, delta=1e-11)
                self.assertEqual(result.band_df, backend)
                self.assertEqual(result.band_exxdiv, "ewald")

    def test_fftdf_band_exxdiv_is_temporary(self):
        original_df = self.kmf.with_df
        original_exxdiv = self.kmf.exxdiv
        calculation = self._smallq("FFTDF", band_exxdiv="vcut_sph")
        _, grids = calculation._build_grids()
        observed = []
        get_bands = self.kmf.get_bands

        def record_environment(kpts):
            observed.append((
                self.kmf.exxdiv,
                type(self.kmf.with_df),
            ))
            return get_bands(kpts)

        with mock.patch.object(
            self.kmf,
            "get_bands",
            side_effect=record_environment,
        ):
            calculation._get_shifted_bands(grids.kGrid2)

        self.assertEqual(observed, [("vcut_sph", df.FFTDF)])
        self.assertIs(self.kmf.with_df, original_df)
        self.assertEqual(self.kmf.exxdiv, original_exxdiv)

    def test_positive_half_shift_and_periodic_mappings(self):
        for mesh in ((1, 1, 1), (1, 1, 2), (2, 2, 2)):
            with self.subTest(mesh=mesh):
                kpts = self.cell.make_kpts(
                    mesh,
                    wrap_around=True,
                    with_gamma_point=True,
                )
                kmf = scf.KRHF(self.cell, kpts)
                kmp = mp.KMP2(kmf)
                calculation = MP2SmallQ(
                    kmf,
                    kmp,
                    band_df="FFTDF",
                    N_local=(3, 3, 3),
                )
                qprime, grids = calculation._build_grids()
                scaled_qprime = self.cell.get_scaled_kpts(qprime)
                self.assertTrue(np.allclose(
                    scaled_qprime,
                    0.5 / np.asarray(mesh),
                    atol=1e-12,
                ))

                ka_map = calculation._map_kpts(
                    self.cell,
                    grids.kGrid1 + qprime,
                    grids.kGrid2,
                    "ka",
                )
                kb_map = calculation._map_kpts(
                    self.cell,
                    grids.kGrid1 - qprime,
                    grids.kGrid3,
                    "kb",
                )
                ka_error = minimum_image(
                    self.cell,
                    grids.kGrid1 + qprime,
                ) - grids.kGrid2[ka_map]
                kb_error = minimum_image(
                    self.cell,
                    grids.kGrid1 - qprime,
                ) - grids.kGrid3[kb_map]
                self.assertLess(np.max(np.abs(ka_error)), 1e-10)
                self.assertLess(np.max(np.abs(kb_error)), 1e-10)

    def test_disabled_option_does_not_construct_smallq(self):
        options = MP2SSOptions(
            smallq_band_df=None,
            t2_store_type="ki",
            pair_density_eval_grid="uniform",
        )
        fake_structure_factor = mock.MagicMock()
        fake_structure_factor.grids.qGrid = np.asarray(self.kmf.kpts)

        with (
            mock.patch(
                "fsec.singularity_subtraction.mp2ss.MP2StructureFactor",
                return_value=fake_structure_factor,
            ),
            mock.patch(
                "fsec.singularity_subtraction.mp2ss.MP2SmallQ"
            ) as smallq_class,
        ):
            mp2ss = MP2SS(self.kmf, self.kmp, options=options)
            result = mp2ss.set_structure_factor(
                direct=True,
                exchange=False,
                dG0=False,
            )

        smallq_class.assert_not_called()
        self.assertIsNone(mp2ss.smallq_result)
        self.assertIsNone(result.smallq_result)


class MP2SmallQ112KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cell = gto.Cell()
        cell.unit = "Bohr"
        cell.atom = """
            H 0.00 0.00 0.00
            H 0.00 0.00 1.80
        """
        cell.a = np.eye(3) * 6.0
        cell.spin = 0
        cell.charge = 0
        cell.basis = "gth-szv"
        cell.pseudo = "gth-hf"
        cell.ke_cutoff = 100.0
        cell.precision = 1e-8
        cell.verbose = 0
        cell.build()

        kpts = cell.make_kpts(
            (1, 1, 2),
            wrap_around=True,
            with_gamma_point=True,
        )
        kmf = scf.KRHF(cell, kpts, exxdiv="ewald")
        kmf.with_df = df.GDF(cell, kpts).build()
        kmf.conv_tol = 1e-10
        kmf.kernel()
        if not kmf.converged:
            raise RuntimeError("Small-q 1x1x2 H2 reference KRHF did not converge")

        kmp = mp.KMP2(kmf)
        _, t2 = kmp.kernel(with_t2=True)

        cls.cell = cell
        cls.kmf = kmf
        cls.kmp = kmp
        cls.t2 = t2
        cls.N_local = cell.cutoff_to_mesh(100.0)

    @staticmethod
    def _closest_10_indices(qG):
        qG_norm = np.linalg.norm(qG, axis=1)
        return np.lexsort(
            (qG[:, 2], qG[:, 1], qG[:, 0], qG_norm)
        )[:10]

    def test_build_structure_factor_112_kmesh_with_smallq(self):
        mp2_sf = MP2StructureFactor(
            self.kmf,
            self.kmp,
            t2=self.t2,
            N_local=self.N_local,
            qG_cutoff=8.0,
            min_points=10,
            pair_density_eval_grid="uniform",
        )
        result = mp2_sf.build_structure_factor(
            direct=True,
            exchange=False,
            dG0=True,
        )
        smallq = MP2SmallQ(
            self.kmf,
            self.kmp,
            band_df="GDF",
            band_exxdiv="ewald",
            N_local=self.N_local,
            pair_density_eval_grid="uniform",
        ).kernel()

        idx10 = self._closest_10_indices(result["qG_full"])
        qG_11 = np.vstack((result["qG_full"][idx10], smallq.qprime))
        direct_11 = np.append(
            result["SqG_full_direct"][idx10],
            smallq.sq_direct,
        )
        q4_11 = np.append(
            result["SqG_full_q4"][idx10],
            smallq.sq_q4,
        )

        reference_qG_11 = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -0.5235987755982988],
            [0.0, 0.0, 0.5235987755982988],
            [-1.0471975511965976, 0.0, 0.0],
            [0.0, -1.0471975511965976, 0.0],
            [0.0, 0.0, -1.0471975511965976],
            [0.0, 0.0, 1.0471975511965976],
            [0.0, 1.0471975511965976, 0.0],
            [1.0471975511965976, 0.0, 0.0],
            [-1.0471975511965976, 0.0, -0.5235987755982988],
            [0.5235987755982988, 0.5235987755982988, 0.2617993877991494],
        ]
        reference_direct_11 = [
            -6.031483988853451e-22,
            -0.0001331479670959755,
            -0.0001331479670959755,
            -5.837718619008481e-22,
            -5.83757449261307e-22,
            -0.000152965545669427,
            -0.000152965545669427,
            -5.83757449261307e-22,
            -5.837718619008481e-22,
            -5.043690481546798e-05,
            -1.6729723494295827e-05,
        ]
        reference_q4_11 = [
            -3.82095320889316e-41,
            -8.559020498960629e-07,
            -8.559020498960629e-07,
            -3.579441904733104e-41,
            -3.5792634797576904e-41,
            -2.437964679354982e-06,
            -2.437964679354982e-06,
            -3.5792634797576904e-41,
            -3.579441904733104e-41,
            -1.2343680816144883e-07,
            -3.205983111109113e-08,
        ]

        np.testing.assert_allclose(qG_11, reference_qG_11, atol=1e-12)
        np.testing.assert_allclose(
            direct_11,
            reference_direct_11,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            q4_11,
            reference_q4_11,
            atol=1e-10,
        )


if __name__ == "__main__":
    unittest.main()
