import unittest
from unittest import mock
import warnings

import numpy as np

from pyscf.pbc import df, gto, mp, scf
from pyscf.pbc.lib import kpts_helper

from fsec.singularity_subtraction.grids import minimum_image
from fsec.singularity_subtraction.mp2ss import MP2SS, MP2SSOptions
from fsec.singularity_subtraction.structure_factor.mp2_smallq import (
    MP2SmallQ,
    _TRSCDERIArray,
    map_kpts,
    required_lov_pairs,
    s2_pair_closure,
    trs_pair_representatives,
)
from fsec.singularity_subtraction.structure_factor.mp2_sf import (
    MP2StructureFactor,
)


class MP2SmallQOptionsTests(unittest.TestCase):
    def test_relative_shift_validation(self):
        invalid = (
            (0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
            (np.nan, 0.0, 0.0),
            (np.inf, 0.0, 0.0),
            (-0.500001, 0.0, 0.0),
            (0.0, 0.500001, 0.0),
        )
        for shift in invalid:
            with self.subTest(shift=shift):
                with self.assertRaises(ValueError):
                    MP2SSOptions(
                        smallq_band_df="FFTDF",
                        smallq_relative_shift=shift,
                    )

        with self.assertRaisesRegex(ValueError, "all-zero"):
            MP2SSOptions(
                smallq_band_df="FFTDF",
                smallq_relative_shift=(0.0, 0.0, 0.0),
            )
        with self.assertRaisesRegex(ValueError, "all-zero"):
            MP2SmallQ(
                mock.Mock(cell=mock.Mock()),
                mock.Mock(),
                band_df="FFTDF",
                relative_shift=(0.0, 0.0, 0.0),
            )
        # Validation still applies when small-q is disabled, but the zero
        # vector remains representable because it is never consumed.
        self.assertEqual(
            MP2SSOptions(
                smallq_band_df=None,
                smallq_relative_shift=(0.0, 0.0, 0.0),
            ).smallq_relative_shift,
            (0.0, 0.0, 0.0),
        )

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

    def test_mixed_shift_builds_distinct_periodic_grids(self):
        calculation = MP2SmallQ(
            self.kmf,
            self.kmp,
            band_df="FFTDF",
            relative_shift=(0.25, 0.0, -0.5),
            N_local=self.N_local,
        )
        qprime, grids = calculation._build_grids()

        np.testing.assert_allclose(
            self.cell.get_scaled_kpts(qprime),
            (0.25, 0.0, -0.5),
            atol=1e-12,
        )
        self.assertTrue(grids.kGrid3_neq_kGrid2)
        ka_map = map_kpts(
            self.cell,
            grids.kGrid1 + qprime,
            grids.kGrid2,
            "ka",
        )
        kb_map = map_kpts(
            self.cell,
            grids.kGrid1 - qprime,
            grids.kGrid3,
            "kb",
        )
        np.testing.assert_allclose(
            minimum_image(self.cell, grids.kGrid1 + qprime),
            grids.kGrid2[ka_map],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            minimum_image(self.cell, grids.kGrid1 - qprime),
            grids.kGrid3[kb_map],
            atol=1e-12,
        )
        self.assertFalse(np.allclose(grids.kGrid2, grids.kGrid3))

        tiny_shift = MP2SmallQ(
            self.kmf,
            self.kmp,
            band_df="FFTDF",
            relative_shift=(1e-10, 0.0, 0.0),
            N_local=self.N_local,
        )
        _, tiny_grids = tiny_shift._build_grids()
        self.assertTrue(tiny_grids.kGrid3_neq_kGrid2)
        self.assertFalse(np.array_equal(tiny_grids.kGrid1, tiny_grids.kGrid2))

    def test_negative_half_shift_keeps_one_shifted_grid(self):
        calculation = MP2SmallQ(
            self.kmf,
            self.kmp,
            band_df="FFTDF",
            relative_shift=(-0.5, -0.5, -0.5),
            N_local=self.N_local,
        )
        qprime, grids = calculation._build_grids()
        np.testing.assert_allclose(
            self.cell.get_scaled_kpts(qprime),
            (-0.5, -0.5, -0.5),
            atol=1e-12,
        )
        self.assertFalse(grids.kGrid3_neq_kGrid2)
        np.testing.assert_allclose(grids.kGrid2, grids.kGrid3, atol=1e-12)

    def test_distinct_shift_recomputes_and_propagates_all_band_blocks(self):
        calculation = MP2SmallQ(
            self.kmf,
            self.kmp,
            band_df="FFTDF",
            band_exxdiv="ewald",
            relative_shift=(0.01, 0.0, 0.0),
            N_local=(3, 3, 3),
            pair_density_eval_grid="uniform",
        )
        band_calls = []
        base_energy = [np.asarray(e) for e in self.kmp.mo_energy]
        base_coeff = [np.asarray(c) for c in self.kmp.mo_coeff]

        fake_structure_factor = mock.MagicMock()
        fake_structure_factor.build_structure_factor.return_value = {
            "SqG_full_direct": np.asarray([-1.0]),
            "SqG_full_q4": np.asarray([-2.0]),
        }

        def fake_bands(kpts, profile=None):
            band_calls.append(np.asarray(kpts).copy())
            energies = []
            coeffs = []
            for index in range(len(kpts)):
                # Give every returned point a distinguishable value so the
                # ordered ki/ka/kb split is observable below.
                energies.append(base_energy[index % len(base_energy)] + index)
                coeffs.append(base_coeff[index % len(base_coeff)] + index)
            return energies, coeffs

        with (
            mock.patch.object(
                calculation, "_get_shifted_bands", side_effect=fake_bands
            ),
            mock.patch.object(
                calculation, "_build_lov", return_value=(None, None)
            ) as build_lov,
            mock.patch(
                "fsec.singularity_subtraction.structure_factor.mp2_smallq."
                "MP2StructureFactor",
                return_value=fake_structure_factor,
            ),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            result = calculation.kernel()

        runtime_warnings = [
            warning for warning in caught
            if warning.category is RuntimeWarning
        ]
        self.assertEqual(len(runtime_warnings), 1)
        self.assertEqual(len(band_calls), 1)
        nkpts = len(self.kmf.kpts)
        self.assertEqual(len(band_calls[0]), 3 * nkpts)
        _, grids = calculation._build_grids()
        np.testing.assert_allclose(
            band_calls[0],
            np.concatenate(
                (grids.kGrid1, grids.kGrid2, grids.kGrid3), axis=0
            ),
        )
        build_kwargs = fake_structure_factor.build_structure_factor.call_args.kwargs
        np.testing.assert_allclose(
            build_kwargs["mo_energy"][0],
            base_energy[0],
        )
        np.testing.assert_allclose(
            build_kwargs["mo_coeff_kpts1"][0],
            base_coeff[0],
        )
        np.testing.assert_allclose(
            build_kwargs["mo_coeff_kpts2"][0],
            base_coeff[0] + nkpts,
        )
        np.testing.assert_allclose(
            build_kwargs["mo_coeff_kpts3"][0],
            base_coeff[0] + 2 * nkpts,
        )
        np.testing.assert_allclose(
            build_kwargs["mo_e_v"][0],
            (base_energy[0] + nkpts)[self.kmp.nocc:],
        )
        np.testing.assert_allclose(
            build_kwargs["mo_e_v_b"][0],
            (base_energy[0] + 2 * nkpts)[self.kmp.nocc:],
        )
        self.assertIsNot(
            build_kwargs["mo_coeff_kpts2"], build_kwargs["mo_coeff_kpts3"]
        )
        self.assertIsNot(build_kwargs["mo_e_v"], build_kwargs["mo_e_v_b"])
        lov_args = build_lov.call_args.args
        np.testing.assert_allclose(lov_args[2][0], base_coeff[0])
        np.testing.assert_allclose(lov_args[3][0], base_coeff[0] + nkpts)
        np.testing.assert_allclose(
            build_lov.call_args.kwargs["mo_coeff_shifted_b"][0],
            base_coeff[0] + 2 * nkpts,
        )
        self.assertEqual(result.qprime.shape, (3,))

        reciprocal_norm = np.linalg.norm(self.cell.reciprocal_vectors()[0])
        above_threshold = MP2SmallQ(
            self.kmf,
            self.kmp,
            band_df="FFTDF",
            band_exxdiv="ewald",
            relative_shift=(0.1001 / reciprocal_norm, 0.0, 0.0),
            N_local=(3, 3, 3),
            pair_density_eval_grid="uniform",
        )
        with (
            mock.patch.object(
                above_threshold, "_get_shifted_bands", side_effect=fake_bands
            ),
            mock.patch.object(
                above_threshold, "_build_lov", return_value=(None, None)
            ),
            mock.patch(
                "fsec.singularity_subtraction.structure_factor.mp2_smallq."
                "MP2StructureFactor",
                return_value=fake_structure_factor,
            ),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            above_threshold.kernel()
        self.assertFalse(
            any(warning.category is RuntimeWarning for warning in caught)
        )

    def test_self_inverse_shift_reuses_shifted_band_block(self):
        calculation = MP2SmallQ(
            self.kmf,
            self.kmp,
            band_df="FFTDF",
            relative_shift=(-0.5, -0.5, -0.5),
            N_local=(3, 3, 3),
            pair_density_eval_grid="uniform",
            check_trs=False,
        )
        band_calls = []
        base_energy = [np.asarray(e) for e in self.kmp.mo_energy]
        base_coeff = [np.asarray(c) for c in self.kmp.mo_coeff]
        fake_structure_factor = mock.MagicMock()
        fake_structure_factor.build_structure_factor.return_value = {
            "SqG_full_direct": np.asarray([-1.0]),
            "SqG_full_q4": np.asarray([-2.0]),
        }

        def fake_bands(kpts, profile=None):
            band_calls.append(np.asarray(kpts).copy())
            energies = [
                base_energy[index % len(base_energy)] + index
                for index in range(len(kpts))
            ]
            coeffs = [
                base_coeff[index % len(base_coeff)] + index
                for index in range(len(kpts))
            ]
            return energies, coeffs

        with (
            mock.patch.object(
                calculation, "_get_shifted_bands", side_effect=fake_bands
            ),
            mock.patch.object(
                calculation, "_build_lov", return_value=(None, None)
            ),
            mock.patch(
                "fsec.singularity_subtraction.structure_factor.mp2_smallq."
                "MP2StructureFactor",
                return_value=fake_structure_factor,
            ),
        ):
            calculation.kernel()

        nkpts = len(self.kmf.kpts)
        self.assertEqual(len(band_calls), 1)
        self.assertEqual(len(band_calls[0]), 2 * nkpts)
        _, grids = calculation._build_grids()
        np.testing.assert_allclose(
            band_calls[0],
            np.concatenate((grids.kGrid1, grids.kGrid2), axis=0),
        )
        build_kwargs = fake_structure_factor.build_structure_factor.call_args.kwargs
        self.assertIs(
            build_kwargs["mo_coeff_kpts2"],
            build_kwargs["mo_coeff_kpts3"],
        )
        np.testing.assert_allclose(
            build_kwargs["mo_coeff_kpts2"][0],
            base_coeff[0] + nkpts,
        )
        np.testing.assert_allclose(
            build_kwargs["mo_coeff_kpts3"][0],
            base_coeff[0] + nkpts,
        )

    def test_recomputed_ki_bands_ignore_stale_kmp_inputs(self):
        calculation = MP2SmallQ(
            self.kmf,
            self.kmp,
            band_df="FFTDF",
            N_local=(3, 3, 3),
            pair_density_eval_grid="uniform",
            check_trs=False,
        )
        original_energy = self.kmp.mo_energy
        original_coeff = self.kmp.mo_coeff
        stale_energy = [np.asarray(energy) + 100.0 for energy in original_energy]
        stale_coeff = [np.asarray(coeff) + 100.0 for coeff in original_coeff]
        pure_energy = [np.asarray(energy) - 10.0 for energy in original_energy]
        pure_coeff = [np.asarray(coeff) - 10.0 for coeff in original_coeff]
        fake_structure_factor = mock.MagicMock()
        fake_structure_factor.build_structure_factor.return_value = {
            "SqG_full_direct": np.asarray([-1.0]),
            "SqG_full_q4": np.asarray([-2.0]),
        }

        def fake_bands(kpts, profile=None):
            npoints = len(kpts)
            return (
                [pure_energy[index % len(pure_energy)] for index in range(npoints)],
                [pure_coeff[index % len(pure_coeff)] for index in range(npoints)],
            )

        self.kmp.mo_energy = stale_energy
        self.kmp.mo_coeff = stale_coeff
        try:
            with (
                mock.patch.object(
                    calculation, "_get_shifted_bands", side_effect=fake_bands
                ),
                mock.patch.object(
                    calculation, "_build_lov", return_value=(None, None)
                ),
                mock.patch(
                    "fsec.singularity_subtraction.structure_factor.mp2_smallq."
                    "MP2StructureFactor",
                    return_value=fake_structure_factor,
                ),
            ):
                calculation.kernel()
        finally:
            self.kmp.mo_energy = original_energy
            self.kmp.mo_coeff = original_coeff

        build_kwargs = fake_structure_factor.build_structure_factor.call_args.kwargs
        np.testing.assert_allclose(
            build_kwargs["mo_energy"][0], pure_energy[0]
        )
        np.testing.assert_allclose(
            build_kwargs["mo_coeff_kpts1"][0], pure_coeff[0]
        )
        self.assertFalse(
            np.allclose(build_kwargs["mo_energy"][0], stale_energy[0])
        )

    def test_band_block_count_mismatch_is_rejected(self):
        calculation = MP2SmallQ(
            self.kmf,
            self.kmp,
            band_df="FFTDF",
            relative_shift=(0.25, 0.0, -0.5),
            N_local=(3, 3, 3),
        )
        _, grids = calculation._build_grids()
        with (
            mock.patch.object(
                calculation,
                "_get_shifted_bands",
                return_value=([self.kmp.mo_energy[0]], []),
            ),
            self.assertRaisesRegex(RuntimeError, "expected 3"),
        ):
            calculation._get_band_blocks(grids)

    def test_rejects_unsupported_correlation_df(self):
        original_df = self.kmf.with_df
        try:
            self.kmf.with_df = mock.Mock()
            calculation = MP2SmallQ(
                self.kmf,
                self.kmp,
                band_df="FFTDF",
                band_exxdiv="vcut_sph",
                relative_shift=(0.25, 0.0, -0.5),
                N_local=(3, 3, 3),
                pair_density_eval_grid="uniform",
            )
            with self.assertRaisesRegex(NotImplementedError, "FFTDF or GDF"):
                calculation.kernel()
        finally:
            self.kmf.with_df = original_df

    def test_fftdf_and_gdf_reference_values(self):
        references = {
            "FFTDF": (-1.4880756934176656e-4, -8.580232969749587e-7),
            "GDF": (-1.4399182726078822e-4, -8.302497872716984e-7),
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

    def test_mixed_shift_fftdf_and_gdf_direct_q4_are_tr_consistent(self):
        for backend in ("FFTDF", "GDF"):
            with self.subTest(backend=backend):
                common = dict(
                    band_df=backend,
                    band_exxdiv="ewald",
                    relative_shift=(0.25, 0.0, -0.5),
                    N_local=(5, 5, 5),
                    pair_density_eval_grid="uniform",
                )
                without_tr = MP2SmallQ(
                    self.kmf,
                    self.kmp,
                    check_trs=False,
                    **common,
                ).kernel()
                with_tr = MP2SmallQ(
                    self.kmf,
                    self.kmp,
                    check_trs=True,
                    **common,
                ).kernel()
                for field in ("sq_direct", "sq_q4"):
                    self.assertTrue(np.isfinite(getattr(without_tr, field)))
                    self.assertTrue(np.isfinite(getattr(with_tr, field)))
                    self.assertAlmostEqual(
                        getattr(without_tr, field),
                        getattr(with_tr, field),
                        places=12,
                    )

    def test_gdf_smallq_uses_supplied_lov_when_flag_is_disabled(self):
        original_with_df_ints = self.kmp.with_df_ints
        self.kmp.with_df_ints = False
        try:
            result = MP2SmallQ(
                self.kmf,
                self.kmp,
                band_df="FFTDF",
                band_exxdiv="vcut_sph",
                relative_shift=(0.25, 0.0, -0.5),
                N_local=(5, 5, 5),
                pair_density_eval_grid="uniform",
            ).kernel()
        finally:
            self.kmp.with_df_ints = original_with_df_ints

        self.assertAlmostEqual(
            result.sq_direct, -4.2940956150106673e-4, delta=1e-10
        )
        self.assertAlmostEqual(
            result.sq_q4, -3.8091174426527914e-6, delta=1e-11
        )
        self.assertEqual(self.kmp.with_df_ints, original_with_df_ints)

    def test_fftdf_correlation_uses_direct_ao2mo(self):
        original_df = self.kmf.with_df
        original_with_df_ints = self.kmp.with_df_ints
        self.kmf.with_df = df.FFTDF(self.cell, self.kmf.kpts)
        self.kmp.with_df_ints = False
        try:
            result = MP2SmallQ(
                self.kmf,
                self.kmp,
                band_df="FFTDF",
                band_exxdiv="vcut_sph",
                relative_shift=(0.25, 0.0, -0.5),
                N_local=(5, 5, 5),
                pair_density_eval_grid="uniform",
            ).kernel()
        finally:
            self.kmf.with_df = original_df
            self.kmp.with_df_ints = original_with_df_ints

        self.assertAlmostEqual(
            result.sq_direct, -4.474443278105192e-4, delta=1e-10
        )
        self.assertAlmostEqual(
            result.sq_q4, -3.8091174426527935e-6, delta=1e-11
        )

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

        with (
            mock.patch.object(
                self.kmf,
                "get_bands",
                side_effect=RuntimeError("band diagonalization failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "diagonalization failed"),
        ):
            calculation._get_shifted_bands(grids.kGrid2)
        self.assertIs(self.kmf.with_df, original_df)
        self.assertEqual(self.kmf.exxdiv, original_exxdiv)

    def test_temporary_gdf_closes_when_band_diagonalization_fails(self):
        calculation = self._smallq("GDF", band_exxdiv="ewald")
        _, grids = calculation._build_grids()
        temporary_df = df.GDF(self.cell, self.kmf.kpts)
        temporary_handle = mock.Mock()
        temporary_df._cderi_to_save = temporary_handle
        original_df = self.kmf.with_df
        original_exxdiv = self.kmf.exxdiv

        with (
            mock.patch.object(
                calculation, "_make_band_df", return_value=temporary_df
            ),
            mock.patch.object(
                self.kmf,
                "get_bands",
                side_effect=RuntimeError("band diagonalization failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "diagonalization failed"),
        ):
            calculation._get_shifted_bands(grids.kGrid2)

        temporary_handle.close.assert_called_once_with()
        self.assertIs(self.kmf.with_df, original_df)
        self.assertEqual(self.kmf.exxdiv, original_exxdiv)

    def test_same_df_ki_occupied_virtual_blocks_are_orthogonal(self):
        calculation = self._smallq("FFTDF", band_exxdiv="vcut_sph")
        _, grids = calculation._build_grids()
        _, band_coeff = calculation._get_shifted_bands(grids.kGrid1)
        overlap = self.kmf.get_ovlp(self.cell, grids.kGrid1)
        for kindex, coeff in enumerate(band_coeff):
            occupied = coeff[:, :self.kmp.nocc]
            virtual = coeff[:, self.kmp.nocc:]
            cross_overlap = occupied.conj().T @ overlap[kindex] @ virtual
            np.testing.assert_allclose(cross_overlap, 0.0, atol=1e-10)

    def test_small_shift_raw_structure_factors_decrease(self):
        results = []
        for shift in (0.01, 0.005):
            results.append(MP2SmallQ(
                self.kmf,
                self.kmp,
                band_df="FFTDF",
                band_exxdiv="vcut_sph",
                relative_shift=(shift, shift, shift),
                N_local=self.N_local,
                pair_density_eval_grid="uniform",
                check_trs=False,
            ).kernel())

        for result in results:
            self.assertTrue(np.isfinite(result.sq_direct))
            self.assertTrue(np.isfinite(result.sq_q4))
        self.assertLess(abs(results[1].sq_direct), abs(results[0].sq_direct))
        self.assertLess(abs(results[1].sq_q4), abs(results[0].sq_q4))

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

                ka_map = map_kpts(
                    self.cell,
                    grids.kGrid1 + qprime,
                    grids.kGrid2,
                    "ka",
                )
                kb_map = map_kpts(
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
                trs_occ = kpts_helper.conj_mapping(
                    self.cell,
                    grids.kGrid1,
                )
                trs_shifted = kpts_helper.conj_mapping(
                    self.cell,
                    grids.kGrid2,
                )
                np.testing.assert_array_equal(
                    trs_shifted[ka_map[trs_occ]],
                    kb_map,
                )

    def test_required_lov_pairs_are_exact_and_unique(self):
        pairs = required_lov_pairs(
            ka_map=np.array([0, 1]),
            kb_map=np.array([1, 0]),
            nkpts=2,
        )
        self.assertEqual(
            pairs,
            [(0, 2), (1, 3), (0, 3), (1, 2)],
        )
        self.assertEqual(
            required_lov_pairs(
                ka_map=np.array([0]),
                kb_map=np.array([0]),
                nkpts=1,
            ),
            [(0, 1)],
        )

    def test_trs_s2_pair_representatives(self):
        requested = [(0, 2), (1, 3), (0, 3), (1, 2)]
        closure = s2_pair_closure(requested)
        self.assertEqual(
            closure,
            [
                (0, 2), (1, 3), (0, 3), (1, 2),
                (2, 0), (3, 1), (3, 0), (2, 1),
            ],
        )
        evaluated = trs_pair_representatives(
            closure,
            np.array([0, 1, 3, 2]),
        )
        self.assertEqual(
            evaluated,
            [(0, 2), (1, 2), (2, 0), (2, 1)],
        )
        self.assertEqual(
            trs_pair_representatives(
                [(0, 1), (1, 0)],
                np.array([0, 1]),
            ),
            [(0, 1), (1, 0)],
        )

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

    def test_mp2ss_passes_smallq_relative_shift(self):
        options = MP2SSOptions(
            smallq_band_df="FFTDF",
            smallq_relative_shift=(0.25, 0.0, -0.5),
            t2_store_type="ki",
            pair_density_eval_grid="uniform",
        )
        fake_structure_factor = mock.MagicMock()
        fake_structure_factor.grids.qGrid = np.asarray(self.kmf.kpts)
        fake_smallq = mock.MagicMock()
        fake_smallq.kernel.return_value = "smallq-result"

        with (
            mock.patch(
                "fsec.singularity_subtraction.mp2ss.MP2StructureFactor",
                return_value=fake_structure_factor,
            ),
            mock.patch(
                "fsec.singularity_subtraction.mp2ss.MP2SmallQ",
                return_value=fake_smallq,
            ) as smallq_class,
        ):
            mp2ss = MP2SS(self.kmf, self.kmp, options=options)
            mp2ss.set_structure_factor(
                direct=True,
                exchange=False,
                dG0=False,
            )

        self.assertEqual(mp2ss.smallq_result, "smallq-result")
        self.assertEqual(
            smallq_class.call_args.kwargs["relative_shift"],
            (0.25, 0.0, -0.5),
        )


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

    def test_selective_trs_cderi_matches_full_gdf(self):
        calculation = MP2SmallQ(
            self.kmf,
            self.kmp,
            band_df="GDF",
            band_exxdiv="ewald",
            N_local=self.N_local,
            pair_density_eval_grid="uniform",
        )
        qprime, grids = calculation._build_grids()
        nkpts = len(grids.kGrid1)
        ka_map = map_kpts(
            self.cell,
            grids.kGrid1 + qprime,
            grids.kGrid2,
            "ka",
        )
        kb_map = map_kpts(
            self.cell,
            grids.kGrid1 - qprime,
            grids.kGrid3,
            "kb",
        )
        # Identify the original-to-shifted k-point pairs needed by Lov and
        # include their reverses so the s2 AO blocks can be reconstructed.
        requested = required_lov_pairs(
            ka_map,
            kb_map,
            nkpts,
        )
        closure = s2_pair_closure(requested)
        trs_occ = kpts_helper.conj_mapping(
            self.cell,
            grids.kGrid1,
        )
        trs_shifted = kpts_helper.conj_mapping(
            self.cell,
            grids.kGrid2,
        )
        combined_trs = np.concatenate(
            (trs_occ, nkpts + trs_shifted)
        )
        # Build only one transpose-closed representative set; its omitted
        # time-reversed partners will be recovered by complex conjugation.
        evaluated = trs_pair_representatives(
            closure,
            combined_trs,
        )
        combined_kpts = np.concatenate(
            (grids.kGrid1, grids.kGrid2),
            axis=0,
        )

        selective_df = calculation._make_correlation_gdf(
            combined_kpts,
            evaluated,
        )
        selective = _TRSCDERIArray(
            selective_df._cderi,
            evaluated,
            combined_trs,
        )

        # Use an otherwise identical, all-k-point-pair GDF build as the
        # numerical reference for the selective construction.
        source_df = self.kmf.with_df
        full_df = df.GDF(self.cell, combined_kpts)
        full_df.auxbasis = source_df.auxbasis
        if source_df.mesh is not None:
            full_df.mesh = np.asarray(source_df.mesh).copy()
        full_df.linear_dep_threshold = source_df.linear_dep_threshold
        full_df.exp_to_discard = source_df.exp_to_discard
        full_df._prefer_ccdf = source_df._prefer_ccdf
        full_df.build(j_only=False)
        full = _TRSCDERIArray(
            full_df._cderi,
            [
                (ki, kj)
                for ki in range(len(combined_kpts))
                for kj in range(len(combined_kpts))
            ],
            np.arange(len(combined_kpts)),
        )

        try:
            self.assertEqual(selective.aosym, "s2")
            # Confirm that the optimization stored exactly the selected
            # representatives, rather than silently building every pair.
            stored_keys = {
                int(key) for key in selective._array.j3c.keys()
            }
            self.assertEqual(
                stored_keys,
                {
                    ki * len(combined_kpts) + kj
                    for ki, kj in evaluated
                },
            )
            # Ensure the comparison below exercises at least one CDERI block
            # reconstructed through time-reversal symmetry.
            self.assertTrue(
                any(pair not in set(evaluated) for pair in requested)
            )
            for pair in requested:
                with self.subTest(pair=pair):
                    np.testing.assert_allclose(
                        selective[pair],
                        full[pair],
                        rtol=1e-10,
                        atol=1e-10,
                    )
        finally:
            selective.close()
            full.close()
            for gdf_object in (selective_df, full_df):
                cderi_temp = gdf_object._cderi_to_save
                if not isinstance(cderi_temp, str):
                    cderi_temp.close()

    def test_selective_three_grid_cderi_matches_full_gdf(self):
        calculation = MP2SmallQ(
            self.kmf,
            self.kmp,
            band_df="GDF",
            band_exxdiv="ewald",
            relative_shift=(0.25, 0.0, -0.5),
            N_local=self.N_local,
            pair_density_eval_grid="uniform",
        )
        qprime, grids = calculation._build_grids()
        nkpts = len(grids.kGrid1)
        ka_map = map_kpts(
            self.cell,
            grids.kGrid1 + qprime,
            grids.kGrid2,
            "ka",
        )
        kb_map = map_kpts(
            self.cell,
            grids.kGrid1 - qprime,
            grids.kGrid3,
            "kb",
        )
        combined_kpts = np.concatenate(
            (grids.kGrid1, grids.kGrid2, grids.kGrid3), axis=0
        )
        requested = required_lov_pairs(
            ka_map,
            kb_map,
            nkpts,
            ka_offset=nkpts,
            kb_offset=2 * nkpts,
        )
        closure = s2_pair_closure(requested)
        combined_trs = kpts_helper.conj_mapping(self.cell, combined_kpts)
        evaluated = trs_pair_representatives(closure, combined_trs)

        selective_df = calculation._make_correlation_gdf(
            combined_kpts,
            evaluated,
        )
        selective = _TRSCDERIArray(
            selective_df._cderi,
            evaluated,
            combined_trs,
        )

        source_df = self.kmf.with_df
        full_df = df.GDF(self.cell, combined_kpts)
        full_df.auxbasis = source_df.auxbasis
        if source_df.mesh is not None:
            full_df.mesh = np.asarray(source_df.mesh).copy()
        full_df.linear_dep_threshold = source_df.linear_dep_threshold
        full_df.exp_to_discard = source_df.exp_to_discard
        full_df._prefer_ccdf = source_df._prefer_ccdf
        full_df.build(j_only=False)
        full = _TRSCDERIArray(
            full_df._cderi,
            [
                (ki, kj)
                for ki in range(len(combined_kpts))
                for kj in range(len(combined_kpts))
            ],
            np.arange(len(combined_kpts)),
        )

        try:
            self.assertEqual(selective.aosym, "s2")
            stored_keys = {int(key) for key in selective._array.j3c.keys()}
            self.assertEqual(
                stored_keys,
                {
                    ki * len(combined_kpts) + kj
                    for ki, kj in evaluated
                },
            )
            for pair in closure:
                with self.subTest(pair=pair):
                    np.testing.assert_allclose(
                        selective[pair],
                        full[pair],
                        rtol=1e-10,
                        atol=1e-10,
                    )
        finally:
            selective.close()
            full.close()
            for gdf_object in (selective_df, full_df):
                cderi_temp = gdf_object._cderi_to_save
                if not isinstance(cderi_temp, str):
                    cderi_temp.close()

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
