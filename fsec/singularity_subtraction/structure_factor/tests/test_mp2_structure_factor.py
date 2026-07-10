import unittest

import numpy as np

try:
    from pyscf.pbc import gto, scf
    from pyscf.pbc import df, mp
    HAS_PYSCF = True
except ImportError:
    HAS_PYSCF = False

try:
    from fsec.singularity_subtraction.structure_factor.mp2_sf import MP2StructureFactor
    from fsec.singularity_subtraction.grids import ExxSSGrids
    from fsec.singularity_subtraction.structure_factor.helpers_sf import (
        make_line_sampling_decay_state,
        normalize_line_sampling_decay_components,
        should_compute_line_sample,
        update_line_sampling_decay_mask,
    )
    HAS_MP2_IMPORT = True
except ImportError:
    HAS_MP2_IMPORT = False


@unittest.skipUnless(HAS_MP2_IMPORT, "fsec structure_factor deps are required")
class LineSamplingDecayHelpers(unittest.TestCase):
    def test_line_sampling_decay_rejects_unimplemented_components(self):
        with self.assertRaises(NotImplementedError):
            normalize_line_sampling_decay_components(
                ("direct_q2",), supported_components={"direct_q4", "exchange"})

    def test_line_sampling_decay_stops_after_consecutive_below_values(self):
        qG_full = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
        ])
        segments = [{
            "line_index": 0,
            "B_index": 0,
            "step_vector": np.array([1.0, 0.0, 0.0]),
            "q_min": qG_full[1],
            "indices": np.arange(1, len(qG_full)),
        }]
        state = make_line_sampling_decay_state(
            qG_full, segments, min_fraction=0.1, consecutive_below=3, power=4)
        mask = np.zeros(len(qG_full), dtype=bool)

        normalized_values = [1.0, 0.05, 0.2, 0.04, 0.03, 0.02, 0.9]
        for index, normalized_value in enumerate(normalized_values, start=1):
            self.assertTrue(should_compute_line_sample(index, state))
            norm = np.linalg.norm(qG_full[index])
            value = normalized_value * norm**4
            update_line_sampling_decay_mask(
                mask, index, value, norm, state, qGpt=qG_full[index])
            if index == 6:
                break

        self.assertTrue(mask[1])
        self.assertFalse(mask[2])
        self.assertTrue(mask[3])
        self.assertFalse(mask[4])
        self.assertFalse(mask[5])
        self.assertFalse(mask[6])
        self.assertFalse(should_compute_line_sample(7, state))
        self.assertEqual(len(state["stop_events"]), 1)
        stop_event = state["stop_events"][0]
        self.assertEqual(stop_event["B_index"], 0)
        self.assertEqual(stop_event["qG_index"], 6)
        self.assertAlmostEqual(stop_event["qG_norm"], 6.0)
        self.assertTrue(np.allclose(stop_event["qG"], qG_full[6]))

    def test_build_qG_line_sampling_stores_segments(self):
        class DummyCell:
            def reciprocal_vectors(self):
                return np.eye(3)

        grids = ExxSSGrids.__new__(ExxSSGrids)
        grids.cell = DummyCell()
        grids.nks = np.ones(3, dtype=int)
        grids.qG_norm_cutoff = 4.1

        qG_full = ExxSSGrids.build_qG_line_sampling(grids)

        self.assertTrue(np.allclose(qG_full[0], np.zeros(3)))
        self.assertEqual(len(grids.qG_line_sampling_segments), 3)
        self.assertEqual(grids.qG_line_sampling_metadata["origin_index"], 0)
        for segment in grids.qG_line_sampling_segments:
            self.assertEqual(len(segment["indices"]), 3)
            self.assertTrue(np.allclose(qG_full[segment["indices"][0]], segment["q_min"]))


@unittest.skipUnless(HAS_PYSCF and HAS_MP2_IMPORT, "PySCF and fsec structure_factor deps are required")
class KnownValues(unittest.TestCase):
    @classmethod
    def _build_system(cls, kmesh):
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

        kpts = cell.make_kpts(kmesh, wrap_around=True, with_gamma_point=True)
        kmf = scf.KRHF(cell, kpts)
        kmf.exxdiv = "ewald"
        kmf.with_df = df.GDF(cell, kpts).build()
        kmf.conv_tol = 1e-10
        kmf.kernel()
        if not kmf.converged:
            raise RuntimeError("KRHF did not converge for the H2 test system")

        kmp = mp.KMP2(kmf)
        _, t2 = kmp.kernel(with_t2=True)

        if not getattr(kmp, "converged", True):
            raise RuntimeError("KMP2 did not converge for the H2 test system")

        return kmf, kmp, t2

    @classmethod
    def setUpClass(cls):
        cls.kmf, cls.kmp, cls.t2 = cls._build_system((1, 1, 1))
        cls.kmf_112, cls.kmp_112, cls.t2_112 = cls._build_system((1, 1, 2))
        cls.sq_ke_cutoff = 100.0
        cls.qG_cutoff = 8.0
        cls.N_local = cls.kmf.cell.cutoff_to_mesh(cls.sq_ke_cutoff)

    @staticmethod
    def _closest_10_indices(qG):
        qG_norm = np.linalg.norm(qG, axis=1)
        return np.lexsort((qG[:, 2], qG[:, 1], qG[:, 0], qG_norm))[:10]

    def test_build_structure_factor_10_closest_qg_points(self):
        mp2_sf = MP2StructureFactor(
            self.kmf,
            self.kmp,
            t2=self.t2,
            N_local=self.N_local,
            qG_cutoff=self.qG_cutoff,
            min_points=10,
        )
        result = mp2_sf.build_structure_factor(direct=True, exchange=True, dG0=True)

        qG = result["qG_full"]
        SqG_full_direct = result["SqG_full_direct"]
        SqG_full_q4 = result["SqG_full_q4"]
        SqG_full_exchange = result["SqG_full_exchange"]

        self.assertEqual(len(SqG_full_direct), len(qG))
        self.assertEqual(len(SqG_full_q4), len(qG))
        self.assertEqual(len(SqG_full_exchange), len(qG))

        self.assertTrue(np.all(np.isfinite(SqG_full_direct)))
        self.assertTrue(np.all(np.isfinite(SqG_full_q4)))
        self.assertTrue(np.all(np.isfinite(SqG_full_exchange)))
        self.assertTrue(np.isrealobj(SqG_full_direct))
        self.assertTrue(np.isrealobj(SqG_full_q4))
        self.assertTrue(np.isrealobj(SqG_full_exchange))

        idx10 = self._closest_10_indices(qG)
        qG_10 = qG[idx10]
        SqG_direct_10 = SqG_full_direct[idx10]
        SqG_q4_10 = SqG_full_q4[idx10]
        SqG_exchange_10 = SqG_full_exchange[idx10]

        self.assertEqual(len(qG_10), 10)
        self.assertEqual(len(SqG_direct_10), 10)
        self.assertEqual(len(SqG_q4_10), 10)
        self.assertEqual(len(SqG_exchange_10), 10)
        self.assertTrue(np.any(np.linalg.norm(qG_10, axis=1) <= 1e-8))

        reference_SqG_direct_10 = [
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
        ]
        reference_SqG_q4_10 = [
            -7.717470607835845e-41,
            -7.230553951086867e-41,
            -7.230159348266723e-41,
            -4.485211663249219e-06,
            -4.485211663249219e-06,
            -7.230159348266723e-41,
            -7.230553951086867e-41,
            -6.772916892974215e-41,
            -8.454795391735687e-07,
            -8.454795391734801e-07,
        ]
        reference_SqG_exchange_10 = [
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
        ]

        for actual, reference in zip(SqG_direct_10, reference_SqG_direct_10):
            self.assertAlmostEqual(actual, reference, places=8)
        for actual, reference in zip(SqG_q4_10, reference_SqG_q4_10):
            self.assertAlmostEqual(actual, reference, places=8)
        for actual, reference in zip(SqG_exchange_10, reference_SqG_exchange_10):
            self.assertAlmostEqual(actual, reference, places=8)

    def test_ki_t2_store_type_matches_kikjka(self):
        reference_sf = MP2StructureFactor(
            self.kmf,
            self.kmp,
            t2=self.t2,
            N_local=self.N_local,
            qG_cutoff=self.qG_cutoff,
            min_points=10,
            t2_store_type="kikjka",
        )
        reference = reference_sf.build_structure_factor(direct=True, exchange=True, dG0=True)

        ki_sf = MP2StructureFactor(
            self.kmf,
            self.kmp,
            N_local=self.N_local,
            qG_cutoff=self.qG_cutoff,
            min_points=10,
            t2_store_type="ki",
        )
        actual = ki_sf.build_structure_factor(direct=True, exchange=True, dG0=True)

        np.testing.assert_allclose(actual["qG_full"], reference["qG_full"], atol=1e-12)
        np.testing.assert_allclose(
            actual["SqG_full_direct"],
            reference["SqG_full_direct"],
            rtol=1e-7,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            actual["SqG_full_exchange"],
            reference["SqG_full_exchange"],
            rtol=1e-7,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            actual["SqG_full_q4"],
            reference["SqG_full_q4"],
            rtol=1e-7,
            atol=1e-10,
        )

    def test_kikj_on_the_fly_matches_kikjka_with_repeated_qi(self):
        reciprocal = self.kmf.cell.reciprocal_vectors()
        qG_full = np.array([
            [0.0, 0.0, 0.0],
            reciprocal[0],
            2 * reciprocal[0],
        ])

        reference_sf = MP2StructureFactor(
            self.kmf,
            self.kmp,
            t2=self.t2,
            N_local=self.N_local,
            qG_cutoff=self.qG_cutoff,
            min_points=10,
            sq_inversion_symm=False,
            t2_store_type="kikjka",
        )
        reference = reference_sf.build_structure_factor(
            qG_full=qG_full, direct=True, exchange=True, dG0=True)

        kikj_sf = MP2StructureFactor(
            self.kmf,
            self.kmp,
            N_local=self.N_local,
            qG_cutoff=self.qG_cutoff,
            min_points=10,
            sq_inversion_symm=False,
            t2_store_type="kikj",
        )
        actual = kikj_sf.build_structure_factor(
            qG_full=qG_full, direct=True, exchange=True, dG0=True)

        np.testing.assert_allclose(actual["qG_full"], reference["qG_full"], atol=1e-12)
        np.testing.assert_allclose(
            actual["SqG_full_direct"],
            reference["SqG_full_direct"],
            rtol=1e-7,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            actual["SqG_full_exchange"],
            reference["SqG_full_exchange"],
            rtol=1e-7,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            actual["SqG_full_q4"],
            reference["SqG_full_q4"],
            rtol=1e-7,
            atol=1e-10,
        )
        self.assertIn("direct t2 cache hit", kikj_sf.last_build_timings)
        self.assertIn("exchange t2 cache hit", kikj_sf.last_build_timings)

    def test_contract_kikj_dG0_matches_old_expression(self):
        rng = np.random.default_rng(12)
        rijab = rng.normal(size=24) + 1j * rng.normal(size=24)
        eijab = rng.normal(size=(2, 3, 4))
        scale = 0.125

        rijab_ovr_e = rijab * np.sqrt(np.abs(eijab)).ravel()
        rijab_ovr_e = rijab_ovr_e * scale
        reference = -2 * np.einsum('i,i->', rijab_ovr_e, rijab_ovr_e.conj())
        actual = MP2StructureFactor.contract_kikj_dG0(rijab, eijab, scale)

        self.assertAlmostEqual(actual.real, reference.real, places=14)
        self.assertAlmostEqual(actual.imag, reference.imag, places=14)

    def test_line_sampling_q4_decay_mask_can_exclude_points(self):
        mp2_sf = MP2StructureFactor(
            self.kmf,
            self.kmp,
            t2=self.t2,
            N_local=self.N_local,
            qG_cutoff=self.qG_cutoff,
            min_points=10,
            sq_inversion_symm=False,
        )
        mp2_sf.set_grids(min_fit_points=10)
        qG_full = mp2_sf.grids.build_qG_line_sampling()
        result = mp2_sf.build_structure_factor(
            qG_full=qG_full,
            direct=True,
            exchange=True,
            dG0=True,
            line_sampling_decay_min_fraction=10.0,
            line_sampling_decay_consecutive_below=1,
            line_sampling_decay_components=("direct_q4",),
            qG_line_sampling_segments=mp2_sf.grids.qG_line_sampling_segments,
        )

        q4_mask = result["SqG_full_q4_mask"]
        direct_mask = result["SqG_full_direct_mask"]
        self.assertEqual(len(q4_mask), len(result["qG_full"]))
        self.assertEqual(len(direct_mask), len(result["qG_full"]))
        self.assertTrue(q4_mask[0])
        self.assertTrue(direct_mask[0])
        self.assertTrue(np.any(~q4_mask))
        np.testing.assert_array_equal(direct_mask, q4_mask)
        self.assertTrue(np.all(np.isfinite(result["SqG_full_q4"])))
        self.assertIn("line_sampling_decay_events", result)

    def test_build_structure_factor_112_kmesh(self):
        mp2_sf = MP2StructureFactor(
            self.kmf_112,
            self.kmp_112,
            t2=self.t2_112,
            N_local=self.N_local,
            qG_cutoff=self.qG_cutoff,
            min_points=10,
        )
        result = mp2_sf.build_structure_factor(direct=True, exchange=True, dG0=True)

        qG = result["qG_full"]
        idx10 = self._closest_10_indices(qG)
        reference_qG_10 = [
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
        ]
        reference_direct_10 = [
            -7.0303133198968355e-22,
            -0.0001194903276263542,
            -0.0001194903276263542,
            -6.781821143014501e-22,
            -6.762184510329533e-22,
            -0.0001765252081289475,
            -0.0001765252081289475,
            -6.762184510329533e-22,
            -6.781821143014501e-22,
            -4.5148932980442695e-05,
        ]
        reference_q4_10 = [
            -3.8888130645417575e-41,
            -8.559020493472634e-07,
            -8.559020493472634e-07,
            -3.6272620975584263e-41,
            -3.6053641934281435e-41,
            -2.4379646780045767e-06,
            -2.4379646780045767e-06,
            -3.6053641934281435e-41,
            -3.6272620975584263e-41,
            -1.234368080915971e-07,
        ]
        reference_exchange_10 = [
            3.515156774224535e-22,
            5.9745163813177126e-05,
            5.9745163813177126e-05,
            3.3909105697810145e-22,
            3.381092268831497e-22,
            8.82626040644735e-05,
            8.82626040644735e-05,
            3.381092268831497e-22,
            3.3909105697810145e-22,
            2.2574466490221357e-05,
        ]

        np.testing.assert_allclose(qG[idx10], reference_qG_10, atol=1e-12)
        np.testing.assert_allclose(
            result["SqG_full_direct"][idx10], reference_direct_10, atol=1e-10
        )
        np.testing.assert_allclose(
            result["SqG_full_q4"][idx10], reference_q4_10, atol=1e-10
        )
        np.testing.assert_allclose(
            result["SqG_full_exchange"][idx10], reference_exchange_10, atol=1e-10
        )


if __name__ == "__main__":
    unittest.main()
