import unittest

import numpy as np
from pyscf.pbc import df, mp
from pyscf.pbc import gto, scf

from fsec.singularity_subtraction.grids import ExxSSGrids
from fsec.singularity_subtraction.structure_factor.helpers_sf import (
    make_line_sampling_decay_state,
    normalize_line_sampling_decay_components,
    should_compute_line_sample,
    update_line_sampling_decay_mask,
)
from fsec.singularity_subtraction.structure_factor.mp2_sf import MP2StructureFactor


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

    def test_trs_pair_representatives_cover_each_pair_once(self):
        trs_map = np.array([0, 2, 1, 4, 3])
        nkpts = len(trs_map)
        representatives = MP2StructureFactor._build_trs_pair_representatives(
            trs_map, nkpts)

        covered = set()
        for m, n, factor in representatives:
            partner = (trs_map[n], trs_map[m])
            if (m, n) == partner:
                self.assertEqual(factor, 1)
            else:
                self.assertEqual(factor, 2)
            covered.add((m, n))
            covered.add(partner)

        self.assertEqual(len(covered), nkpts * nkpts)
        self.assertEqual(sum(factor for _, _, factor in representatives), nkpts * nkpts)

    def test_trs_representative_contractions_match_full_reductions(self):
        rng = np.random.default_rng(14)
        trs_map = np.array([0, 2, 1, 3])
        nkpts = len(trs_map)
        nocc = 2
        nvir = 3
        representatives = MP2StructureFactor._build_trs_pair_representatives(
            trs_map, nkpts)

        rho_ia = (
            rng.normal(size=(nkpts, nocc, nvir))
            + 1j * rng.normal(size=(nkpts, nocc, nvir))
        )
        rho_jb = rho_ia[trs_map].transpose(0, 2, 1)
        rijab = np.einsum("mia,nbj->mnijab", rho_ia, rho_jb.conj())

        direct_t2 = np.zeros((nkpts, nkpts, nocc, nocc, nvir, nvir), dtype=complex)
        exchange_t2 = np.zeros_like(direct_t2)
        eijab = np.zeros((nkpts, nkpts, nocc, nocc, nvir, nvir))
        for m, n, _ in representatives:
            partner_m = trs_map[n]
            partner_n = trs_map[m]
            direct_block = (
                rng.normal(size=(nocc, nocc, nvir, nvir))
                + 1j * rng.normal(size=(nocc, nocc, nvir, nvir))
            )
            exchange_block = (
                rng.normal(size=(nocc, nocc, nvir, nvir))
                + 1j * rng.normal(size=(nocc, nocc, nvir, nvir))
            )
            denom_block = rng.random(size=(nocc, nocc, nvir, nvir)) + 0.1
            direct_t2[m, n] = direct_block
            exchange_t2[m, n] = exchange_block
            eijab[m, n] = denom_block
            if (partner_m, partner_n) != (m, n):
                direct_t2[partner_m, partner_n] = direct_block.transpose(1, 0, 3, 2).conj()
                exchange_t2[partner_m, partner_n] = exchange_block.transpose(1, 0, 3, 2).conj()
                eijab[partner_m, partner_n] = denom_block.transpose(1, 0, 3, 2)

        full_direct = np.einsum("mnijab,mnijab->", rijab, direct_t2).real
        full_exchange = np.einsum("mnijab,mnijba->", rijab, exchange_t2).real
        full_q4 = np.sum(np.abs(rijab)**2 * np.abs(eijab))

        direct, exchange, q4 = MP2StructureFactor._contract_trs_representative_rijab(
            rho_ia,
            rho_jb,
            representatives,
            direct_t2=direct_t2,
            exchange_t2=exchange_t2,
            eijab_recip=eijab,
        )

        self.assertAlmostEqual(direct, full_direct, places=12)
        self.assertAlmostEqual(exchange, full_exchange, places=12)
        self.assertAlmostEqual(q4, full_q4, places=12)

    def test_trs_representative_lov_contractions_match_t2_blocks(self):
        rng = np.random.default_rng(37)
        trs_map = np.array([0, 2, 1])
        nkpts = len(trs_map)
        naux = 4
        nocc = 2
        nvir = 3
        representatives = MP2StructureFactor._build_trs_pair_representatives(
            trs_map, nkpts)

        rho_ia = (
            rng.normal(size=(nkpts, nocc, nvir))
            + 1j * rng.normal(size=(nkpts, nocc, nvir))
        )
        rho_jb = (
            rng.normal(size=(nkpts, nvir, nocc))
            + 1j * rng.normal(size=(nkpts, nvir, nocc))
        )
        Lov = np.empty((nkpts, nkpts), dtype=object)
        Lov_b = np.empty((nkpts, nkpts), dtype=object)
        for ko in range(nkpts):
            for kv in range(nkpts):
                Lov[ko, kv] = (
                    rng.normal(size=(naux, nocc, nvir))
                    + 1j * rng.normal(size=(naux, nocc, nvir))
                )
                Lov_b[ko, kv] = (
                    rng.normal(size=(naux, nocc, nvir))
                    + 1j * rng.normal(size=(naux, nocc, nvir))
                )

        kas_at_qi = np.array([0, 1, 2])
        kbs_at_qi = np.array([2, 0, 1])
        mo_e_o = -rng.random(size=(nkpts, nocc)) - 0.5
        mo_e_v = rng.random(size=(nkpts, nvir)) + 0.5
        mo_e_v_b = rng.random(size=(nkpts, nvir)) + 0.75
        nonzero_opadding = [np.arange(nocc) for _ in range(nkpts)]
        nonzero_vpadding = [np.arange(nvir) for _ in range(nkpts)]

        direct_t2 = np.zeros((nkpts, nkpts, nocc, nocc, nvir, nvir), dtype=complex)
        exchange_t2 = np.zeros_like(direct_t2)
        eijab_recip = np.zeros((nkpts, nkpts, nocc, nocc, nvir, nvir))

        for m, n, _ in representatives:
            ka = kas_at_qi[m]
            kb = kbs_at_qi[n]
            eia = mo_e_o[m, :, None] - mo_e_v[ka]
            ejb = mo_e_o[n, :, None] - mo_e_v_b[kb]
            direct_recip = 1 / (eia[:, None, :, None] + ejb[None, :, None, :])
            direct_oovv = (
                np.einsum("Lia,Ljb->iajb", Lov[m, ka], Lov_b[n, kb])
                .transpose(0, 2, 1, 3)
                / nkpts
            )
            direct_t2[m, n] = direct_oovv.conj() * direct_recip
            eijab_recip[m, n] = direct_recip

            eib = mo_e_o[m, :, None] - mo_e_v_b[kb]
            eja = mo_e_o[n, :, None] - mo_e_v[ka]
            exchange_recip = 1 / (eib[:, None, :, None] + eja[None, :, None, :])
            exchange_oovv = (
                np.einsum("Lib,Lja->ijba", Lov_b[m, kb], Lov[n, ka])
                / nkpts
            )
            exchange_t2[m, n] = exchange_oovv.conj() * exchange_recip

        reference = MP2StructureFactor._contract_trs_representative_rijab(
            rho_ia,
            rho_jb,
            representatives,
            direct_t2=direct_t2,
            exchange_t2=exchange_t2,
            eijab_recip=eijab_recip,
        )
        actual = MP2StructureFactor._contract_trs_representative_rijab_lov(
            rho_ia,
            rho_jb,
            representatives,
            Lov,
            Lov_b,
            kas_at_qi,
            kbs_at_qi,
            mo_e_o,
            mo_e_v,
            mo_e_v_b,
            nonzero_opadding,
            nonzero_vpadding,
            nkpts,
            compute_direct=True,
            compute_exchange=True,
            compute_q4=True,
        )

        for actual_value, reference_value in zip(actual, reference):
            self.assertAlmostEqual(actual_value, reference_value, places=12)


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
            pair_density_eval_grid="uniform",
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

    def test_build_structure_factor_10_closest_qg_points_becke_level0(self):
        mp2_sf = MP2StructureFactor(
            self.kmf,
            self.kmp,
            t2=self.t2,
            N_local=self.N_local,
            qG_cutoff=self.qG_cutoff,
            min_points=10,
            pair_density_eval_grid="becke",
            pair_density_becke_grid_level=0,
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

        reference_qG_10 = [
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
        ]
        reference_SqG_direct_10 = [
            -4.294521938834457e-09,
            -7.652429628603002e-10,
            -7.652429628959698e-10,
            -0.00026828638583244375,
            -0.00026828638583244375,
            -7.652429628959698e-10,
            -7.652429628603002e-10,
            -1.0870652872799016e-10,
            -0.0001126844199236413,
            -0.00011268441992364891,
        ]
        reference_SqG_q4_10 = [
            -1.1994133199422713e-15,
            -3.808359212315172e-17,
            -3.808359212670204e-17,
            -4.680976772564127e-06,
            -4.680976772564127e-06,
            -3.808359212670204e-17,
            -3.808359212315172e-17,
            -7.68511679024734e-19,
            -8.257849487100997e-07,
            -8.257849487102111e-07,
        ]
        reference_SqG_exchange_10 = [
            2.1472609694172284e-09,
            3.826214814301501e-10,
            3.826214814479849e-10,
            0.00013414319291622188,
            0.00013414319291622188,
            3.826214814479849e-10,
            3.826214814301501e-10,
            5.435326436399508e-11,
            5.634220996182065e-05,
            5.6342209961824454e-05,
        ]

        np.testing.assert_allclose(qG_10, reference_qG_10, atol=1e-12)
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
            pair_density_eval_grid="uniform",
        )
        reference = reference_sf.build_structure_factor(direct=True, exchange=True, dG0=True)

        ki_sf = MP2StructureFactor(
            self.kmf,
            self.kmp,
            N_local=self.N_local,
            qG_cutoff=self.qG_cutoff,
            min_points=10,
            t2_store_type="ki",
            pair_density_eval_grid="uniform",
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
            check_trs=False,
            t2_store_type="kikjka",
            pair_density_eval_grid="uniform",
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
            pair_density_eval_grid="uniform",
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

    def test_kikj_lov_matches_kikjka_without_t2_materialization(self):
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
            check_trs=True,
            t2_store_type="kikjka",
            pair_density_eval_grid="uniform",
        )
        reference = reference_sf.build_structure_factor(
            qG_full=qG_full, direct=True, exchange=True, dG0=True)

        lov_sf = MP2StructureFactor(
            self.kmf,
            self.kmp,
            N_local=self.N_local,
            qG_cutoff=self.qG_cutoff,
            min_points=10,
            sq_inversion_symm=False,
            check_trs=True,
            t2_store_type="kikj_lov",
            pair_density_eval_grid="uniform",
        )
        actual = lov_sf.build_structure_factor(
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
        self.assertIn(
            "rijab/Lov TRS representative contraction",
            lov_sf.last_build_timings,
        )
        self.assertNotIn("direct t2 cache miss", lov_sf.last_build_timings)
        self.assertNotIn("exchange t2 cache miss", lov_sf.last_build_timings)

    def test_trs_representative_kikjka_runs_with_exact_fallback_112_kmesh(self):
        reciprocal = self.kmf_112.cell.reciprocal_vectors()
        qG_full = np.array([
            [0.0, 0.0, 0.0],
            0.5 * reciprocal[2],
            -0.5 * reciprocal[2],
        ])

        fallback_sf = MP2StructureFactor(
            self.kmf_112,
            self.kmp_112,
            t2=self.t2_112,
            N_local=self.N_local,
            qG_cutoff=self.qG_cutoff,
            min_points=10,
            sq_inversion_symm=False,
            check_trs=False,
            t2_store_type="kikjka",
            pair_density_eval_grid="uniform",
        )
        fallback = fallback_sf.build_structure_factor(
            qG_full=qG_full, direct=True, exchange=True, dG0=True)

        optimized_sf = MP2StructureFactor(
            self.kmf_112,
            self.kmp_112,
            t2=self.t2_112,
            N_local=self.N_local,
            qG_cutoff=self.qG_cutoff,
            min_points=10,
            sq_inversion_symm=False,
            check_trs=True,
            t2_store_type="kikjka",
            pair_density_eval_grid="uniform",
        )
        optimized = optimized_sf.build_structure_factor(
            qG_full=qG_full, direct=True, exchange=True, dG0=True)

        np.testing.assert_allclose(optimized["qG_full"], fallback["qG_full"], atol=1e-12)
        for key in ("SqG_full_direct", "SqG_full_exchange", "SqG_full_q4"):
            self.assertTrue(np.all(np.isfinite(optimized[key])))
            self.assertTrue(np.all(np.isfinite(fallback[key])))
        self.assertFalse(
            np.allclose(
                optimized["SqG_full_direct"],
                fallback["SqG_full_direct"],
                rtol=1e-7,
                atol=1e-10,
            )
        )
        self.assertIn(
            "rijab/t2 TRS representative contraction",
            optimized_sf.last_build_timings,
        )
        self.assertIn("rijab tensor contraction", fallback_sf.last_build_timings)

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
            pair_density_eval_grid="uniform",
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

    def test_adaptive_sq_ke_cutoff_matches_uniform_references(self):
        reciprocal = self.kmf.cell.reciprocal_vectors()
        qG_full = np.array([
            [0.0, 0.0, 0.0],
            reciprocal[0],
            2 * reciprocal[0],
        ])
        switch_radius = np.linalg.norm(reciprocal[0]) + 1e-8
        half_sq_ke_cutoff = 0.5 * self.sq_ke_cutoff

        adaptive_sf = MP2StructureFactor(
            self.kmf,
            self.kmp,
            t2=self.t2,
            sq_ke_cutoff=self.sq_ke_cutoff,
            qG_cutoff=self.qG_cutoff,
            min_points=10,
            sq_inversion_symm=False,
            sq_ke_cutoff_switch_radius=switch_radius,
            pair_density_eval_grid="uniform",
        )
        adaptive = adaptive_sf.build_structure_factor(
            qG_full=qG_full, direct=True, exchange=True, dG0=True)

        full_sf = MP2StructureFactor(
            self.kmf,
            self.kmp,
            t2=self.t2,
            sq_ke_cutoff=self.sq_ke_cutoff,
            qG_cutoff=self.qG_cutoff,
            min_points=10,
            sq_inversion_symm=False,
            pair_density_eval_grid="uniform",
        )
        full_reference = full_sf.build_structure_factor(
            qG_full=qG_full, direct=True, exchange=True, dG0=True)

        half_sf = MP2StructureFactor(
            self.kmf,
            self.kmp,
            t2=self.t2,
            sq_ke_cutoff=half_sq_ke_cutoff,
            qG_cutoff=self.qG_cutoff,
            min_points=10,
            sq_inversion_symm=False,
            pair_density_eval_grid="uniform",
        )
        half_reference = half_sf.build_structure_factor(
            qG_full=qG_full, direct=True, exchange=True, dG0=True)

        inner = np.linalg.norm(qG_full, axis=1) <= switch_radius + 1e-8
        outer = ~inner
        np.testing.assert_allclose(adaptive["qG_full"], qG_full, atol=1e-12)
        np.testing.assert_allclose(
            adaptive["sq_ke_cutoff_by_qG"][inner],
            np.full(np.count_nonzero(inner), self.sq_ke_cutoff),
        )
        np.testing.assert_allclose(
            adaptive["sq_ke_cutoff_by_qG"][outer],
            np.full(np.count_nonzero(outer), half_sq_ke_cutoff),
        )
        np.testing.assert_array_equal(
            adaptive["N_local_by_region"]["inner"],
            self.kmf.cell.cutoff_to_mesh(self.sq_ke_cutoff),
        )
        np.testing.assert_array_equal(
            adaptive["N_local_by_region"]["outer"],
            self.kmf.cell.cutoff_to_mesh(half_sq_ke_cutoff),
        )
        for key in ("SqG_full_direct", "SqG_full_exchange", "SqG_full_q4"):
            np.testing.assert_allclose(
                adaptive[key][inner],
                full_reference[key][inner],
                rtol=1e-7,
                atol=1e-10,
            )
            np.testing.assert_allclose(
                adaptive[key][outer],
                half_reference[key][outer],
                rtol=1e-7,
                atol=1e-10,
            )

    def test_adaptive_sq_ke_cutoff_preserves_line_sampling_masks(self):
        mp2_sf = MP2StructureFactor(
            self.kmf,
            self.kmp,
            t2=self.t2,
            sq_ke_cutoff=self.sq_ke_cutoff,
            qG_cutoff=self.qG_cutoff,
            min_points=10,
            sq_inversion_symm=False,
            sq_ke_cutoff_switch_radius=1.5,
            pair_density_eval_grid="uniform",
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
            line_sampling_decay_components=("direct_q4", "exchange"),
            qG_line_sampling_segments=mp2_sf.grids.qG_line_sampling_segments,
        )

        np.testing.assert_allclose(result["qG_full"], qG_full, atol=1e-12)
        self.assertEqual(len(result["SqG_full_direct_mask"]), len(result["qG_full"]))
        self.assertEqual(len(result["SqG_full_exchange_mask"]), len(result["qG_full"]))
        self.assertEqual(len(result["SqG_full_q4_mask"]), len(result["qG_full"]))
        self.assertEqual(len(result["sq_ke_cutoff_by_qG"]), len(result["qG_full"]))
        self.assertIn("inner", result["N_local_by_region"])
        self.assertIn("outer", result["N_local_by_region"])
        self.assertIn("line_sampling_decay_events", result)

    def test_build_structure_factor_112_kmesh(self):
        mp2_sf = MP2StructureFactor(
            self.kmf_112,
            self.kmp_112,
            t2=self.t2_112,
            N_local=self.N_local,
            qG_cutoff=self.qG_cutoff,
            min_points=10,
            pair_density_eval_grid="uniform",
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
        ]
        reference_q4_10 = [
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
        ]
        reference_exchange_10 = [
            3.948083901297239e-22,
            5.2916344116689204e-05,
            5.2916344116689204e-05,
            3.821253042962879e-22,
            3.8211585588923894e-22,
            0.00010004243537396696,
            0.00010004243537396696,
            3.8211585588923894e-22,
            3.821253042962879e-22,
            1.99304805849594e-05,
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
