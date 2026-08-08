import unittest
from unittest.mock import patch
from types import SimpleNamespace
import gc
import importlib.util
import weakref

import numpy as np
from pyscf.pbc import df, mp
from pyscf.pbc.mp import kmp2
from pyscf.pbc import gto, scf

from fsec.singularity_subtraction.grids import ExxSSGrids
from fsec.singularity_subtraction.mp2ss import convert_t2_to_kikjq_format
from fsec.singularity_subtraction.structure_factor.helpers import (
    TimingProfile,
    make_line_sampling_decay_state,
    normalize_line_sampling_decay_components,
    should_compute_line_sample,
    update_line_sampling_decay_mask,
)
from fsec.singularity_subtraction.structure_factor.laplace import (
    make_minimax_laplace_grid,
)
from fsec.singularity_subtraction.structure_factor.mp2_contractions import (
    build_trs_unique_pairs,
    contract_direct_q4_lov_laplace,
    contract_trs_unique_pair_rijab,
    contract_trs_unique_pair_rijab_lov,
    contract_trs_unique_pair_rijab_lov_laplace,
)
from fsec.singularity_subtraction.structure_factor.mp2_rsdf_direct_helpers import (
    accumulate_direct_rsdf_occ_blocks,
    estimate_direct_rsdf_block_memory_bytes,
    select_direct_rsdf_occ_block_size,
)
from fsec.singularity_subtraction.structure_factor.mp2_sf import MP2StructureFactor


HAS_KMP2_DIRECT = (
    importlib.util.find_spec("pyscf.pbc.mp.kmp2_direct") is not None
)


class LineSamplingDecayHelpers(unittest.TestCase):
    def test_direct_rsdf_block_memory_selection_and_preflight(self):
        dimensions = {
            "nocc": 4,
            "nkpts": 2,
            "naux": 11,
            "nvir": 3,
            "ngrid": 17,
            "laplace": True,
            "max_laplace_points": 8,
        }
        estimates = {
            block_size: estimate_direct_rsdf_block_memory_bytes(
                dimensions["nkpts"],
                dimensions["naux"],
                dimensions["nvir"],
                block_size,
                dimensions["ngrid"],
                nocc=dimensions["nocc"],
                laplace=dimensions["laplace"],
                max_laplace_points=dimensions["max_laplace_points"],
            )
            for block_size in range(1, dimensions["nocc"] + 1)
        }
        available = estimates[2]
        expected = max(
            block_size
            for block_size, estimate in estimates.items()
            if estimate <= available
        )
        selected, estimate = select_direct_rsdf_occ_block_size(
            available_bytes=available, **dimensions)
        self.assertEqual(selected, expected)
        self.assertEqual(estimate, estimates[expected])

        selected, estimate = select_direct_rsdf_occ_block_size(
            available_bytes=estimates[1],
            requested=1,
            **dimensions,
        )
        self.assertEqual(selected, 1)
        self.assertEqual(estimate, estimates[1])
        with self.assertRaisesRegex(MemoryError, "B=2"):
            select_direct_rsdf_occ_block_size(
                available_bytes=estimates[2] - 1,
                requested=2,
                **dimensions,
            )
        with self.assertRaisesRegex(MemoryError, "even B=1"):
            select_direct_rsdf_occ_block_size(
                available_bytes=estimates[1] - 1,
                **dimensions,
            )

    def test_direct_rsdf_scheduler_retains_at_most_two_lov_blocks(self):
        cell = gto.Cell()
        cell.unit = "Bohr"
        cell.atom = "H 0 0 0"
        cell.a = np.eye(3) * 5
        cell.basis = "gth-szv"
        cell.pseudo = "gth-hf"
        cell.spin = 1
        cell.verbose = 0
        cell.build()
        nocc, nvir, naux, ngrid = 3, 1, 2, 2
        auxcell = SimpleNamespace(nao_nr=lambda: naux)
        mydf = SimpleNamespace(auxcell=auxcell)
        scf_stub = SimpleNamespace(cell=cell, with_df=mydf)
        kmp_stub = SimpleNamespace(
            nkpts=1,
            nocc=nocc,
            nmo=nocc + nvir,
            max_memory=10000,
            _scf=scf_stub,
        )
        kmf_stub = SimpleNamespace(cell=cell)
        mesh_context = {
            "rptGrid3D": np.zeros((ngrid, 3)),
            "quadrature_weights": None,
            "quadrature_product_scale": 1.0,
            "uKpts_i": np.ones((1, nocc, ngrid), dtype=complex),
            "uKpts_a": np.ones((1, nvir, ngrid), dtype=complex),
        }
        live = {"count": 0, "maximum": 0}
        requested_slices = []

        def release_block():
            live["count"] -= 1

        def provider(_kmp, occ_slice):
            requested_slices.append(tuple(occ_slice))
            live["count"] += 1
            live["maximum"] = max(live["maximum"], live["count"])
            result = np.empty((1, 1), dtype=object)
            result[0, 0] = np.ones((naux, 1, nvir), dtype=complex)
            weakref.finalize(result, release_block)
            return result

        outputs = [np.zeros(1) for _ in range(3)]
        masks = [np.ones(1, dtype=bool) for _ in range(3)]
        with patch(
            "fsec.singularity_subtraction.structure_factor."
            "mp2_rsdf_direct_helpers."
            "build_direct_rsdf_lov_block",
            side_effect=provider,
        ):
            stats = accumulate_direct_rsdf_occ_blocks(
                kmp=kmp_stub,
                kmf=kmf_stub,
                qG_full=np.zeros((1, 3)),
                qi_map=np.zeros(1, dtype=int),
                qGrid=np.zeros((1, 3)),
                kGrid1=np.zeros((1, 3)),
                kas=np.zeros((1, 1), dtype=int),
                kbs=np.zeros((1, 1), dtype=int),
                qG_uses_outer_mesh=np.zeros(1, dtype=bool),
                mesh_contexts={"inner": mesh_context},
                trs_map=np.zeros(1, dtype=int),
                trs_unique_pairs=[(0, 0, 1)],
                mo_e_o=np.array([[-1.0, -0.9, -0.8]]),
                mo_e_v=np.array([[0.5]]),
                mo_e_v_b=np.array([[0.5]]),
                nonzero_opadding=[np.arange(nocc)],
                nonzero_vpadding=[np.arange(nvir)],
                omega_cell=125.0,
                direct=True,
                exchange=True,
                dG0=True,
                sq_inversion_symm=False,
                inversion_partner=None,
                requested_block_size=1,
                laplace=False,
                laplace_direct_tol=1e-8,
                laplace_direct_max_points=16,
                laplace_exchange_tol=1e-8,
                laplace_exchange_max_points=16,
                SqG_full_direct=outputs[0],
                SqG_full_exchange=outputs[1],
                SqG_full_q4=outputs[2],
                SqG_full_direct_mask=masks[0],
                SqG_full_exchange_mask=masks[1],
                SqG_full_q4_mask=masks[2],
                q4_decay_state=None,
                exchange_decay_state=None,
                profile=TimingProfile(),
                log=SimpleNamespace(note=lambda *args, **kwargs: None),
            )
        gc.collect()
        self.assertEqual(stats[3], 6)
        self.assertEqual(len(requested_slices), 6)
        self.assertLessEqual(live["maximum"], 2)
        self.assertEqual(live["count"], 0)

    def test_minimax_laplace_grid_meets_error_bound(self):
        denominator_min = 0.4
        denominator_max = 18.0
        grid = make_minimax_laplace_grid(
            denominator_min, denominator_max, tolerance=1e-8, max_points=16)

        self.assertIsNotNone(grid)
        self.assertEqual(grid.npoints, 12)
        self.assertEqual(grid.table_ratio, 100.0)
        denominators = np.geomspace(denominator_min, denominator_max, 20001)
        approximate = (
            np.exp(-denominators[:, None] * grid.points) @ grid.weights)
        error = np.max(np.abs(approximate - 1 / denominators))
        self.assertLessEqual(error, grid.normalized_error / denominator_min)

        self.assertIsNone(make_minimax_laplace_grid(1.0, 1001.0))
        self.assertIsNone(make_minimax_laplace_grid(
            1.0, 100.0, tolerance=1e-8, max_points=8))

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

    def test_trs_unique_pairs_cover_each_pair_once(self):
        trs_map = np.array([0, 2, 1, 4, 3])
        nkpts = len(trs_map)
        trs_unique_pairs = build_trs_unique_pairs(
            trs_map, nkpts)

        covered = set()
        for m, n, factor in trs_unique_pairs:
            partner = (trs_map[n], trs_map[m])
            if (m, n) == partner:
                self.assertEqual(factor, 1)
            else:
                self.assertEqual(factor, 2)
            covered.add((m, n))
            covered.add(partner)

        self.assertEqual(len(covered), nkpts * nkpts)
        self.assertEqual(sum(factor for _, _, factor in trs_unique_pairs), nkpts * nkpts)

    def test_trs_unique_pair_contractions_match_full_reductions(self):
        rng = np.random.default_rng(14)
        trs_map = np.array([0, 2, 1, 3])
        nkpts = len(trs_map)
        nocc = 2
        nvir = 3
        trs_unique_pairs = build_trs_unique_pairs(
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
        for m, n, _ in trs_unique_pairs:
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

        direct, exchange, q4 = contract_trs_unique_pair_rijab(
            rho_ia,
            rho_jb,
            trs_unique_pairs,
            direct_t2=direct_t2,
            exchange_t2=exchange_t2,
            eijab_recip=eijab,
        )

        self.assertAlmostEqual(direct, full_direct, places=12)
        self.assertAlmostEqual(exchange, full_exchange, places=12)
        self.assertAlmostEqual(q4, full_q4, places=12)

    def test_trs_unique_pair_lov_contractions_match_t2_blocks(self):
        rng = np.random.default_rng(37)
        trs_map = np.array([0, 2, 1])
        nkpts = len(trs_map)
        naux = 4
        nocc = 2
        nvir = 3
        trs_unique_pairs = build_trs_unique_pairs(
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

        for m, n, _ in trs_unique_pairs:
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

        reference = contract_trs_unique_pair_rijab(
            rho_ia,
            rho_jb,
            trs_unique_pairs,
            direct_t2=direct_t2,
            exchange_t2=exchange_t2,
            eijab_recip=eijab_recip,
        )
        actual = contract_trs_unique_pair_rijab_lov(
            rho_ia,
            rho_jb,
            trs_unique_pairs,
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

        laplace_contract = (
            contract_trs_unique_pair_rijab_lov_laplace
        )
        laplace_actual = laplace_contract(
            rho_ia,
            rho_jb,
            trs_unique_pairs,
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
            direct_tolerance=1e-8,
            direct_max_points=16,
            exchange_tolerance=1e-8,
            exchange_max_points=16,
        )
        for actual_value, reference_value in zip(laplace_actual, reference):
            np.testing.assert_allclose(
                actual_value, reference_value, rtol=2e-8, atol=1e-10)

        direct_profile = TimingProfile()
        laplace_contract(
            rho_ia,
            rho_jb,
            trs_unique_pairs,
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
            compute_q4=True,
            profile=direct_profile,
            direct_tolerance=1e-8,
            direct_max_points=16,
        )
        self.assertIn(
            "TRS Lov direct/q4 Laplace contraction", direct_profile._times)
        self.assertNotIn(
            "TRS Lov direct/q4 denominator build", direct_profile._times)
        self.assertNotIn("TRS Lov direct X@denom", direct_profile._times)

        laplace_exchange = laplace_contract(
            rho_ia,
            rho_jb,
            trs_unique_pairs,
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
            compute_exchange=True,
            exchange_tolerance=1e-8,
            exchange_max_points=16,
        )
        np.testing.assert_allclose(
            laplace_exchange[1], reference[1], rtol=2e-8, atol=1e-10)

        # Insufficient Laplace rank must take the exact path, not return a
        # lower-accuracy approximation.
        exact_fallback = laplace_contract(
            rho_ia,
            rho_jb,
            trs_unique_pairs,
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
            compute_exchange=True,
            exchange_tolerance=1e-8,
            exchange_max_points=1,
        )
        self.assertAlmostEqual(exact_fallback[1], reference[1], places=12)

        direct_exact_fallback = laplace_contract(
            rho_ia,
            rho_jb,
            trs_unique_pairs,
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
            compute_q4=True,
            direct_tolerance=1e-8,
            direct_max_points=1,
        )
        self.assertAlmostEqual(
            direct_exact_fallback[0], reference[0], places=12)
        self.assertAlmostEqual(
            direct_exact_fallback[2], reference[2], places=12)

        # Exercise the independent allocation paths as well.  In particular,
        # exchange-only must not construct or depend on the (ia, jb) direct
        # denominator matrix.
        for contraction, component, expected_index in (
                (laplace_contract, {"compute_direct": True}, 0),
                (contract_trs_unique_pair_rijab_lov,
                 {"compute_exchange": True}, 1),
                (laplace_contract, {"compute_q4": True}, 2)):
            component_actual = (
                contraction(
                    rho_ia,
                    rho_jb,
                    trs_unique_pairs,
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
                    **component,
                )
            )
            np.testing.assert_allclose(
                component_actual[expected_index],
                reference[expected_index],
                rtol=2e-8,
                atol=1e-10,
            )

    def test_rectangular_occupied_lov_blocks_sum_to_full_contraction(self):
        rng = np.random.default_rng(913)
        trs_map = np.array([0, 2, 1])
        nkpts = len(trs_map)
        naux, nocc, nvir = 4, 3, 2
        trs_unique_pairs = build_trs_unique_pairs(trs_map, nkpts)
        rho_ia = (
            rng.normal(size=(nkpts, nocc, nvir))
            + 1j * rng.normal(size=(nkpts, nocc, nvir))
        )
        rho_jb = (
            rng.normal(size=(nkpts, nvir, nocc))
            + 1j * rng.normal(size=(nkpts, nvir, nocc))
        )
        Lov = np.empty((nkpts, nkpts), dtype=object)
        for ko in range(nkpts):
            for kv in range(nkpts):
                Lov[ko, kv] = (
                    rng.normal(size=(naux, nocc, nvir))
                    + 1j * rng.normal(size=(naux, nocc, nvir))
                )
        kas_at_qi = np.array([0, 1, 2])
        kbs_at_qi = np.array([2, 0, 1])
        mo_e_o = -rng.random(size=(nkpts, nocc)) - 0.5
        mo_e_v = rng.random(size=(nkpts, nvir)) + 0.5
        mo_e_v_b = mo_e_v.copy()
        nonzero_opadding = [
            np.array([0, 1, 2]),
            np.array([0, 2]),
            np.array([1, 2]),
        ]
        nonzero_vpadding = [np.arange(nvir) for _ in range(nkpts)]
        for ko, occupied in enumerate(nonzero_opadding):
            padded = np.setdiff1d(np.arange(nocc), occupied)
            rho_ia[ko, padded, :] = 0
            rho_jb[ko, :, padded] = 0
            for kv in range(nkpts):
                Lov[ko, kv][:, padded, :] = 0
        common = (
            trs_unique_pairs,
            kas_at_qi,
            kbs_at_qi,
            mo_e_o,
            mo_e_v,
            mo_e_v_b,
            nonzero_opadding,
            nonzero_vpadding,
            nkpts,
        )

        def sliced_lov(occ_slice):
            start, stop = occ_slice
            result = np.empty_like(Lov)
            for ko in range(nkpts):
                for kv in range(nkpts):
                    result[ko, kv] = Lov[ko, kv][:, start:stop, :]
            return result

        def blocked_sum(contraction, **extra):
            slices = [(0, 1), (1, 3)]
            tables = [sliced_lov(occ_slice) for occ_slice in slices]
            total = np.zeros(3)
            for block_i, occ_slice_i in enumerate(slices):
                for block_j in range(block_i + 1):
                    occ_slice_j = slices[block_j]
                    directed = contraction(
                        rho_ia[:, occ_slice_i[0]:occ_slice_i[1], :],
                        rho_jb[:, :, occ_slice_j[0]:occ_slice_j[1]],
                        common[0],
                        tables[block_i],
                        tables[block_j],
                        *common[1:],
                        compute_direct=True,
                        compute_exchange=True,
                        compute_q4=True,
                        occ_slice_i=occ_slice_i,
                        occ_slice_j=occ_slice_j,
                        Lov_exchange_i=tables[block_i],
                        Lov_exchange_j=tables[block_j],
                        **extra,
                    )
                    total += directed
                    if block_i != block_j:
                        reverse = contraction(
                            rho_ia[:, occ_slice_j[0]:occ_slice_j[1], :],
                            rho_jb[:, :, occ_slice_i[0]:occ_slice_i[1]],
                            common[0],
                            tables[block_j],
                            tables[block_i],
                            *common[1:],
                            compute_direct=True,
                            compute_exchange=True,
                            compute_q4=True,
                            occ_slice_i=occ_slice_j,
                            occ_slice_j=occ_slice_i,
                            Lov_exchange_i=tables[block_j],
                            Lov_exchange_j=tables[block_i],
                            **extra,
                        )
                        total += reverse
            return total

        exact_full = contract_trs_unique_pair_rijab_lov(
            rho_ia,
            rho_jb,
            common[0],
            Lov,
            Lov,
            *common[1:],
            compute_direct=True,
            compute_exchange=True,
            compute_q4=True,
        )
        np.testing.assert_allclose(
            blocked_sum(contract_trs_unique_pair_rijab_lov),
            exact_full,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            blocked_sum(
                contract_trs_unique_pair_rijab_lov_laplace,
                direct_tolerance=1e-8,
                direct_max_points=16,
                exchange_tolerance=1e-8,
                exchange_max_points=16,
            ),
            exact_full,
            rtol=2e-8,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            blocked_sum(
                contract_trs_unique_pair_rijab_lov_laplace,
                direct_max_points=1,
                exchange_max_points=1,
            ),
            exact_full,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_direct_q4_laplace_respects_padding_masks(self):
        rng = np.random.default_rng(83)
        naux, nocc, nvir = 3, 2, 3
        rho_ia = rng.normal(size=(nocc, nvir)) + 1j * rng.normal(
            size=(nocc, nvir))
        rho_jb = rng.normal(size=(nvir, nocc)) + 1j * rng.normal(
            size=(nvir, nocc))
        Lov_mka = rng.normal(size=(naux, nocc, nvir)) + 1j * rng.normal(
            size=(naux, nocc, nvir))
        Lov_nkb = rng.normal(size=(naux, nocc, nvir)) + 1j * rng.normal(
            size=(naux, nocc, nvir))
        eia = -(0.5 + rng.random(size=(nocc, nvir)))
        ejb = -(0.75 + rng.random(size=(nocc, nvir)))
        active_ia = np.array([[True, False, True], [False, False, False]])
        active_jb = np.array([[True, False, True], [False, False, False]])

        x_lia = Lov_mka * rho_ia[None, :, :] * active_ia[None, :, :]
        y_ljb = (
            Lov_nkb * rho_jb.conj().T[None, :, :] * active_jb[None, :, :])
        reciprocal = 1 / np.add.outer(eia.ravel(), ejb.ravel())
        direct_reference = np.sum(
            (x_lia.reshape(naux, -1) @ reciprocal)
            * y_ljb.reshape(naux, -1)
        )
        q4_reference = np.dot(
            (np.abs(rho_ia)**2 * active_ia).ravel() @ np.abs(reciprocal),
            (np.abs(rho_jb.conj().T)**2 * active_jb).ravel(),
        )

        result = contract_direct_q4_lov_laplace(
            rho_ia,
            rho_jb,
            Lov_mka,
            Lov_nkb,
            eia,
            ejb,
            active_ia,
            active_jb,
            True,
            True,
            1e-8,
            16,
        )
        self.assertIsNotNone(result)
        np.testing.assert_allclose(
            result[0], direct_reference, rtol=2e-8, atol=1e-10)
        np.testing.assert_allclose(
            result[1], q4_reference, rtol=2e-8, atol=1e-10)

        nonpositive_eia = eia.copy()
        nonpositive_eia[active_ia] = 0.1
        self.assertIsNone(
            contract_direct_q4_lov_laplace(
                rho_ia,
                rho_jb,
                Lov_mka,
                Lov_nkb,
                nonpositive_eia,
                ejb,
                active_ia,
                active_jb,
                True,
                True,
                1e-8,
                16,
            )
        )


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

        exact_lov_sf = MP2StructureFactor(
            self.kmf,
            self.kmp,
            N_local=self.N_local,
            qG_cutoff=self.qG_cutoff,
            min_points=10,
            sq_inversion_symm=False,
            check_trs=True,
            t2_store_type="kikj_lov",
            laplace=False,
            pair_density_eval_grid="uniform",
        )
        exact = exact_lov_sf.build_structure_factor(
            qG_full=qG_full, direct=True, exchange=True, dG0=True)

        np.testing.assert_allclose(actual["qG_full"], reference["qG_full"], atol=1e-12)
        for key in (
                "SqG_full_direct", "SqG_full_exchange", "SqG_full_q4"):
            np.testing.assert_allclose(
                actual[key], reference[key], rtol=1e-7, atol=1e-10)
            np.testing.assert_allclose(
                exact[key], reference[key], rtol=1e-7, atol=1e-10)
        self.assertIn(
            "rijab/Lov TRS unique-pair contraction",
            lov_sf.last_build_timings,
        )
        self.assertIn(
            "TRS Lov exchange Laplace contraction",
            lov_sf.last_build_timings,
        )
        self.assertIn(
            "TRS Lov direct/q4 Laplace contraction",
            lov_sf.last_build_timings,
        )
        self.assertNotIn(
            "TRS Lov direct/q4 denominator build",
            lov_sf.last_build_timings,
        )
        self.assertNotIn("TRS Lov direct X@denom", lov_sf.last_build_timings)
        self.assertNotIn(
            "TRS Lov exchange ERI matmul",
            lov_sf.last_build_timings,
        )
        self.assertIn(
            "TRS Lov direct/q4 denominator build",
            exact_lov_sf.last_build_timings,
        )
        self.assertIn(
            "TRS Lov exchange ERI matmul",
            exact_lov_sf.last_build_timings,
        )
        self.assertNotIn(
            "TRS Lov direct/q4 Laplace contraction",
            exact_lov_sf.last_build_timings,
        )
        self.assertNotIn(
            "TRS Lov exchange Laplace contraction",
            exact_lov_sf.last_build_timings,
        )
        self.assertNotIn("direct t2 cache miss", lov_sf.last_build_timings)
        self.assertNotIn("exchange t2 cache miss", lov_sf.last_build_timings)

    def test_direct_rsdf_uses_occupied_lov_provider(self):
        reciprocal = self.kmf.cell.reciprocal_vectors()
        qG_full = np.array([
            [0.0, 0.0, 0.0],
            reciprocal[0],
            -reciprocal[0],
        ])
        reference_sf = MP2StructureFactor(
            self.kmf,
            self.kmp,
            N_local=self.N_local,
            qG_cutoff=self.qG_cutoff,
            sq_inversion_symm=True,
            t2_store_type="kikj_lov",
            pair_density_eval_grid="uniform",
        )
        reference = reference_sf.build_structure_factor(
            qG_full=qG_full, direct=True, exchange=True, dG0=True)
        full_lov = kmp2._init_mp_df_eris(self.kmp)
        original_df = self.kmf.with_df
        direct_df = df.RSDF(self.kmf.cell, self.kmf.kpts)
        direct_df.direct = True
        direct_df.semidirect = False
        direct_df.ksym = "s2"
        requested_slices = []

        def block_provider(_kmp, occ_slice):
            requested_slices.append(tuple(occ_slice))
            start, stop = occ_slice
            result = np.empty_like(full_lov)
            for ko in range(self.kmp.nkpts):
                for kv in range(self.kmp.nkpts):
                    result[ko, kv] = full_lov[ko, kv][:, start:stop, :]
                    self.assertEqual(
                        result[ko, kv].shape[1], stop - start)
            return result

        try:
            self.kmf.with_df = direct_df
            direct_sf = MP2StructureFactor(
                self.kmf,
                self.kmp,
                N_local=self.N_local,
                qG_cutoff=self.qG_cutoff,
                sq_inversion_symm=True,
                t2_store_type="kikjka",
                rsdf_occ_block_size=1,
                pair_density_eval_grid="uniform",
            )
            with patch(
                "fsec.singularity_subtraction.structure_factor."
                "mp2_rsdf_direct_helpers."
                "build_direct_rsdf_lov_block",
                side_effect=block_provider,
            ):
                actual = direct_sf.build_structure_factor(
                    qG_full=qG_full,
                    direct=True,
                    exchange=True,
                    dG0=True,
                )
        finally:
            self.kmf.with_df = original_df

        np.testing.assert_allclose(actual["qG_full"], reference["qG_full"])
        for key in (
                "SqG_full_direct", "SqG_full_exchange", "SqG_full_q4"):
            np.testing.assert_allclose(
                actual[key], reference[key], rtol=1e-7, atol=1e-10)
        self.assertEqual(requested_slices, [(0, self.kmp.nocc)])
        self.assertEqual(
            actual["direct_rsdf_block_stats"]["lov_build_count"], 1)
        self.assertEqual(
            actual["direct_rsdf_block_stats"]["block_size"], 1)

    def test_explicit_lov_bypasses_direct_rsdf_provider(self):
        reciprocal = self.kmf.cell.reciprocal_vectors()
        qG_full = np.array([
            [0.0, 0.0, 0.0],
            reciprocal[0],
            -reciprocal[0],
        ])
        full_lov = kmp2._init_mp_df_eris(self.kmp)
        zero_lov = np.empty_like(full_lov)
        zero_lov_b = np.empty_like(full_lov)
        for ko in range(self.kmp.nkpts):
            for kv in range(self.kmp.nkpts):
                zero_lov[ko, kv] = np.zeros_like(full_lov[ko, kv])
                zero_lov_b[ko, kv] = np.zeros_like(full_lov[ko, kv])

        original_df = self.kmf.with_df
        direct_df = df.RSDF(self.kmf.cell, self.kmf.kpts)
        direct_df.direct = True
        direct_df.semidirect = False
        direct_df.ksym = "s2"
        try:
            self.kmf.with_df = direct_df
            direct_sf = MP2StructureFactor(
                self.kmf,
                self.kmp,
                N_local=self.N_local,
                qG_cutoff=self.qG_cutoff,
                sq_inversion_symm=True,
                t2_store_type="ki",
                pair_density_eval_grid="uniform",
            )
            with patch(
                "fsec.singularity_subtraction.structure_factor.mp2_sf."
                "accumulate_direct_rsdf_occ_blocks",
                side_effect=AssertionError(
                    "explicit Lov must bypass the blocked direct-RSDF route"
                ),
            ):
                actual = direct_sf.build_structure_factor(
                    qG_full=qG_full,
                    direct=True,
                    exchange=False,
                    dG0=True,
                    Lov=zero_lov,
                    Lov_b=zero_lov_b,
                )
        finally:
            self.kmf.with_df = original_df

        self.assertIsNone(actual["direct_rsdf_block_stats"])
        np.testing.assert_allclose(actual["SqG_full_direct"], 0.0)

    @unittest.skipUnless(
        HAS_KMP2_DIRECT,
        "requires the PySCF fork with pyscf.pbc.mp.kmp2_direct",
    )
    def test_direct_rsdf_real_backend_matches_stored_gdf(self):
        systems = (
            ("gamma", self.kmf, self.kmp),
            ("1x1x2", self.kmf_112, self.kmp_112),
        )
        for label, kmf, kmp_object in systems:
            with self.subTest(system=label):
                nonzero_q = (
                    kmf.kpts[1]
                    if len(kmf.kpts) > 1
                    else kmf.cell.reciprocal_vectors()[0]
                )
                qG_full = np.array([
                    [0.0, 0.0, 0.0],
                    nonzero_q,
                    -nonzero_q,
                ])
                reference_sf = MP2StructureFactor(
                    kmf,
                    kmp_object,
                    N_local=self.N_local,
                    qG_cutoff=self.qG_cutoff,
                    sq_inversion_symm=True,
                    t2_store_type="kikj_lov",
                    pair_density_eval_grid="uniform",
                )
                reference = reference_sf.build_structure_factor(
                    qG_full=qG_full,
                    direct=True,
                    exchange=True,
                    dG0=True,
                )

                original_df = kmf.with_df
                direct_df = df.RSDF(kmf.cell, kmf.kpts)
                direct_df.direct = True
                direct_df.semidirect = False
                direct_df.ksym = "s2"
                direct_df.build()
                try:
                    kmf.with_df = direct_df
                    direct_kmp = mp.KMP2(kmf)
                    direct_sf = MP2StructureFactor(
                        kmf,
                        direct_kmp,
                        N_local=self.N_local,
                        qG_cutoff=self.qG_cutoff,
                        sq_inversion_symm=True,
                        t2_store_type="kikjka",
                        rsdf_occ_block_size=1,
                        pair_density_eval_grid="uniform",
                    )
                    actual = direct_sf.build_structure_factor(
                        qG_full=qG_full,
                        direct=True,
                        exchange=True,
                        dG0=True,
                    )
                finally:
                    kmf.with_df = original_df

                np.testing.assert_allclose(
                    actual["qG_full"], reference["qG_full"])
                for key in (
                        "SqG_full_direct",
                        "SqG_full_exchange",
                        "SqG_full_q4"):
                    np.testing.assert_allclose(
                        actual[key],
                        reference[key],
                        rtol=1e-7,
                        atol=1e-10,
                    )
                self.assertEqual(
                    actual["direct_rsdf_block_stats"]["lov_build_count"], 1)
                self.assertEqual(
                    actual["direct_rsdf_block_stats"]["block_size"], 1)

    @unittest.skipUnless(
        HAS_KMP2_DIRECT,
        "requires the PySCF fork with pyscf.pbc.mp.kmp2_direct",
    )
    def test_direct_rsdf_real_backend_occupied_blocks_match_stored_gdf(self):
        cell = gto.Cell()
        cell.unit = "Bohr"
        cell.atom = """
            H 0.0 0.0 0.0
            H 0.0 0.0 1.6
            H 3.0 3.0 3.0
            H 3.0 3.0 4.6
        """
        cell.a = np.eye(3) * 8.0
        cell.spin = 0
        cell.charge = 0
        cell.basis = "gth-szv"
        cell.pseudo = "gth-hf"
        cell.ke_cutoff = 60.0
        cell.precision = 1e-8
        cell.verbose = 0
        cell.build()
        kpts = cell.make_kpts(
            (1, 1, 1), wrap_around=True, with_gamma_point=True)
        kmf = scf.KRHF(cell, kpts)
        kmf.exxdiv = "ewald"
        kmf.with_df = df.GDF(cell, kpts).build()
        kmf.conv_tol = 1e-10
        kmf.kernel()
        self.assertTrue(kmf.converged)
        stored_kmp = mp.KMP2(kmf)
        self.assertEqual(stored_kmp.nocc, 2)

        reciprocal = cell.reciprocal_vectors()
        qG_full = np.array([
            [0.0, 0.0, 0.0],
            reciprocal[0],
            -reciprocal[0],
        ])
        N_local = cell.cutoff_to_mesh(cell.ke_cutoff)
        reference_sf = MP2StructureFactor(
            kmf,
            stored_kmp,
            N_local=N_local,
            qG_cutoff=8.0,
            sq_inversion_symm=True,
            t2_store_type="kikj_lov",
            pair_density_eval_grid="uniform",
        )
        reference = reference_sf.build_structure_factor(
            qG_full=qG_full, direct=True, exchange=True, dG0=True)

        original_df = kmf.with_df
        direct_df = df.RSDF(cell, kpts)
        direct_df.direct = True
        direct_df.semidirect = False
        direct_df.ksym = "s2"
        direct_df.build()
        try:
            kmf.with_df = direct_df
            direct_kmp = mp.KMP2(kmf)
            direct_sf = MP2StructureFactor(
                kmf,
                direct_kmp,
                N_local=N_local,
                qG_cutoff=8.0,
                sq_inversion_symm=True,
                rsdf_occ_block_size=1,
                pair_density_eval_grid="uniform",
            )
            actual = direct_sf.build_structure_factor(
                qG_full=qG_full,
                direct=True,
                exchange=True,
                dG0=True,
            )
        finally:
            kmf.with_df = original_df

        for key in (
                "SqG_full_direct", "SqG_full_exchange", "SqG_full_q4"):
            np.testing.assert_allclose(
                actual[key], reference[key], rtol=1e-7, atol=1e-10)
        self.assertEqual(
            actual["direct_rsdf_block_stats"]["lov_build_count"], 3)
        self.assertEqual(
            actual["direct_rsdf_block_stats"]["block_size"], 1)

    def test_trs_unique_pair_matches_full_contraction_112_kmesh(self):
        fallback_sf = MP2StructureFactor(
            self.kmf_112,
            self.kmp_112,
            N_local=self.N_local,
            qG_cutoff=self.qG_cutoff,
            min_points=10,
            sq_inversion_symm=False,
            check_trs=False,
            t2_store_type="kikj",
            pair_density_eval_grid="uniform",
        )
        fallback_sf.set_grids(min_fit_points=10)
        qG_candidates = fallback_sf.grids.qG_grid_local
        qG_candidates = qG_candidates[
            np.linalg.norm(qG_candidates, axis=1) < self.qG_cutoff + 1e-8
        ]
        qG_full = qG_candidates[self._closest_10_indices(qG_candidates)]
        fallback = fallback_sf.build_structure_factor(
            qG_full=qG_full,
            grids=fallback_sf.grids,
            direct=True,
            exchange=True,
            dG0=True,
        )

        optimized_sf = MP2StructureFactor(
            self.kmf_112,
            self.kmp_112,
            N_local=self.N_local,
            qG_cutoff=self.qG_cutoff,
            min_points=10,
            sq_inversion_symm=False,
            check_trs=True,
            t2_store_type="kikj",
            pair_density_eval_grid="uniform",
        )
        optimized = optimized_sf.build_structure_factor(
            qG_full=qG_full,
            grids=fallback_sf.grids,
            direct=True,
            exchange=True,
            dG0=True,
        )

        np.testing.assert_allclose(
            optimized["qG_full"], fallback["qG_full"], atol=1e-12)
        for key in ("SqG_full_direct", "SqG_full_exchange", "SqG_full_q4"):
            with self.subTest(structure_factor=key):
                np.testing.assert_allclose(
                    optimized[key],
                    fallback[key],
                    rtol=1e-7,
                    atol=1e-10,
                )
        self.assertIn(
            "rijab/t2 TRS unique-pair contraction",
            optimized_sf.last_build_timings,
        )
        self.assertNotIn(
            "rijab tensor contraction",
            optimized_sf.last_build_timings,
        )
        self.assertIn("rijab tensor contraction", fallback_sf.last_build_timings)
        self.assertNotIn(
            "rijab/t2 TRS unique-pair contraction",
            fallback_sf.last_build_timings,
        )

    def test_kikjka_trs_matches_non_trs_112_kmesh(self):
        non_trs_sf = MP2StructureFactor(
            self.kmf_112,
            self.kmp_112,
            N_local=self.N_local,
            qG_cutoff=self.qG_cutoff,
            min_points=10,
            sq_inversion_symm=False,
            check_trs=False,
            t2_store_type="kikjka",
            pair_density_eval_grid="uniform",
        )
        non_trs_sf.set_grids(min_fit_points=10)
        t2_kikjq = self.t2_112.copy()
        convert_t2_to_kikjq_format(
            t2_kikjq,
            self.kmp_112.kpts,
            non_trs_sf.grids.qGrid,
            self.kmf_112.cell,
        )
        non_trs_sf.t2 = t2_kikjq
        qG_candidates = non_trs_sf.grids.qG_grid_local
        qG_candidates = qG_candidates[
            np.linalg.norm(qG_candidates, axis=1) < self.qG_cutoff + 1e-8
        ]
        qG_full = qG_candidates[self._closest_10_indices(qG_candidates)]
        non_trs = non_trs_sf.build_structure_factor(
            qG_full=qG_full,
            grids=non_trs_sf.grids,
            direct=True,
            exchange=True,
            dG0=True,
        )

        trs_sf = MP2StructureFactor(
            self.kmf_112,
            self.kmp_112,
            t2=t2_kikjq,
            N_local=self.N_local,
            qG_cutoff=self.qG_cutoff,
            min_points=10,
            sq_inversion_symm=False,
            check_trs=True,
            t2_store_type="kikjka",
            pair_density_eval_grid="uniform",
        )
        trs = trs_sf.build_structure_factor(
            qG_full=qG_full,
            grids=non_trs_sf.grids,
            direct=True,
            exchange=True,
            dG0=True,
        )

        np.testing.assert_allclose(trs["qG_full"], non_trs["qG_full"], atol=1e-12)
        for key in ("SqG_full_direct", "SqG_full_exchange", "SqG_full_q4"):
            with self.subTest(structure_factor=key):
                np.testing.assert_allclose(
                    trs[key],
                    non_trs[key],
                    rtol=1e-7,
                    atol=1e-10,
                )
        self.assertIn(
            "rijab/t2 TRS unique-pair contraction",
            trs_sf.last_build_timings,
        )
        self.assertNotIn("rijab tensor contraction", trs_sf.last_build_timings)
        self.assertIn("rijab tensor contraction", non_trs_sf.last_build_timings)
        self.assertNotIn(
            "rijab/t2 TRS unique-pair contraction",
            non_trs_sf.last_build_timings,
        )

    def test_kikj_dG0_reduction_matches_old_expression(self):
        rng = np.random.default_rng(12)
        rijab = rng.normal(size=24) + 1j * rng.normal(size=24)
        eijab = rng.normal(size=(2, 3, 4))
        scale = 0.125

        rijab_ovr_e = rijab * np.sqrt(np.abs(eijab)).ravel()
        rijab_ovr_e = rijab_ovr_e * scale
        reference = -2 * np.einsum('i,i->', rijab_ovr_e, rijab_ovr_e.conj())
        actual = -2 * scale**2 * np.sum(
            np.abs(rijab)**2 * np.abs(eijab).ravel())

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
            N_local=self.N_local,
            qG_cutoff=self.qG_cutoff,
            min_points=10,
            t2_store_type="kikjka",
            pair_density_eval_grid="uniform",
        )
        mp2_sf.set_grids(min_fit_points=10)
        t2_kikjq = self.t2_112.copy()
        convert_t2_to_kikjq_format(
            t2_kikjq,
            self.kmp_112.kpts,
            mp2_sf.grids.qGrid,
            self.kmf_112.cell,
        )
        mp2_sf.t2 = t2_kikjq
        result = mp2_sf.build_structure_factor(
            grids=mp2_sf.grids,
            direct=True,
            exchange=True,
            dG0=True,
        )

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
            -5.5472046120633615e-22,
            -0.0001399607560043585,
            -0.0001399607560043585,
            -5.3690298464512465e-22,
            -5.368898301328439e-22,
            -0.00014013849432096997,
            -0.00014013849432096997,
            -5.368898301328439e-22,
            -5.3690298464512465e-22,
            -5.34169807626042e-05,
        ]
        reference_q4_10 = [
            -3.8209490334935407e-41,
            -8.559020498960629e-07,
            -8.559020498960629e-07,
            -3.579442222239576e-41,
            -3.579266530713441e-41,
            -2.437964679354979e-06,
            -2.437964679354979e-06,
            -3.579266530713441e-41,
            -3.579442222239576e-41,
            -1.2343680816144936e-07,
        ]
        reference_exchange_10 = [
            3.7059436888971183e-22,
            5.632273857088071e-05,
            5.632273857088071e-05,
            3.5869087000632887e-22,
            3.586820862473833e-22,
            9.362890969973841e-05,
            9.362890969973841e-05,
            3.586820862473833e-22,
            3.5869087000632887e-22,
            2.1420518558527496e-05,
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
