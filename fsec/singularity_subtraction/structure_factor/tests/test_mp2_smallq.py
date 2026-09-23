"""Small, explicit regressions for the sTC small-q MP2 evaluator."""

import copy
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from pyscf.pbc import df, gto, mp, scf
from pyscf.pbc.mp import kmp2
from pyscf.pbc.tools import get_monkhorst_pack_size

from fsec.singularity_subtraction.grids import MP2SSGrids, minimum_image
from fsec.singularity_subtraction.mp2ss import MP2SS, MP2SSOptions
from fsec.singularity_subtraction.structure_factor.mp2_smallq import (
    MP2SmallQ,
    MP2SmallQOptions,
)
from fsec.singularity_subtraction.structure_factor.mp2_sf import (
    MP2StructureFactor,
)


def make_h2_cell():
    cell = gto.Cell()
    cell.unit = "Bohr"
    cell.atom = "H 0 0 0; H 0 0 1.8"
    cell.a = np.eye(3) * 6.0
    cell.basis = "gth-szv"
    cell.pseudo = "gth-hf"
    cell.mesh = [9, 9, 9]
    cell.precision = 1e-7
    cell.verbose = 0
    cell.build()
    return cell


def make_reference_pair(kmf, kmp, options, N_local, pair_grid):
    """Return the explicit-denominator structure-factor reference inputs."""
    smallq = MP2SmallQ(
        kmf,
        kmp,
        options,
        N_local=N_local,
        pair_density_eval_grid=pair_grid,
    )
    grids = smallq._build_grids()
    (energy_plus, coeff_plus), (energy_minus, coeff_minus) = (
        smallq._get_shifted_bands(grids)
    )
    coeff_occ, energy_occ = kmp2._add_padding(
        kmp, kmp.mo_coeff, kmp.mo_energy
    )
    # The legacy ki driver allocates its ERI buffer from this dtype; shifted
    # integrals can be complex even when the occupied Gamma orbitals are real.
    coeff_occ = [np.asarray(coeff, dtype=np.complex128) for coeff in coeff_occ]
    coeff_plus, energy_plus = kmp2._add_padding(
        kmp, coeff_plus, energy_plus
    )
    coeff_minus, energy_minus = kmp2._add_padding(
        kmp, coeff_minus, energy_minus
    )

    ref_grids = MP2SSGrids(
        kmf.cell,
        kmf.kpts,
        N_local=N_local,
        qG_norm_cutoff=np.inf,
        relative_shift=options.relative_shift,
        shift_occ=False,
    )
    # The generic MP2SS constructor reuses k+q for self-inverse shifts. Keep
    # the separately ordered k-q grid so its per-k frozen masks follow the
    # same convention as the small-q evaluator.
    ref_grids.kGrid2 = grids.plus.copy()
    ref_grids.kGrid3 = grids.minus.copy()
    ref_grids.kGrid3_neq_kGrid2 = not np.array_equal(
        ref_grids.kGrid2, ref_grids.kGrid3
    )
    ref_grids.build_RptGrid3D_coarse()
    band_kpts = np.concatenate((grids.plus, grids.minus), axis=0)
    ordinary_df = df.GDF(kmf.cell, grids.occupied)
    source_df = kmp._scf.with_df
    ordinary_df.auxbasis = getattr(source_df, "auxbasis", None)
    source_mesh = getattr(source_df, "mesh", None)
    if source_mesh is not None:
        ordinary_df.mesh = np.asarray(source_mesh).copy()
    for name in ("linear_dep_threshold", "exp_to_discard", "_prefer_ccdf"):
        if hasattr(source_df, name):
            setattr(ordinary_df, name, getattr(source_df, name))
    ordinary_df.build(kpts_band=band_kpts)

    kmp_ref = copy.copy(kmp)
    kmp_ref._scf = copy.copy(kmf)
    kmp_ref._scf.with_df = ordinary_df
    kmp_ref.with_df_ints = False
    structure_factor = MP2StructureFactor(
        kmf,
        kmp_ref,
        t2=None,
        N_local=N_local,
        qG_cutoff=np.inf,
        sq_inversion_symm=False,
        check_trs=False,
        t2_store_type="ki",
        pair_density_eval_grid=pair_grid,
        pair_density_becke_grid_level=0,
        verbose=0,
    )
    return (
        smallq,
        grids,
        ref_grids,
        structure_factor,
        kmp_ref,
        ordinary_df,
        coeff_occ,
        energy_occ,
        coeff_plus,
        energy_plus,
        coeff_minus,
        energy_minus,
    )


def evaluate_explicit_reference(
    structure_factor, grids, reference_grids, kmp, coeff_occ, energy_occ,
    coeff_plus, energy_plus, coeff_minus, energy_minus,
):
    """Evaluate the existing ki driver with exact denominators and public ERIs."""
    _, active_virtual = kmp2.padding_k_idx(kmp, kind="split")

    def virtual_energies(energies):
        result = np.asarray(energies)[:, kmp.nocc:].copy()
        # The general driver forms a dense denominator tensor. Give unused
        # padded virtual slots a finite gap so their zero coefficients remain
        # harmless instead of producing 0/0.
        for kpt, active in enumerate(active_virtual):
            inactive = np.ones(result.shape[1], dtype=bool)
            inactive[np.asarray(active, dtype=int)] = False
            result[kpt, inactive] = 1e6
        return result

    return structure_factor.build_structure_factor(
        direct=True,
        exchange=False,
        dG0=True,
        qG_full=grids.qprime.reshape(1, 3),
        update_class=False,
        qG_cutoff=np.inf,
        grids=reference_grids,
        mo_coeff_kpts1=coeff_occ,
        mo_coeff_kpts2=coeff_plus,
        mo_coeff_kpts3=coeff_minus,
        mo_energy=energy_occ,
        mo_e_o=np.asarray(energy_occ)[:, :kmp.nocc],
        mo_e_v=virtual_energies(energy_plus),
        mo_e_v_b=virtual_energies(energy_minus),
        t2_store_type="ki",
        kmp=kmp,
        verbose=0,
    )


class MP2SmallQOptionsTests(unittest.TestCase):
    def test_options_validate_shift_and_smoothing_settings(self):
        """The shift and smoothing controls define a finite, nonzero q."""
        defaults = MP2SmallQOptions()
        self.assertEqual(defaults.relative_shift, (0.5, 0.5, 0.5))
        self.assertEqual(defaults.eta, 4.0)
        self.assertEqual(defaults.cutoff, "ws")
        self.assertEqual(
            MP2SmallQOptions((0.0, -0.5, 0.25), 3.0, "SPH").cutoff,
            "sph",
        )
        for shift in ((0, 0, 0), (0.6, 0, 0), (0, np.nan, 0), (0, 1)):
            with self.subTest(shift=shift), self.assertRaises(ValueError):
                MP2SmallQOptions(relative_shift=shift)
        with self.assertRaises(ValueError):
            MP2SmallQOptions(eta=0)
        with self.assertRaises(ValueError):
            MP2SmallQOptions(cutoff="ewald")

    def test_mp2ss_smallq_is_opt_in_and_old_backend_options_are_removed(self):
        """The new configuration is optional and replaces backend switches."""
        self.assertIsNone(MP2SSOptions().smallq)
        self.assertEqual(
            MP2SSOptions(smallq={"relative_shift": (0, 0, 0.5)}).smallq,
            MP2SmallQOptions((0, 0, 0.5)),
        )
        with self.assertRaises(TypeError):
            MP2SSOptions(smallq_band_df="GDF")


class MP2SmallQScientificTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cell = make_h2_cell()
        cls.kpts = cls.cell.make_kpts(
            (1, 1, 1), wrap_around=True, with_gamma_point=True
        )
        cls.kmf = scf.KRHF(cls.cell, cls.kpts, exxdiv="ewald")
        cls.kmf.with_df = df.GDF(cls.cell, cls.kpts).build()
        cls.kmf.conv_tol = 1e-9
        cls.kmf.kernel()
        if not cls.kmf.converged:
            raise RuntimeError("small-q H2 reference KRHF did not converge")
        cls.kmp = mp.KMP2(cls.kmf)
        cls.kmp.with_df_ints = True
        cls.kmp.kernel(with_t2=False)

    def _compare_to_explicit_driver(self, pair_grid):
        options = MP2SmallQOptions(relative_shift=(0.0, 0.0, 0.5))
        N_local = (7, 7, 7)
        (
            smallq,
            grids,
            reference_grids,
            structure_factor,
            kmp_ref,
            ordinary_df,
            coeff_occ,
            energy_occ,
            coeff_plus,
            energy_plus,
            coeff_minus,
            energy_minus,
        ) = make_reference_pair(
            self.kmf, self.kmp, options, N_local, pair_grid
        )
        try:
            # Preserve a supplied occupied-energy shift and verify its effect
            # through an independent explicit-denominator contraction.
            original_energy = self.kmp.mo_energy
            shifted_energy = [np.asarray(values).copy() for values in original_energy]
            shifted_energy[0][0] += 0.013
            self.kmp.mo_energy = shifted_energy
            kmp_ref.mo_energy = shifted_energy
            try:
                smallq.kmp = self.kmp
                shifted_result = smallq.kernel()
                _, shifted_energy_occ = kmp2._add_padding(
                    self.kmp, self.kmp.mo_coeff, self.kmp.mo_energy
                )
                np.testing.assert_allclose(
                    np.asarray(shifted_energy_occ)[0, :self.kmp.nocc],
                    shifted_energy[0][:self.kmp.nocc],
                    atol=1e-12,
                )
                shifted_reference = evaluate_explicit_reference(
                    structure_factor,
                    grids,
                    reference_grids,
                    kmp_ref,
                    coeff_occ,
                    shifted_energy_occ,
                    coeff_plus,
                    energy_plus,
                    coeff_minus,
                    energy_minus,
                )
            finally:
                self.kmp.mo_energy = original_energy

            np.testing.assert_allclose(
                shifted_result.sq_direct,
                shifted_reference["SqG_full_direct"][0],
                rtol=2e-8,
                atol=2e-11,
            )
            np.testing.assert_allclose(
                shifted_result.sq_q4,
                shifted_reference["SqG_full_q4"][0],
                rtol=2e-8,
                atol=2e-11,
            )
            self.assertEqual(shifted_result.cutoff, "ws")
            self.assertEqual(shifted_result.eta, 4.0)
            self.assertEqual(shifted_result.qprime[2] > 0, True)
        finally:
            handle = ordinary_df._cderi_to_save
            if not isinstance(handle, str) and not handle.closed:
                handle.close()

    def test_uniform_pair_density_matches_explicit_ki_driver(self):
        """Uniform-grid direct and q4 values match explicit exact gaps."""
        self._compare_to_explicit_driver("uniform")

    def test_becke_pair_density_matches_explicit_ki_driver(self):
        """Becke quadrature uses the same pair-density and normalization rules."""
        self._compare_to_explicit_driver("becke")

    def test_stc_virtual_bands_match_independent_pyscf_builder(self):
        """Stored sTC bands use the original SCF k-mesh cutoff anchor."""
        from pyscf.pbc.df import rsdf_stc

        options = MP2SmallQOptions(relative_shift=(0.0, 0.0, 0.5))
        calc = MP2SmallQ(self.kmf, self.kmp, options, N_local=(5, 5, 5))
        grids = calc._build_grids()
        band_mf, stc_df, j_df = calc._make_band_mean_field(grids.plus)

        # The helper makes an independent sTC/GDF pair on a fresh source copy.
        source_df = self.kmf.with_df
        ref_source = copy.copy(self.kmf)
        ref_source.with_df = df.GDF(self.cell, self.kmf.kpts)
        ref_source.rsjk = None
        ref_mf = rsdf_stc.density_fit(
            ref_source,
            auxbasis=source_df.auxbasis,
            mesh=None if source_df.mesh is None else np.asarray(source_df.mesh).copy(),
            eta=options.eta,
            exxdiv="vcut_ws",
            rc_type="ws",
        )
        try:
            dm_kpts = self.kmf.make_rdm1()
            energy, coeff = band_mf.get_bands(
                grids.plus, dm_kpts=dm_kpts, kpts=self.kmf.kpts
            )
            ref_energy, ref_coeff = ref_mf.get_bands(
                grids.plus, dm_kpts=dm_kpts, kpts=self.kmf.kpts
            )
            np.testing.assert_allclose(energy, ref_energy, atol=1e-10)
            overlap = self.kmf.get_ovlp(kpts=grids.plus)
            for actual, expected, s1e in zip(coeff, ref_coeff, overlap):
                mo_overlap = actual.conj().T @ s1e @ expected
                np.testing.assert_allclose(
                    np.abs(mo_overlap), np.eye(mo_overlap.shape[0]), atol=1e-7
                )

            expected_mesh = np.asarray(
                get_monkhorst_pack_size(self.cell, self.kmf.kpts)
            )
            np.testing.assert_array_equal(stc_df._scf_kpts, self.kmf.kpts)
            np.testing.assert_array_equal(stc_df._stc_kmesh, expected_mesh)
            np.testing.assert_array_equal(
                ref_mf.with_df._stc_kmesh, expected_mesh
            )
        finally:
            for obj in (
                j_df,
                stc_df,
                ref_mf.with_df,
                ref_mf.with_df.with_df_j,
            ):
                handle = getattr(obj, "_cderi_to_save", None)
                if handle is not None and not isinstance(handle, str) and not handle.closed:
                    handle.close()

    def test_uniform_pair_density_converges_on_refined_grid(self):
        """Refining quadrature reduces error while holding the bands fixed."""
        calc = MP2SmallQ(
            self.kmf, self.kmp,
            MP2SmallQOptions(relative_shift=(0.0, 0.0, 0.5)),
            pair_density_eval_grid="uniform",
        )
        grids = calc._build_grids()
        (_, coeff_plus), (_, coeff_minus) = calc._get_shifted_bands(grids)
        densities = []
        for mesh_size in (7, 13, 25):
            calc.N_local = np.full(3, mesh_size)
            densities.append(np.asarray(calc._build_pair_densities(
                grids, self.kmp.mo_coeff, coeff_plus, coeff_minus,
            )))
        coarse_error = np.max(np.abs(densities[0] - densities[-1]))
        finer_error = np.max(np.abs(densities[1] - densities[-1]))
        self.assertLess(finer_error, 0.1 * coarse_error)
        self.assertLess(finer_error, 1e-3)

    def test_custom_shift_mapping_keeps_tiny_and_signed_half_shifts(self):
        """Signed halves reuse bands; an arbitrary tiny shift stays distinct."""
        for shift, self_inverse in (
            ((0.0, -0.5, 0.0), True),
            ((0.17, -0.1, 0.23), False),
            ((1e-12, 0.0, 0.0), False),
        ):
            with self.subTest(shift=shift):
                calc = MP2SmallQ(
                    self.kmf,
                    self.kmp,
                    MP2SmallQOptions(relative_shift=shift),
                    N_local=(3, 3, 3),
                )
                grids = calc._build_grids()
                self.assertNotEqual(np.linalg.norm(grids.qprime), 0.0)
                self.assertEqual(grids.minus_from_plus is not None, self_inverse)
                np.testing.assert_allclose(
                    grids.plus,
                    minimum_image(self.cell, grids.occupied + grids.qprime),
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    grids.minus,
                    minimum_image(self.cell, grids.occupied - grids.qprime),
                    atol=1e-12,
                )

    def test_source_objects_and_gdf_files_survive_success_and_band_failure(self):
        """Band work leaves source arrays, DF caches, and RSJK intact."""
        original_rsjk = getattr(self.kmf, "rsjk", None)
        try:
            self._assert_source_state_survives()
        finally:
            self.kmf.rsjk = original_rsjk

    def _assert_source_state_survives(self):
        """Temporary sTC/J objects close without changing source SCF state."""
        source_df = self.kmf.with_df
        source_cderi = source_df._cderi
        source_auxcell = source_df.auxcell
        source_exxdiv = self.kmf.exxdiv
        source_rsjk = object()
        self.kmf.rsjk = source_rsjk
        energy_before = [np.asarray(x).copy() for x in self.kmp.mo_energy]
        coeff_before = [np.asarray(x).copy() for x in self.kmp.mo_coeff]
        calc = MP2SmallQ(
            self.kmf,
            self.kmp,
            MP2SmallQOptions(relative_shift=(0.0, 0.0, 0.5)),
            N_local=(5, 5, 5),
            pair_density_eval_grid="uniform",
        )
        result = calc.kernel()
        self.assertTrue(np.isfinite(result.sq_direct))
        self.assertIs(self.kmf.with_df, source_df)
        self.assertEqual(source_df._cderi, source_cderi)
        self.assertIs(source_df.auxcell, source_auxcell)
        self.assertEqual(self.kmf.exxdiv, source_exxdiv)
        self.assertIs(self.kmf.rsjk, source_rsjk)
        for actual, expected in zip(self.kmp.mo_energy, energy_before):
            np.testing.assert_array_equal(actual, expected)
        for actual, expected in zip(self.kmp.mo_coeff, coeff_before):
            np.testing.assert_array_equal(actual, expected)

        from pyscf.pbc.df import rsdf_stc

        grids = calc._build_grids()
        created_j = []
        created_stc = []
        original_gdf_build = df.GDF.build

        def record_j_build(obj, *args, **kwargs):
            if obj is not source_df:
                created_j.append(obj)
            return original_gdf_build(obj, *args, **kwargs)

        def fail_stc_build(obj, *args, **kwargs):
            created_stc.append(obj)
            raise RuntimeError("injected sTC band build failure")

        with mock.patch.object(df.GDF, "build", record_j_build), mock.patch.object(
            rsdf_stc.RSGDF_STC, "build", fail_stc_build
        ):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                calc._get_shifted_bands(grids)

        self.assertTrue(created_j)
        self.assertTrue(created_stc)
        self.assertTrue(created_j[0]._cderi_to_save.closed)
        self.assertTrue(created_stc[0]._cderi_to_save.closed)
        self.assertIs(self.kmf.with_df, source_df)
        self.assertEqual(source_df._cderi, source_cderi)
        self.assertIs(self.kmf.rsjk, source_rsjk)


class MP2SmallQUnwrappedKPointTests(unittest.TestCase):
    def test_unwrapped_regular_mesh_uses_matching_df_kpoint_labels(self):
        """Nk=2, frozen padding, and unwrapped labels match the ki reference."""
        cell = make_h2_cell()
        kpts = cell.make_kpts(
            (1, 1, 2), wrap_around=False, with_gamma_point=True
        )
        self.assertGreater(kpts[1, 2], 0.0)
        kmf = scf.KRHF(cell, kpts, exxdiv="ewald")
        kmf.with_df = df.GDF(cell, kpts).build()
        kmf.conv_tol = 1e-9
        kmf.kernel()
        kmp = mp.KMP2(kmf, frozen=[[], [1]])
        kmp.with_df_ints = True
        kmp.kernel(with_t2=False)
        _, active_virtual = kmp2.padding_k_idx(kmp, kind="split")
        self.assertEqual(len(active_virtual[0]), 1)
        self.assertEqual(len(active_virtual[1]), 0)
        options = MP2SmallQOptions(relative_shift=(0.0, 0.0, 0.5))
        N_local = (5, 5, 5)
        (
            smallq,
            grids,
            reference_grids,
            structure_factor,
            kmp_ref,
            ordinary_df,
            coeff_occ,
            energy_occ,
            coeff_plus,
            energy_plus,
            coeff_minus,
            energy_minus,
        ) = make_reference_pair(kmf, kmp, options, N_local, "uniform")
        try:
            reference = evaluate_explicit_reference(
                structure_factor,
                grids,
                reference_grids,
                kmp_ref,
                coeff_occ,
                energy_occ,
                coeff_plus,
                energy_plus,
                coeff_minus,
                energy_minus,
            )
            result = smallq.kernel()
            np.testing.assert_allclose(
                result.sq_direct,
                reference["SqG_full_direct"][0],
                rtol=2e-8,
                atol=2e-11,
            )
            np.testing.assert_allclose(
                result.sq_q4,
                reference["SqG_full_q4"][0],
                rtol=2e-8,
                atol=2e-11,
            )
        finally:
            handle = ordinary_df._cderi_to_save
            if not isinstance(handle, str) and not handle.closed:
                handle.close()

        self.assertTrue(np.isfinite(result.sq_direct))
        self.assertTrue(np.isfinite(result.sq_q4))


class MP2SSSmallQFitIntegrationTests(unittest.TestCase):
    @staticmethod
    def _bare_mp2ss(separate):
        mp2ss = MP2SS.__new__(MP2SS)
        mp2ss.correct_q2_q4_separately = separate
        mp2ss.dG0 = True
        mp2ss.grids = object()
        mp2ss.nks = np.asarray((1, 1, 1))
        mp2ss._build_direct_second_order_correction_config = lambda: object()
        mp2ss._build_direct_fourth_order_correction_config = lambda: object()
        mp2ss._build_direct_full_correction_config = lambda: object()
        mp2ss.mp2_structure_factor = SimpleNamespace(
            SqG_full_direct=np.asarray((0.5, 0.6)),
            SqG_full_q4=np.asarray((0.01, 0.02)),
            qG_full=np.asarray(((1.0, 0.0, 0.0), (0.0, 2.0, 0.0))),
            SqG_full_direct_mask=np.asarray((True, True)),
            SqG_full_q4_mask=np.asarray((True, True)),
        )
        mp2ss.mp2_structure_factor.smallq_result = SimpleNamespace(
            qprime=np.asarray((0.1, 0.0, 0.0)),
            sq_direct=0.2,
            sq_q4=0.03,
        )
        return mp2ss

    def test_combined_direct_fit_gets_appended_smallq_sample(self):
        """The combined fit receives an extra sample without changing data."""
        mp2ss = self._bare_mp2ss(separate=False)
        observed = {}

        class Direct:
            def __init__(self, config):
                pass

            def compute_direct_correction(self, **kwargs):
                observed.update(kwargs)
                return SimpleNamespace(
                    direct_integral_term=1.0,
                    direct_quadrature_term=2.0,
                    total_direct_correction=3.0,
                )

        with mock.patch(
            "fsec.singularity_subtraction.mp2ss.MP2DirectFullSS", Direct
        ):
            self.assertEqual(mp2ss.compute_direct_correction(), 3.0)
        self.assertEqual(len(observed["qG_full"]), 2)
        self.assertEqual(len(observed["SqG_full_direct"]), 2)
        self.assertEqual(len(observed["fit_qG_full"]), 3)
        self.assertEqual(len(observed["fit_SqG_full_direct"]), 3)

    def test_separate_direct_fits_get_q2_and_q4_smallq_samples(self):
        """Separate q2/q4 fits receive their matching derived small-q values."""
        mp2ss = self._bare_mp2ss(separate=True)
        observed = {}

        class SecondOrder:
            def __init__(self, config):
                pass

            def compute_correction(self, **kwargs):
                observed["q2"] = kwargs
                return (1.0, 2.0, 3.0)

        class FourthOrder:
            def __init__(self, config):
                pass

            def compute_correction(self, **kwargs):
                observed["q4"] = kwargs
                return (4.0, 5.0, 9.0)

        class Full:
            def __init__(self, config):
                pass

        with mock.patch(
            "fsec.singularity_subtraction.mp2ss.MP2DirectSecondOrderSS",
            SecondOrder,
        ), mock.patch(
            "fsec.singularity_subtraction.mp2ss.MP2DirectFourthOrderSS",
            FourthOrder,
        ), mock.patch(
            "fsec.singularity_subtraction.mp2ss.MP2DirectFullSS", Full
        ):
            self.assertEqual(mp2ss.compute_direct_correction(), 12.0)
        self.assertEqual(len(observed["q2"]["qG_full"]), 3)
        self.assertAlmostEqual(
            observed["q2"]["SqG_full_q2_part"][-1],
            0.2 - 4 * np.pi * 0.03 / 0.01,
        )
        self.assertEqual(len(observed["q4"]["qG_full"]), 3)
        self.assertAlmostEqual(observed["q4"]["SqG_full_q4"][-1], 0.03)
        self.assertTrue(observed["q4"]["q4_fit_mask"][-1])


if __name__ == "__main__":
    unittest.main()
