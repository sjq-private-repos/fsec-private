import contextlib
import io
import unittest

import numpy as np

from fsec.singularity_subtraction.mp2_variants import get_mp2_variant
from fsec.singularity_subtraction.structure_factor.mp2_sf import (
    compute_mp2_variant_energies,
    MP2StructureFactor,
)
from fsec.singularity_subtraction.structure_factor.mp2_smallq import MP2SmallQ


class MP2VariantStructureFactorRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import the fixture locally so unittest discovery does not collect
        # the imported legacy TestCase a second time from this module.
        from fsec.singularity_subtraction.structure_factor.tests.test_mp2_structure_factor import (
            KnownValues as structure_factor_known_values,
        )

        structure_factor_known_values.setUpClass()
        cls.kmf = structure_factor_known_values.kmf
        cls.kmp = structure_factor_known_values.kmp
        cls.t2 = structure_factor_known_values.t2
        cls.N_local = structure_factor_known_values.N_local
        reciprocal = cls.kmf.cell.reciprocal_vectors()
        cls.qG_full = np.array([
            [0.0, 0.0, 0.0],
            reciprocal[0],
            -reciprocal[0],
        ])

    def _build(self, variant, *, t2=None, t2_store_type="kikjka",
               laplace=True, direct=True, exchange=True, dG0=True,
               check_trs=False):
        with contextlib.redirect_stdout(io.StringIO()):
            structure_factor = MP2StructureFactor(
                self.kmf,
                self.kmp,
                t2=t2,
                N_local=self.N_local,
                qG_cutoff=8.0,
                min_points=10,
                sq_inversion_symm=False,
                check_trs=check_trs,
                pair_density_eval_grid="uniform",
                t2_store_type=t2_store_type,
                laplace=laplace,
                variant=variant,
            )
            result = structure_factor.build_structure_factor(
                qG_full=self.qG_full,
                direct=direct,
                exchange=exchange,
                dG0=dG0,
            )
        return structure_factor, result

    def test_sos_scales_direct_and_q4_once_and_skips_exchange(self):
        _, mp2 = self._build(get_mp2_variant("mp2"), t2=self.t2)
        _, sos = self._build(get_mp2_variant("sos"), t2=self.t2)

        np.testing.assert_allclose(
            sos["SqG_full_direct"],
            0.65 * mp2["SqG_full_direct"],
            rtol=1e-11,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            sos["SqG_full_q4"],
            0.65 * mp2["SqG_full_q4"],
            rtol=1e-11,
            atol=1e-14,
        )
        self.assertTrue(np.array_equal(
            sos["SqG_full_exchange"],
            np.zeros_like(sos["SqG_full_exchange"]),
        ))

        # An exchange-only request with a zero SOS exchange scale is a no-op,
        # rather than an attempt to enter an exchange contraction.
        _, exchange_only = self._build(
            get_mp2_variant("sos"),
            direct=False,
            exchange=True,
            dG0=False,
        )
        self.assertTrue(np.array_equal(
            exchange_only["SqG_full_exchange"],
            np.zeros_like(exchange_only["SqG_full_exchange"]),
        ))

    def test_supplied_ordinary_t2_matches_internal_regularized_t2(self):
        supplied_t2 = self.t2.copy()
        original_t2 = supplied_t2.copy()
        supplied_sf, supplied = self._build(
            get_mp2_variant("kappa", 1.1), t2=supplied_t2,
            laplace=False,
        )
        _, internal = self._build(
            get_mp2_variant("kappa", 1.1), t2=None, laplace=False,
        )

        np.testing.assert_allclose(supplied_t2, original_t2)
        for key in ("SqG_full_direct", "SqG_full_exchange", "SqG_full_q4"):
            np.testing.assert_allclose(
                supplied[key], internal[key], rtol=1e-10, atol=1e-13)
        self.assertEqual(supplied_sf.variant.name, "kappa-mp2")

    def test_low_memory_and_lov_routes_use_regularized_policy(self):
        _, full = self._build(
            get_mp2_variant("sigma", 0.7), laplace=False,
        )
        _, low_memory = self._build(
            get_mp2_variant("sigma", 0.7),
            t2_store_type="ki",
            laplace=False,
        )
        for key in ("SqG_full_direct", "SqG_full_exchange", "SqG_full_q4"):
            np.testing.assert_allclose(
                low_memory[key], full[key], rtol=2e-7, atol=1e-12)

        lov_sf, lov = self._build(
            get_mp2_variant("sigma2", 0.4),
            t2_store_type="kikj_lov",
            laplace=True,
            check_trs=True,
        )
        for key in ("SqG_full_direct", "SqG_full_exchange", "SqG_full_q4"):
            self.assertTrue(np.all(np.isfinite(lov[key])))
        self.assertIn(
            "TRS Lov direct/q4 denominator build",
            lov_sf.last_build_timings,
        )

    def test_large_strength_energy_recovers_ordinary_mp2(self):
        ordinary = compute_mp2_variant_energies(
            self.kmp,
            variant=get_mp2_variant("mp2"),
        )
        limiting = compute_mp2_variant_energies(
            self.kmp,
            variant=get_mp2_variant("sigma", 1e6),
            t2=self.t2,
        )
        np.testing.assert_allclose(limiting, ordinary, rtol=1e-8, atol=1e-10)


class MP2VariantSmallQRoute(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Keep the imported legacy fixture out of this module's discovery.
        from fsec.singularity_subtraction.structure_factor.tests.test_mp2_smallq import (
            MP2SmallQKnownValues as smallq_known_values,
        )

        smallq_known_values.setUpClass()
        cls.kmf = smallq_known_values.kmf
        cls.kmp = smallq_known_values.kmp
        cls.N_local = smallq_known_values.N_local

    def test_smallq_receives_and_retains_variant(self):
        variant = get_mp2_variant("sigma", 0.7)
        calculation = MP2SmallQ(
            self.kmf,
            self.kmp,
            band_df="FFTDF",
            band_exxdiv="ewald",
            N_local=self.N_local,
            pair_density_eval_grid="uniform",
            check_trs=False,
            variant=variant,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            result = calculation.kernel()
        self.assertEqual(calculation.variant, variant)
        self.assertEqual(
            calculation.smallq_structure_factor.variant,
            variant,
        )
        self.assertTrue(np.isfinite(result.sq_direct))
        self.assertTrue(np.isfinite(result.sq_q4))


if __name__ == "__main__":
    unittest.main()
