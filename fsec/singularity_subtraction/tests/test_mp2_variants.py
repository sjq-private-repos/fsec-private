import unittest
from dataclasses import FrozenInstanceError
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace

import numpy as np
from pyscf.lib.parameters import LARGE_DENOM

from fsec.singularity_subtraction.mp2_variants import get_mp2_variant
from fsec.singularity_subtraction.mp2ss import MP2SS, MP2SSOptions
from fsec.singularity_subtraction.structure_factor.mp2_contractions import (
    contract_trs_unique_pair_rijab,
    contract_trs_unique_pair_rijab_lov,
    contract_trs_unique_pair_rijab_lov_laplace,
)


class MP2VariantPolicyTests(unittest.TestCase):
    def test_aliases_and_preset_scales(self):
        presets = {
            "mp2": (1.0, 1.0, 1.0, 1.0),
            "SOS": (1.3, 0.0, 0.65, 0.0),
            "SCS-MP2": (1.2, 1.0 / 3.0, (1.2 + 1.0 / 3.0) / 2, 1.0 / 3.0),
            "scs_mi": (0.40, 1.29, (0.40 + 1.29) / 2, 1.29),
        }
        for name, (c_os, c_ss, direct, exchange) in presets.items():
            with self.subTest(method=name):
                variant = get_mp2_variant(name)
                self.assertEqual(variant.cOS, c_os)
                self.assertEqual(variant.cSS, c_ss)
                self.assertEqual(variant.direct_scale, direct)
                self.assertEqual(variant.exchange_scale, exchange)

        self.assertEqual(get_mp2_variant("κ-MP2").regularization_strength, 1.1)
        self.assertEqual(get_mp2_variant("SIGMA_MP2").regularization_strength, 0.7)
        self.assertEqual(get_mp2_variant("sosmp2").name, "sos-mp2")
        self.assertEqual(
            get_mp2_variant("sigma-squared-mp2").regularization_strength,
            0.4,
        )
        self.assertEqual(get_mp2_variant("sigma^2 MP2").name, "sigma2-mp2")
        self.assertEqual(get_mp2_variant("σ²-MP2").name, "sigma2-mp2")

    def test_options_normalize_and_variant_is_immutable(self):
        options = MP2SSOptions(
            correlation_method="KAPPA",
            regularization_strength="0.25",
        )
        self.assertEqual(options.correlation_method, "kappa-mp2")
        self.assertEqual(options.regularization_strength, 0.25)
        variant = get_mp2_variant("mp2")
        with self.assertRaises(FrozenInstanceError):
            variant.c_os = 2.0

    def test_invalid_methods_and_strengths(self):
        for method in ("not-mp2", "", None, 2):
            with self.subTest(method=method):
                with self.assertRaises((TypeError, ValueError)):
                    get_mp2_variant(method)
        for strength in (-1, np.nan, np.inf, "not-a-number"):
            with self.subTest(strength=strength):
                with self.assertRaises(ValueError):
                    get_mp2_variant("sigma", strength)
        with self.assertRaises(ValueError):
            get_mp2_variant("mp2", 0.1)
        with self.assertRaises(ValueError):
            MP2SSOptions(correlation_method="sos-mp2", regularization_strength=0.1)

    def test_damping_reciprocal_limits_and_padding_mask(self):
        gaps = np.array([0.0, 1e-12, 0.35, 1.7])
        signed = -gaps
        expected_zero_gap = {
            "kappa": 0.0,
            "sigma": -0.7,
            "sigma2": 0.0,
        }
        for method, regularizer in (
            ("kappa", "kappa"),
            ("sigma", "sigma"),
            ("sigma2", "sigma2"),
        ):
            variant = get_mp2_variant(method, 0.7)
            damping = variant.damping_from_gap(gaps)
            if regularizer == "kappa":
                expected_damping = -np.expm1(-0.7 * gaps)
                expected_damping **= 2
            elif regularizer == "sigma":
                expected_damping = -np.expm1(-0.7 * gaps)
            else:
                expected_damping = -np.expm1(-0.7 * gaps * gaps)
            np.testing.assert_allclose(damping, expected_damping, rtol=1e-14)
            reciprocal = variant.reciprocal(signed)
            self.assertAlmostEqual(reciprocal[0], expected_zero_gap[regularizer])

        # The inactive padded entry is deliberately a huge positive value.
        # It must not be selected or exponentiated by a regularizer.
        padded_denominator = np.array([-0.8, LARGE_DENOM, -1.2])
        active = np.array([True, False, True])
        variant = get_mp2_variant("kappa", 1.1)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            damping = variant.damping(padded_denominator, active)
            reciprocal = variant.reciprocal(padded_denominator, active)
        self.assertEqual(damping[1], 0.0)
        self.assertEqual(reciprocal[1], 0.0)

    def test_strength_limits_and_laplace_expansions(self):
        gaps = np.array([0.2, 0.8, 2.1])
        for method in ("kappa", "sigma", "sigma2"):
            zero = get_mp2_variant(method, 0.0)
            np.testing.assert_allclose(zero.damping_from_gap(gaps), 0.0)
            np.testing.assert_allclose(zero.reciprocal_from_gap(gaps), 0.0)

        for method in ("kappa", "sigma"):
            variant = get_mp2_variant(method, 0.7)
            signed_terms = variant.laplace_terms()
            absolute_terms = variant.laplace_terms(absolute=True)
            signed = sum(
                coefficient * np.exp(-shift * gaps) / gaps
                for shift, coefficient in signed_terms
            )
            absolute = sum(
                coefficient * np.exp(-shift * gaps) / gaps
                for shift, coefficient in absolute_terms
            )
            np.testing.assert_allclose(signed, variant.reciprocal_from_gap(gaps))
            np.testing.assert_allclose(
                absolute, variant.absolute_reciprocal_from_gap(gaps))

        self.assertIsNone(get_mp2_variant("sigma2").laplace_terms())
        large = get_mp2_variant("sigma", 1e6)
        np.testing.assert_allclose(
            large.reciprocal_from_gap(gaps), -1.0 / gaps, rtol=1e-12)

    def test_spin_component_decomposition(self):
        e_os = -0.12
        e_ss = 0.04
        variant = get_mp2_variant("scs-mi-mp2")
        direct = variant.direct_scale * (2.0 * e_os)
        exchange = variant.exchange_scale * (e_ss - e_os)
        self.assertAlmostEqual(
            direct + exchange,
            variant.cOS * e_os + variant.cSS * e_ss,
        )

    def test_restricted_gapped_applicability_validation(self):
        unrestricted = SimpleNamespace(
            mo_coeff=(np.eye(2), np.eye(2)),
            mo_occ=np.array([1.0, 1.0]),
        )
        restricted = SimpleNamespace(
            mo_coeff=np.eye(2),
            mo_occ=np.array([2.0, 0.0]),
        )
        gapless = SimpleNamespace(
            mo_energy=np.array([[-0.5, -0.5]]),
            nocc=1,
        )
        indirect_overlap = SimpleNamespace(
            # Both direct gaps are positive, but the valence maximum at the
            # second k-point lies above the conduction minimum at the first.
            mo_energy=np.array([[-1.0, -0.2], [0.2, 1.0]]),
            nocc=1,
        )
        with self.assertRaisesRegex(ValueError, "unrestricted"):
            MP2SS._validate_reference(unrestricted, gapless)
        with self.assertRaisesRegex(ValueError, "gapped"):
            MP2SS._validate_reference(restricted, gapless)
        with self.assertRaisesRegex(ValueError, "gapped"):
            MP2SS._validate_reference(restricted, indirect_overlap)


class RegularizedContractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(23)
        cls.nkpts = 1
        cls.nocc = 2
        cls.nvir = 2
        cls.naux = 3
        cls.rho_ia = rng.normal(size=(1, 2, 2)) + 1j * rng.normal(size=(1, 2, 2))
        cls.rho_jb = rng.normal(size=(1, 2, 2)) + 1j * rng.normal(size=(1, 2, 2))
        block = rng.normal(size=(3, 2, 2)) + 1j * rng.normal(size=(3, 2, 2))
        block_b = rng.normal(size=(3, 2, 2)) + 1j * rng.normal(size=(3, 2, 2))
        cls.Lov = np.empty((1, 1), dtype=object)
        cls.Lov_b = np.empty((1, 1), dtype=object)
        cls.Lov[0, 0] = block
        cls.Lov_b[0, 0] = block_b
        cls.mo_e_o = np.array([[-1.3, -0.9]])
        cls.mo_e_v = np.array([[0.45, 0.85]])
        cls.mo_e_v_b = np.array([[0.55, 0.95]])
        cls.nonzero = [np.arange(2)]
        cls.pairs = [(0, 0, 1)]
        cls.rijab = np.einsum(
            "mia,nbj->mnijab", cls.rho_ia, cls.rho_jb.conj())

    def _reference(self, variant):
        eia = self.mo_e_o[0, :, None] - self.mo_e_v[0]
        ejb = self.mo_e_o[0, :, None] - self.mo_e_v_b[0]
        direct_denominator = eia[:, None, :, None] + ejb[None, :, None, :]
        direct_reciprocal = variant.reciprocal(direct_denominator)
        direct_oovv = (
            np.einsum("Lia,Ljb->iajb", self.Lov[0, 0], self.Lov_b[0, 0])
            .transpose(0, 2, 1, 3)
        )
        direct_t2 = direct_oovv.conj() * direct_reciprocal

        eib = self.mo_e_o[0, :, None] - self.mo_e_v_b[0]
        eja = self.mo_e_o[0, :, None] - self.mo_e_v[0]
        exchange_denominator = eib[:, None, :, None] + eja[None, :, None, :]
        exchange_reciprocal = variant.reciprocal(exchange_denominator)
        exchange_oovv = np.einsum(
            "Lib,Lja->ijba", self.Lov_b[0, 0], self.Lov[0, 0])
        exchange_t2 = exchange_oovv.conj() * exchange_reciprocal
        return contract_trs_unique_pair_rijab(
            self.rho_ia,
            self.rho_jb,
            self.pairs,
            direct_t2=direct_t2[None, None],
            exchange_t2=exchange_t2[None, None],
            eijab_recip=direct_reciprocal[None, None],
        )

    def _lov(self, variant, laplace=False):
        function = (
            contract_trs_unique_pair_rijab_lov_laplace
            if laplace else contract_trs_unique_pair_rijab_lov
        )
        kwargs = dict(
            compute_direct=True,
            compute_exchange=True,
            compute_q4=True,
            variant=variant,
        )
        if laplace:
            kwargs.update(
                direct_tolerance=1e-8,
                direct_max_points=16,
                exchange_tolerance=1e-8,
                exchange_max_points=16,
            )
        return function(
            self.rho_ia,
            self.rho_jb,
            self.pairs,
            self.Lov,
            self.Lov_b,
            np.array([0]),
            np.array([0]),
            self.mo_e_o,
            self.mo_e_v,
            self.mo_e_v_b,
            self.nonzero,
            self.nonzero,
            self.nkpts,
            **kwargs,
        )

    def test_exact_route_matches_explicit_numpy_reference(self):
        variant = get_mp2_variant("sigma", 0.7)
        expected = self._reference(variant)
        actual = self._lov(variant)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_kappa_and_sigma_laplace_match_exact_and_sigma2_falls_back(self):
        for method in ("kappa", "sigma", "sigma2"):
            with self.subTest(method=method):
                variant = get_mp2_variant(method, 0.7)
                exact = self._lov(variant)
                laplace = self._lov(variant, laplace=True)
                np.testing.assert_allclose(
                    laplace, exact, rtol=3e-8, atol=1e-10)


class MP2SSEnergyIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Keep the reference TestCase local so unittest does not collect it
        # again from this module.
        from fsec.singularity_subtraction.tests.test_mp2ss import (
            KnownValues as mp2ss_known_values,
        )

        mp2ss_known_values.setUpClass()
        cls.kmf = mp2ss_known_values.kmf
        cls.kmp = mp2ss_known_values.kmp
        cls.t2 = mp2ss_known_values.t2

    def test_regularized_energy_identity_and_public_attributes(self):
        from fsec.singularity_subtraction import MP2SS

        options = MP2SSOptions(
            correlation_method="sigma2",
            regularization_strength=0.4,
            pair_density_eval_grid="uniform",
            qG_norm_cutoff=4.0,
        )
        original_t2 = self.t2.copy()
        with redirect_stdout(StringIO()):
            mp2ss = MP2SS(
                self.kmf,
                self.kmp,
                t2=self.t2,
                options=options,
            )
            correction = mp2ss.compute_correction()

        self.assertEqual(mp2ss.correlation_method, "sigma2-mp2")
        self.assertTrue(np.isfinite(mp2ss.e_corr_uncorrected))
        self.assertTrue(np.isfinite(mp2ss.e_corr_corrected))
        self.assertAlmostEqual(
            mp2ss.e_corr_corrected,
            mp2ss.e_corr_uncorrected + float(correction),
        )
        self.assertEqual(mp2ss.e_corr_uncorrected, mp2ss.emp2_uncorr)
        self.assertEqual(mp2ss.e_corr_corrected, mp2ss.emp2ss)
        np.testing.assert_allclose(self.t2, original_t2)


if __name__ == "__main__":
    unittest.main()
