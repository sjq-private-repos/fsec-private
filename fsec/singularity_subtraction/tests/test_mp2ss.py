import unittest
from contextlib import redirect_stdout
from io import StringIO
import numpy as np
from pyscf.pbc import df
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import mp
from pyscf.pbc import scf as pbcscf

from fsec.singularity_subtraction import MP2SS
from fsec.singularity_subtraction.mp2ss import (
    MP2DirectFourthOrderSS,
    MP2DirectFullSS,
    MP2DirectSecondOrderSS,
    MP2ExchangeSS,
    MP2SSOptions,
)


class PrintResultsTests(unittest.TestCase):
    @staticmethod
    def _calculator(calculator_class, integral, quadrature, correction):
        calculator = calculator_class.__new__(calculator_class)
        calculator.integral_term = integral
        calculator.quadrature_term = quadrature
        calculator.correction = correction
        return calculator

    def _mp2ss(self, separate):
        mp2ss = MP2SS.__new__(MP2SS)
        mp2ss.with_df_ints = True
        mp2ss.auxfunc_direct = "direct-model"
        mp2ss.auxfunc_exchange = "exchange-model"
        mp2ss.edi_uncorr = -1.0
        mp2ss.exi_uncorr = 0.25
        mp2ss.emp2_uncorr = -0.75
        mp2ss.emp2ss_direct = -1.1
        mp2ss.emp2ss_exchange = 0.3
        mp2ss.emp2ss = -0.8
        mp2ss.correct_q2_q4_separately = separate
        mp2ss.exchange_correction = self._calculator(MP2ExchangeSS, 4.0, 1.5, 2.5)
        return mp2ss

    def test_print_results_with_full_direct_term(self):
        mp2ss = self._mp2ss(separate=False)
        mp2ss.direct_correction = self._calculator(MP2DirectFullSS, 3.0, 2.0, 1.0)

        output = StringIO()
        with redirect_stdout(output):
            mp2ss.print_results()
        result = output.getvalue()

        self.assertIn("Uncorrected MP2 Energies:", result)
        self.assertIn("Direct Term:\n", result)
        self.assertNotIn("Direct Term (2nd order):", result)
        self.assertIn("Exchange Term:\n", result)
        self.assertIn("Final Energies:", result)
        self.assertIn(" Integral term (hartree)    = 3.0", result)
        self.assertIn(" Total correction (hartree) = 1.0", result)

    def test_print_results_with_separate_direct_terms(self):
        mp2ss = self._mp2ss(separate=True)
        mp2ss.direct_second_order_correction = self._calculator(
            MP2DirectSecondOrderSS, 2.0, 1.5, 0.5
        )
        mp2ss.direct_fourth_order_correction = self._calculator(
            MP2DirectFourthOrderSS, 1.0, 0.5, 0.5
        )
        mp2ss.direct_correction = self._calculator(MP2DirectFullSS, 3.0, 2.0, 1.0)
        mp2ss.direct_correction.results_title = "Direct Term (Full: 2nd + 4th)"

        output = StringIO()
        with redirect_stdout(output):
            mp2ss.print_results()
        result = output.getvalue()

        self.assertIn("Direct Term (2nd order):", result)
        self.assertIn("Direct Term (4th order):", result)
        self.assertIn("Direct Term (Full: 2nd + 4th):", result)
        self.assertNotIn("Direct Term:\n", result)
        self.assertIn("Exchange Term:\n", result)
        self.assertIn("Final Energies:", result)


class MP2SSOptionsTests(unittest.TestCase):
    def test_adaptive_sq_ke_cutoff_options(self):
        options = MP2SSOptions(
            sq_ke_cutoff=100.0,
            sq_ke_cutoff_switch_radius=2.5,
            outer_sq_ke_cutoff_scale=0.25,
        )

        self.assertEqual(options.sq_ke_cutoff, 100.0)
        self.assertEqual(options.sq_ke_cutoff_switch_radius, 2.5)
        self.assertEqual(options.outer_sq_ke_cutoff_scale, 0.25)


class KnownValues(unittest.TestCase):
    TOL = 1e-6
    REFERENCES = {
        "emp2ss": -0.020743652149576734,
        "mp2ss_total_correction": -0.0042350890410775485,
        "mp2ss_direct_correction": -0.013108450253663102,
        "mp2ss_exchange_correction": 0.008873361212585553,
        "direct_integral_term": -0.03495204165711477,
        "direct_quadrature_term": -0.021843591403451663,
        "exchange_integral_term": 0.018810916817992317,
        "exchange_quadrature_term": 0.009937555605406764,
        "direct_integral_term_q2": -0.028229618267205235,
        "direct_quadrature_term_q2": -0.01748981155330432,
        "direct_total_correction_q2": -0.010739806713900915,
        "direct_integral_term_q4": -0.006722423389909531,
        "direct_quadrature_term_q4": -0.004353779850147344,
        "direct_total_correction_q4": -0.0023686435397621865,
        "direct_total_correction_q2_q4": -0.013108450253663102,
        "emp2_uncorr": -0.016508563108499186,
        "edi_uncorr": -0.031180712917439485,
        "exi_uncorr": 0.014672149808940297,
        "emp2ss_direct": -0.04428916317110258,
        "emp2ss_exchange": 0.02354551102152585,
    }

    @classmethod
    def setUpClass(cls):
        cls.kmf, cls.kmp, cls.t2 = cls._run_mp2()

    @staticmethod
    def _build_h2_cell():
        cell = pbcgto.Cell()
        cell.unit = "Bohr"
        cell.atom = """
            H 0.00 0.00 0.00
            H 0.00 0.00 1.80
        """
        cell.a = np.eye(3) * 6.0
        cell.verbose = 0
        cell.spin = 0
        cell.charge = 0
        cell.basis = "gth-szv"
        cell.pseudo = "gth-hf"
        cell.ke_cutoff = 100
        cell.max_memory = 1000
        cell.precision = 1e-8
        cell.build()
        kpts = cell.make_kpts((1, 1, 2), wrap_around=True, with_gamma_point=True)
        return cell, kpts

    @classmethod
    def _run_mp2(cls):
        cell, kpts = cls._build_h2_cell()
        kmf = pbcscf.KRHF(cell, kpts)
        kmf.exxdiv = "ewald"
        kmf.with_df = df.GDF(cell, kpts).build()
        kmf.kernel()
        if not kmf.converged:
            raise RuntimeError("Reference KRHF calculation did not converge")

        kmp = mp.KMP2(kmf)
        kmp.with_df_ints = True
        _, t2 = kmp.kernel(with_t2=True)
        return kmf, kmp, t2

    def _assert_matches_references(self, mp2ss, correction):
        actual = {
            "emp2ss": mp2ss.emp2ss,
            "mp2ss_total_correction": mp2ss.mp2ss_total_correction,
            "mp2ss_direct_correction": mp2ss.mp2ss_direct_correction,
            "mp2ss_exchange_correction": mp2ss.mp2ss_exchange_correction,
            "direct_integral_term": mp2ss.direct_integral_term,
            "direct_quadrature_term": mp2ss.direct_quadrature_term,
            "exchange_integral_term": mp2ss.exchange_integral_term,
            "exchange_quadrature_term": mp2ss.exchange_quadrature_term,
            "direct_integral_term_q2": mp2ss.direct_integral_term_q2,
            "direct_quadrature_term_q2": mp2ss.direct_quadrature_term_q2,
            "direct_total_correction_q2": mp2ss.direct_total_correction_q2,
            "direct_integral_term_q4": mp2ss.direct_integral_term_q4,
            "direct_quadrature_term_q4": mp2ss.direct_quadrature_term_q4,
            "direct_total_correction_q4": mp2ss.direct_total_correction_q4,
            "direct_total_correction_q2_q4": mp2ss.direct_total_correction_q2_q4,
            "emp2_uncorr": mp2ss.emp2_uncorr,
            "edi_uncorr": mp2ss.edi_uncorr,
            "exi_uncorr": mp2ss.exi_uncorr,
            "emp2ss_direct": mp2ss.emp2ss_direct,
            "emp2ss_exchange": mp2ss.emp2ss_exchange,
        }

        for quantity, reference in self.REFERENCES.items():
            self.assertAlmostEqual(
                actual[quantity],
                reference,
                delta=self.TOL,
                msg=quantity,
            )

        self.assertAlmostEqual(float(correction), self.REFERENCES["mp2ss_total_correction"], delta=self.TOL)
        self.assertAlmostEqual(
            correction.mp2ss_direct_correction,
            self.REFERENCES["mp2ss_direct_correction"],
            delta=self.TOL,
        )
        self.assertAlmostEqual(
            correction.mp2ss_exchange_correction,
            self.REFERENCES["mp2ss_exchange_correction"],
            delta=self.TOL,
        )

    def test_mp2ss_stacked_singularity_split_q2_q4(self):
        mp2ss = MP2SS(
            kmf=self.kmf,
            kmp=self.kmp,
            t2=self.t2,
        )
        self.assertEqual(mp2ss.options.auxfunc_direct, "Gauss")
        self.assertEqual(mp2ss.options.auxfunc_direct_q2, "Gauss")
        self.assertEqual(mp2ss.options.auxfunc_direct_q4, "Gauss")
        self.assertEqual(mp2ss.options.auxfunc_exchange, "Gauss")
        self.assertEqual(mp2ss.options.qG_norm_cutoff, 4.0)
        self.assertEqual(mp2ss.options.fit_method, "scipy_least_squares")
        self.assertTrue(mp2ss.options.fit_with_coul)
        self.assertTrue(mp2ss.options.fit_with_coul_q2)
        self.assertTrue(mp2ss.options.correct_q2_q4_separately)
        self.assertTrue(mp2ss.options.check_trs)
        self.assertEqual(mp2ss.options.t2_store_type, "kikjka")
        correction = mp2ss.compute_correction(direct=True, exchange=True)

        self._assert_matches_references(mp2ss, correction)

    def test_mp2ss_check_trs_false(self):
        mp2ss = MP2SS(
            kmf=self.kmf,
            kmp=self.kmp,
            t2=self.t2,
            check_trs=False,
        )
        self.assertFalse(mp2ss.options.check_trs)
        correction = mp2ss.compute_correction(direct=True, exchange=True)

        self.assertFalse(mp2ss.mp2_structure_factor.check_trs)
        self._assert_matches_references(mp2ss, correction)


if __name__ == "__main__":
    unittest.main()
