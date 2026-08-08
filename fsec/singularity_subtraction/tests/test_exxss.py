import unittest
from unittest import mock
import numpy as np
from pyscf.pbc import df
from pyscf.pbc import dft as pbcdft
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbcscf

from fsec.singularity_subtraction import ExxSS, ExxSSQuarticExponential


class ExxUncorrectedCallShapeTests(unittest.TestCase):
    @staticmethod
    def _call_shape(df_class, rsjk=None, direct=False):
        cell = pbcgto.Cell()
        cell.unit = "Bohr"
        cell.atom = "H 0.0 0.0 0.0"
        cell.spin = 1
        cell.a = np.eye(3) * 6.0
        cell.basis = "gth-szv"
        cell.pseudo = "gth-hf"
        cell.build()
        kpts = cell.make_kpts(
            (1, 1, 1), wrap_around=True, with_gamma_point=True)
        dm_kpts = np.zeros((1, cell.nao_nr(), cell.nao_nr()))

        mf = mock.Mock()
        mf.cell = cell
        mf.kpts = kpts
        mf.with_df = df_class(cell, kpts)
        if direct:
            mf.with_df.direct = True
        mf.rsjk = rsjk
        mf.exxdiv = "ewald"
        mf.make_rdm1.return_value = dm_kpts
        call_shape = {}

        def get_jk(**kwargs):
            call_shape.update(kwargs)
            return np.zeros_like(dm_kpts), np.zeros_like(dm_kpts)

        mf.get_jk.side_effect = get_jk
        exxss = ExxSS.__new__(ExxSS)
        exxss.kmf = mf
        exxss.Ek_uncorr = None
        exxss.compute_uncorrected_exx()
        return call_shape

    def test_gdf_and_fftdf_request_exchange_only(self):
        for df_class in (df.GDF, df.FFTDF):
            with self.subTest(backend=df_class.__name__):
                call_shape = self._call_shape(df_class)
                self.assertFalse(call_shape["with_j"])
                self.assertIn("kpts_band", call_shape)

    def test_rsjk_requests_jk_cache_shape(self):
        call_shape = self._call_shape(df.GDF, rsjk=object())
        self.assertTrue(call_shape["with_j"])

        inactive_call_shape = self._call_shape(df.GDF, rsjk=False)
        self.assertFalse(inactive_call_shape["with_j"])

    def test_direct_rsdf_omits_band_kpoints(self):
        call_shape = self._call_shape(df.RSDF, direct=True)
        self.assertFalse(call_shape["with_j"])
        self.assertNotIn("kpts_band", call_shape)


class KnownValues(unittest.TestCase):
    TOL = 1e-6
    REFERENCES = {
        "pbe0_contracted_gaussian": {
            "Ek_ss": -0.5291272996690393,
            "correction": -0.26247787610465045,
            "integral_term": -0.489449527737625,
            "quadrature_term": 0.22697165163297459,
        },
        "hf_contracted_gaussian": {
            "Ek_ss": -0.5291272967126897,
            "correction": -0.26247787314965876,
            "integral_term": -0.4894495084688119,
            "quadrature_term": 0.22697163531915313,
        },
        "pbe0_quartic_exponential": {
            "Ek_ss": -0.5283008162655474,
            "correction": -0.26165139270115867,
            "integral_term": -0.5435525114152486,
            "quadrature_term": 0.28190111871408996,
        },
        "hf_quartic_exponential": {
            "Ek_ss": -0.5283008133413027,
            "correction": -0.2616513897782719,
            "integral_term": -0.5435524885236986,
            "quadrature_term": 0.28190109874542674,
        },
    }

    @classmethod
    def setUpClass(cls):
        cls.pbe0_mf = cls._run_pbe0()
        cls.hf_mf = cls._run_hf()

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
    def _run_pbe0(cls):
        cell, kpts = cls._build_h2_cell()
        mf = pbcdft.KRKS(cell, kpts)
        mf.xc = "PBE0"
        return cls._run_scf(mf, cell, kpts)

    @classmethod
    def _run_hf(cls):
        cell, kpts = cls._build_h2_cell()
        mf = pbcscf.KRHF(cell, kpts)
        return cls._run_scf(mf, cell, kpts)

    @staticmethod
    def _run_scf(mf, cell, kpts):
        mf.exxdiv = "ewald"
        mf.with_df = df.GDF(cell, kpts).build()
        mf.kernel()
        if not mf.converged:
            raise RuntimeError("Reference mean-field calculation did not converge")
        return mf

    def _assert_exxss_reference(self, label, mf, exxss_cls, **kwargs):
        exxss = exxss_cls(
            mf,
            fit_method="scipy_least_squares",
            fit_with_coul=True,
            **kwargs,
        )
        exxss.compute_correction()

        actual = {
            "Ek_ss": exxss.Ek_ss,
            "correction": exxss.correction,
            "integral_term": exxss.integral_term,
            "quadrature_term": exxss.quadrature_term,
        }
        for quantity, reference in self.REFERENCES[label].items():
            self.assertAlmostEqual(
                actual[quantity],
                reference,
                delta=self.TOL,
                msg=f"{label} {quantity}",
            )

    def test_pair_density_eval_grid_option(self):
        exxss = ExxSS(
            self.hf_mf,
            pair_density_eval_grid=" Becke ",
            pair_density_becke_grid_level="1",
        )
        self.assertEqual(exxss.pair_density_eval_grid, "becke")
        self.assertEqual(exxss.pair_density_becke_grid_level, 1)

        default_level = ExxSS(
            self.hf_mf,
            pair_density_becke_grid_level=None,
        )
        self.assertEqual(default_level.pair_density_eval_grid, "becke")
        self.assertEqual(default_level.pair_density_becke_grid_level, 0)

        with self.assertRaisesRegex(
                ValueError, "pair_density_eval_grid"):
            ExxSS(self.hf_mf, pair_density_eval_grid="atom")
        with self.assertRaisesRegex(
                ValueError, "pair_density_becke_grid_level"):
            ExxSS(self.hf_mf, pair_density_becke_grid_level=-1)

    def test_pbe0_exxss_contracted_gaussian(self):
        self._assert_exxss_reference(
            "pbe0_contracted_gaussian",
            self.pbe0_mf,
            ExxSS,
        )

    def test_hf_exxss_contracted_gaussian(self):
        self._assert_exxss_reference(
            "hf_contracted_gaussian",
            self.hf_mf,
            ExxSS,
        )

    def test_pbe0_exxss_quartic_exponential(self):
        self._assert_exxss_reference(
            "pbe0_quartic_exponential",
            self.pbe0_mf,
            ExxSSQuarticExponential,
            qG_norm_cutoff_sigma=2.0,
        )

    def test_hf_exxss_quartic_exponential(self):
        self._assert_exxss_reference(
            "hf_quartic_exponential",
            self.hf_mf,
            ExxSSQuarticExponential,
            qG_norm_cutoff_sigma=2.0,
        )


if __name__ == "__main__":
    unittest.main()
