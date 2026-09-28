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
    HAS_MP2_IMPORT = True
except ImportError:
    HAS_MP2_IMPORT = False


@unittest.skipUnless(HAS_PYSCF and HAS_MP2_IMPORT, "PySCF and fsec structure_factor deps are required")
class KnownValues(unittest.TestCase):
    @classmethod
    def _build_system(cls, kmesh, backend="gdf"):
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
        if backend == "gdf":
            kmf = scf.KRHF(cell, kpts, exxdiv="ewald")
            kmf.with_df = df.GDF(cell, kpts).build()
        elif backend == "fft_ws":
            kmf = scf.KRHF(cell, kpts, exxdiv="vcut_ws")
            kmf.with_df = df.FFTDF(cell, kpts)
            kmf.with_df.mesh = list(cell.mesh)
        else:
            raise ValueError("unknown test backend: {}".format(backend))
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
        cls.kmf_fft_113, cls.kmp_fft_113, cls.t2_fft_113 = cls._build_system(
            (1, 1, 3), backend="fft_ws"
        )
        cls.sq_ke_cutoff = 100.0
        cls.qG_cutoff = 8.0
        cls.N_local = cls.kmf.cell.cutoff_to_mesh(cls.sq_ke_cutoff)

    @staticmethod
    def _closest_10_indices(qG):
        qG_norm = np.linalg.norm(qG, axis=1)
        return np.lexsort((qG[:, 2], qG[:, 1], qG[:, 0], qG_norm))[:10]

    def _check_structure_factor_10_closest_qg_points(self, kmf, kmp, t2, references):
        """Check the three structure-factor terms against backend references."""
        mp2_sf = MP2StructureFactor(
            kmf,
            kmp,
            t2=t2,
            N_local=self.N_local,
            qG_cutoff=self.qG_cutoff,
            min_points=10,
        )
        result = mp2_sf.build_structure_factor(direct=True, exchange=True, dG0=True)

        qG = result["qG_full"]
        idx10 = self._closest_10_indices(qG)
        qG_10 = qG[idx10]
        self.assertEqual(len(qG_10), 10)
        self.assertTrue(np.any(np.linalg.norm(qG_10, axis=1) <= 1e-8))

        for key, reference in references.items():
            with self.subTest(term=key):
                actual = result[key]
                self.assertEqual(len(actual), len(qG))
                self.assertTrue(np.all(np.isfinite(actual)))
                self.assertTrue(np.isrealobj(actual))
                np.testing.assert_allclose(
                    actual[idx10], reference, rtol=0, atol=5e-9,
                    err_msg=key,
                )

    def test_build_structure_factor_10_closest_qg_points(self):
        """Preserve the Gamma-point GDF structure-factor reference values."""
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

        self._check_structure_factor_10_closest_qg_points(
            self.kmf, self.kmp, self.t2,
            {
                "SqG_full_direct": reference_SqG_direct_10,
                "SqG_full_q4": reference_SqG_q4_10,
                "SqG_full_exchange": reference_SqG_exchange_10,
            },
        )

    def test_build_structure_factor_10_closest_qg_points_fftdf(self):
        """Check Gamma-point FFTDF direct, exchange, and dG0 reference values."""
        kmf, kmp, t2 = self._build_system((1, 1, 1), backend="fft_ws")
        # PySCF 2.14.0, WS-truncated HF, full FFTDF KMP2 amplitudes,
        # and a 100-Ha cutoff (29^3 mesh); sorted by _closest_10_indices.
        references = {
            "SqG_full_direct": [
                -1.172058899966748e-21,
                -7.906080127100997e-17,
                -7.906079309717075e-17,
                -0.00028254532096928315,
                -0.00028254532096928315,
                -7.906079309717075e-17,
                -7.906080127100997e-17,
                -1.21281663965706e-16,
                -0.0001226727777199658,
                -0.00012267277771995896,
            ],
            "SqG_full_q4": [
                -8.303279465169691e-41,
                -3.778097873770974e-31,
                -3.778097092560502e-31,
                -4.825330009799675e-06,
                -4.825330009799675e-06,
                -3.778097092560502e-31,
                -3.778097873770974e-31,
                -8.890804790723498e-31,
                -9.09593147291765e-07,
                -9.095931472916641e-07,
            ],
            "SqG_full_exchange": [
                5.86029449983374e-22,
                3.9530400635504984e-17,
                3.9530396548585376e-17,
                0.00014127266048464157,
                0.00014127266048464157,
                3.9530396548585376e-17,
                3.9530400635504984e-17,
                6.0640831982853e-17,
                6.13363888599829e-05,
                6.133638885997948e-05,
            ],
        }
        self._check_structure_factor_10_closest_qg_points(
            kmf, kmp, t2, references,
        )

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

    def test_fft_kikj_matches_full_kikjka(self):
        """Match direct and exchange against full FFTDF amplitudes off Gamma."""
        from fsec.singularity_subtraction.mp2ss import convert_t2_to_kikjq_format

        common = dict(
            N_local=self.N_local,
            qG_cutoff=1.2,
            min_points=10,
            sq_inversion_symm=False,
        )
        reference_sf = MP2StructureFactor(
            self.kmf_fft_113,
            self.kmp_fft_113,
            t2_store_type="kikjka",
            **common,
        )
        reference_sf.set_grids(min_fit_points=common["min_points"])
        # The full structure-factor path consumes q-indexed amplitudes;
        # convert a copy of PySCF's ka-indexed tensor to preserve the fixture.
        t2 = self.t2_fft_113.copy()
        convert_t2_to_kikjq_format(
            t2, self.kmp_fft_113.kpts, reference_sf.grids.qGrid,
            self.kmf_fft_113.cell,
        )
        reference = reference_sf.build_structure_factor(
            direct=True, exchange=True, t2=t2, grids=reference_sf.grids,
        )
        actual = MP2StructureFactor(
            self.kmf_fft_113,
            self.kmp_fft_113,
            t2_store_type="kikj",
            **common,
        ).build_structure_factor(direct=True, exchange=True)

        np.testing.assert_allclose(
            actual["qG_full"], reference["qG_full"], atol=1e-12,
        )
        for key in ("SqG_full_direct", "SqG_full_exchange"):
            np.testing.assert_allclose(
                actual[key], reference[key], rtol=1e-7, atol=1e-10,
                err_msg=key,
            )

    def test_fft_kikj_exchange_matches_ki_and_legacy_bug_is_distinct(self):
        common = dict(
            N_local=self.N_local,
            qG_cutoff=4.0,
            min_points=10,
            sq_inversion_symm=False,
        )
        reference = MP2StructureFactor(
            self.kmf_fft_113,
            self.kmp_fft_113,
            t2_store_type="ki",
            **common,
        ).build_structure_factor(exchange=True)

        fixed = MP2StructureFactor(
            self.kmf_fft_113,
            self.kmp_fft_113,
            t2_store_type="kikj",
            **common,
        ).build_structure_factor(exchange=True)
        np.testing.assert_allclose(
            fixed["qG_full"], reference["qG_full"], atol=1e-12
        )
        np.testing.assert_allclose(
            fixed["SqG_full_exchange"],
            reference["SqG_full_exchange"],
            rtol=1e-7,
            atol=1e-10,
        )

        legacy = MP2StructureFactor(
            self.kmf_fft_113,
            self.kmp_fft_113,
            t2_store_type="kikj",
            legacy_fft_exchange_orbital_order=True,
            **common,
        ).build_structure_factor(exchange=True)
        self.assertGreater(
            np.max(
                np.abs(
                    legacy["SqG_full_exchange"]
                    - fixed["SqG_full_exchange"]
                )
            ),
            1.0e-12,
        )

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
