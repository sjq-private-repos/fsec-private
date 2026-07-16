import unittest

import numpy as np
from pyscf.pbc import df
from pyscf.pbc import gto, scf

from fsec.singularity_subtraction.structure_factor.exx_sf import ExxStructureFactor


class KnownValues(unittest.TestCase):
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
        cell.ke_cutoff = 100.
        cell.precision = 1e-8
        cell.verbose = 0
        cell.build()

        kpts = cell.make_kpts((1, 1, 1), wrap_around=True, with_gamma_point=True)
        kmf = scf.KRHF(cell, kpts)
        kmf.exxdiv = "ewald"
        kmf.with_df = df.GDF(cell, kpts).build()
        kmf.conv_tol = 1e-10
        kmf.kernel()
        if not kmf.converged:
            raise RuntimeError("KRHF did not converge for the H2 test system")

        cls.kmf = kmf
        cls.sq_ke_cutoff = 100.
        cls.N_local = cls.kmf.cell.cutoff_to_mesh(cls.sq_ke_cutoff)

    def test_build_structure_factor_10_closest_qg_points(self):
        exx_sf = ExxStructureFactor(self.kmf, N_local=self.N_local, qG_cutoff=None, min_points=10)
        SqG = exx_sf.build_structure_factor()
        qG = exx_sf.grids.qG_grid_truncated

        self.assertEqual(len(SqG), len(qG))
        self.assertTrue(np.all(np.isfinite(SqG)))
        self.assertGreaterEqual(len(qG), 10)

        qG_norm = np.linalg.norm(qG, axis=1)
        idx10 = np.lexsort((qG[:, 2], qG[:, 1], qG[:, 0], qG_norm))[:10]
        qG_10 = qG[idx10]
        SqG_10 = SqG[idx10]

        self.assertEqual(len(qG_10), 10)
        self.assertEqual(len(SqG_10), 10)
        self.assertTrue(np.all(np.isfinite(SqG_10)))

        self.assertTrue(np.any(np.linalg.norm(qG_10, axis=1) <= 1e-8))

        tenth_radius = np.sort(qG_norm)[9]
        self.assertTrue(np.all(np.linalg.norm(qG_10, axis=1) <= tenth_radius + 1e-8))

        reference_SqG_10 = [
            1.0000000040758936,
            0.32744907530353146,
            0.3274490753036006,
            0.13484040769431632,
            0.13484040769431632,
            0.3274490753036006,
            0.32744907530353146,
            0.1329367270735302,
            0.055346487300111935,
            0.055346487300113774,
        ]
        for actual, reference in zip(SqG_10, reference_SqG_10):
            self.assertAlmostEqual(actual, reference, places=8)

    def test_build_structure_factor_with_line_sampling(self):
        exx_sf = ExxStructureFactor(
            self.kmf,
            N_local=self.N_local,
            qG_cutoff=4.0,
            line_sampling=True,
        )
        SqG = exx_sf.build_structure_factor()
        qG = exx_sf.grids.qG_grid_truncated
        expected_qG = exx_sf.grids.build_qG_line_sampling()

        self.assertTrue(exx_sf.line_sampling)
        self.assertEqual(len(SqG), len(qG))
        self.assertTrue(np.all(np.isfinite(SqG)))
        np.testing.assert_allclose(qG, expected_qG)


if __name__ == "__main__":
    unittest.main()
