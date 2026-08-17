#!/usr/bin/env python
# Copyright 2026 Stephen Quiton. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# These tests were written independently by Stephen Quiton.  They credit Xin
# Xing's experimental KRCCD code and local commit e37974c, Xin Xing and Lin
# Lin for the method in Phys. Rev. X 14, 011059, and the PySCF developers,
# including J. D. McClain and T. Berkelbach, for the upstream CCSD code.

"""Tests for the minimal periodic restricted CCD implementation."""

import os
from types import SimpleNamespace
import unittest
from unittest import TestCase, mock

import numpy as np

from pyscf.pbc import gto, scf
from pyscf.pbc.cc import kccsd_rhf
from pyscf.lib.parameters import LARGE_DENOM

from fsec.staggered_mesh.cc import KCCD, KRCCD
from fsec.staggered_mesh.cc import krccd as krccd_module


try:
    import pytest
except ImportError:  # pragma: no cover - unittest remains the test runner
    pytest = None


def slow_test(test):
    """Mark the full periodic reference test when pytest is available."""
    if pytest is not None:
        return pytest.mark.slow(test)
    return test


def archived_reference_test(test):
    """Keep the historical FFTDF comparison opt-in on slow local machines."""
    test = slow_test(test)
    return unittest.skipUnless(
        os.environ.get("FSEC_RUN_ARCHIVED_REFERENCE"),
        "set FSEC_RUN_ARCHIVED_REFERENCE=1 to run the historical FFTDF reference",
    )(test)


def make_uninitialized_solver(nkpts=1, nocc=1, nmo=2):
    """Create a solver shell for tests of algebra independent of AO2MO."""
    solver = object.__new__(KRCCD)
    solver.kpts = np.zeros((nkpts, 3))
    solver.mo_occ = np.asarray(
        [np.r_[np.full(nocc, 2.0), np.zeros(nmo - nocc)] for _ in range(nkpts)]
    )
    solver._nocc = None
    solver._nmo = None
    solver.frozen = None
    solver.khelper = SimpleNamespace(
        kconserv=np.zeros((nkpts, nkpts, nkpts), dtype=int)
    )
    solver.level_shift = 0.0
    solver.madelung_constant = 0.0
    solver.madelung_eri = False
    solver.last_madelung_eri_residual_norm = 0.0
    return solver


def build_h2_2x2x2():
    """Build the archived small periodic H2 reference calculation.

    The archived ``CC_stagger/kccd_rhf.py`` uses a 0.4-bohr bond centered in
    the cell.  The user-facing example intentionally uses the paper's 1.8-
    bohr geometry instead.
    """
    cell = gto.Cell()
    cell.unit = "Bohr"
    cell.atom = "H 3.0 3.0 2.8; H 3.0 3.0 3.2"
    cell.a = np.eye(3) * 6.0
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pade"
    cell.build()
    kpts = cell.make_kpts([2, 2, 2])
    mf = scf.KRHF(cell, kpts)
    mf.exxdiv = None
    mf.kernel()
    return mf


class InterfaceTests(TestCase):
    def test_alias_and_public_exports(self):
        self.assertIs(KCCD, KRCCD)

        import fsec

        self.assertFalse(hasattr(fsec, "KRCCD"))

    def test_madelung_sign_and_explicit_orbital_baseline(self):
        cell = gto.Cell()
        cell.unit = "Bohr"
        cell.atom = "H 0.0 0.0 0.0; H 1.8 0.0 0.0"
        cell.a = np.eye(3) * 6.0
        cell.basis = "gth-szv"
        cell.pseudo = "gth-pade"
        cell.build()
        kpts = cell.make_kpts([1, 1, 1], with_gamma_point=True)
        mf = scf.KRHF(cell, kpts)
        mf.exxdiv = None
        mf.mo_coeff = np.asarray([np.eye(2)])
        mf.mo_occ = np.asarray([[2.0, 0.0]])
        mf.mo_energy = np.asarray([[1.0, 3.0]])

        original = {
            "mo_coeff": mf.mo_coeff.copy(),
            "mo_occ": mf.mo_occ.copy(),
            "mo_energy": mf.mo_energy.copy(),
            "exxdiv": mf.exxdiv,
        }

        def make_eris(_mo_coeff=None):
            return SimpleNamespace(
                fock=np.asarray([[[1.0, 0.2], [0.2, 3.0]]]),
                mo_energy=[np.zeros(2)],
            )

        with mock.patch.object(krccd_module.tools, "madelung", return_value=2.5):
            cc = KRCCD(mf, mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ)
            self.assertEqual(cc.madelung_constant, -2.5)
            with mock.patch.object(
                kccsd_rhf.RCCSD, "ao2mo", side_effect=make_eris
            ):
                baseline = cc.ao2mo()
                corrected = KRCCD(
                    mf,
                    mo_coeff=mf.mo_coeff,
                    mo_occ=mf.mo_occ,
                    madelung_orbital=True,
                ).ao2mo()

        np.testing.assert_allclose(baseline.mo_energy, [[1.0, 3.0]])
        np.testing.assert_allclose(corrected.mo_energy, [[-1.5, 3.0]])
        np.testing.assert_allclose(
            baseline.fock, [[[1.0, 0.2], [0.2, 3.0]]]
        )
        np.testing.assert_allclose(
            corrected.fock, [[[-1.5, 0.2], [0.2, 3.0]]]
        )
        np.testing.assert_allclose(mf.mo_coeff, original["mo_coeff"])
        np.testing.assert_allclose(mf.mo_occ, original["mo_occ"])
        np.testing.assert_allclose(mf.mo_energy, original["mo_energy"])
        self.assertEqual(mf.exxdiv, original["exxdiv"])

    def test_unsupported_inputs_have_clear_errors(self):
        cell = gto.Cell()
        cell.unit = "Bohr"
        cell.atom = "H 0.0 0.0 0.0; H 1.0 0.0 0.0"
        cell.a = np.eye(3) * 5.0
        cell.dimension = 2
        cell.basis = "gth-szv"
        cell.pseudo = "gth-pade"
        cell.build()
        mf = scf.KRHF(cell, np.zeros((1, 3)))
        mf.mo_coeff = np.asarray([np.eye(2)])
        mf.mo_occ = np.asarray([[2.0, 0.0]])

        with self.assertRaisesRegex(ValueError, "three-dimensional"):
            KRCCD(mf, mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ)

        cell.dimension = 3
        mf.mo_occ = np.asarray([[1.0, 1.0]])
        with self.assertRaisesRegex(ValueError, "closed-shell"):
            KRCCD(mf, mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ)


class AlgebraTests(TestCase):
    def test_eri_correction_uses_padding_level_shift_and_denominators(self):
        solver = make_uninitialized_solver(nkpts=2, nocc=2, nmo=4)
        solver.madelung_constant = 0.7
        solver.level_shift = 0.5
        eris = SimpleNamespace(
            mo_energy=[
                np.asarray([1.0, 2.0, 10.0, 11.0]),
                np.asarray([3.0, 4.0, 12.0, 13.0]),
            ]
        )
        t2 = np.ones((2, 2, 2, 2, 2, 2, 2))
        occupied = [np.asarray([0]), np.asarray([0, 1])]
        virtual = [np.asarray([0, 1]), np.asarray([1])]

        with mock.patch.object(
            krccd_module,
            "padding_k_idx",
            return_value=(occupied, virtual),
        ):
            correction = np.zeros_like(t2)
            returned = solver._add_madelung_eri_correction(
                correction, t2, eris
            )

        self.assertIs(returned, correction)
        residual_norm = np.linalg.norm(2.0 * 0.7 * t2)
        self.assertAlmostEqual(
            solver.last_madelung_eri_residual_norm, residual_norm
        )
        self.assertAlmostEqual(
            correction[0, 0, 0, 0, 0, 0, 0], 1.4 / -19.0
        )
        self.assertAlmostEqual(
            correction[0, 0, 0, 1, 0, 0, 0],
            1.4 / (LARGE_DENOM - 9.5),
        )

    def test_update_amps_zeros_singles_before_and_after_upstream_call(self):
        solver = make_uninitialized_solver()
        solver.madelung_eri = False
        t1 = np.full((1, 1, 1), 4.0)
        t2 = np.ones((1, 1, 1, 1, 1, 1, 1))
        returned_t1 = np.full_like(t1, 9.0)
        returned_t2 = np.full_like(t2, 3.0)
        eris = SimpleNamespace()

        received_t1 = []

        def fake_upstream(_cc, upstream_t1, upstream_t2, upstream_eris):
            received_t1.append(upstream_t1.copy())
            return returned_t1, returned_t2.copy()

        with mock.patch.object(kccsd_rhf.RCCSD, "update_amps", fake_upstream):
            out_t1, out_t2 = solver.update_amps(t1, t2, eris)

        np.testing.assert_array_equal(received_t1[0], np.zeros_like(t1))
        np.testing.assert_array_equal(out_t1, np.zeros_like(t1))
        np.testing.assert_array_equal(out_t2, returned_t2)

    def test_zero_amplitude_eri_correction_is_inert(self):
        solver = make_uninitialized_solver()
        solver.madelung_constant = -0.4
        eris = SimpleNamespace(mo_energy=[np.asarray([1.0, 4.0])])
        zero_t2 = np.zeros((1, 1, 1, 1, 1, 1, 1))
        target = np.zeros_like(zero_t2)
        solver._add_madelung_eri_correction(target, zero_t2, eris)
        np.testing.assert_array_equal(target, zero_t2)
        self.assertEqual(solver.last_madelung_eri_residual_norm, 0.0)


class FixedPointTests(TestCase):
    def test_kernel_n_performs_exactly_n_raw_updates_from_zero(self):
        solver = make_uninitialized_solver()
        solver.dump_flags = lambda verbose=None: solver
        solver.get_e_hf = lambda: 0.0
        updates = []

        def update(_solver, t1, t2, eris):
            updates.append((t1.copy(), t2.copy()))
            return np.ones_like(t1), t2 + 1.0

        def energy(_solver, t1, t2, eris):
            np.testing.assert_array_equal(t1, np.zeros_like(t1))
            return float(np.sum(t2))

        solver.update_amps = update.__get__(solver, KRCCD)
        solver.energy = energy.__get__(solver, KRCCD)
        eris = SimpleNamespace(dtype=np.dtype(float), fock=np.zeros((1, 1, 1)))

        result, t1, t2 = KRCCD.kernel_n(solver, 3, eris=eris)

        self.assertEqual(len(updates), 3)
        np.testing.assert_array_equal(updates[0][0], np.zeros_like(t1))
        np.testing.assert_array_equal(updates[0][1], np.zeros_like(t2))
        np.testing.assert_array_equal(t1, np.zeros_like(t1))
        np.testing.assert_array_equal(t2, np.full_like(t2, 3.0))
        self.assertEqual(result, 3.0)
        self.assertEqual(solver.cycles, 3)

        with self.assertRaisesRegex(ValueError, "n >= 1"):
            KRCCD.kernel_n(solver, 0, eris=eris)


@archived_reference_test
def test_archived_h2_ccd2_and_converged_cancellation():
    """Reproduce the archived 2x2x2 H2 converged values and cancellation."""
    mf = build_h2_2x2x2()
    settings = {}
    for orbital in (False, True):
        for eri in (False, True):
            cc = KRCCD(
                mf,
                madelung_orbital=orbital,
                madelung_eri=eri,
            )
            settings[(orbital, eri)] = cc.kernel_n(2)[0]

    e_ccd1 = KRCCD(mf).kernel_n(1)[0]
    assert not np.isclose(e_ccd1, settings[(False, False)])
    assert not np.isclose(settings[(False, False)], settings[(False, True)])

    cc_none = KRCCD(mf)
    e_none = cc_none.kernel()[0]
    cc_eri = KRCCD(mf, madelung_eri=True)
    e_eri = cc_eri.kernel()[0]
    np.testing.assert_allclose(e_none, -0.006273269059592043, atol=1e-5)
    np.testing.assert_allclose(e_eri, -0.007563169456805278, atol=1e-5)

    cc_both = KRCCD(mf, madelung_orbital=True, madelung_eri=True)
    e_both = cc_both.kernel()[0]
    assert cc_none.converged
    assert cc_eri.converged
    assert cc_both.converged
    np.testing.assert_allclose(e_none, e_both, atol=1e-7)
