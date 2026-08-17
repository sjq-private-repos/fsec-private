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
# This implementation was written independently by Stephen Quiton; it does
# not copy the experimental code.  It credits Xin Xing's experimental KRCCD
# code and local commit e37974c, Xin Xing and Lin Lin for the method in
# Phys. Rev. X 14, 011059, and the PySCF developers, including J. D. McClain
# and T. Berkelbach, for the upstream CCSD implementation.

"""Minimal restricted k-point CCD built on PySCF 2.14 KCCSD."""

from __future__ import annotations

import numbers

import numpy as np

from pyscf import scf as molecular_scf
from pyscf.lib import logger
from pyscf.pbc import scf, tools
from pyscf.pbc.cc import kccsd_rhf
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.mp.kmp2 import padding_k_idx


class KRCCD(kccsd_rhf.RCCSD):
    """Restricted periodic CCD with optional Madelung corrections.

    The doubles residual and energy contraction are inherited from PySCF's
    restricted k-point CCSD implementation.  With a zero singles amplitude,
    those expressions are the CCD expressions.  ``madelung_constant`` is the
    signed paper convention ``xi = -pyscf.pbc.tools.madelung(...)``.
    """

    _keys = kccsd_rhf.RCCSD._keys.union(
        {
            "madelung_orbital",
            "madelung_eri",
            "madelung_constant",
            "last_madelung_eri_residual_norm",
        }
    )

    def __init__(
        self,
        mf,
        frozen=None,
        mo_coeff=None,
        mo_occ=None,
        madelung_orbital=False,
        madelung_eri=False,
    ):
        if not isinstance(mf, scf.khf.KSCF):
            raise TypeError(
                "KRCCD requires a restricted periodic k-point mean-field "
                "object (KRHF or a compatible KSCF)."
            )
        if not isinstance(mf, scf.khf.KRHF):
            raise TypeError(
                "KRCCD supports restricted k-point references only; "
                "unrestricted or generalized references are unsupported."
            )
        if isinstance(mf, molecular_scf.hf.KohnShamDFT):
            raise TypeError(
                "KRCCD requires a Hartree-Fock reference, not a k-point DFT "
                "object.  Convert the mean-field object to HF first."
            )
        if getattr(mf.cell, "dimension", None) != 3:
            raise ValueError(
                "KRCCD currently supports three-dimensional periodic cells "
                "only."
            )
        if getattr(mf.cell, "spin", 0) != 0:
            raise ValueError(
                "KRCCD requires a closed-shell restricted reference "
                "(cell.spin must be zero)."
            )

        # Let PySCF initialize missing orbitals in the same way as its KCCSD
        # class, then validate the effective occupations used by CCD.
        super().__init__(mf, frozen=frozen, mo_coeff=mo_coeff, mo_occ=mo_occ)
        self._validate_restricted_occupations(self.mo_occ)

        self.madelung_orbital = bool(madelung_orbital)
        self.madelung_eri = bool(madelung_eri)
        self.madelung_constant = -float(tools.madelung(mf.cell, self.kpts))
        self.last_madelung_eri_residual_norm = 0.0

    def _validate_restricted_occupations(self, mo_occ):
        if mo_occ is None:
            raise ValueError(
                "KRCCD requires converged restricted occupations; run the "
                "mean-field calculation or provide mo_occ explicitly."
            )
        try:
            occupations = list(mo_occ)
        except TypeError as exc:
            raise ValueError(
                "KRCCD expects one restricted occupation array per k-point."
            ) from exc
        if len(occupations) != self.nkpts:
            raise ValueError(
                "KRCCD received an occupation array count inconsistent with "
                "the k-point mesh."
            )
        for kpt, occ in enumerate(occupations):
            occ = np.asarray(occ)
            if occ.ndim != 1:
                raise ValueError(
                    "KRCCD requires one-dimensional restricted occupations "
                    f"at k-point {kpt}."
                )
            if not np.all(np.isclose(occ, 0.0) | np.isclose(occ, 2.0)):
                raise ValueError(
                    "KRCCD requires closed-shell occupations containing only "
                    "0 or 2; fractional or open-shell occupations are "
                    f"unsupported at k-point {kpt}."
                )
        if not any(np.count_nonzero(np.asarray(occ)) for occ in occupations):
            raise ValueError("KRCCD requires at least one occupied orbital.")

    def dump_flags(self, verbose=None):
        """Log the CCD settings, including the signed Madelung constant."""
        result = super().dump_flags(verbose)
        log = logger.new_logger(self, verbose)
        log.info("Madelung orbital correction = %s", self.madelung_orbital)
        log.info("Madelung ERI correction = %s", self.madelung_eri)
        log.info("Signed Madelung xi = %.15g", self.madelung_constant)
        return result

    def ao2mo(self, mo_coeff=None):
        """Build PySCF ERIs and restore an explicitly uncorrected baseline.

        PySCF's k-point CCSD ERI builder applies its own occupied-orbital
        Madelung shift to ``eris.mo_energy``.  Take the Fock matrix diagonal
        before that shift as the explicit baseline.  When the orbital
        correction is enabled, shift the occupied Fock diagonal together with
        the occupied orbital energies so PySCF's residual intermediates see a
        consistent one-body operator.
        """
        if self.keep_exxdiv and self.madelung_orbital:
            logger.warn(
                self,
                "keep_exxdiv=True and madelung_orbital=True may double-count "
                "the occupied-orbital Madelung correction; set "
                "keep_exxdiv=False to use KRCCD's explicit correction",
            )

        eris = super().ao2mo(mo_coeff)
        fock = np.array(eris.fock, copy=True)
        mo_energy = [
            np.asarray(fock_k.diagonal().real).copy() for fock_k in fock
        ]

        if self.madelung_orbital:
            nonzero_opadding, _ = padding_k_idx(self, kind="split")
            for k, occupied in enumerate(nonzero_opadding):
                fock[k][occupied, occupied] += self.madelung_constant
                mo_energy[k][occupied] += self.madelung_constant

        eris.fock = fock
        eris.mo_energy = mo_energy
        return eris

    def _add_madelung_eri_correction(self, t2new, t2, eris):
        """Add the denominator-divided ``2*xi*T2`` residual to ``t2new``.

        Applying the correction one momentum block at a time avoids allocating
        another full doubles-sized denominator or residual tensor.
        """
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        nonzero_opadding, nonzero_vpadding = padding_k_idx(self, kind="split")
        mo_e_o = [e[:nocc] for e in eris.mo_energy]
        mo_e_v = [e[nocc:] + self.level_shift for e in eris.mo_energy]
        residual_factor = 2.0 * self.madelung_constant
        self.last_madelung_eri_residual_norm = float(
            abs(residual_factor) * np.linalg.norm(t2)
        )

        for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
            kb = self.khelper.kconserv[ki, ka, kj]
            eia = kccsd_rhf._get_epq(
                [0, nocc, ki, mo_e_o, nonzero_opadding],
                [0, nvir, ka, mo_e_v, nonzero_vpadding],
                fac=[1.0, -1.0],
            )
            ejb = kccsd_rhf._get_epq(
                [0, nocc, kj, mo_e_o, nonzero_opadding],
                [0, nvir, kb, mo_e_v, nonzero_vpadding],
                fac=[1.0, -1.0],
            )
            # This is the same broadcasting expression used by the PySCF
            # kccsd_rhf update_amps implementation.
            eijab = eia[:, None, :, None] + ejb[:, None, :]
            t2new[ki, kj, ka] += residual_factor * t2[ki, kj, ka] / eijab
        return t2new

    def update_amps(self, t1, t2, eris):
        """Perform an upstream doubles update with singles fixed to zero."""
        zero_t1 = np.zeros_like(t1)
        _, t2new = super().update_amps(zero_t1, t2, eris)
        if self.madelung_eri:
            self._add_madelung_eri_correction(t2new, t2, eris)
        else:
            self.last_madelung_eri_residual_norm = 0.0
        return zero_t1, t2new

    def energy(self, t1, t2, eris):
        """Use PySCF's energy contraction with the CCD singles set to zero."""
        return super().energy(np.zeros_like(t1), t2, eris)

    def kernel(self, t1=None, t2=None, eris=None, mbpt2=False):
        """Delegate ordinary converged CCD, forcing any supplied T1 to zero."""
        if t1 is not None:
            t1 = np.zeros_like(t1)
        return super().kernel(t1=t1, t2=t2, eris=eris, mbpt2=mbpt2)

    def kernel_n(self, n, eris=None):
        """Run exactly ``n`` undamped, non-DIIS CCD fixed-point updates.

        The amplitudes start at zero, as in the CCD(n) construction of Phys.
        Rev. X 14, 011059.  This method intentionally does not use the
        inherited converged ``kernel`` machinery.
        """
        if isinstance(n, bool) or not isinstance(n, numbers.Integral) or n < 1:
            raise ValueError("kernel_n requires an integer n >= 1.")

        self.dump_flags()
        self.e_hf = self.get_e_hf()
        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        self.eris = eris

        nocc = self.nocc
        nvir = self.nmo - nocc
        shape_t1 = (self.nkpts, nocc, nvir)
        shape_t2 = (self.nkpts, self.nkpts, self.nkpts, nocc, nocc, nvir, nvir)
        dtype = getattr(eris, "dtype", None)
        if dtype is None:
            dtype = eris.fock.dtype
        t1 = np.zeros(shape_t1, dtype=dtype)
        t2 = np.zeros(shape_t2, dtype=dtype)

        for _ in range(n):
            t1, t2 = self.update_amps(t1, t2, eris)
            # Keep the invariant explicit even if a future upstream routine
            # returns a nonzero singles array.
            t1 = np.zeros_like(t1)

        self.converged = False
        self.cycles = int(n)
        self.t1 = t1
        self.t2 = t2
        self.e_corr = self.energy(t1, t2, eris)
        return self.e_corr, self.t1, self.t2


KCCD = KRCCD

__all__ = ["KRCCD", "KCCD"]
