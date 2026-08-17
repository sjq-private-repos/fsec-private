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
# This example was written independently by Stephen Quiton.  It credits Xin
# Xing's experimental KRCCD code and local commit e37974c, Xin Xing and Lin
# Lin for the method in Phys. Rev. X 14, 011059, and the PySCF developers,
# including J. D. McClain and T. Berkelbach, for the upstream CCSD code.

"""CCD(n) Madelung-correction example for periodic H2."""

import numpy as np

from pyscf.pbc import df, gto, scf, tools

from fsec.staggered_mesh.cc import KRCCD


cell = gto.Cell()
cell.unit = "Bohr"
cell.atom = "H 0.0 0.0 0.0; H 1.8 0.0 0.0"
cell.a = np.eye(3) * 6.0
cell.basis = "gth-szv"
cell.pseudo = "gth-pade"
cell.precision = 1e-8
cell.build()

kpts = cell.make_kpts([2, 2, 2], wrap_around=True, with_gamma_point=True)
kmf = scf.KRHF(cell, kpts)
kmf.exxdiv = None
kmf.with_df = df.GDF(cell, kpts).build()
kmf.kernel()

madelung_magnitude = tools.madelung(cell, kpts)
print("PySCF Madelung magnitude =", madelung_magnitude)
print("Signed xi                =", KRCCD(kmf).madelung_constant)

for madelung_orbital in (False, True):
    for madelung_eri in (False, True):
        cc = KRCCD(
            kmf,
            madelung_orbital=madelung_orbital,
            madelung_eri=madelung_eri,
        )
        e_corr, _, _ = cc.kernel_n(2)
        print(
            "CCD(2) orbital=%s eri=%s E_corr=%.15f "
            "||2*xi*T2||=%.8e"
            % (
                madelung_orbital,
                madelung_eri,
                e_corr,
                cc.last_madelung_eri_residual_norm,
            )
        )

cc_none = KRCCD(kmf)
e_none, _, _ = cc_none.kernel()
cc_both = KRCCD(kmf, madelung_orbital=True, madelung_eri=True)
e_both, _, _ = cc_both.kernel()
if not (cc_none.converged and cc_both.converged):
    raise RuntimeError("The converged CCD example did not converge.")

print("Converged none E_corr    =", e_none)
print("Converged both E_corr    =", e_both)
print("Converged |difference|    =", abs(e_none - e_both))
print("Agree within 1e-7         =", np.isclose(e_none, e_both, atol=1e-7))
