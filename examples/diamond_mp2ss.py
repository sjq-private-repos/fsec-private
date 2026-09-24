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

"""
MP2 singularity subtraction for diamond.

This example runs KMP2 on a 2x2x2 k-point mesh and applies the direct and
exchange MP2SS corrections to the stored doubles amplitudes.

Method reference:
    S. J. Quiton, J. D. F. Pottecher, M. Head-Gordon, and L. Lin,
    arXiv:2605.12727 (2026).
    https://doi.org/10.48550/arXiv.2605.12727
"""

from pyscf.pbc import df, gto, mp, scf
from pyscf.lib import logger

from fsec.singularity_subtraction import MP2SS


cell = gto.Cell()
cell.unit = "Bohr"
cell.atom = """
    C 0.0 0.0 0.0
    C 1.68516327271508 1.68516327271508 1.68516327271508
"""
cell.a = """
    0.0 3.370326545430162 3.370326545430162
    3.370326545430162 0.0 3.370326545430162
    3.370326545430162 3.370326545430162 0.0
"""
cell.basis = "gth-szv"
cell.pseudo = "gth-pbe"
cell.ke_cutoff = 100
cell.precision = 1e-8
cell.max_memory = 4000
cell.verbose = 4
cell.build()

kpts = cell.make_kpts(
    [2, 2, 2], wrap_around=True, with_gamma_point=True
)

# Generate the orbitals and retain the KMP2 amplitudes needed by MP2SS.
kmf = scf.KRHF(cell, kpts)
kmf.verbose = logger.DEBUG
kmf.exxdiv = "ewald"
kmf.with_df = df.GDF(cell, kpts).build()
kmf.kernel()

kmp = mp.KMP2(kmf)
kmp.verbose = logger.DEBUG
kmp.with_df_ints = True
e_corr, t2 = kmp.kernel(with_t2=True)

print("KMP2 correlation energy (hartree) =", e_corr)
print("KMP2 total energy (hartree)       =", kmp.e_tot)
print("KMP2 same-spin energy (hartree)   =", kmp.e_corr_ss)
print("KMP2 opposite-spin energy (hartree) =", kmp.e_corr_os)

# Compute both components of the MP2 finite-size correction.
mp2ss = MP2SS(kmf=kmf, kmp=kmp, t2=t2)
correction = mp2ss.compute_correction(direct=True, exchange=True)

print("MP2SS total correction (hartree)    =", float(correction))
print(
    "MP2SS direct correction (hartree)   =",
    correction.mp2ss_direct_correction,
)
print(
    "MP2SS exchange correction (hartree) =",
    correction.mp2ss_exchange_correction,
)
print("MP2SS corrected direct energy (hartree)   =", mp2ss.emp2ss_direct)
print("MP2SS corrected exchange energy (hartree) =", mp2ss.emp2ss_exchange)
print("MP2SS corrected total energy (hartree)    =", mp2ss.emp2ss)

# Report the quadratic and quartic pieces of the direct correction separately.
print("q2 integral term (hartree)   =", mp2ss.direct_integral_term_q2)
print("q2 quadrature term (hartree) =", mp2ss.direct_quadrature_term_q2)
print("q2 correction (hartree)      =", mp2ss.direct_total_correction_q2)
print("q4 integral term (hartree)   =", mp2ss.direct_integral_term_q4)
print("q4 quadrature term (hartree) =", mp2ss.direct_quadrature_term_q4)
print("q4 correction (hartree)      =", mp2ss.direct_total_correction_q4)
print(
    "q2 + q4 direct correction (hartree) =",
    mp2ss.direct_total_correction_q2_q4,
)
