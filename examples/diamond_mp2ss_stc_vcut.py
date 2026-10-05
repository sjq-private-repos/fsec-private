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

"""Diamond sTC MP2 followed by a combined q2 + q4 direct MP2SS correction.

Uses the cell, basis, SCF settings, and 2x2x2 mesh of diamond_mp2ss.py.
Both KMP2_STC and XNGaussVcut use the Wigner-Seitz cutoff. Exchange is
included in the sTC MP2 energy, but only the direct term is corrected.
XNGaussVcut retains its implemented untruncated reference Coulomb integral.
The correction quadrature uses the complete sTC kernel at eta=4.0, including
q+G=0, while the auxiliary model and its fitting procedure stay unchanged.

After activating fsec-312, run from the repository root:
    LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    python -u examples/diamond_mp2ss_stc_vcut.py \
        > examples/diamond_mp2ss_stc_vcut.log 2>&1
"""

import sys
import time

from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import df, gto, scf

from fsec.singularity_subtraction import MP2SS
from fsec.singularity_subtraction.model_function.mp2_direct_modfunc import (
    XNGaussVcut,
)
from fsec.singularity_subtraction.mp2ss import DIRECT_MODEL_SPECS, Q2_MODEL_SPECS
from fsec.vcut.kmp2_stc import KMP2_STC


def main():
    start = time.perf_counter()
    print("Started:", time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    print("Python:", sys.executable)
    print("PySCF threads:", lib.num_threads())
    print("MP2SS: direct only; combined q2 + q4; XNGaussVcut (vcut_ws)")
    print("XNGaussVcut uses its inherited untruncated reference integral.")
    print("Correction quadrature: complete sTC vcut_ws kernel, eta=4.0.")

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

    kmf = scf.KRHF(cell, kpts)
    kmf.verbose = logger.DEBUG
    kmf.exxdiv = "ewald"
    kmf.with_df = df.GDF(cell, kpts).build()
    kmf.kernel()
    if not kmf.converged:
        raise RuntimeError("KRHF did not converge")

    kmp = KMP2_STC(kmf, eta=4.0, exxdiv="vcut_ws", rc_type="ws")
    kmp.verbose = logger.DEBUG
    e_corr, t2 = kmp.kernel(with_t2=True, with_direct=True)

    print("KRHF total energy (hartree/cell)        =", kmf.e_tot)
    print("sTC MP2 direct energy (hartree/cell)    =", kmp.e_corr_direct)
    print("sTC MP2 exchange energy (hartree/cell)  =", kmp.e_corr_exchange)
    print("sTC MP2 correlation energy (hartree/cell) =", e_corr)
    print("sTC MP2 total energy (hartree/cell)     =", kmf.e_tot + e_corr)

    # Supply the PySCF spin-component interface consumed by MP2SS:
    # E_direct = 2 E_OS and E_exchange = E_SS - E_OS.
    kmp.e_corr_os = 0.5 * kmp.e_corr_direct
    kmp.e_corr_ss = kmp.e_corr_os + kmp.e_corr_exchange
    kmp.with_df_ints = True

    # Register the new model for this example's process. The complete,
    # Gamma-centered q mesh also defines XNGaussVcut's cutoff supercell.
    # Its default vcut_ws therefore matches the KMP2_STC builder above.
    DIRECT_MODEL_SPECS["XNGaussVcut"] = {
        "cls_name": XNGaussVcut.__name__,
        "type": "qmesh",
        "c0_scale": 1.0e-4,
        "fit_multipliers": [1e4, 1.0],
    }
    # MP2SS resolves both direct model names even for the full-direct route.
    # If separate fits are later enabled, q2 excludes deltaG=0 and q4 handles it.
    Q2_MODEL_SPECS["XNGaussVcut"] = {
        **DIRECT_MODEL_SPECS["XNGaussVcut"],
        "c0_scale": 1.0e-5,
    }
    mp2ss = MP2SS(
        kmf=kmf,
        kmp=kmp,
        t2=t2,
        auxfunc_direct="XNGaussVcut",
        auxfunc_direct_q2="XNGaussVcut",
        correct_q2_q4_separately=False,
        quadrature_coulomb_kernel="vcut_ws",
        eta=4.0,
    )
    correction = mp2ss.compute_correction(direct=True, exchange=False)

    print("q2 + q4 integral term (hartree/cell)   =", mp2ss.direct_integral_term)
    print("q2 + q4 quadrature term (hartree/cell) =", mp2ss.direct_quadrature_term)
    print("q2 + q4 direct correction (hartree/cell) =", float(correction))
    print("MP2SS corrected direct energy (hartree/cell) =", mp2ss.emp2ss_direct)
    print("sTC exchange energy (hartree/cell)          =", mp2ss.emp2ss_exchange)
    print("MP2SS corrected correlation (hartree/cell)  =", mp2ss.emp2ss)
    print("MP2SS corrected total energy (hartree/cell) =", kmf.e_tot + mp2ss.emp2ss)
    print("Completed in %.1f seconds" % (time.perf_counter() - start))


if __name__ == "__main__":
    main()
