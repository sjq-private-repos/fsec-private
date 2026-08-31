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

"""Iterative CCD singularity subtraction for periodic H2."""

import numpy as np

from pyscf.pbc import gto, scf

from fsec.singularity_subtraction import KRCCD_SS


cell = gto.Cell()
cell.unit = "Bohr"
cell.atom = "H 0.0 0.0 0.0; H 1.8 0.0 0.0"
cell.a = np.eye(3) * 6.0
cell.basis = "gth-szv"
cell.pseudo = "gth-pade"
cell.precision = 1e-8
cell.build()

kpts = cell.make_kpts([1, 1, 2], wrap_around=True, with_gamma_point=True)
kmf = scf.KRHF(cell, kpts).density_fit()
kmf.exxdiv = None
kmf.conv_tol = 1e-9
kmf.kernel()
if not kmf.converged:
    raise RuntimeError("The KRHF calculation did not converge.")


def run_ccdss(label, fixed_sigma=None):
    """Run one fitted or fixed-width CCDSS calculation."""
    cc = KRCCD_SS(kmf, fixed_sigma=fixed_sigma)
    cc.keep_exxdiv = False
    cc.conv_tol = 1e-7
    cc.max_cycle = 30
    e_corr, t1, t2 = cc.kernel()
    if not cc.converged:
        raise RuntimeError(f"The {label} CCDSS calculation did not converge.")
    print(f"{label:24s} E_corr = {e_corr:.15f}")
    print(
        f"{'':24s} prepares = {cc.ss_prepare_count}, "
        f"fits = {cc.ss_fit_count}, ||R_SS|| = {cc.last_ss_residual_norm:.8e}"
    )
    if cc.ss_sigmas is not None:
        print(
            f"{'':24s} six sigmas = "
            f"{np.array2string(cc.ss_sigmas, precision=8)}"
        )
    return e_corr, t1, t2


# Default: rebuild the six complete amplitude-dependent structure factors and
# fit six Gaussian widths on every CCD update.  The latest sigma and xi states
# each contain six elements.
run_ccdss("fitted CCDSS")

# Fixed widths are amplitude-independent and are prepared once during
# initialization (or lazily if initialization is bypassed).  Zero gives the
# orbital-only correction; infinity gives the exact
# orbital-plus-ERI Madelung correction.
run_ccdss("sigma = 0", fixed_sigma=0.0)
run_ccdss("sigma = infinity", fixed_sigma=np.inf)
