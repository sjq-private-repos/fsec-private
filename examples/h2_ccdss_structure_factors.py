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

"""Print aggregate fitted CCDSS structure-factor samples for periodic H2."""

import numpy as np

from pyscf.lib import logger
from pyscf.pbc import gto, scf

from fsec.singularity_subtraction import KRCCD_SS


# Match the small H2 CCDSS example: GTH-SZV on a 1x1x2 k-point mesh.
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


# DEBUG2 emits six CCDSS_SF_NORM rows with the raw L1-L6 origins, followed by
# one CCDSS_SF row for every aggregate curve and q sample.  Fitted CCDSS
# prepares these six density-only curves once before the iterations.
# There are no tensor indices in the output.
cc = KRCCD_SS(kmf, line_points=3)
cc.keep_exxdiv = False
cc.verbose = logger.DEBUG2
cc.conv_tol = 1e-7
cc.max_cycle = 30
e_corr, _, _ = cc.kernel()
if not cc.converged:
    raise RuntimeError("The fitted CCDSS calculation did not converge.")

print(f"fitted CCDSS E_corr = {e_corr:.15f}")
print(
    f"prepares = {cc.ss_prepare_count}, fits = {cc.ss_fit_count}, "
    f"||R_SS|| = {cc.last_ss_residual_norm:.8e}"
)
