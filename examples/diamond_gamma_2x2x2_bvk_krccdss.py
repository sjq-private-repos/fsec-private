#!/usr/bin/env python
"""Gamma-point KRCCDSS for a 2x2x2 BvK diamond supercell."""

import numpy as np

from pyscf.pbc import gto, scf
from pyscf.pbc.tools import pbc as pbctools

from fsec.singularity_subtraction import KRCCD_SS


BVK_MESH = np.asarray([2, 2, 2])
N_PRIMITIVE_CELLS = int(np.prod(BVK_MESH))

# Gamma-containing primitive-cell 2x2x2 reference from
# diamond_2x2x2_krccdss_constraint_1_relaxed_no_exxss.log.
REFERENCE_SCF_PER_CELL = -9.469387390682694
REFERENCE_CCDSS_CORR_PER_CELL = -0.09914847536862217
REFERENCE_CCDSS_TOTAL_PER_CELL = -9.568535866051317


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
cell.pseudo = "gth-hf"
cell.precision = 1e-8
cell.ke_cutoff = 100
cell.verbose = 4
cell.build()

# A Gamma calculation in this BvK supercell folds the Gamma-containing
# primitive-cell 2x2x2 k-point mesh into a single supercell k-point.
supercell = pbctools.super_cell(cell, BVK_MESH, wrap_around=True)
kpts = supercell.make_kpts(
    [1, 1, 1], wrap_around=True, with_gamma_point=True
)

print("Diamond Gamma-point KRCCDSS in a 2x2x2 BvK supercell")
print(f"BvK mesh: {BVK_MESH}")
print(f"number of primitive cells: {N_PRIMITIVE_CELLS}")
print(f"number of atoms: {supercell.natm}")
print(f"number of electrons: {supercell.nelectron}")
print(f"supercell volume: {supercell.vol:.15f}")
print(f"supercell mesh: {supercell.mesh}")
print(f"number of k-points: {len(kpts)}")
print(f"k-points: {kpts}")
print("constraint (1): False")
print("constraint (2): True")
print("ExxSS applied: False")

kmf = scf.KRHF(supercell, kpts).density_fit()
kmf.exxdiv = None
kmf.conv_tol = 1e-9
kmf.chkfile = "diamond_gamma_2x2x2_bvk_scf.chk"
e_scf = kmf.kernel()
if not kmf.converged:
    raise RuntimeError("The diamond BvK-supercell KRHF calculation did not converge.")

e_scf_per_cell = e_scf / N_PRIMITIVE_CELLS
print(f"SCF energy, supercell = {e_scf:.15f}")
print(f"SCF energy / primitive cell = {e_scf_per_cell:.15f}")
print(
    "SCF delta from primitive 2x2x2 Gamma-containing mesh / cell = "
    f"{e_scf_per_cell - REFERENCE_SCF_PER_CELL:+.15e}"
)

cc = KRCCD_SS(kmf, use_constraint_1=False, use_constraint_2=True)
cc.keep_exxdiv = False
cc.conv_tol = 1e-7
cc.max_cycle = 30
e_corr, _, _ = cc.kernel()
if not cc.converged:
    raise RuntimeError("The diamond BvK-supercell KRCCDSS calculation did not converge.")

e_corr_per_cell = e_corr / N_PRIMITIVE_CELLS
e_total = e_scf + e_corr
e_total_per_cell = e_total / N_PRIMITIVE_CELLS
print(f"SCF energy = {e_scf:.15f}")
print(f"Madelung xi = {cc.madelung_constant:.15f}")
print(f"CCDSS E_corr, supercell = {e_corr:.15f}")
print(f"CCDSS E_corr / primitive cell = {e_corr_per_cell:.15f}")
print(f"CCDSS total energy, supercell = {e_total:.15f}")
print(f"CCDSS total energy / primitive cell = {e_total_per_cell:.15f}")
print(
    "CCDSS E_corr delta from primitive 2x2x2 Gamma-containing mesh / cell = "
    f"{e_corr_per_cell - REFERENCE_CCDSS_CORR_PER_CELL:+.15e}"
)
print(
    "CCDSS total delta from primitive 2x2x2 Gamma-containing mesh / cell = "
    f"{e_total_per_cell - REFERENCE_CCDSS_TOTAL_PER_CELL:+.15e}"
)
print(f"ss_prepare_count = {cc.ss_prepare_count}")
print(f"ss_fit_count = {cc.ss_fit_count}")
print(f"||R_SS|| = {cc.last_ss_residual_norm:.8e}")
print(
    "CCDSS six sigmas = "
    f"{np.array2string(cc.ss_sigmas, precision=8)}"
)
print(
    "CCDSS six xi = "
    f"{np.array2string(cc.ss_xi, precision=8)}"
)
