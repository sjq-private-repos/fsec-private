"""Run restricted periodic MP3 with ordinary and staggered k-point meshes.

The staggered contractions are based on Xin Xing's implementation and
Hirata et al., J. Chem. Phys. 120, 2581 (2004).
"""

from pyscf.pbc import df, gto, scf

from fsec.staggered_mesh.mp import KMP3, KMP3_2mesh, KMP3_3mesh


cell = gto.Cell()
cell.atom = """
    H 3.00  3.00  2.10
    H 3.00  3.00  3.90
"""
cell.a = """
    6.0  0.0  0.0
    0.0  6.0  0.0
    0.0  0.0  6.0
"""
cell.unit = "B"
cell.basis = "gth-szv"
cell.pseudo = "gth-pade"
cell.ke_cutoff = 100
cell.precision = 1e-9
cell.verbose = 4
cell.build()

# A 1x1x6 mesh is divisible by both two and three for the submesh modes.
kpts = cell.make_kpts([1, 1, 6], with_gamma_point=True)
kmf = scf.KRHF(cell, kpts, exxdiv="ewald")
kmf.with_df = df.GDF(cell, kpts).build()
kmf.conv_tol = 1e-12
kmf.conv_tol_grad = 1e-9
kmf.kernel()

# Set this to True to partition the SCF mesh into commensurate submeshes.
# False evaluates band orbitals on an extended mesh.
flag_submesh = False

methods = (
    KMP3(kmf),
    KMP3_2mesh(kmf, flag_submesh=flag_submesh),
    KMP3_3mesh(kmf, flag_submesh=flag_submesh),
)
for method in methods:
    energy, t2 = method.kernel()
    print(
        f"{type(method).__name__}: E2={method.e_corr_mp2:.12f} Ha, "
        f"E3={method.e_corr_mp3:.12f} Ha, "
        f"E2+E3={energy:.12f} Ha, first-order t2 shape={t2.shape}"
    )
