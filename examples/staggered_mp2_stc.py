"""Run staggered MP2 with FFTDF and with GDF+sTC band orbitals.

With ``flag_submesh=False``, orbitals and orbital energies on the shifted mesh
come from a fixed-density band calculation. ``band_df='stc'`` evaluates it with
density-fitted smoothed truncated Coulomb exchange instead of a temporary FFTDF.
The correlation integrals use the bare Coulomb interaction in both cases. The
sTC paths need the custom ``pyscf.pbc.df.rsdf_stc`` module.
"""

from pyscf.pbc import df, gto, scf
from pyscf.pbc.df import rsdf_stc

from fsec.staggered_mesh.mp import KMP2_stagger


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

# The sTC path reuses the reference auxiliary basis, and the band energies are
# sensitive to it. Converge it for production calculations.
auxbasis = "weigend"
kpts = cell.make_kpts([2, 2, 2], with_gamma_point=True)

kmf = scf.KRHF(cell, kpts, exxdiv="ewald")
kmf.with_df = df.GDF(cell, kpts)
kmf.with_df.auxbasis = auxbasis
kmf.conv_tol = 1e-12
kmf.conv_tol_grad = 1e-9
kmf.kernel()

# An sTC reference is detected automatically and fixes eta and the cutoff.
kmf_stc = rsdf_stc.density_fit(scf.KRHF(cell, kpts), auxbasis=auxbasis, eta=4.0)
kmf_stc.conv_tol = 1e-12
kmf_stc.conv_tol_grad = 1e-9
kmf_stc.kernel()

methods = (
    ("GDF reference, FFTDF+vcut_sph bands", KMP2_stagger(kmf)),
    ("GDF reference, GDF+sTC bands", KMP2_stagger(kmf, band_df="stc")),
    ("sTC reference, GDF+sTC bands", KMP2_stagger(kmf_stc)),
)
for name, method in methods:
    print(f"{name}: E2={method.kernel():.12f} Ha")
