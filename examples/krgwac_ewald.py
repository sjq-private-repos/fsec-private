"""Compare q -> 0 finite-size corrections in restricted periodic G0W0-AC.

For each Monkhorst-Pack mesh, print the diamond Gamma-point HOMO/LUMO
quasiparticle energies and gap (hartree) without the correction, with PySCF's
sphere integration, with FHI-gap Gaussian auxiliary functions, and with the
Madelung constant, each with PySCF's single q direction and with a 26-point
Lebedev average of the k.p dielectric head/wings.
"""

from pyscf.pbc import df, dft, gto

from fsec.standard import KRGWAC_ewald


cell = gto.Cell()
cell.build(
    unit='angstrom',
    a='''
        0.000000     1.783500     1.783500
        1.783500     0.000000     1.783500
        1.783500     1.783500     0.000000
    ''',
    atom='C 1.337625 1.337625 1.337625; C 2.229375 2.229375 2.229375',
    dimension=3,
    verbose=0,
    pseudo='gth-pade',
    basis='gth-szv',
    precision=1e-10,
)
homo, lumo = 3, 4

settings = [('none', None)] + [
    (method, qdirs) for method in ('sphere', 'aux', 'madelung') for qdirs in (None, 26)
]

for nk in (2, 3):
    kpts = cell.make_kpts([nk] * 3)
    gdf = df.GDF(cell, kpts)
    gdf.build()
    mf = dft.KRKS(cell, kpts).density_fit(with_df=gdf)
    mf.xc = 'pbe'
    mf.conv_tol = 1e-10
    mf.kernel()

    print(f'Nk = {nk}^3')
    print(f'{"method":>10s} {"qdirs":>6s} {"c_head":>10s} {"c_wing":>10s} '
          f'{"HOMO":>12s} {"LUMO":>12s} {"gap":>12s}')
    for method, qdirs in settings:
        gw = KRGWAC_ewald(mf)
        gw.fc = method != 'none'
        if gw.fc:
            gw.fc_method = method
            gw.fc_qdirs = qdirs
            c_head, c_wing = gw.get_fc_coeffs()
        else:
            c_head = c_wing = 0.0
        gw.kernel(orbs=[homo, lumo], kptlist=[0])
        e_homo, e_lumo = gw.mo_energy[0][homo], gw.mo_energy[0][lumo]
        print(f'{method:>10s} {str(qdirs):>6s} {c_head:10.6f} {c_wing:10.6f} '
              f'{e_homo:12.8f} {e_lumo:12.8f} {e_lumo - e_homo:12.8f}')
    print()
