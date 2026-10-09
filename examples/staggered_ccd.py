"""Run two- and three-mesh restricted CCD for periodic hydrogen."""

from pyscf.pbc import df, gto, scf

from fsec.staggered_mesh.cc import KRCCD_2mesh, KRCCD_3mesh


def make_cell():
    cell = gto.Cell()
    cell.atom = "H 3 3 2.1; H 3 3 3.9"
    cell.a = "6 0 0; 0 6 0; 0 0 6"
    cell.unit = "B"
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pade"
    cell.mesh = [17, 17, 17]
    cell.verbose = 4
    cell.build()
    return cell


def run_ccd(cell, mesh, solver, **solver_options):
    kpts = cell.make_kpts(mesh)
    mf = scf.KRHF(cell, kpts=kpts, exxdiv="ewald")
    mf.with_df = df.GDF(cell, kpts)
    mf.conv_tol = 1e-9
    e_hf = mf.kernel()

    cc = solver(mf, **solver_options)
    e_corr, t1, t2 = cc.kernel()
    print(f"{solver.__name__} on {mesh}: E(HF)={e_hf:.12f}  "
          f"E(CCD)={e_corr:.12f}  max|t1|={abs(t1).max():.1e}")
    return e_corr, t1, t2


def main():
    cell = make_cell()

    # Use even-sized reference meshes to form occupied and virtual submeshes.
    run_ccd(cell, [2, 2, 2], KRCCD_2mesh, flag_submesh=True)

    # Three submeshes require reference mesh sizes divisible by three.
    run_ccd(cell, [3, 1, 1], KRCCD_3mesh, flag_submesh=True)

    # For band extensions, set flag_submesh=False.  KRCCD_2mesh also accepts
    # nks_ccd=[nx, ny, nz] to choose its mesh independently of the reference.


if __name__ == "__main__":
    main()
