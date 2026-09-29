import numpy as np
import scipy

from pyscf.lib import logger
from pyscf.pbc.tools import get_monkhorst_pack_size
from pyscf.pbc.dft import gen_grid as pbc_gen_grid

from fsec.singularity_subtraction.grids import ExxSSGrids, minimum_image

from fsec.singularity_subtraction.structure_factor import StructureFactor
from fsec.singularity_subtraction.structure_factor.helpers import build_uKpts, TimingProfile


def normalize_pair_density_eval_grid(pair_density_eval_grid):
    pair_density_eval_grid = str(pair_density_eval_grid).strip().lower()
    if pair_density_eval_grid not in ("uniform", "becke"):
        raise ValueError(
            "pair_density_eval_grid must be 'uniform' or 'becke'"
        )
    return pair_density_eval_grid


def normalize_pair_density_becke_grid_level(
        pair_density_becke_grid_level):
    pair_density_becke_grid_level = (
        0 if pair_density_becke_grid_level is None
        else int(pair_density_becke_grid_level)
    )
    if pair_density_becke_grid_level < 0:
        raise ValueError(
            "pair_density_becke_grid_level must be non-negative"
        )
    return pair_density_becke_grid_level


class ExxStructureFactor(StructureFactor):
    def __init__(self, kmf, N_local=None, sq_ke_cutoff=None, qG_cutoff=None, relative_shift=0.0, **kwargs):
        """
        Initialize the exchange structure factor.
        """
        self.kmf = kmf
        self.dm_kpts1 = kwargs.get("dm_kpts1", kmf.make_rdm1())
        self.dm_kpts2 = kwargs.get("dm_kpts2", kmf.make_rdm1())
        self.mo_coeff_kpts1 = kwargs.get("mo_coeff_kpts1", kmf.mo_coeff_kpts)
        self.mo_coeff_kpts2 = kwargs.get("mo_coeff_kpts2", kmf.mo_coeff_kpts)
        self.kGrid1 = minimum_image(kmf.cell, kwargs.get("kGrid1", kmf.kpts))
        self.kGrid2 = kwargs.get("kGrid2", None)
        self.relative_shift = relative_shift
        self.min_points = kwargs.get("min_points", 6)
        self.line_sampling = kwargs.get("line_sampling", False)
        self.pair_density_eval_grid = normalize_pair_density_eval_grid(
            kwargs.get("pair_density_eval_grid", "becke")
        )
        self.pair_density_becke_grid_level = (
            normalize_pair_density_becke_grid_level(
                kwargs.get("pair_density_becke_grid_level", 0)
            )
        )

        self.debug_options = kwargs.get("debug_options", {})
        super().__init__(self.kmf.cell, N_local, sq_ke_cutoff, qG_cutoff, **kwargs)

    def set_grids(self, min_fit_points=6):
        self.min_points = min_fit_points
        self.grids = ExxSSGrids(
            self.kmf.cell,
            self.kGrid1,
            N_local=self.N_local,
            qG_norm_cutoff=self.qG_cutoff,
            min_points=self.min_points,
        )
        self.grids.build_grids()
        if self.qG_cutoff is None:
            self.grids.build_truncated_qG_grid()
        if self.line_sampling:
            qG_line = self.grids.build_qG_line_sampling()
            if len(qG_line) < self.min_points:
                raise ValueError(
                    f"Line sampling produced {len(qG_line)} q+G points; "
                    f"increase qG_cutoff to provide at least {self.min_points}."
                )
            self.grids.qG_grid_truncated = qG_line

    def build_structure_factor(
            self, verbose=None, pair_density_eval_grid=None,
            pair_density_becke_grid_level=None):
        """Build the exchange structure factor.

        Parameters
        ----------
        pair_density_eval_grid : {"uniform", "becke"}, optional
            Real-space quadrature grid used to evaluate pair-density
            overlaps. The default is the value supplied at construction,
            which defaults to ``"becke"``.
        pair_density_becke_grid_level : int, optional
            PySCF periodic Becke grid level. The construction-time default is
            0.
        """
        kmf = self.kmf
        log = logger.new_logger(kmf, verbose)
        profile = TimingProfile()
        total_t0 = profile.start()

        if pair_density_eval_grid is None:
            pair_density_eval_grid = self.pair_density_eval_grid
        pair_density_eval_grid = normalize_pair_density_eval_grid(
            pair_density_eval_grid
        )
        if pair_density_becke_grid_level is None:
            pair_density_becke_grid_level = (
                self.pair_density_becke_grid_level
            )
        pair_density_becke_grid_level = (
            normalize_pair_density_becke_grid_level(
                pair_density_becke_grid_level
            )
        )

        phase_t0 = profile.start()
        self.set_grids(min_fit_points=self.min_points)
        profile.stop("grid construction", phase_t0)
        qG_full = self.grids.qG_grid_truncated
        kGrid1 = self.grids.kGrid1
        kGrid2 = self.grids.kGrid2
        rptGrid3D = self.grids.RptGrid3D_coarse
        quadrature_weights = None
        if pair_density_eval_grid == "becke":
            pair_grid = pbc_gen_grid.BeckeGrids(kmf.cell)
            pair_grid.level = pair_density_becke_grid_level
            pair_grid.build(with_non0tab=False)
            rptGrid3D = np.asarray(pair_grid.coords)
            quadrature_weights = np.asarray(pair_grid.weights)

        nocc = kmf.cell.tot_electrons() // 2
        phase_t0 = profile.start()
        uKpts1 = build_uKpts(kmf, kGrid1, self.mo_coeff_kpts1, rptGrid3D=rptGrid3D, nbands=nocc)
        uKpts2 = build_uKpts(kmf, kGrid2, self.mo_coeff_kpts2, rptGrid3D=rptGrid3D, nbands=nocc)
        profile.stop("uKpts construction", phase_t0)

        phase_t0 = profile.start()
        NsCell = np.array(self.N_local)
        nks = get_monkhorst_pack_size(kmf.cell, kmf.kpts)
        nkpts = np.prod(nks)

        Lvec_real = kmf.cell.lattice_vectors()
        L_delta = Lvec_real / NsCell[:, None]
        dvol = np.abs(np.linalg.det(L_delta))
        quadrature_product_scale = (
            dvol**2 if pair_density_eval_grid == "uniform" else 1.0
        )

        nqG = qG_full.shape[0]
        nG = rptGrid3D.shape[0]
        print("ExxStructureFactor nG: ", nG)
        print("ExxStructureFactor nqG: ", nqG)
        SqG_full = np.zeros(nqG, dtype=np.float64)
        print("SqG MEM USAGE (KB) IS: {:.3f}".format(SqG_full.nbytes / 1024))
        profile.stop("structure-factor setup", phase_t0)

        num_equiv_qG = 0
        kgrid2_tree = scipy.spatial.KDTree(kGrid2)

        loop_t0 = profile.start()
        for qG in range(qG_full.shape[0]):
            qGpt = qG_full[qG, :]

            region_t0 = profile.start()
            if self.sq_inversion_symm and qG > 1:
                qg_tree = scipy.spatial.KDTree(qG_full[:qG, :])
                _, equiv_qG_index = qg_tree.query(-qGpt, distance_upper_bound=1e-8)
                if equiv_qG_index != len(qG_full[:qG, :]):
                    num_equiv_qG += 1
                    SqG_full[qG] = SqG_full[equiv_qG_index]
                    profile.stop("inversion-symmetry reuse", region_t0)
                    continue
            profile.stop("inversion-symmetry lookup", region_t0)

            region_t0 = profile.start()
            kpt2s = kGrid1 + qGpt
            kpt2s_BZ = minimum_image(kmf.cell, kpt2s)
            _, idx_kpt2s = kgrid2_tree.query(kpt2s_BZ, distance_upper_bound=np.sqrt(1e-8))
            if np.any(idx_kpt2s == kgrid2_tree.n):
                raise TypeError("Cannot locate (k+q) in the kmesh.")
            kGdiffs = kpt2s - kpt2s_BZ
            exp_terms = np.exp(-1j * (rptGrid3D @ kGdiffs.T)).T
            profile.stop("per-qG index/phase setup", region_t0)

            region_t0 = profile.start()
            u2 = uKpts2[idx_kpt2s] * exp_terms[:, None, :]
            if quadrature_weights is not None:
                u2 *= quadrature_weights[None, None, :]
            profile.stop("pair-density orbital setup", region_t0)

            region_t0 = profile.start()
            rho12 = np.conj(uKpts1) @ u2.transpose(0, 2, 1)
            profile.stop("pair-density matrix multiply", region_t0)

            region_t0 = profile.start()
            rho12 = np.abs(rho12) ** 2
            SqG_full[qG] = (
                np.sum(rho12) * quadrature_product_scale / nkpts
            )
            profile.stop("pair-density square/reduction", region_t0)

        profile.stop("main qG loop (inclusive)", loop_t0)
        log.note("Number of equivalent qG points: %d", num_equiv_qG)

        phase_t0 = profile.start()
        self.SqG_full = SqG_full
        self.SqG_truncated = SqG_full
        profile.stop("class result update", phase_t0)
        print("ExxStructureFactor: nocc = ", self.SqG_full[0])

        self.last_build_timings = profile.summary(total_t0)
        profile.log_summary(log, self.last_build_timings)
        return SqG_full
