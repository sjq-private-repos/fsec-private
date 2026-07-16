from abc import ABC, abstractmethod
from fsec.singularity_subtraction.grids import minimum_image, MP2SSGrids
from pyscf.pbc.tools import get_monkhorst_pack_size
import scipy
from pyscf.pbc.tools.pbc import mesh_to_cutoff, cutoff_to_mesh
from pyscf.lib import logger
from pyscf import lib
from pyscf.lib.parameters import LARGE_DENOM
from pyscf.pbc import df
from pyscf.pbc.dft import gen_grid as pbc_gen_grid
from pyscf.pbc.lib import kpts_helper
from pyscf.lib import logger, einsum
from pyscf.pbc.mp import kmp2
import numpy as np
from scipy.spatial import KDTree
from pyscf.lib.numpy_helper import einsum as pyscf_einsum




from fsec.singularity_subtraction.structure_factor import StructureFactor
from fsec.singularity_subtraction.structure_factor.helpers_sf import (
    TimingProfile,
    build_uKpts,
    filter_line_sampling_segments,
    make_line_sampling_decay_state,
    normalize_line_sampling_decay_components,
    should_compute_line_sample,
    update_line_sampling_decay_mask,
)


def compute_t2_amplitudes(kmp, mo_energy, mo_coeff, qGrid=None, qGrid_sample=None,
                          skip_if_no_qpt=False, mode='direct', Lov=None, verbose=logger.DEBUG,
                          return_eijab_recip=False, with_t2=True):
    """Computes k-point RMP2 energy.

    Args:
        mp (KMP2): an instance of KMP2
        mo_energy (list): a list of numpy.ndarray. Each array contains MO energies of
                        shape (Nmo,) for one kpt. If frozen orbitals or per-k ragged
                        shapes are present, they are canonicalized via
                        kmp2._add_padding so the body of this function can safely
                        use kmp.nocc / kmp.nmo and kmp2.padding_k_idx.
        mo_coeff (list): a list of numpy.ndarray. Each array contains MO coefficients
                        of shape (Nao, Nmo) for one kpt. Padded analogously.
        verbose (int, optional): level of verbosity. Defaults to logger.NOTE (=3).
        with_t2 (bool, optional): whether to compute t2 amplitudes. Defaults to WITH_T2 (=True).
        mode (str, optional): 'direct' means ka = ki + q. 'exchange' means ka = kj - q. Default is 'direct'
        return_eijab_recip (bool, optional): whether to also return reciprocal
                        MP2 denominators in the same layout as t2.

    Returns:
        t2 amplitudes by default. If return_eijab_recip is True, returns
        (t2, eijab_recip); t2 is None when with_t2 is False.
    """
    if mode not in ['direct', 'exchange']:
        raise ValueError(f"Mode {mode} not recognized. Must be 'direct' or 'exchange'.")
    if not with_t2:
        return_eijab_recip = True
    profile = TimingProfile()
    cput0 = profile.start()
    log = logger.new_logger(kmp, verbose)

    region_t0 = profile.start()
    kmp.dump_flags()
    cell = kmp._scf.cell
    mo_coeff, mo_energy = kmp2._add_padding(kmp, mo_coeff, mo_energy)
    nmo = kmp.nmo
    nocc = kmp.nocc
    nvir = nmo - nocc
    nkpts = kmp.nkpts
    qGrid = kmp.kpts if qGrid is None else qGrid
    nka = qGrid_sample.shape[0] if qGrid_sample is not None else len(qGrid)
    profile.stop("compute_t2 setup/padding", region_t0)

    region_t0 = profile.start()
    with_df_ints = kmp.with_df_ints and isinstance(kmp._scf.with_df, df.GDF)

    mem_avail = kmp.max_memory - lib.current_memory()[0]
    mem_usage = 0
    if with_t2:
        mem_usage += (nkpts * (nocc * nvir)**2) * 16 / 1e6
    if with_t2 and with_df_ints:
        mydf = kmp._scf.with_df
        if mydf.auxcell is None:
            # Calculate naux based on precomputed GDF integrals
            naux = mydf.get_naoaux()
        else:
            naux = mydf.auxcell.nao_nr()

        mem_usage += (nkpts**2 * naux * nocc * nvir) * 16 / 1e6
    if with_t2:
        mem_usage += (nkpts**2 * nka * (nocc * nvir)**2) * 16 / 1e6
    if return_eijab_recip:
        mem_usage += (nkpts**2 * nka * (nocc * nvir)**2) * 8 / 1e6
    if mem_usage > mem_avail:
        raise MemoryError('Insufficient memory! MP2 memory usage %d MB (currently available %d MB)'
                          % (mem_usage, mem_avail))
    profile.stop("compute_t2 memory estimate", region_t0)

    region_t0 = profile.start()
    fao2mo = kmp._scf.with_df.ao2mo if with_t2 else None
    kconserv = kmp.khelper.kconserv
    profile.stop("compute_t2 workspace allocation", region_t0)

    region_t0 = profile.start()
    mo_e_o = [mo_energy[k][:nocc] for k in range(nkpts)]
    mo_e_v = [mo_energy[k][nocc:] for k in range(nkpts)]

    # Get location of non-zero/padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = kmp2.padding_k_idx(kmp, kind="split")
    profile.stop("compute_t2 energy/padding setup", region_t0)

    if qGrid_sample is not None:
        nka = qGrid_sample.shape[0]
        skip_if_no_qpt = True

    region_t0 = profile.start()
    t2 = None
    if with_t2:
        t2 = np.zeros((nkpts, nkpts, nka, nocc, nocc, nvir, nvir), dtype=complex)
    eijab_recip = None
    if return_eijab_recip:
        eijab_recip = np.zeros((nkpts, nkpts, nka, nocc, nocc, nvir, nvir),
                               dtype=mo_energy[0].dtype)
    profile.stop("compute_t2 output allocation", region_t0)

    # Build 3-index DF tensor Lov
    if with_t2 and with_df_ints and Lov is None:
        region_t0 = profile.start()
        Lov = kmp2._init_mp_df_eris(kmp)
        profile.stop("compute_t2 DF integral init", region_t0)

    region_t0 = profile.start()
    q_tree = scipy.spatial.KDTree(qGrid)
    num_skipped_qpts_oovv = 0
    num_skipped_qpts_t2 = 0

    qpts = kmp.kpts[:,None,:] - kmp.kpts[None,:,:]
    qpts = minimum_image(cell, qpts.reshape(-1,3))
    _, qi_map = q_tree.query(qpts, distance_upper_bound=1e-8)
    qi_map = qi_map.reshape(nkpts,nkpts)
    active_qi_map = qi_map
    active_qgrid_size = len(qGrid)
    profile.stop("compute_t2 q-map setup", region_t0)

    # Find qpt
    if qGrid_sample is not None:
        region_t0 = profile.start()
        qGrid_sample = minimum_image(cell, qGrid_sample.reshape(-1,3))
        qsample_tree = scipy.spatial.KDTree(qGrid_sample)
        _, qi_map_sample = qsample_tree.query(qpts, distance_upper_bound=1e-8)
        qi_map_sample = qi_map_sample.reshape(nkpts,nkpts)
        active_qi_map = qi_map_sample
        active_qgrid_size = len(qGrid_sample)
        profile.stop("compute_t2 sampled q-map setup", region_t0)

    loop_t0 = profile.start()
    for ki in range(nkpts):
        for kj in range(nkpts):
            # kref = ki if mode == 'direct' else kj
            for ka in range(nkpts):
                if active_qi_map[ka,ki] == active_qgrid_size:
                    if with_t2:
                        num_skipped_qpts_oovv += 1
                        if not skip_if_no_qpt:
                            raise ValueError(f"Cannot locate qpt for (k+q) in the qmesh.")
                    num_skipped_qpts_t2 += 1
                    if not skip_if_no_qpt:
                        raise ValueError(f"Cannot locate qpt for (k+q) in the qmesh.")
                    continue

                kb = kconserv[ki,ka,kj]
                kvirt = ka if mode == 'direct' else kb
                kvirt2 = kb if mode == 'direct' else ka
                qi = active_qi_map[ka,ki]

                oovv_block = None
                if with_t2:
                    # (ia|jb)
                    if with_df_ints:
                        region_t0 = profile.start()
                        oovv_block = (1./nkpts) * einsum(
                            "Lia,Ljb->iajb",
                            Lov[ki, kvirt],
                            Lov[kj, kvirt2],
                        ).transpose(0,2,1,3)
                        profile.stop("compute_t2 DF oovv block", region_t0)
                    else:
                        region_t0 = profile.start()
                        orbo_i = mo_coeff[ki][:,:nocc]
                        orbo_j = mo_coeff[kj][:,:nocc]
                        orbv_a = mo_coeff[ka][:,nocc:]
                        orbv_b = mo_coeff[kb][:,nocc:]
                        oovv_block = fao2mo(
                            (orbo_i,orbv_a,orbo_j,orbv_b),
                            (kmp.kpts[ki],kmp.kpts[kvirt],kmp.kpts[kj],kmp.kpts[kvirt2]),
                            compact=False
                        ).reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3) / nkpts
                        profile.stop("compute_t2 AO2MO oovv block", region_t0)

                # Remove zero/padded elements from denominator
                region_t0 = profile.start()
                eia = LARGE_DENOM * np.ones((nocc, nvir), dtype=mo_energy[0].dtype)
                n0_ovp_ia = np.ix_(nonzero_opadding[ki], nonzero_vpadding[kvirt])
                eia[n0_ovp_ia] = (mo_e_o[ki][:,None] - mo_e_v[kvirt])[n0_ovp_ia]

                ejb = LARGE_DENOM * np.ones((nocc, nvir), dtype=mo_energy[0].dtype)
                n0_ovp_jb = np.ix_(nonzero_opadding[kj], nonzero_vpadding[kvirt2])
                ejb[n0_ovp_jb] = (mo_e_o[kj][:,None] - mo_e_v[kvirt2])[n0_ovp_jb]

                eijab = lib.direct_sum('ia,jb->ijab',eia,ejb)
                eijab_recip_ijab = 1 / eijab
                profile.stop("compute_t2 denominator reciprocal", region_t0)
                if return_eijab_recip:
                    region_t0 = profile.start()
                    eijab_recip[ki, kj, qi] = eijab_recip_ijab
                    profile.stop("compute_t2 denominator store", region_t0)
                if with_t2:
                    region_t0 = profile.start()
                    out = t2[ki, kj, qi]
                    np.conjugate(oovv_block, out=out)
                    out *= eijab_recip_ijab
                    profile.stop("compute_t2 amplitude store", region_t0)
    profile.stop("compute_t2 main loops", loop_t0)

    log.timer("KMP2", *cput0)
    print(f"Number of skipped qpts for oovv: {num_skipped_qpts_oovv}")
    print(f"Number of skipped qpts for t2: {num_skipped_qpts_t2}")
    timings = profile.summary(cput0)
    total = timings["total"]
    log.debug(
        "compute_t2 CPU %.2f sec, wall %.2f sec",
        total["cpu"], total["wall"],
    )
    for label, values in sorted(
            ((key, value) for key, value in timings.items() if key != "total"),
            key=lambda item: item[1]["wall"], reverse=True):
        cpu_fraction = 100.0 * values["cpu"] / total["cpu"] if total["cpu"] else 0.0
        wall_fraction = 100.0 * values["wall"] / total["wall"] if total["wall"] else 0.0
        log.debug(
            "  %-36s CPU %9.2f sec (%5.1f%%), wall %9.2f sec (%5.1f%%)",
            label, values["cpu"], cpu_fraction,
            values["wall"], wall_fraction,
        )

    if return_eijab_recip:
        return t2, eijab_recip
    return t2


class MP2StructureFactor(StructureFactor):
    def __init__(self, kmf, kmp, t2=None, N_local=None, sq_ke_cutoff=None, qG_cutoff=None, relative_shift=0.0, **kwargs):
        """
        Initialize the structure factor with mean-field object (mf), density matrix (dm), cell, and optional parameters.
        """
        self.kmf = kmf
        self.kmp = kmp
        self.t2 = t2
        self.kGrid1 = minimum_image(kmf.cell, kwargs.get('kGrid1', kmf.kpts))
        self.kGrid2 = kwargs.get('kGrid2', None)
        self.min_points = kwargs.get('min_points', 6)
        self.check_trs = kwargs.get('check_trs', True)
        self.input_sq_ke_cutoff = sq_ke_cutoff
        self.sq_ke_cutoff_switch_radius = kwargs.get('sq_ke_cutoff_switch_radius', None)
        self.outer_sq_ke_cutoff_scale = kwargs.get('outer_sq_ke_cutoff_scale', 0.5)
        self.pair_density_eval_grid = kwargs.get('pair_density_eval_grid', 'becke')
        pair_density_becke_grid_level = kwargs.get('pair_density_becke_grid_level', 0)
        self.pair_density_becke_grid_level = (
            0 if pair_density_becke_grid_level is None
            else int(pair_density_becke_grid_level)
        )

        self.t2_store_type = kwargs.get('t2_store_type', 'kikjka') # 'kikjka' or 'kikj'
        super().__init__(self.kmf.cell, N_local, sq_ke_cutoff, qG_cutoff, **kwargs)
        
    @staticmethod
    def surviving_mo_energy(mmp):
        """Return mo_energy with frozen orbitals removed, per k-point.

        Returns
        -------
        list of np.ndarray
            One array per k-point. Lengths can differ between k-points if you
            used a per-k-point `frozen` list.
        """
        from pyscf.pbc.mp.kmp2 import get_frozen_mask

        moidx = get_frozen_mask(mmp)
        return [e[mask] for e, mask in zip(mmp.mo_energy, moidx)]
    

    def set_grids(self,min_fit_points=6):
        self.min_points = min_fit_points
        self.grids = MP2SSGrids(self.kmf.cell,
                                self.kGrid1,
                                self.N_local,
                                qG_norm_cutoff=self.qG_cutoff,
                                min_points=self.min_points)
        self.grids.build_grids()
        self.grids.build_truncated_qG_grid()

    @staticmethod
    def contract_kikj_dG0(rijab, eijab_recip, scale):
        """Compute the kikj dG0 contraction without materializing sqrt-weighted rijab."""
        return -2 * scale**2 * np.sum(np.abs(rijab)**2 * np.abs(eijab_recip).ravel())

    @staticmethod
    def _build_trs_pair_representatives(trs_map, nkpts=None):
        """Return canonical (m, n, factor) pairs for the TRS rijab involution."""
        trs_map = np.asarray(trs_map, dtype=int)
        if nkpts is None:
            nkpts = len(trs_map)
        representatives = []
        for m in range(nkpts):
            for n in range(nkpts):
                flat = m * nkpts + n
                partner_m = trs_map[n]
                partner_n = trs_map[m]
                partner_flat = partner_m * nkpts + partner_n
                if flat <= partner_flat:
                    factor = 1 if flat == partner_flat else 2
                    representatives.append((m, n, factor))
        return representatives

    @staticmethod
    def _contract_trs_representative_rijab(
            rho_ia_full, rho_jb_full, pair_representatives,
            direct_t2=None, exchange_t2=None, eijab_recip=None, profile=None):
        """Contract rijab/t2 over TRS representative pairs without full materialization."""
        direct_value = 0.0
        exchange_value = 0.0
        q4_weighted_norm = 0.0

        for m, n, factor in pair_representatives:
            if direct_t2 is not None:
                region_t0 = profile.start() if profile is not None else None
                tmp_jb = np.einsum(
                    'ia,ijab->jb',
                    rho_ia_full[m],
                    direct_t2[m, n],
                    optimize=True,
                )
                direct_block = np.einsum(
                    'jb,bj->',
                    tmp_jb,
                    rho_jb_full[n].conj(),
                    optimize=True,
                )
                direct_value += factor * direct_block.real
                if profile is not None:
                    profile.stop("TRS direct rho/t2 contraction", region_t0)
            if exchange_t2 is not None:
                region_t0 = profile.start() if profile is not None else None
                tmp_jb = np.einsum(
                    'ia,ijba->jb',
                    rho_ia_full[m],
                    exchange_t2[m, n],
                    optimize=True,
                )
                exchange_block = np.einsum(
                    'jb,bj->',
                    tmp_jb,
                    rho_jb_full[n].conj(),
                    optimize=True,
                )
                exchange_value += factor * exchange_block.real
                if profile is not None:
                    profile.stop("TRS exchange rho/t2 contraction", region_t0)
            if eijab_recip is not None:
                region_t0 = profile.start() if profile is not None else None
                tmp_jb = np.einsum(
                    'ia,ijab->jb',
                    np.abs(rho_ia_full[m])**2,
                    np.abs(eijab_recip[m, n]),
                    optimize=True,
                )
                q4_weighted_norm += factor * np.einsum(
                    'jb,bj->',
                    tmp_jb,
                    np.abs(rho_jb_full[n])**2,
                    optimize=True,
                )
                if profile is not None:
                    profile.stop("TRS dG0 rho/denominator contraction", region_t0)

        return direct_value, exchange_value, q4_weighted_norm

    @staticmethod
    def _build_coarse_real_space_grid(cell, N_local):
        Lvec_real = cell.lattice_vectors()
        N_local = np.asarray(N_local, dtype=int)
        L_delta = Lvec_real / N_local
        xv, yv, zv = np.meshgrid(
            np.arange(N_local[0]),
            np.arange(N_local[1]),
            np.arange(N_local[2]),
            indexing='ij',
        )
        mesh_idx = np.hstack([xv.reshape(-1, 1), yv.reshape(-1, 1), zv.reshape(-1, 1)])
        return mesh_idx @ L_delta

    def build_structure_factor(self,direct=False,exchange=False,qG_full=None,
                               update_class=True, qG_cutoff=None, dG0=False,
                               grids=None, mo_coeff_kpts1=None, mo_coeff_kpts2=None, mo_coeff_kpts3=None, 
                               kmf=None, t2=None, mo_energy=None, mo_e_o=None, mo_e_v=None, mo_e_v_b=None,
                               t2_store_type=None, Lov=None, Lov_b=None, kmp=None, verbose=None,
                               line_sampling_decay_min_fraction=None,
                               line_sampling_decay_consecutive_below=3,
                               line_sampling_decay_components=(),
                               qG_line_sampling_segments=None,
                               sq_ke_cutoff_switch_radius=None,
                               outer_sq_ke_cutoff_scale=None,
                               pair_density_eval_grid=None,
                               pair_density_becke_grid_level=None):
        """
        Build the MP2 structure factor, either direct term, exchange term, or both.

        Parameters
        ----------
        direct : bool
            Whether to compute the direct term of the structure factor.
        exchange : bool
            Whether to compute the exchange term of the structure factor.
        qG_full : np.ndarray, optional
            The (N, 3) array of q+G points where the structure factor is evaluated.
            If None, uses the grids.qG_grid_local constructed from the class grid setup.
        update_class : bool
            If True, update the class instance attributes with computed structure factor arrays.
        qG_cutoff : float, optional
            Maximum norm of q+G to include in the computed structure factor.
        dG0 : bool, optional
            Whether to compute the DeltaG=0 term of the unfolded structure factor.
        grids : MP2SSGrids, optional
            The grids object to use for the structure factor calculation.
            If None, uses the grids constructed from the class grid setup.
        verbose : int or pyscf.lib.logger.Logger, optional
            PySCF verbosity level or logger used for the timing summary. If
            omitted, the verbosity and output stream are inherited from kmp.
        pair_density_eval_grid : {"uniform", "becke"}, optional
            Real-space quadrature grid used to evaluate pair-density overlaps.
            ``"uniform"`` uses the existing equally weighted mesh. ``"becke"``
            uses PySCF PBC Becke coordinates and weights.
        pair_density_becke_grid_level : int, optional
            PySCF Becke grid level used when ``pair_density_eval_grid="becke"``.
            Defaults to the class setting, which defaults to 0.
        Returns
        -------
        SqG_full_direct : np.ndarray
            The direct term of the structure factor.
        SqG_full_exchange : np.ndarray
            The exchange term of the structure factor.
        qG_full : np.ndarray
            The (N, 3) array of q+G points where the structure factor is evaluated.

        Notes
        -----
        - The structure factor S(qG) is computed for all relevant q+G points.
        - Builds the structure factor for either or both of the direct and exchange MP2 terms.
        - May use pre-provided t2 amplitudes if available or compute them on-the-fly.
        """
        
        if kmf is None:
            kmf = self.kmf
        if kmp is None:
            kmp = self.kmp

        log = logger.new_logger(kmp, verbose)
        profile = TimingProfile()
        total_t0 = profile.start()
        
        if not direct and not exchange and not dG0:
            raise ValueError("Either direct or exchange or dG0 must be requested")
        
        phase_t0 = profile.start()
        mo_coeff_padded, mo_energy_padded = kmp2._add_padding(
            kmp, kmp.mo_coeff, kmp.mo_energy)
        profile.stop("MO padding", phase_t0)

        if grids is None:
            phase_t0 = profile.start()
            self.set_grids(min_fit_points=self.min_points)
            grids = self.grids
            profile.stop("grid construction", phase_t0)

        if qG_full is None:
            qG_full = grids.qG_grid_local
        
        if qG_cutoff is None:
            qG_cutoff = self.qG_cutoff

        if sq_ke_cutoff_switch_radius is None:
            sq_ke_cutoff_switch_radius = self.sq_ke_cutoff_switch_radius
        if outer_sq_ke_cutoff_scale is None:
            outer_sq_ke_cutoff_scale = self.outer_sq_ke_cutoff_scale
        if pair_density_eval_grid is None:
            pair_density_eval_grid = self.pair_density_eval_grid
        pair_density_eval_grid = str(pair_density_eval_grid).strip().lower()
        if pair_density_eval_grid not in ('uniform', 'becke'):
            raise ValueError("pair_density_eval_grid must be 'uniform' or 'becke'")
        if pair_density_becke_grid_level is None:
            pair_density_becke_grid_level = self.pair_density_becke_grid_level
        pair_density_becke_grid_level = int(pair_density_becke_grid_level)
        if pair_density_becke_grid_level < 0:
            raise ValueError("pair_density_becke_grid_level must be non-negative")
        adaptive_sq_ke_cutoff = sq_ke_cutoff_switch_radius is not None
        if adaptive_sq_ke_cutoff and pair_density_eval_grid == 'becke':
            raise NotImplementedError(
                "pair_density_eval_grid='becke' is not implemented with adaptive "
                "sq_ke_cutoff real-space meshes"
            )
        full_sq_ke_cutoff = self.input_sq_ke_cutoff
        if full_sq_ke_cutoff is None and np.isscalar(self.sq_ke_cutoff):
            full_sq_ke_cutoff = self.sq_ke_cutoff
        if adaptive_sq_ke_cutoff:
            if full_sq_ke_cutoff is None or not np.isscalar(full_sq_ke_cutoff):
                raise ValueError("Adaptive sq_ke_cutoff requires a scalar user-provided sq_ke_cutoff")
            if outer_sq_ke_cutoff_scale <= 0:
                raise ValueError("outer_sq_ke_cutoff_scale must be positive")
            sq_ke_cutoff_switch_radius = float(sq_ke_cutoff_switch_radius)
            outer_sq_ke_cutoff_scale = float(outer_sq_ke_cutoff_scale)
            full_sq_ke_cutoff = float(full_sq_ke_cutoff)

        line_sampling_decay_components = normalize_line_sampling_decay_components(
            line_sampling_decay_components, supported_components={"direct_q4", "exchange"})
        
        if mo_coeff_kpts1 is None:
            mo_coeff_kpts1 = mo_coeff_padded
        
        if mo_coeff_kpts2 is None:
            mo_coeff_kpts2 = mo_coeff_padded

        if mo_coeff_kpts3 is None:
            mo_coeff_kpts3 = mo_coeff_padded
        
        if t2_store_type is None:
            t2_store_type = self.t2_store_type
        
        kGrid1 = grids.kGrid1 # occupied
        kGrid2 = grids.kGrid2 # virtual
        kGrid3 = grids.kGrid3 # virtual b

        try:
            trs_map = kpts_helper.conj_mapping(kmf.cell, kGrid1)
            kgrid_occ_trs = True
        except kpts_helper.KPointSymmetryError:
            kgrid_occ_trs = False
            trs_map = None
        check_trs = self.check_trs

        print("kgrid_occ has time reversal symmetry: ", kgrid_occ_trs)

        qGrid = grids.qGrid

        # Set up constants
        # NsCell = np.array(kmf.cell.mesh)
        NsCell = np.array(self.N_local)
        nks = get_monkhorst_pack_size(kmf.cell, kmf.kpts)
        nkpts = np.prod(nks)
        
        t2_required = True # if full t2 in kikjka format is needed
        t2_given = t2 is not None
        
        with_df_ints = self.kmp.with_df_ints and isinstance(self.kmp._scf.with_df, df.GDF)
        if not direct and not exchange:
            # Only dG0 term. No need to compute t2.
            print("Only dG0 term. No need to compute or use t2.")
            t2_required = False
        elif not t2_given:
            if t2_store_type == 'kikjka':
                # NOTE: t2 MUST be in the kikjq format
                if self.t2 is None:
                    phase_t0 = profile.start()
                    t2 = compute_t2_amplitudes(self.kmp, self.kmp.mo_energy, self.kmp.mo_coeff) # nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir
                    profile.stop("full t2 construction", phase_t0)
                else:
                    t2 = self.t2

            else:
                if with_df_ints:
                    phase_t0 = profile.start()
                    Lov = kmp2._init_mp_df_eris(kmp) if Lov is None else Lov
                    if grids.kGrid3_neq_kGrid2 and Lov_b is None:
                        raise NotImplementedError("Lov_b is not implemented for kGrid3_neq_kGrid2")
                    elif Lov_b is None:
                        Lov_b = Lov.copy()
                    profile.stop("DF integral initialization", phase_t0)
                print("Computing t2 on the fly for each q+G point.")
                t2_required = False # we compute t2 on the fly for each q+G point.
                
        else:
            print("Using provided t2, assuming kikjq format")
                
        # periodic parts, uKpts_i, uKpts_j, uKpts_a, uKpts_b
        # nocc = kmf.cell.tot_electrons() // 2
        nocc = self.kmp.nocc
        # nbands = kmf.cell.nao_nr()
        nbands = self.kmp.nmo
        fao2mo = self.kmp._scf.with_df.ao2mo
        nvir = nbands - nocc
        phase_t0 = profile.start()
        
        kgrids_equal = kGrid1 is kGrid2 or np.allclose(kGrid1, kGrid2, atol=1e-8)
        mo_coeffs_equal = mo_coeff_kpts1 is mo_coeff_kpts2 or np.allclose(mo_coeff_kpts1, mo_coeff_kpts2, atol=1e-8)

        if mo_energy is None:
            mo_energy = mo_energy_padded
        mo_energy = np.asarray(mo_energy)
        mo_e_o = mo_energy[:, :nocc] if mo_e_o is None else np.asarray(mo_e_o)
        mo_e_v = mo_energy[:, nocc:] if mo_e_v is None else np.asarray(mo_e_v)
        mo_e_v_b = mo_e_v.copy() if mo_e_v_b is None else np.asarray(mo_e_v_b)
        
        Lvec_real = kmf.cell.lattice_vectors()
        omega_cell = np.abs(np.linalg.det(Lvec_real))
        becke_coords = None
        becke_weights = None
        if pair_density_eval_grid == 'becke':
            pair_grid = pbc_gen_grid.BeckeGrids(kmf.cell)
            pair_grid.level = pair_density_becke_grid_level
            pair_grid.build(with_non0tab=False)
            becke_coords = np.asarray(pair_grid.coords)
            becke_weights = np.asarray(pair_grid.weights)

        def build_mesh_context(label, N_local, rptGrid3D, sq_ke_cutoff_value):
            local_NsCell = np.asarray(N_local, dtype=int)
            quadrature_weights = None
            if pair_density_eval_grid == 'becke':
                rptGrid3D = becke_coords
                quadrature_weights = becke_weights
            if kgrids_equal and mo_coeffs_equal:
                uKpts_i_local = build_uKpts(kmf, kGrid1, mo_coeff_kpts1, rptGrid3D=rptGrid3D, nbands=nbands)
                uKpts_j_local = uKpts_i_local
                uKpts_a_local = uKpts_i_local
                uKpts_b_local = uKpts_i_local
            else:
                uKpts_i_local = build_uKpts(kmf, kGrid1, mo_coeff_kpts1, rptGrid3D=rptGrid3D, nbands=nbands)
                uKpts_j_local = build_uKpts(kmf, kGrid1, mo_coeff_kpts1, rptGrid3D=rptGrid3D, nbands=nbands)
                uKpts_a_local = build_uKpts(kmf, kGrid2, mo_coeff_kpts2, rptGrid3D=rptGrid3D, nbands=nbands)
                uKpts_b_local = build_uKpts(kmf, kGrid3, mo_coeff_kpts3, rptGrid3D=rptGrid3D, nbands=nbands) # SJQ checked

            L_delta = Lvec_real / local_NsCell[:, None]
            quadrature_product_scale = 1.0
            dvol = np.abs(np.linalg.det(L_delta))
            if pair_density_eval_grid == 'uniform':
                quadrature_product_scale = dvol**2
            return {
                'label': label,
                'N_local': local_NsCell.copy(),
                'sq_ke_cutoff': sq_ke_cutoff_value,
                'rptGrid3D': rptGrid3D,
                'dvol': dvol,
                'quadrature_weights': quadrature_weights,
                'quadrature_product_scale': quadrature_product_scale,
                'uKpts_i': uKpts_i_local[:,:nocc,:],
                'uKpts_j': uKpts_j_local[:,:nocc,:],
                'uKpts_a': uKpts_a_local[:,nocc:,:],
                'uKpts_b': uKpts_b_local[:,nocc:,:],
                'conj_uKpts_i': np.conj(uKpts_i_local[:,:nocc,:]),
            }

        full_context = build_mesh_context(
            'inner',
            NsCell,
            grids.RptGrid3D_coarse,
            None if full_sq_ke_cutoff is None else float(full_sq_ke_cutoff),
        )
        mesh_contexts = {'inner': full_context}
        if adaptive_sq_ke_cutoff:
            outer_sq_ke_cutoff = outer_sq_ke_cutoff_scale * full_sq_ke_cutoff
            outer_NsCell = cutoff_to_mesh(kmf.cell.lattice_vectors(), outer_sq_ke_cutoff)
            outer_rptGrid3D = self._build_coarse_real_space_grid(kmf.cell, outer_NsCell)
            mesh_contexts['outer'] = build_mesh_context(
                'outer',
                outer_NsCell,
                outer_rptGrid3D,
                outer_sq_ke_cutoff,
            )
        profile.stop("uKpts construction", phase_t0)

        # SqG = pymp.shared.array((nkpts, nG), dtype=np.float64)
        nqG = qG_full.shape[0]
        nG = full_context['rptGrid3D'].shape[0]
        print("MP2StructureFactorTruncated nG: ", nG)
        if adaptive_sq_ke_cutoff:
            print("MP2StructureFactorTruncated outer nG: ", mesh_contexts['outer']['rptGrid3D'].shape[0])
        print("MP2StructureFactorTruncated nqG: ", nqG)
        SqG_full_direct = np.zeros(nqG, dtype=np.float64)
        SqG_full_exchange = np.zeros(nqG, dtype=np.float64)
        print("SqG direct MEM USAGE (KB) IS: {:.3f}".format(SqG_full_direct.nbytes / (1024)))
        print("SqG exchange MEM USAGE (KB) IS: {:.3f}".format(SqG_full_exchange.nbytes / (1024)))

        kgrid2_tree = KDTree(kGrid2)
        kgrid3_tree = KDTree(kGrid3)
        num_equiv_qG = 0

        phase_t0 = profile.start()
        qG_keep_mask = np.linalg.norm(qG_full, axis=1) < qG_cutoff + 1e-8
        qG_line_sampling_segments = filter_line_sampling_segments(
            qG_line_sampling_segments, qG_keep_mask)
        qG_full = qG_full[qG_keep_mask,:]
        SqG_full_direct = np.zeros(qG_full.shape[0], dtype=np.float64)
        SqG_full_exchange = np.zeros(qG_full.shape[0], dtype=np.float64)
        SqG_full_q4 = np.zeros(qG_full.shape[0], dtype=np.float64)
        SqG_full_direct_mask = np.ones(qG_full.shape[0], dtype=bool)
        SqG_full_exchange_mask = np.ones(qG_full.shape[0], dtype=bool)
        SqG_full_q4_mask = np.ones(qG_full.shape[0], dtype=bool)
        qG_norms = np.linalg.norm(qG_full, axis=1)
        if adaptive_sq_ke_cutoff:
            qG_uses_outer_mesh = qG_norms > sq_ke_cutoff_switch_radius + 1e-8
            sq_ke_cutoff_by_qG = np.where(
                qG_uses_outer_mesh,
                mesh_contexts['outer']['sq_ke_cutoff'],
                mesh_contexts['inner']['sq_ke_cutoff'],
            )
        else:
            qG_uses_outer_mesh = np.zeros(qG_full.shape[0], dtype=bool)
            sq_ke_cutoff_by_qG = None
        q4_decay_state = None
        exchange_decay_state = None
        if dG0 and "direct_q4" in line_sampling_decay_components:
            q4_decay_state = make_line_sampling_decay_state(
                qG_full,
                qG_line_sampling_segments,
                line_sampling_decay_min_fraction,
                line_sampling_decay_consecutive_below,
                power=4,
            )
            if q4_decay_state is not None:
                if direct:
                    SqG_full_direct_mask[:] = False
                SqG_full_q4_mask[:] = False
        if exchange and "exchange" in line_sampling_decay_components:
            exchange_decay_state = make_line_sampling_decay_state(
                qG_full,
                qG_line_sampling_segments,
                line_sampling_decay_min_fraction,
                line_sampling_decay_consecutive_below,
                power=2,
            )
            if exchange_decay_state is not None:
                SqG_full_exchange_mask[:] = False
        if t2_required and t2_store_type == 'kikjka':
            # Convert ki,kj,q -> qi,ki,kj
            t2 = t2.transpose(2, 0, 1, 3, 4, 5, 6)
        # t2_test = np.zeros_like(t2)
        
        # Map all qG points to the first Brillouin zone
        qG_full_BZ = minimum_image(kmf.cell, qG_full)
        
        qtree = scipy.spatial.KDTree(grids.qGrid)
        
        # Find indices of qG_full_BZ in the qGrid
        _, qi_map = qtree.query(qG_full_BZ, distance_upper_bound=1e-8)
        if np.any(qi_map == len(qGrid)):
            raise TypeError("Cannot locate qG_full_BZ in the qmesh.")

        kikj_on_the_fly = t2_store_type == 'kikj' and not t2_given
        t2_cache = {}
        t2_cache_counts = {
            'direct_hits': 0,
            'direct_misses': 0,
            'exchange_hits': 0,
            'exchange_misses': 0,
        }
        kii = None
        kjj = None
        repeated_qis = set()
        if kikj_on_the_fly:
            unique_qis, qi_counts = np.unique(qi_map, return_counts=True)
            repeated_qis = {int(qi) for qi, count in zip(unique_qis, qi_counts)
                            if count > 1}
            log.note(
                "kikj t2 cache: %d unique qi, %d repeated qi",
                len(unique_qis), len(repeated_qis),
            )
    
        kptas = kGrid1[None,:,:] + qGrid[:,None,:]
        kptbs = kGrid1[None,:,:] - qGrid[:,None,:]
        kptas_BZ = minimum_image(kmf.cell, kptas.reshape(-1,3))
        kptbs_BZ = minimum_image(kmf.cell, kptbs.reshape(-1,3))
        _, kas = kgrid2_tree.query(kptas_BZ, distance_upper_bound=1e-8)
        _, kbs = kgrid3_tree.query(kptbs_BZ, distance_upper_bound=1e-8)
        
        kas = kas.reshape(nkpts,nkpts) # qi, ki -> kas
        kbs = kbs.reshape(nkpts,nkpts) # qi, kj -> kbs

        # Precomoute the qpis
        qpis_full = None
        eijab_full = None
        if t2_store_type == 'kikjka':
            # precompute qp and eijab for all q, ki, kj.
            # Results in qpis_full and eijab_full which are both O(Nk^3)
            
            if exchange:
                qp_pts = np.zeros((nkpts,nkpts,nkpts,3), dtype=np.float64)
                qp_pts[:,:,:,:] =  - kGrid1[:,  None, None, :] + kGrid1[None, :, None, :] - qGrid[None, None, :, :] # q' = -ki + kj + q
                qp_pts = minimum_image(kmf.cell, qp_pts.reshape(-1,3))
                _, qpis_full = qtree.query(qp_pts, distance_upper_bound=1e-8)
                if np.any(qpis_full == len(qGrid)):
                    raise TypeError("Cannot locate qpi in the qmesh.")
                qpis_full = qpis_full.reshape(nkpts,nkpts,nkpts)
                qpis_full = qpis_full.transpose(2,0,1) # q', ki, kj
            
            # Compute Delta E matrix for q, ki, kj.
            eijab_full = np.zeros((nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=mo_energy[0][0].dtype)
            
            for qi in range(nkpts):
                ka_at_qi = kas[qi]
                kb_at_qi = kbs[qi]
                eijab = mo_e_o[:,None,:,None,None,None] + mo_e_o[None,:,None,:,None,None] \
                    -mo_e_v[ka_at_qi,None,None,None,:,None] - mo_e_v_b[None,kb_at_qi,None,None,None,:]
                eijab_full[qi,:,:,:,:,:,:] = 1/(eijab)
            if exchange:
                kii, kjj = np.indices((nkpts, nkpts))
                
        contract_expression_rijab = 'mia,nbj->mnijab'
        use_trs_representative_rijab = (
            kgrid_occ_trs and trs_map is not None and check_trs
            and t2_store_type in ('kikjka', 'kikj')
        )
        trs_pair_representatives = None
        if use_trs_representative_rijab:
            trs_pair_representatives = self._build_trs_pair_representatives(
                trs_map, nkpts)
            log.note(
                "TRS rijab representative pairs: %d of %d k-point pairs",
                len(trs_pair_representatives), nkpts * nkpts,
            )
        profile.stop("qG/k-point precomputation", phase_t0)
            
        nqG_full = qG_full.shape[0]
        interval = nqG_full // 10
        interval = max(interval, 1)
        inversion_partner = None
        if self.sq_inversion_symm:
            qg_tree = scipy.spatial.KDTree(qG_full)
            _, inversion_partner = qg_tree.query(-qG_full, distance_upper_bound=1e-8)

        loop_t0 = profile.start()
        for qG in range(qG_full.shape[0]):
            if qG % interval == 0:
                print(f"Progress: {qG/nqG_full*100:.2f}%")
                print(f"Wall time: {logger.perf_counter()-total_t0[1]:.2f}s")
                
            # First, see if S(-qG) has already been computed
            # precompute all idx_kpta and idx_kptb for all kpts
            qGpt = qG_full[qG, :]
            compute_direct = direct
            if direct and q4_decay_state is not None:
                compute_direct = should_compute_line_sample(qG, q4_decay_state)
            compute_exchange = exchange
            if exchange and exchange_decay_state is not None:
                compute_exchange = should_compute_line_sample(qG, exchange_decay_state)
            compute_q4 = dG0 and should_compute_line_sample(qG, q4_decay_state)
            region_t0 = profile.start()
            if self.sq_inversion_symm and qG > 1:
                equiv_qG_index = inversion_partner[qG]
                if equiv_qG_index < qG:
                    num_equiv_qG += 1
                    if compute_direct:
                        SqG_full_direct[qG] = SqG_full_direct[equiv_qG_index]
                        SqG_full_direct_mask[qG] = SqG_full_direct_mask[equiv_qG_index]
                    if compute_exchange:
                        SqG_full_exchange[qG] = SqG_full_exchange[equiv_qG_index]
                        SqG_full_exchange_mask[qG] = SqG_full_exchange_mask[equiv_qG_index]
                    if compute_q4:
                        SqG_full_q4[qG] = SqG_full_q4[equiv_qG_index]
                        SqG_full_q4_mask[qG] = SqG_full_q4_mask[equiv_qG_index]
                    profile.stop("inversion-symmetry reuse", region_t0)
                    continue
            profile.stop("inversion-symmetry lookup", region_t0)

            if not compute_direct and not compute_exchange and not compute_q4:
                continue

            # Find qi index
            region_t0 = profile.start()
            mesh_context = mesh_contexts['outer'] if qG_uses_outer_mesh[qG] else mesh_contexts['inner']
            rptGrid3D = mesh_context['rptGrid3D']
            quadrature_weights = mesh_context['quadrature_weights']
            quadrature_product_scale = mesh_context['quadrature_product_scale']
            uKpts_i = mesh_context['uKpts_i']
            uKpts_j = mesh_context['uKpts_j']
            uKpts_a = mesh_context['uKpts_a']
            uKpts_b = mesh_context['uKpts_b']
            conj_uKpts_i = mesh_context['conj_uKpts_i']
            qi = qi_map[qG]
            qpt = qGrid[qi]
            if compute_exchange and t2_store_type == 'kikjka':
                qpis = qpis_full[qi] # nkpts x nkpts.

            t2_qi = None
            if t2_required and t2_store_type == 'kikjka':  
                t2_qi = t2[qi]

            kptas = kGrid1 + qGpt
            kptbs = kGrid1 - qGpt
            kptas_BZ = minimum_image(kmf.cell, kptas)
            kptbs_BZ = minimum_image(kmf.cell, kptbs)

            # combined_kptgrid = np.concatenate((kptas_BZ, kptbs_BZ), axis=0)
            # _, unique_indices, _ = kpts_helper.unique(combined_kptgrid)
            # combined_kptgrid = combined_kptgrid[unique_indices]
            # try:
            #     trs_map = kpts_helper.conj_mapping(kmf.cell, combined_kptgrid)
            #     kgrid_occ_trs = True
            # except kpts_helper.KPointSymmetryError:
            #     trs_map = None
            #     kgrid_occ_trs = False
            
            kGdiffas = kptas - kptas_BZ
            kGdiffbs = kptbs - kptbs_BZ
            
            kas_at_qi = kas[qi]
            kbs_at_qi = kbs[qi]
            if kgrid_occ_trs and trs_map is not None and check_trs:
                # Use time-reversal symmetry 
                exp_term_as = np.exp(-1j * (rptGrid3D @ kGdiffas.T)).T
                profile.stop("per-qG index/phase setup", region_t0)
                
                # Build pair densities, rho_ikiaka and rho_jkjbkb.
                # Apply the phase on the occupied side to avoid nvir-sized
                # temporaries on the real-space grid.
                region_t0 = profile.start()
                phased_conj_ui = conj_uKpts_i * exp_term_as[:,None,:] # nkpts x nocc x nG
                if quadrature_weights is not None:
                    phased_conj_ui = phased_conj_ui * quadrature_weights[None,None,:]
                profile.stop("pair-density elementwise products", region_t0)
                region_t0 = profile.start()
                rho_ia_full = phased_conj_ui @ uKpts_a[kas_at_qi].transpose(0,2,1) # nkpts x nocc x nvir
                profile.stop("pair-density matrix multiply", region_t0)
                    
                rho_jb_full = rho_ia_full[trs_map,:,:].transpose(0,2,1) # nkpts x nvir x nocc
            else:
                # Precompute exp_term for kGdiffas and kGdiffbs
                exp_term_as = np.exp(-1j * (rptGrid3D @ kGdiffas.T)).T
                exp_term_bs = np.exp(1j * (rptGrid3D @ kGdiffbs.T)).T
                profile.stop("per-qG index/phase setup", region_t0)

                region_t0 = profile.start()
                phased_conj_ui = conj_uKpts_i * exp_term_as[:,None,:] # nkpts x nocc x nG
                if quadrature_weights is not None:
                    phased_conj_ui = phased_conj_ui * quadrature_weights[None,None,:]
                profile.stop("pair-density elementwise products", region_t0)
                region_t0 = profile.start()
                rho_ia_full = phased_conj_ui @ uKpts_a[kas_at_qi].transpose(0,2,1) # nkpts x nocc x nvir
                profile.stop("pair-density matrix multiply", region_t0)
                    
                region_t0 = profile.start()
                phased_uj = uKpts_j * exp_term_bs[:,None,:]
                if quadrature_weights is not None:
                    phased_uj = phased_uj * quadrature_weights[None,None,:]
                phased_uj_T = phased_uj.transpose(0,2,1) # nkpts x nG x nocc
                profile.stop("pair-density elementwise products", region_t0)
                region_t0 = profile.start()
                rho_jb_full = np.conj(uKpts_b[kbs_at_qi]) @ phased_uj_T # nkpts x nvir x nocc
                profile.stop("pair-density matrix multiply", region_t0)
        

            if t2_store_type == 'ki' and not t2_given:
                # O(Nk) memory scaling pathway
                oovv_ij = None
                oovv_ji = None
                if compute_exchange:
                    oovv_ji = np.zeros((nkpts,nocc,nocc,nvir,nvir), dtype=mo_coeff_kpts1[0].dtype)
                if compute_direct:
                    oovv_ij = np.zeros((nkpts,nocc,nocc,nvir,nvir), dtype=mo_coeff_kpts1[0].dtype)

                for ki in range(nkpts):
                    # Build ERIs with DF integrals
                    region_t0 = profile.start()
                    ka = kas_at_qi[ki]
                    for kj in range(nkpts):
                        kb = kbs_at_qi[kj]
                        if with_df_ints:
                            assert Lov.ndim == Lov_b.ndim, "Lov and Lov_b must have the same number of dimensions"
                            
                            if Lov.ndim == 4:
                                Lov_linear_scaling = True
                            elif Lov.ndim == 2: # HACK: should in principle be a 5D array.
                                Lov_linear_scaling = False
                            else: 
                                raise ValueError("Lov must be a 3D or 4D array")

                            if compute_direct:
                                Lov_kika = None
                                Lov_kjkb = None
                                if Lov_linear_scaling:
                                    Lov_kika = Lov[ki]
                                    Lov_kjkb = Lov_b[kj]
                                else:
                                    Lov_kika = Lov[ki, ka]
                                    Lov_kjkb = Lov_b[kj, kb]
                                
                                oovv_ij[kj] = (1./nkpts) * lib.einsum(
                                    "Lia,Ljb->iajb",
                                    Lov_kika, Lov_kjkb
                                ).transpose(0,2,1,3)

                            if compute_exchange:
                                oovv_ji[kj] = (1./nkpts) * lib.einsum(
                                    "Lia,Ljb->iajb",
                                    Lov_b[ki, kb], Lov[kj, ka]
                                ).transpose(0,2,1,3)
                                
                        else:
                            orbo_i = mo_coeff_kpts1[ki][:,:nocc]
                            orbo_j = mo_coeff_kpts1[kj][:,:nocc]
                            orbv_a = mo_coeff_kpts2[ka][:,nocc:]
                            orbv_b = mo_coeff_kpts3[kb][:,nocc:]
                            if compute_direct:
                                oovv_ij[kj] = fao2mo(
                                    (orbo_i,orbv_a,orbo_j,orbv_b),
                                    (kGrid1[ki],kGrid2[ka],kGrid1[kj],kGrid3[kb]),
                                    compact=False
                                ).reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3) / nkpts
                            if compute_exchange:
                                oovv_ji[kj] = fao2mo(
                                    (orbo_i,orbv_b,orbo_j,orbv_a),
                                    (kGrid1[ki],kGrid3[kb],kGrid1[kj],kGrid2[ka]),
                                    compact=False
                                ).reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3) / nkpts              
                    profile.stop("ERI/DF construction (ki path)", region_t0)
                    # Build eijab for direct


                    # Compute structure factor contribution
                    region_t0 = profile.start()
                    rijab_ki = np.einsum('ia,nbj->nijab', rho_ia_full[ki], rho_jb_full.conj()) * quadrature_product_scale / (nkpts * omega_cell)
                    profile.stop("rijab construction (ki path)", region_t0)
                    if compute_direct or compute_q4:
                        region_t0 = profile.start()
                        eijab_ki = mo_e_o[ki,None,:,None,None,None] + mo_e_o[:,None,:,None,None] \
                            -mo_e_v[ka,None,None,None,:,None] - mo_e_v_b[kbs_at_qi,None,None,None,:]
                        profile.stop("energy denominators (ki path)", region_t0)

                    if compute_direct:
                        region_t0 = profile.start()
                        t2_ki = np.conj(oovv_ij / eijab_ki) # kj, i, j, a, b
                        temp_SqG_k = 2 * pyscf_einsum('nijab,nijab->', rijab_ki, t2_ki) 
                        SqG_full_direct[qG] += temp_SqG_k.real / nkpts
                        profile.stop("direct contraction (ki path)", region_t0)
                    if compute_exchange:
                        region_t0 = profile.start()
                        eijba_ki = mo_e_o[ki,None,:,None,None,None] + mo_e_o[:,None,:,None,None] \
                            -mo_e_v_b[kbs_at_qi,None,None,:,None] - mo_e_v[ka,None,None,None,None,:]
                        t2_ki_x = np.conj(oovv_ji / eijba_ki) # kj, i, j, a, b
                        temp_SqG_k = - pyscf_einsum('nijab,nijba->', rijab_ki, t2_ki_x) 
                        SqG_full_exchange[qG] += temp_SqG_k.real / nkpts
                        profile.stop("exchange contraction (ki path)", region_t0)
                    if compute_q4:
                        region_t0 = profile.start()
                        temp_SqG_k =  2 * np.sum(np.abs(rijab_ki)**2 / eijab_ki)
                        SqG_full_q4[qG] += temp_SqG_k.real / nkpts
                        profile.stop("dG0 contraction (ki path)", region_t0)

                if compute_q4:
                    update_line_sampling_decay_mask(
                        SqG_full_q4_mask,
                        qG,
                        SqG_full_q4[qG],
                        np.linalg.norm(qGpt),
                        q4_decay_state,
                        qGpt=qGpt,
                    )
                    if direct and q4_decay_state is not None:
                        SqG_full_direct_mask[qG] = SqG_full_q4_mask[qG]
                if compute_exchange:
                    update_line_sampling_decay_mask(
                        SqG_full_exchange_mask,
                        qG,
                        SqG_full_exchange[qG],
                        np.linalg.norm(qGpt),
                        exchange_decay_state,
                        qGpt=qGpt,
                    )

                oovv_ij = None
                oovv_ji = None
                eijab_ki = None

            else:
                # O(Nk^2) or O(Nk^3) memory scaling pathway
                t2_qpi = None
                eijab = None
                if compute_direct:
                    if t2_store_type == 'kikj' and not t2_given:
                        cache_key = ('direct', int(qi))
                        eijab_cache_key = ('eijab', int(qi))
                        if cache_key in t2_cache:
                            region_t0 = profile.start()
                            t2_qi = t2_cache[cache_key]
                            t2_cache_counts['direct_hits'] += 1
                            profile.stop("direct t2 cache hit", region_t0)
                        else:
                            region_t0 = profile.start()
                            return_eijab_recip = compute_q4
                            t2_result = compute_t2_amplitudes(self.kmp, self.kmp.mo_energy, self.kmp.mo_coeff, qGrid, qGrid_sample=qpt.reshape(1,3),
                                                                    skip_if_no_qpt=True, mode='direct', Lov=Lov, verbose=logger.NOTE,
                                                                    return_eijab_recip=return_eijab_recip)
                            if return_eijab_recip:
                                t2_qi, eijab = t2_result
                                eijab = eijab.transpose(2,0,1,3,4,5,6)
                                eijab = eijab[0]
                            else:
                                t2_qi = t2_result
                            t2_qi = t2_qi.transpose(2,0,1,3,4,5,6)
                            t2_qi = t2_qi[0]
                            if int(qi) in repeated_qis:
                                t2_cache[cache_key] = t2_qi
                                if return_eijab_recip:
                                    t2_cache[eijab_cache_key] = eijab
                            t2_cache_counts['direct_misses'] += 1
                            profile.stop("direct t2 cache miss", region_t0)

                if compute_exchange:
                    if t2_store_type == 'kikj' and not t2_given:
                        cache_key = ('exchange', int(qi))
                        if cache_key in t2_cache:
                            region_t0 = profile.start()
                            t2_qpi = t2_cache[cache_key]
                            t2_cache_counts['exchange_hits'] += 1
                            profile.stop("exchange t2 cache hit", region_t0)
                        else:
                            region_t0 = profile.start()
                            t2_qpi = compute_t2_amplitudes(self.kmp, self.kmp.mo_energy, self.kmp.mo_coeff, qGrid, qGrid_sample=qpt.reshape(1,3),
                                                                skip_if_no_qpt=True, mode='exchange', Lov=Lov, verbose=logger.NOTE)
                            t2_qpi = t2_qpi.transpose(2,0,1,3,4,5,6) # qpi, ki, kj, i, j, a, b
                            t2_qpi = t2_qpi[0]
                            if int(qi) in repeated_qis:
                                t2_cache[cache_key] = t2_qpi
                            t2_cache_counts['exchange_misses'] += 1
                            profile.stop("exchange t2 cache miss", region_t0)
                    else:
                        region_t0 = profile.start()
                        t2_qpi = t2[qpis, kii, kjj]
                        profile.stop("exchange t2 gather", region_t0)

                if compute_q4:
                    region_t0 = profile.start()
                    if t2_store_type == 'kikj':
                        eijab_cache_key = ('eijab', int(qi))
                        if eijab is None:
                            if eijab_cache_key in t2_cache:
                                eijab = t2_cache[eijab_cache_key]
                            else:
                                _, eijab = compute_t2_amplitudes(
                                    self.kmp,
                                    self.kmp.mo_energy,
                                    self.kmp.mo_coeff,
                                    qGrid,
                                    qGrid_sample=qpt.reshape(1,3),
                                    skip_if_no_qpt=True,
                                    mode='direct',
                                    Lov=Lov,
                                    verbose=logger.NOTE,
                                    with_t2=False,
                                    return_eijab_recip=True,
                                )
                                eijab = eijab.transpose(2,0,1,3,4,5,6)
                                eijab = eijab[0]
                                if int(qi) in repeated_qis:
                                    t2_cache[eijab_cache_key] = eijab
                    else:
                        eijab = eijab_full[qi]
                    profile.stop("dG0 denominator setup", region_t0)

                if use_trs_representative_rijab:
                    region_t0 = profile.start()
                    direct_sum, exchange_sum, q4_weighted_norm = (
                        self._contract_trs_representative_rijab(
                            rho_ia_full,
                            rho_jb_full,
                            trs_pair_representatives,
                            direct_t2=t2_qi if compute_direct else None,
                            exchange_t2=t2_qpi if compute_exchange else None,
                            eijab_recip=eijab if compute_q4 else None,
                            profile=profile,
                        )
                    )
                    profile.stop("rijab/t2 TRS representative contraction", region_t0)

                    if compute_direct:
                        temp_SqG_k = (
                            2 / (omega_cell * nkpts)
                            * direct_sum
                            * quadrature_product_scale
                        )
                        SqG_full_direct[qG] += temp_SqG_k / nkpts
                    if compute_exchange:
                        temp_SqG_k_x = (
                            -1 / (omega_cell * nkpts)
                            * exchange_sum
                            * quadrature_product_scale
                        )
                        SqG_full_exchange[qG] += temp_SqG_k_x / nkpts
                    if compute_q4:
                        scale = quadrature_product_scale / (nkpts * omega_cell)
                        temp_SqG_k_q4 = -2 * scale**2 * q4_weighted_norm
                        SqG_full_q4[qG] += temp_SqG_k_q4.real / nkpts

                else:
                    region_t0 = profile.start()
                    rijab = np.einsum(contract_expression_rijab,rho_ia_full,rho_jb_full.conj(),optimize=True)
                    rijab = rijab.ravel()
                    profile.stop("rijab tensor contraction", region_t0)

                    if compute_direct:
                        region_t0 = profile.start()
                        temp_SqG_k =2/(omega_cell*nkpts) * np.dot(rijab, t2_qi.ravel()) * quadrature_product_scale #ORIGINAL 3/3/26
                        # temp_SqG_k =2/(omega_cell*nkpts) * pyscf_einsum('i,i->', rijab, t2_qi.ravel()) * dvol**2 #NEW 3/3/26

                        SqG_full_direct[qG] += temp_SqG_k.real / nkpts
                        profile.stop("direct final contraction", region_t0)

                    if compute_exchange:
                        region_t0 = profile.start()
                        t2_qpi_flat = np.ascontiguousarray(
                            t2_qpi.transpose(0,1,2,3,5,4)
                        ).ravel()
                        profile.stop("exchange t2 transpose/flatten", region_t0)

                        region_t0 = profile.start()
                        temp_SqG_k_x = -1/(omega_cell*nkpts) * np.dot(rijab, t2_qpi_flat) * quadrature_product_scale #ORIGINAL 3/3/26
                        # temp_SqG_k_x = -1/(omega_cell*nkpts) * pyscf_einsum('i,i->', rijab, t2_qpi) * dvol**2 #NEW 3/3/26

                        SqG_full_exchange[qG] += temp_SqG_k_x.real / nkpts
                        profile.stop("exchange final contraction", region_t0)

                    if compute_q4:
                        region_t0 = profile.start()
                        scale = quadrature_product_scale / (nkpts * omega_cell)
                        temp_SqG_k_q4 = self.contract_kikj_dG0(rijab, eijab, scale)
                        SqG_full_q4[qG] += temp_SqG_k_q4.real / nkpts
                        profile.stop("dG0 final contraction", region_t0)

                if compute_exchange:
                    update_line_sampling_decay_mask(
                        SqG_full_exchange_mask,
                        qG,
                        SqG_full_exchange[qG],
                        np.linalg.norm(qGpt),
                        exchange_decay_state,
                        qGpt=qGpt,
                    )

                if compute_q4:
                    update_line_sampling_decay_mask(
                        SqG_full_q4_mask,
                        qG,
                        SqG_full_q4[qG],
                        np.linalg.norm(qGpt),
                        q4_decay_state,
                        qGpt=qGpt,
                    )
                    if direct and q4_decay_state is not None:
                        SqG_full_direct_mask[qG] = SqG_full_q4_mask[qG]
        profile.stop("main qG loop (inclusive)", loop_t0)

        log.note("Number of equivalent qG points: %d", num_equiv_qG)
        line_sampling_decay_events = []
        if q4_decay_state is not None:
            line_sampling_decay_events = list(q4_decay_state["stop_events"])
            for event in line_sampling_decay_events:
                event["component"] = "direct_q4"
                log.note(
                    "Line-sampling decay reached for direct_q4 on B_index %d at |q+G| %.8g "
                    "(qG index %d, normalized %.8g < threshold %.8g)",
                    event["B_index"], event["qG_norm"], event["qG_index"],
                    event["normalized_contribution"], event["threshold"],
                )
        if exchange_decay_state is not None:
            exchange_decay_events = list(exchange_decay_state["stop_events"])
            line_sampling_decay_events.extend(exchange_decay_events)
            for event in exchange_decay_events:
                event["component"] = "exchange"
                log.note(
                    "Line-sampling decay reached for exchange on B_index %d at |q+G| %.8g "
                    "(qG index %d, normalized %.8g < threshold %.8g)",
                    event["B_index"], event["qG_norm"], event["qG_index"],
                    event["normalized_contribution"], event["threshold"],
                )
        if kikj_on_the_fly:
            cache_mem = sum(value.nbytes for value in t2_cache.values()) / 1024**2
            log.note(
                "kikj t2 cache: direct hits/misses %d/%d, exchange hits/misses %d/%d, retained %.3f MB",
                t2_cache_counts['direct_hits'], t2_cache_counts['direct_misses'],
                t2_cache_counts['exchange_hits'], t2_cache_counts['exchange_misses'],
                cache_mem,
            )
        if update_class:
            phase_t0 = profile.start()
            self.SqG_full_direct = SqG_full_direct
            self.SqG_full_exchange = SqG_full_exchange
            self.SqG_full_q4 = SqG_full_q4
            self.SqG_full_direct_mask = SqG_full_direct_mask
            self.SqG_full_exchange_mask = SqG_full_exchange_mask
            self.SqG_full_q4_mask = SqG_full_q4_mask
            self.line_sampling_decay_events = line_sampling_decay_events
            self.qG_full = qG_full
            self.sq_ke_cutoff_by_qG = sq_ke_cutoff_by_qG
            self.N_local_by_region = {
                key: value['N_local'].copy()
                for key, value in mesh_contexts.items()
            }
            profile.stop("class result update", phase_t0)
                
        # Find zero index
        norms = np.linalg.norm(qG_full, axis=1)
        zero_idx = np.where(norms < 1e-8)[0]
        if len(zero_idx) > 0:
            print("MP2StructureFactorTruncated: S_direct(0) = ",SqG_full_direct[zero_idx])
            print("MP2StructureFactorTruncated: S_exchange(0) = ",SqG_full_exchange[zero_idx])
            print("MP2StructureFactorTruncated: S_q4(0) = ",SqG_full_q4[zero_idx])


        result_dict = {
            'SqG_full_direct': SqG_full_direct,
            'SqG_full_exchange': SqG_full_exchange,
            'SqG_full_q4': SqG_full_q4,
            'SqG_full_direct_mask': SqG_full_direct_mask,
            'SqG_full_exchange_mask': SqG_full_exchange_mask,
            'SqG_full_q4_mask': SqG_full_q4_mask,
            'line_sampling_decay_events': line_sampling_decay_events,
            'sq_ke_cutoff_by_qG': sq_ke_cutoff_by_qG,
            'N_local_by_region': {
                key: value['N_local'].copy()
                for key, value in mesh_contexts.items()
            },
            'qG_full': qG_full,
        }
        self.last_build_timings = profile.summary(total_t0)
        profile.log_summary(log, self.last_build_timings)
        return result_dict
