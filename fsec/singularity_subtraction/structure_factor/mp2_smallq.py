"""Half-shifted small-q direct MP2 structure factors."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import KDTree

from pyscf import lib
from pyscf.ao2mo import _ao2mo
from pyscf.lib import logger
from pyscf.pbc import df
from pyscf.pbc.df.df import (
    CDERIArray,
    _CCGDFBuilder,
    _RSGDFBuilder,
)
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.mp import kmp2
from pyscf.pbc.tools import get_monkhorst_pack_size

from fsec.singularity_subtraction.grids import MP2SSGrids, minimum_image
from fsec.singularity_subtraction.structure_factor.helpers import TimingProfile
from fsec.singularity_subtraction.structure_factor.mp2_sf import (
    MP2StructureFactor,
)


def normalize_band_df(value):
    if value is None:
        raise ValueError("band_df must be 'FFTDF' or 'GDF'")
    value = str(value).strip().upper()
    if value not in ("FFTDF", "GDF"):
        raise ValueError("band_df must be 'FFTDF' or 'GDF'")
    return value


def map_kpts(cell, targets, grid, label):
    targets = minimum_image(cell, np.asarray(targets).reshape(-1, 3))
    _, indices = KDTree(grid).query(targets, distance_upper_bound=1e-8)
    if np.any(indices == len(grid)):
        raise ValueError(f"Cannot map {label} points onto the shifted k-grid")
    return np.asarray(indices, dtype=int)


def required_lov_pairs(ka_map, kb_map, nkpts):
    """Return unique combined-grid indices required by Lov and Lov_b."""
    pairs = [
        (ki, nkpts + int(ka))
        for ki, ka in enumerate(ka_map)
    ]
    pairs.extend(
        (kj, nkpts + int(kb))
        for kj, kb in enumerate(kb_map)
    )
    # dict preserves the first occurrence and therefore keeps the output
    # deterministic without sorting the two physical pair families.
    return list(dict.fromkeys(pairs))


def s2_pair_closure(kpt_pairs):
    """Add reverse orientations required to reconstruct s2 AO blocks."""
    pairs = list(kpt_pairs)
    pairs.extend((kj, ki) for ki, kj in kpt_pairs)
    return list(dict.fromkeys(pairs))


def trs_pair_representatives(s2_pairs, trs_map):
    """Choose a transpose-closed half of a TR-closed pair set."""
    s2_pairs = {
        (int(ki), int(kj)) for ki, kj in s2_pairs
    }
    trs_map = np.asarray(trs_map, dtype=int)

    def transpose(pair):
        return pair[1], pair[0]

    def time_reverse(pair):
        return int(trs_map[pair[0]]), int(trs_map[pair[1]])

    evaluated = set()
    visited = set()
    for pair in sorted(s2_pairs):
        if pair in visited:
            continue
        pair_t = transpose(pair)
        pair_tr = time_reverse(pair)
        pair_tr_t = transpose(pair_tr)
        symmetry_group = {pair, pair_t, pair_tr, pair_tr_t}
        if not symmetry_group.issubset(s2_pairs):
            raise ValueError(
                "s2 pair closure is not closed under time reversal"
            )
        direct_choice = {pair, pair_t}
        tr_choice = {pair_tr, pair_tr_t}
        chosen = min(
            (direct_choice, tr_choice),
            key=lambda choice: tuple(sorted(choice)),
        )
        evaluated.update(chosen)
        visited.update(symmetry_group)
    return sorted(evaluated)


class _SelectiveKPairBuilderMixin:
    """Restrict j3c work while retaining PySCF's s2 and q/-q symmetry."""

    selective_kk_idx = None

    def outcore_auxe2(
        self,
        cderi_file,
        intor="int3c2e",
        aosym="s2",
        comp=None,
        j_only=False,
        dataname="j3c",
        shls_slice=None,
        fft_dd_block=None,
        kk_idx=None,
    ):
        if kk_idx is not None:
            raise ValueError("selective kk_idx must be configured on the builder")
        return super().outcore_auxe2(
            cderi_file,
            intor,
            aosym,
            comp,
            j_only,
            dataname,
            shls_slice,
            fft_dd_block,
            kk_idx=self.selective_kk_idx,
        )

    def gen_uniq_kpts_groups(self, j_only, h5swap, kk_idx=None):
        if kk_idx is not None:
            raise ValueError("selective kk_idx must be configured on the builder")
        selected = set(np.asarray(self.selective_kk_idx, dtype=int))
        # Use the normal full-grid metric grouping. It factorizes one member
        # of each q/-q pair and generates the other by conjugation, preserving
        # the auxiliary gauge used by a regular GDF build.
        for kpt, group_indices, cd_j2c in super().gen_uniq_kpts_groups(
            j_only,
            h5swap,
            kk_idx=None,
        ):
            filtered = np.asarray(
                [idx for idx in group_indices if idx in selected],
                dtype=np.int32,
            )
            if filtered.size:
                yield kpt, filtered, cd_j2c


class _SelectiveRSGDFBuilder(
    _SelectiveKPairBuilderMixin,
    _RSGDFBuilder,
):
    pass


class _SelectiveCCGDFBuilder(
    _SelectiveKPairBuilderMixin,
    _CCGDFBuilder,
):
    pass


class _TRSCDERIArray:
    """Expose omitted time-reversal CDERI blocks by conjugation."""

    def __init__(self, cderi, evaluated_pairs, trs_map):
        self._array = CDERIArray(cderi)
        self.aosym = self._array.aosym
        self.evaluated_pairs = {
            (int(ki), int(kj)) for ki, kj in evaluated_pairs
        }
        self.trs_map = np.asarray(trs_map, dtype=int)

    def __getitem__(self, indices):
        if not (
            isinstance(indices, tuple)
            and len(indices) == 2
            and all(isinstance(index, (int, np.integer)) for index in indices)
        ):
            raise TypeError("selective CDERI access requires two integer indices")
        pair = (int(indices[0]), int(indices[1]))
        if pair in self.evaluated_pairs:
            return self._array[pair]
        partner = (
            int(self.trs_map[pair[0]]),
            int(self.trs_map[pair[1]]),
        )
        if partner not in self.evaluated_pairs:
            raise KeyError(
                f"CDERI pair {pair} and its TR partner {partner} were not built"
            )
        return self._array[partner].conj()

    def close(self):
        self._array.data_group.close()


@dataclass(frozen=True)
class MP2SmallQResult:
    """Direct structure-factor values at the positive half-shifted q point."""

    qprime: np.ndarray
    sq_direct: float
    sq_q4: float
    band_df: str
    band_exxdiv: Optional[str]


class MP2SmallQ:
    """Compute direct and fourth-order MP2 structure factors at one small q.

    The occupied orbitals are taken from the converged mean-field k-mesh.  The
    virtual orbitals and energies are evaluated non-self-consistently on the
    half-shifted mesh.
    """

    def __init__(
        self,
        kmf,
        kmp,
        *,
        band_df,
        band_exxdiv="ewald",
        N_local=None,
        sq_ke_cutoff=None,
        check_trs=True,
        pair_density_eval_grid="becke",
        pair_density_becke_grid_level=0,
        sq_ke_cutoff_switch_radius=None,
        outer_sq_ke_cutoff_scale=0.5,
        verbose=None,
    ):
        self.kmf = kmf
        self.kmp = kmp
        self.cell = kmf.cell
        self.band_df = normalize_band_df(band_df)
        self.band_exxdiv = (
            band_exxdiv.strip().lower()
            if isinstance(band_exxdiv, str)
            else band_exxdiv
        )
        self.N_local = (
            np.asarray(N_local, dtype=int)
            if N_local is not None
            else np.asarray(self.cell.mesh, dtype=int)
        )
        self.sq_ke_cutoff = sq_ke_cutoff
        self.check_trs = check_trs
        self.pair_density_eval_grid = pair_density_eval_grid
        self.pair_density_becke_grid_level = pair_density_becke_grid_level
        self.sq_ke_cutoff_switch_radius = sq_ke_cutoff_switch_radius
        self.outer_sq_ke_cutoff_scale = outer_sq_ke_cutoff_scale
        self.verbose = verbose
        self.result = None
        self.smallq_structure_factor = None
        self.last_kernel_timings = None

        self._validate_band_exxdiv()

    def _validate_band_exxdiv(self):
        if (
            self.band_df == "GDF"
            and self.band_exxdiv is not None
            and str(self.band_exxdiv).strip().lower() != "ewald"
        ):
            raise ValueError(
                "GDF bands only support smallq_band_exxdiv=None or 'ewald'; "
                f"got {self.band_exxdiv!r}"
            )

    def _build_grids(self):
        nks = np.asarray(
            get_monkhorst_pack_size(self.cell, self.kmf.kpts), dtype=int
        )
        scaled_half_shift = 0.5 / nks
        qprime = np.asarray(self.cell.get_abs_kpts(scaled_half_shift))
        grids = MP2SSGrids(
            self.cell,
            self.kmf.kpts,
            N_local=self.N_local,
            qG_norm_cutoff=np.inf,
            relative_shift=np.full(3, 0.5),
            shift_occ=False,
        )
        # The single-point evaluator only needs the coarse real-space
        # quadrature mesh; avoid constructing the full q+G grids.
        grids.build_RptGrid3D_coarse()
        return qprime, grids

    def _make_band_df(self, shifted_kpts):
        if self.band_df == "FFTDF":
            return df.FFTDF(self.cell, self.kmf.kpts)

        band_df = df.GDF(self.cell, self.kmf.kpts)
        source_df = self.kmf.with_df
        if isinstance(source_df, df.GDF):
            band_df.auxbasis = source_df.auxbasis
            if source_df.mesh is not None:
                band_df.mesh = np.asarray(source_df.mesh).copy()
            band_df.linear_dep_threshold = source_df.linear_dep_threshold
            band_df.exp_to_discard = source_df.exp_to_discard
            band_df._prefer_ccdf = source_df._prefer_ccdf
        band_df.build(kpts_band=shifted_kpts)
        return band_df

    def _get_shifted_bands(self, shifted_kpts, profile=None):
        phase_t0 = profile.start() if profile is not None else None
        band_df = self._make_band_df(shifted_kpts)
        if profile is not None:
            profile.stop("shifted-band DF construction", phase_t0)

        phase_t0 = profile.start() if profile is not None else None
        try:
            with lib.temporary_env(
                self.kmf,
                with_df=band_df,
                exxdiv=self.band_exxdiv,
            ):
                mo_energy, mo_coeff = self.kmf.get_bands(shifted_kpts)
        finally:
            if isinstance(band_df, df.GDF):
                cderi_temp = band_df._cderi_to_save
                if not isinstance(cderi_temp, str):
                    cderi_temp.close()
        if profile is not None:
            profile.stop("shifted-band Fock diagonalization", phase_t0)
        return list(np.asarray(mo_energy)), list(np.asarray(mo_coeff))

    def _make_correlation_gdf(self, combined_kpts, evaluated_pairs):
        source_df = self.kmf.with_df
        correlation_df = df.GDF(self.cell, combined_kpts)
        correlation_df.auxbasis = source_df.auxbasis
        if source_df.mesh is not None:
            correlation_df.mesh = np.asarray(source_df.mesh).copy()
        correlation_df.linear_dep_threshold = source_df.linear_dep_threshold
        correlation_df.exp_to_discard = source_df.exp_to_discard
        correlation_df._prefer_ccdf = source_df._prefer_ccdf
        # Initializing without j3c constructs the auxiliary basis while
        # avoiding the full Cartesian product of the combined k-point grid.
        correlation_df.build(j_only=False, with_j3c=False)
        cderi_file = correlation_df._cderi_to_save
        if not isinstance(cderi_file, str):
            cderi_file = cderi_file.name
        nkpts = len(combined_kpts)
        kk_idx = np.asarray(
            [ki * nkpts + kj for ki, kj in evaluated_pairs],
            dtype=np.int32,
        )
        if correlation_df._prefer_ccdf or self.cell.omega > 0:
            builder = _SelectiveCCGDFBuilder(
                self.cell,
                correlation_df.auxcell,
                combined_kpts,
            )
            builder.eta = correlation_df.eta
        else:
            builder = _SelectiveRSGDFBuilder(
                self.cell,
                correlation_df.auxcell,
                combined_kpts,
            )
        builder.mesh = correlation_df.mesh
        builder.linear_dep_threshold = (
            correlation_df.linear_dep_threshold
        )
        builder.selective_kk_idx = kk_idx
        builder.make_j3c(
            cderi_file,
            j_only=False,
            dataname=correlation_df._dataname,
            aosym="s2",
        )
        correlation_df._cderi = cderi_file
        return correlation_df

    def _build_lov(
        self,
        grids,
        qprime,
        mo_coeff_occ,
        mo_coeff_shifted,
        profile=None,
    ):
        """Build only the occupied-to-shifted GDF tensors needed at qprime."""
        nkpts = len(grids.kGrid1)
        combined_kpts = np.concatenate((grids.kGrid1, grids.kGrid2), axis=0)
        phase_t0 = profile.start() if profile is not None else None
        ka_map = map_kpts(
            self.cell,
            grids.kGrid1 + qprime,
            grids.kGrid2,
            "k_i + qprime",
        )
        kb_map = map_kpts(
            self.cell,
            grids.kGrid1 - qprime,
            grids.kGrid3,
            "k_j - qprime",
        )
        kpt_pairs = required_lov_pairs(ka_map, kb_map, nkpts)
        s2_pairs = s2_pair_closure(kpt_pairs)
        log = logger.new_logger(self.kmp, self.verbose)

        trs_enabled = True
        try:
            trs_occ = kpts_helper.conj_mapping(
                self.cell,
                grids.kGrid1,
            )
            trs_shifted = kpts_helper.conj_mapping(
                self.cell,
                grids.kGrid2,
            )
            if not np.array_equal(
                trs_shifted[ka_map],
                kb_map[trs_occ],
            ):
                raise ValueError(
                    "ia and jb mappings are not time-reversal equivariant"
                )
            combined_trs = np.concatenate(
                (trs_occ, nkpts + trs_shifted)
            )
            evaluated_pairs = trs_pair_representatives(
                s2_pairs,
                combined_trs,
            )
        except (kpts_helper.KPointSymmetryError, ValueError) as err:
            trs_enabled = False
            combined_trs = np.arange(len(combined_kpts), dtype=int)
            evaluated_pairs = s2_pairs
            log.warn(
                "Cannot reduce selective Lov GDF with time reversal (%s); "
                "using the transpose-closed s2 pair set",
                err,
            )

        log.note(
            "MP2 small-q selective Lov GDF pairs: %d requested, %d s2 "
            "closure, %d evaluated%s (%d full ordered pairs)",
            len(kpt_pairs),
            len(s2_pairs),
            len(evaluated_pairs),
            " with TRS" if trs_enabled else "",
            len(combined_kpts) ** 2,
        )
        if profile is not None:
            profile.stop("Lov setup and k-point mapping", phase_t0)

        phase_t0 = profile.start() if profile is not None else None
        correlation_df = self._make_correlation_gdf(
            combined_kpts,
            evaluated_pairs,
        )
        if profile is not None:
            profile.stop("correlation GDF construction", phase_t0)

        cderi_array = _TRSCDERIArray(
            correlation_df._cderi,
            evaluated_pairs,
            combined_trs,
        )
        if cderi_array.aosym != "s2":
            raise RuntimeError(
                "selective Lov GDF must retain aosym='s2'"
            )

        nocc = self.kmp.nocc
        nmo = self.kmp.nmo
        nvir = nmo - nocc
        nao = self.cell.nao_nr()
        dtype = np.result_type(
            np.complex128, *mo_coeff_occ, *mo_coeff_shifted
        )

        lov_a = np.empty(nkpts, dtype=object)
        lov_b = np.empty(nkpts, dtype=object)
        transform_cache = {}

        def transform(ko, kv):
            key = (int(ko), int(kv))
            if key in transform_cache:
                return transform_cache[key]

            phase_t0 = profile.start() if profile is not None else None
            Lpq_ao = cderi_array[ko, nkpts + kv]
            mo = np.hstack((mo_coeff_occ[ko], mo_coeff_shifted[kv]))
            mo = np.asarray(mo, dtype=dtype, order="F")
            if Lpq_ao[0].size != nao**2:
                Lpq_ao = lib.unpack_tril(Lpq_ao).astype(np.complex128)
            if profile is not None:
                profile.stop("Lov CDERI load/unpack", phase_t0)

            phase_t0 = profile.start() if profile is not None else None
            out = _ao2mo.r_e2(
                Lpq_ao,
                mo,
                (0, nocc, nmo + nocc, nmo + nocc + nvir),
                [],
                None,
            )
            if profile is not None:
                profile.stop("Lov AO-to-MO transformation", phase_t0)
            out = out.reshape(-1, nocc, nvir)
            transform_cache[key] = out
            return out

        try:
            for ki, ka in enumerate(ka_map):
                lov_a[ki] = transform(ki, ka)
            for kj, kb in enumerate(kb_map):
                lov_b[kj] = transform(kj, kb)
        finally:
            cderi_array.close()
            cderi_temp = correlation_df._cderi_to_save
            if not isinstance(cderi_temp, str):
                cderi_temp.close()
        return lov_a, lov_b

    def kernel(self):
        log = logger.new_logger(self.kmp, self.verbose)
        profile = TimingProfile()
        total_t0 = profile.start()

        phase_t0 = profile.start()
        qprime, grids = self._build_grids()
        profile.stop("small-q grid construction", phase_t0)

        shifted_energy, shifted_coeff = self._get_shifted_bands(
            grids.kGrid2,
            profile=profile,
        )

        phase_t0 = profile.start()
        mo_coeff_occ, mo_energy_occ = kmp2._add_padding(
            self.kmp, self.kmp.mo_coeff, self.kmp.mo_energy
        )
        mo_coeff_shifted, mo_energy_shifted = kmp2._add_padding(
            self.kmp, shifted_coeff, shifted_energy
        )
        # Shifted-grid integrals are generally complex even when the
        # origin-centered SCF coefficients happen to be real (for example at
        # Gamma).  The low-memory contraction allocates from this dtype.
        mo_coeff_occ = [
            np.asarray(coeff, dtype=np.complex128) for coeff in mo_coeff_occ
        ]
        mo_coeff_shifted = [
            np.asarray(coeff, dtype=np.complex128)
            for coeff in mo_coeff_shifted
        ]

        nocc = self.kmp.nocc
        mo_e_o = np.asarray(mo_energy_occ)[:, :nocc]
        mo_e_v = np.asarray(mo_energy_shifted)[:, nocc:]
        profile.stop("MO padding and dtype preparation", phase_t0)

        lov = lov_b = None
        if (
            self.kmp.with_df_ints
            and isinstance(self.kmf.with_df, df.GDF)
        ):
            lov, lov_b = self._build_lov(
                grids,
                qprime,
                mo_coeff_occ,
                mo_coeff_shifted,
                profile=profile,
            )

        phase_t0 = profile.start()
        structure_factor = MP2StructureFactor(
            self.kmf,
            self.kmp,
            t2=None,
            N_local=self.N_local,
            sq_ke_cutoff=self.sq_ke_cutoff,
            qG_cutoff=np.inf,
            sq_inversion_symm=False,
            check_trs=self.check_trs,
            t2_store_type="ki",
            pair_density_eval_grid=self.pair_density_eval_grid,
            pair_density_becke_grid_level=self.pair_density_becke_grid_level,
            sq_ke_cutoff_switch_radius=self.sq_ke_cutoff_switch_radius,
            outer_sq_ke_cutoff_scale=self.outer_sq_ke_cutoff_scale,
        )
        profile.stop("structure-factor setup", phase_t0)

        phase_t0 = profile.start()
        values = structure_factor.build_structure_factor(
            direct=True,
            exchange=False,
            dG0=True,
            qG_full=qprime.reshape(1, 3),
            update_class=False,
            qG_cutoff=np.inf,
            grids=grids,
            mo_coeff_kpts1=mo_coeff_occ,
            mo_coeff_kpts2=mo_coeff_shifted,
            mo_coeff_kpts3=mo_coeff_shifted,
            mo_energy=mo_energy_occ,
            mo_e_o=mo_e_o,
            mo_e_v=mo_e_v,
            mo_e_v_b=mo_e_v,
            t2_store_type="ki",
            Lov=lov,
            Lov_b=lov_b,
            kmp=self.kmp,
            verbose=self.verbose,
        )
        profile.stop("single-point structure factor", phase_t0)

        phase_t0 = profile.start()
        self.smallq_structure_factor = structure_factor
        self.result = MP2SmallQResult(
            qprime=qprime.copy(),
            sq_direct=float(values["SqG_full_direct"][0]),
            sq_q4=float(values["SqG_full_q4"][0]),
            band_df=self.band_df,
            band_exxdiv=self.band_exxdiv,
        )
        profile.stop("result packaging", phase_t0)
        self.last_kernel_timings = profile.summary(total_t0)
        profile.log_summary(
            log,
            self.last_kernel_timings,
            title="MP2 small-q kernel",
        )
        return self.result
