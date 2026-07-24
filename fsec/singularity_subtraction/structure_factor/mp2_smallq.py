"""Half-shifted small-q direct MP2 structure factors."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import KDTree

from pyscf import lib
from pyscf.ao2mo import _ao2mo
from pyscf.pbc import df
from pyscf.pbc.df.df import CDERIArray
from pyscf.pbc.mp import kmp2
from pyscf.pbc.tools import get_monkhorst_pack_size

from fsec.singularity_subtraction.grids import MP2SSGrids, minimum_image
from fsec.singularity_subtraction.structure_factor.mp2_sf import (
    MP2StructureFactor,
)


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
        self.band_df = self._normalize_band_df(band_df)
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

        self._validate_band_exxdiv()

    @staticmethod
    def _normalize_band_df(value):
        if value is None:
            raise ValueError("band_df must be 'FFTDF' or 'GDF'")
        value = str(value).strip().upper()
        if value not in ("FFTDF", "GDF"):
            raise ValueError("band_df must be 'FFTDF' or 'GDF'")
        return value

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

    def _get_shifted_bands(self, shifted_kpts):
        band_df = self._make_band_df(shifted_kpts)
        with lib.temporary_env(
            self.kmf,
            with_df=band_df,
            exxdiv=self.band_exxdiv,
        ):
            mo_energy, mo_coeff = self.kmf.get_bands(shifted_kpts)
        return list(np.asarray(mo_energy)), list(np.asarray(mo_coeff))

    @staticmethod
    def _map_kpts(cell, targets, grid, label):
        targets = minimum_image(cell, np.asarray(targets).reshape(-1, 3))
        _, indices = KDTree(grid).query(targets, distance_upper_bound=1e-8)
        if np.any(indices == len(grid)):
            raise ValueError(f"Cannot map {label} points onto the shifted k-grid")
        return np.asarray(indices, dtype=int)

    def _make_correlation_gdf(self, combined_kpts):
        source_df = self.kmf.with_df
        correlation_df = df.GDF(self.cell, combined_kpts)
        correlation_df.auxbasis = source_df.auxbasis
        if source_df.mesh is not None:
            correlation_df.mesh = np.asarray(source_df.mesh).copy()
        correlation_df.linear_dep_threshold = source_df.linear_dep_threshold
        correlation_df.exp_to_discard = source_df.exp_to_discard
        correlation_df._prefer_ccdf = source_df._prefer_ccdf
        correlation_df.build()
        return correlation_df

    def _build_lov(
        self,
        grids,
        qprime,
        mo_coeff_occ,
        mo_coeff_shifted,
    ):
        """Build only the occupied-to-shifted GDF tensors needed at qprime."""
        nkpts = len(grids.kGrid1)
        combined_kpts = np.concatenate((grids.kGrid1, grids.kGrid2), axis=0)
        correlation_df = self._make_correlation_gdf(combined_kpts)
        cderi_array = CDERIArray(correlation_df._cderi)

        ka_map = self._map_kpts(
            self.cell,
            grids.kGrid1 + qprime,
            grids.kGrid2,
            "k_i + qprime",
        )
        kb_map = self._map_kpts(
            self.cell,
            grids.kGrid1 - qprime,
            grids.kGrid3,
            "k_j - qprime",
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

        def transform(ko, kv):
            Lpq_ao = cderi_array[ko, nkpts + kv]
            mo = np.hstack((mo_coeff_occ[ko], mo_coeff_shifted[kv]))
            mo = np.asarray(mo, dtype=dtype, order="F")
            if Lpq_ao[0].size != nao**2:
                Lpq_ao = lib.unpack_tril(Lpq_ao).astype(np.complex128)
            out = _ao2mo.r_e2(
                Lpq_ao,
                mo,
                (0, nocc, nmo + nocc, nmo + nocc + nvir),
                [],
                None,
            )
            return out.reshape(-1, nocc, nvir)

        for ki, ka in enumerate(ka_map):
            lov_a[ki] = transform(ki, ka)
        for kj, kb in enumerate(kb_map):
            lov_b[kj] = transform(kj, kb)
        return lov_a, lov_b

    def kernel(self):
        qprime, grids = self._build_grids()
        shifted_energy, shifted_coeff = self._get_shifted_bands(grids.kGrid2)

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
            )

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
        self.smallq_structure_factor = structure_factor
        self.result = MP2SmallQResult(
            qprime=qprime.copy(),
            sq_direct=float(values["SqG_full_direct"][0]),
            sq_q4=float(values["SqG_full_q4"][0]),
            band_df=self.band_df,
            band_exxdiv=self.band_exxdiv,
        )
        return self.result
