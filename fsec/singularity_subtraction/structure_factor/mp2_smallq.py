"""MP2 direct structure factors at one shifted small-q point."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import KDTree

from pyscf.lib import logger
from pyscf.pbc import df, dft
from pyscf.pbc.mp import kmp2
from pyscf.pbc.tools import get_monkhorst_pack_size
from pyscf.pbc.tools.pbc import cutoff_to_mesh, mesh_to_cutoff

from fsec.singularity_subtraction.grids import minimum_image
from fsec.singularity_subtraction.structure_factor.helpers import build_uKpts


@dataclass(frozen=True)
class MP2SmallQOptions:
    """sTC settings for the non-self-consistent small-q virtual bands.

    ``relative_shift`` is in units of the original Monkhorst-Pack mesh and
    must be a nonzero vector in ``[-0.5, 0.5]^3``. ``cutoff`` selects the
    Wigner-Seitz or spherical sTC cutoff.
    """

    relative_shift: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    eta: float = 4.0
    cutoff: str = "ws"

    def __post_init__(self):
        shift = np.asarray(self.relative_shift, dtype=float)
        if shift.shape != (3,) or not np.all(np.isfinite(shift)):
            raise ValueError("relative_shift must be a finite three-vector")
        if np.all(shift == 0):
            raise ValueError("relative_shift must be nonzero")
        if np.any(np.abs(shift) > 0.5):
            raise ValueError("relative_shift components must lie in [-0.5, 0.5]")
        eta = float(self.eta)
        if not np.isfinite(eta) or eta <= 0:
            raise ValueError("eta must be finite and positive")
        cutoff = str(self.cutoff).strip().lower()
        if cutoff not in ("ws", "sph"):
            raise ValueError("cutoff must be 'ws' or 'sph'")
        object.__setattr__(self, "relative_shift", tuple(shift.tolist()))
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "cutoff", cutoff)


@dataclass(frozen=True)
class MP2SmallQResult:
    """Direct structure factors at ``qprime`` and their sTC band settings."""

    qprime: np.ndarray
    sq_direct: float
    sq_q4: float
    relative_shift: Tuple[float, float, float]
    eta: float
    cutoff: str


@dataclass(frozen=True)
class _SmallQGrids:
    """Original and shifted grids, with virtual point maps in original order."""

    qprime: np.ndarray
    occupied: np.ndarray
    plus: np.ndarray
    minus: np.ndarray
    minus_from_plus: Optional[np.ndarray]


def _map_kpts(cell, targets, grid, label):
    targets = minimum_image(cell, np.asarray(targets).reshape(-1, 3))
    grid = minimum_image(cell, np.asarray(grid).reshape(-1, 3))
    _, indices = KDTree(grid).query(targets, distance_upper_bound=1e-8)
    if np.any(indices == len(grid)):
        raise ValueError(f"Cannot map {label} points onto the shifted k-grid")
    return np.asarray(indices, dtype=int)


def _close_temporary_gdf(gdf_object):
    handle = getattr(gdf_object, "_cderi_to_save", None)
    if handle is not None and not isinstance(handle, str) and not handle.closed:
        handle.close()


def _copy_df_settings(source, target, mf):
    """Copy auxiliary-basis accuracy controls without sharing DF state."""
    target.max_memory = getattr(source, "max_memory", mf.max_memory)
    target.stdout = getattr(source, "stdout", mf.stdout)
    target.verbose = getattr(source, "verbose", mf.verbose)
    for name in (
        "auxbasis",
        "mesh",
        "linear_dep_threshold",
        "exp_to_discard",
        "_prefer_ccdf",
    ):
        if hasattr(source, name):
            value = getattr(source, name)
            if name == "mesh" and value is not None:
                value = np.asarray(value).copy()
            if name != "mesh" or value is not None:
                setattr(target, name, value)


class MP2SmallQ:
    """Evaluate direct MP2 structure factors at one arbitrary small shift.

    Occupied orbitals and energies always come from ``kmp`` (including any
    KMP2 energy shifts). Only virtual orbitals are recomputed on the shifted
    grids with smoothed truncated Coulomb exchange. Pair densities are
    integrated over the configured Becke or uniform real-space grid. Here
    ``qprime`` is a Cartesian vector in inverse Bohr; ``sq_q4`` is the
    fourth-order direct contribution from the same pair densities.
    """

    def __init__(
        self,
        kmf,
        kmp,
        options=None,
        *,
        N_local=None,
        sq_ke_cutoff=None,
        pair_density_eval_grid="becke",
        pair_density_becke_grid_level=0,
        sq_ke_cutoff_switch_radius=None,
        outer_sq_ke_cutoff_scale=0.5,
        verbose=None,
    ):
        self.kmf = kmf
        self.kmp = kmp
        self.cell = kmf.cell
        if options is None:
            options = MP2SmallQOptions()
        elif isinstance(options, dict):
            options = MP2SmallQOptions(**options)
        elif not isinstance(options, MP2SmallQOptions):
            raise TypeError("options must be MP2SmallQOptions, dict, or None")
        self.options = options

        self.sq_ke_cutoff = sq_ke_cutoff
        self.sq_ke_cutoff_switch_radius = sq_ke_cutoff_switch_radius
        self.outer_sq_ke_cutoff_scale = float(outer_sq_ke_cutoff_scale)
        self.pair_density_eval_grid = str(pair_density_eval_grid).strip().lower()
        if self.pair_density_eval_grid not in ("uniform", "becke"):
            raise ValueError("pair_density_eval_grid must be 'uniform' or 'becke'")
        self.pair_density_becke_grid_level = int(pair_density_becke_grid_level)
        if self.pair_density_becke_grid_level < 0:
            raise ValueError("pair_density_becke_grid_level must be non-negative")
        self.verbose = kmp.verbose if verbose is None else verbose

        if N_local is None:
            N_local = np.asarray(self.cell.mesh, dtype=int)
        if np.isscalar(N_local):
            N_local = np.full(3, int(N_local), dtype=int)
        else:
            N_local = np.asarray(N_local, dtype=int)
        if N_local.shape != (3,) or np.any(N_local < 1):
            raise ValueError("N_local must be a positive scalar or three-vector")
        if sq_ke_cutoff is not None:
            cutoff = float(sq_ke_cutoff)
            if not np.isfinite(cutoff) or cutoff <= 0:
                raise ValueError("sq_ke_cutoff must be finite and positive")
            mesh_cutoff = cutoff_to_mesh(self.cell.lattice_vectors(), cutoff)
            mesh_energy = mesh_to_cutoff(self.cell.lattice_vectors(), N_local)
            if np.min(mesh_energy) <= cutoff:
                N_local = np.asarray(mesh_cutoff, dtype=int)
        self.N_local = N_local
        self.result = None

        if self.sq_ke_cutoff_switch_radius is not None:
            if self.pair_density_eval_grid == "becke":
                raise NotImplementedError(
                    "adaptive sq_ke_cutoff is not implemented with Becke quadrature"
                )
            if sq_ke_cutoff is None or not np.isscalar(sq_ke_cutoff):
                raise ValueError(
                    "adaptive sq_ke_cutoff requires a scalar sq_ke_cutoff"
                )
            self.sq_ke_cutoff_switch_radius = float(
                self.sq_ke_cutoff_switch_radius
            )
            if self.sq_ke_cutoff_switch_radius < 0:
                raise ValueError("sq_ke_cutoff_switch_radius must be non-negative")
            if (
                not np.isfinite(self.outer_sq_ke_cutoff_scale)
                or self.outer_sq_ke_cutoff_scale <= 0
            ):
                raise ValueError("outer_sq_ke_cutoff_scale must be positive")

    def _build_grids(self):
        """Build the one physical q and the minimum-image plus/minus meshes."""
        nks = np.asarray(
            get_monkhorst_pack_size(self.cell, self.kmf.kpts), dtype=int
        )
        shift = np.asarray(self.options.relative_shift)
        qprime = np.asarray(self.cell.get_abs_kpts(shift / nks))
        occupied = minimum_image(self.cell, self.kmf.kpts)
        plus = minimum_image(self.cell, occupied + qprime)
        minus = minimum_image(self.cell, occupied - qprime)
        # A shift is self-inverse only when 2*s is exactly an integer mesh
        # shift. Exact comparison keeps valid tiny nonzero shifts distinct.
        self_inverse = np.all(2 * shift == np.rint(2 * shift))
        minus_from_plus = None
        if self_inverse:
            minus_from_plus = _map_kpts(
                self.cell, minus, plus, "self-inverse k-q"
            )
        return _SmallQGrids(
            qprime=qprime,
            occupied=np.asarray(occupied),
            plus=np.asarray(plus),
            minus=np.asarray(minus),
            minus_from_plus=minus_from_plus,
        )

    def _make_band_mean_field(self, band_kpts):
        """Build a fresh sTC exchange backend and its ordinary-GDF J partner."""
        try:
            from pyscf.pbc.df import rsdf_stc
        except ImportError as err:
            raise ImportError(
                "MP2 small-q bands require PySCF's rsdf_stc extension"
            ) from err

        source_df = self.kmf.with_df
        cutoff_name = "vcut_" + self.options.cutoff
        stc_df = rsdf_stc.RSGDF_STC(
            self.cell,
            self.kmf.kpts,
            eta=self.options.eta,
            exxdiv=cutoff_name,
            rc_type=self.options.cutoff,
        )
        j_df = None
        try:
            j_df = df.GDF(self.cell, self.kmf.kpts)
            _copy_df_settings(source_df, stc_df, self.kmf)
            _copy_df_settings(source_df, j_df, self.kmf)
            stc_df.with_df_j = j_df
            j_df.build(kpts_band=band_kpts)
            stc_df.build(kpts_band=band_kpts)
            # Do not call rsdf_stc.density_fit: its implementation resets a
            # copied mean field, which can reset DF/RSJK state on the source.
            band_mf = self.kmf.copy()
            band_mf.with_df = stc_df
            band_mf.exxdiv = None
            band_mf.rsjk = None
            band_mf._eri = None
            return band_mf, stc_df, j_df
        except Exception:
            if j_df is not None:
                _close_temporary_gdf(j_df)
            _close_temporary_gdf(stc_df)
            raise

    def _get_shifted_bands(self, grids):
        """Evaluate shifted virtual bands using the original SCF density."""
        if grids.minus_from_plus is None:
            band_kpts = np.concatenate((grids.plus, grids.minus), axis=0)
        else:
            band_kpts = grids.plus

        band_mf, stc_df, j_df = self._make_band_mean_field(band_kpts)
        try:
            dm_kpts = self.kmf.make_rdm1()
            energies, coeffs = band_mf.get_bands(
                band_kpts,
                dm_kpts=dm_kpts,
                kpts=self.kmf.kpts,
            )
            energies = list(np.asarray(energies))
            coeffs = list(np.asarray(coeffs))
            nkpts = len(grids.occupied)
            plus_energy = energies[:nkpts]
            plus_coeff = coeffs[:nkpts]
            if grids.minus_from_plus is None:
                minus_energy = energies[nkpts:]
                minus_coeff = coeffs[nkpts:]
            else:
                # Reorder raw bands first so per-k-point KMP2 frozen masks
                # are applied to the corresponding k-q orbital blocks.
                indices = grids.minus_from_plus
                minus_energy = [plus_energy[i] for i in indices]
                minus_coeff = [plus_coeff[i] for i in indices]
            return (plus_energy, plus_coeff), (minus_energy, minus_coeff)
        finally:
            _close_temporary_gdf(j_df)
            _close_temporary_gdf(stc_df)

    def _pair_density_grid(self, mesh=None):
        if self.pair_density_eval_grid == "becke":
            pair_grid = dft.gen_grid.BeckeGrids(self.cell)
            pair_grid.level = self.pair_density_becke_grid_level
            pair_grid.build(with_non0tab=False)
            return np.asarray(pair_grid.coords), np.asarray(pair_grid.weights)

        mesh = np.asarray(self.N_local if mesh is None else mesh, dtype=int)
        lattice = self.cell.lattice_vectors()
        indices = np.stack(
            np.meshgrid(
                np.arange(mesh[0]),
                np.arange(mesh[1]),
                np.arange(mesh[2]),
                indexing="ij",
            ),
            axis=-1,
        ).reshape(-1, 3)
        coords = indices @ (lattice / mesh[:, None])
        dvol = abs(np.linalg.det(lattice)) / np.prod(mesh)
        return coords, np.full(len(coords), dvol)

    def _build_pair_densities(self, grids, mo_coeff_occ, mo_coeff_plus,
                              mo_coeff_minus):
        """Integrate rho_ia(q) and rho_jb(-q) in the cell volume."""
        mesh = self.N_local
        if self.sq_ke_cutoff_switch_radius is not None:
            if np.linalg.norm(grids.qprime) > self.sq_ke_cutoff_switch_radius:
                outer_cutoff = (
                    float(self.sq_ke_cutoff) * self.outer_sq_ke_cutoff_scale
                )
                mesh = cutoff_to_mesh(
                    self.cell.lattice_vectors(), outer_cutoff
                )

        coords, weights = self._pair_density_grid(mesh=mesh)
        nmo = self.kmp.nmo
        nocc = self.kmp.nocc
        u_occ = build_uKpts(
            self.kmf, grids.occupied, mo_coeff_occ,
            rptGrid3D=coords, nbands=nmo,
        )
        u_plus = build_uKpts(
            self.kmf, grids.plus, mo_coeff_plus,
            rptGrid3D=coords, nbands=nmo,
        )
        u_minus = build_uKpts(
            self.kmf, grids.minus, mo_coeff_minus,
            rptGrid3D=coords, nbands=nmo,
        )

        gdiff_plus = grids.occupied + grids.qprime - grids.plus
        gdiff_minus = grids.occupied - grids.qprime - grids.minus
        phase_plus = np.exp(-1j * (coords @ gdiff_plus.T)).T * weights
        phase_minus = np.exp(-1j * (coords @ gdiff_minus.T)).T * weights
        rho_ia = np.empty(
            (len(grids.occupied), nocc, nmo - nocc), dtype=np.complex128
        )
        rho_jb = np.empty_like(rho_ia)
        for ki in range(len(grids.plus)):
            ka = ki
            rho_ia[ki] = (
                (u_occ[ki, :nocc].conj() * phase_plus[ki][None, :])
                @ u_plus[ka, nocc:].T
            )
        for kj in range(len(grids.minus)):
            kb = kj
            rho_jb[kj] = (
                (u_occ[kj, :nocc].conj() * phase_minus[kj][None, :])
                @ u_minus[kb, nocc:].T
            )
        return rho_ia, rho_jb

    def _make_correlation_gdf(self, occupied_kpts, band_kpts):
        """Build full ordinary GDF factors for occupied and shifted points."""
        correlation_df = df.GDF(self.cell, occupied_kpts)
        try:
            _copy_df_settings(self.kmp._scf.with_df, correlation_df, self.kmf)
            correlation_df.build(kpts_band=band_kpts)
        except Exception:
            _close_temporary_gdf(correlation_df)
            raise
        return correlation_df

    def _contract(self, grids, mo_coeff_occ, mo_energy_occ,
                  mo_coeff_plus, mo_energy_plus, mo_coeff_minus,
                  mo_energy_minus, rho_ia, rho_jb, correlation_df):
        """Contract one occupied pair at a time using exact denominators.

        ``rho_ia`` and ``rho_jb`` have shape ``(Nk, nocc, nvir)`` and are
        cell-volume integrals at ``q`` and ``-q``. For each ``ki,kj`` block,
        ``R = rho_ia * rho_jb / (Nk * V)`` and ``I = ERI / Nk``; the direct
        and fourth-order reductions are ``2 Re sum(R * conj(I / D)) / Nk``
        and ``2 sum(abs(R)**2 / D) / Nk``.
        """
        nocc = self.kmp.nocc
        nkpts = len(grids.occupied)
        volume = abs(np.linalg.det(self.cell.lattice_vectors()))
        occ_active, vir_active = kmp2.padding_k_idx(self.kmp, kind="split")
        direct = 0.0
        q4 = 0.0

        for ki in range(len(grids.plus)):
            ka = ki
            oi = np.asarray(occ_active[ki], dtype=int)
            va = np.asarray(vir_active[ka], dtype=int)
            for kj in range(len(grids.minus)):
                kb = kj
                oj = np.asarray(occ_active[kj], dtype=int)
                vb = np.asarray(vir_active[kb], dtype=int)
                if not (len(oi) and len(oj) and len(va) and len(vb)):
                    continue
                coeffs = (
                    np.asarray(mo_coeff_occ[ki])[:, oi],
                    np.asarray(mo_coeff_plus[ka])[:, nocc + va],
                    np.asarray(mo_coeff_occ[kj])[:, oj],
                    np.asarray(mo_coeff_minus[kb])[:, nocc + vb],
                )
                kpts = (
                    grids.occupied[ki], grids.plus[ka],
                    grids.occupied[kj], grids.minus[kb],
                )
                eri = correlation_df.ao2mo(coeffs, kpts, compact=False)
                eri = np.asarray(eri).reshape(
                    len(oi), len(va), len(oj), len(vb)
                ).transpose(0, 2, 1, 3) / nkpts
                denominator = (
                    np.asarray(mo_energy_occ[ki])[oi, None, None, None]
                    + np.asarray(mo_energy_occ[kj])[None, oj, None, None]
                    - np.asarray(mo_energy_plus[ka])[None, None, nocc + va, None]
                    - np.asarray(mo_energy_minus[kb])[None, None, None, nocc + vb]
                )
                r_ia = rho_ia[ki][np.ix_(oi, va)]
                r_jb = rho_jb[kj][np.ix_(oj, vb)]
                pair_density = (
                    r_ia[:, None, :, None] * r_jb[None, :, None, :]
                ) / (nkpts * volume)
                direct += (
                    2 * np.real(np.sum(pair_density * np.conj(eri / denominator)))
                    / nkpts
                )
                q4 += (
                    2 * np.sum(np.abs(pair_density) ** 2 / denominator).real
                    / nkpts
                )
        return float(direct), float(q4)

    def kernel(self):
        """Build the q-point result and release all temporary GDF files."""
        log = logger.new_logger(self.kmp, self.verbose)
        grids = self._build_grids()
        (energy_plus, coeff_plus), (energy_minus, coeff_minus) = (
            self._get_shifted_bands(grids)
        )

        # Keep the exact KMP2 occupied data, including user-supplied energy
        # shifts, while applying KMP2's frozen and padding conventions.
        coeff_occ, energy_occ = kmp2._add_padding(
            self.kmp, self.kmp.mo_coeff, self.kmp.mo_energy
        )
        coeff_plus, energy_plus = kmp2._add_padding(
            self.kmp, coeff_plus, energy_plus
        )
        coeff_minus, energy_minus = kmp2._add_padding(
            self.kmp, coeff_minus, energy_minus
        )
        rho_ia, rho_jb = self._build_pair_densities(
            grids, coeff_occ, coeff_plus, coeff_minus
        )

        band_kpts = np.concatenate((grids.plus, grids.minus), axis=0)
        correlation_df = self._make_correlation_gdf(grids.occupied, band_kpts)
        try:
            sq_direct, sq_q4 = self._contract(
                grids,
                coeff_occ,
                energy_occ,
                coeff_plus,
                energy_plus,
                coeff_minus,
                energy_minus,
                rho_ia,
                rho_jb,
                correlation_df,
            )
        finally:
            _close_temporary_gdf(correlation_df)

        self.result = MP2SmallQResult(
            qprime=grids.qprime.copy(),
            sq_direct=sq_direct,
            sq_q4=sq_q4,
            relative_shift=self.options.relative_shift,
            eta=self.options.eta,
            cutoff=self.options.cutoff,
        )
        log.note(
            "MP2 small-q: q=%s, direct=% .9e, q4=% .9e (sTC eta=%g, cutoff=%s)",
            self.result.qprime,
            self.result.sq_direct,
            self.result.sq_q4,
            self.result.eta,
            self.result.cutoff,
        )
        return self.result
