from __future__ import annotations

from dataclasses import dataclass
import numbers

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import KDTree

from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.pbc.cc import kccsd_rhf
from pyscf.pbc.dft import gen_grid as pbc_gen_grid
from pyscf.pbc.mp import kmp2
from pyscf.pbc.mp.kmp2 import padding_k_idx

from fsec.staggered_mesh.cc.krccd import KRCCD

"""Iterative singularity subtraction for restricted periodic CCD.

Let ``F_n(k,Q)`` be the normalized diagonal transition-density factor linking
orbital ``n,k`` to ``n,k+q``, including the reciprocal wrap in ``Q = q+G``.
The corresponding full transition density is ``rho[p,s](k,Q)``.  PySCF
stores doubles as ``t2[ki,kj,ka,i,j,a,b]`` and determines
``kb = kconserv[ki,ka,kj]``.

The six fitted curves are density-only aggregates.  With constraint (1)
enabled, each channel uses the existing diagonal factor product.  With it
disabled, the two factors are contracted over the internal orbital indices as
``ki,jl->ij``, ``ac,db->ab``, ``ac,ik->ia``, ``bc,jk->jb``, ``ac,jk->ja``,
and ``bc,ik->ib``.  Every active external orbital and k-point index is then
summed.  Constraint (2) fixes the T2 momentum at its q=0 block, so T2 is
factored out of fitting entirely.  Each curve is normalized by the exact
number of physical active entries at q=0 and its origin is set exactly to one.

The six fitted corrections are reused throughout CCD.  The correction added
to the current doubles residual is
``(xi1 + xi2 - xi3 - xi4 - xi5 - xi6) * t2``.
"""


@dataclass(frozen=True)
class CCDSSOptions:
    """Configuration for :class:`KRCCD_SS`.

    ``line_points`` is the number of positive samples on each reciprocal
    direction.  Samples are separated by ``b_i / n_i``, where ``n_i`` is the
    Monkhorst--Pack mesh size.  The Gaussian is
    ``exp(-|Q|**2 / (2*sigma**2))`` and its coefficient is fixed to one.
    ``use_constraint_1`` selects diagonal pair factors or full transition
    densities.  Constraint (2) relaxation is not currently supported.
    """

    use_constraint_2: bool = True
    use_constraint_1: bool = True
    line_points: int = 3
    pair_density_becke_grid_level: int = 0
    fit_with_coul: bool = True
    fixed_sigma: float | None = None
    amplitude_fit_tol: float = 1e-12

    def __post_init__(self):
        if isinstance(self.line_points, bool) or not isinstance(
            self.line_points, numbers.Integral
        ) or self.line_points < 1:
            raise ValueError("line_points must be an integer >= 1")
        if (
            isinstance(self.pair_density_becke_grid_level, bool)
            or not isinstance(self.pair_density_becke_grid_level, numbers.Integral)
            or self.pair_density_becke_grid_level < 0
        ):
            raise ValueError(
                "pair_density_becke_grid_level must be an integer >= 0"
            )
        if self.fixed_sigma is not None:
            try:
                sigma = float(self.fixed_sigma)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "fixed_sigma must be None or a non-negative number"
                ) from exc
            if np.isnan(sigma) or sigma < 0:
                raise ValueError("fixed_sigma must be None or a non-negative number")
            object.__setattr__(self, "fixed_sigma", sigma)
        try:
            tol = float(self.amplitude_fit_tol)
        except (TypeError, ValueError) as exc:
            raise ValueError("amplitude_fit_tol must be a finite positive number") from exc
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError("amplitude_fit_tol must be a finite positive number")
        object.__setattr__(self, "amplitude_fit_tol", tol)
        object.__setattr__(self, "use_constraint_1", bool(self.use_constraint_1))
        object.__setattr__(self, "use_constraint_2", bool(self.use_constraint_2))
        object.__setattr__(self, "fit_with_coul", bool(self.fit_with_coul))


def _merge_options(options, **values):
    defaults = CCDSSOptions()
    if options is None:
        return CCDSSOptions(**values)
    if not isinstance(options, CCDSSOptions):
        raise TypeError("options must be a CCDSSOptions instance or None")
    conflicts = [
        key for key, value in values.items() if value != getattr(defaults, key)
    ]
    if conflicts:
        raise ValueError(
            "options cannot be combined with non-default keyword overrides: "
            + ", ".join(conflicts)
        )
    return options


def _normalize_occupied_orbital_shift(value, nkpts, nocc):
    """Return a real ``(nkpts, nocc)`` shift array or ``None``."""
    if value is None:
        return None

    shift = np.asarray(value)
    if not np.issubdtype(shift.dtype, np.number) or np.issubdtype(
        shift.dtype, np.bool_
    ):
        raise ValueError("occupied_orbital_shift must contain real numbers")
    if np.iscomplexobj(shift):
        raise ValueError("occupied_orbital_shift must contain real numbers")

    expected = (nkpts, nocc)
    if shift.ndim == 0:
        shift = np.full(expected, float(shift), dtype=float)
    elif shift.shape == expected:
        shift = np.array(shift, dtype=float, copy=True)
    else:
        raise ValueError(
            "occupied_orbital_shift must be a scalar or have shape "
            f"{expected}; received {shift.shape}"
        )
    if not np.all(np.isfinite(shift)):
        raise ValueError("occupied_orbital_shift values must be finite")
    return shift


def _line_samples(cell, kpts, line_points):
    """Return origin and positive ``b_i/n_i`` line samples."""
    nks = np.asarray(tools.get_monkhorst_pack_size(cell, kpts), dtype=int)
    if nks.shape != (3,) or np.any(nks < 1):
        raise ValueError("could not determine a three-dimensional k-point mesh")
    steps = np.asarray(cell.reciprocal_vectors()) / nks[:, None]
    points = [np.zeros(3)]
    metadata = []
    for direction, step in enumerate(steps):
        indices = []
        for multiple in range(1, line_points + 1):
            indices.append(len(points))
            points.append(multiple * step)
        metadata.append((direction, step.copy(), np.asarray(indices, dtype=int)))
    return np.asarray(points), tuple(metadata), nks


def _fit_unit_gaussian(q_vectors, values, fit_with_coul=True):
    """Fit ``exp(-Q^2/(2 sigma^2))`` with its coefficient fixed to one."""
    q_vectors = np.asarray(q_vectors, dtype=float)
    values = np.asarray(values)
    if q_vectors.ndim != 2 or q_vectors.shape[1] != 3:
        raise ValueError("q_vectors must have shape (n, 3)")
    if values.shape != (len(q_vectors),):
        raise ValueError("Gaussian fit values must have shape (n_samples,)")
    if not np.all(np.isfinite(q_vectors)) or not np.all(np.isfinite(values)):
        raise ValueError("Gaussian fitting data must be finite")
    norms = np.linalg.norm(q_vectors, axis=1)
    active = norms > 1e-14
    if not np.any(active):
        raise ValueError("Gaussian fitting requires at least one nonzero sample")
    q2 = norms[active] ** 2
    target = values[active]
    weights = 1.0 / q2 if fit_with_coul else np.ones_like(q2)
    scale = max(float(np.min(norms[active])), 1e-12)

    def residual(log_sigma):
        sigma = np.exp(log_sigma[0])
        delta = (np.exp(-q2 / (2.0 * sigma * sigma)) - target) * np.sqrt(weights)
        return np.r_[delta.real, delta.imag]

    result = least_squares(
        residual,
        np.asarray([np.log(scale)]),
        bounds=(np.log(scale * 1e-8), np.log(scale * 1e8)),
    )
    sigma = float(np.exp(result.x[0]))
    if not result.success or not np.isfinite(sigma):
        raise ValueError("Gaussian fitting produced a non-finite sigma")
    return sigma


class KRCCD_SS(KRCCD):
    """Restricted k-point CCD with six-channel singularity subtraction.

    The active path evaluates six scalar, density-only ``S_tilde_n(Q)``
    aggregates.  Constraint (1) selects diagonal transition factors or full
    transition-density matrices.  Constraint (2) keeps the T2 momentum at its
    q=0 value; relaxing constraint (2) is unsupported.
    """

    _keys = KRCCD._keys.union(
        {
            "options",
            "occupied_orbital_shift",
            "ss_fit_count",
            "ss_prepare_count",
            "last_ss_residual_norm",
            "ss_sigmas",
            "ss_xi",
        }
    )

    def __init__(
        self,
        mf,
        frozen=None,
        mo_coeff=None,
        mo_occ=None,
        options=None,
        use_constraint_2=True,
        line_points=3,
        pair_density_becke_grid_level=0,
        fit_with_coul=True,
        fixed_sigma=None,
        amplitude_fit_tol=1e-12,
        occupied_orbital_shift=None,
        use_constraint_1=True,
    ):
        self.options = _merge_options(
            options,
            use_constraint_1=use_constraint_1,
            use_constraint_2=use_constraint_2,
            line_points=line_points,
            pair_density_becke_grid_level=pair_density_becke_grid_level,
            fit_with_coul=fit_with_coul,
            fixed_sigma=fixed_sigma,
            amplitude_fit_tol=amplitude_fit_tol,
        )
        self._validate_constraints()
        super().__init__(
            mf,
            frozen=frozen,
            mo_coeff=mo_coeff,
            mo_occ=mo_occ,
            madelung_orbital=occupied_orbital_shift is None,
            madelung_eri=False,
        )
        self.occupied_orbital_shift = _normalize_occupied_orbital_shift(
            occupied_orbital_shift, self.nkpts, self.nocc
        )
        self.ss_fit_count = 0
        self.ss_prepare_count = 0
        self.last_ss_residual_norm = 0.0
        self.ss_sigmas = None
        self.ss_xi = None
        self._ss_pair_factors = None
        self._ss_pair_densities = None
        self._ss_pair_orbitals = None
        self._ss_q_vectors = None
        self._ss_plus = None
        self._ss_minus = None
        self._ss_active_masks = None
        self._ss_active_count = None

    @property
    def use_constraint_1(self):
        return self.options.use_constraint_1

    @property
    def use_constraint_2(self):
        return self.options.use_constraint_2

    def _validate_constraints(self):
        """Reject the unsupported constraint-(2)-relaxed contraction."""
        if not self.options.use_constraint_2:
            raise NotImplementedError(
                "constraint (2) relaxation is unsupported; "
                "use_constraint_2 must be True"
            )

    def dump_flags(self, verbose=None):
        result = super().dump_flags(verbose)
        log = logger.new_logger(self, verbose)
        log.info("CCD SS constraint (1) = %s", self.options.use_constraint_1)
        log.info("CCD SS constraint (2) = %s", self.options.use_constraint_2)
        log.info("CCD SS positive line samples = %d", self.options.line_points)
        log.info(
            "CCD SS Becke grid level = %d",
            self.options.pair_density_becke_grid_level,
        )
        log.info("CCD SS fixed sigma = %s", self.options.fixed_sigma)
        if self.occupied_orbital_shift is None:
            log.info("CCD SS occupied-orbital shift = Madelung default")
        else:
            log.info(
                "CCD SS occupied-orbital shift shape = %s, range = [%g, %g]",
                self.occupied_orbital_shift.shape,
                np.min(self.occupied_orbital_shift),
                np.max(self.occupied_orbital_shift),
            )
            log.info(
                "CCD SS occupied-orbital shift replaces Madelung and "
                "keep_exxdiv corrections"
            )
        return result

    def ao2mo(self, mo_coeff=None):
        """Build ERIs and apply any custom occupied-orbital shifts.

        A custom shift is defined relative to the uncorrected one-body
        baseline.  It therefore replaces both KRCCD's explicit Madelung shift
        and an exchange-divergence correction requested through
        ``keep_exxdiv``.
        """
        if self.occupied_orbital_shift is None:
            return super().ao2mo(mo_coeff)

        keep_exxdiv = self.keep_exxdiv
        madelung_orbital = self.madelung_orbital
        try:
            self.keep_exxdiv = False
            self.madelung_orbital = False
            eris = super().ao2mo(mo_coeff)
        finally:
            self.keep_exxdiv = keep_exxdiv
            self.madelung_orbital = madelung_orbital

        fock = np.array(eris.fock, copy=True)
        mo_energy = [np.asarray(energy).copy() for energy in eris.mo_energy]
        nonzero_opadding, _ = padding_k_idx(self, kind="split")
        for k, occupied in enumerate(nonzero_opadding):
            shift = self.occupied_orbital_shift[k, occupied]
            fock[k][occupied, occupied] += shift
            mo_energy[k][occupied] += shift

        eris.fock = fock
        eris.mo_energy = mo_energy
        return eris

    def init_amps(self, eris):
        emp2, t1, t2 = super().init_amps(eris)
        t1 = np.zeros_like(t1)
        # The density-only curves do not depend on the initial amplitudes.  Do
        # the one-time preparation here so the first CCD update can reuse the
        # cached six widths and corrections.
        self._prepare_ss(t2)
        return emp2, t1, t2

    def _validate_t2(self, t2):
        expected = (
            self.nkpts,
            self.nkpts,
            self.nkpts,
            self.nocc,
            self.nocc,
            self.nmo - self.nocc,
            self.nmo - self.nocc,
        )
        if np.shape(t2) != expected:
            raise ValueError(f"t2 must have shape {expected}; received {np.shape(t2)}")

    def _build_k_shift_maps(self, q_vectors):
        cell = self._scf.cell
        scaled = np.remainder(cell.get_scaled_kpts(self.kpts), 1.0)
        # Floating-point remainders can round a value just below zero to
        # exactly 1.0.  Canonicalize that boundary before constructing the
        # periodic tree, whose input must lie in [0, 1).
        scaled[scaled >= 1.0] = 0.0
        tree = KDTree(scaled, boxsize=1.0)
        plus = np.empty((len(q_vectors), self.nkpts), dtype=int)
        minus = np.empty_like(plus)
        for iq, q in enumerate(q_vectors):
            qscaled = cell.get_scaled_kpts(q)
            for sign, output in ((1.0, plus), (-1.0, minus)):
                targets = np.remainder(scaled + sign * qscaled, 1.0)
                targets[targets >= 1.0] = 0.0
                distance, indices = tree.query(targets)
                if np.any(distance > 1e-7):
                    raise ValueError("line sample does not map onto the k-point mesh")
                output[iq] = indices
        return plus, minus

    def _prepare_sampling_maps(self):
        """Cache the sampled q-vectors and their discrete momentum maps."""
        if (
            getattr(self, "_ss_q_vectors", None) is not None
            and getattr(self, "_ss_plus", None) is not None
            and getattr(self, "_ss_minus", None) is not None
        ):
            return
        q_vectors, _, _ = _line_samples(
            self._scf.cell, self.kpts, self.options.line_points
        )
        plus, minus = self._build_k_shift_maps(q_vectors)
        self._ss_q_vectors = q_vectors
        self._ss_plus = plus
        self._ss_minus = minus

    def _build_pair_orbitals(self):
        """Build periodic orbital parts and their Becke-grid norms."""
        cached = getattr(self, "_ss_pair_orbitals", None)
        if cached is not None:
            return cached
        self._prepare_sampling_maps()

        cell = self._scf.cell
        pair_grid = pbc_gen_grid.BeckeGrids(cell)
        pair_grid.level = self.options.pair_density_becke_grid_level
        pair_grid.build(with_non0tab=False)
        coords = np.asarray(pair_grid.coords)
        weights = np.asarray(pair_grid.weights)
        coeff = kmp2.padded_mo_coeff(self, self.mo_coeff)

        # Periodic parts u_nk(r), following the sibling ExxStructureFactor
        # implementation.  Their weighted overlaps are transition densities.
        ao = cell.pbc_eval_gto("GTOval_sph", coords=coords, kpts=self.kpts)
        scaled = cell.get_scaled_kpts(self.kpts)
        scaled -= np.floor(scaled)
        scaled[scaled > 0.5 - 1e-8] -= 1.0
        wrapped_kpts = cell.get_abs_kpts(scaled)
        phase = np.exp(-1j * (coords @ wrapped_kpts.T)).T
        u = (ao @ coeff).transpose(0, 2, 1) * phase[:, None, :]
        norms = lib.einsum("knr,r,knr->kn", u.conj(), weights, u).real
        cached = coords, weights, u, norms, wrapped_kpts
        self._ss_pair_orbitals = cached
        return cached

    def _build_pair_factors(self):
        """Build the diagonal, constraint-(1) transition-density factors."""
        cached = getattr(self, "_ss_pair_factors", None)
        if cached is not None:
            return cached
        self._prepare_sampling_maps()
        coords, weights, u, norms, wrapped_kpts = self._build_pair_orbitals()
        factors = np.ones(
            (len(self._ss_q_vectors), self.nkpts, self.nmo), dtype=np.complex128
        )
        for iq, q in enumerate(self._ss_q_vectors[1:], start=1):
            target = self._ss_plus[iq]
            unwrapped = self.kpts + q
            wrapped = wrapped_kpts[target]
            gdiff = unwrapped - wrapped
            wrap_phase = np.exp(-1j * (coords @ gdiff.T)).T
            u_target = u[target] * wrap_phase[:, None, :]
            overlap = lib.einsum("knr,r,knr->kn", u.conj(), weights, u_target)
            valid = (np.abs(norms) > 1e-14) & (
                np.abs(norms[target]) > 1e-14
            )
            factors[iq, valid] = overlap[valid] / np.sqrt(
                norms[valid] * norms[target][valid]
            )
            # Preserve the historical padded-orbital safeguard.  Those
            # elements are removed from CCD denominators later.
            factors[iq, ~valid] = 1.0
        factors[0] = 1.0
        self._ss_pair_factors = factors
        return factors

    def _build_pair_densities(self):
        """Build normalized transition-density matrices for relaxed constraint 1."""
        cached = getattr(self, "_ss_pair_densities", None)
        if cached is not None:
            return cached
        self._prepare_sampling_maps()
        coords, weights, u, norms, wrapped_kpts = self._build_pair_orbitals()
        densities = np.zeros(
            (len(self._ss_q_vectors), self.nkpts, self.nmo, self.nmo),
            dtype=np.complex128,
        )
        densities[0] = np.eye(self.nmo, dtype=np.complex128)

        for iq, q in enumerate(self._ss_q_vectors[1:], start=1):
            target = self._ss_plus[iq]
            unwrapped = self.kpts + q
            wrapped = wrapped_kpts[target]
            gdiff = unwrapped - wrapped
            wrap_phase = np.exp(-1j * (coords @ gdiff.T)).T
            u_target = u[target] * wrap_phase[:, None, :]
            overlap = lib.einsum(
                "kpr,r,ksr->kps", u.conj(), weights, u_target
            )
            denominator = np.sqrt(
                np.maximum(
                    norms[:, :, None] * norms[target][:, None, :], 0.0
                )
            )
            valid = denominator > 1e-14
            np.divide(overlap, denominator, out=densities[iq], where=valid)

            # A padded state has no grid norm.  Keep its diagonal finite and
            # compatible with the legacy diagonal factors while leaving
            # off-diagonal transitions zero.  The origin is set exactly above.
            diagonal = np.diag_indices(self.nmo)
            for k in range(self.nkpts):
                invalid_diagonal = ~valid[k].diagonal()
                densities[
                    iq,
                    k,
                    diagonal[0][invalid_diagonal],
                    diagonal[1][invalid_diagonal],
                ] = 1.0

        self._ss_pair_densities = densities
        self._ss_pair_factors = np.diagonal(densities, axis1=2, axis2=3).copy()
        self._ss_pair_factors[0] = 1.0
        return densities

    def _build_active_masks(self):
        """Return boolean occupied and virtual masks for every k point."""
        cached = getattr(self, "_ss_active_masks", None)
        if cached is not None:
            if getattr(self, "_ss_active_count", None) is None:
                self._ss_active_count = self._count_active_entries(*cached)
            return cached

        nonzero_occupied, nonzero_virtual = padding_k_idx(self, kind="split")
        if len(nonzero_occupied) != self.nkpts or len(nonzero_virtual) != self.nkpts:
            raise ValueError("padding masks do not match the k-point mesh")

        occupied = np.zeros((self.nkpts, self.nocc), dtype=bool)
        virtual = np.zeros(
            (self.nkpts, self.nmo - self.nocc), dtype=bool
        )
        for k, indices in enumerate(nonzero_occupied):
            indices = np.asarray(indices, dtype=int)
            if np.any(indices < 0) or np.any(indices >= self.nocc):
                raise ValueError("occupied padding mask contains an invalid index")
            occupied[k, indices] = True
        for k, indices in enumerate(nonzero_virtual):
            indices = np.asarray(indices, dtype=int)
            if np.any(indices < 0) or np.any(indices >= self.nmo - self.nocc):
                raise ValueError("virtual padding mask contains an invalid index")
            virtual[k, indices] = True

        self._ss_active_masks = occupied, virtual
        self._ss_active_count = self._count_active_entries(occupied, virtual)
        return self._ss_active_masks

    def _count_active_entries(self, occupied, virtual):
        """Count physical doubles entries over all momentum blocks."""
        occupied_count = occupied.sum(axis=1).astype(int)
        virtual_count = virtual.sum(axis=1).astype(int)
        kb_map = np.asarray(self.khelper.kconserv)
        return int(
            sum(
                int(occupied_count[ki])
                * int(occupied_count[kj])
                * int(virtual_count[ka])
                * int(virtual_count[kb_map[ki, ka, kj]])
                for ki in range(self.nkpts)
                for kj in range(self.nkpts)
                for ka in range(self.nkpts)
            )
        )

    @staticmethod
    def _masked_density(density, row_mask, column_mask):
        """Mask both endpoints of a transition density without T2 storage."""
        return density * row_mask[:, None] * column_mask[None, :]

    @staticmethod
    def _masked_factor(factors, source, target, active):
        """Mask a diagonal factor at both physical ends of its transition."""
        valid = active[source] & active[target]
        return factors[source] * valid

    def _diagonal_channel_partials(
        self, factors, q_index, ki, kj, ka, kb, occupied, virtual
    ):
        """Return the six diagonal density-only partial matrices."""
        occupied_factors = factors[:, : self.nocc]
        virtual_factors = factors[:, self.nocc :]
        plus = self._ss_plus[q_index]
        minus = self._ss_minus[q_index]
        occupied_i = self._masked_factor(
            occupied_factors, minus[ki], ki, occupied
        )
        occupied_j = self._masked_factor(
            occupied_factors, kj, plus[kj], occupied
        )
        occupied_i_at_ki = self._masked_factor(
            occupied_factors, ki, plus[ki], occupied
        )
        virtual_a = self._masked_factor(
            virtual_factors, ka, plus[ka], virtual
        )
        virtual_b = self._masked_factor(
            virtual_factors, kb, plus[kb], virtual
        )
        virtual_b_at_minus = self._masked_factor(
            virtual_factors, minus[kb], kb, virtual
        )
        return (
            lib.einsum("i,j->ij", occupied_i, occupied_j.conj()),
            lib.einsum("a,b->ab", virtual_a, virtual_b_at_minus.conj()),
            lib.einsum("a,i->ia", virtual_a, occupied_i_at_ki.conj()),
            lib.einsum("b,j->jb", virtual_b, occupied_j.conj()),
            lib.einsum("a,j->ja", virtual_a, occupied_j.conj()),
            lib.einsum("b,i->ib", virtual_b, occupied_i_at_ki.conj()),
        )

    def _full_density_channel_partials(
        self, densities, q_index, ki, kj, ka, kb, occupied, virtual
    ):
        """Return the six full-density-only partial matrices.

        A density row is masked at its source momentum and its column at the
        corresponding plus-momentum.  L1 and L2 use the inverse minus maps for
        the factors whose external index is at ``ki`` or ``kb``.
        """
        plus = self._ss_plus[q_index]
        minus = self._ss_minus[q_index]
        density = densities[q_index]
        occupied_density = density[:, : self.nocc, : self.nocc]
        virtual_density = density[:, self.nocc :, self.nocc :]
        rho_ki = self._masked_density(
            occupied_density[minus[ki]], occupied[minus[ki]], occupied[ki]
        )
        rho_j = self._masked_density(
            occupied_density[kj], occupied[kj], occupied[plus[kj]]
        )
        rho_i = self._masked_density(
            occupied_density[ki], occupied[ki], occupied[plus[ki]]
        )
        rho_a = self._masked_density(
            virtual_density[ka], virtual[ka], virtual[plus[ka]]
        )
        rho_b_minus = self._masked_density(
            virtual_density[minus[kb]], virtual[minus[kb]], virtual[kb]
        )
        rho_b = self._masked_density(
            virtual_density[kb], virtual[kb], virtual[plus[kb]]
        )
        return (
            lib.einsum("ki,jl->ij", rho_ki, rho_j.conj()),
            lib.einsum("ac,db->ab", rho_a, rho_b_minus.conj()),
            lib.einsum("ac,ik->ia", rho_a, rho_i.conj()),
            lib.einsum("bc,jk->jb", rho_b, rho_j.conj()),
            lib.einsum("ac,jk->ja", rho_a, rho_j.conj()),
            lib.einsum("bc,ik->ib", rho_b, rho_i.conj()),
        )

    def _contract_aggregate_structure_factors(self, t2=None):
        """Accumulate six scalar, density-only structure-factor curves.

        ``t2`` is accepted for compatibility with the old private helper and
        is deliberately ignored after shape validation.  The common driver
        handles active-index counting, momentum traversal, constraint-(1)
        dispatch, and scalar accumulation for both modes.
        """
        if t2 is not None:
            self._validate_t2(t2)
        self._prepare_sampling_maps()
        occupied, virtual = self._build_active_masks()
        occupied_count = occupied.sum(axis=1).astype(int)
        virtual_count = virtual.sum(axis=1).astype(int)
        nsamples = len(self._ss_q_vectors)
        raw = np.zeros((6, nsamples), dtype=np.complex128)
        kb_map = np.asarray(self.khelper.kconserv)
        active_count = 0

        factors = None
        densities = None
        if self.options.use_constraint_1:
            factors = self._build_pair_factors()
        else:
            densities = self._build_pair_densities()

        for ki in range(self.nkpts):
            noi = int(occupied_count[ki])
            for kj in range(self.nkpts):
                noj = int(occupied_count[kj])
                for ka in range(self.nkpts):
                    kb = int(kb_map[ki, ka, kj])
                    nva = int(virtual_count[ka])
                    nvb = int(virtual_count[kb])
                    active_count += noi * noj * nva * nvb
                    for q_index in range(1, nsamples):
                        if self.options.use_constraint_1:
                            partials = self._diagonal_channel_partials(
                                factors[q_index],
                                q_index,
                                ki,
                                kj,
                                ka,
                                kb,
                                occupied,
                                virtual,
                            )
                        else:
                            partials = self._full_density_channel_partials(
                                densities,
                                q_index,
                                ki,
                                kj,
                                ka,
                                kb,
                                occupied,
                                virtual,
                            )
                        raw[:, q_index] += (
                            np.sum(partials[0]) * nva * nvb,
                            np.sum(partials[1]) * noi * noj,
                            np.sum(partials[2]) * noj * nvb,
                            np.sum(partials[3]) * noi * nva,
                            np.sum(partials[4]) * noi * nvb,
                            np.sum(partials[5]) * noj * nva,
                        )

        if active_count <= 0:
            raise ValueError("CCD SS fitting requires physical active entries")
        self._ss_active_count = active_count
        raw[:, 0] = active_count
        return raw

    def _normalize_aggregate_structure_factors(self, raw, active_count=None):
        """Normalize curves by the exact physical q=0 entry count."""
        expected = (6, len(self._ss_q_vectors))
        if np.shape(raw) != expected:
            raise ValueError(
                "aggregate CCD SS structure factors must have shape "
                f"{expected}; received {np.shape(raw)}"
            )
        if active_count is None or np.ndim(active_count) != 0 or not isinstance(
            active_count, numbers.Integral
        ):
            active_count = getattr(self, "_ss_active_count", None)
        if active_count is None:
            self._build_active_masks()
            active_count = self._ss_active_count
        active_count = int(active_count)
        if active_count <= 0:
            raise ValueError("CCD SS fitting requires physical active entries")
        samples = np.asarray(raw, dtype=np.complex128).copy()
        samples /= active_count
        samples[:, 0] = 1.0
        return samples

    def _aggregate_channel_samples(self, t2=None):
        """Return six normalized, complex density-only aggregate curves."""
        raw = self._contract_aggregate_structure_factors(t2)
        samples = self._normalize_aggregate_structure_factors(raw)
        if self.verbose >= logger.DEBUG2:
            self._log_ss_norms(raw[:, 0], self.ss_prepare_count)
        return samples

    def _gaussian_xi(self, sigma):
        if sigma == 0.0:
            return 0.0
        if np.isinf(sigma):
            # Each channel xi is the positive Madelung quadrature defect.  The
            # 2-positive/4-negative sign pattern then gives 2*signed_xi*T2.
            return -self.madelung_constant

        cache = getattr(self, "_ss_xi_cache", None)
        if cache is None:
            cache = self._ss_xi_cache = {}
        if sigma in cache:
            return cache[sigma]

        cell = self._scf.cell
        volume = float(cell.vol)
        reciprocal = np.asarray(cell.reciprocal_vectors())
        qmesh = np.asarray(self.kpts) - np.asarray(self.kpts[0])
        # Include enough reciprocal shells for the Gaussian to be negligible.
        bnorm = np.linalg.norm(reciprocal, axis=1)
        shell = np.maximum(1, np.ceil(8.0 * sigma / bnorm).astype(int))
        shell_size = len(qmesh)
        for extent in shell:
            shell_size *= 2 * int(extent) + 1
            if shell_size > 2_000_000:
                return -self.madelung_constant
        integers = np.stack(
            np.meshgrid(
                *[np.arange(-n, n + 1) for n in shell], indexing="ij"
            ),
            axis=-1,
        ).reshape(-1, 3)
        gvectors = integers @ reciprocal
        qg = (qmesh[:, None, :] + gvectors[None, :, :]).reshape(-1, 3)
        q2 = lib.einsum("qi,qi->q", qg, qg)
        active = q2 > 1e-20
        quadrature = (
            np.sum(4.0 * np.pi * np.exp(-q2[active] / (2.0 * sigma * sigma)) / q2[active])
            / (self.nkpts * volume)
        )
        integral = np.sqrt(2.0 / np.pi) * sigma
        xi = float(integral - quadrature)
        if not np.isfinite(xi):
            raise ValueError("Gaussian correction produced a non-finite value")
        cache[sigma] = xi
        return xi

    def _log_ss_samples(self, samples, sigmas, xi, preparation_id):
        """Log one DEBUG2 row per normalized aggregate channel sample."""
        for channel in range(6):
            sigma = float(sigmas[channel])
            correction = float(xi[channel])
            for q_index, q_vector in enumerate(self._ss_q_vectors):
                qx, qy, qz = (float(value) for value in q_vector)
                q_squared = qx * qx + qy * qy + qz * qz
                raw = complex(samples[channel, q_index])
                if sigma == 0.0:
                    fit = 1.0 if q_squared == 0.0 else 0.0
                elif np.isinf(sigma):
                    fit = 1.0
                else:
                    fit = float(np.exp(-q_squared / (2.0 * sigma * sigma)))
                residual = fit - raw
                logger.debug2(
                    self,
                    "CCDSS_SF prep=%d channel=L%d q_index=%d "
                    "qx=%.16e qy=%.16e qz=%.16e "
                    "raw_real=%.16e raw_imag=%.16e sigma=%.16e "
                    "xi=%.16e fit=%.16e residual_real=%.16e "
                    "residual_imag=%.16e",
                    int(preparation_id),
                    channel + 1,
                    int(q_index),
                    qx,
                    qy,
                    qz,
                    raw.real,
                    raw.imag,
                    sigma,
                    correction,
                    fit,
                    residual.real,
                    residual.imag,
                )

    def _log_ss_norms(self, origins, preparation_id):
        """Log one DEBUG2 row per raw aggregate channel origin."""
        for channel, origin in enumerate(origins):
            value = complex(origin)
            logger.debug2(
                self,
                "CCDSS_SF_NORM prep=%d channel=L%d "
                "norm_real=%.16e norm_imag=%.16e norm_abs=%.16e",
                int(preparation_id),
                channel + 1,
                value.real,
                value.imag,
                abs(value),
            )

    def _prepare_ss(self, t2=None):
        self._validate_constraints()
        if t2 is not None:
            self._validate_t2(t2)
        if self.ss_xi is not None:
            return self.ss_xi
        report_timing = self.verbose >= logger.INFO
        if report_timing:
            cpu0 = logger.process_clock()
            wall0 = logger.perf_counter()
        self.ss_prepare_count += 1
        if self.options.fixed_sigma is not None:
            sigmas = np.full(6, self.options.fixed_sigma, dtype=float)
            xi = np.full(
                6, self._gaussian_xi(self.options.fixed_sigma), dtype=float
            )
        else:
            samples = self._aggregate_channel_samples()
            sigmas = np.empty(6, dtype=float)
            xi = np.empty(6, dtype=float)
            for channel in range(6):
                sigma = _fit_unit_gaussian(
                    self._ss_q_vectors,
                    samples[channel],
                    fit_with_coul=self.options.fit_with_coul,
                )
                sigmas[channel] = sigma
                xi[channel] = self._gaussian_xi(sigma)
                self.ss_fit_count += 1
            if self.verbose >= logger.DEBUG2:
                self._log_ss_samples(
                    samples, sigmas, xi, self.ss_prepare_count
                )
        self.ss_sigmas = sigmas
        self.ss_xi = xi
        if report_timing:
            logger.info(
                self,
                "KRCCDSS xi precomputation CPU %.2f sec, wall %.2f sec",
                logger.process_clock() - cpu0,
                logger.perf_counter() - wall0,
            )
        return xi

    def _ss_residual_coefficient(self, t2):
        self._validate_constraints()
        if self.ss_xi is None:
            self._prepare_ss(t2)
        xi = self.ss_xi
        return xi[0] + xi[1] - xi[2] - xi[3] - xi[4] - xi[5]

    def _inject_ss_residual(self, t2new, t2, eris):
        """Inject the SS term into the residual before denominator division."""
        coefficient = self._ss_residual_coefficient(t2)
        residual = coefficient * t2
        self.last_ss_residual_norm = float(np.linalg.norm(residual))
        nocc = self.nocc
        nvir = self.nmo - nocc
        nonzero_o, nonzero_v = padding_k_idx(self, kind="split")
        mo_e_o = [e[:nocc] for e in eris.mo_energy]
        mo_e_v = [e[nocc:] + self.level_shift for e in eris.mo_energy]
        for ki in range(self.nkpts):
            for kj in range(self.nkpts):
                for ka in range(self.nkpts):
                    kb = self.khelper.kconserv[ki, ka, kj]
                    eia = kccsd_rhf._get_epq(
                        [0, nocc, ki, mo_e_o, nonzero_o],
                        [0, nvir, ka, mo_e_v, nonzero_v],
                        fac=[1.0, -1.0],
                    )
                    ejb = kccsd_rhf._get_epq(
                        [0, nocc, kj, mo_e_o, nonzero_o],
                        [0, nvir, kb, mo_e_v, nonzero_v],
                        fac=[1.0, -1.0],
                    )
                    denominator = eia[:, None, :, None] + ejb[:, None, :]
                    numerator = t2new[ki, kj, ka] * denominator
                    numerator += residual[ki, kj, ka]
                    t2new[ki, kj, ka] = numerator / denominator
        return t2new

    def update_amps(self, t1, t2, eris):
        self._validate_t2(t2)
        zero_t1 = np.zeros_like(t1)
        # Call the PySCF implementation directly so KRCCD's exact Madelung
        # correction cannot be applied in addition to singularity subtraction.
        _, t2new = kccsd_rhf.RCCSD.update_amps(self, zero_t1, t2, eris)
        self._inject_ss_residual(t2new, t2, eris)
        return zero_t1, t2new


KCCD_SS = KRCCD_SS

__all__ = [
    "CCDSSOptions",
    "KRCCD_SS",
    "KCCD_SS",
    "_fit_unit_gaussian",
    "_line_samples",
]
