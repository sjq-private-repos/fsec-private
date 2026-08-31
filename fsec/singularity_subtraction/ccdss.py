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

The fitted curves are the complete, k-point-summed structure factors
``S1``--``S6`` from Eqs. (2.1) and (2.3)--(2.7) of ``ccd-ss.pdf``.  PySCF
stores doubles as ``t2[ki,kj,ka,i,j,a,b]`` and determines the fourth momentum
with ``kb = kconserv[ki,ka,kj]``.  Each curve therefore depends on the current
doubles amplitudes and is rebuilt on every fitted CCD update.

The resulting correction added to the doubles residual is
``(xi1 + xi2 - xi3 - xi4 - xi5 - xi6) * t2``.
"""


@dataclass(frozen=True)
class CCDSSOptions:
    """Configuration for :class:`KRCCD_SS`.

    ``line_points`` is the number of positive samples on each reciprocal
    direction.  Samples are separated by ``b_i / n_i``, where ``n_i`` is the
    Monkhorst--Pack mesh size.  The Gaussian is
    ``exp(-|Q|**2 / (2*sigma**2))`` and its coefficient is fixed to one.
    ``structure_factor_imag_tol`` bounds both raw and normalized imaginary
    residues relative to the magnitude of the channel origin.
    """

    line_points: int = 3
    pair_density_becke_grid_level: int = 2
    fit_with_coul: bool = True
    fixed_sigma: float | None = None
    amplitude_fit_tol: float = 1e-12
    structure_factor_imag_tol: float = 1e-5

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
        try:
            imag_tol = float(self.structure_factor_imag_tol)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "structure_factor_imag_tol must be a finite non-negative number"
            ) from exc
        if not np.isfinite(imag_tol) or imag_tol < 0:
            raise ValueError(
                "structure_factor_imag_tol must be a finite non-negative number"
            )
        object.__setattr__(self, "structure_factor_imag_tol", imag_tol)
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

    The active path evaluates the six complete scalar structure factors using
    full occupied--occupied and virtual--virtual transition densities.
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
        line_points=3,
        pair_density_becke_grid_level=2,
        fit_with_coul=True,
        fixed_sigma=None,
        amplitude_fit_tol=1e-12,
        structure_factor_imag_tol=1e-5,
        occupied_orbital_shift=None,
    ):
        self.options = _merge_options(
            options,
            line_points=line_points,
            pair_density_becke_grid_level=pair_density_becke_grid_level,
            fit_with_coul=fit_with_coul,
            fixed_sigma=fixed_sigma,
            amplitude_fit_tol=amplitude_fit_tol,
            structure_factor_imag_tol=structure_factor_imag_tol,
        )
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
        self._ss_pair_densities = None
        self._ss_pair_orbitals = None
        self._ss_q_vectors = None
        self._ss_plus = None
        self._ss_minus = None
        self._ss_active_masks = None

    def dump_flags(self, verbose=None):
        result = super().dump_flags(verbose)
        log = logger.new_logger(self, verbose)
        log.info("CCD SS positive line samples = %d", self.options.line_points)
        log.info(
            "CCD SS Becke grid level = %d",
            self.options.pair_density_becke_grid_level,
        )
        log.info("CCD SS fixed sigma = %s", self.options.fixed_sigma)
        log.info(
            "CCD SS structure-factor imaginary tolerance = %g",
            self.options.structure_factor_imag_tol,
        )
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
        # Fitted curves depend on T2 and are prepared by each residual update.
        # Fixed widths are amplitude-independent and may be cached now.
        if self.options.fixed_sigma is not None:
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

    def _build_pair_densities(self):
        """Build normalized occupied and virtual transition densities.

        ``rho[q, k, p, s]`` links an orbital ``p`` at ``k`` to an orbital
        ``s`` at ``plus[q, k]``.  Occupied--virtual blocks are never needed by
        the six chapter-2 structure factors and are not stored.
        """
        cached = getattr(self, "_ss_pair_densities", None)
        if cached is not None:
            return cached
        self._prepare_sampling_maps()
        coords, weights, u, norms, wrapped_kpts = self._build_pair_orbitals()
        nsamples = len(self._ss_q_vectors)
        nvir = self.nmo - self.nocc
        rho_oo = np.zeros(
            (nsamples, self.nkpts, self.nocc, self.nocc), dtype=np.complex128
        )
        rho_vv = np.zeros(
            (nsamples, self.nkpts, nvir, nvir), dtype=np.complex128
        )
        rho_oo[0] = np.eye(self.nocc, dtype=np.complex128)
        rho_vv[0] = np.eye(nvir, dtype=np.complex128)

        for iq, q in enumerate(self._ss_q_vectors[1:], start=1):
            target = self._ss_plus[iq]
            unwrapped = self.kpts + q
            wrapped = wrapped_kpts[target]
            gdiff = unwrapped - wrapped
            wrap_phase = np.exp(-1j * (coords @ gdiff.T)).T
            u_target = u[target] * wrap_phase[:, None, :]
            for output, orbital_slice in (
                (rho_oo, slice(None, self.nocc)),
                (rho_vv, slice(self.nocc, None)),
            ):
                source_orbitals = u[:, orbital_slice]
                target_orbitals = u_target[:, orbital_slice]
                overlap = lib.einsum(
                    "kpr,r,ksr->kps",
                    source_orbitals.conj(),
                    weights,
                    target_orbitals,
                )
                source_norms = norms[:, orbital_slice]
                target_norms = norms[target, orbital_slice]
                denominator = np.sqrt(
                    np.maximum(
                        source_norms[:, :, None] * target_norms[:, None, :],
                        0.0,
                    )
                )
                np.divide(
                    overlap,
                    denominator,
                    out=output[iq],
                    where=denominator > 1e-14,
                )

        self._ss_pair_densities = rho_oo, rho_vv
        return self._ss_pair_densities

    def _build_active_masks(self):
        """Return boolean occupied and virtual masks for every k point."""
        cached = getattr(self, "_ss_active_masks", None)
        if cached is not None:
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
        return self._ss_active_masks

    def _contract_aggregate_structure_factors(self, t2):
        """Evaluate the complete scalar ``S1``--``S6`` curves.

        Every density endpoint and amplitude axis is sliced with the padding
        mask at its actual, possibly shifted momentum.  Each ``einsum``
        contracts directly to a scalar, so no aggregate T2-sized intermediate
        is constructed.
        """
        self._validate_t2(t2)
        self._prepare_sampling_maps()
        occupied, virtual = self._build_active_masks()
        occupied_indices = tuple(np.flatnonzero(mask) for mask in occupied)
        virtual_indices = tuple(np.flatnonzero(mask) for mask in virtual)
        nsamples = len(self._ss_q_vectors)
        raw = np.zeros((6, nsamples), dtype=np.complex128)
        rho_oo_all, rho_vv_all = self._build_pair_densities()
        kconserv = np.asarray(self.khelper.kconserv)

        def density_block(density, source, row_indices, column_indices):
            return density[source][np.ix_(row_indices, column_indices)]

        def amplitude_block(k1, k2, ka):
            kb = int(kconserv[k1, ka, k2])
            indices = np.ix_(
                occupied_indices[k1],
                occupied_indices[k2],
                virtual_indices[ka],
                virtual_indices[kb],
            )
            return t2[k1, k2, ka][indices]

        for q_index in range(nsamples):
            plus = self._ss_plus[q_index]
            minus = self._ss_minus[q_index]
            rho_oo = rho_oo_all[q_index]
            rho_vv = rho_vv_all[q_index]
            for ki in range(self.nkpts):
                plus_i = int(plus[ki])
                minus_i = int(minus[ki])
                rho_i = density_block(
                    rho_oo, ki, occupied_indices[ki], occupied_indices[plus_i]
                )
                rho_minus_i = density_block(
                    rho_oo,
                    minus_i,
                    occupied_indices[minus_i],
                    occupied_indices[ki],
                )
                for kj in range(self.nkpts):
                    plus_j = int(plus[kj])
                    rho_j = density_block(
                        rho_oo,
                        kj,
                        occupied_indices[kj],
                        occupied_indices[plus_j],
                    )
                    for ka in range(self.nkpts):
                        kb = int(kconserv[ki, ka, kj])
                        plus_a = int(plus[ka])
                        plus_b = int(plus[kb])
                        minus_b = int(minus[kb])
                        rho_a = density_block(
                            rho_vv,
                            ka,
                            virtual_indices[ka],
                            virtual_indices[plus_a],
                        )
                        rho_b = density_block(
                            rho_vv,
                            kb,
                            virtual_indices[kb],
                            virtual_indices[plus_b],
                        )
                        rho_minus_b = density_block(
                            rho_vv,
                            minus_b,
                            virtual_indices[minus_b],
                            virtual_indices[kb],
                        )

                        raw[0, q_index] += lib.einsum(
                            "ki,jl,klab->",
                            rho_minus_i,
                            rho_j.conj(),
                            amplitude_block(minus_i, plus_j, ka),
                        )
                        raw[1, q_index] += lib.einsum(
                            "ac,db,ijcd->",
                            rho_a,
                            rho_minus_b.conj(),
                            amplitude_block(ki, kj, plus_a),
                        )
                        raw[2, q_index] += lib.einsum(
                            "ac,ik,kjcb->",
                            rho_a,
                            rho_i.conj(),
                            amplitude_block(plus_i, kj, plus_a),
                        )
                        raw[3, q_index] += lib.einsum(
                            "bc,jk,kica->",
                            rho_b,
                            rho_j.conj(),
                            amplitude_block(plus_j, ki, plus_b),
                        )
                        raw[4, q_index] += lib.einsum(
                            "ac,jk,kibc->",
                            rho_a,
                            rho_j.conj(),
                            amplitude_block(plus_j, ki, kb),
                        )
                        raw[5, q_index] += lib.einsum(
                            "bc,ik,kjac->",
                            rho_b,
                            rho_i.conj(),
                            amplitude_block(plus_i, kj, ka),
                        )
        return raw

    def _normalize_aggregate_structure_factors(self, raw):
        """Validate and independently normalize the six complex curves."""
        expected = (6, len(self._ss_q_vectors))
        if np.shape(raw) != expected:
            raise ValueError(
                "aggregate CCD SS structure factors must have shape "
                f"{expected}; received {np.shape(raw)}"
            )
        raw = np.asarray(raw, dtype=np.complex128)
        origins = raw[:, 0]
        for channel, origin in enumerate(origins, start=1):
            if abs(origin) < self.options.amplitude_fit_tol:
                raise ValueError(
                    f"CCD SS channel L{channel} origin magnitude {abs(origin):.16e} "
                    "is below amplitude_fit_tol="
                    f"{self.options.amplitude_fit_tol:.16e}"
                )

        normalized = raw / origins[:, None]
        tolerance = self.options.structure_factor_imag_tol
        for channel in range(6):
            origin_magnitude = abs(origins[channel])
            for q_index in range(len(self._ss_q_vectors)):
                raw_imag_scaled = abs(raw[channel, q_index].imag) / origin_magnitude
                if raw_imag_scaled > tolerance:
                    raise ValueError(
                        f"CCD SS channel L{channel + 1} sample {q_index} raw "
                        "imaginary magnitude scaled by |S(0)| "
                        f"{raw_imag_scaled:.16e} exceeds "
                        f"structure_factor_imag_tol={tolerance:.16e}"
                    )
                normalized_imag = abs(normalized[channel, q_index].imag)
                if normalized_imag > tolerance:
                    raise ValueError(
                        f"CCD SS channel L{channel + 1} sample {q_index} "
                        f"normalized imaginary magnitude {normalized_imag:.16e} "
                        "exceeds structure_factor_imag_tol="
                        f"{tolerance:.16e}"
                    )
        # Preserve the exact unit-coefficient condition used by the fit after
        # validating the computed origins through the same equations.
        normalized[:, 0] = 1.0
        return normalized

    def _aggregate_channel_samples(self, t2):
        """Return unnormalized and normalized complex structure factors."""
        raw = self._contract_aggregate_structure_factors(t2)
        return raw, self._normalize_aggregate_structure_factors(raw)

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

    def _log_ss_samples(self, raw, normalized, sigmas, xi, preparation_id):
        """Log raw and normalized values for every fitted channel sample."""
        for channel in range(6):
            sigma = float(sigmas[channel])
            correction = float(xi[channel])
            for q_index, q_vector in enumerate(self._ss_q_vectors):
                qx, qy, qz = (float(value) for value in q_vector)
                q_squared = qx * qx + qy * qy + qz * qz
                raw_value = complex(raw[channel, q_index])
                normalized_value = complex(normalized[channel, q_index])
                if sigma == 0.0:
                    fit = 1.0 if q_squared == 0.0 else 0.0
                elif np.isinf(sigma):
                    fit = 1.0
                else:
                    fit = float(np.exp(-q_squared / (2.0 * sigma * sigma)))
                residual = fit - normalized_value
                logger.debug2(
                    self,
                    "CCDSS_SF prep=%d channel=L%d q_index=%d "
                    "qx=%.16e qy=%.16e qz=%.16e "
                    "raw_real=%.16e raw_imag=%.16e "
                    "normalized_real=%.16e normalized_imag=%.16e "
                    "sigma=%.16e xi=%.16e fit=%.16e "
                    "residual_real=%.16e residual_imag=%.16e",
                    int(preparation_id),
                    channel + 1,
                    int(q_index),
                    qx,
                    qy,
                    qz,
                    raw_value.real,
                    raw_value.imag,
                    normalized_value.real,
                    normalized_value.imag,
                    sigma,
                    correction,
                    fit,
                    residual.real,
                    residual.imag,
                )

    def _prepare_ss(self, t2=None):
        if t2 is not None:
            self._validate_t2(t2)
        fixed_sigma = self.options.fixed_sigma
        if fixed_sigma is not None and self.ss_xi is not None:
            return self.ss_xi
        report_timing = self.verbose >= logger.INFO
        if report_timing:
            cpu0 = logger.process_clock()
            wall0 = logger.perf_counter()
        preparation_id = self.ss_prepare_count + 1
        if fixed_sigma is not None:
            sigmas = np.full(6, fixed_sigma, dtype=float)
            xi = np.full(
                6, self._gaussian_xi(fixed_sigma), dtype=float
            )
        else:
            if t2 is None:
                self._validate_t2(t2)
            raw, normalized = self._aggregate_channel_samples(t2)
            sigmas = np.empty(6, dtype=float)
            xi = np.empty(6, dtype=float)
            for channel in range(6):
                sigma = _fit_unit_gaussian(
                    self._ss_q_vectors,
                    normalized[channel].real,
                    fit_with_coul=self.options.fit_with_coul,
                )
                sigmas[channel] = sigma
                xi[channel] = self._gaussian_xi(sigma)
            if self.verbose >= logger.DEBUG2:
                self._log_ss_samples(
                    raw, normalized, sigmas, xi, preparation_id
                )
        self.ss_sigmas = sigmas
        self.ss_xi = xi
        self.ss_prepare_count += 1
        if fixed_sigma is None:
            self.ss_fit_count += 6
        if report_timing:
            logger.info(
                self,
                "KRCCDSS xi precomputation CPU %.2f sec, wall %.2f sec",
                logger.process_clock() - cpu0,
                logger.perf_counter() - wall0,
            )
        return xi

    def _ss_residual_coefficient(self, t2):
        if self.options.fixed_sigma is None or self.ss_xi is None:
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
