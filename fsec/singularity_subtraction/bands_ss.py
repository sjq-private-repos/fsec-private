"""Orbital-resolved singularity subtraction for periodic band energies.

The implementation uses analytic Fourier AO-pair integrals throughout.  This
keeps the zero-transfer mask inside the exchange operator, before any exchange
matrix is formed, and also makes the transfer convention explicit for shifted
band k-points.
"""

from __future__ import annotations

import copy
import itertools
import os
import tempfile

import numpy as np
from pyscf.pbc.df import ft_ao
from pyscf.pbc.scf import hf as pbc_hf
from pyscf.pbc.tools import get_monkhorst_pack_size

from fsec.singularity_subtraction import model_function
from fsec.singularity_subtraction.function_fitting import ExxScipyLeastSquares


DEFAULT_Q_ZERO_TOL = 1.0e-4  # bohr^-1
_FIT_MIN_POINTS = 6


def _canonical_kpts(cell, kpts):
    """Map k-points to a consistent first-BZ image for finite G grids."""
    scaled = cell.get_scaled_kpts(np.asarray(kpts, dtype=float).reshape(-1, 3))
    scaled -= np.floor(scaled + 0.5)
    return cell.get_abs_kpts(scaled)


def _reciprocal_vectors(cell, mesh):
    """Return one deterministic reciprocal-vector list for a three-axis mesh."""
    mesh = np.asarray(mesh, dtype=int)
    if mesh.shape != (3,) or np.any(mesh < 1):
        raise ValueError("The reciprocal mesh must contain three positive integers")
    return np.asarray(cell.get_Gv(mesh=mesh))


def _exchange_mesh(cell):
    """Return the physical cell mesh used by the default Fourier exchange."""
    return np.asarray(cell.mesh, dtype=int)


def fourier_exchange_matrices(kmf, dm_kpts, kpts_band, q_zero_tol=DEFAULT_Q_ZERO_TOL,
                              block_size=512, mesh=None):
    r"""Build unscaled masked exchange matrices at arbitrary target k-points.

    The density matrices must include their physical orbital occupations (two
    for a doubly occupied restricted orbital).  The returned matrix is the
    unscaled ``K`` in a restricted Fock operator ``-alpha_x * K / 2``.

    Args:
        kmf: Converged periodic restricted PySCF mean-field object.
        dm_kpts: Source density matrices, shape ``(nk, nao, nao)``.
        kpts_band: Target points, shape ``(nband, 3)``.
        q_zero_tol: Exclude transfers with norm less than or equal to this
            value, in inverse bohr.
        block_size: Maximum number of reciprocal vectors passed to one AO-pair
            integral call.
        mesh: Optional reciprocal mesh. Defaults to ``cell.mesh``.

    Returns:
        Complex Hermitian matrices of shape ``(nband, nao, nao)``.
    """
    cell = kmf.cell
    source_kpts = np.asarray(kmf.kpts, dtype=float).reshape(-1, 3)
    target_kpts = np.asarray(kpts_band, dtype=float).reshape(-1, 3)
    if dm_kpts.shape[0] != len(source_kpts):
        raise ValueError("dm_kpts and kmf.kpts must have the same source-k dimension")
    if q_zero_tol < 0:
        raise ValueError("q_zero_tol must be nonnegative")
    if block_size < 1:
        raise ValueError("block_size must be positive")

    source_internal = _canonical_kpts(cell, source_kpts)
    target_internal = _canonical_kpts(cell, target_kpts)
    Gv = _reciprocal_vectors(cell, _exchange_mesh(cell) if mesh is None else mesh)
    volume = abs(float(np.linalg.det(cell.lattice_vectors())))
    nk = len(source_kpts)
    nao = cell.nao_nr()
    out = np.zeros((len(target_kpts), nao, nao), dtype=np.complex128)

    for itarget, target_k in enumerate(target_internal):
        K = out[itarget]
        for isource, source_k in enumerate(source_internal):
            q = target_k - source_k
            dm = np.asarray(dm_kpts[isource])
            for start in range(0, len(Gv), block_size):
                G_block = Gv[start:start + block_size]
                p = q + G_block
                p2 = np.einsum("gi,gi->g", p, p)
                keep = p2 > q_zero_tol**2
                if not np.any(keep):
                    continue
                p2 = p2[keep]
                B = ft_ao.ft_aopair(
                    cell,
                    G_block[keep],
                    kpti_kptj=np.asarray([source_k, target_k]),
                )
                weight = 4.0 * np.pi / (p2 * volume * nk)
                K += np.einsum(
                    "g,gmi,mn,gnj->ij", weight, B.conj(), dm, B,
                    optimize=True,
                )
        out[itarget] = 0.5 * (K + K.conj().T)
    return out


def _sample_structure_factor(kmf, kpt_band, mo_coeff_band, Gv,
                             g_indices_by_source, occupied_indices,
                             target_indices,
                             block_size=128):
    """Evaluate selected source/G samples in source-major order."""
    cell = kmf.cell
    source_kpts = np.asarray(kmf.kpts, dtype=float).reshape(-1, 3)
    source_internal = _canonical_kpts(cell, source_kpts)
    target_internal = _canonical_kpts(cell, np.asarray(kpt_band).reshape(1, 3))[0]
    coeff_source = np.asarray(kmf.mo_coeff_kpts)
    coeff_target = np.asarray(mo_coeff_band)
    occupied_indices = np.asarray(occupied_indices, dtype=int)
    target_indices = np.asarray(target_indices, dtype=int)
    ntarget = len(target_indices)
    if coeff_target.ndim != 2 or coeff_target.shape[1] <= int(np.max(target_indices)):
        raise ValueError("mo_coeff_band does not contain the requested target orbitals")

    p_samples = []
    factors = []
    source_indices = []
    g_indices = []
    for isource, source_k in enumerate(source_internal):
        selected_g = np.asarray(g_indices_by_source[isource], dtype=int)
        if selected_g.size == 0:
            continue
        q = target_internal - source_k
        p_samples.append(q + Gv[selected_g])
        source_indices.append(np.full(len(selected_g), isource, dtype=int))
        g_indices.append(selected_g)
        S_source = np.zeros((len(selected_g), ntarget), dtype=float)
        C_source = coeff_source[isource][:, occupied_indices]
        C_target = coeff_target[:, target_indices]
        for start in range(0, len(selected_g), block_size):
            stop = min(start + block_size, len(selected_g))
            B = ft_ao.ft_aopair(
                cell,
                Gv[selected_g[start:stop]],
                kpti_kptj=np.asarray([source_k, target_internal]),
            )
            rho = np.einsum(
                "mj,gmn,nl->gjl", C_source.conj(), B, C_target,
                optimize=True,
            )
            S_source[start:stop] = np.sum(np.abs(rho) ** 2, axis=1).real
        factors.append(S_source)

    if not p_samples:
        return (np.empty((0, 3)), np.empty((0, ntarget)),
                np.empty(0, dtype=int), np.empty(0, dtype=int))

    return (
        np.concatenate(p_samples, axis=0),
        np.concatenate(factors, axis=0),
        np.concatenate(source_indices),
        np.concatenate(g_indices),
    )


def orbital_resolved_structure_factor(kmf, kpt_band, mo_coeff_band,
                                      q_zero_tol=DEFAULT_Q_ZERO_TOL,
                                      mesh=None, block_size=512,
                                      occupied_indices=None,
                                      target_indices=None):
    r"""Evaluate ``S[n,p]`` for all source k-points and reciprocal vectors.

    Samples are returned in source-k-major, reciprocal-vector-minor order.
    The structure factor is ``sum_j |rho[j,n]|**2`` for each source point;
    there is deliberately no source-k average.  ``q_zero_tol`` is accepted to
    make the shared cutoff convention explicit; the raw structure factor is
    retained at zero for normalization checks, while callers mask it before
    fitting.
    """
    if mesh is None:
        mesh = kmf.cell.mesh
    Gv = _reciprocal_vectors(kmf.cell, mesh)
    if occupied_indices is None:
        occupations = np.asarray(kmf.mo_occ[0])
        occupied_indices = np.flatnonzero(np.isclose(occupations, 2.0, atol=1e-8))
    if target_indices is None:
        target_indices = occupied_indices
    del q_zero_tol
    all_g = np.arange(len(Gv), dtype=int)
    return _sample_structure_factor(
        kmf, kpt_band, mo_coeff_band, Gv,
        [all_g for _ in range(len(kmf.kpts))], occupied_indices,
        target_indices, block_size,
    )


def _gaussian_transfer_grid(cell, q, sigma, max_sigma=8.0):
    """Generate all reciprocal images needed inside ``max_sigma * sigma``."""
    reciprocal = np.asarray(cell.reciprocal_vectors())
    radius = max_sigma * float(sigma)
    inverse = np.linalg.inv(reciprocal)
    component_bounds = np.ceil((radius + np.linalg.norm(q)) *
                               np.linalg.norm(inverse, axis=0)).astype(int) + 1
    integer_axes = [range(-bound, bound + 1) for bound in component_bounds]
    indices = np.asarray(list(itertools.product(*integer_axes)), dtype=int)
    Gv = indices @ reciprocal
    p = np.asarray(q) + Gv
    keep = np.linalg.norm(p, axis=1) <= radius + 1e-12
    return p[keep]


def _fit_gaussian(p, values, initial_guess=None, fit_with_coul=True):
    """Fit one normalized isotropic Gaussian and return sigma plus diagnostics."""
    model = model_function.ContractedGaussianModel(
        num_gaussians=1, isotropic=True, parameters=[1.0, 1.0]
    )
    fitter = ExxScipyLeastSquares(
        model,
        fit_with_coul=fit_with_coul,
        initial_guess=initial_guess,
    )
    parameters = fitter.fit_model(p, values, force_positive_params=True)
    sigma = float(parameters[1])
    prediction = np.exp(-np.einsum("ij,ij->i", p, p) / (2.0 * sigma**2))
    residual = prediction - values
    return sigma, {
        "parameters": np.asarray(parameters, dtype=float),
        "n_samples": int(len(values)),
        "residual_norm": float(np.linalg.norm(residual)),
        "rms_residual": float(np.sqrt(np.mean(residual**2))),
    }


def _whole_shell_mask(values, npoints):
    """Select at least ``npoints``, including ties at the boundary radius."""
    values = np.asarray(values)
    if len(values) < npoints:
        return np.ones(len(values), dtype=bool)
    boundary = np.partition(values, npoints - 1)[npoints - 1]
    tolerance = max(1e-12, abs(float(boundary)) * 1e-10)
    return values <= boundary + tolerance


class BandsSS:
    r"""Orbital-resolved Gaussian singularity subtraction for band energies.

    The source mean-field density is held fixed.  A Gaussian is fitted
    independently for each occupied band orbital at every target k-point.
    The stored ``xi`` is unscaled; ``correction`` includes the functional's
    global exact-exchange fraction and has zeros for virtual orbitals.

    Args:
        kmf: Converged 3D restricted KRHF or global-hybrid KRKS calculation
            with a full, uniformly weighted source k-point mesh.
        kpts_band: Arbitrary target k-points in inverse bohr. A single point
            may be passed as shape ``(3,)``; output arrays retain a leading
            band-k dimension.
        mo_energy_band: Optional bare energies, shape ``(nband_k, nmo)``.
        mo_coeff_band: Optional matching coefficients, shape
            ``(nband_k, nao, nmo)``.
        q_zero_tol: Shared exclusion radius in inverse bohr. Defaults to
            ``1e-4``.
        **kwargs: ``N_local`` or ``sq_ke_cutoff`` set structure-factor samples;
            ``qG_norm_cutoff_sigma`` and ``qG_norm_cutoff`` set the fit radius;
            ``min_points`` sets the minimum fit-sample count; ``sigma`` (or
            ``model_params=[1, sigma]``) fixes a common Gaussian width;
            ``exchange_mesh`` controls Fourier exchange convergence;
            ``fit_with_coul`` and ``initial_guess`` configure the existing
            Gaussian fit; and ``block_size`` controls AO-pair memory.

    If ``mo_energy_band`` and ``mo_coeff_band`` are supplied, their bare
    energies must use the same ``q_zero_tol`` exchange mask used here.  The
    class cannot infer or repair a different convention in external energies.

    Result arrays ``xi``, ``sigma``, ``integral_term`` and
    ``quadrature_term`` have shape ``(nband_k, nocc)``; ``xi`` is in hartree
    and is unscaled. ``correction`` and ``mo_energy_band_corrected`` have
    shape ``(nband_k, nmo)`` with virtual corrections set to zero.
    """

    def __init__(self, kmf, kpts_band, mo_energy_band=None, mo_coeff_band=None,
                 q_zero_tol=DEFAULT_Q_ZERO_TOL, **kwargs):
        self.kmf = kmf
        self.cell = kmf.cell
        self.kpts_band_requested = np.asarray(kpts_band, dtype=float)
        self.kpts_band = self.kpts_band_requested.reshape(-1, 3)
        self.q_zero_tol = float(q_zero_tol)
        self.nocc = self._validate_mean_field()
        self.hybrid_fraction = self._hybrid_fraction()
        if (mo_energy_band is None) != (mo_coeff_band is None):
            raise ValueError("Supply both band energies and coefficients, or neither")

        self.mo_energy_band_input = mo_energy_band
        self.mo_coeff_band_input = mo_coeff_band
        self.mo_energy_band_baseline = None
        self.mo_energy_band_corrected = None
        self.mo_coeff_band = None
        self._baseline_ready = False
        self._correction_ready = False

        if self.q_zero_tol < 0:
            raise ValueError("q_zero_tol must be nonnegative")
        if len(self.kpts_band) == 0:
            raise ValueError("At least one band k-point is required")

        # Fit settings mirror the applicable one-Gaussian ExxSS options.
        self.fit_with_coul = kwargs.get("fit_with_coul", True)
        self.qG_norm_cutoff_sigma = kwargs.get("qG_norm_cutoff_sigma", 1.0)
        self.qG_norm_cutoff = kwargs.get("qG_norm_cutoff", None)
        self.min_points = max(_FIT_MIN_POINTS, int(kwargs.get("min_points", _FIT_MIN_POINTS)))
        self.block_size = int(kwargs.get("block_size", 512))
        self.exchange_mesh = np.asarray(kwargs.get("exchange_mesh", self.cell.mesh), dtype=int)
        if self.exchange_mesh.ndim == 0:
            self.exchange_mesh = np.repeat(self.exchange_mesh, 3)
        self.N_local = kwargs.get("N_local", self.cell.mesh)
        if kwargs.get("sq_ke_cutoff") is not None:
            self.N_local = self.cell.cutoff_to_mesh(kwargs["sq_ke_cutoff"])
        self.N_local = np.asarray(self.N_local, dtype=int)
        if self.N_local.ndim == 0:
            self.N_local = np.repeat(self.N_local, 3)
        if self.qG_norm_cutoff is not None and self.qG_norm_cutoff <= 0:
            raise ValueError("qG_norm_cutoff must be positive")
        if (self.qG_norm_cutoff_sigma is not None and
                self.qG_norm_cutoff_sigma <= 0 and self.qG_norm_cutoff is None):
            raise ValueError("qG_norm_cutoff_sigma must be positive")
        self.fixed_sigma = kwargs.get("sigma", None)
        if self.fixed_sigma is None and kwargs.get("model_params") is not None:
            params = np.asarray(kwargs["model_params"], dtype=float)
            if params.shape == (2,):
                if not np.isclose(params[0], 1.0, atol=1e-12):
                    raise ValueError("A normalized single Gaussian requires amplitude 1")
                self.fixed_sigma = float(params[1])
            else:
                raise ValueError("model_params must be [1, sigma] for one normalized Gaussian")
        if self.fixed_sigma is not None and self.fixed_sigma <= 0:
            raise ValueError("sigma must be positive")
        if kwargs.get("fit_method", "scipy_least_squares") != "scipy_least_squares":
            raise ValueError("BandsSS currently supports fit_method='scipy_least_squares'")
        self.initial_guess = kwargs.get("initial_guess", None)

        self.xi = None
        self.sigma = None
        self.integral_term = None
        self.quadrature_term = None
        self.correction = None
        self.fit_diagnostics = None
        self.fit_parameters = None
        self.p_samples = None
        self.structure_factors = None
        self.source_indices = None
        self.g_indices = None

        if mo_energy_band is not None:
            self._set_supplied_bands(mo_energy_band, mo_coeff_band)

    def _validate_mean_field(self):
        cell = self.cell
        if int(getattr(cell, "dimension", 3)) != 3:
            raise NotImplementedError("BandsSS is implemented for three-dimensional cells")
        if getattr(self.kmf, "spin", 0) not in (0, None) or getattr(cell, "spin", 0) != 0:
            raise NotImplementedError("BandsSS requires a closed-shell restricted mean-field object")
        if not hasattr(self.kmf, "mo_occ") or np.asarray(self.kmf.mo_occ).ndim != 2:
            raise NotImplementedError("Unrestricted mean-field objects are not supported")
        occupations = np.asarray(self.kmf.mo_occ, dtype=float)
        if not np.all(np.isclose(occupations, 0.0, atol=1e-8) |
                      np.isclose(occupations, 2.0, atol=1e-8)):
            raise NotImplementedError("Fractional occupations are not supported")
        occupancy_mask = np.isclose(occupations, 2.0, atol=1e-8)
        if not np.all(np.sum(occupancy_mask, axis=1) == np.sum(occupancy_mask[0])):
            raise NotImplementedError("The occupied subspace must have the same size at every source k-point")
        nocc = int(np.sum(occupancy_mask[0]))
        standard_occupation_order = np.arange(occupations.shape[1]) < nocc
        if not np.all(occupancy_mask == standard_occupation_order[None, :]):
            raise NotImplementedError("Occupied orbitals must precede virtual orbitals in MO order")
        self.occupied_indices = np.arange(nocc)
        if nocc == 0:
            raise ValueError("BandsSS requires at least one occupied orbital")

        source_kpts = np.asarray(self.kmf.kpts, dtype=float).reshape(-1, 3)
        try:
            mesh = np.asarray(get_monkhorst_pack_size(cell, source_kpts), dtype=int)
        except Exception as exc:
            raise NotImplementedError("BandsSS requires a full uniform occupied k-point mesh") from exc
        if int(np.prod(mesh)) != len(source_kpts):
            raise NotImplementedError("BandsSS requires the full occupied k-point mesh")
        scaled = cell.get_scaled_kpts(source_kpts)
        wrapped = scaled - np.floor(scaled)
        rounded = np.round(wrapped, decimals=10) % 1.0
        if len(np.unique(rounded, axis=0)) != len(source_kpts):
            raise NotImplementedError("The occupied k-point mesh contains duplicate points")
        for axis, npoints in enumerate(mesh):
            values = np.unique(rounded[:, axis])
            if len(values) != npoints:
                raise NotImplementedError("The occupied k-point mesh is not a full uniform grid")
            if npoints > 1:
                spacings = np.diff(np.r_[values, values[0] + 1.0])
                if not np.allclose(spacings, 1.0 / npoints, atol=1e-8, rtol=0):
                    raise NotImplementedError("The occupied k-point mesh is not uniformly spaced")
        return nocc

    def _hybrid_fraction(self):
        if not np.isclose(float(getattr(self.cell, "omega", 0.0)), 0.0, atol=1e-12):
            raise NotImplementedError("Range-separated Coulomb operators are not supported")
        if not hasattr(self.kmf, "xc"):
            return 1.0
        ni = self.kmf._numint
        omega, _alpha, hyb = ni.rsh_and_hybrid_coeff(self.kmf.xc, spin=self.cell.spin)
        if not np.isclose(omega, 0.0, atol=1e-12):
            raise NotImplementedError("Range-separated hybrid functionals are not supported")
        return float(hyb)

    def _set_supplied_bands(self, mo_energy_band, mo_coeff_band):
        energies = np.asarray(mo_energy_band)
        coefficients = np.asarray(mo_coeff_band)
        if energies.ndim == 1:
            energies = energies[None, :]
        if coefficients.ndim == 2:
            coefficients = coefficients[None, :, :]
        if energies.ndim != 2 or coefficients.ndim != 3:
            raise ValueError("Band arrays must have shapes (nk, nmo) and (nk, nao, nmo)")
        if energies.shape[0] != len(self.kpts_band) or coefficients.shape[0] != len(self.kpts_band):
            raise ValueError("The leading band-array dimension must match kpts_band")
        if energies.shape[1] != coefficients.shape[2] or coefficients.shape[1] != self.cell.nao_nr():
            raise ValueError("Band energy and coefficient dimensions do not match the cell basis")
        if energies.shape[1] < self.nocc:
            raise ValueError("Band arrays do not contain all occupied orbitals")
        self.mo_energy_band_baseline = np.array(energies, copy=True)
        self.mo_coeff_band = np.array(coefficients, copy=True)
        self._baseline_ready = True

    def _make_isolated_mean_field(self):
        # PySCF's DFT get_veff initializes grids and updates summary state.
        # Deep-copy the source so these changes and the exchange adapter stay local.
        try:
            return copy.deepcopy(self.kmf)
        except Exception as exc:
            raise RuntimeError("Could not isolate the PySCF mean-field object for band evaluation") from exc

    @staticmethod
    def _nonzero_helper_kpt(cell):
        """Return an unused nonzero target that makes tiny-k APIs complex-safe."""
        reciprocal = cell.reciprocal_vectors()
        return 0.23 * reciprocal[0] + 0.17 * reciprocal[1] + 0.11 * reciprocal[2]

    def _get_nonexchange_veff(self, mf_copy, dm_kpts, kpts_band):
        with_df = getattr(mf_copy, "with_df", None)
        temporary_df_dir = None
        try:
            if (with_df is not None and hasattr(with_df, "build") and
                    hasattr(with_df, "_cderi") and
                    hasattr(with_df, "_cderi_to_save") and with_df._cderi is None):
                if with_df._cderi_to_save is None:
                    temporary_df_dir = tempfile.TemporaryDirectory(prefix="bands_ss_df_")
                    with_df._cderi_to_save = os.path.join(temporary_df_dir.name, "cderi.h5")
                # Only Coulomb integrals are needed for the nonexchange Fock part.
                with_df.build(j_only=True, kpts_band=kpts_band)

            original_get_jk = mf_copy.get_jk

            def get_jk_without_exchange(cell=None, dm_kpts=None, hermi=1, kpts=None,
                                        kpts_band=None, with_j=True, with_k=True,
                                        omega=None, **kwargs):
                vj, _ = original_get_jk(
                    cell=cell, dm_kpts=dm_kpts, hermi=hermi, kpts=kpts,
                    kpts_band=kpts_band, with_j=with_j, with_k=False,
                    omega=omega, **kwargs,
                )
                if not with_k:
                    return vj, None
                nk = len(kpts if kpts_band is None else kpts_band)
                nao = self.cell.nao_nr()
                dtype = np.result_type(dm_kpts, np.complex128)
                return vj, np.zeros((nk, nao, nao), dtype=dtype)

            mf_copy.get_jk = get_jk_without_exchange
            if hasattr(mf_copy, "xc"):
                return mf_copy.get_veff(
                    cell=self.cell,
                    dm=dm_kpts,
                    kpts=self.kmf.kpts,
                    kpts_band=kpts_band,
                )
            return mf_copy.get_veff(
                cell=self.cell,
                dm_kpts=dm_kpts,
                kpts=self.kmf.kpts,
                kpts_band=kpts_band,
            )
        finally:
            if temporary_df_dir is not None:
                temporary_df_dir.cleanup()

    def _build_baseline_bands(self):
        if self._baseline_ready:
            return
        mf_copy = self._make_isolated_mean_field()
        dm_kpts = np.asarray(self.kmf.make_rdm1())
        # PySCF's local pseudopotential path can cast-fail for a lone Gamma or
        # tiny-k call. Include a nonzero helper only in the batched nonexchange
        # evaluation; it is discarded before diagonalizing requested points.
        helper = self._nonzero_helper_kpt(self.cell)
        eval_kpts = np.vstack((self.kpts_band, helper))

        hcore = mf_copy.get_hcore(cell=self.cell, kpts=eval_kpts)
        vnonexchange = self._get_nonexchange_veff(mf_copy, dm_kpts, eval_kpts)
        K = fourier_exchange_matrices(
            self.kmf,
            dm_kpts,
            self.kpts_band,
            q_zero_tol=self.q_zero_tol,
            block_size=self.block_size,
            mesh=self.exchange_mesh,
        )
        fock = np.asarray(hcore[:len(self.kpts_band)] +
                          vnonexchange[:len(self.kpts_band)])
        fock -= 0.5 * self.hybrid_fraction * K
        overlap = mf_copy.get_ovlp(cell=self.cell, kpts=eval_kpts)
        energies, coefficients = pbc_hf.eigh_with_canonical_orth(fock, overlap[:len(self.kpts_band)])
        self.mo_energy_band_baseline = np.asarray(energies)
        self.mo_coeff_band = np.asarray(coefficients)
        self._baseline_ready = True

    def _compute_auxiliary_terms(self):
        nband = len(self.kpts_band)
        integral = np.zeros((nband, self.nocc), dtype=float)
        quadrature = np.zeros_like(integral)
        sigma_values = np.zeros_like(integral)
        diagnostics = []
        fit_parameters = []
        source_internal = _canonical_kpts(self.cell, self.kmf.kpts)
        target_internal = _canonical_kpts(self.cell, self.kpts_band)
        volume = abs(float(np.linalg.det(self.cell.lattice_vectors())))
        nk = len(source_internal)
        Gv = _reciprocal_vectors(self.cell, self.N_local)
        self.p_samples = []
        self.structure_factors = []
        self.source_indices = []
        self.g_indices = []

        for ik, target_k in enumerate(target_internal):
            p_by_source = [target_k - source_k + Gv for source_k in source_internal]
            valid_indices = []
            candidate_indices = []
            for p_all in p_by_source:
                p_norm = np.linalg.norm(p_all, axis=1)
                valid = np.flatnonzero(p_norm > self.q_zero_tol)
                valid_indices.append(valid)
                local_mask = _whole_shell_mask(p_norm[valid], self.min_points)
                candidate_indices.append(valid[local_mask])

            if sum(len(indices) for indices in candidate_indices) < self.min_points:
                raise ValueError(
                    f"Fewer than {self.min_points} nonexcluded structure-factor samples are available; "
                    "increase N_local or sq_ke_cutoff"
                )

            pilot_p, pilot_S, _, _ = _sample_structure_factor(
                self.kmf,
                self.kpts_band[ik],
                self.mo_coeff_band[ik],
                Gv,
                candidate_indices,
                self.occupied_indices,
                self.occupied_indices,
                self.block_size,
            )
            pilot_norms = np.linalg.norm(pilot_p, axis=1)
            pilot_mask = _whole_shell_mask(pilot_norms, self.min_points)
            target_sigmas = np.zeros(self.nocc)
            target_diagnostics = []
            target_params = []
            pilot_infos = []
            fit_radii = np.zeros(self.nocc)
            all_norms = np.concatenate([np.linalg.norm(p, axis=1)[valid]
                                        for p, valid in zip(p_by_source, valid_indices)])
            for n in range(self.nocc):
                if self.fixed_sigma is not None:
                    sigma = float(self.fixed_sigma)
                    pilot_info = {"fixed_sigma": True, "n_samples": int(np.count_nonzero(pilot_mask))}
                else:
                    sigma, pilot_info = _fit_gaussian(
                        pilot_p[pilot_mask], pilot_S[pilot_mask, n],
                        initial_guess=self.initial_guess,
                        fit_with_coul=self.fit_with_coul,
                    )
                pilot_infos.append(pilot_info)
                if not np.isfinite(sigma) or sigma <= 0:
                    raise ValueError("The Gaussian pilot fit did not produce a positive sigma")
                target_sigmas[n] = sigma
                if self.qG_norm_cutoff is not None:
                    fit_radius = float(self.qG_norm_cutoff)
                elif self.qG_norm_cutoff_sigma is None:
                    fit_radius = np.inf
                else:
                    fit_radius = float(self.qG_norm_cutoff_sigma) * sigma
                selected = all_norms <= fit_radius + 1e-12
                if np.count_nonzero(selected) < self.min_points:
                    minimum_shell = _whole_shell_mask(all_norms, self.min_points)
                    minimum_radius = float(np.max(all_norms[minimum_shell]))
                    fit_radius = max(fit_radius, minimum_radius)
                fit_radii[n] = fit_radius

            union_radius = float(np.max(fit_radii))
            final_indices = []
            for p_all, valid in zip(p_by_source, valid_indices):
                p_norm = np.linalg.norm(p_all, axis=1)
                final_indices.append(valid[p_norm[valid] <= union_radius + max(1e-12, union_radius * 1e-10)])
            p_samples, S_samples, source_ids, g_ids = _sample_structure_factor(
                self.kmf,
                self.kpts_band[ik],
                self.mo_coeff_band[ik],
                Gv,
                final_indices,
                self.occupied_indices,
                self.occupied_indices,
                self.block_size,
            )
            p_norms = np.linalg.norm(p_samples, axis=1)
            self.p_samples.append(p_samples)
            self.structure_factors.append(S_samples)
            self.source_indices.append(source_ids)
            self.g_indices.append(g_ids)

            for n in range(self.nocc):
                selected = p_norms <= fit_radii[n] + max(1e-12, fit_radii[n] * 1e-10)
                if self.fixed_sigma is None:
                    sigma, fit_info = _fit_gaussian(
                        p_samples[selected], S_samples[selected, n],
                        initial_guess=self.initial_guess,
                        fit_with_coul=self.fit_with_coul,
                    )
                else:
                    sigma = float(self.fixed_sigma)
                    residual = np.exp(-p_norms[selected]**2 / (2.0 * sigma**2)) - S_samples[selected, n]
                    fit_info = {
                        "parameters": np.asarray([1.0, sigma]),
                        "n_samples": int(np.count_nonzero(selected)),
                        "residual_norm": float(np.linalg.norm(residual)),
                        "rms_residual": float(np.sqrt(np.mean(residual**2))),
                    }
                fit_info.update({
                    "pilot_sigma": float(target_sigmas[n]),
                    "pilot_n_samples": int(np.count_nonzero(pilot_mask)),
                    "fit_radius": float(fit_radii[n]),
                    "pilot_diagnostics": pilot_infos[n],
                })
                target_sigmas[n] = sigma
                target_diagnostics.append(fit_info)
                target_params.append(fit_info["parameters"])
                integral[ik, n] = np.sqrt(2.0 / np.pi) * sigma

                q_values = []
                for source_k in source_internal:
                    q = target_k - source_k
                    p_aux = _gaussian_transfer_grid(self.cell, q, sigma)
                    p2 = np.einsum("gi,gi->g", p_aux, p_aux)
                    keep = p2 > self.q_zero_tol**2
                    if np.any(keep):
                        h = np.exp(-p2[keep] / (2.0 * sigma**2))
                        q_values.append(np.sum(4.0 * np.pi * h / p2[keep]))
                quadrature[ik, n] = np.sum(q_values) / (volume * nk)
            sigma_values[ik] = target_sigmas
            diagnostics.append(target_diagnostics)
            fit_parameters.append(target_params)

        self.sigma = sigma_values
        self.integral_term = integral
        self.quadrature_term = quadrature
        self.xi = quadrature - integral
        self.fit_diagnostics = diagnostics
        self.fit_parameters = fit_parameters

    def compute_correction(self):
        """Fit the occupied orbitals and return the full band correction."""
        if self._correction_ready:
            return self.correction
        self._build_baseline_bands()
        self._compute_auxiliary_terms()
        correction = np.zeros_like(self.mo_energy_band_baseline, dtype=float)
        correction[:, :self.nocc] = self.hybrid_fraction * self.xi
        self.correction = correction
        self.mo_energy_band_corrected = self.mo_energy_band_baseline + correction
        self._correction_ready = True
        return self.correction

    def get_bands(self):
        """Return corrected band energies and the unchanged band coefficients."""
        self.compute_correction()
        return self.mo_energy_band_corrected.copy(), self.mo_coeff_band.copy()


__all__ = [
    "BandsSS",
    "DEFAULT_Q_ZERO_TOL",
    "fourier_exchange_matrices",
    "orbital_resolved_structure_factor",
]
