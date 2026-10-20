"""Even-tempered Gaussian basis (ETB) helpers shared by the ETB auxiliary functions.

An ETB decay function is a nonnegative linear combination of Gaussians with fixed exponents

    h(r) = sum_k b_k exp(-alpha_k r^2),   alpha_k = alpha_sigma * ratio**(k - (N-1)/2),

centered (in log space) on alpha_sigma = 1/(2 sigma^2), the exponent of a Gaussian with the expected
width sigma. Because the exponents are fixed, the ETB parameters are the linear coefficients b_k.
"""

import numpy as np
import scipy.special

SHELL_TOL = 1e-8
DEFAULT_RCOND = 1e-8
DEFAULT_MIN_SHELL_MARGIN = 1


def even_tempered_exponents(sigma=1.0, num_gaussians=5, ratio=2.0):
    """Return N even-tempered exponents (ascending) centered on alpha_sigma = 1/(2 sigma^2)."""
    sigma = float(sigma)
    num_gaussians = int(num_gaussians)
    ratio = float(ratio)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if num_gaussians < 1:
        raise ValueError("num_gaussians must be at least 1")
    if not np.isfinite(ratio) or ratio <= 1:
        raise ValueError("ratio must be finite and greater than 1")

    alpha_sigma = 1.0 / (2.0 * sigma**2)
    k = np.arange(num_gaussians) - (num_gaussians - 1) / 2
    return alpha_sigma * ratio**k


class EvenTemperedMixin:
    """Shared ETB machinery. ``self.parameters`` holds the coefficients b_k (ascending exponent)."""

    def _init_etb(self, sigma=1.0, num_gaussians=5, ratio=2.0, exponents=None):
        if exponents is None:
            exponents = even_tempered_exponents(sigma, num_gaussians, ratio)
        else:
            exponents = np.sort(np.asarray(exponents, dtype=float).ravel())
            if exponents.size == 0 or np.any(~np.isfinite(exponents)) or np.any(exponents <= 0):
                raise ValueError("exponents must be a nonempty array of finite positive values")
        if not np.isfinite(float(sigma)) or float(sigma) <= 0:
            raise ValueError("sigma must be finite and positive")

        self.etb_sigma = float(sigma)
        self.etb_ratio = float(ratio)
        self.etb_exponents = exponents
        self.num_gaussians = len(exponents)
        alpha_sigma = 1.0 / (2.0 * self.etb_sigma**2)
        self.etb_ref_index = int(np.argmin(np.abs(np.log(exponents) - np.log(alpha_sigma))))

    def one_hot_reference(self, scale=1.0):
        """Coefficients selecting only the Gaussian closest to the expected width sigma."""
        coeffs = np.zeros(self.num_gaussians)
        coeffs[self.etb_ref_index] = scale
        return coeffs

    def initial_coefficients(self, scale=1.0, floor=1e-3):
        """Reference Gaussian with weight scale; the others get floor * scale to start inside positive bounds."""
        coeffs = np.full(self.num_gaussians, floor * scale)
        coeffs[self.etb_ref_index] = scale
        return coeffs

    def gaussian_columns(self, r):
        """Return exp(-alpha_k r^2) with shape r.shape + (N,)."""
        r = np.asarray(r, dtype=float)
        return np.exp(-np.multiply.outer(r**2, self.etb_exponents))

    def etb_h(self, r, coeffs=None):
        """Evaluate sum_k b_k exp(-alpha_k r^2) for any shape of r."""
        coeffs = np.asarray(self.parameters if coeffs is None else coeffs, dtype=float)
        r2 = np.asarray(r, dtype=float) ** 2
        result = np.zeros(r2.shape)
        for b_k, alpha_k in zip(coeffs, self.etb_exponents):
            if b_k != 0.0:
                result += b_k * np.exp(-alpha_k * r2)
        return result

    def _radial_moments(self, p):
        """int_0^inf r^p exp(-alpha_k r^2) dr for each primitive."""
        return scipy.special.gamma((p + 1) / 2) / (2 * self.etb_exponents ** ((p + 1) / 2))

    def _volume_moments(self):
        """int_{R^3} exp(-alpha_k q^2) d^3q for each primitive."""
        return (np.pi / self.etb_exponents) ** 1.5

    def _pair_overlap_sum(self, coeffs, deltaG_norms):
        """sum_dG sum_kl b_k b_l int exp(-alpha_k q^2) exp(-alpha_l |q+dG|^2) d^3q."""
        coeffs = np.asarray(coeffs, dtype=float)
        dG2 = np.asarray(deltaG_norms, dtype=float) ** 2
        total = 0.0
        for k, (b_k, a_k) in enumerate(zip(coeffs, self.etb_exponents)):
            for b_l, a_l in zip(coeffs[k:], self.etb_exponents[k:]):
                if b_k == 0.0 or b_l == 0.0:
                    continue
                s = a_k + a_l
                term = b_k * b_l * (np.pi / s) ** 1.5 * np.sum(np.exp(-a_k * a_l / s * dG2))
                total += term if a_k == a_l else 2 * term
        return total

    def with_exponents(self, exponents):
        """Rebuild this model with a subset of its exponents (default coefficients)."""
        kwargs = dict(self._etb_ctor_kwargs)
        kwargs.update(sigma=self.etb_sigma, ratio=self.etb_ratio,
                      exponents=np.asarray(exponents, dtype=float))
        return type(self)(**kwargs)


def _radii(coords):
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 1:
        return np.abs(coords)
    if coords.ndim == 2 and coords.shape[1] == 1:
        return np.abs(coords[:, 0])
    if coords.ndim == 2 and coords.shape[1] == 3:
        return np.linalg.norm(coords, axis=1)
    raise ValueError(f"coords must have shape (N,), (N, 1), or (N, 3). Got {coords.shape}.")


def radial_shells(coords, tol=SHELL_TOL):
    """Distinct nonzero |q| values (ascending). Isotropic models cannot resolve more than this."""
    r = np.sort(_radii(coords))
    r = r[r > tol]
    if r.size == 0:
        return r
    keep = np.concatenate(([True], np.diff(r) > tol))
    return r[keep]


def cutoff_for_min_shells(coords, n_shells, tol=SHELL_TOL):
    """Smallest strict-inequality |q| cutoff containing n_shells nonzero shells, or None."""
    shells = radial_shells(coords, tol)
    if n_shells < 1:
        return None
    if len(shells) < n_shells:
        return None
    return shells[n_shells - 1] + tol


def numerical_rank(A, rcond=DEFAULT_RCOND):
    """Rank of A after scaling each column to unit norm."""
    A = np.asarray(A, dtype=float)
    if A.size == 0:
        return 0
    col_norms = np.linalg.norm(A, axis=0)
    nonzero = col_norms > 0
    if not np.any(nonzero):
        return 0
    s = np.linalg.svd(A[:, nonzero] / col_norms[nonzero], compute_uv=False)
    return int(np.sum(s > rcond * s[0]))


def etb_windows(num_gaussians, ref_index):
    """Index windows centered on ref_index, from the full basis down to the reference alone."""
    windows = []
    for half_width in range(max(ref_index, num_gaussians - 1 - ref_index), -1, -1):
        idx = np.arange(max(0, ref_index - half_width), min(num_gaussians, ref_index + half_width + 1))
        if not windows or len(idx) != len(windows[-1]):
            windows.append(idx)
    return windows


def select_etb_window(A, ref_index, constraint_rows=None, rcond=DEFAULT_RCOND,
                      n_shells=None, min_shell_margin=DEFAULT_MIN_SHELL_MARGIN):
    """Largest window of columns around ref_index that the samples can determine.

    A window is accepted when its (weighted) design matrix, stacked with any constraint rows, has full
    numerical column rank and, if n_shells is given, the number of distinct nonzero |q| shells exceeds
    the window's degrees of freedom by at least min_shell_margin.
    """
    A = np.asarray(A, dtype=float)
    n_constraints = 0 if constraint_rows is None else np.atleast_2d(constraint_rows).shape[0]
    for idx in etb_windows(A.shape[1], ref_index):
        sub = A[:, idx]
        if constraint_rows is not None:
            sub = np.vstack([sub, np.atleast_2d(constraint_rows)[:, idx]])
        dof = len(idx) - n_constraints
        if n_shells is not None and dof > 0 and n_shells < dof + min_shell_margin:
            continue
        if numerical_rank(sub, rcond) == len(idx):
            return idx
    return None


def trim_etb_model_to_data(model, coords, weights=None, rcond=DEFAULT_RCOND,
                           min_shell_margin=DEFAULT_MIN_SHELL_MARGIN, label="ETB"):
    """Return model, or a rebuilt model with fewer exponents if the fit data cannot determine all of them."""
    if hasattr(model, "linear_design_matrix"):
        A = model.linear_design_matrix(coords)
    else:
        A = model.linear_proxy_design_matrix(coords)
    if weights is not None:
        A = np.asarray(weights, dtype=float)[:, None] * A

    constraint_rows = None
    if getattr(model, "linear_sum_constraint", None) is not None:
        constraint_rows = np.ones((1, model.num_gaussians))

    n_shells = len(radial_shells(coords))
    idx = select_etb_window(A, model.etb_ref_index, constraint_rows=constraint_rows, rcond=rcond,
                            n_shells=n_shells, min_shell_margin=min_shell_margin)
    if idx is None:
        raise ValueError(
            f"{label}: the fit data ({n_shells} distinct nonzero |q| shells) cannot determine even the "
            "reference Gaussian. Increase qG_norm_cutoff or reduce num_gaussians."
        )
    if len(idx) == model.num_gaussians:
        return model

    print(f"WARNING: {label} reduced from {model.num_gaussians} to {len(idx)} primitives: only "
          f"{n_shells} distinct nonzero |q| shells within the fit region. Increase qG_norm_cutoff to "
          "use the full basis.")
    return model.with_exponents(model.etb_exponents[idx])
