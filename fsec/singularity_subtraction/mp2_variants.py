"""Correlation-method policies used by :mod:`fsec.singularity_subtraction.mp2ss`.

The structure-factor code works with the signed MP2 denominator

``d = eps_i + eps_j - eps_a - eps_b``.

For a gapped reference, the excitation gap is ``Delta = -d > 0``.  Keeping
that convention in one small policy object makes it much harder for the
different storage routes to disagree about a regularizer or a spin scale.
"""

from dataclasses import dataclass
import re
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class _VariantSpec:
    """Immutable metadata for a named MP2 variant."""

    name: str
    c_os: float
    c_ss: float
    regularizer: Optional[str] = None
    default_strength: Optional[float] = None


_VARIANT_SPECS = {
    "mp2": _VariantSpec("mp2", 1.0, 1.0),
    "sos-mp2": _VariantSpec("sos-mp2", 1.3, 0.0),
    "scs-mp2": _VariantSpec("scs-mp2", 1.2, 1.0 / 3.0),
    "scs-mi-mp2": _VariantSpec("scs-mi-mp2", 0.40, 1.29),
    "kappa-mp2": _VariantSpec(
        "kappa-mp2", 1.0, 1.0, "kappa", 1.1),
    "sigma-mp2": _VariantSpec(
        "sigma-mp2", 1.0, 1.0, "sigma", 0.7),
    "sigma2-mp2": _VariantSpec(
        "sigma2-mp2", 1.0, 1.0, "sigma2", 0.4),
}


def _method_key(value: str) -> str:
    """Return a forgiving, case-insensitive lookup key for a method name."""
    if not isinstance(value, str):
        raise TypeError("correlation_method must be a string")
    value = value.strip().lower()
    value = value.replace("κ", "kappa").replace("σ", "sigma")
    value = value.replace("²", "2")
    value = value.replace("–", "-").replace("—", "-")
    value = re.sub(r"[\s_]+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value


_ALIASES = {
    "mp2": "mp2",
    "rmp2": "mp2",
    "standard-mp2": "mp2",
    "sos": "sos-mp2",
    "sosmp2": "sos-mp2",
    "sos-mp2": "sos-mp2",
    "scaled-opposite-spin-mp2": "sos-mp2",
    "scaled-opposite-spin": "sos-mp2",
    "scs": "scs-mp2",
    "scsmp2": "scs-mp2",
    "scs-mp2": "scs-mp2",
    "spin-component-scaled-mp2": "scs-mp2",
    "spin-component-scaled": "scs-mp2",
    "scs-mi": "scs-mi-mp2",
    "scsmimp2": "scs-mi-mp2",
    "scs-mi-mp2": "scs-mi-mp2",
    "scsmi-mp2": "scs-mi-mp2",
    "spin-component-scaled-mi-mp2": "scs-mi-mp2",
    "kappa": "kappa-mp2",
    "kappamp2": "kappa-mp2",
    "kappa-mp2": "kappa-mp2",
    "k-mp2": "kappa-mp2",
    "sigma": "sigma-mp2",
    "sigmamp2": "sigma-mp2",
    "sigma-mp2": "sigma-mp2",
    "s-mp2": "sigma-mp2",
    "sigma2": "sigma2-mp2",
    "sigma-2": "sigma2-mp2",
    "sigma^2": "sigma2-mp2",
    "sigma^2-mp2": "sigma2-mp2",
    "sigma2mp2": "sigma2-mp2",
    "sigma2-mp2": "sigma2-mp2",
    "sigma-squared": "sigma2-mp2",
    "sigma-squared-mp2": "sigma2-mp2",
}


def normalize_correlation_method(value: str) -> str:
    """Normalize a method name and validate that it is supported."""
    key = _method_key(value)
    try:
        return _ALIASES[key]
    except KeyError as error:
        choices = ", ".join(sorted(_VARIANT_SPECS))
        raise ValueError(
            f"Unsupported correlation_method {value!r}; choose one of {choices}"
        ) from error


def _validate_strength(spec: _VariantSpec, strength) -> Optional[float]:
    if spec.regularizer is None:
        if strength is not None:
            raise ValueError(
                f"regularization_strength is only valid for regularized methods; "
                f"{spec.name!r} does not use a denominator regularizer"
            )
        return None

    if strength is None:
        strength = spec.default_strength
    try:
        strength = float(strength)
    except (TypeError, ValueError) as error:
        raise ValueError("regularization_strength must be a finite non-negative number") from error
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("regularization_strength must be a finite non-negative number")
    return strength


@dataclass(frozen=True)
class MP2Variant:
    """Resolved, immutable MP2 method policy.

    ``direct_scale`` and ``exchange_scale`` apply to the repository's
    decomposition ``D = 2 E_OS`` and ``X = E_SS - E_OS``.  Thus the final
    spin-component energy is ``c_os * E_OS + c_ss * E_SS``.

    The methods :meth:`damping`, :meth:`reciprocal`, and
    :meth:`absolute_reciprocal` accept an explicit ``active_mask``.  Padded
    PySCF denominator entries must be masked before evaluating an exponential;
    this is especially important for the ``LARGE_DENOM`` padding value.
    """

    name: str
    c_os: float
    c_ss: float
    regularizer: Optional[str] = None
    regularization_strength: Optional[float] = None

    @property
    def cOS(self) -> float:
        """Opposite-spin scale using the conventional MP2 notation."""
        return self.c_os

    @property
    def cSS(self) -> float:
        """Same-spin scale using the conventional MP2 notation."""
        return self.c_ss

    @property
    def direct_scale(self) -> float:
        """Scale for the accumulated direct structure factor."""
        return (self.c_os + self.c_ss) / 2.0

    @property
    def exchange_scale(self) -> float:
        """Scale for the accumulated exchange structure factor."""
        return self.c_ss

    @property
    def is_regularized(self) -> bool:
        return self.regularizer is not None

    @property
    def supports_laplace(self) -> bool:
        """Whether the regularized reciprocal has a separable shift form."""
        return self.regularizer in (None, "kappa", "sigma")

    def _active_values(self, values, active_mask):
        values = np.asarray(values)
        if active_mask is None:
            active_mask = np.ones(values.shape, dtype=bool)
        else:
            active_mask = np.asarray(active_mask, dtype=bool)
            if active_mask.shape != values.shape:
                raise ValueError(
                    "active_mask must have the same shape as the denominator"
                )
        return values, active_mask, values[active_mask]

    def _damping_from_gaps(self, gaps):
        """Evaluate the damping on already-selected active gaps only."""
        gaps = np.asarray(gaps)
        if np.any(~np.isfinite(gaps)) or np.any(gaps < 0):
            raise ValueError("active excitation gaps must be finite and non-negative")
        if self.regularizer is None:
            return np.ones(gaps.shape, dtype=float)

        strength = self.regularization_strength
        x = strength * gaps
        one_minus_exp = -np.expm1(-x)
        if self.regularizer == "kappa":
            return one_minus_exp * one_minus_exp
        if self.regularizer == "sigma":
            return one_minus_exp
        if self.regularizer == "sigma2":
            return -np.expm1(-strength * gaps * gaps)
        raise RuntimeError(f"Unknown MP2 regularizer {self.regularizer!r}")

    def damping_from_gap(self, gap, active_mask=None):
        """Return the amplitude damping ``f(Delta)``.

        Inactive entries are returned as zero.  This convention lets callers
        safely multiply padded blocks without ever exponentiating padding.
        """
        gap, active_mask, active_gap = self._active_values(gap, active_mask)
        result = np.zeros(gap.shape, dtype=float)
        if active_gap.size:
            result[active_mask] = self._damping_from_gaps(active_gap)
        return result

    def damping(self, signed_denominator, active_mask=None):
        """Return ``f(-d)`` for the signed denominator ``d``."""
        denominator, active_mask, active_denominator = self._active_values(
            signed_denominator, active_mask)
        result = np.zeros(denominator.shape, dtype=float)
        if active_denominator.size:
            result[active_mask] = self._damping_from_gaps(-active_denominator)
        return result

    def reciprocal_from_gap(self, gap, active_mask=None):
        """Return the signed regularized reciprocal ``-f(Delta)/Delta``."""
        gap, active_mask, active_gap = self._active_values(gap, active_mask)
        result = np.zeros(gap.shape, dtype=np.result_type(gap, float))
        if not active_gap.size:
            return result

        active_gap = np.asarray(active_gap)
        if np.any(~np.isfinite(active_gap)) or np.any(active_gap < 0):
            raise ValueError("active excitation gaps must be finite and non-negative")
        if self.regularizer is None:
            with np.errstate(divide="ignore", invalid="ignore"):
                active_result = -1.0 / active_gap
        else:
            damping = self._damping_from_gaps(active_gap)
            active_result = np.zeros_like(active_gap, dtype=np.result_type(active_gap, float))
            nonzero = active_gap != 0
            with np.errstate(divide="ignore", invalid="ignore"):
                np.divide(-damping, active_gap, out=active_result, where=nonzero)
            zero = ~nonzero
            if np.any(zero):
                if self.regularizer == "sigma":
                    active_result[zero] = -self.regularization_strength
                else:
                    active_result[zero] = 0.0
        result[active_mask] = active_result
        return result

    def reciprocal(self, signed_denominator, active_mask=None):
        """Return the signed regularized reciprocal for ``d = -Delta``."""
        denominator, active_mask, active_denominator = self._active_values(
            signed_denominator, active_mask)
        result = np.zeros(denominator.shape, dtype=np.result_type(denominator, float))
        if active_denominator.size:
            result[active_mask] = self.reciprocal_from_gap(-active_denominator)
        return result

    def absolute_reciprocal_from_gap(self, gap, active_mask=None):
        """Return ``|f(Delta)/d|`` with the same zero-gap limits."""
        return -self.reciprocal_from_gap(gap, active_mask=active_mask)

    def absolute_reciprocal(self, signed_denominator, active_mask=None):
        """Return the magnitude of :meth:`reciprocal`."""
        return -self.reciprocal(signed_denominator, active_mask=active_mask)

    def laplace_terms(self, absolute=False) -> Optional[Tuple[Tuple[float, float], ...]]:
        """Return separable ``(shift, coefficient)`` terms for Laplace use.

        The terms represent ``sum(coefficient * exp(-shift*Delta) / Delta)``.
        Signed terms describe :meth:`reciprocal`; absolute terms describe
        :meth:`absolute_reciprocal`.  ``None`` marks sigma-squared, whose
        quadratic total-gap damping is not separable.
        """
        sign = 1.0 if absolute else -1.0
        if self.regularizer is None:
            return ((0.0, sign),)
        strength = self.regularization_strength
        if self.regularizer == "kappa":
            if strength == 0:
                return tuple()
            coefficients = (1.0, -2.0, 1.0) if absolute else (-1.0, 2.0, -1.0)
            return (
                (0.0, coefficients[0]),
                (strength, coefficients[1]),
                (2.0 * strength, coefficients[2]),
            )
        if self.regularizer == "sigma":
            if strength == 0:
                return tuple()
            coefficients = (1.0, -1.0) if absolute else (-1.0, 1.0)
            return ((0.0, coefficients[0]), (strength, coefficients[1]))
        if self.regularizer == "sigma2":
            return None
        raise RuntimeError(f"Unknown MP2 regularizer {self.regularizer!r}")


def get_mp2_variant(
        correlation_method: str = "mp2",
        regularization_strength=None,
) -> MP2Variant:
    """Resolve a user-facing method name and regularization strength."""
    name = normalize_correlation_method(correlation_method)
    spec = _VARIANT_SPECS[name]
    strength = _validate_strength(spec, regularization_strength)
    return MP2Variant(
        name=spec.name,
        c_os=spec.c_os,
        c_ss=spec.c_ss,
        regularizer=spec.regularizer,
        regularization_strength=strength,
    )


# A descriptive alias makes the intended public role obvious to callers that
# prefer the word ``method`` over ``variant``.
resolve_mp2_method = get_mp2_variant


__all__ = [
    "MP2Variant",
    "get_mp2_variant",
    "resolve_mp2_method",
    "normalize_correlation_method",
]
