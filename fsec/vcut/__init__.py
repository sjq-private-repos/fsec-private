"""Methods using smoothed truncated Coulomb interactions."""

from importlib import import_module

from .kmp2_stc import KMP2, KMP2_HYBRID, KMP2_STC

__all__ = [
    "KMP2", "KMP2_STC", "KMP2_HYBRID",
    "KMP2_STC_DIRECT", "KMP2_HYBRID_DIRECT",
]


def __getattr__(name):
    """Load optional direct-RSDF calculators only when explicitly requested."""
    if name in {"KMP2_STC_DIRECT", "KMP2_HYBRID_DIRECT"}:
        module = import_module(".kmp2_stc_direct", __name__)
        return getattr(module, name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
