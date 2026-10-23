"""Standard (single-mesh) periodic methods with alternative finite-size corrections.

``KRGWAC_ewald`` is PySCF's restricted G0W0-AC with Gaussian auxiliary-function
or Madelung q -> 0 corrections in place of spherical integration.
"""

from .krgwac_ewald import KRGWAC_ewald

__all__ = ['KRGWAC_ewald']
