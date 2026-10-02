from abc import ABC, abstractmethod

import numpy as np


class ModelFunction(ABC):
    _registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        ModelFunction._registry[cls.__name__] = cls

    @classmethod
    def get_class(cls, name):
        if name not in cls._registry:
            raise ValueError(
                f"Unknown model function: {name}. "
                f"Available: {sorted(cls._registry.keys())}"
            )
        return cls._registry[name]

    def __init__(self, parameters=None, is_contraction=False, **kwargs):
        self.parameters = parameters if parameters is not None else self.default_parameters()
        self.is_contraction = is_contraction
        self.num_params = len(self.parameters)

    def set_parameters(self, parameters):
        self.parameters = parameters

    @abstractmethod
    def eval_model(self, coords: np.ndarray):
        """
        Evaluate the model at given coordinates.
        """

    @abstractmethod
    def coulomb_integral(self):
        """
        Compute the integral of the model times the Coulomb kernel over all space. Normalization follows
        pyscf.pbc.gto.cell.ewald() i.e. 1/(2*np.pi)**3 * 4*np.pi * int_{R^3} f(x,y,z)/(x^2+y^2+z^2) dx dy dz
        """

    @abstractmethod
    def default_parameters(self):
        """
        Return default parameters for the model.
        """


class IsotropicModel(ModelFunction):
    """Adds some default functions specific to isotropic models."""

    def __init__(self, parameters, num_params, is_contraction=False):
        self.num_params = num_params
        self.isotropic = True
        self.parameters = parameters if parameters is not None else self.default_parameters()
        self.is_contraction = is_contraction

    @abstractmethod
    def eval_model_r(self, r: np.ndarray):
        pass

    def default_parameters(self):
        self.parameters = [1.0] * self.num_params
        return self.parameters

    def eval_model(self, coords: np.ndarray):
        return self.eval_model_r(np.linalg.norm(coords, axis=1))

    def coulomb_integral(self):
        """Compute the coulomb integral as a 1d scipy.integrate.quad integral."""
        from scipy.integrate import quad

        def radial_eval(r):
            # SciPy quad requires a scalar return value. Some radial model
            # implementations return a length-1 ndarray for scalar input, so
            # normalize the output here.
            return np.asarray(self.eval_model_r(r)).item()

        integral, abserror = quad(radial_eval, 0, np.inf)
        prefactor = (4 * np.pi) * 2 * 2 * np.pi / (2 * np.pi) ** 3
        print("Computed integral with scipy.integrate.quad. Estimated error: ", abserror)
        return prefactor * integral


from fsec.singularity_subtraction.model_function.exx_modfunc import ContractedGaussianModel, ExponentialModel, ExpoLorentzianModel, QuarticExponentialModel
from fsec.singularity_subtraction.model_function.mp2_direct_modfunc import (
    PolynomialModel,
    XNExpAbs,
    XNExpAbs2,
    XNExpAbsStackedSingularity,
    XNExpAbsStackedSingularityQMesh,
    XNExponential,
    XNGauss,
    XNGaussStackedSingularity,
    XNGaussStackedSingularityQMesh,
    XNGaussVcut,
    XNGeneral,
    XNQuarticExponential,
)
from fsec.singularity_subtraction.model_function.mp2_exchange_modfunc import (
    MP2StackedSingularityExchange,
    XNExpAbs2StackedSingularityExchange,
    XNExpAbsStackedSingularityExchange,
    XNExponentialStackedSingularityExchange,
    XNGaussStackedSingularityExchange,
    XNQuarticExponentialStackedSingularityExchange,
)
