import numpy as np
from scipy.integrate import quad

from fsec.singularity_subtraction.model_function import ModelFunction
from fsec.singularity_subtraction.model_function.etb import EvenTemperedMixin


class MP2StackedSingularityExchange(ModelFunction):
    """Auxiliary model for MP2 exchange correction."""

    def __init__(self, parameters=None, q2s=None, dvol=None):
        super().__init__(parameters=parameters, is_contraction=False)
        self.q2s = None if q2s is None else np.asarray(q2s, dtype=float)
        self.dvol = dvol
        self.uncorrected_q2_quad = None
        if self.q2s is not None:
            self.compute_uncorrected_q2_quad()

    def decay_func_r(self, r: np.ndarray):
        pass

    def default_parameters(self):
        pass

    def set_parameters(self, parameters):
        self.parameters = parameters
        if self.q2s is not None:
            self.compute_uncorrected_q2_quad()

    def set_q2s(self, q2s, dvol=None):
        self.q2s = np.asarray(q2s, dtype=float)
        if dvol is not None:
            self.dvol = dvol
        self.compute_uncorrected_q2_quad()

    def compute_uncorrected_q2_quad(self):
        squared_norm_q2s = np.sum(self.q2s**2, axis=1)
        squared_norm_q2s[squared_norm_q2s < 1e-8] = np.inf
        uncorrected_q2_quad = self.dvol * np.sum(self.decay_func_r(np.sqrt(squared_norm_q2s)))
        self.uncorrected_q2_quad = uncorrected_q2_quad
        return uncorrected_q2_quad

    def eval_model(self, coords: np.ndarray):
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("coords must be an (N,3) array: [q_x, q_y, q_z]")
        if self.q2s is None:
            raise ValueError("q2s must be set on MP2StackedSingularityExchange before evaluation")
        if self.q2s.ndim != 2 or self.q2s.shape[1] != 3:
            raise ValueError("q2s must be an (M,3) array")
        if self.dvol is None:
            raise ValueError("dvol must be set on MP2StackedSingularityExchange before evaluation")
        if self.c0 is None:
            raise ValueError("c0 must be set on MP2StackedSingularityExchange before evaluation")

        c0 = self.c0

        uncorrected_q2_quad = self.uncorrected_q2_quad

        q_norm = np.linalg.norm(coords, axis=1)
        q_norm_squared = q_norm**2

        decay_func_r = self.decay_func_r(q_norm)
        results = c0 * q_norm_squared * decay_func_r * uncorrected_q2_quad

        return results

    def coulomb_integral(self):
        """Compute the coulomb integral as a 1d scipy.integrate.quad integral."""

        def integrand(r):
            return r**2 * self.decay_func_r(r)

        integral, abserror = quad(integrand, 0, np.inf)
        prefactor = 4 * np.pi
        print("Computed integral with scipy.integrate.quad. Estimated error: ", abserror)
        return self.c0 * (prefactor * integral) ** 2


class XNGaussStackedSingularityExchange(MP2StackedSingularityExchange):
    def __init__(self, parameters=None, q2s=None, dvol=None):
        self.sigma = parameters[1]
        self.c0 = parameters[0]
        super().__init__(parameters=parameters, q2s=q2s, dvol=dvol)

    def default_parameters(self):
        return [1.0, 1.0]

    def set_parameters(self, parameters):
        self.sigma = parameters[1]
        self.c0 = parameters[0]
        super().set_parameters(parameters)

    def decay_func_r(self, r: np.ndarray):
        if np.isclose(self.sigma, 0):
            return np.zeros_like(r)
        return np.exp(-r**2 / (2 * self.sigma**2))

    def coulomb_integral(self):
        c0, sigma = self.c0, self.sigma
        if self.c0 is None or self.sigma is None:
            raise ValueError("c0 and sigma must be set on XNGaussStackedSingularityExchange before evaluation")
        if np.isclose(sigma, 0):
            return 0.0
        return c0 * (2 * sigma**2 * np.pi) ** 3


class XNExponentialStackedSingularityExchange(MP2StackedSingularityExchange):
    def __init__(self, parameters=None, q2s=None, dvol=None):
        self.c0 = parameters[0]
        self.alpha = parameters[1]
        self.gamma = parameters[2]
        super().__init__(parameters=parameters, q2s=q2s, dvol=dvol)

    def default_parameters(self):
        return [1.0, 1.0, 1.0]

    def set_parameters(self, parameters):
        self.alpha = parameters[1]
        self.gamma = parameters[2]
        self.c0 = parameters[0]
        super().set_parameters(parameters)

    def decay_func_r(self, r: np.ndarray):
        if np.isclose(self.alpha, 0):
            return np.zeros_like(r)
        return np.exp(-self.alpha * (np.sqrt(1 + self.gamma * r**2) - 1))


class XNExpAbsStackedSingularityExchange(MP2StackedSingularityExchange):
    def __init__(self, parameters=None, q2s=None, dvol=None):
        self.c0 = parameters[0]
        self.alpha = parameters[1]
        super().__init__(parameters=parameters, q2s=q2s, dvol=dvol)

    def default_parameters(self):
        return [1.0, 1.0]

    def set_parameters(self, parameters):
        self.alpha = parameters[1]
        self.c0 = parameters[0]
        super().set_parameters(parameters)

    def decay_func_r(self, r: np.ndarray):
        if np.isclose(self.alpha, 0):
            return np.zeros_like(r)
        return np.exp(-self.alpha * r)

    def coulomb_integral(self):
        c0, alpha = self.c0, self.alpha
        if self.c0 is None or self.alpha is None:
            raise ValueError("c0 and alpha must be set on XNExpAbsStackedSingularityExchange before evaluation")
        if np.isclose(alpha, 0):
            return 0.0
        return c0 * (8 * np.pi / (alpha**3)) ** 2


class XNExpAbs2StackedSingularityExchange(MP2StackedSingularityExchange):
    def __init__(self, parameters=None, q2s=None, dvol=None):
        self.c0 = parameters[0]
        self.alpha = parameters[1]
        self.beta = parameters[2]
        super().__init__(parameters=parameters, q2s=q2s, dvol=dvol)

    def default_parameters(self):
        return [1.0, 1.0, 1.0]

    def set_parameters(self, parameters):
        self.alpha = parameters[1]
        self.beta = parameters[2]
        self.c0 = parameters[0]
        super().set_parameters(parameters)

    def decay_func_r(self, r: np.ndarray):
        if np.isclose(self.alpha, 0):
            return np.zeros_like(r)
        return np.exp(-self.alpha * r - self.beta * r**2)


class XNQuarticExponentialStackedSingularityExchange(MP2StackedSingularityExchange):
    def __init__(self, parameters=None, q2s=None, dvol=None):
        self.c0 = parameters[0]
        self.alpha = parameters[1]
        self.beta = parameters[2]
        self.kappa = parameters[3]
        super().__init__(parameters=parameters, q2s=q2s, dvol=dvol)

    def default_parameters(self):
        return [1.0, 1.0, 1.0, 1.0]

    def set_parameters(self, parameters):
        self.alpha = parameters[1]
        self.beta = parameters[2]
        self.kappa = parameters[3]
        self.c0 = parameters[0]
        super().set_parameters(parameters)

    def decay_func_r(self, r: np.ndarray):
        if np.isclose(self.alpha, 0):
            return np.zeros_like(r)
        gamma1 = self.beta / self.alpha
        gamma2 = self.kappa / self.alpha
        return np.exp(-self.alpha * (np.sqrt(1 + gamma1 * r**2 + gamma2 * r**4) - 1))


class XNETBStackedSingularityExchange(EvenTemperedMixin, MP2StackedSingularityExchange):
    """Exchange model with an even-tempered Gaussian decay function.

    f(q) = c0 |q|^2 h(q) dvol sum_{q2} h(q2) = |q|^2 sum_k beta_k exp(-alpha_k |q|^2),
    with h(r) = sum_k w_k exp(-alpha_k r^2) normalized to h(0) = 1.

    Parameter order:
        [beta_0, ..., beta_{N-1}] in order of ascending exponent, so the model is linear in the parameters.
        The normalized weights w = beta / sum(beta) and c0 = sum(beta) / (dvol sum_{q2} h(q2)) are derived.
    """

    def __init__(self, parameters=None, q2s=None, dvol=None, sigma=1.0, num_gaussians=5, ratio=2.0, exponents=None):
        self._init_etb(sigma, num_gaussians, ratio, exponents)
        self._etb_ctor_kwargs = {'q2s': q2s, 'dvol': dvol}
        if parameters is not None:
            parameters = np.asarray(parameters, dtype=float)
        super().__init__(parameters=parameters, q2s=q2s, dvol=dvol)
        self.parameters = np.asarray(self.parameters, dtype=float)

    def default_parameters(self):
        return self.one_hot_reference()

    def initial_parameters(self, c0_scale):
        """Reference Gaussian with c0 = c0_scale."""
        self.set_parameters(self.one_hot_reference())
        return self.initial_coefficients(c0_scale * self.uncorrected_q2_quad)

    def set_parameters(self, parameters):
        super().set_parameters(np.asarray(parameters, dtype=float))

    @property
    def etb_weights(self):
        total = np.sum(self.parameters)
        if np.isclose(total, 0):
            return np.zeros_like(self.parameters)
        return self.parameters / total

    @property
    def c0(self):
        if self.uncorrected_q2_quad is None:
            return None
        if np.isclose(np.sum(self.parameters), 0):
            return 0.0
        if np.isclose(self.uncorrected_q2_quad, 0):
            raise ValueError("The q2 quadrature of the ETB decay function vanishes; c0 is undefined")
        return float(np.sum(self.parameters) / self.uncorrected_q2_quad)

    def decay_func_r(self, r: np.ndarray):
        return self.etb_h(r, self.etb_weights)

    def eval_model(self, coords: np.ndarray):
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("coords must be an (N,3) array: [q_x, q_y, q_z]")
        r = np.linalg.norm(coords, axis=1)
        return r**2 * self.etb_h(r)

    def linear_design_matrix(self, coords):
        r = np.linalg.norm(np.asarray(coords, dtype=float), axis=1)
        return (r**2)[:, None] * self.gaussian_columns(r)

    def coulomb_integral(self):
        if self.c0 is None:
            raise ValueError("q2s and dvol must be set on XNETBStackedSingularityExchange before evaluation")
        return self.c0 * float(np.dot(self.etb_weights, self._volume_moments())) ** 2
