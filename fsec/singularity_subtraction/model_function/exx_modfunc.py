import numpy as np
import scipy
from scipy.integrate import quad

from fsec.singularity_subtraction.model_function import IsotropicModel, ModelFunction
from fsec.singularity_subtraction.model_function.etb import EvenTemperedMixin, _radii


class ContractedGaussianModel(ModelFunction):
    """Linear combination of Gaussians.

    Parameter order:
        [c_i, sigma_x, sigma_y, sigma_z]* num_gaussians if anisotropic
        [c_i, sigma]* num_gaussians if isotropic
    """

    def __init__(self, num_gaussians, isotropic, parameters=None):
        self.num_primitives = num_gaussians
        self.isotropic = isotropic
        self.parameters = parameters if parameters is not None else self.default_parameters()
        self.num_primitive_params = 4 if isotropic else 2

        super().__init__(parameters, is_contraction=True)

    def default_parameters(self):
        if self.isotropic:
            parameters = [1.0 / self.num_primitives, 1] * self.num_primitives
        else:
            parameters = [1.0 / self.num_primitives, 1, 1, 1] * self.num_primitives
        return parameters

    def contracted_gaussian_model_centered(self, params, xyz, num_gaussians=1, isotropic=True):
        num_gauss_params = self.num_primitive_params

        def gaussian_model(params, xyz):
            sigma_x, sigma_y, sigma_z = params
            exponent = -(
                (xyz[:, 0]) ** 2 / (2 * sigma_x**2)
                + (xyz[:, 1]) ** 2 / (2 * sigma_y**2)
                + (xyz[:, 2]) ** 2 / (2 * sigma_z**2)
            )
            return np.exp(exponent)

        result = np.zeros(xyz.shape[0])

        if isotropic:
            for i in range(num_gaussians):
                c_i, sigma = params[i * num_gauss_params : (i + 1) * num_gauss_params]
                result += c_i * gaussian_model([sigma, sigma, sigma], xyz)
        else:
            for i in range(num_gaussians):
                c_i, sigma_x, sigma_y, sigma_z = params[
                    i * num_gauss_params : (i + 1) * num_gauss_params
                ]
                result += c_i * gaussian_model([sigma_x, sigma_y, sigma_z], xyz)

        return result

    def eval_model(self, coords: np.ndarray):
        return self.contracted_gaussian_model_centered(
            self.parameters, coords, self.num_primitives, self.isotropic
        )

    def coulomb_integral(self, parameters: list = None):
        parameters = self.parameters if parameters is None else parameters
        assert self.isotropic, "Anisotropic Gaussian Coulomb integral not implemented"
        out = 0.0
        for i in range(self.num_primitives):
            c_i, sigma = parameters[
                i * self.num_primitive_params : (i + 1) * self.num_primitive_params
            ]
            ew_eta = 1.0 / np.sqrt(2.0) * sigma
            out += c_i * 2 * np.mean(ew_eta) / np.sqrt(np.pi)

        return out


class ExponentialModel(ModelFunction):
    """Exponential.

    Parameter order:
        [c_i, alpha_i, gamma_i]* num_primitives
    """

    def __init__(self, num_primitives, isotropic, parameters=None):
        self.num_primitives = num_primitives
        self.isotropic = isotropic
        self.parameters = parameters if parameters is not None else self.default_parameters()
        self.num_primitive_params = 3
        self.num_params = self.num_primitive_params * self.num_primitives
        super().__init__(parameters, is_contraction=True)

    def default_parameters(self):
        parameters = [1.0 / self.num_primitives, 0.8, 2.0] * self.num_primitives
        return parameters

    def exponential_model_centered(self, params, xyz, num_primitives=1):
        num_params = 3

        def exponential_model(params, xyz):
            alpha, gamma = params
            exponent = -alpha * (np.sqrt(1 + gamma * np.sum(xyz**2, axis=1)) - 1)
            return np.exp(exponent)

        result = np.zeros(xyz.shape[0])

        for i in range(num_primitives):
            c_i, alpha, gamma = params[i * num_params : (i + 1) * num_params]
            result += c_i * exponential_model([alpha, gamma], xyz)

        return result

    def eval_model(self, coords: np.ndarray):
        return self.exponential_model_centered(self.parameters, coords, self.num_primitives)

    def coulomb_integral(self, parameters: list = None):
        parameters = self.parameters if parameters is None else parameters
        out = 0.0
        for i in range(self.num_primitives):
            c_i, alpha, gamma = parameters[
                i * self.num_primitive_params : (i + 1) * self.num_primitive_params
            ]
            out += c_i * 2 * scipy.special.kn(1, alpha) * np.exp(alpha) / (
                np.pi * np.sqrt(gamma)
            )

        return out


class ExpoLorentzianModel(IsotropicModel):
    """Exponential Lorentzian."""

    def __init__(self, num_primitives, parameters=None):
        self.num_primitives = num_primitives
        self.isotropic = True
        self.parameters = parameters if parameters is not None else self.default_parameters()
        self.num_primitive_params = 4
        self.num_params = self.num_primitive_params * self.num_primitives

        super().__init__(parameters, num_params=self.num_params, is_contraction=True)

    def default_parameters(self):
        parameters = [1.0 / self.num_primitives, 1.0, 1.0, 1.0] * self.num_primitives
        return parameters

    def expo_lorentzian_model_centered_r(self, params, r, num_primitives=1):
        scalar_input = np.isscalar(r)
        if np.isscalar(r):
            r = np.array([r])

        def exponential_model(params, r):
            alpha, gamma1, gamma2 = params
            exponent = -alpha * (np.sqrt(1 + gamma1 * r**2) - 1)
            return np.exp(exponent) / (1 + gamma2 * (np.sqrt(1 + gamma1 * r**2) - 1))

        result = np.zeros(r.shape[0])

        num_params = 4
        for i in range(num_primitives):
            c_i, alpha, gamma1, gamma2 = params[i * num_params : (i + 1) * num_params]
            result += c_i * exponential_model([alpha, gamma1, gamma2], r)

        return result.item() if scalar_input else result

    def expo_lorentzian_model_centered(self, params, xyz, num_primitives=1):
        return self.expo_lorentzian_model_centered_r(
            params, np.linalg.norm(xyz, axis=1), num_primitives
        )

    def eval_model(self, coords: np.ndarray):
        return self.expo_lorentzian_model_centered(self.parameters, coords, self.num_primitives)

    def eval_model_r(self, r):
        return self.expo_lorentzian_model_centered_r(self.parameters, r, self.num_primitives)


class QuarticExponentialModel(IsotropicModel):
    """Quartic exponential parametrized via [c_i, alpha, beta, kappa]."""

    def __init__(self, num_primitives, parameters=None):
        self.num_primitives = num_primitives
        self.isotropic = True
        self.parameters = parameters if parameters is not None else self.default_parameters()
        self.num_primitive_params = 4
        self.num_params = self.num_primitive_params * self.num_primitives
        self.is_contraction = True
        super().__init__(parameters, self.num_params, self.is_contraction)

    def default_parameters(self):
        parameters = [1.0 / self.num_primitives, 1.0, 1.0, 1.0] * self.num_primitives
        return parameters

    def quartic_exponential_model_centered_r(self, params, r, num_primitives=1):
        scalar_input = np.isscalar(r)
        if np.isscalar(r):
            r = np.array([r])

        def quartic_exponential_model(params, r):
            alpha, beta, kappa = params
            if np.isclose(alpha, 0):
                return np.zeros_like(r)
            gamma1 = beta / alpha
            gamma2 = kappa / alpha
            exponent = -alpha * (np.sqrt(1 + gamma1 * r**2 + gamma2 * r**4) - 1)
            return np.exp(exponent)

        result = np.zeros(r.shape[0])

        num_params = 4
        for i in range(num_primitives):
            c_i, alpha, beta, kappa = params[i * num_params : (i + 1) * num_params]
            result += c_i * quartic_exponential_model([alpha, beta, kappa], r)

        return result.item() if scalar_input else result

    def quartic_exponential_model_centered(self, params, xyz, num_primitives=1):
        return self.quartic_exponential_model_centered_r(
            params, np.linalg.norm(xyz, axis=1), num_primitives
        )

    def eval_model(self, coords: np.ndarray):
        return self.quartic_exponential_model_centered(self.parameters, coords, self.num_primitives)

    def eval_model_r(self, r):
        return self.quartic_exponential_model_centered_r(self.parameters, r, self.num_primitives)


class EvenTemperedGaussianModel(EvenTemperedMixin, IsotropicModel):
    """Linear combination of even-tempered Gaussians with fixed exponents.

    f(r) = sum_k c_k exp(-alpha_k r^2), alpha_k = alpha_sigma * ratio**(k - (N-1)/2), alpha_sigma = 1/(2 sigma^2)

    Parameter order:
        [c_0, ..., c_{N-1}] in order of ascending exponent. Fits enforce c_k >= 0 and sum_k c_k = 1, i.e. f(0) = 1.

    A nonnegative combination of Gaussians is log-convex in |q|^2, i.e. its tail is at least as heavy as a
    Gaussian's. It therefore improves on a single Gaussian for heavier-than-Gaussian S(q), but not for S(q) that
    decays faster than a Gaussian.
    """

    linear_sum_constraint = 1.0

    def __init__(self, sigma=1.0, num_gaussians=5, ratio=2.0, exponents=None, parameters=None):
        self._init_etb(sigma, num_gaussians, ratio, exponents)
        self._etb_ctor_kwargs = {}
        self.num_primitives = self.num_gaussians
        self.num_primitive_params = 1
        if parameters is not None:
            parameters = np.asarray(parameters, dtype=float)
        super().__init__(parameters, num_params=self.num_gaussians, is_contraction=True)

    def default_parameters(self):
        return self.one_hot_reference()

    def set_parameters(self, parameters):
        self.parameters = np.asarray(parameters, dtype=float)

    def eval_model_r(self, r):
        scalar_input = np.isscalar(r)
        result = self.etb_h(r)
        return float(result) if scalar_input else result

    def linear_design_matrix(self, coords):
        return self.gaussian_columns(_radii(coords))

    def coulomb_integral(self, parameters: list = None):
        parameters = self.parameters if parameters is None else parameters
        return float(np.sum(np.asarray(parameters, dtype=float) / np.sqrt(np.pi * self.etb_exponents)))
