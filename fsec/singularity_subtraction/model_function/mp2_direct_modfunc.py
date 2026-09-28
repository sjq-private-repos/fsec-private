import numpy as np
import scipy
from scipy.integrate import quad
from scipy.spatial import KDTree

from fsec.singularity_subtraction.grids import minimum_image

from fsec.singularity_subtraction.model_function import ModelFunction


class XNGeneral(ModelFunction):
    """Auxiliary model: f(q) = x^n * decay_func(x), where decay_func(x) is a decay function and n is an integer."""

    def __init__(self, parameters=None, deg=2, is_contraction=False, negative=True):
        super().__init__(parameters=parameters, is_contraction=is_contraction)
        self.deg = deg
        self.c0 = self.parameters[0]
        self.sign = -1 if negative else 1

    def decay_func(self, r: np.ndarray):
        """Decay function based on radius r."""
        pass

    def eval_model(self, coords: np.ndarray):
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("coords must be an (N,3) array: [q_x, q_y, q_z]")
        if self.c0 is None:
            raise ValueError("c0 must be set on XNGeneral before evaluation")
        if self.deg < 0:
            raise ValueError("deg must be non-negative")

        r = np.linalg.norm(coords, axis=1)
        return self.sign * self.c0 * r**self.deg * self.decay_func(r)

    def coulomb_integral(self, coul_deg=2):
        def integrand(r):
            return r ** (self.deg + 2 - coul_deg) * self.decay_func(r)

        integral, abserror = quad(integrand, 0, np.inf)
        prefactor = 4 * np.pi
        print("Computed integral with scipy.integrate.quad. Estimated error: ", abserror)
        return self.sign * prefactor * self.c0 * integral


class XNGauss(XNGeneral):
    def __init__(self, parameters=None, deg=2, is_contraction=False, negative=True):
        self.c0 = parameters[0] if parameters is not None else 1.0
        self.sigma = parameters[1] if parameters is not None else 1.0
        super().__init__(parameters=parameters, deg=deg, is_contraction=is_contraction, negative=negative)

    def decay_func(self, r: np.ndarray):
        return np.exp(-r**2 / (2.0 * self.sigma * self.sigma))

    def default_parameters(self):
        return [1.0, 1.0]

    def set_parameters(self, parameters):
        self.parameters = parameters
        self.c0 = parameters[0] if parameters is not None else 1.0
        self.sigma = parameters[1] if parameters is not None else 1.0

    def coulomb_integral(self, coul_deg=2):
        if np.isclose(self.sigma, 0):
            return 0.0

        b = 1 / (2.0 * self.sigma**2)
        n = self.deg + 2 - coul_deg

        integral = scipy.special.gamma((n + 1) / 2) / (2 * b ** ((n + 1) / 2))

        prefactor = 4 * np.pi
        return self.sign * prefactor * self.c0 * integral


class XNExpAbs(XNGeneral):
    def __init__(self, parameters=None, deg=2, is_contraction=False, negative=True):
        self.c0 = parameters[0]
        self.alpha = parameters[1]
        super().__init__(parameters=parameters, deg=deg, is_contraction=is_contraction, negative=negative)

    def decay_func(self, r: np.ndarray):
        if np.isclose(self.alpha, 0):
            return np.zeros_like(r)

        return np.exp(-self.alpha * r)

    def default_parameters(self):
        return [1.0, 1.0]

    def set_parameters(self, parameters):
        self.parameters = parameters
        self.c0 = parameters[0]
        self.alpha = parameters[1]
        super().set_parameters(parameters)


class XNExpAbs2(XNGeneral):
    def __init__(self, parameters=None, deg=2, is_contraction=False, negative=True):
        self.c0 = parameters[0]
        self.alpha = parameters[1]
        self.beta = parameters[2]
        super().__init__(parameters=parameters, deg=deg, is_contraction=is_contraction, negative=negative)

    def decay_func(self, r: np.ndarray):
        if np.isclose(self.alpha, 0):
            return np.zeros_like(r)

        return np.exp(-self.alpha * r - self.beta * r**2)

    def default_parameters(self):
        return [1.0, 1.0, 1.0]

    def set_parameters(self, parameters):
        self.parameters = parameters
        self.c0 = parameters[0]
        self.alpha = parameters[1]
        self.beta = parameters[2]
        super().set_parameters(parameters)


class XNExponential(XNGeneral):
    def __init__(self, parameters=None, deg=2, is_contraction=False, negative=True):
        self.c0 = parameters[0]
        self.alpha = parameters[1]
        self.gamma = parameters[2]
        super().__init__(parameters=parameters, deg=deg, is_contraction=is_contraction, negative=negative)

    def decay_func(self, r: np.ndarray):
        if np.isclose(self.alpha, 0):
            return np.zeros_like(r)
        return np.exp(-self.alpha * (np.sqrt(1 + self.gamma * r**2) - 1))

    def default_parameters(self):
        return [1.0, 1.0, 1.0]

    def set_parameters(self, parameters):
        self.parameters = parameters
        self.c0 = parameters[0]
        self.alpha = parameters[1]
        self.gamma = parameters[2]
        super().set_parameters(parameters)


class XNQuarticExponential(XNGeneral):
    """Quartic exponential parametrized via [c0, alpha, beta, kappa]."""

    def __init__(self, parameters=None, deg=2, is_contraction=False, negative=True):
        self.c0 = parameters[0]
        self.alpha = parameters[1]
        self.beta = parameters[2]
        self.kappa = parameters[3]
        super().__init__(parameters=parameters, deg=deg, is_contraction=is_contraction, negative=negative)

    def decay_func(self, r: np.ndarray):
        if np.isclose(self.alpha, 0):
            return np.zeros_like(r)
        gamma1 = self.beta / self.alpha
        gamma2 = self.kappa / self.alpha
        return np.exp(-self.alpha * (np.sqrt(1 + gamma1 * r**2 + gamma2 * r**4) - 1))

    def default_parameters(self):
        return [1.0, 1.0, 1.0, 1.0]

    def set_parameters(self, parameters):
        self.parameters = parameters
        self.c0 = parameters[0]
        self.alpha = parameters[1]
        self.beta = parameters[2]
        self.kappa = parameters[3]
        super().set_parameters(parameters)


class XNGaussStackedSingularity(ModelFunction):
    """Auxiliary model with stacked singularity over q + deltaG."""

    def __init__(self, parameters=None, deltaGs=None, is_contraction=False, remove_deltaG_zero=False):
        super().__init__(parameters=parameters, is_contraction=is_contraction)
        self.deltaGs = None if deltaGs is None else np.asarray(deltaGs, dtype=float)
        self.remove_deltaG_zero = remove_deltaG_zero
        if parameters is None:
            parameters = self.default_parameters()
        self.c0 = parameters[0]
        self.sigma = parameters[1]
        self.num_params = 2

    def default_parameters(self):
        return [1.0, 1.0]

    def set_deltaGs(self, deltaGs):
        self.deltaGs = np.asarray(deltaGs, dtype=float)

    def set_parameters(self, parameters):
        self.parameters = parameters
        self.c0 = parameters[0]
        self.sigma = parameters[1]

    def check_parameters(self):
        return_all_zeros = False
        if len(self.parameters) != 2:
            raise ValueError("parameters must be [c0, sigma]")
        if self.c0 is None or self.sigma is None:
            raise ValueError("c0 and sigma must be set on XNGaussStackedSingularityQMesh before evaluation")
        if self.sigma < 0:
            raise ValueError("sigma must be positive")
        if np.isclose(self.sigma, 0):
            return_all_zeros = True

        return return_all_zeros

    def eval_model(self, coords: np.ndarray):
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("coords must be an (N,3) array: [q_x, q_y, q_z]")
        if self.deltaGs is None:
            raise ValueError("deltaGs must be set on XNGaussStackedSingularity before evaluation")
        if self.deltaGs.shape != (len(self.deltaGs), 3):
            raise ValueError("deltaGs must be a (N,3) array")
        if len(self.parameters) != 2:
            raise ValueError("parameters must be [c0, sigma]")

        cn_coeffs = self.parameters[0]

        return_all_zeros = self.check_parameters()
        if return_all_zeros:
            return np.zeros(coords.shape[0])

        deltaGs = self.deltaGs
        q = coords
        q_expanded = q[:, np.newaxis, :]
        deltaGs_expanded = deltaGs[np.newaxis, :, :]
        q_plus_dG = q_expanded + deltaGs_expanded

        q_norm = np.linalg.norm(q_expanded, axis=2)
        qp_norm = np.linalg.norm(q_plus_dG, axis=2)

        denominator = qp_norm**2
        denominator[denominator < 1e-8] = np.inf

        h = self.decay_func(q_norm) * self.decay_func(qp_norm)
        result_matrix = -cn_coeffs * q_norm**2 * qp_norm**2 * h / denominator

        if self.remove_deltaG_zero:
            dG_norms = np.linalg.norm(deltaGs, axis=1)
            zero_mask = dG_norms < 1e-8
            result_matrix = result_matrix[:, ~zero_mask]

        result = np.sum(result_matrix, axis=1)
        return result

    def decay_func(self, r: np.ndarray):
        if np.isclose(self.sigma, 0, atol=1e-8):
            return np.zeros_like(r, dtype=float)
        return np.exp(-r**2 / (2.0 * self.sigma**2))

    def coulomb_integral(self):
        result = 0
        norm_deltaGs = np.sum(self.deltaGs**2, axis=1) ** (1 / 2)
        sigma = self.parameters[1]
        c0 = self.parameters[0]
        alpha = 1.0 / (2.0 * sigma * sigma)
        result = np.sum(np.exp(-alpha / 2 * norm_deltaGs**2))

        if self.remove_deltaG_zero:
            result = result - 1

        return -c0 * result * (np.pi / (2 * alpha)) ** (3 / 2)


class XNGaussStackedSingularityQMesh(XNGaussStackedSingularity):
    """q-mesh-only stacked singularity model."""

    def __init__(self, qGrid, cell, parameters=None, deltaGs=None, is_contraction=False, remove_deltaG_zero=False):
        super().__init__(
            parameters=parameters,
            is_contraction=is_contraction,
            remove_deltaG_zero=remove_deltaG_zero,
        )
        self.deltaGs = None if deltaGs is None else np.asarray(deltaGs, dtype=float)
        self.qGrid = qGrid
        self.cell = cell
        self.kdtree_qGrid = KDTree(self.qGrid)
        self.sigma = self.parameters[1]
        self.c0 = self.parameters[0]

    def default_parameters(self):
        return [1.0, 1.0]

    def compute_sum_g_q_deltaG(self):
        if self.qGrid is None:
            raise ValueError("qGrid must be set on XNGaussStackedSingularityQMesh before evaluation")
        if self.deltaGs is None:
            raise ValueError("deltaGs must be set on XNGaussStackedSingularityQMesh before evaluation")

        deltaGs = self.deltaGs
        q = self.qGrid
        q_expanded = q[:, None, :]
        deltaGs_expanded = deltaGs[None, :, :]
        q_plus_dG = q_expanded + deltaGs_expanded

        qp_norm = np.linalg.norm(q_plus_dG, axis=2)
        qp_norm[qp_norm < 1e-8] = np.inf
        N = qp_norm.shape[0]
        M = qp_norm.shape[1]
        g_q_deltaG = self.decay_func(qp_norm.reshape(N * M)).reshape(N, M)

        result = np.sum(g_q_deltaG, axis=1)

        zero_idx = self.kdtree_qGrid.query(np.zeros((1, 3)), k=1)[1][0]
        self.g0 = result[zero_idx]

        self.sum_g_q_deltaG = result
        return result

    def set_parameters(self, parameters):
        self.parameters = parameters
        self.sigma = self.parameters[1]
        self.c0 = self.parameters[0]
        self.compute_sum_g_q_deltaG()

    def eval_model(self, coords: np.ndarray):
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("coords must be an (N,3) array: [q_x, q_y, q_z]")
        if self.deltaGs is None:
            raise ValueError("deltaGs must be set on XNGaussStackedSingularity before evaluation")
        if self.deltaGs.shape != (len(self.deltaGs), 3):
            raise ValueError("deltaGs must be a (N,3) array")

        return_all_zeros = self.check_parameters()
        if return_all_zeros:
            return np.zeros(coords.shape[0])

        cn_coeffs = self.parameters[0]

        coords_FBZ = minimum_image(self.cell, coords)
        idxs_coords_FBZ_in_qGrid = self.kdtree_qGrid.query(
            coords_FBZ, k=1, distance_upper_bound=1e-8
        )[1]
        cached = idxs_coords_FBZ_in_qGrid != len(self.qGrid)
        sum_g_q_deltaG = np.empty(coords.shape[0], dtype=float)
        sum_g_q_deltaG[cached] = self.sum_g_q_deltaG[
            idxs_coords_FBZ_in_qGrid[cached]
        ]
        if np.any(~cached):
            q_plus_dG = (
                coords_FBZ[~cached, None, :] + self.deltaGs[None, :, :]
            )
            sum_g_q_deltaG[~cached] = np.sum(
                self.decay_func(np.linalg.norm(q_plus_dG, axis=2)),
                axis=1,
            )

        q_norm = np.linalg.norm(coords, axis=1)
        result = -cn_coeffs * q_norm**2 * self.decay_func(q_norm) * sum_g_q_deltaG

        if self.remove_deltaG_zero:
            g_q = self.decay_func(q_norm)
            g_q[q_norm < 1e-8] = 0
            result = result + cn_coeffs * q_norm**2 * self.decay_func(q_norm) * g_q

        return result


class XNExpAbsStackedSingularity(XNGaussStackedSingularity):
    """Stacked singularity model with exponential decay."""

    def __init__(self, parameters=None, deltaGs=None, is_contraction=False, remove_deltaG_zero=False):
        self.deltaGs = None if deltaGs is None else np.asarray(deltaGs, dtype=float)
        if parameters is None:
            parameters = self.default_parameters()
        self.c0 = parameters[0]
        self.alpha = parameters[1]
        self.num_params = 2
        self.is_contraction = is_contraction
        self.remove_deltaG_zero = remove_deltaG_zero

    def default_parameters(self):
        return [1.0, 1.0]

    def set_parameters(self, parameters):
        self.parameters = parameters
        self.c0 = parameters[0]
        self.alpha = parameters[1]

    def check_parameters(self):
        return_all_zeros = False
        if len(self.parameters) != 2:
            raise ValueError("parameters must be [c0, alpha]")
        if self.c0 is None or self.alpha is None:
            raise ValueError("c0 and alpha must be set on XNExpAbsStackedSingularity before evaluation")
        if self.alpha < 0:
            raise ValueError("alpha must be positive")
        if np.isclose(self.alpha, 0):
            return_all_zeros = True

        return return_all_zeros

    def decay_func(self, r: np.ndarray):
        if np.isclose(self.alpha, 0, atol=1e-8):
            return np.zeros_like(r, dtype=float)
        return np.exp(-self.alpha * r)

    def coulomb_integral(self):
        result = 0
        norm_deltaGs = np.sum(self.deltaGs**2, axis=1) ** (1 / 2)
        alpha = self.alpha
        c0 = self.c0

        p = alpha * norm_deltaGs
        normalization_factor = np.pi / (alpha**3)

        S_100100R = np.sum(np.exp(-p) * (1 + p + (1 / 3) * p**2))
        if self.remove_deltaG_zero:
            S_100100R = S_100100R - 1

        result = normalization_factor * S_100100R
        return -c0 * result


class XNExpAbsStackedSingularityQMesh(XNExpAbsStackedSingularity, XNGaussStackedSingularityQMesh):
    """q-mesh-only stacked singularity model with exponential decay."""

    def __init__(self, qGrid, cell, parameters=None, deltaGs=None, is_contraction=False, remove_deltaG_zero=False):
        self.parameters = parameters if parameters is not None else self.default_parameters()
        self.remove_deltaG_zero = remove_deltaG_zero
        self.is_contraction = is_contraction
        self.num_params = len(self.parameters)

        self.deltaGs = None if deltaGs is None else np.asarray(deltaGs, dtype=float)
        self.qGrid = qGrid
        self.cell = cell
        self.kdtree_qGrid = KDTree(self.qGrid)
        self.alpha = self.parameters[1]
        self.c0 = self.parameters[0]

    def default_parameters(self):
        return [1.0, 1.0]

    def set_parameters(self, parameters):
        self.parameters = parameters
        self.alpha = self.parameters[1]
        self.c0 = self.parameters[0]
        self.compute_sum_g_q_deltaG()


class PolynomialModel(ModelFunction):
    """Polynomial model: f(q) = c0 + c1 * q + c2 * q^2 + ... + cn * q^n."""

    def __init__(self, parameters=None, orders=[2]):
        self.orders = orders
        super().__init__(parameters=parameters, is_contraction=False)
        self.cn_coeffs = self.parameters
        self.set_parameters(self.parameters)
        if len(self.parameters) != len(self.orders):
            raise ValueError("parameters must be [c0, c1, c2, ..., cn]")

    def set_parameters(self, parameters):
        self.parameters = parameters
        self.cn_coeffs = self.parameters

    def default_parameters(self):
        return [1.0] * len(self.orders)

    def eval_model(self, coords: np.ndarray):
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("coords must be an (N,3) array: [q_x, q_y, q_z]")
        if len(self.parameters) != len(self.orders):
            raise ValueError("parameters must be [c0, c1, c2, ..., cn]")

        r = np.linalg.norm(coords, axis=1)
        result = np.zeros(r.shape)
        for i, order in enumerate(self.orders):
            result += self.cn_coeffs[i] * r**order
        return result

    def coulomb_integral(self):
        raise NotImplementedError("Coulomb integral for PolynomialModel is not implemented.")
