import numpy as np
import scipy
from scipy.integrate import quad
from scipy.spatial import KDTree

from fsec.singularity_subtraction.grids import minimum_image

from fsec.singularity_subtraction.model_function import ModelFunction
from fsec.singularity_subtraction.model_function.etb import EvenTemperedMixin


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

        return_all_zeros = self.check_parameters()
        if return_all_zeros:
            return np.zeros(coords.shape[0])

        cn_coeffs = self.c0

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

    def stacked_sum(self, coords: np.ndarray):
        """sum_dG decay_func(|q + dG|) for each q, skipping q + dG = 0 and, if remove_deltaG_zero, dG = 0."""
        deltaGs = self.deltaGs
        if self.remove_deltaG_zero:
            deltaGs = deltaGs[np.linalg.norm(deltaGs, axis=1) >= 1e-8]
        qp_norm = np.linalg.norm(coords[:, None, :] + deltaGs[None, :, :], axis=2)
        h = self.decay_func(qp_norm)
        h[qp_norm**2 < 1e-8] = 0.0
        return np.sum(h, axis=1)

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

    def initial_parameters(self, c0_scale):
        """Default shape parameters with c0 chosen so that the model is ~c0_scale near q = 0."""
        self.compute_sum_g_q_deltaG()
        parameters = np.array(self.default_parameters(), dtype=float)
        parameters[0] = c0_scale / self.g0
        return parameters

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

        q_norm = np.linalg.norm(coords, axis=1)
        return -self.c0 * q_norm**2 * self.decay_func(q_norm) * self.stacked_sum(coords)

    def stacked_sum(self, coords: np.ndarray):
        """sum_dG decay_func(|q + dG|) for each q from the cached q-mesh sums, removing dG = 0 if requested."""
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

        if self.remove_deltaG_zero:
            q_norm = np.linalg.norm(coords, axis=1)
            g_q = self.decay_func(q_norm)
            g_q[q_norm < 1e-8] = 0
            sum_g_q_deltaG = sum_g_q_deltaG - g_q

        return sum_g_q_deltaG


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


class XNETB(EvenTemperedMixin, XNGeneral):
    """f(q) = sign * |q|^deg * sum_k b_k exp(-alpha_k |q|^2) with fixed even-tempered exponents.

    Parameter order:
        [b_0, ..., b_{N-1}] in order of ascending exponent. c0 = sum_k b_k and decay_func is normalized to 1 at
        r = 0, so the model is linear in the parameters.
    """

    def __init__(self, parameters=None, deg=2, is_contraction=False, negative=True,
                 sigma=1.0, num_gaussians=5, ratio=2.0, exponents=None):
        self._init_etb(sigma, num_gaussians, ratio, exponents)
        self._etb_ctor_kwargs = {'deg': deg, 'is_contraction': is_contraction, 'negative': negative}
        super().__init__(parameters=parameters, deg=deg, is_contraction=is_contraction, negative=negative)
        self.set_parameters(self.parameters)

    def default_parameters(self):
        return self.one_hot_reference()

    def initial_parameters(self, c0_scale):
        return self.initial_coefficients(c0_scale)

    def set_parameters(self, parameters):
        self.parameters = np.asarray(parameters, dtype=float)
        self.c0 = float(np.sum(self.parameters))
        self.num_params = len(self.parameters)

    def decay_func(self, r: np.ndarray):
        if np.isclose(self.c0, 0):
            return np.zeros_like(np.asarray(r, dtype=float))
        return self.etb_h(r) / self.c0

    def eval_model(self, coords: np.ndarray):
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("coords must be an (N,3) array: [q_x, q_y, q_z]")
        r = np.linalg.norm(coords, axis=1)
        return self.sign * r**self.deg * self.etb_h(r)

    def linear_design_matrix(self, coords):
        r = np.linalg.norm(np.asarray(coords, dtype=float), axis=1)
        return self.sign * (r**self.deg)[:, None] * self.gaussian_columns(r)

    def coulomb_integral(self, coul_deg=2):
        moments = self._radial_moments(self.deg + 2 - coul_deg)
        return self.sign * 4 * np.pi * float(np.dot(self.parameters, moments))


class XNETBStackedSingularity(EvenTemperedMixin, XNGaussStackedSingularity):
    """Stacked singularity model with an even-tempered Gaussian decay function.

    f(q) = -|q|^2 h(q) sum_dG h(q + dG), h(r) = sum_k b_k exp(-alpha_k r^2)

    Parameter order:
        [b_0, ..., b_{N-1}] in order of ascending exponent. The overall scale is carried by the b_k (c0 = 1), so
        the model is quadratic in the parameters.
    """

    def __init__(self, parameters=None, deltaGs=None, is_contraction=False, remove_deltaG_zero=False,
                 sigma=1.0, num_gaussians=5, ratio=2.0, exponents=None):
        self._init_etb(sigma, num_gaussians, ratio, exponents)
        self._etb_ctor_kwargs = {'deltaGs': deltaGs, 'is_contraction': is_contraction,
                                 'remove_deltaG_zero': remove_deltaG_zero}
        self.deltaGs = None if deltaGs is None else np.asarray(deltaGs, dtype=float)
        self.is_contraction = is_contraction
        self.remove_deltaG_zero = remove_deltaG_zero
        self.c0 = 1.0
        self.parameters = np.asarray(self.default_parameters() if parameters is None else parameters, dtype=float)
        self.num_params = self.num_gaussians

    def default_parameters(self):
        return self.one_hot_reference()

    def set_parameters(self, parameters):
        self.parameters = np.asarray(parameters, dtype=float)

    def check_parameters(self):
        if self.parameters is None or len(self.parameters) != self.num_gaussians:
            raise ValueError(f"parameters must be the {self.num_gaussians} ETB coefficients")
        if np.any(np.asarray(self.parameters) < 0):
            raise ValueError("ETB coefficients must be nonnegative")
        return bool(np.allclose(self.parameters, 0))

    def decay_func(self, r: np.ndarray):
        return self.etb_h(r)

    def linear_proxy_design_matrix(self, coords):
        """Columns -|q|^2 g_k(q) H_ref(q), with the stacked sum H frozen at the reference Gaussian."""
        saved = np.array(self.parameters, dtype=float)
        self.set_parameters(self.one_hot_reference())
        try:
            H_ref = self.stacked_sum(coords)
        finally:
            self.set_parameters(saved)
        q_norm = np.linalg.norm(coords, axis=1)
        return -self.c0 * (q_norm**2 * H_ref)[:, None] * self.gaussian_columns(q_norm)

    def coulomb_integral(self):
        norm_deltaGs = np.linalg.norm(self.deltaGs, axis=1)
        result = self._pair_overlap_sum(self.parameters, norm_deltaGs)
        if self.remove_deltaG_zero:
            result -= self._pair_overlap_sum(self.parameters, np.zeros(1))
        return -self.c0 * result


class XNETBStackedSingularityQMesh(XNETBStackedSingularity, XNGaussStackedSingularityQMesh):
    """q-mesh-only stacked singularity model with an even-tempered Gaussian decay function."""

    def __init__(self, qGrid, cell, parameters=None, deltaGs=None, is_contraction=False, remove_deltaG_zero=False,
                 sigma=1.0, num_gaussians=5, ratio=2.0, exponents=None):
        XNETBStackedSingularity.__init__(
            self, parameters=parameters, deltaGs=deltaGs, is_contraction=is_contraction,
            remove_deltaG_zero=remove_deltaG_zero, sigma=sigma, num_gaussians=num_gaussians, ratio=ratio,
            exponents=exponents,
        )
        self._etb_ctor_kwargs.update(qGrid=qGrid, cell=cell)
        self.qGrid = qGrid
        self.cell = cell
        self.kdtree_qGrid = KDTree(self.qGrid)

    def set_parameters(self, parameters):
        XNETBStackedSingularity.set_parameters(self, parameters)
        self.compute_sum_g_q_deltaG()

    def initial_parameters(self, c0_scale):
        """Reference Gaussian scaled to match XNGaussStackedSingularityQMesh.initial_parameters."""
        self.set_parameters(self.one_hot_reference())
        return self.initial_coefficients(np.sqrt(c0_scale / self.g0))


class XNETBLiftedStackedSingularityQMesh(EvenTemperedMixin, ModelFunction):
    """Lifted (linearized) stacked singularity model with even-tempered Gaussian primitives, q-mesh only.

    The stacked model -|q|^2 h(q) sum_dG h(q + dG) with h = sum_k b_k g_k, g_k(r) = exp(-alpha_k r^2), is quadratic
    in b. Writing it in terms of the pair matrix B = b b^T,

        f(q) = -|q|^2 sum_{k,l} B_kl g_k(q) H_l(q),   H_l(q) = sum_dG g_l(|q + dG|),

    and dropping the rank-1 requirement on B makes the model linear in B, so the fit is a convex nonnegative least
    squares problem. The analytic Coulomb integral stays exact. The result is a nonnegative sum of pairwise stacked
    Gaussian terms; it reduces to the stacked model whenever the fitted B has rank 1.

    Parameter order:
        Upper triangle of B (k <= l, row-major as numpy.triu_indices), N (N + 1) / 2 entries, all >= 0.
    """

    def __init__(self, qGrid, cell, parameters=None, deltaGs=None, is_contraction=False, remove_deltaG_zero=False,
                 sigma=1.0, num_gaussians=5, ratio=2.0, exponents=None):
        self._init_etb(sigma, num_gaussians, ratio, exponents)
        self._etb_ctor_kwargs = {'qGrid': qGrid, 'cell': cell, 'deltaGs': deltaGs, 'is_contraction': is_contraction,
                                 'remove_deltaG_zero': remove_deltaG_zero}
        self.pair_rows, self.pair_cols = np.triu_indices(self.num_gaussians)
        self.qGrid = np.asarray(qGrid, dtype=float)
        self.cell = cell
        self.kdtree_qGrid = KDTree(self.qGrid)
        self.deltaGs = None if deltaGs is None else np.asarray(deltaGs, dtype=float)
        self.remove_deltaG_zero = remove_deltaG_zero
        self.c0 = 1.0
        self.primitive_sums = None
        self._pair_integrals = None
        super().__init__(parameters=parameters, is_contraction=is_contraction)
        self.parameters = np.asarray(self.parameters, dtype=float)

    def num_window_params(self, num_gaussians):
        return num_gaussians * (num_gaussians + 1) // 2

    def window_columns(self, idx):
        """Pair columns (k, l) with both primitives inside the window idx."""
        inside = np.isin(self.pair_rows, idx) & np.isin(self.pair_cols, idx)
        return np.flatnonzero(inside)

    def _pair_index(self, k, l):
        return int(np.flatnonzero((self.pair_rows == min(k, l)) & (self.pair_cols == max(k, l)))[0])

    def default_parameters(self):
        parameters = np.zeros(len(self.pair_rows))
        parameters[self._pair_index(self.etb_ref_index, self.etb_ref_index)] = 1.0
        return parameters

    def set_parameters(self, parameters):
        self.parameters = np.asarray(parameters, dtype=float)

    @property
    def pair_matrix(self):
        """Symmetric pair matrix B (B = b b^T for a stacked model)."""
        B = np.zeros((self.num_gaussians, self.num_gaussians))
        B[self.pair_rows, self.pair_cols] = self.parameters
        B[self.pair_cols, self.pair_rows] = self.parameters
        return B

    def compute_primitive_sums(self):
        """sum_dG g_l(|q + dG|) for every q in qGrid and primitive l (independent of the parameters)."""
        if self.deltaGs is None:
            raise ValueError("deltaGs must be set on XNETBLiftedStackedSingularityQMesh before evaluation")
        qp_norm = np.linalg.norm(self.qGrid[:, None, :] + self.deltaGs[None, :, :], axis=2)
        qp_norm[qp_norm < 1e-8] = np.inf
        qp2 = qp_norm**2
        self.primitive_sums = np.stack(
            [np.sum(np.exp(-alpha * qp2), axis=1) for alpha in self.etb_exponents], axis=1)
        zero_idx = self.kdtree_qGrid.query(np.zeros((1, 3)), k=1)[1][0]
        self.g0 = self.primitive_sums[zero_idx, self.etb_ref_index]
        return self.primitive_sums

    def initial_parameters(self, c0_scale, floor=1e-3):
        """Reference pair scaled like XNGaussStackedSingularityQMesh.initial_parameters; others at floor."""
        if self.primitive_sums is None:
            self.compute_primitive_sums()
        scale = c0_scale / self.g0
        parameters = np.full(len(self.pair_rows), floor * scale)
        parameters[self._pair_index(self.etb_ref_index, self.etb_ref_index)] = scale
        return parameters

    def _stacked_sums(self, coords):
        """H_l(q) for each coordinate, using the cached q-mesh sums and removing dG = 0 if requested."""
        if self.primitive_sums is None:
            self.compute_primitive_sums()
        coords_FBZ = minimum_image(self.cell, coords)
        idxs = self.kdtree_qGrid.query(coords_FBZ, k=1, distance_upper_bound=1e-8)[1]
        cached = idxs != len(self.qGrid)
        H = np.empty((coords.shape[0], self.num_gaussians))
        H[cached] = self.primitive_sums[idxs[cached]]
        if np.any(~cached):
            qp_norm = np.linalg.norm(coords_FBZ[~cached, None, :] + self.deltaGs[None, :, :], axis=2)
            qp_norm[qp_norm < 1e-8] = np.inf
            H[~cached] = np.stack(
                [np.sum(np.exp(-alpha * qp_norm**2), axis=1) for alpha in self.etb_exponents], axis=1)
        if self.remove_deltaG_zero:
            q_norm = np.linalg.norm(coords, axis=1)
            g_q = self.gaussian_columns(q_norm)
            g_q[q_norm < 1e-8] = 0.0
            H = H - g_q
        return H

    def linear_design_matrix(self, coords):
        coords = np.asarray(coords, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("coords must be an (N,3) array: [q_x, q_y, q_z]")
        q_norm = np.linalg.norm(coords, axis=1)
        g = self.gaussian_columns(q_norm)
        H = self._stacked_sums(coords)
        k, l = self.pair_rows, self.pair_cols
        phi = g[:, k] * H[:, l] + np.where(k == l, 0.0, 1.0) * g[:, l] * H[:, k]
        return -(q_norm**2)[:, None] * phi

    def eval_model(self, coords: np.ndarray):
        return self.linear_design_matrix(coords) @ self.parameters

    def pair_integrals(self):
        """int g_k(q) sum_dG g_l(|q + dG|) d^3q for each pair (k <= l), with dG = 0 removed if requested."""
        if self._pair_integrals is None:
            dG2 = np.sum(self.deltaGs**2, axis=1)
            values = []
            for k, l in zip(self.pair_rows, self.pair_cols):
                a_k, a_l = self.etb_exponents[k], self.etb_exponents[l]
                s = a_k + a_l
                overlaps = np.sum(np.exp(-a_k * a_l / s * dG2))
                if self.remove_deltaG_zero:
                    overlaps -= 1.0
                values.append((np.pi / s) ** 1.5 * overlaps)
            self._pair_integrals = np.array(values)
        return self._pair_integrals

    def coulomb_integral(self):
        multiplicity = np.where(self.pair_rows == self.pair_cols, 1.0, 2.0)
        return -float(np.sum(self.parameters * multiplicity * self.pair_integrals()))


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
