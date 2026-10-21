from abc import ABC, abstractmethod
import numpy as np
from fsec.singularity_subtraction.model_function import ModelFunction, IsotropicModel
from fsec.singularity_subtraction.model_function.etb import DEFAULT_RCOND, numerical_rank, radial_shells
from scipy.optimize import least_squares, lsq_linear, minimize


def coulomb_residual_weights(input_data, isotropic, fit_with_coul, coul_deg):
    """
    Residual weights used by the fit methods. Isotropic models take radial input and use 1/r^coul_deg; other
    models take (N,3) input and use 1/sum_i q_i^coul_deg. Points at q = 0 keep weight 1.
    """
    input_data = np.asarray(input_data)
    weights = np.ones(input_data.shape[0])
    if not fit_with_coul:
        return weights
    if isotropic:
        nonzero_mask = np.abs(input_data) > 1e-8
        weights[nonzero_mask] = 1./input_data[nonzero_mask]**coul_deg
    else:
        nonzero_mask = np.linalg.norm(input_data, axis=1) > 1e-8
        weights[nonzero_mask] = 1./np.sum(input_data[nonzero_mask]**coul_deg, axis=1)
    return weights

class FitMethod(ABC):
    def __init__(self, model_function: ModelFunction, **kwargs):
        self.model_function = model_function
        if isinstance(model_function, IsotropicModel):
            self.abs_diff = self.abs_diff_r
        else:
            self.abs_diff = self.abs_diff_xyz
        self.initial_guess = kwargs.get('initial_guess')
        print("Using initial_guess: ", self.initial_guess)
        if self.initial_guess is None:
            self.initial_guess = self.default_initial_guess()
        self.fit_with_coul = kwargs.get('fit_with_coul', False)
        self.coul_deg = kwargs.get('coul_deg', 2)
        self.dump_flags()

    @abstractmethod
    def fit_model(self, input_data: np.ndarray, output_data: np.ndarray,fit_multipliers=None):
        """
        Fit the model based on input_data and output_data. Output data should be normalized (i.e.S(0)=1 for exx)
        Should return the optimized parameters.
        """
        pass

    @abstractmethod
    def dump_flags(self):
        """
        For printing the fitting method used
        """
        pass

    def residuals(self, params_input, input_data: np.ndarray, output_data: np.ndarray, pow=2):
        """
        Compute the residuals between the model and the data.
        """
        result = np.sum(self.abs_diff(params_input, input_data, output_data) ** pow)
        return result

    def abs_diff_r(self, params_input, input_data: np.ndarray, output_data: np.ndarray):
        """
        Compute the residuals between the model and the data.
        """
        input_data = self._prepare_input_data(input_data)
        self.model_function.set_parameters(params_input)
        f_fit = self.model_function.eval_model_r(input_data)
        abs_diff = np.abs(f_fit - output_data)
        if self.fit_with_coul:
            abs_diff *= self.residual_weights(input_data)
        return abs_diff

    def abs_diff_xyz(self, params_input, input_data: np.ndarray, output_data: np.ndarray):
        """
        Compute the residuals between the model and the data.
        """
        self.model_function.set_parameters(params_input)
        f_fit = self.model_function.eval_model(input_data)
        abs_diff = f_fit - output_data
        if self.fit_with_coul:
            abs_diff *= self.residual_weights(input_data)
        return abs_diff

    def residual_weights(self, input_data: np.ndarray):
        """
        Per-point residual weights: 1/|q|^coul_deg away from q = 0 when fitting with the Coulomb kernel, else 1.
        """
        isotropic = isinstance(self.model_function, IsotropicModel)
        return coulomb_residual_weights(input_data, isotropic, self.fit_with_coul, self.coul_deg)

    def _prepare_input_data(self, input_data: np.ndarray):
        """
        Normalize isotropic model inputs to radial data.
        """
        input_data = np.asarray(input_data)
        if not isinstance(self.model_function, IsotropicModel):
            return input_data

        if input_data.ndim == 1:
            return input_data

        if input_data.ndim == 2 and input_data.shape[1] == 1:
            return input_data[:, 0]

        if input_data.ndim == 2 and input_data.shape[1] == 3:
            return np.linalg.norm(input_data, axis=1)

        raise ValueError(
            "For IsotropicModel, input_data must have shape (N,), (N, 1), or (N, 3). "
            f"Got {input_data.shape}."
        )


    def default_initial_guess(self):
        """
        Return an initial guess for the model parameters.
        """
        return np.ones_like(self.model_function.default_parameters())
    
    
class MP2ScipyMinimize(FitMethod):
    def __init__(self, model_function, **kwargs):
        self.is_contraction = kwargs.get('is_contraction', False)
        super().__init__(model_function, **kwargs)
    
    def fit_model(self, input_data: np.ndarray, output_data: np.ndarray, force_positive_params=True, fit_multipliers=None, x0_points=None,fixed_params=None):
        """
        Finds optimal parameters for the model using scipy.optimize.minimize
        """
        input_data = self._prepare_input_data(input_data)
        
        if fit_multipliers is None:
            fit_multipliers = np.ones(len(self.model_function.parameters))
        if x0_points is None:
            if input_data.ndim == 1:
                x0_points = [0.0]
            else:
                x0_points = [np.array([0.,0.,0.])]
            
        lb = 1e-8 if force_positive_params else -np.inf
        
        normalization = np.max(np.abs(output_data)) # To avoid numerical issues with very small residuals

            
        def residuals(params, input_data, output_data):
            self.model_function.set_parameters(params / fit_multipliers)
            if isinstance(self.model_function, IsotropicModel):
                f_fit = self.model_function.eval_model_r(input_data)
            else:
                f_fit = self.model_function.eval_model(input_data)
            abs_diff = np.abs(f_fit - output_data)
            if self.fit_with_coul:
                input_data_updated = input_data.copy()
                for x0 in x0_points:
                    if input_data_updated.ndim == 1:
                        abs_diff *= 1./(input_data_updated - x0)**2
                        mask = np.abs(input_data_updated - x0) > 1e-8
                    else:
                        abs_diff *= 1./np.sum((input_data_updated - x0)**2, axis=1)
                        mask = np.linalg.norm(input_data_updated-x0, axis=1) > 1e-8
                    abs_diff = abs_diff[mask] # take out the singular points
                    input_data_updated = input_data_updated[mask] # take out the singular points
            return np.sum(abs_diff**2)/normalization**2
        
        if fixed_params is not None:
            def residuals_to_fit(params, input_data, output_data):
                params_updated = params.copy()
                params_updated[fixed_params] = self.model_function.parameters[fixed_params]
                return residuals(params_updated, input_data, output_data)
        else:
            residuals_to_fit = residuals
            
        input_initial_guess = self.initial_guess * fit_multipliers
        constraints = ()
        bounds = [(lb, np.inf)] * len(self.model_function.parameters)
        result = minimize(residuals_to_fit, input_initial_guess, args=(input_data, output_data), constraints=constraints,
                          bounds=bounds, options={'ftol':1e-12, 'gtol':1e-12})
        if not result.success:
            print("WARNING:Fitting failed. Result: ", result)
            raise ValueError("Fitting failed. Result: ", result)
        self.fitted_parameters = result.x / fit_multipliers
        return self.fitted_parameters
    
    def dump_flags(self):
        """
        Dump the flags used in the fitting.
        """
        print("<class MP2ScipyMinimize> with model function", self.model_function)
        print("Fitting with coul: ", self.fit_with_coul)


class ExxScipyMinimize(FitMethod):
    def __init__(self, model_function : ModelFunction, **kwargs):
        self.is_contraction = kwargs.get('is_contraction', False)
        super().__init__(model_function, **kwargs)

    def residuals_set_c0(self, params_input, input_data: np.ndarray, output_data: np.ndarray):
        """
        Manually set first parameter to 1. To avoid using `constraint` object.
        """
        params = np.zeros(len(params_input)+1)
        params[1:] = params_input
        params[0] = 1
        # self.model_function.set_parameters(params)
        # return self.residuals(params, input_data, output_data, pow)
        return self.residuals(params, input_data, output_data)

    def fit_model(self, input_data: np.ndarray, output_data: np.ndarray, force_positive_params=True, fit_multipliers=None):
        """
        Finds optimal parameters for the model using scipy.optimize.minimize
        """
        constraints = ()
        lb = 1e-8 if force_positive_params else -np.inf
        bounds = [(lb, np.inf)] * len(self.model_function.parameters)


        input_initial_guess = self.initial_guess
        residuals = self.residuals
        is_contraction = self.model_function.is_contraction
        if is_contraction:
            # Constraint where all c_i must be positive and sum to 1
            if self.model_function.num_primitives > 1:
                def normalization(params):
                    return np.sum(params[::self.model_function.num_primitive_params]) - 1
                constraints = [
                    {'type': 'eq', 'fun': normalization},
                ]
                bounds[::self.model_function.num_primitive_params] = [(0.0, np.inf)]
            elif self.model_function.num_primitives == 1:
                # Try to avoid setting a `constraint` to use LBFGS instead of SLSQP
                input_initial_guess = input_initial_guess[1:]
                bounds = bounds[1:]
                residuals = self.residuals_set_c0
            else:
                raise ValueError("num_primitives must be > 0")


        result = minimize(residuals, input_initial_guess, args=(input_data, output_data), constraints=constraints,
                          bounds=bounds,options={'ftol':1e-8, 'gtol':1e-8})
        self.fitted_parameters = result.x
        if not result.success:
            print("WARNING:Fitting failed. Result: ", result)
            raise ValueError("Fitting failed. Result: ", result)

        if is_contraction and self.model_function.num_primitives == 1:
            params = np.zeros(len(result.x)+1)
            params[1:] = result.x
            params[0] = 1.0
            self.fitted_parameters = params

        print("Fitting complete. Parameters: ", self.fitted_parameters)
        print("Sum residual: ", result.fun)
        print("Force positive params: ", force_positive_params)
        return self.fitted_parameters

    def dump_flags(self):
        """
        Dump the flags used in the fitting.
        """
        print("<class ExxScipyMinimize> with model function", self.model_function)
        print("Fitting with coul: ", self.fit_with_coul)


class ExxScipyLeastSquares(FitMethod):
    def __init__(self, model_function : ModelFunction, **kwargs):
        super().__init__(model_function, **kwargs)

    def abs_diff_set_c0(self, params_input, input_data: np.ndarray, output_data: np.ndarray):
        """
        Instead of setting constraint sum_i c_i = 1, we force c_0 = 1 - sum_i c_i and remove c_0 from
        the optimization.
        """
        num_primitive_params = self.model_function.num_primitive_params
        params = np.zeros(len(params_input)+1)
        params[1:] = params_input
        params[0] = 1 - np.sum(params_input[num_primitive_params-1::num_primitive_params])
        self.model_function.set_parameters(params)
        # return self.residuals(params, input_data, output_data, pow)
        return self.abs_diff(params, input_data, output_data)

    def fit_model(self, input_data: np.ndarray, output_data: np.ndarray, force_positive_params=False, fit_multipliers=None):
        """
        Finds optimal parameters for the model using scipy.optimize.least_squares
        """
        # Setup bounds for the parameters
        is_contraction = self.model_function.is_contraction
        lb = 0. if force_positive_params else -np.inf
        upper_bounds = [np.inf]*len(self.model_function.parameters)
        lower_bounds = [lb]*len(self.model_function.parameters)
        input_initial_guess = self.initial_guess
        if is_contraction:
            lower_bounds[::self.model_function.num_primitive_params] = [0.0]*self.model_function.num_primitives

            # Remove the first parameter (c_i), because we are fitting with normalization
            upper_bounds = upper_bounds[1:]
            lower_bounds = lower_bounds[1:]

            input_initial_guess = self.initial_guess[1:]
        bounds = (lower_bounds,upper_bounds)


        abs_diff = self.abs_diff_set_c0 if is_contraction else self.abs_diff
        max_nfev = 1000 * self.model_function.num_params
        result = least_squares(
            abs_diff,
            input_initial_guess,
            args=(input_data, output_data),
            bounds=bounds,
            verbose=2,
            max_nfev=max_nfev,
        )
        if not result.success:
            print("WARNING: Fitting failed. Result: ", result)
            raise ValueError("Fitting failed. Result: ", result)
        
        # Check if fit is sigificantly incorrect.
        normalization = np.sum(np.abs(output_data))
        npoints = len(input_data)
        abs_rel_error = np.sum(abs_diff(result.x, input_data, output_data))/normalization/npoints
        if abs_rel_error > 0.5:
            print("WARNING: Relative error (", abs_rel_error, ") is too large.")
            
        if is_contraction:
            params = np.zeros(len(result.x)+1)
            params[1:] = result.x
            params[0] = 1 - \
                np.sum(result.x[self.model_function.num_primitive_params-1::self.model_function.num_primitive_params])
            self.fitted_parameters = params
        else:
            self.fitted_parameters = result.x

        print("Fitting complete. Parameters: ", self.fitted_parameters)
        print("Sum residual: ", np.sum((result.fun)**2))
        print("Force positive params: ", force_positive_params)
        return self.fitted_parameters

    def dump_flags(self):
        """
        Dump the flags used in the fitting.
        """
        print("<class MP2ScipyLeastSquares> with model function", self.model_function)
        print("Fitting with coul: ", self.fit_with_coul)


class MP2ScipyLeastSquares(FitMethod):
    def __init__(self, model_function : ModelFunction, **kwargs):
        super().__init__(model_function, **kwargs)
        
    def fit_model(self, input_data: np.ndarray, output_data: np.ndarray, force_positive_params=False, fit_multipliers=None,
                  fixed_params=None,jac='2-point',x_scale=1.0,max_nfev=None):
        """
        Finds optimal parameters for the model using scipy.optimize.least_squares
        """
        # Setup bounds for the parameters
        normalization = np.sum(np.abs(output_data)) # To avoid numerical issues with very small residuals

        lb = 1e-14 * normalization if force_positive_params else -np.inf
        upper_bounds = [np.inf]*len(self.model_function.parameters)
        lower_bounds = [lb]*len(self.model_function.parameters)
        input_initial_guess = self.initial_guess
        bounds = (lower_bounds,upper_bounds)
        
        
        def abs_diff(params, input_data, output_data):
            return self.abs_diff(params, input_data, output_data)/normalization

        result = least_squares(abs_diff, input_initial_guess, args=(input_data, output_data), bounds=bounds, verbose=2, jac=jac,
                               x_scale=x_scale, max_nfev=max_nfev,ftol=1e-12,gtol=1e-12)
        if not result.success:
            print("WARNING:Fitting failed. Result: ", result)
            raise ValueError("Fitting failed. Result: ", result)
        
        # Check if fit is sigificantly incorrect.
        npoints = len(input_data)
        abs_rel_error = np.sum(np.abs(abs_diff(result.x, input_data, output_data)))#/normalization/npoints # dont normalize by normalization/npoints?
        if abs_rel_error > 0.5:
            print("WARNING: Relative error (", abs_rel_error, ") is too large.")

        self.fitted_parameters = result.x


        print("Fitting complete. Parameters: ", self.fitted_parameters)
        print("Sum residual: ", np.sum((result.fun)**2))
        print("Relative error: ", abs_rel_error)
        print("Force positive params: ", force_positive_params)
        return self.fitted_parameters

    def dump_flags(self):
        """
        Dump the flags used in the fitting.
        """
        print("<class MP2ScipyLeastSquares> with model function", self.model_function)
        print("Fitting with coul: ", self.fit_with_coul)


class ETBLinearLeastSquares(FitMethod):
    """
    Direct nonnegative least squares for auxiliary functions that are linear in their parameters, i.e. models
    with fixed even-tempered exponents that implement linear_design_matrix(coords) so that
    eval_model(coords) = linear_design_matrix(coords) @ parameters.

    Minimizes ||W (A b - y)||^2 subject to b >= 0 (and sum(b) = model.linear_sum_constraint if the model defines
    one), where W are the same residual weights as the nonlinear fit methods. The problem is strictly convex when
    W A has full column rank, so the solution is unique and needs no initial guess.
    """
    def __init__(self, model_function: ModelFunction, **kwargs):
        if not hasattr(model_function, 'linear_design_matrix'):
            raise TypeError(f"{model_function.__class__.__name__} is not linear in its parameters; "
                            "ETBLinearLeastSquares requires a model with linear_design_matrix()")
        self.rcond = kwargs.get('rcond', DEFAULT_RCOND)
        super().__init__(model_function, **kwargs)

    def default_initial_guess(self):
        return np.asarray(self.model_function.default_parameters(), dtype=float)

    def fit_model(self, input_data: np.ndarray, output_data: np.ndarray, **kwargs):
        """
        Solve for the coefficients directly. Keyword arguments used only by the nonlinear fit methods
        (fit_multipliers, fixed_params, force_positive_params, jac, x_scale, max_nfev) are accepted and ignored;
        the coefficients are always nonnegative.
        """
        model = self.model_function
        input_data = np.asarray(input_data, dtype=float)
        output_data = np.real_if_close(np.asarray(output_data)).astype(float).ravel()
        if isinstance(model, IsotropicModel):
            weight_input = self._prepare_input_data(input_data)
        else:
            weight_input = input_data
        weights = self.residual_weights(weight_input)

        A = weights[:, None] * model.linear_design_matrix(input_data)
        y = weights * output_data
        num_coeffs = A.shape[1]

        constraint = getattr(model, 'linear_sum_constraint', None)
        rank_matrix = A if constraint is None else np.vstack([A, np.ones((1, num_coeffs))])
        rank = numerical_rank(rank_matrix, self.rcond)
        if rank < num_coeffs:
            n_shells = len(radial_shells(input_data))
            raise ValueError(
                f"ETB design matrix has numerical rank {rank} < {num_coeffs} coefficients "
                f"({len(input_data)} points, {n_shells} distinct nonzero |q| shells): the coefficients are not "
                "uniquely determined. Increase qG_norm_cutoff or reduce num_gaussians."
            )

        if constraint is None:
            coeffs = self._nnls(A, y)
        else:
            coeffs = self._nnls_fixed_sum(A, y, constraint)

        self.fitted_parameters = coeffs
        model.set_parameters(coeffs)
        residual = weights * (model.eval_model(input_data) if not isinstance(model, IsotropicModel)
                              else model.eval_model_r(weight_input)) - y

        print("Fitting complete. Parameters: ", self.fitted_parameters)
        print("Sum residual: ", np.sum(residual**2))
        return self.fitted_parameters

    def _nnls(self, A, y):
        """
        min ||A b - y|| subject to b >= 0, by bounded-variable least squares. scipy.optimize.nnls is avoided: it
        returns inaccurate solutions for the moderately ill-conditioned ETB design matrices (cond ~ 1e6). Columns
        and right-hand side are scaled to unit norm because MP2 structure factors are small.
        """
        col_norms = np.linalg.norm(A, axis=0)
        col_norms[col_norms == 0] = 1.0
        y_norm = np.linalg.norm(y)
        if y_norm == 0:
            return np.zeros(A.shape[1])
        result = lsq_linear(A / col_norms, y / y_norm, bounds=(0, np.inf), method='bvls', tol=1e-14)
        return np.maximum(result.x, 0.0) * y_norm / col_norms

    def _nnls_fixed_sum(self, A, y, total):
        """
        min ||A b - y|| subject to b >= 0 and sum(b) = total.

        Eliminating b_j = total - sum_{i != j} b_i leaves a bound-constrained problem that relaxes b_j >= 0. If its
        solution has b_j >= 0 it is optimal for the original convex problem. Candidates j are tried in order of
        decreasing weight in an approximate (penalized) solution.
        """
        num_coeffs = A.shape[1]
        if num_coeffs == 1:
            return np.array([float(total)])

        penalty = 1e3 * max(np.linalg.norm(A), 1.0)
        approx = self._nnls(np.vstack([A, penalty * np.ones((1, num_coeffs))]),
                            np.concatenate([y, [penalty * total]]))
        for j in np.argsort(-approx):
            others = np.arange(num_coeffs) != j
            A_reduced = A[:, others] - A[:, [j]]
            col_norms = np.linalg.norm(A_reduced, axis=0)
            col_norms[col_norms == 0] = 1.0
            result = lsq_linear(A_reduced / col_norms, y - total * A[:, j], bounds=(0, np.inf),
                                method='bvls', tol=1e-14)
            b_others = result.x / col_norms
            b_j = total - np.sum(b_others)
            if b_j >= -1e-12 * abs(total):
                coeffs = np.zeros(num_coeffs)
                coeffs[others] = b_others
                coeffs[j] = max(b_j, 0.0)
                return coeffs * (total / np.sum(coeffs))
        raise ValueError(f"ETB fit failed to find nonnegative coefficients summing to {total}")

    def dump_flags(self):
        """
        Dump the flags used in the fitting.
        """
        print("<class ETBLinearLeastSquares> with model function", self.model_function)
        print("Fitting with coul: ", self.fit_with_coul)
