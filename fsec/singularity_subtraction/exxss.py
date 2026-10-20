from pyscf.pbc.tools import get_monkhorst_pack_size, madelung
from pyscf.pbc import df
import time
from fsec.singularity_subtraction import model_function
# import traceback
from fsec.singularity_subtraction.function_fitting import (
    ETBLinearLeastSquares,
    ExxScipyLeastSquares,
    ExxScipyMinimize,
    coulomb_residual_weights,
)
from fsec.singularity_subtraction.grids import ExxSSGrids
from fsec.singularity_subtraction.model_function.etb import (
    DEFAULT_MIN_SHELL_MARGIN,
    DEFAULT_RCOND,
    cutoff_for_min_shells,
    radial_shells,
    trim_etb_model_to_data,
)
from fsec.singularity_subtraction.structure_factor import ExxStructureFactor
from fsec.singularity_subtraction.structure_factor.exx_sf import (
    normalize_pair_density_becke_grid_level,
    normalize_pair_density_eval_grid,
)
from fsec.singularity_subtraction.structure_factor.helpers import build_uKpts as _build_uKpts
from fsec.singularity_subtraction import SingularitySubtraction

import numpy as np

def build_uKpts(kmf, kpts, mo_coeff_kpts, NsCell=None, rptGrid3D=None, nbands=None):
    return _build_uKpts(kmf, kpts, mo_coeff_kpts, NsCell, rptGrid3D, nbands)


class ExxSS(SingularitySubtraction):
    """
    A class for performing singularity subtraction to correct the finite size error in
    exact exchange (Exx)

    Key attributes:
        kmf (object): PySCF Mean-field object. Should have converged SCF calculation.
        cell (object): The periodic cell object from the mean-field calculation.
        nocc (int): Number of occupied orbitals
        auxfunc (model_function.ModelFunction): Auxiliary function f(q) used in the singularity subtraction
        fit_method (callable): Fitting method for the auxiliary function.
    """
    fit_method_dict = {
        'scipy_minimize': ExxScipyMinimize,
        'scipy_least_squares': ExxScipyLeastSquares,
        'etb_linear': ETBLinearLeastSquares,
    }
    results_title = "ExxSS"


    def __init__(self, kmf, auxfunc: model_function.ModelFunction = None, dim=3, model_params=None, **kwargs):
        """
        Initialize the SingularitySubtraction class.
        Parameters:
            kmf (object): Kohn-Sham mean-field object containing the computational cell and related properties.
            auxfunc (model_function.ModelFunction): Auxiliary model function used for fitting.
            dim (int, optional): Dimensionality of the system. Default is 3.
            model_params (dict, optional): Parameters for the auxiliary model function. Default is None.
        kwargs: Additional optional keyword arguments:
            fit_with_coul (bool): Whether to fit with Coulomb interaction. Default is True.
            fit_to_structure_factor (bool): Whether to fit to the structure factor. Default is True.
            qG_norm_cutoff_sigma (float or None): Cutoff for qG norm in terms of Gaussian-fit sigmas. Default is None.
            initial_guess (array-like or None): Initial guess for the fitting parameters. Default is None.
            fit_method (str): Method used for fitting. Default is 'scipy_least_squares'.
            force_positive_parameters (bool): Whether to enforce positivity of fitting parameters. Default is True.
            N_local (array-like or int): Controls the real-space grid for the structure factor. Default is
                kmf.cell.mesh.
            sq_ke_cutoff (float or None): kecutoff to control real space grid for the structure factor calculation. 
            sq_inversion_symm (bool): Whether to enforce inversion symmetry in the structure factor. Default is True.
            line_sampling (bool): Whether to sample q+G along reciprocal-lattice directions only. Default is False.
            pair_density_eval_grid (str): Real-space quadrature grid used for
                pair-density overlaps. Supported values are "uniform" and
                "becke". Default is "becke".
            pair_density_becke_grid_level (int): PySCF periodic Becke grid
                level. Default is 0.
        """


        using_default_auxfunc = auxfunc is None
        if using_default_auxfunc:
            auxfunc = model_function.ContractedGaussianModel(
                num_gaussians=1,
                isotropic=True,
                parameters=model_params,
            )

        self.kmf = kmf
        self.cell = kmf.cell
        self.nocc = self.cell.tot_electrons() // 2
        self.auxfunc = auxfunc
        self.model_params = model_params

        # Fit Settings
        self.fit_with_coul = kwargs.get('fit_with_coul', True)
        self.fit_to_structure_factor = kwargs.get('fit_to_structure_factor', True)
        default_qG_norm_cutoff_sigma = getattr(
            self,
            'qG_norm_cutoff_sigma',
            1.0 if using_default_auxfunc else None,
        )
        self.qG_norm_cutoff_sigma = kwargs.get('qG_norm_cutoff_sigma', default_qG_norm_cutoff_sigma) # qcut = number of gaussian-fit sigmas
        self.qG_norm_cutoff = kwargs.get('qG_norm_cutoff', None)
        self.min_points = kwargs.get('min_points', 6)

        self.initial_guess = kwargs.get('initial_guess', None)
        self.fit_method  =  self.set_fit_method(kwargs.get('fit_method', 'scipy_least_squares'))
        self.force_positive_parameters = kwargs.get('force_positive_parameters', True)

        # Structure factor build settings
        self.N_local = kwargs.get('N_local', kmf.cell.mesh) # Array (3), controls real space grid for structure factor
        self.sq_ke_cutoff = kwargs.get('sq_ke_cutoff', None) 
        self.sq_inversion_symm = kwargs.get('sq_inversion_symm', True)
        self.line_sampling = kwargs.get('line_sampling', False)
        self.pair_density_eval_grid = normalize_pair_density_eval_grid(
            kwargs.get('pair_density_eval_grid', 'becke')
        )
        self.pair_density_becke_grid_level = (
            normalize_pair_density_becke_grid_level(
                kwargs.get('pair_density_becke_grid_level', 0)
            )
        )

        if self.sq_ke_cutoff is not None:
            print("sq_ke_cutoff provided to ExxSS, overriding N_local")
            self.N_local = self.cell.cutoff_to_mesh(self.sq_ke_cutoff)
            print("Using N_local = ", self.N_local)
        self.Ek_uncorr = None
        self.Ek_probe = None

    def __str__(self):
        return f"ExxSS: {self.__class__.__name__} with auxfunc {self.auxfunc.model}"

    def set_fit_method(self, method):
        print("singularity subtraction initial_guess", self.initial_guess)
        self.fit_method = ExxSS.fit_method_dict[method](self.auxfunc, fit_with_coul=self.fit_with_coul,
                                                        is_contraction=self.auxfunc.is_contraction,
                                                        initial_guess=self.initial_guess)
        return self.fit_method

    def set_structure_factor(self):
        # Build structure factor and set qG grid for fitting
        # assert self.qG_norm_cutoff is not None, "qG_norm_cutoff must be set before building structure factor"
        # traceback.print_stack()
        print("qcut: ", self.qG_norm_cutoff)
        self.structure_factor = ExxStructureFactor(self.kmf, self.N_local,self.sq_ke_cutoff,
                                                            self.qG_norm_cutoff,
                                                            min_points=self.min_points,
                                                            sq_inversion_symm=self.sq_inversion_symm,
                                                            line_sampling=(
                                                                self.line_sampling
                                                                and self.qG_norm_cutoff is not None
                                                            ),
                                                            pair_density_eval_grid=self.pair_density_eval_grid,
                                                            pair_density_becke_grid_level=self.pair_density_becke_grid_level)

        self.SqG = self.structure_factor.build_structure_factor()

        self.qG_grid_fit = self.structure_factor.grids.qG_grid_truncated
        self.SqG_fit = self.structure_factor.SqG_truncated


    def compute_uncorrected_exx(self):
        if self.Ek_uncorr is None:
            mf = self.kmf
            dm_kpts = mf.make_rdm1()
            kpts = mf.kpts
            nks = get_monkhorst_pack_size(mf.cell, kpts)
            nk = np.prod(nks)


            mf.exxdiv = None  # so that standard energy is computed without madelung
            jk_kwargs = {
                "cell": mf.cell,
                "dm_kpts": dm_kpts,
                "kpts": kpts,
                # RSJK caches the combined J/K result from SCF.  Requesting
                # only K here can make its incremental cache shapes
                # incompatible (one output versus two), so retain the SCF
                # J/K call shape and discard J below.
                "with_j": True,
                "exxdiv": None,
            }
            # Direct RSDF does not implement the band-k-point interface.  This
            # calculation evaluates K on the SCF mesh itself, so kpts_band=kpts
            # is redundant and can be omitted for that backend.
            if not (
                isinstance(mf.with_df, df.RSDF)
                and bool(getattr(mf.with_df, "direct", False))
            ):
                jk_kwargs["kpts_band"] = kpts
            J, K = mf.get_jk(**jk_kwargs)
            mf.exxdiv = 'ewald'

            Ek_uncorr = -1. / nk * np.einsum('kij,kji', dm_kpts, K) * 0.5
            Ek_uncorr /= 2.
            Ek_uncorr = Ek_uncorr.real
            self.Ek_uncorr = Ek_uncorr

        return self.Ek_uncorr

    def compute_exx_probe(self):
        chi_regular = madelung(self.cell, self.kmf.kpts)
        Ek_probe = self.compute_uncorrected_exx() - self.nocc * chi_regular
        self.Ek_probe = Ek_probe
        return Ek_probe

    @staticmethod
    def build_probe_cell(cell,Nk):
        import copy
        cell_input = copy.deepcopy(cell)
        cell_input._atm = np.array([[1, cell._env.size, 0, 0, 0, 0]])
        cell_input._env = np.append(cell._env, [0., 0., 0.])
        cell_input.unit = 'B' # ecell.verbose = 0
        cell_input.a = np.einsum('xi,x->xi', cell.lattice_vectors(), Nk)
        cell_input._mesh = cell._mesh # hack to overcome attribute error
        cell_input.build()
        return cell_input

    def compute_integral_term(self):
        self.integral_term = self.nocc * self.auxfunc.coulomb_integral()
        print("Integral term = %.15g" % (self.integral_term))
        return self.integral_term

    def compute_quadrature_term(self):
        printstr = "Computing quadrature term for Exact Exchange SS"

        print(printstr)
        # Make ew_eta into array to allow for anisotropy if len==3

        Nk = get_monkhorst_pack_size(self.cell, self.kmf.kpts)
        cell_input = ExxSS.build_probe_cell(self.cell,Nk)

        ke_cutoff = self.cell.ke_cutoff
        # Get FFT mesh from cutoff value
        mesh = cell_input.cutoff_to_mesh(ke_cutoff)

        # Get grid
        Gv, Gvbase, weights = cell_input.get_Gv_weights(mesh=mesh)

        # Get q+G points
        shifted = np.array([0., 0., 0.])
        G_combined = Gv + shifted
        # Calculate |q+G|^2 values of the shifted points
        qG2 = np.einsum('gi,gi->g', G_combined, G_combined)

        qG2[qG2 == 0] = 1e200
        component = 4 * np.pi / qG2 * self.auxfunc.eval_model(G_combined)

        sum_term = weights*np.einsum('i->',component).real
        sum_term *= self.nocc
        self.quadrature_term = sum_term

        print("Quadrature term = %.15g" % (sum_term))
        return sum_term

    def optimize_parameters(self):
        if self.qG_norm_cutoff_sigma is not None and self.qG_norm_cutoff is None:
            sigma = self.compute_approximate_sigma()
            # Then set qg_cutoff to a multiple of sigma
            self.qG_norm_cutoff = sigma * self.qG_norm_cutoff_sigma
            self.set_structure_factor()
        elif self.qG_norm_cutoff is None:
            raise ValueError("qG_norm_cutoff must be set before optimizing parameters")

        fitted_parameters = self.fit_method.fit_model(self.qG_grid_fit, self.SqG_fit/self.nocc,)
        print('optimize_parameters: fitted_parameters = ', fitted_parameters)
        self.auxfunc.set_parameters(fitted_parameters)
        self.model_params = fitted_parameters
        return fitted_parameters

    def compute_correction(self):

        ss_start = time.time()
        print("SS Correction for Exact Exchange")
        print(f"Using {self.__class__.__name__} with {self.auxfunc.__class__.__name__}")
        nocc = self.nocc

        self.set_structure_factor()

        # If sigma is not provided, fit to structure factor
        if self.model_params is None or self.fit_to_structure_factor:
            self.optimize_parameters()

        # Compute Exchange Energies
        Ek_uncorr = self.compute_uncorrected_exx()
        Ek_probe = self.compute_exx_probe()

        self.Ek_uncorr = Ek_uncorr
        self.Ek_probe = Ek_probe

        # Compute SS-NG exchange energy
        quadrature_term = self.compute_quadrature_term()
        integral_term = self.compute_integral_term()
        correction = quadrature_term - integral_term
        chi = - correction / nocc
        Ek = Ek_uncorr - nocc * chi

        self.Ek_ss = Ek
        self.integral_term = -integral_term
        self.quadrature_term = quadrature_term

        # Set general attributes
        self.E_uncorr = Ek_uncorr
        self.E_ss = Ek
        self.correction = correction
        self.chi = chi

        self.print_results()

        print('Total time for ExxSS: %.2f seconds' % (time.time() - ss_start))
        return correction

    def compute_approximate_sigma(self):
        print(f"{self.__class__.__name__} optimizing parameters with {self.qG_norm_cutoff_sigma} Gaussian sigmas")
        gauss_func = model_function.ContractedGaussianModel(num_gaussians=1, isotropic=True, parameters=[1., 1.])
        gaussian_fitting = ExxScipyLeastSquares(gauss_func, fit_with_coul=self.fit_with_coul)
        temp_structure_factor = ExxStructureFactor(self.kmf, self.N_local,self.sq_ke_cutoff,
                                                            self.qG_norm_cutoff,
                                                            min_points=self.min_points,
                                                            sq_inversion_symm=self.sq_inversion_symm,
                                                            pair_density_eval_grid=self.pair_density_eval_grid,
                                                            pair_density_becke_grid_level=self.pair_density_becke_grid_level)
        temp_SqG = temp_structure_factor.build_structure_factor()
        temp_qG_grid = temp_structure_factor.grids.qG_grid_truncated

        fitted_parameters = gaussian_fitting.fit_model(temp_qG_grid, temp_SqG/self.nocc)
        _, sigma = fitted_parameters

        return sigma


    def print_results(self):
        print(f"=== Results for {self.__class__.__name__} with {self.auxfunc.__class__.__name__} ===")

        print("Uncorrected EXX Energies:")
        print(f" HF exact exchange energy (hartree)       = {self.Ek_uncorr}")
        print(f" HF exact exchange probe energy (hartree) = {self.Ek_probe}")
        print()

        super().print_results()

        print("Final Energies:")
        print(f" Corrected HF exact exchange energy (hartree) = {self.Ek_ss}")
        print(f" Chi per occupied orbital (hartree)           = {self.chi}")


class ExxSSGaussian(ExxSS):
    def __init__(self, kmf, dim=3, **kwargs):
        self.kmf = kmf

        # Get parameter from sigma. If model_params is provided, override sigma.
        self.sigma = kwargs.get('sigma', None)
        self.model_params = None
        if self.sigma is not None:
            self.model_params = [1., self.sigma]
        self.model_params = kwargs.get('model_params') if kwargs.get('model_params') is not None else self.model_params
        self.qG_norm_cutoff_sigma = kwargs.get('qG_norm_cutoff_sigma', 1.0)

        self.min_points = kwargs.get('min_points', 6)
        self.N_local = kwargs.get('N_local', kmf.cell.mesh)
        self.qG_norm_cutoff = kwargs.get('qG_norm_cutoff', None)
        print(self.model_params)
        auxfunc = model_function.ContractedGaussianModel(num_gaussians=1, isotropic=True, parameters=self.model_params)
        super().__init__(kmf, auxfunc, dim, model_params=self.model_params, **kwargs)

    def optimize_parameters(self):
        if self.qG_norm_cutoff_sigma is not None:
            sigma = self.compute_approximate_sigma()
            # Then set qg_cutoff to a multiple of sigma
            self.qG_norm_cutoff = sigma * self.qG_norm_cutoff_sigma
            self.set_structure_factor()

        # Otherwise, the minimum truncated grid will serve as the cutoff
        fitted_parameters = self.fit_method.fit_model(self.qG_grid_fit, self.SqG_fit/self.nocc,)
        print('optimize_parameters: fitted_parameters = ', fitted_parameters)
        self.auxfunc.set_parameters(fitted_parameters)
        self.model_params = fitted_parameters
        return fitted_parameters

class ExxSSQuarticExponential(ExxSS):
    def __init__(self, kmf, dim=3, **kwargs):
        self.kmf = kmf

        # Get parameter from alpha, gamma. If model_params is provided, override both.
        self.alpha = kwargs.get('alpha', None)
        self.gamma1 = kwargs.get('gamma1', None)
        self.gamma2 = kwargs.get('gamma2', None)
        self.model_params = None
        if self.alpha is not None and self.gamma1 is not None and self.gamma2 is not None:
            self.model_params = [1., self.alpha, self.gamma1, self.gamma2]
        self.model_params = kwargs.get('model_params') if kwargs.get('model_params') is not None else self.model_params

        self.min_points = kwargs.get('min_points', 6)
        self.N_local = kwargs.get('N_local', kmf.cell.mesh)
        self.qG_norm_cutoff = kwargs.get('qG_norm_cutoff', None)
        auxfunc = model_function.QuarticExponentialModel(num_primitives=1, parameters=self.model_params,)
        super().__init__(kmf, auxfunc, dim, **kwargs)

    def optimize_parameters(self):
        if self.qG_norm_cutoff is None and self.qG_norm_cutoff_sigma is not None:
            # First, find sigma via Gaussian fitting
            print(f"{self.__class__.__name__} optimizing parameters with {self.qG_norm_cutoff_sigma} Gaussian sigmas")
            gauss_func = model_function.ContractedGaussianModel(num_gaussians=1, isotropic=True, parameters=[1., 1.])
            gaussian_fitting = ExxScipyLeastSquares(gauss_func, fit_with_coul=self.fit_with_coul)
            temp_structure_factor = ExxStructureFactor(self.kmf, self.N_local,self.sq_ke_cutoff,
                                                                self.qG_norm_cutoff,
                                                                min_points=self.min_points,
                                                                sq_inversion_symm=self.sq_inversion_symm,
                                                                pair_density_eval_grid=self.pair_density_eval_grid,
                                                                pair_density_becke_grid_level=self.pair_density_becke_grid_level)
            temp_SqG = temp_structure_factor.build_structure_factor()
            temp_qG_grid = temp_structure_factor.grids.qG_grid_truncated


            fitted_parameters = gaussian_fitting.fit_model(temp_qG_grid, temp_SqG/self.nocc)
            _, sigma = fitted_parameters

            # Then set qg_cutoff to sigma
            self.qG_norm_cutoff = sigma*self.qG_norm_cutoff_sigma
            self.set_structure_factor()
        elif self.qG_norm_cutoff is None:
            raise ValueError("qG_norm_cutoff must be set before optimizing parameters")

        # Now fit the Exponential model
        norm_qG = np.linalg.norm(self.qG_grid_fit, axis=1) # workaround for the model_function.QuarticExponentialModel being isotropic
        fitted_parameters = self.fit_method.fit_model(norm_qG, self.SqG_fit/self.nocc, force_positive_params=True)
        self.auxfunc.set_parameters(fitted_parameters)
        self.model_params = fitted_parameters
        return fitted_parameters


class ExxSSEvenTempered(ExxSS):
    """
    ExxSS with an even-tempered Gaussian auxiliary function f(q) = sum_k c_k exp(-alpha_k |q|^2), c_k >= 0,
    sum_k c_k = 1. The exponents are fixed, centered on the expected width sigma (alpha_k = alpha_sigma *
    ratio**(k - (N-1)/2), alpha_sigma = 1/(2 sigma^2)), so only the coefficients are fit. By default they are
    found by direct nonnegative least squares ('etb_linear').

    Additional keyword arguments:
        sigma (float): Expected width of S(q) in reciprocal space. Default is 1.0.
        num_gaussians (int): Number of even-tempered Gaussians N. Default is 5.
        ratio (float): Ratio between consecutive exponents. Default is 2.0.
        exponents (array-like or None): Explicit exponents, overriding sigma/num_gaussians/ratio grid.
        rcond (float): Relative singular-value cutoff for deciding whether the fit data determine the
            coefficients. Default is 1e-8.
        min_shell_margin (int): Number of distinct nonzero |q| shells required beyond the number of free
            coefficients. Default is 1.

    If the q+G cutoff is chosen automatically (qG_norm_cutoff_sigma), it is enlarged as needed so that the fit
    region contains enough distinct |q| shells to determine all N coefficients. If the fit data still cannot
    determine them (e.g. an explicit small qG_norm_cutoff), the basis is reduced around the Gaussian at sigma.
    """

    def __init__(self, kmf, dim=3, sigma=1.0, num_gaussians=5, ratio=2.0, exponents=None, **kwargs):
        self.kmf = kmf
        self.rcond = kwargs.pop('rcond', DEFAULT_RCOND)
        self.min_shell_margin = kwargs.pop('min_shell_margin', DEFAULT_MIN_SHELL_MARGIN)
        self.qG_norm_cutoff_sigma = kwargs.get('qG_norm_cutoff_sigma', 1.0)
        self.fit_method_name = kwargs.setdefault('fit_method', 'etb_linear')
        model_params = kwargs.pop('model_params', None)

        auxfunc = model_function.EvenTemperedGaussianModel(
            sigma=sigma, num_gaussians=num_gaussians, ratio=ratio, exponents=exponents, parameters=model_params,
        )
        self.user_initial_guess = kwargs.get('initial_guess') is not None
        if not self.user_initial_guess:
            kwargs['initial_guess'] = auxfunc.default_parameters()
        super().__init__(kmf, auxfunc, dim, model_params=model_params, **kwargs)
        self.fit_method.rcond = self.rcond

    def required_shells(self):
        """Distinct nonzero |q| shells needed to determine the N - 1 free coefficients, plus the margin."""
        return self.auxfunc.num_gaussians - 1 + self.min_shell_margin

    def ensure_min_shells(self):
        """Enlarge an automatically chosen qG_norm_cutoff until the fit region has enough |q| shells."""
        grids = ExxSSGrids(self.cell, self.kmf.kpts, N_local=self.N_local)
        grids.build_GptGrid3D()
        if self.line_sampling:
            candidates = grids.build_qG_line_sampling(
                qG_norm_cutoff=np.max(np.linalg.norm(grids.GptGrid3D, axis=1)))
        else:
            candidates = (grids.qGrid[:, None, :] + grids.GptGrid3D[None, :, :]).reshape(-1, 3)
        candidate_norms = np.linalg.norm(candidates, axis=1)

        required = self.required_shells()
        n_shells = len(radial_shells(candidate_norms[candidate_norms < self.qG_norm_cutoff]))
        if n_shells >= required:
            return self.qG_norm_cutoff

        cutoff = cutoff_for_min_shells(candidate_norms, required)
        if cutoff is None:
            print(f"WARNING: only {len(radial_shells(candidate_norms))} distinct |q| shells are available; "
                  "the ETB basis will be reduced.")
            return self.qG_norm_cutoff
        print(f"ETB: qG_norm_cutoff {self.qG_norm_cutoff} contains {n_shells} distinct nonzero |q| shells; "
              f"enlarging to {cutoff} to include {required}.")
        self.qG_norm_cutoff = cutoff
        return cutoff

    def trim_auxfunc_to_fit_data(self):
        """Reduce the ETB basis if the fit data cannot determine all coefficients."""
        input_data = self.qG_grid_fit
        if isinstance(self.auxfunc, model_function.IsotropicModel):
            input_data = np.linalg.norm(input_data, axis=1)
        weights = coulomb_residual_weights(input_data, True, self.fit_with_coul, self.fit_method.coul_deg)
        trimmed = trim_etb_model_to_data(self.auxfunc, self.qG_grid_fit, weights, rcond=self.rcond,
                                         min_shell_margin=self.min_shell_margin, label="ExxSS ETB")
        if trimmed is not self.auxfunc:
            self.auxfunc = trimmed
            if not self.user_initial_guess:
                self.initial_guess = trimmed.default_parameters()
            self.set_fit_method(self.fit_method_name)
            self.fit_method.rcond = self.rcond
        return self.auxfunc

    def optimize_parameters(self):
        if self.qG_norm_cutoff_sigma is not None and self.qG_norm_cutoff is None:
            sigma = self.compute_approximate_sigma()
            # Then set qg_cutoff to a multiple of sigma, with enough |q| shells for the ETB coefficients
            self.qG_norm_cutoff = sigma * self.qG_norm_cutoff_sigma
            self.ensure_min_shells()
            self.set_structure_factor()
        elif self.qG_norm_cutoff is None:
            raise ValueError("qG_norm_cutoff must be set before optimizing parameters")

        self.trim_auxfunc_to_fit_data()
        fitted_parameters = self.fit_method.fit_model(self.qG_grid_fit, self.SqG_fit/self.nocc,
                                                      force_positive_params=self.force_positive_parameters)
        print('optimize_parameters: fitted_parameters = ', fitted_parameters)
        print('ETB exponents: ', self.auxfunc.etb_exponents)
        self.auxfunc.set_parameters(fitted_parameters)
        self.model_params = fitted_parameters
        return fitted_parameters
