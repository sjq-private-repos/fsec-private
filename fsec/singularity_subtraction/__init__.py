from abc import ABC, abstractmethod


class SingularitySubtraction(ABC):
    results_title = None
    integral_term = None
    quadrature_term = None
    correction = None

    @abstractmethod
    def compute_integral_term(self):
        """
        Compute the integral term for singularity subtraction.
        """
        pass

    @abstractmethod
    def compute_quadrature_term(self):
        """
        Compute the quadrature term for singularity subtraction.
        """
        pass

    @abstractmethod
    def compute_correction(self):
        """
        Compute the overall correction using singularity subtraction.
        Should set the following attributes:
            self.E_ss: the SS corrected energy value (e.g. exchange, mp2 correlation)
            self.correction: the correction to the exchange energy (E_ss = E_uncorr + correction)
            self.chi: total correction divided by the number of occupied orbitals
        """
        pass

    def print_results(self):
        """Print the integral, quadrature, and total correction for one term."""
        title = self.results_title or self.__class__.__name__
        print(f"{title}:")
        print(f" Integral term (hartree)    = {self.integral_term}")
        print(f" Quadrature term (hartree)  = {self.quadrature_term}")
        print(f" Total correction (hartree) = {self.correction}")
        print()


from fsec.singularity_subtraction.exxss import ExxSS, ExxSSGaussian, ExxSSQuarticExponential
from fsec.singularity_subtraction.mp2ss import (
    DirectFourthOrderCorrectionConfig,
    DirectFullCorrectionConfig,
    DirectSecondOrderCorrectionConfig,
    DirectCorrectionResult,
    ExchangeCorrectionConfig,
    ExchangeCorrectionResult,
    MP2SS,
    MP2SSOptions,
    MP2DirectSS,
    MP2ExchangeSS,
)
from fsec.singularity_subtraction.analysis import OriginDiagnostics
from fsec.singularity_subtraction.ccdss import CCDSSOptions, KRCCD_SS
