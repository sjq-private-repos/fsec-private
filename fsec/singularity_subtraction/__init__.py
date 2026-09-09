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


def __getattr__(name):
    """Load the optional periodic G0W0 implementation on first use."""
    if name == "G0W0SS":
        from fsec.singularity_subtraction.g0w0ss import G0W0SS

        return G0W0SS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SingularitySubtraction",
    "ExxSS",
    "ExxSSGaussian",
    "ExxSSQuarticExponential",
    "DirectFourthOrderCorrectionConfig",
    "DirectFullCorrectionConfig",
    "DirectSecondOrderCorrectionConfig",
    "DirectCorrectionResult",
    "ExchangeCorrectionConfig",
    "ExchangeCorrectionResult",
    "MP2SS",
    "MP2SSOptions",
    "MP2DirectSS",
    "MP2ExchangeSS",
    "OriginDiagnostics",
    "G0W0SS",
]
