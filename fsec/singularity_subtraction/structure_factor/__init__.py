from abc import ABC, abstractmethod
import time
from fsec.singularity_subtraction.grids import minimum_image, ExxSSGrids, MP2SSGrids
from pyscf.pbc.tools import get_monkhorst_pack_size
import scipy
from pyscf.pbc.tools.pbc import mesh_to_cutoff, cutoff_to_mesh
from pyscf.lib import logger
from pyscf import lib
from pyscf.lib.parameters import LARGE_DENOM
from pyscf.pbc import df, dft, scf
from pyscf.lib import logger, einsum
from pyscf.pbc.mp import kmp2
import numpy as np
from fsec.singularity_subtraction.structure_factor.helpers import build_uKpts as _build_uKpts
from scipy.spatial import KDTree
from pyscf.lib.numpy_helper import einsum as pyscf_einsum



class StructureFactor(ABC):
    def __init__(self, cell, N_local=None, sq_ke_cutoff=None, qG_cutoff=None,**kwargs):
        """
        Initialize the structure factor with density matrix (dm), cell, and optional parameters.
        """
        self.cell = cell
        self.N_local = N_local
        self.sq_ke_cutoff = sq_ke_cutoff

        # if N_local and sq_ke_cutoff both provided, go with denser mesh.
        if self.N_local is not None:
            if self.sq_ke_cutoff is not None:
                ke_cutoff_Nlocal = mesh_to_cutoff(self.cell.lattice_vectors(), self.N_local)
                print(ke_cutoff_Nlocal)
                if np.min(ke_cutoff_Nlocal) > self.sq_ke_cutoff:
                    print("Using N_local mesh")
                    self.sq_ke_cutoff = ke_cutoff_Nlocal
                else:
                    print("Using sq_ke_cutoff mesh")
                    self.N_local = cutoff_to_mesh(self.cell.lattice_vectors(), self.sq_ke_cutoff)
            else:
                self.sq_ke_cutoff = mesh_to_cutoff(self.cell.lattice_vectors(), self.N_local)
        else:
            if self.sq_ke_cutoff is not None:
                self.N_local = cutoff_to_mesh(self.cell.lattice_vectors(), self.sq_ke_cutoff)

        self.qG_cutoff = qG_cutoff
        self.sq_inversion_symm = kwargs.get('sq_inversion_symm', True) # S(x,y,z) = S(-x,-y,-z)
        self.sq_isotropic_symm = kwargs.get('sq_isotropic_symm', True) # S(x,y,z) = S(-x,y,z) = S(x,-y,z) = S(x,y,-z)
        self.dump_flags()

    def update_sq_ke_cutoff(self, sq_ke_cutoff):
        """
        Update the sq_ke_cutoff value, and change corresponding N_local mesh.
        """
        self.sq_ke_cutoff = sq_ke_cutoff
        self.N_local = cutoff_to_mesh(self.cell.lattice_vectors(), self.sq_ke_cutoff)
        print("Updated sq_ke_cutoff: ", self.sq_ke_cutoff)
        print("Updated N_local: ", self.N_local)

    def update_N_local(self, N_local):
        """
        Update the N_local value, and change corresponding sq_ke_cutoff.
        """
        self.N_local = N_local
        self.sq_ke_cutoff = mesh_to_cutoff(self.cell.lattice_vectors(), self.N_local)
        print("Updated N_local: ", self.N_local)
        print("Updated sq_ke_cutoff: ", self.sq_ke_cutoff)

    @abstractmethod
    def build_structure_factor(self):
        """
        Build the structure factor. Must be implemented by subclasses.
        """
        pass

    def dump_flags(self):
        """
        Dump the flags used in the fitting. Prints class name and key parameters
        """
        print("StructureFactor class <{}>".format(self.__class__.__name__))
        print("    N_local (ke_cutoff): {} ({})".format(self.N_local, self.sq_ke_cutoff))
        print("    Inversion symmetry: ", self.sq_inversion_symm)
        print("    qG_cutoff: ", self.qG_cutoff)
    
    @staticmethod
    def build_uKpts(kmf, kpts, mo_coeff_kpts, NsCell=None, rptGrid3D=None, nbands=None):
        return _build_uKpts(kmf, kpts, mo_coeff_kpts, NsCell=NsCell, rptGrid3D=rptGrid3D, nbands=nbands)

from fsec.singularity_subtraction.structure_factor.exx_sf import ExxStructureFactor
from fsec.singularity_subtraction.structure_factor.mp2_sf import MP2StructureFactor
from fsec.singularity_subtraction.structure_factor.mp2_smallq import (
    MP2SmallQ,
    MP2SmallQResult,
)
