from abc import ABC, abstractmethod
import numpy as np
from pyscf.pbc.tools import get_monkhorst_pack_size

def minimum_image(cell, kpts):
    """
    Compute the minimum image of k-points in 'kpts' in the first Brillouin zone

    Arguments:
        cell -- a cell instance
        kpts -- a list of k-points

    Returns:
        kpts_bz -- a list of k-point in the first Brillouin zone
    """
    tmp_kpt = cell.get_scaled_kpts(kpts)
    tmp_kpt = tmp_kpt - np.floor(tmp_kpt)
    tmp_kpt[tmp_kpt > 0.5 - 1e-8] -= 1
    kpts_bz = cell.get_abs_kpts(tmp_kpt)
    return kpts_bz

def build_N_local_grid(N_local_x, N_local_y, N_local_z, Lvec_recip):
    if N_local_x % 2 == 1:
        Grid_1D_x = np.concatenate((np.arange(0, (N_local_x - 1) // 2 + 1), np.arange(-(N_local_x - 1) // 2, 0)))
    else:
        # At low Nlocal/Nk, this matters, because we want the direction where G is incremented to be opposite of
        # the default direction of a boundary-value q.
        Grid_1D_x = np.concatenate((np.arange(0, N_local_x // 2 + 1), np.arange(-N_local_x // 2 +1, 0)))
    if N_local_y % 2 == 1:
        Grid_1D_y = np.concatenate((np.arange(0, (N_local_y - 1) // 2 + 1), np.arange(-(N_local_y - 1) // 2, 0)))
    else:
        Grid_1D_y = np.concatenate((np.arange(0, N_local_y // 2 + 1), np.arange(-N_local_y // 2 +1, 0)))

    if N_local_z % 2 == 1:
        Grid_1D_z = np.concatenate((np.arange(0, (N_local_z - 1) // 2 + 1), np.arange(-(N_local_z - 1) // 2, 0)))
    else:
        Grid_1D_z = np.concatenate((np.arange(0, N_local_z // 2 + 1), np.arange(-N_local_z // 2 +1, 0)))

    Gxx_local, Gyy_local, Gzz_local = np.meshgrid(Grid_1D_x, Grid_1D_y, Grid_1D_z, indexing='ij')
    GptGrid3D_local = np.hstack(
        (Gxx_local.reshape(-1, 1), Gyy_local.reshape(-1, 1), Gzz_local.reshape(-1, 1))) @ Lvec_recip
    return GptGrid3D_local

class SSGrids(ABC):
    def __init__(self, cell, N_local=None):
        """
        Initialize the grid structure with a given cell and optional local grid size.
        """
        self.cell = cell
        if N_local is not None:
            self.N_local = np.array(N_local)
        else:
            self.N_local = None

    @abstractmethod
    def build_grids(self):
        """
        Construct the relevant grids. Must be implemented by subclasses.
        """
        pass

    def build_GptGrid3D(self,NsCell=None):
        """
        Build the reciprocal lattice for the cell.
        """
        NsCell = self.cell.mesh if NsCell is None else NsCell
        Lvec_recip = self.cell.reciprocal_vectors()

        Gx = np.fft.fftfreq(NsCell[0], d=1 / NsCell[0])
        Gy = np.fft.fftfreq(NsCell[1], d=1 / NsCell[1])
        Gz = np.fft.fftfreq(NsCell[2], d=1 / NsCell[2])
        Gxx, Gyy, Gzz = np.meshgrid(Gx, Gy, Gz, indexing='ij')
        self.GptGrid3D = np.hstack((Gxx.reshape(-1, 1), Gyy.reshape(-1, 1), Gzz.reshape(-1, 1))) @ Lvec_recip
        return self.GptGrid3D
    
    def build_RptGrid3D(self):
        """
        Build the real space grid for the cell.
        """
        Lvec_real = self.Lvec_real
        NsCell = self.cell.mesh
        L_delta = Lvec_real / NsCell[:, None]

        # Evaluate wavefunction on all real space grid points
        # # Establishing real space grid (Generalized for arbitary volume defined by 3 vectors)
        xv, yv, zv = np.meshgrid(np.arange(NsCell[0]), np.arange(NsCell[1]), np.arange(NsCell[2]), indexing='ij')
        mesh_idx = np.hstack([xv.reshape(-1, 1), yv.reshape(-1, 1), zv.reshape(-1, 1)])
        self.RptGrid3D = mesh_idx @ L_delta
        return self.RptGrid3D


class ExxSSGrids(SSGrids):
    def __init__(self, cell, kGrid1, kGrid2=None, N_local=None, dim=3,
                 relative_shift=[0.0, 0.0, 0.0], shift_occ=True,
                 qG_norm_cutoff=None, min_points=6):
        self.cell = cell
        self.kGrid1 = minimum_image(cell,kGrid1)
        self.nks = get_monkhorst_pack_size(cell,kGrid1)
        self.Lvec_recip = cell.reciprocal_vectors()
        self.qG_norm_cutoff = qG_norm_cutoff
        self.min_points = min_points
        if kGrid2 is None:
            if np.all(np.isclose(relative_shift, 0.0)):
                self.kGrid2 = kGrid1
            else:
                kshift_abs = cell.get_abs_kpts([shift / n for shift,n in zip(relative_shift,self.nks)])
                if shift_occ:
                    kGrid2 = self.kGrid1.copy()
                    self.kGrid2 = kGrid2
                    kGrid1 = minimum_image(cell, kGrid1 + kshift_abs) # q is opposite sign of shift
                    self.kGrid1 = kGrid1

                else:
                    self.kGrid2 = minimum_image(cell, kGrid1 + kshift_abs)

        else:
            self.kGrid2 = kGrid2
        self.qGrid = minimum_image(cell, self.kGrid2 - self.kGrid1[0, :]) # assumes both grids are same size and spacing
        if np.isscalar(N_local):
            self.N_local = np.array([N_local]*dim)
        else:
            self.N_local = np.array(N_local)
        self.Lvec_real = cell.lattice_vectors()

    def build_qG_Grid(self):
        """
        Build the q+G grid.
        """
        self.qG_grid = np.einsum('ij,kj->ikj', self.qGrid,np.ones_like(self.GptGrid3D)).reshape(-1, 3) \
            + np.tile(self.GptGrid3D, (self.qGrid.shape[0], 1))
        self.qG_grid_local = np.einsum('ij,kj->ikj', self.qGrid,np.ones_like(self.GptGrid3D_local)).reshape(-1, 3) \
            + np.tile(self.GptGrid3D_local, (self.qGrid.shape[0], 1))
        if self.qG_norm_cutoff is not None:
            qG_norm = np.linalg.norm(self.qG_grid, axis=1)
            self.qG_grid_truncated = self.qG_grid[qG_norm < self.qG_norm_cutoff]
        else:
            self.qG_grid_truncated = None

        return self.qG_grid

    def build_qG_line_sampling(self, qG_norm_cutoff=None):
        """
        Build q+G samples along each reciprocal lattice direction.
        """
        qG_norm_cutoff = qG_norm_cutoff if qG_norm_cutoff is not None else self.qG_norm_cutoff
        qG_full = []
        line_segments = []
        B_over_nk = self.cell.reciprocal_vectors() / self.nks[:, None]
        next_index = 1
        for i in range(B_over_nk.shape[0]):
            B_i = B_over_nk[i, :]
            B_i_norm = np.linalg.norm(B_i)
            npoints = max(0, int(np.floor(qG_norm_cutoff / B_i_norm)) - 1)
            qG_full_i = np.zeros((npoints, 3))
            for j in range(npoints):
                qG_full_i[j, :] = (j + 1) * B_i
            qG_full.append(qG_full_i)
            indices = np.arange(next_index, next_index + npoints, dtype=int)
            line_segments.append({
                "line_index": i,
                "B_index": i,
                "step_vector": B_i.copy(),
                "q_min": qG_full_i[0].copy() if npoints > 0 else None,
                "indices": indices,
            })
            next_index += npoints
        qG_full = np.concatenate(qG_full, axis=0)
        qG_full = np.concatenate([np.zeros((1, 3)), qG_full], axis=0)
        self.qG_line_sampling_segments = line_segments
        self.qG_line_sampling_metadata = {
            "origin_index": 0,
            "segments": line_segments,
        }
        return qG_full


    def build_RptGrid3D_local(self):
        """
        Build the dual lattice to the q+G grid on the Nlocal^3 BZs.
        """
        nks = self.nks

        # lattice size along each dimension in the real-space (equal to q + G size)
        nqG_local = np.array(self.N_local) * np.array(nks)
        Lvec_real_local = self.Lvec_real / self.N_local  # dual real cell of local domain LsCell_bz_local

        Rx = np.fft.fftfreq(nqG_local[0], d=1 / nqG_local[0])
        Ry = np.fft.fftfreq(nqG_local[1], d=1 / nqG_local[1])
        Rz = np.fft.fftfreq(nqG_local[2], d=1 / nqG_local[2])
        Rxx, Ryy, Rzz = np.meshgrid(Rx, Ry, Rz, indexing='ij')
        self.RptGrid3D_local = np.hstack((Rxx.reshape(-1, 1), Ryy.reshape(-1, 1), Rzz.reshape(-1, 1))) @ Lvec_real_local
        return self.RptGrid3D_local

    def build_RptGrid3D_coarse(self,N_local=None):
        """
        Similar to build_RptGrid3D_local, but only creates the coarse grid in the Wigner-Seitz cell.
        Meant to be used with ExxSS to accelerate construction of the structure factor S(q+G).
        """
        Lvec_real = self.Lvec_real
        NsCell = self.N_local if N_local is None else N_local
        L_delta = Lvec_real / NsCell
        xv, yv, zv = np.meshgrid(np.arange(NsCell[0]), np.arange(NsCell[1]), np.arange(NsCell[2]), indexing='ij')
        mesh_idx = np.hstack([xv.reshape(-1, 1), yv.reshape(-1, 1), zv.reshape(-1, 1)])
        self.RptGrid3D_coarse = mesh_idx @ L_delta
        return self.RptGrid3D_coarse


    def build_grids(self):
        """
        Build all grids
        """
        self.build_GptGrid3D()
        self.GptGrid3D_local = build_N_local_grid(self.N_local[0], self.N_local[1], self.N_local[2],
                                                  self.cell.reciprocal_vectors())
        self.build_qG_Grid()
        self.build_RptGrid3D()
        self.build_RptGrid3D_local()
        self.build_RptGrid3D_coarse()
        if self.qG_norm_cutoff is not None:
            self.build_truncated_qG_grid()

    def build_truncated_qG_grid(self):
        """
        Build a truncated q+G grid.
        """
        GptGrid3D = self.GptGrid3D
        qGrid = self.qGrid
        qG_norm_cutoff = self.qG_norm_cutoff
        cell = self.cell
        N_local = self.N_local
        min_fitting_pts = self.min_points

        print("Computing only necessary SqG")
        import time
        temp_time = time.time()
        qG_full = np.einsum('ij,kj->ikj', qGrid, np.ones_like(GptGrid3D)).reshape(-1, 3) \
            + np.tile(GptGrid3D, (qGrid.shape[0], 1))

        qG_norm = np.linalg.norm(qG_full, axis=1)
        if qG_norm_cutoff is None:
            print("Automatically finding qG norm cutoff")
            unique_qG_norms = np.unique(qG_norm)
            unique_qG_norms = unique_qG_norms[:25]

            num_points = []
            for norm in unique_qG_norms:
                num_points.append(np.sum(qG_norm <= norm + 1e-8))

            qG_norm_cutoff = unique_qG_norms[np.argmax(np.array(num_points) >= min_fitting_pts)] + 1e-8
            print('Computed qG norm cutoff is', qG_norm_cutoff)
            if qG_norm_cutoff < 1e-8:
                raise ValueError("qG_norm_cutoff is too small")

            self.qG_norm_cutoff = qG_norm_cutoff

        print("Using qG norm cutoff, fitting to qG with norm less than", qG_norm_cutoff)

        if max(np.linalg.norm(cell.reciprocal_vectors() * N_local, axis=1)) < qG_norm_cutoff:
            print("NOTE: qG_norm_cutoff is outside the longest dimension of the NlocalBZs")

        temp_time2 = time.time()
        print("Time to compute qG_full", temp_time2 - temp_time)
        self.qG_grid_truncated = qG_full[qG_norm < qG_norm_cutoff]
        print("Number of fitting points: ", self.qG_grid_truncated.shape[0])
        return self.qG_grid_truncated


class MP2SSGrids(ExxSSGrids):
    def __init__(self, cell, kGrid1, N_local=None,
                 qG_norm_cutoff=None, min_points=6, relative_shift=[0.0, 0.0, 0.0], shift_occ=True, **kwargs):
        super().__init__(cell, kGrid1, N_local=N_local, relative_shift=relative_shift, shift_occ=shift_occ, **kwargs)

        kGrid3_neq_kGrid2 = False
        for shift_i in relative_shift:
            if ~np.isclose(shift_i, 0.0, atol=1e-8) and ~np.isclose(shift_i, 0.5, atol=1e-8):
                kGrid3_neq_kGrid2 = True
                break

        if np.all(np.isclose(relative_shift, 0.0)):
            self.kGrid3 = self.kGrid1.copy()
        elif not kGrid3_neq_kGrid2:
            self.kGrid3 = self.kGrid2.copy()
        else:
            kshift_abs = cell.get_abs_kpts([shift / n for shift, n in zip(relative_shift, self.nks)])
            if shift_occ:
                kGrid3 = self.kGrid1.copy()
                self.kGrid3 = minimum_image(cell, kGrid3 + kshift_abs)
            else:
                self.kGrid3 = minimum_image(cell, self.kGrid1.copy() - kshift_abs)

        if qG_norm_cutoff is None:
            print("No qG_norm_cutoff provided, using two reciprocal lattice vectors to find qG_norm_cutoff")
            qG_norm_cutoff = max(np.linalg.norm(cell.reciprocal_vectors() * 2, axis=1))
            print("Using qG_norm_cutoff: ", qG_norm_cutoff)
        else:
            print("Using qG_norm_cutoff: ", qG_norm_cutoff)

        self.qG_norm_cutoff = qG_norm_cutoff
        self.min_points = min_points
        self.kGrid3_neq_kGrid2 = kGrid3_neq_kGrid2

    def build_qG_grid(self, qGrid, GptGrid3D, faster_dim='q'):
        if faster_dim == 'G':
            return np.einsum('ij,kj->ikj', qGrid, np.ones_like(GptGrid3D)).reshape(-1, 3) + np.tile(GptGrid3D, (qGrid.shape[0], 1))
        elif faster_dim == 'q':
            return np.einsum('ij,kj->ikj', GptGrid3D, np.ones_like(qGrid)).reshape(-1, 3) + np.tile(qGrid, (GptGrid3D.shape[0], 1))
        else:
            raise ValueError("faster_dim must be 'q' or 'G'")


    def build_truncated_qG_grid(self, qG_norm_cutoff=None):
        """
        Build a truncated q+G grid.
        """
        GptGrid3D = self.GptGrid3D
        qGrid = self.qGrid
        qG_norm_cutoff = qG_norm_cutoff if qG_norm_cutoff is not None else self.qG_norm_cutoff
        cell = self.cell
        N_local = self.N_local
        min_fitting_pts = self.min_points

        print("Computing only necessary SqG")
        import time
        temp_time = time.time()
        qG_full = np.einsum('ij,kj->ikj', qGrid, np.ones_like(GptGrid3D)).reshape(-1, 3) + np.tile(GptGrid3D, (qGrid.shape[0], 1))

        qG_norm = np.linalg.norm(qG_full, axis=1)
        if qG_norm_cutoff is None:
            print("Automatically finding qG norm cutoff")
            unique_qG_norms = np.unique(qG_norm)
            unique_qG_norms = unique_qG_norms[:25]

            num_points = []
            for norm in unique_qG_norms:
                num_points.append(np.sum(qG_norm <= norm + 1e-8))

            qG_norm_cutoff = unique_qG_norms[np.argmax(np.array(num_points) >= min_fitting_pts)] + 1e-8
            print('Computed qG norm cutoff is', qG_norm_cutoff)
            if qG_norm_cutoff < 1e-8:
                raise ValueError("qG_norm_cutoff is too small")

            self.qG_norm_cutoff = qG_norm_cutoff

        print("Using qG norm cutoff, fitting to qG with norm less than", qG_norm_cutoff)

        if max(np.linalg.norm(cell.reciprocal_vectors() * N_local, axis=1)) < qG_norm_cutoff:
            print("NOTE: qG_norm_cutoff is outside the longest dimension of the NlocalBZs")

        temp_time2 = time.time()
        print("Time to compute qG_full", temp_time2 - temp_time)
        self.qG_grid_truncated = qG_full[qG_norm < qG_norm_cutoff]
        print("Number of fitting points: ", self.qG_grid_truncated.shape[0])
        return self.qG_grid_truncated

    def build_grids(self):
        """
        Build all grids
        """
        super().build_grids()
        self.build_truncated_qG_grid()
