"""Private Coulomb-kernel evaluation for MP2SS direct quadratures."""

from types import SimpleNamespace

import numpy as np
from pyscf.pbc import tools


_KERNEL_NAMES = ("bare", "vcut_ws", "vcut_sph")


def _validate_kernel_options(name, eta):
    name = str(name).strip().lower()
    if name not in _KERNEL_NAMES:
        raise ValueError(
            "quadrature_coulomb_kernel must be 'bare', 'vcut_ws', or "
            "'vcut_sph'"
        )
    if eta is not None:
        try:
            eta = float(eta)
        except (TypeError, ValueError) as err:
            raise ValueError("eta must be finite and positive or None") from err
        if not np.isfinite(eta) or eta <= 0:
            raise ValueError("eta must be finite and positive or None")
        if name == "bare":
            raise ValueError("eta is only valid with a vcut quadrature kernel")
    return name, eta


class _QuadratureCoulombKernel:
    """Evaluate a direct-quadrature Coulomb kernel on unwrapped q+G points."""

    def __init__(self, cell, kpts, name="bare", eta=None):
        name, eta = _validate_kernel_options(name, eta)

        self.cell = cell
        self.name = name
        self.eta = eta
        self.kpts = None
        self._context = None
        self._stc_prepared = False

        if name != "bare":
            if getattr(cell, "dimension", None) != 3:
                raise ValueError(
                    "vcut quadrature kernels require a three-dimensional cell"
                )
            if kpts is None:
                raise ValueError(
                    "vcut quadrature kernels require the complete regular k mesh"
                )
            kpts = np.asarray(getattr(kpts, "kpts", kpts)).reshape(-1, 3)
            if len(kpts) == 0:
                raise ValueError(
                    "vcut quadrature kernels require the complete regular k mesh"
                )
            try:
                kmesh = np.asarray(
                    tools.get_monkhorst_pack_size(cell, kpts), dtype=int
                )
            except (AssertionError, RuntimeError, ValueError) as err:
                raise ValueError(
                    "vcut quadrature kernels require a complete regular k mesh"
                ) from err
            scaled = cell.get_scaled_kpts(kpts - kpts[0])
            mesh_coordinates = scaled * kmesh
            indices = np.rint(mesh_coordinates).astype(int) % kmesh
            if (len(kpts) != np.prod(kmesh)
                    or np.max(np.abs(mesh_coordinates - np.rint(mesh_coordinates)))
                    >= 1e-6
                    or len(np.unique(indices, axis=0)) != len(kpts)):
                raise ValueError(
                    "vcut quadrature kernels require a complete regular k mesh"
                )

            self.kpts = np.array(kpts, copy=True)
            self._context = SimpleNamespace(
                cell=cell,
                kpts=self.kpts,
                _ws_exx=None,
            )

    def _prepare_stc(self):
        try:
            from pyscf.pbc.df.fft_stc import _prepare_stc
        except ImportError as err:
            raise RuntimeError(
                "finite-eta MP2SS quadrature requires the custom PySCF sTC "
                "support (pyscf.pbc.df.fft_stc)"
            ) from err

        context = self._context
        context._scf_kpts = self.kpts
        context.eta = self.eta
        context.exxdiv = self.name
        context.rc_type = "ws" if self.name == "vcut_ws" else "sph"
        _prepare_stc(context)
        self._stc_prepared = True

    def __call__(self, qG):
        """Return V(q+G), retaining finite values at q+G=0."""
        qG = np.asarray(qG)
        if self.name == "bare":
            raise ValueError("the bare quadrature kernel uses legacy arithmetic")

        if self.eta is None:
            return tools.get_coulG(
                self.cell,
                Gv=qG,
                exx=self.name,
                mf=self._context,
                wrap_around=False,
                omega=0,
            )

        if not self._stc_prepared:
            self._prepare_stc()
        vcut = tools.get_coulG(
            self.cell,
            Gv=qG,
            exx=self.name,
            mf=self._context,
            wrap_around=False,
            omega=0,
        )
        omega = self._context.omega_stc
        qG2 = np.einsum("gi,gi->g", qG, qG)
        exponent = -qG2 / (4 * omega**2)
        damping = np.exp(exponent)
        kernel = vcut * damping
        nonzero = qG2 != 0
        kernel[nonzero] += -4 * np.pi * np.expm1(exponent[nonzero]) / qG2[nonzero]
        kernel[~nonzero] = vcut[~nonzero] + np.pi / omega**2
        return kernel
