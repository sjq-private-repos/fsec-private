# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Portions of the in-core response and self-energy implementation are adapted
# from pyscf.pbc.gw.krgw_ac.py by Tianyu Zhu, Christopher Hillenbrand, and
# Jiachen Li.  The Gaussian finite-size terms and local orchestration are
# original to FSEC.

"""Gaussian finite-size correction for periodic G0W0.

This module adapts the in-core periodic GW-AC implementation in
``pyscf.pbc.gw.krgw_ac``.  PySCF's response, density-fitting transformation,
analytic-continuation, and quasiparticle routines are retained; only the
kernel orchestration and in-core self-energy loop are local so that the
Gaussian head and wing corrections are evaluated before continuation.

The implementation currently targets three-dimensional, spin-restricted,
gapped calculations on a complete uniform Monkhorst-Pack mesh with GDF
integrals.  The correction is enabled by ``fc=True`` (the default).
"""

from dataclasses import dataclass
from functools import reduce
import itertools
import time

import numpy as np
import scipy.linalg
import scipy.optimize

from pyscf import lib
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.incore import _conc_mos
from pyscf.gw.utils.ac_grid import PadeAC, TwoPoleAC
from pyscf.gw.utils.gw_np_helper import array_scale
from pyscf.lib import einsum, logger, temporary_env
from pyscf.pbc import df
from pyscf.pbc.gw import krgw_ac as _pyscf_krgw_ac
from pyscf.pbc.mp.kmp2 import get_frozen_mask


KRGWAC = _pyscf_krgw_ac.KRGWAC


@dataclass(frozen=True)
class GaussianCoefficients:
    """Analytic and finite-mesh Gaussian coefficients.

    ``head`` and ``wing`` are the coefficients multiplying the inverse
    dielectric head and the diagonal wing contraction, respectively.  The
    integral and quadrature fields are retained separately for diagnostics.
    All reciprocal vectors and the cell volume use atomic units.
    """

    sigma: float
    head_integral: float
    head_quadrature: float
    head: float
    wing_integral: float
    wing_quadrature: float
    wing: float

    def as_dict(self):
        """Return coefficients in a readable mapping for diagnostics."""
        return {
            "sigma": self.sigma,
            "head_integral": self.head_integral,
            "head_quadrature": self.head_quadrature,
            "head": self.head,
            "wing_integral": self.wing_integral,
            "wing_quadrature": self.wing_quadrature,
            "wing": self.wing,
        }


def _periodic_unique(values, tol):
    """Cluster values on a unit-periodic interval."""
    representatives = []
    for value in np.sort(np.mod(np.asarray(values, dtype=float), 1.0)):
        if not any(min(abs(value - rep), 1.0 - abs(value - rep)) < tol for rep in representatives):
            representatives.append(float(value))
    return np.asarray(representatives, dtype=float)


def monkhorst_pack_shape(cell, kpts, tol=1e-8):
    """Return the shape of a complete uniform Monkhorst-Pack mesh.

    The result is independent of k-point ordering and of a common fractional
    shift.  A ``ValueError`` is raised for incomplete or nonuniform meshes.
    """
    scaled = np.mod(np.asarray(cell.get_scaled_kpts(kpts), dtype=float), 1.0)
    if scaled.ndim != 2 or scaled.shape[1] != 3 or len(scaled) == 0:
        raise ValueError("kpts must contain a nonempty array of three-vectors")

    axes = [_periodic_unique(scaled[:, axis], tol) for axis in range(3)]
    shape = np.asarray([len(axis_values) for axis_values in axes], dtype=int)
    if int(np.prod(shape)) != len(scaled):
        raise ValueError("k-points do not form a complete Monkhorst-Pack mesh")

    for axis_values, ngrid in zip(axes, shape):
        if ngrid > 1:
            ordered = np.sort(axis_values)
            spacings = np.diff(np.r_[ordered, ordered[0] + 1.0])
            if not np.allclose(spacings, 1.0 / ngrid, atol=10 * tol, rtol=0):
                raise ValueError("k-points are not uniformly spaced")

    # Check the Cartesian-product topology explicitly.  This catches meshes
    # with the right axis counts but one missing point and one duplicate.
    for candidate in itertools.product(*axes):
        distances = np.max(
            np.minimum(abs(scaled - np.asarray(candidate)), 1.0 - abs(scaled - np.asarray(candidate))),
            axis=1,
        )
        if np.min(distances) > 10 * tol:
            raise ValueError("k-points do not form a complete Cartesian mesh")
    return shape


def _reciprocal_supercell_basis(cell, mesh):
    """Build the unshifted reciprocal-supercell basis as row vectors."""
    mesh = np.asarray(mesh, dtype=float)
    if mesh.shape != (3,) or np.any(mesh < 1) or not np.all(np.equal(mesh, np.rint(mesh))):
        raise ValueError("mesh must contain three positive integers")
    return np.asarray(cell.reciprocal_vectors(), dtype=float) / mesh[:, None]


def _iter_reciprocal_q_chunks(cell, mesh, cutoff, chunk_size=8192):
    """Yield nonzero reciprocal-supercell vectors inside a spherical cutoff.

    Bounds are obtained from the inverse reciprocal-supercell basis, so the
    enumeration remains complete for skew cells.  Chunks keep the Gaussian
    sums bounded in memory.
    """
    basis = _reciprocal_supercell_basis(cell, mesh)
    inverse_basis = np.linalg.inv(basis)
    # m = q @ inverse_basis.  The norm of each column gives a valid integer
    # bound for all q with |q| <= cutoff, including nonorthogonal cells.
    bounds = np.ceil(float(cutoff) * np.linalg.norm(inverse_basis, axis=0)).astype(int)
    pending = []
    cutoff2 = float(cutoff) ** 2
    for integer_vector in itertools.product(
        *[range(-int(bound), int(bound) + 1) for bound in bounds]
    ):
        if not any(integer_vector):
            continue
        q_vector = np.dot(np.asarray(integer_vector, dtype=float), basis)
        if np.dot(q_vector, q_vector) <= cutoff2 * (1.0 + 1e-13):
            pending.append(q_vector)
            if len(pending) == chunk_size:
                yield np.asarray(pending)
                pending = []
    if pending:
        yield np.asarray(pending)


def reciprocal_supercell_lattice(cell, mesh, cutoff, chunk_size=8192):
    """Return nonzero reciprocal-supercell vectors up to ``cutoff``."""
    chunks = list(_iter_reciprocal_q_chunks(cell, mesh, cutoff, chunk_size=chunk_size))
    if not chunks:
        return np.zeros((0, 3), dtype=float)
    return np.concatenate(chunks, axis=0)


def compute_gaussian_coefficients(
    cell,
    nkpts=None,
    mesh=None,
    cutoff_sigma=8.0,
    chunk_size=8192,
):
    """Compute Gaussian head and wing coefficients.

    Parameters
    ----------
    cell
        Three-dimensional periodic PySCF cell.
    nkpts
        Number of k-points.  If ``mesh`` is omitted, a three-element integer
        sequence may be supplied here as the mesh shape.
    mesh
        Three-element Monkhorst-Pack mesh shape.  The common k-point shift is
        intentionally not used.
    cutoff_sigma
        Reciprocal-vector cutoff in units of the Gaussian width.
    chunk_size
        Maximum number of reciprocal vectors accumulated in one chunk.
    """
    if mesh is None:
        if nkpts is None:
            mesh = np.ones(3, dtype=int)
            nk = 1
        else:
            candidate = np.asarray(nkpts)
            if candidate.shape == (3,):
                mesh = candidate.astype(int)
                nk = int(np.prod(mesh))
            else:
                raise ValueError("mesh is required when nkpts is a scalar")
    else:
        mesh = np.asarray(mesh, dtype=int)
        nk = int(np.prod(mesh))
        if nkpts is not None and int(nkpts) != nk:
            raise ValueError("nkpts must equal the product of mesh dimensions")

    if mesh.shape != (3,) or np.any(mesh < 1):
        raise ValueError("mesh must contain three positive integers")
    if not np.isfinite(cell.vol) or cell.vol <= 0:
        raise ValueError("cell volume must be positive")

    volume = float(cell.vol)
    sigma = (6.0 * np.pi**2 / (volume * nk)) ** (1.0 / 3.0)
    cutoff = float(cutoff_sigma) * sigma
    sum_head = 0.0
    sum_wing = 0.0
    for q_chunk in _iter_reciprocal_q_chunks(cell, mesh, cutoff, chunk_size=chunk_size):
        q2 = np.einsum("gi,gi->g", q_chunk, q_chunk)
        gaussian = np.exp(-q2 / (2.0 * sigma**2))
        sum_head += np.sum(gaussian / q2)
        sum_wing += np.sum(gaussian / np.sqrt(q2))

    head_integral = np.sqrt(2.0 / np.pi) * sigma
    head_quadrature = -4.0 * np.pi / (volume * nk) * sum_head
    wing_integral = np.sqrt(4.0 * np.pi * volume) * sigma**2 / (2.0 * np.pi**2)
    wing_quadrature = -np.sqrt(4.0 * np.pi * volume) / (volume * nk) * sum_wing
    return GaussianCoefficients(
        sigma=sigma,
        head_integral=head_integral,
        head_quadrature=head_quadrature,
        head=head_integral + head_quadrature,
        wing_integral=wing_integral,
        wing_quadrature=wing_quadrature,
        wing=wing_integral + wing_quadrature,
    )


def inverse_dielectric_head_wing(eps_00, eps_p0, body_inverse):
    """Invert a dielectric matrix through its full body Schur complement.

    ``body_inverse`` is the inverse of the body block, and ``eps_p0`` is the
    column wing.  The returned pair is ``(eps_inv_00, eps_inv_p0)``.
    """
    eps_p0 = np.asarray(eps_p0, dtype=np.complex128)
    body_inverse = np.asarray(body_inverse, dtype=np.complex128)
    schur = eps_00 - np.vdot(eps_p0, np.matmul(body_inverse, eps_p0))
    eps_inv_00 = 1.0 / schur
    eps_inv_p0 = -eps_inv_00 * np.matmul(body_inverse, eps_p0)
    return eps_inv_00, eps_inv_p0


def _validate_support(gw):
    """Validate the supported corrected periodic G0W0 problem class."""
    cell = gw.mol
    if getattr(cell, "dimension", 3) != 3:
        raise NotImplementedError("Gaussian G0W0SS currently supports three-dimensional cells only")
    if not isinstance(gw.with_df, df.GDF) or isinstance(gw.with_df, df.MDF):
        raise NotImplementedError("G0W0SS requires a pure GDF object")
    if gw.fc and gw.outcore:
        raise NotImplementedError("Gaussian finite-size correction is implemented for in-core GW only")
    if gw.fc and hasattr(gw._scf, "sigma"):
        raise NotImplementedError("Gaussian finite-size correction does not support metallic or smeared occupations")
    if getattr(gw._scf, "spin", 0) != 0:
        raise NotImplementedError("G0W0SS currently supports spin-restricted calculations only")

    mo_occ = np.asarray(gw._scf.mo_occ)
    if mo_occ.ndim != 2 or not np.all(np.isclose(mo_occ, 0.0) | np.isclose(mo_occ, 2.0)):
        raise NotImplementedError("Gaussian finite-size correction requires integer restricted occupations")
    occupied = np.count_nonzero(mo_occ > 1.0, axis=1)
    if not np.all(occupied == occupied[0]):
        raise NotImplementedError("Gaussian finite-size correction requires the same occupation count at every k-point")
    mo_energy = np.asarray(gw._scf.mo_energy)
    if occupied[0] == 0 or occupied[0] == mo_energy.shape[1]:
        raise NotImplementedError("Gaussian finite-size correction requires an insulating occupied/virtual partition")
    homo = np.max(mo_energy[:, occupied[0] - 1])
    lumo = np.min(mo_energy[:, occupied[0]])
    if lumo <= homo + 1e-10:
        raise NotImplementedError("Gaussian finite-size correction requires a positive indirect gap")
    monkhorst_pack_shape(cell, gw.kpts)


def _set_frozen_orbs(gw):
    """Map full-space orbital selections to the active frozen space."""
    masks = get_frozen_mask(gw)
    first_mask = np.asarray(masks[0], dtype=bool)
    if any(not np.array_equal(first_mask, np.asarray(mask, dtype=bool)) for mask in masks[1:]):
        raise NotImplementedError("k-point-dependent frozen orbital masks are not supported")
    active_full = np.flatnonzero(first_mask)
    full_to_active = {int(full): active for active, full in enumerate(active_full)}

    if gw.orbs is None:
        full_orbs = [int(index) for index in active_full]
    else:
        full_orbs = [int(index) for index in gw.orbs]
    if len(set(full_orbs)) != len(full_orbs):
        raise ValueError("orbs must not contain duplicate orbital indices")
    try:
        active_orbs = [full_to_active[index] for index in full_orbs]
    except KeyError as error:
        raise ValueError("orbs contains a frozen or out-of-range orbital") from error
    gw.orbs = full_orbs
    gw.orbs_frz = active_orbs


def _finite_q_points(fc_grid):
    """Return small fractional reciprocal vectors for dielectric limits."""
    if not fc_grid:
        return np.asarray([[1.0e-3, 0.0, 0.0]], dtype=float)
    q_points = []
    for i, j, k in itertools.product(range(3), repeat=3):
        if i == 0 and j == 0 and k == 0:
            continue
        q_points.append([k * 5.0e-4, j * 5.0e-4, i * 5.0e-4])
    return np.asarray(q_points, dtype=float)


def _fit_two_pole_per_element(data, omega, orbs, nocc):
    """Fit TwoPoleAC element-by-element with the correct orbital axis."""
    data = np.asarray(data)
    element_shape = data.shape[:-1]
    coeff = np.zeros((10, *element_shape), dtype=float)
    if data.ndim < 2:
        raise ValueError("unexpected self-energy rank for two-pole continuation")
    for index in np.ndindex(element_shape):
        orbital = int(orbs[index[1]])
        fit = TwoPoleAC([orbital], nocc)
        # TwoPoleAC expects an orbital axis even when fitting one scalar
        # self-energy element.  The extra axis also makes idx[0] meaningful.
        fit.ac_fit(np.asarray(data[index])[None, :], omega)
        coeff[(slice(None), *index)] = fit.coeff[:, 0]
    acobj = TwoPoleAC(orbs, nocc)
    acobj.coeff = coeff
    acobj.omega = np.asarray(omega).copy()
    acobj.shape = element_shape
    return acobj


def _analytic_continue(gw, sigma_i, omega, orbs_frz):
    """Construct an AC object while preserving k-point/orbital axes."""
    if gw.ac == "twopole":
        return _fit_two_pole_per_element(sigma_i, omega, orbs_frz, gw.nocc)
    if gw.ac != "pade":
        raise ValueError("Unknown GW-AC type %s" % str(gw.ac))
    acobj = PadeAC(npts=gw.ac_pade_npts, step_ratio=gw.ac_pade_step_ratio)
    acobj.ac_fit(sigma_i, omega, axis=-1)
    return acobj


def get_sigma(
    gw,
    freqs,
    wts,
    ef,
    mo_energy,
    orbs=None,
    kptlist=None,
    mo_coeff=None,
    mo_occ=None,
    iw_cutoff=None,
    fullsigma=False,
):
    """Get in-core GW self-energy with Gaussian head and wing corrections.

    The contraction and frequency weights follow PySCF's ``krgw_ac``
    implementation.  Gaussian corrections are added on the imaginary axis,
    before analytic continuation, so the Green function and sign convention
    are unchanged.
    """
    nocc = gw.nocc
    nmo = gw.nmo
    nkpts = gw.nkpts
    kpts = gw.kpts
    if orbs is None:
        orbs = list(range(nmo))
    else:
        orbs = list(orbs)
    if kptlist is None:
        kptlist = list(range(nkpts))
    else:
        kptlist = list(kptlist)
    norbs = len(orbs)
    nklist = len(kptlist)
    nw = len(freqs)

    if mo_coeff is None:
        mo_coeff = _pyscf_krgw_ac._mo_frozen(gw, gw.mo_coeff)
    if mo_occ is None:
        mo_occ = _pyscf_krgw_ac._mo_occ_frozen(gw, gw.mo_occ)
    nao = mo_coeff[0].shape[0]

    # Keep PySCF's frequency-grid convention, including the zero-frequency
    # point used by the analytic-continuation fit.
    kscaled = gw.mol.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]
    if iw_cutoff is not None and gw.rdm is False:
        nw_sigma = sum(iw < iw_cutoff for iw in freqs) + 1
    else:
        nw_sigma = nw + 1
    omega = np.zeros(shape=[nw_sigma], dtype=np.complex128)
    omega[1:] = 1j * np.asarray(freqs)[: nw_sigma - 1] + ef
    emo = omega[None, None, :] - mo_energy[:, :, None]

    if fullsigma:
        sigma = np.zeros(shape=[nklist, norbs, norbs, nw_sigma], dtype=np.complex128)
    else:
        sigma = np.zeros(shape=[nklist, norbs, nw_sigma], dtype=np.complex128)

    gaussian = getattr(gw, "gaussian_coefficients", None)
    qij = None
    q_abs = None
    if gw.fc:
        if gaussian is None:
            gaussian = compute_gaussian_coefficients(
                gw.mol,
                nkpts=nkpts,
                mesh=monkhorst_pack_shape(gw.mol, gw.kpts),
            )
            gw.gaussian_coefficients = gaussian
        q_points = _finite_q_points(gw.fc_grid)
        q_abs = gw.mol.get_abs_kpts(q_points)
        qij = np.asarray(
            [_pyscf_krgw_ac.get_qij(gw, q, mo_energy, mo_coeff) for q in q_abs]
        )
        gw.fc_eps_inv_00 = np.zeros(nw, dtype=np.complex128)
        gw.fc_eps_inv_p0 = None
        gw.fc_sigma_head = np.zeros((nklist, norbs, nw_sigma), dtype=np.complex128)
        gw.fc_sigma_wing = np.zeros((nklist, norbs, nw_sigma), dtype=np.complex128)
    else:
        gw.fc_eps_inv_00 = None
        gw.fc_eps_inv_p0 = None
        gw.fc_sigma_head = None
        gw.fc_sigma_wing = None

    cderiarr = gw.with_df.cderi_array()
    for kL in range(nkpts):
        # Lij: (ki, L, i, j), with kj fixed by momentum conservation.
        Lij = []
        kidx = np.zeros(shape=[nkpts], dtype=np.int64)
        kidx_r = np.zeros(shape=[nkpts], dtype=np.int64)
        for i, kpti in enumerate(kpts):
            for j, kptj in enumerate(kpts):
                kconserv = -kscaled[i] + kscaled[j] + kscaled[kL]
                if np.linalg.norm(np.round(kconserv) - kconserv) < 1e-12:
                    kidx[i] = j
                    kidx_r[j] = i
                    Lpq = cderiarr.load(kpti, kptj)
                    if Lpq.shape[-1] == (nao * (nao + 1)) // 2:
                        Lpq = lib.unpack_tril(Lpq).reshape(-1, nao**2)
                    else:
                        Lpq = Lpq.reshape(-1, nao**2)
                    Lpq = Lpq.astype(np.complex128)
                    moij, ijslice = _conc_mos(mo_coeff[i], mo_coeff[j])[2:]
                    Lij_out = _ao2mo.r_e2(Lpq, moij, ijslice, tao=[], ao_loc=None, out=None)
                    Lij.append(Lij_out.reshape(-1, nmo, nmo))
        Lij = np.ascontiguousarray(Lij)
        naux = Lij.shape[1]
        Lia = None if hasattr(gw._scf, "sigma") else np.ascontiguousarray(Lij[:, :, :nocc, nocc:])

        naux_ones = np.ones(shape=[1, naux], dtype=np.complex128)
        mnQ = np.zeros(shape=[nmo * norbs, naux], dtype=np.complex128)
        if fullsigma:
            Wmn = np.zeros(shape=[nmo, norbs, norbs], dtype=np.complex128)
            Lij_kmQn = np.ascontiguousarray(Lij.transpose(0, 2, 1, 3))
        else:
            Qmn = np.zeros(shape=[naux, nmo * norbs], dtype=np.complex128)
            Wmn = np.zeros(shape=[nmo, norbs], dtype=np.complex128)

        for w in range(nw):
            if hasattr(gw._scf, "sigma"):
                Pi = _pyscf_krgw_ac.get_rho_response_metal(freqs[w], mo_energy, mo_occ, Lij, kidx)
            else:
                Pi = _pyscf_krgw_ac.get_rho_response(freqs[w], mo_energy, Lia, kidx)
            dielectric_inverse = np.linalg.inv(np.eye(naux, dtype=np.complex128) - Pi)

            if gw.fc and kL == 0:
                inverse_heads = []
                inverse_wings = []
                for iq in range(len(q_abs)):
                    Pi_00 = _pyscf_krgw_ac.get_rho_response_head(freqs[w], mo_energy, qij[iq])
                    qnorm = np.linalg.norm(q_abs[iq])
                    eps_00 = 1.0 - 4.0 * np.pi / qnorm**2 * Pi_00
                    Pi_p0 = _pyscf_krgw_ac.get_rho_response_wing(freqs[w], mo_energy, Lia, qij[iq])
                    eps_p0 = -np.sqrt(4.0 * np.pi) / qnorm * Pi_p0
                    inv_head, inv_wing = inverse_dielectric_head_wing(
                        eps_00, eps_p0, dielectric_inverse
                    )
                    inverse_heads.append(inv_head)
                    inverse_wings.append(inv_wing)
                # Each small-q direction is inverted independently before
                # averaging.  This avoids the cumulative-head bug in the
                # upstream implementation and applies to both blocks.
                eps_inv_00 = np.mean(inverse_heads)
                eps_inv_p0 = np.mean(np.asarray(inverse_wings), axis=0)
                gw.fc_eps_inv_00[w] = eps_inv_00
                if gw.fc_eps_inv_p0 is None:
                    gw.fc_eps_inv_p0 = np.zeros((nw, naux), dtype=np.complex128)
                gw.fc_eps_inv_p0[w] = eps_inv_p0

            dielectric_inverse -= np.eye(naux, dtype=np.complex128)
            g0 = wts[w] * emo / (emo**2 + freqs[w] ** 2)
            for k, kn in enumerate(kptlist):
                km = kidx_r[kn]
                # Index directly with the selected list so arbitrary ordering
                # survives; mkslice() is only valid for monotonic selections.
                l_slice = np.ascontiguousarray(Lij[km][:, :, orbs].reshape(naux, -1))
                if fullsigma:
                    l_slice_mQn = np.ascontiguousarray(Lij_kmQn[km][:, :, orbs])

                scipy.linalg.blas.zgemm(
                    alpha=1.0,
                    a=dielectric_inverse.T,
                    b=l_slice.T,
                    c=mnQ.T,
                    overwrite_c=1,
                    trans_b=2,
                )

                if not fullsigma:
                    Qmn[:] = mnQ.T * l_slice
                    np.matmul(naux_ones, Qmn, out=Wmn.reshape(1, nmo * norbs))
                    array_scale(Wmn, 1.0 / nkpts / np.pi)
                    sigma[k] -= np.matmul(Wmn.reshape(nmo, norbs).T, g0[km])
                else:
                    np.matmul(mnQ.reshape(nmo, norbs, naux), l_slice_mQn, out=Wmn)
                    array_scale(Wmn, 1.0 / nkpts / np.pi)
                    sigma[k] -= np.matmul(
                        Wmn.reshape(nmo, norbs * norbs).T, g0[km]
                    ).reshape(norbs, norbs, nw_sigma)

                if gw.fc and kL == 0:
                    assert kn == km
                    delta_head = gaussian.head * (gw.fc_eps_inv_00[w] - 1.0)
                    head_sigma = -delta_head * g0[kn][orbs] / np.pi
                    if not fullsigma:
                        sigma[k] += head_sigma
                    else:
                        sigma[k, np.arange(norbs), np.arange(norbs)] += head_sigma
                    gw.fc_sigma_head[k, :, :] += head_sigma

                    wing_contract = 2.0 * np.real(
                        einsum("Pnn,P->n", Lij[kn], gw.fc_eps_inv_p0[w])
                    )
                    delta_wing = gaussian.wing * wing_contract
                    wing_sigma = -delta_wing[orbs, None] * g0[kn][orbs] / np.pi
                    if not fullsigma:
                        sigma[k] += wing_sigma
                    else:
                        sigma[k, np.arange(norbs), np.arange(norbs)] += wing_sigma
                    gw.fc_sigma_wing[k, :, :] += wing_sigma

    if gw.rdm:
        gw.sigmaI = sigma
    return sigma, omega


def kernel(gw):
    """Run the local Gaussian-corrected G0W0 kernel."""
    mf = gw._scf
    nocc = gw.nocc
    nmo = gw.nmo
    nkpts = gw.nkpts

    _set_frozen_orbs(gw)
    orbs = list(gw.orbs)
    orbs_frz = list(gw.orbs_frz)
    kptlist = list(range(gw.nkpts)) if gw.kptlist is None else list(gw.kptlist)
    gw.kptlist = kptlist
    mo_energy_frz = _pyscf_krgw_ac._mo_energy_frozen(gw, gw.mo_energy)
    mo_coeff_frz = _pyscf_krgw_ac._mo_frozen(gw, gw.mo_coeff)

    with temporary_env(mf, verbose=0), temporary_env(mf.mol, verbose=0), temporary_env(mf.with_df, verbose=0):
        dm = mf.make_rdm1()
        v_mf_ao = mf.get_veff() - mf.get_j(dm_kpts=dm)
    v_mf = np.zeros(shape=[nkpts, nmo, nmo], dtype=np.complex128)
    for k in range(nkpts):
        v_mf[k] = reduce(np.matmul, (mo_coeff_frz[k].T.conj(), v_mf_ao[k], mo_coeff_frz[k]))
    gw.vxc = v_mf

    vk = gw.get_sigma_exchange()
    if gw.fc:
        gw.gaussian_coefficients = compute_gaussian_coefficients(
            gw.mol,
            nkpts=nkpts,
            mesh=monkhorst_pack_shape(gw.mol, gw.kpts),
        )
        vk_corr = -2.0 / np.pi * gw.gaussian_coefficients.sigma
        for k in range(nkpts):
            for i in range(nocc):
                vk[k][i, i] += vk_corr
    gw.vk = vk

    gw.ef = ef = _pyscf_krgw_ac.get_ef(kmf=mf, mo_energy=mf.mo_energy)
    gw.freqs, gw.wts = freqs, wts = _pyscf_krgw_ac._get_scaled_legendre_roots(gw.nw)
    if gw.outcore:
        # fc=False is the explicitly supported uncorrected out-of-core path.
        sigma_i, omega = _pyscf_krgw_ac.get_sigma_outcore(
            gw,
            freqs,
            wts,
            ef=ef,
            mo_energy=mo_energy_frz,
            orbs=orbs_frz,
            kptlist=kptlist,
            iw_cutoff=gw.ac_iw_cutoff,
            fullsigma=gw.fullsigma,
        )
    else:
        sigma_i, omega = get_sigma(
            gw,
            freqs,
            wts,
            ef=ef,
            mo_energy=mo_energy_frz,
            orbs=orbs_frz,
            kptlist=kptlist,
            iw_cutoff=gw.ac_iw_cutoff,
            fullsigma=gw.fullsigma,
        )

    acobj = _analytic_continue(gw, sigma_i, omega, orbs_frz)
    diag_acobj = acobj.diagonal(axis1=1, axis2=2) if gw.fullsigma else acobj
    mo_energy = np.zeros_like(mf.mo_energy)
    for ik, k in enumerate(kptlist):
        for ip, p in enumerate(orbs_frz):
            full_orbital = orbs[ip]
            if gw.qpe_linearized:
                de = 1e-6
                ep = mf.mo_energy[k][full_orbital]
                sigma_r = diag_acobj[ik, ip].ac_eval(ep).real
                dsigma = diag_acobj[ik, ip].ac_eval(ep + de).real - sigma_r
                zn = 1.0 / (1.0 - dsigma / de)
                if gw.qpe_linearized_range is not None:
                    if zn < gw.qpe_linearized_range[0] or zn > gw.qpe_linearized_range[1]:
                        zn = 1.0
                mo_energy[k, full_orbital] = ep + zn * (
                    sigma_r + vk[k, p, p] - v_mf[k, p, p]
                ).real
            else:
                def quasiparticle(energy):
                    sigma_r = diag_acobj[ik, ip].ac_eval(energy)
                    return energy - mf.mo_energy[k][full_orbital] - (
                        sigma_r + vk[k, p, p] - v_mf[k, p, p]
                    ).real

                try:
                    mo_energy[k, full_orbital] = scipy.optimize.newton(
                        quasiparticle,
                        mf.mo_energy[k][full_orbital],
                        tol=gw.qpe_tol,
                        maxiter=gw.qpe_max_iter,
                    )
                except RuntimeError:
                    logger.warn(gw, "QPE for k=%d orbital=%d not converged!", k, full_orbital)

    gw.mo_energy = mo_energy
    gw.acobj = acobj
    if gw.writefile > 0:
        import h5py

        with h5py.File("vxc.h5", "w") as feri:
            feri["vk"] = np.asarray(vk)
            feri["v_mf"] = np.asarray(v_mf)
        with h5py.File("sigma_imag.h5", "w") as feri:
            feri["sigmaI"] = np.asarray(sigma_i)
            feri["omega"] = np.asarray(omega)
            if gw.sigmaI is not None:
                feri["sigmaI_full"] = np.asarray(gw.sigmaI)
        acobj.save("ac_coeff.h5")
    logger.warn(gw, "GW QP energies may not be sorted from min to max")
    return


class G0W0SS(KRGWAC):
    """Periodic spin-restricted G0W0 with Gaussian finite-size correction.

    With ``fc=True``, the auxiliary function is ``exp(-Q**2/(2*sigma**2))``
    with ``sigma=(6*pi**2/(cell.vol*nkpts))**(1/3)`` in inverse Bohr.  The
    integral-minus-quadrature coefficients use all nonzero reciprocal
    supercell vectors within ``8*sigma``.  Exchange retains PySCF's Eq. 46.

    After a corrected calculation, ``gaussian_coefficients`` stores the
    width and separate signed integral/quadrature terms.  ``fc_eps_inv_00``
    and ``fc_eps_inv_p0`` have shapes ``(nw,)`` and ``(nw, naux)`` on
    ``freqs``.  ``fc_sigma_head`` and ``fc_sigma_wing`` contain diagonal
    additions of shape ``(len(kptlist), len(orbs), nw_sigma)``, even with
    ``fullsigma=True``.  Energies and self-energies are in hartree.
    """

    def kernel(self, orbs=None, kptlist=None):
        """Run G0W0, optionally selecting full-space orbitals and k-points."""
        if self.fc is False:
            # Keep PySCF's uncorrected behavior, including its out-of-core and
            # metallic paths, available as an exact drop-in mode.
            return super().kernel(orbs=orbs, kptlist=kptlist)
        _validate_support(self)
        if self.mo_energy is None:
            self.mo_energy = np.array(self._scf.mo_energy, copy=True)
        if self.mo_coeff is None:
            self.mo_coeff = np.array(self._scf.mo_coeff, copy=True)
        if self.mo_occ is None:
            self.mo_occ = np.array(self._scf.mo_occ, copy=True)
        self.orbs = orbs
        self.kptlist = kptlist
        nmo = self.nmo
        naux = self.with_df.get_naoaux()
        mem_incore = (2 * self.nkpts * nmo**2 * naux) * 16 / 1e6
        if mem_incore + lib.current_memory()[0] > 0.99 * self.max_memory:
            logger.warn(self, "Memory may not be enough!")
        cput0 = (time.process_time(), time.perf_counter())
        self.dump_flags()
        kernel(self)
        logger.timer(self, "G0W0SS", *cput0)
        return

    def set_frozen_orbs(self):
        """Use explicit active-space mapping for corrected calculations."""
        if self.fc:
            return _set_frozen_orbs(self)
        return KRGWAC.set_frozen_orbs(self)


__all__ = [
    "G0W0SS",
    "GaussianCoefficients",
    "compute_gaussian_coefficients",
    "inverse_dielectric_head_wing",
    "monkhorst_pack_shape",
    "reciprocal_supercell_lattice",
]
