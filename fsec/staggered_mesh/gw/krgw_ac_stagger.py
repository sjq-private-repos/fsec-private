# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
#
# Staggered-mesh formulation and implementation: Xin Xing.
# Based on periodic GW-AC work by Tianyu Zhu and the PySCF Developers.
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

"""Restricted staggered-mesh G0W0 with analytic continuation.

The target mesh carries the quasiparticle states and the shifted mesh supplies
the occupied/virtual propagation.  Density-fitting integrals are built for
the union of both meshes.  Only the diagonal, one-shot G0W0 equations are
implemented here.
"""

from functools import reduce

import numpy as np
import scipy.optimize

from pyscf import lib
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.incore import _conc_mos
from pyscf.gw.utils.ac_grid import _get_scaled_legendre_roots, PadeAC, TwoPoleAC
from pyscf.lib import logger
from pyscf.pbc import df, dft, gto, scf
from pyscf.pbc.scf import khf


einsum = lib.einsum


def minimum_image(cell, kpts):
    """Wrap Cartesian k-points into the first Brillouin zone."""
    scaled = cell.get_scaled_kpts(kpts)
    scaled -= np.floor(scaled)
    scaled[scaled >= 0.5] -= 1.0
    return cell.get_abs_kpts(scaled)


def _mesh_shape(nks):
    """Return a validated three-dimensional Monkhorst-Pack mesh shape."""
    values = np.asarray(nks)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError('nks_target must contain three positive integers')
    rounded = np.rint(values).astype(int)
    if np.any(rounded <= 0) or not np.allclose(values, rounded):
        raise ValueError('nks_target must contain three positive integers')
    return rounded


def _match_kpoints(cell, candidates, reference, label):
    """Map each reference k-point to one unique point in candidates."""
    candidates_scaled = cell.get_scaled_kpts(candidates)
    reference_scaled = cell.get_scaled_kpts(reference)
    indices = []
    for point in reference_scaled:
        delta = candidates_scaled - point
        delta -= np.rint(delta)
        matches = np.flatnonzero(np.linalg.norm(delta, axis=1) < 1e-9)
        if len(matches) != 1:
            raise ValueError(
                f'{label} k-point is not present exactly once in the mean-field mesh'
            )
        indices.append(matches[0])
    if len(set(indices)) != len(indices):
        raise ValueError(f'{label} mesh contains duplicate k-points')
    return np.asarray(indices, dtype=int)


def _madelung_stagger(cell, nks, mesh_shift=None):
    """Madelung singularity subtraction for the half-shifted mesh.

    This evaluates the reciprocal-grid quadrature explicitly, avoiding the
    private ``_cut_mesh_for_ewald`` helper removed from current PySCF.
    """
    nks = _mesh_shape(nks)
    mesh_qg = np.asarray(cell.mesh, dtype=int) * nks
    mesh_max = np.asarray(
        2.0 * np.linalg.norm(cell.lattice_vectors() * nks[:, None], axis=1),
        dtype=int,
    )
    mesh_max = np.maximum(mesh_max, 80)
    mesh_qg = np.minimum(mesh_qg, mesh_max)

    bvec = cell.reciprocal_vectors() / nks[:, None]
    gmax = np.min(mesh_qg // 2 * np.linalg.norm(bvec, axis=1))
    log_precision = np.log(cell.precision / (4.0 * np.pi * gmax**2))
    eta = np.sqrt(-gmax**2 / (4.0 * log_precision))

    gbase = [np.fft.fftfreq(n, 1.0 / n) for n in mesh_qg]
    gv = lib.cartesian_prod(gbase) @ bvec
    if mesh_shift is None:
        mesh_shift = cell.get_abs_kpts((0.5 / nks).reshape(1, 3))[0]
    gv += np.asarray(mesh_shift)
    g2 = np.einsum('gi,gi->g', gv, gv)
    g2[g2 < 1e-24] = 1e200
    integrand = 4.0 * np.pi / (cell.vol * g2) * np.exp(-g2 / (4.0 * eta**2))
    quadrature = np.sum(integrand) / np.prod(nks)
    integral = 2.0 * eta / np.sqrt(np.pi)
    return float(integral - quadrature)


def _load_lmn(gw, left_kpts, right_kpts, left_coeff, right_coeff, qpt):
    """Read and transform all momentum-conserving DF blocks for one q-point."""
    cell = gw.cell
    nao = cell.nao_nr()
    nleft = len(left_kpts)
    right_scaled = cell.get_scaled_kpts(right_kpts)
    left_scaled = cell.get_scaled_kpts(left_kpts)
    q_scaled = cell.get_scaled_kpts(np.asarray(qpt).reshape(1, 3))[0]
    lmn = []
    right_index = np.full(nleft, -1, dtype=int)

    for im, kptm in enumerate(left_kpts):
        matches = []
        for jn, kptn in enumerate(right_kpts):
            delta = -left_scaled[im] + right_scaled[jn] + q_scaled
            if np.linalg.norm(delta - np.rint(delta)) < 1e-9:
                matches.append(jn)
        if len(matches) != 1:
            raise ValueError('staggered meshes do not have unique momentum conservation')
        jn = matches[0]
        right_index[im] = jn

        blocks = []
        for l_pq_r, l_pq_i, sign in gw.with_df.sr_loop(
            [kptm, right_kpts[jn]], max_memory=max(1.0, 0.1 * gw.max_memory),
            compact=False,
        ):
            if sign != 1:
                raise NotImplementedError(
                    'staggered GW supports only 3D positive-metric GDF blocks'
                )
            blocks.append(l_pq_r + 1j * l_pq_i)
        if not blocks:
            raise RuntimeError('GDF returned no three-center integrals')

        lpq = np.vstack(blocks)
        if lpq.shape[1] == nao * (nao + 1) // 2:
            lpq = lib.unpack_tril(lpq).reshape(-1, nao**2)
        elif lpq.shape[1] != nao**2:
            raise ValueError('unexpected AO density-fitting block shape')
        moij, ijslice = _conc_mos(left_coeff[im], right_coeff[jn])[2:]
        lmn_out = _ao2mo.r_e2(lpq, moij, ijslice, tao=[], ao_loc=None, out=None)
        nmo_left = left_coeff.shape[2]
        nmo_right = right_coeff.shape[2]
        lmn.append(lmn_out.reshape(-1, nmo_left, nmo_right))

    return np.asarray(lmn), right_index


def _rho_response_stagger(gw, omega, lmn, right_index, metallic=False):
    """Independent-particle response for target-to-shifted transitions."""
    naux = lmn.shape[1]
    pi = np.zeros((naux, naux), dtype=np.complex128)
    nocc = gw.nocc
    target_energy = gw._mo_energy[gw.kpts_target_idx]
    shift_energy = gw._mo_energy[gw.kpts_shift_idx]
    nkpts = gw.nkpts_target

    if metallic:
        target_occ = gw.mo_occ[gw.kpts_target_idx] / 2.0
        shift_occ = gw.mo_occ[gw.kpts_shift_idx] / 2.0
        for im, jn in enumerate(right_index):
            eia = target_energy[im, :, None] - shift_energy[jn, None, :]
            fia = target_occ[im, :, None] - shift_occ[jn, None, :]
            eia = eia * fia / (omega**2 + eia**2)
            pia = lmn[im] * eia[None, :, :]
            pi += 2.0 / nkpts * einsum('Pia,Qia->PQ', pia, lmn[im].conj())
    else:
        for im, jn in enumerate(right_index):
            eia = target_energy[im, :nocc, None] - shift_energy[jn, None, nocc:]
            eia = eia / (omega**2 + eia**2)
            pia = lmn[im, :, :nocc, nocc:] * eia[None, :, :]
            pi += 4.0 / nkpts * einsum(
                'Pia,Qia->PQ', pia, lmn[im, :, :nocc, nocc:].conj()
            )
    return pi


def _rho_response_target(gw, omega, lmn, metallic=False):
    """Independent-particle response within the target mesh at q=0."""
    naux = lmn.shape[1]
    pi = np.zeros((naux, naux), dtype=np.complex128)
    energy = gw._mo_energy[gw.kpts_target_idx]
    nocc = gw.nocc
    nkpts = gw.nkpts_target

    if metallic:
        occ = gw.mo_occ[gw.kpts_target_idx] / 2.0
        for ik in range(nkpts):
            eia = energy[ik, :, None] - energy[ik, None, :]
            fia = occ[ik, :, None] - occ[ik, None, :]
            eia = eia * fia / (omega**2 + eia**2)
            pia = lmn[ik] * eia[None, :, :]
            pi += 2.0 / nkpts * einsum('Pia,Qia->PQ', pia, lmn[ik].conj())
    else:
        for ik in range(nkpts):
            eia = energy[ik, :nocc, None] - energy[ik, None, nocc:]
            eia = eia / (omega**2 + eia**2)
            pia = lmn[ik, :, :nocc, nocc:] * eia[None, :, :]
            pi += 4.0 / nkpts * einsum(
                'Pia,Qia->PQ', pia, lmn[ik, :, :nocc, nocc:].conj()
            )
    return pi


def _get_rmat(gw):
    """Compute target-mesh dipoles from AO gradients."""
    cell = gw.cell
    target_coeff = gw.mo_coeff[gw.kpts_target_idx]
    target_kpts = gw.kpts_target
    coords, weights = dft.gen_grid.get_becke_grids(cell, level=4)
    nocc = gw.nocc
    nmo = gw.nmo
    rmat = np.zeros((gw.nkpts_target, 3, nocc, nmo - nocc), dtype=np.complex128)

    for ik, kpt in enumerate(target_kpts):
        ao_p = dft.numint.eval_ao(cell, coords, kpt=kpt, deriv=1)
        ao, ao_grad = ao_p[0], ao_p[1:4]
        ao_ao_grad = einsum('g,mg,xgn->xmn', weights, ao.T.conj(), ao_grad)
        mo_grad = np.empty_like(rmat[ik])
        for xyz in range(3):
            mo_grad[xyz] = reduce(
                np.dot,
                (target_coeff[ik][:, :nocc].T.conj(), ao_ao_grad[xyz],
                 target_coeff[ik][:, nocc:]),
            )
        denom = 1.0 / (
            gw._mo_energy[gw.kpts_target_idx[ik], nocc:, None]
            - gw._mo_energy[gw.kpts_target_idx[ik], None, :nocc]
        )
        rmat[ik] = mo_grad * denom.T[None, :, :]
    return rmat


def _get_fab(gw, omega, rmat):
    """Long-wavelength head of the dielectric response."""
    energy = gw._mo_energy[gw.kpts_target_idx]
    nocc = gw.nocc
    fab = np.eye(3, dtype=np.complex128)
    for ik in range(gw.nkpts_target):
        eia = energy[ik, :nocc, None] - energy[ik, None, nocc:]
        eia /= omega**2 + eia**2
        fab -= 16.0 * np.pi / (gw.nkpts_target * gw.cell.vol) * einsum(
            'xia,yia,ia->xy', rmat[ik], rmat[ik].conj(), eia
        )
    return fab


def _get_uap(gw, omega, lmn, rmat):
    """Long-wavelength wings coupling the dipole and DF basis."""
    energy = gw._mo_energy[gw.kpts_target_idx]
    nocc = gw.nocc
    uap = np.zeros((3, lmn.shape[1]), dtype=np.complex128)
    for ik in range(gw.nkpts_target):
        eia = energy[ik, :nocc, None] - energy[ik, None, nocc:]
        eia /= omega**2 + eia**2
        uap += (
            4j * np.sqrt(4.0 * np.pi / gw.cell.vol) / gw.nkpts_target
            * einsum('Pia,xia,ia->xP', lmn[ik, :, :nocc, nocc:],
                     rmat[ik].conj(), eia)
        )
    return uap


def _correlation_finite_size_correction(gw, freqs, wts, omega_grid, orbs, kptlist,
                                        sigma, metallic):
    """Apply the separate long-wavelength correlation Madelung correction."""
    if not (gw.fc and gw.fc_corr_madelung) or metallic:
        return sigma
    if gw.cell.dimension != 3:
        raise NotImplementedError('Madelung GW corrections require a 3D cell')

    xi_ex = gw.cell.vol / (4.0 * np.pi) * _madelung_stagger(
        gw.cell, gw.Nk_target, gw.mesh_shift
    )
    lmn, _ = _load_lmn(
        gw, gw.kpts_target, gw.kpts_target,
        gw.mo_coeff[gw.kpts_target_idx], gw.mo_coeff[gw.kpts_target_idx],
        np.zeros(3),
    )
    rmat = _get_rmat(gw)
    target_energy = gw._mo_energy[gw.kpts_target_idx]
    nw_sigma = len(omega_grid)
    emo = omega_grid[None, None, :] - target_energy[:, :, None]

    for iw, iw_freq in enumerate(freqs):
        pi = _rho_response_target(gw, iw_freq, lmn, metallic=False)
        eps_body_inv = np.linalg.inv(np.eye(lmn.shape[1]) - pi)
        fab = _get_fab(gw, iw_freq, rmat)
        uap = _get_uap(gw, iw_freq, lmn, rmat)
        sap = einsum('PQ,xQ->xP', eps_body_inv, uap)
        lw = fab - einsum('xP,yP->xy', uap.conj(), sap)
        if np.linalg.norm(lw.imag) > 1e-6:
            raise ValueError('long-wavelength dielectric matrix is not real')
        evals, evecs = np.linalg.eigh((lw + lw.conj().T) * 0.5)
        if np.min(evals) <= 0:
            raise np.linalg.LinAlgError('long-wavelength dielectric matrix is not positive')
        sqrt_inv = (evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.conj().T).real

        tmp_cell = gto.Cell()
        tmp_cell.build(
            unit='Bohr', a=gw.cell.lattice_vectors() @ sqrt_inv,
            atom='H 0 0 0; H 0 0 1e-10', dimension=3,
            max_memory=gw.max_memory, verbose=0,
            pseudo='gth-pade', basis='gth-szv', precision=gw.cell.precision,
        )
        shift_scaled = gw.cell.get_scaled_kpts(gw.mesh_shift.reshape(1, 3))[0]
        tmp_mesh_shift = tmp_cell.get_abs_kpts(shift_scaled.reshape(1, 3))[0]
        xi_lw = tmp_cell.vol / (4.0 * np.pi) * _madelung_stagger(
            tmp_cell, gw.Nk_target, tmp_mesh_shift
        )
        xi = xi_lw - xi_ex
        g0 = wts[iw] * emo / (emo**2 + iw_freq**2)
        for out_k, target_k in enumerate(kptlist):
            sigma[out_k] -= g0[target_k, orbs] * (4.0 * xi / gw.cell.vol)
    return sigma


def _get_sigma_diag_stagger(gw, orbs, kptlist, nw, iw_cutoff):
    """Build diagonal correlation self-energy samples on absolute frequencies."""
    freqs, wts = _get_scaled_legendre_roots(nw)
    nocc = gw.nocc
    target_idx = gw.kpts_target_idx
    shift_idx = gw.kpts_shift_idx
    shift_energy = gw._mo_energy[shift_idx]
    metallic = np.any(np.abs(gw.mo_occ / 2.0 - np.rint(gw.mo_occ / 2.0)) > 1e-6)

    if metallic:
        if gw.ef is None:
            raise ValueError('fractional occupations require setting gw.ef')
        ef = gw.ef
        if nw < 200:
            logger.warn(gw, 'Increasing imaginary-frequency grid to 200 for a metal')
            nw = 200
            freqs, wts = _get_scaled_legendre_roots(nw)
    else:
        homo = np.max(gw._mo_energy[:, :nocc])
        lumo = np.min(gw._mo_energy[:, nocc:])
        ef = 0.5 * (homo + lumo)
        gw.ef = ef

    if iw_cutoff is None:
        nw_sigma = nw + 1
    else:
        if iw_cutoff <= 0:
            raise ValueError('ac_iw_cutoff must be positive or None')
        nw_sigma = np.count_nonzero(freqs < iw_cutoff) + 1
    omega = ef + 1j * np.concatenate(([0.0], freqs[:nw_sigma - 1]))
    emo = omega[None, None, :] - shift_energy[:, :, None]
    sigma = np.zeros((len(kptlist), len(orbs), nw_sigma), dtype=np.complex128)

    kpts_diff = minimum_image(gw.cell, gw.kpts_target - gw.kpts_shift[0])
    for ikl, qpt in enumerate(kpts_diff):
        lmn, right_index = _load_lmn(
            gw, gw.kpts_target, gw.kpts_shift,
            gw.mo_coeff[target_idx], gw.mo_coeff[shift_idx], qpt,
        )
        naux = lmn.shape[1]
        eye = np.eye(naux)
        for iw, iw_freq in enumerate(freqs):
            pi = _rho_response_stagger(gw, iw_freq, lmn, right_index, metallic)
            wc = np.linalg.inv(eye - pi) - eye
            g0 = wts[iw] * emo / (emo**2 + iw_freq**2)
            for out_k, target_k in enumerate(kptlist):
                shift_k = right_index[target_k]
                l_slice = lmn[target_k][:, orbs, :]
                qnm = einsum('Pnm,PQ->Qnm', l_slice.conj(), wc)
                wnm = einsum('Qnm,Qnm->nm', qnm, l_slice) / gw.nkpts_target
                sigma[out_k] -= einsum(
                    'nm,mw->nw', wnm, g0[shift_k]
                ) / np.pi

    sigma = _correlation_finite_size_correction(
        gw, freqs, wts, omega, orbs, kptlist, sigma, metallic
    )
    return sigma, omega, metallic


class KRGWAC_stagger(lib.StreamObject):
    """Shared state and kernel for restricted staggered-mesh G0W0."""

    def __init__(self, mf, nks_target, kpts_target, kpts_shift,
                 mo_energy, mo_coeff, mo_occ, frozen=0, kpts=None,
                 target_idx=None, shift_idx=None):
        if frozen not in (0, None):
            raise NotImplementedError('frozen orbitals are not supported')
        if mf.cell.dimension != 3:
            raise NotImplementedError('staggered GW currently supports 3D cells only')
        if mf.cell.spin != 0:
            raise NotImplementedError('staggered GW requires a spin-restricted closed shell')

        self.cell = mf.cell
        if not (
            isinstance(mf, dft.krks.KRKS) or isinstance(mf, scf.khf.KRHF)
        ):
            raise TypeError('mf must be a restricted periodic KRHF or KRKS object')
        self._scf = mf
        self.verbose = mf.verbose
        self.stdout = mf.stdout
        self.max_memory = mf.max_memory
        self.frozen = 0
        self.Nk_target = _mesh_shape(nks_target)
        self.kpts_target = np.asarray(kpts_target)
        self.kpts_shift = np.asarray(kpts_shift)
        if len(self.kpts_target) != len(self.kpts_shift):
            raise ValueError('target and shifted meshes must have the same size')
        target_scaled = self.cell.get_scaled_kpts(self.kpts_target)
        shift_scaled = self.cell.get_scaled_kpts(self.kpts_shift)
        actual_shift = shift_scaled[0] - target_scaled[0]
        actual_shift -= np.rint(actual_shift)
        expected_shift = np.where(self.Nk_target > 1, 0.5 / self.Nk_target, 0.0)
        if np.all(self.Nk_target == 1):
            expected_shift[:] = 0.5
        delta_shift = actual_shift - expected_shift
        delta_shift -= np.rint(delta_shift)
        if np.linalg.norm(delta_shift) > 1e-9:
            raise ValueError('shifted mesh must use the expected half-grid shift')
        self.mesh_shift = self.cell.get_abs_kpts(actual_shift.reshape(1, 3))[0]
        all_scaled = np.concatenate((target_scaled, shift_scaled))
        for i in range(len(all_scaled)):
            delta = all_scaled[i + 1:] - all_scaled[i]
            delta -= np.rint(delta)
            if np.any(np.linalg.norm(delta, axis=1) < 1e-9):
                raise ValueError('target and shifted meshes must contain unique points')

        self.nkpts_target = len(self.kpts_target)
        self.kpts = np.asarray(
            np.concatenate((self.kpts_target, self.kpts_shift)) if kpts is None else kpts
        )
        self.kpts_target_idx = (
            np.arange(self.nkpts_target) if target_idx is None
            else np.asarray(target_idx, dtype=int)
        )
        self.kpts_shift_idx = (
            np.arange(self.nkpts_target, 2 * self.nkpts_target) if shift_idx is None
            else np.asarray(shift_idx, dtype=int)
        )
        self._mo_energy = np.asarray(mo_energy)
        self.mo_coeff = np.asarray(mo_coeff)
        self.mo_occ = np.asarray(mo_occ)
        if (len(self.kpts_target_idx) != self.nkpts_target
                or len(self.kpts_shift_idx) != self.nkpts_target
                or len(self.kpts) != len(self._mo_energy)
                or np.max(self.kpts_target_idx) >= len(self.kpts)
                or np.max(self.kpts_shift_idx) >= len(self.kpts)):
            raise ValueError('orbital data must contain both complete staggered meshes')
        self.nkpts = len(self.kpts)
        self.mo_energy_ref = self._mo_energy[self.kpts_target_idx].copy()
        self.nmo = self._mo_energy.shape[1]
        self.nocc = self.cell.nelectron // 2
        if self.nocc <= 0 or self.nocc >= self.nmo:
            raise ValueError('G0W0 requires at least one occupied and one virtual orbital')

        self.ac = 'pade'
        self.ac_pade_npts = 18
        self.ac_pade_step_ratio = 2.0 / 3.0
        self.qpe_linearized = False
        self.qpe_linearized_range = [0.5, 1.5]
        self.qpe_tol = 1e-6
        self.qpe_max_iter = 100
        self.fc = True
        self.fc_ex_madelung = True
        self.fc_corr_madelung = True
        self.ef = None
        self.mo_energy = np.full_like(self.mo_energy_ref, np.nan, dtype=float)
        self.qpe_converged = np.zeros_like(self.mo_energy_ref, dtype=bool)
        self.converged = None
        self.sigma = None
        self.acobj = None
        self.v_mf = self._get_v_mf()

        # A fresh GDF is necessary because cross-mesh pairs need not occur in
        # the mean-field density-fitting mesh.
        self.with_df = df.GDF(self.cell, self.kpts).build(j_only=False)
        self._keys = set(self.__dict__.keys()) | {
            'ac', 'ac_pade_npts', 'ac_pade_step_ratio', 'qpe_linearized',
            'qpe_linearized_range', 'qpe_tol', 'qpe_max_iter', 'fc',
            'fc_ex_madelung', 'fc_corr_madelung', 'nw', 'ac_iw_cutoff',
        }
        self.nw = 100
        self.ac_iw_cutoff = 5.0

    def _get_v_mf(self):
        """Project the reference exchange-correlation potential on target MOs."""
        dm = np.asarray(self._scf.make_rdm1())
        if isinstance(self._scf, dft.krks.KRKS):
            veff = self._scf.get_veff(self.cell, dm=dm, kpts_band=self.kpts_target)
        else:
            veff = self._scf.get_veff(
                self.cell, dm_kpts=dm, kpts_band=self.kpts_target
            )
        vj = self._scf.get_j(self.cell, dm_kpts=dm, kpts_band=self.kpts_target)
        vxc_ao = np.asarray(veff) - np.asarray(vj)
        coeff = self.mo_coeff[self.kpts_target_idx]
        return np.asarray([
            reduce(np.dot, (c.T.conj(), v, c)) for c, v in zip(coeff, vxc_ao)
        ])

    def dump_flags(self):
        logger.info(self, 'staggered restricted G0W0: %s target k-points', self.nkpts_target)
        logger.info(self, 'analytic continuation = %s', self.ac)
        logger.info(self, 'linearized QPE = %s', self.qpe_linearized)
        logger.info(self, 'finite-size correction = %s', self.fc)
        logger.info(self, 'exchange Madelung correction = %s', self.fc_ex_madelung)
        logger.info(self, 'correlation Madelung correction = %s', self.fc_corr_madelung)
        return self

    def _get_sigma_exchange(self, kptlist):
        """Evaluate Fock exchange at selected target k-points from shift density."""
        shift_coeff = self.mo_coeff[self.kpts_shift_idx]
        shift_occ = self.mo_occ[self.kpts_shift_idx]
        dm_shift = khf.make_rdm1(shift_coeff, shift_occ)
        rhf = scf.khf.KRHF(self.cell, self.kpts_shift, exxdiv=None)
        target_kpts = self.kpts_target[kptlist]
        vk_ao = rhf.get_veff(
            self.cell, dm_kpts=dm_shift, kpts_band=target_kpts
        ) - rhf.get_j(self.cell, dm_kpts=dm_shift, kpts_band=target_kpts)
        coeff = self.mo_coeff[self.kpts_target_idx[kptlist]]
        vk = np.asarray([
            reduce(np.dot, (c.T.conj(), v, c)) for c, v in zip(coeff, vk_ao)
        ])
        if self.fc and self.fc_ex_madelung:
            correction = _madelung_stagger(
                self.cell, self.Nk_target, self.mesh_shift
            )
            for ik in range(len(kptlist)):
                for i in range(self.nocc):
                    vk[ik, i, i] -= correction
        return vk

    def kernel(self, orbs=None, kptlist=None):
        """Run G0W0 and store selected quasiparticle energies in ``mo_energy``.

        ``orbs`` contains orbital indices in the reference MO basis and
        ``kptlist`` contains indices in the target mesh.  Unrequested or
        unconverged results remain NaN.  The method returns ``None``.
        """
        if self.frozen != 0:
            raise NotImplementedError('frozen orbitals are not supported')
        if self.ac not in ('pade', 'twopole'):
            raise ValueError("ac must be 'pade' or 'twopole'")
        if orbs is None:
            orbs = list(range(self.nmo))
        else:
            orbs = list(orbs)
        if kptlist is None:
            kptlist = list(range(self.nkpts_target))
        else:
            kptlist = list(kptlist)
        if not orbs or not kptlist:
            raise ValueError('orbs and kptlist must select at least one state')
        if len(set(orbs)) != len(orbs) or any(p < 0 or p >= self.nmo for p in orbs):
            raise ValueError('orbs must contain unique valid orbital indices')
        if (len(set(kptlist)) != len(kptlist)
                or any(k < 0 or k >= self.nkpts_target for k in kptlist)):
            raise ValueError('kptlist must contain unique target-mesh indices')
        nw = int(self.nw)
        if nw <= 0:
            raise ValueError('nw must be a positive integer')

        self.dump_flags()
        self.mo_energy[:] = np.nan
        self.qpe_converged[:] = False
        self.converged = False
        sigma, omega, metallic = _get_sigma_diag_stagger(
            self, orbs, kptlist, nw, self.ac_iw_cutoff
        )
        if self.ac == 'pade':
            acobj = PadeAC(
                npts=self.ac_pade_npts,
                step_ratio=self.ac_pade_step_ratio,
            )
            acobj.ac_fit(sigma, omega, axis=-1)
        else:
            acobj = TwoPoleAC(orbs, self.nocc)
            acobj.ac_fit(sigma.transpose(1, 0, 2), omega, axis=-1)
        self.sigma = sigma
        self.acobj = acobj

        vk = self._get_sigma_exchange(kptlist)
        conv = True
        for ik, target_k in enumerate(kptlist):
            for ip, orb in enumerate(orbs):
                ep = self.mo_energy_ref[target_k, orb]
                diag_sigma = acobj[ik, ip] if self.ac == 'pade' else acobj[ip, ik]
                if self.qpe_linearized:
                    de = 1e-6
                    sigma_r = diag_sigma.ac_eval(ep).real
                    dsigma = diag_sigma.ac_eval(ep + de).real - sigma_r
                    z_factor = 1.0 / (1.0 - dsigma / de)
                    if self.qpe_linearized_range is not None:
                        zlo, zhi = self.qpe_linearized_range
                        if z_factor < zlo or z_factor > zhi:
                            z_factor = 1.0
                    energy = ep + z_factor * (
                        sigma_r + vk[ik, orb, orb] - self.v_mf[target_k, orb, orb]
                    ).real
                    if np.isfinite(energy):
                        self.mo_energy[target_k, orb] = energy
                        self.qpe_converged[target_k, orb] = True
                else:
                    def quasiparticle(energy):
                        sigma_r = diag_sigma.ac_eval(energy).real
                        return energy - ep - (
                            sigma_r + vk[ik, orb, orb]
                            - self.v_mf[target_k, orb, orb]
                        ).real

                    try:
                        energy = scipy.optimize.newton(
                            quasiparticle, ep, tol=self.qpe_tol,
                            maxiter=self.qpe_max_iter,
                        )
                    except (RuntimeError, ValueError, OverflowError):
                        logger.warn(
                            self, 'QPE for target k=%d orbital=%d did not converge',
                            target_k, orb,
                        )
                        conv = False
                        continue
                    self.mo_energy[target_k, orb] = energy
                    self.qpe_converged[target_k, orb] = np.isfinite(energy)
                conv = conv and self.qpe_converged[target_k, orb]

        self.converged = bool(conv)
        return None


def _make_meshes(cell, nks, anchor):
    """Create disjoint gamma and half-grid-shifted meshes."""
    nks = _mesh_shape(nks)
    target = anchor + cell.make_kpts(nks, with_gamma_point=True)
    shift_scaled = np.where(nks > 1, 0.5 / nks, 0.0)
    if np.all(nks == 1):
        shift_scaled[:] = 0.5
    half_shift = cell.get_abs_kpts(shift_scaled.reshape(1, 3))[0]
    shifted = target + half_shift
    return nks, target, shifted


class KRGWAC_stagger_submesh(KRGWAC_stagger):
    """G0W0 on a target/shifted pair sampled from a restricted SCF mesh."""

    def __init__(self, mf, nks_target, frozen=0):
        nks = _mesh_shape(nks_target)
        mf_kpts = np.asarray(mf.kpts)
        gamma_dist = np.linalg.norm(minimum_image(mf.cell, mf_kpts), axis=1)
        anchor = mf_kpts[np.argmin(gamma_dist)]
        _, target, shifted = _make_meshes(mf.cell, nks, anchor)
        target_mf_idx = _match_kpoints(mf.cell, mf_kpts, target, 'target')
        shift_mf_idx = _match_kpoints(mf.cell, mf_kpts, shifted, 'shifted')
        super().__init__(
            mf, nks, target, shifted,
            np.asarray(mf.mo_energy), np.asarray(mf.mo_coeff), np.asarray(mf.mo_occ),
            frozen=frozen, kpts=mf_kpts, target_idx=target_mf_idx,
            shift_idx=shift_mf_idx,
        )


class KRGWAC_stagger_extend(KRGWAC_stagger):
    """G0W0 using non-SCF target and half-shifted bands from a restricted SCF."""

    def __init__(self, mf, nks_target, frozen=0):
        nks = _mesh_shape(nks_target)
        _, target, shifted = _make_meshes(mf.cell, nks, np.zeros(3))
        kpts = np.concatenate((target, shifted))

        # KRKS inherits from KRHF, so test it first.  The HF interpolation
        # uses the spherical-vcut exchange treatment available in current PySCF.
        if isinstance(mf, dft.krks.KRKS):
            energy, coeff = mf.get_bands(kpts)
            occ = khf.get_occ(mf, mo_energy_kpts=energy)
        elif isinstance(mf, scf.khf.KRHF):
            with lib.temporary_env(
                mf, exxdiv='vcut_sph', with_df=df.FFTDF(mf.cell, mf.kpts)
            ):
                energy, coeff = mf.get_bands(kpts)
                occ = khf.get_occ(mf, mo_energy_kpts=energy)
        else:
            raise TypeError('mf must be a restricted periodic KRHF or KRKS object')
        super().__init__(
            mf, nks, target, shifted, energy, coeff, occ, frozen=frozen,
            kpts=kpts, target_idx=np.arange(len(target)),
            shift_idx=np.arange(len(target), len(kpts)),
        )


__all__ = ['KRGWAC_stagger_submesh', 'KRGWAC_stagger_extend']
