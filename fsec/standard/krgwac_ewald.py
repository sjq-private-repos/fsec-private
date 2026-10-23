# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
#
# This file adapts the KRGWAC driver from PySCF 2.14.0. It is distributed
# under the Apache License, Version 2.0; see the license text below.
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

"""Restricted periodic G0W0-AC with Ewald finite-size corrections.

The Ewald driver uses PySCF's ordinary correlation self-energy and adds the
scalar long-wavelength head correction before analytic continuation. The
leading self-energy wing is odd on the supported symmetric transfer meshes,
so it has no separate finite-size term in this implementation.
"""

from copy import copy
from functools import reduce
import tempfile
import time

import h5py
import numpy as np
import scipy

from pyscf import lib
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.incore import _conc_mos
from pyscf.lib import logger, temporary_env
from pyscf.pbc import df, tools
from pyscf.pbc.gw import krgw_ac as _krgw_ac


def _uniform_mesh_shape(cell, kpts, tol=1e-8):
    """Return the shape of a complete uniform MP mesh, including any shift."""
    scaled = np.mod(cell.get_scaled_kpts(np.asarray(kpts).reshape(-1, 3)), 1.0)
    scaled[np.isclose(scaled, 1.0, atol=tol, rtol=0.0)] = 0.0
    scaled = np.round(scaled, 10)
    scaled[np.isclose(scaled, 1.0, atol=tol, rtol=0.0)] = 0.0

    axes = [np.unique(scaled[:, axis]) for axis in range(3)]
    shape = np.asarray([len(axis) for axis in axes], dtype=int)
    if np.prod(shape) != len(scaled):
        raise ValueError('Ewald correction requires a complete uniform MP mesh')

    for axis in axes:
        spacings = np.diff(np.r_[axis, axis[0] + 1.0])
        if not np.allclose(spacings, 1.0 / len(axis), atol=tol, rtol=0.0):
            raise ValueError('Ewald correction requires a complete uniform MP mesh')

    points = {tuple(point) for point in scaled}
    if len(points) != len(scaled):
        raise ValueError('Ewald correction does not support duplicate k-points')
    if len(points) != int(np.prod(shape)):
        raise ValueError('Ewald correction requires a complete uniform MP mesh')
    return shape


def _madelung_constant(cell, kpts):
    """Compute the supercell Madelung constant in inverse Bohr."""
    return float(tools.madelung(cell, kpts=kpts, omega=0.0))


def _add_exchange_head(vk, chi, nocc):
    """Apply the scalar exchange head to active occupied diagonal elements."""
    occupied = np.arange(nocc)
    vk[:, occupied, occupied] -= chi
    return vk


def _validate_ewald_setup(gw):
    """Reject Ewald cases outside the supported 3D restricted insulator scope."""
    if gw.outcore:
        raise NotImplementedError('Ewald finite-size corrections require the in-core GW driver')
    if gw.mol.dimension != 3:
        raise NotImplementedError('Ewald finite-size corrections require a 3D periodic cell')
    if not isinstance(gw.with_df, df.GDF) or isinstance(gw.with_df, df.MDF):
        raise NotImplementedError('Ewald finite-size corrections require GDF or RSDF integrals')
    if gw.mol.omega != 0:
        raise NotImplementedError('Ewald finite-size corrections require the full Coulomb interaction')
    if hasattr(gw._scf, 'sigma'):
        raise NotImplementedError('Ewald finite-size corrections do not support smeared occupations')

    kpts = np.asarray(gw.kpts)
    if kpts.ndim != 2 or kpts.shape[1] != 3:
        raise NotImplementedError('Ewald correction requires an explicit full k-point mesh')
    _uniform_mesh_shape(gw.mol, gw.kpts)
    mo_occ = np.asarray(gw.mo_occ)
    mo_energy = np.asarray(gw.mo_energy)
    mo_coeff = np.asarray(gw.mo_coeff)
    if (mo_occ.ndim != 2 or mo_energy.shape != mo_occ.shape
            or mo_coeff.ndim != 3 or mo_coeff.shape[1] != gw.mol.nao_nr()):
        raise NotImplementedError('Ewald finite-size corrections require restricted scalar orbitals')
    if not np.all(
        np.isclose(mo_occ, 0.0, atol=1e-9, rtol=0.0)
        | np.isclose(mo_occ, 2.0, atol=1e-9, rtol=0.0)
    ):
        raise NotImplementedError(
            'Ewald finite-size corrections require restricted integer occupations'
        )
    nocc_by_k = np.count_nonzero(mo_occ > 1.0, axis=1)
    if len(set(nocc_by_k.tolist())) != 1:
        raise NotImplementedError(
            'Ewald finite-size corrections require the same occupations at each k-point'
        )

    nocc = int(nocc_by_k[0])
    if nocc == 0 or nocc == mo_energy.shape[1]:
        raise NotImplementedError(
            'Ewald finite-size corrections require occupied and virtual orbitals'
        )
    gap = np.min(mo_energy[:, nocc:]) - np.max(mo_energy[:, :nocc])
    if not np.isfinite(gap) or gap <= 1e-8:
        raise NotImplementedError('Ewald finite-size corrections require a gapped system')
    if not np.all(mo_occ[:, :nocc] > 1.0) or np.any(mo_occ[:, nocc:] > 1.0):
        raise NotImplementedError('Ewald correction requires occupied orbitals before virtual orbitals')
    active_occ = _krgw_ac._mo_occ_frozen(gw, gw.mo_occ)
    counts = np.count_nonzero(active_occ > 1.0, axis=1)
    if (np.any(counts != counts[0]) or counts[0] == 0
            or counts[0] == active_occ.shape[1]):
        raise NotImplementedError('Ewald correction requires a common occupied/virtual active space')


def _long_wavelength_qpoints(gw):
    """Return the same dielectric-head directions used by PySCF's fc_grid."""
    if not gw.fc_grid:
        return np.array([[1e-3, 0.0, 0.0]], dtype=float)

    qpts = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i == 0 and j == 0 and k == 0:
                    continue
                qpts.append([k * 5e-4, j * 5e-4, i * 5e-4])
    return np.asarray(qpts, dtype=float)


def _load_zero_transfer_lia(gw, mo_coeff):
    """Load and transform the q=0 occupied-virtual DF blocks once."""
    nkpts = gw.nkpts
    naux = gw.with_df.get_naoaux()
    nocc = gw.nocc
    nmo = gw.nmo
    nao = gw.mol.nao_nr()
    cderi = gw.with_df.cderi_array()
    lia = np.empty((nkpts, naux, nocc, nmo - nocc), dtype=np.complex128)

    for ik, kpt in enumerate(gw.kpts):
        lpq = cderi.load(kpt, kpt)
        if lpq.shape[-1] == nao * (nao + 1) // 2:
            lpq = lib.unpack_tril(lpq).reshape(-1, nao**2)
        else:
            lpq = lpq.reshape(-1, nao**2)
        lpq = lpq.astype(np.complex128, copy=False)
        moij, ijslice = _conc_mos(
            mo_coeff[ik, :, :nocc], mo_coeff[ik, :, nocc:]
        )[2:]
        lij = _ao2mo.r_e2(lpq, moij, ijslice, tao=[], ao_loc=None, out=None)
        lia[ik] = lij.reshape(-1, nocc, nmo - nocc)
    return lia


def _inverse_dielectric_head(gw, freqs, mo_energy, mo_coeff, lia):
    """Evaluate the direction-averaged inverse dielectric head by Schur complement."""
    qpts = _long_wavelength_qpoints(gw)
    q_abs = gw.mol.get_abs_kpts(qpts)
    if gw.fc_grid:
        # Upstream's k.p transition density is exactly linear in q. Three
        # evaluations therefore reproduce its 26 directions without repeating
        # the real-space AO-gradient integration for every direction.
        basis_q = gw.mol.get_abs_kpts(np.eye(3) * 5e-4)
        basis_qij = np.asarray([
            _krgw_ac.get_qij(gw, q, mo_energy, mo_coeff) for q in basis_q
        ])
        qij = np.einsum('dx,xkia->dkia', qpts / 5e-4, basis_qij)
    else:
        qij = np.asarray([_krgw_ac.get_qij(gw, q_abs[0], mo_energy, mo_coeff)])
    kidx = np.arange(gw.nkpts, dtype=int)
    eye = np.eye(lia.shape[1], dtype=np.complex128)
    head = np.empty(len(freqs), dtype=np.complex128)

    for iw, nu in enumerate(freqs):
        pi_body = _krgw_ac.get_rho_response(nu, mo_energy, lia, kidx)
        pi_body_inv = np.linalg.inv(eye - pi_body)
        h_nu = 0.0j
        for iq, q in enumerate(q_abs):
            qnorm = np.linalg.norm(q)
            pi_00 = _krgw_ac.get_rho_response_head(nu, mo_energy, qij[iq])
            eps_00 = 1.0 - 4.0 * np.pi / qnorm**2 * pi_00
            pi_p0 = _krgw_ac.get_rho_response_wing(nu, mo_energy, lia, qij[iq])
            eps_p0 = -np.sqrt(4.0 * np.pi) / qnorm * pi_p0
            h_nu += 1.0 / len(q_abs) * _schur_inverse_head(
                eps_00, eps_p0, pi_body_inv
            )
        head[iw] = h_nu
    return head


def _schur_inverse_head(eps_00, eps_p0, pi_body_inv):
    """Invert the dielectric head after coupling it to the body through wings."""
    schur = eps_00 - np.dot(eps_p0.conj(), pi_body_inv @ eps_p0)
    return 1.0 / schur


def _correlation_head_correction(chi, head, freqs, wts, omega, energy):
    """Return the scalar head correction for one external orbital and k-point."""
    delta = np.asarray(omega)[:, None] - energy
    kernel = delta / (delta**2 + np.asarray(freqs)[None, :] ** 2)
    return -chi / np.pi * np.sum(
        kernel * (np.asarray(wts) * (np.asarray(head) - 1.0))[None, :], axis=1
    )


def _add_correlation_head(sigma, fullsigma, chi, head, freqs, wts, omega,
                          mo_energy, orbs, kptlist):
    """Add the Ewald head to selected diagonal self-energy samples."""
    for ik, kpt in enumerate(kptlist):
        for ip, orb in enumerate(orbs):
            correction = _correlation_head_correction(
                chi, head, freqs, wts, omega, mo_energy[kpt, orb]
            )
            if fullsigma:
                sigma[ik, ip, ip] += correction
            else:
                sigma[ik, ip] += correction
    return sigma


class KRGWAC(_krgw_ac.KRGWAC):
    """PySCF KRGWAC with scalar Ewald or upstream spherical corrections.

    ``fc_method`` is ``'ewald'`` by default. Set it to ``'spherical'`` to use
    PySCF's native finite-size correction. Setting ``fc=False`` disables both
    exchange and correlation finite-size corrections.
    """

    def __init__(self, mf, frozen=None):
        super().__init__(mf, frozen=frozen)
        self.fc_method = 'ewald'
        self._keys = self._keys | {'fc_method'}

    def kernel(self, orbs=None, kptlist=None):
        """Run restricted periodic G0W0 with the selected finite-size method."""
        if self.fc_method not in ('ewald', 'spherical'):
            raise ValueError("fc_method must be 'ewald' or 'spherical'")
        if self.fc_method == 'spherical' or not self.fc:
            return super().kernel(orbs=orbs, kptlist=kptlist)

        if self.mo_energy is None:
            self.mo_energy = np.array(self._scf.mo_energy, copy=True)
        if self.mo_coeff is None:
            self.mo_coeff = np.array(self._scf.mo_coeff, copy=True)
        if self.mo_occ is None:
            self.mo_occ = np.array(self._scf.mo_occ, copy=True)
        self.orbs = orbs
        self.kptlist = kptlist
        _validate_ewald_setup(self)
        if self.with_df._cderi is None or self.with_df._j_only:
            # Pure DFT often builds only Coulomb (k,k) blocks. GW also needs
            # (k,k') blocks; build them in a private cache, preserving SCF DF.
            self.with_df = copy(self.with_df).reset()
            self.with_df._cderi_to_save = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
            self.with_df.build(j_only=False)
        nmo = self.nmo
        naux = self.with_df.get_naoaux()
        mem_incore = (2 * self.nkpts * nmo**2 * naux) * 16 / 1e6
        if mem_incore + lib.current_memory()[0] > 0.99 * self.max_memory:
            logger.warn(self, 'Memory may not be enough!')
        cput0 = (time.process_time(), time.perf_counter())
        self.dump_flags()
        _kernel_ewald(self)
        logger.timer(self, 'KRGWAC', *cput0)
        return


def _kernel_ewald(gw):
    """PySCF 2.14 KRGWAC driver with scalar Ewald corrections before AC."""
    mf = gw._scf
    nocc = gw.nocc
    nmo = gw.nmo
    nkpts = gw.nkpts

    gw.set_frozen_orbs()
    orbs = gw.orbs
    orbs_frz = gw.orbs_frz
    kptlist = gw.kptlist
    if kptlist is None:
        gw.kptlist = kptlist = range(nkpts)
    mo_energy_frz = _krgw_ac._mo_energy_frozen(gw, gw.mo_energy)
    mo_coeff_frz = _krgw_ac._mo_frozen(gw, gw.mo_coeff)

    with temporary_env(mf, verbose=0), \
            temporary_env(mf.mol, verbose=0), \
            temporary_env(mf.with_df, verbose=0):
        dm = mf.make_rdm1()
        v_mf_ao = mf.get_veff() - mf.get_j(dm_kpts=dm)
    v_mf = np.zeros((nkpts, nmo, nmo), dtype=np.complex128)
    for k in range(nkpts):
        v_mf[k] = reduce(
            np.matmul,
            (mo_coeff_frz[k].T.conj(), v_mf_ao[k], mo_coeff_frz[k]),
        )
    gw.vxc = v_mf

    chi = _madelung_constant(gw.mol, gw.kpts)
    vk = _add_exchange_head(gw.get_sigma_exchange(), chi, nocc)
    gw.vk = vk

    gw.ef = ef = _krgw_ac.get_ef(kmf=mf, mo_energy=mf.mo_energy)
    gw.freqs, gw.wts = freqs, wts = _krgw_ac._get_scaled_legendre_roots(gw.nw)

    lia = _load_zero_transfer_lia(gw, mo_coeff_frz)
    head = _inverse_dielectric_head(gw, freqs, mo_energy_frz, mo_coeff_frz, lia)

    # The ordinary self-energy uses PySCF's full DF, frequency, and MO
    # contractions. The shallow copy disables its spherical finite-size term
    # without changing the user's GW object or the SCF object.
    working_gw = copy(gw)
    working_gw.fc = False
    # PySCF 2.14's advanced indexing mishandles noncontiguous band lists and
    # ignores their order when all bands are requested. A contiguous range
    # keeps its contractions intact; select the requested order afterwards.
    sigma_orbs = range(min(orbs_frz), max(orbs_frz) + 1)
    sigma_i, omega = _krgw_ac.get_sigma(
        working_gw, freqs, wts, ef=ef, mo_energy=mo_energy_frz,
        orbs=sigma_orbs, kptlist=kptlist, iw_cutoff=gw.ac_iw_cutoff,
        fullsigma=gw.fullsigma,
    )
    selected = np.asarray(orbs_frz) - sigma_orbs.start
    sigma_i = np.take(sigma_i, selected, axis=1)
    if gw.fullsigma:
        sigma_i = np.take(sigma_i, selected, axis=2)
    _add_correlation_head(
        sigma_i, gw.fullsigma, chi, head, freqs, wts, omega,
        mo_energy_frz, orbs_frz, kptlist,
    )
    if gw.rdm:
        gw.sigmaI = sigma_i

    if gw.ac == 'twopole':
        acobj = _krgw_ac.TwoPoleAC(list(range(nmo)), nocc)
    elif gw.ac == 'pade':
        acobj = _krgw_ac.PadeAC(
            npts=gw.ac_pade_npts, step_ratio=gw.ac_pade_step_ratio
        )
    else:
        raise ValueError('Unknown GW-AC type %s' % (str(gw.ac)))
    acobj.ac_fit(sigma_i, omega, axis=-1)
    diag_acobj = acobj.diagonal(axis1=1, axis2=2) if gw.fullsigma else acobj

    qp_energy = np.zeros_like(mf.mo_energy)
    for ik, kpt in enumerate(kptlist):
        for ip, p in enumerate(orbs_frz):
            if gw.qpe_linearized:
                de = 1e-6
                ep = mf.mo_energy[kpt][orbs[ip]]
                sigma_r = diag_acobj[ik, ip].ac_eval(ep).real
                dsigma = diag_acobj[ik, ip].ac_eval(ep + de).real - sigma_r.real
                zn = 1.0 / (1.0 - dsigma / de)
                if gw.qpe_linearized_range is not None:
                    low, high = gw.qpe_linearized_range
                    zn = 1.0 if zn < low or zn > high else zn
                qp_energy[kpt, orbs[ip]] = ep + zn * (
                    sigma_r + vk[kpt, p, p] - v_mf[kpt, p, p]
                ).real
            else:
                def quasiparticle(energy):
                    sigma_r = diag_acobj[ik, ip].ac_eval(energy)
                    return energy - mf.mo_energy[kpt][orbs[ip]] - (
                        sigma_r + vk[kpt, p, p] - v_mf[kpt, p, p]
                    ).real

                try:
                    qp_energy[kpt, orbs[ip]] = scipy.optimize.newton(
                        quasiparticle, mf.mo_energy[kpt][orbs[ip]],
                        tol=gw.qpe_tol, maxiter=gw.qpe_max_iter,
                    )
                except RuntimeError:
                    logger.warn(gw, 'QPE for k=%d orbital=%d not converged!', kpt, orbs[ip])

    gw.mo_energy = qp_energy
    gw.acobj = acobj
    with np.printoptions(threshold=len(mf.mo_energy[0])):
        for k in range(nkpts):
            logger.debug(gw, '  GW mo_energy @ k%d =\n%s', k, qp_energy[k])
    logger.warn(gw, 'GW QP energies may not be sorted from min to max')

    if gw.writefile > 0:
        with h5py.File('vxc.h5', 'w') as feri:
            feri['vk'] = np.asarray(vk)
            feri['v_mf'] = np.asarray(v_mf)
        with h5py.File('sigma_imag.h5', 'w') as feri:
            feri['sigmaI'] = np.asarray(sigma_i)
            feri['omega'] = np.asarray(omega)
            if gw.sigmaI is not None:
                feri['sigmaI_full'] = np.asarray(gw.sigmaI)
        acobj.save('ac_coeff.h5')
    return
