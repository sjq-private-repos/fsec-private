# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
#
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

'''
Periodic spin-restricted G0W0-AC with lattice-aware q -> 0 finite-size corrections.

This is pyscf.pbc.gw.krgw_ac.KRGWAC, except that the geometric constants
multiplying the integrable q -> 0 singularities of the exchange and correlation
self-energies can be chosen among

    'aux'      : Gaussian auxiliary functions, FHI-gap App. B (default)
    'madelung' : the infinite-width limit of 'aux' (supercell Madelung constant)
    'sphere'   : equal-volume sphere integration, as in PySCF

and the k.p dielectric head/wings can be averaged over q directions on a
Lebedev grid.  With fc_method='sphere' and fc_qdirs=None the results agree
with PySCF's KRGWAC.

References:
    T. Zhu and G.K.-L. Chan, J. Chem. Theory. Comput. 17, 727-741 (2021)
    H. Jiang et al., Comput. Phys. Commun. 184, 348-366 (2013), Appendix B
    S. Massidda, M. Posternak, A. Baldereschi, Phys. Rev. B 48, 5058 (1993)
'''

from copy import copy
from functools import reduce
import tempfile
import time

import h5py
import numpy as np
import scipy.optimize
from scipy.special import erfc

from pyscf import lib
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.incore import _conc_mos
from pyscf.dft.LebedevGrid import LEBEDEV_NGRID, MakeAngularGrid
from pyscf.lib import einsum, logger, temporary_env
from pyscf.pbc import df, tools
from pyscf.pbc.gw import krgw_ac
from pyscf.pbc.gw.krgw_ac import (
    _get_scaled_legendre_roots,
    _mo_energy_frozen,
    _mo_frozen,
    _mo_occ_frozen,
    PadeAC,
    TwoPoleAC,
    get_ef,
    get_qij,
    get_rho_response,
    get_rho_response_wing,
    get_sigma,
    get_sigma_outcore,
)

# exp(-x) cutoff for the Gaussian-damped lattice sums
_EXP_CUTOFF = 40.0


def _uniform_mesh_shape(cell, kpts, tol=1e-8):
    """Return the shape of a complete uniform MP mesh, including any shift."""
    scaled = np.mod(cell.get_scaled_kpts(np.asarray(kpts).reshape(-1, 3)), 1.0)
    scaled[np.isclose(scaled, 1.0, atol=tol, rtol=0.0)] = 0.0
    scaled = np.round(scaled, 10)
    scaled[np.isclose(scaled, 1.0, atol=tol, rtol=0.0)] = 0.0

    axes = [np.unique(scaled[:, axis]) for axis in range(3)]
    shape = np.asarray([len(axis) for axis in axes], dtype=int)
    if np.prod(shape) != len(scaled):
        raise ValueError('Finite-size correction requires a complete uniform MP mesh')

    for axis in axes:
        spacings = np.diff(np.r_[axis, axis[0] + 1.0])
        if not np.allclose(spacings, 1.0 / len(axis), atol=tol, rtol=0.0):
            raise ValueError('Finite-size correction requires a complete uniform MP mesh')

    if len({tuple(point) for point in scaled}) != len(scaled):
        raise ValueError('Finite-size correction does not support duplicate k-points')
    return shape


def _validate_fc_setup(gw):
    """Reject finite-size corrections outside the 3D restricted insulator scope."""
    if gw.mol.dimension != 3:
        raise NotImplementedError('Finite-size corrections require a 3D periodic cell')
    if not isinstance(gw.with_df, df.GDF) or isinstance(gw.with_df, df.MDF):
        raise NotImplementedError('Finite-size corrections require GDF or RSDF integrals')
    if gw.mol.omega != 0:
        raise NotImplementedError('Finite-size corrections require the full Coulomb interaction')
    _uniform_mesh_shape(gw.mol, gw.kpts)

    mo_occ = np.asarray(gw.mo_occ)
    mo_energy = np.asarray(gw.mo_energy)
    mo_coeff = np.asarray(gw.mo_coeff)
    if (mo_occ.ndim != 2 or mo_energy.shape != mo_occ.shape
            or mo_coeff.ndim != 3 or mo_coeff.shape[1] != gw.mol.nao_nr()):
        raise NotImplementedError('Finite-size corrections require restricted scalar orbitals')
    if not np.all(np.isclose(mo_occ, 0.0, atol=1e-9, rtol=0.0)
                  | np.isclose(mo_occ, 2.0, atol=1e-9, rtol=0.0)):
        raise NotImplementedError('Finite-size corrections require restricted integer occupations')
    nocc_by_k = np.count_nonzero(mo_occ > 1.0, axis=1)
    if len(set(nocc_by_k.tolist())) != 1:
        raise NotImplementedError('Finite-size corrections require the same occupations at each k-point')
    nocc = int(nocc_by_k[0])
    if nocc == 0 or nocc == mo_energy.shape[1]:
        raise NotImplementedError('Finite-size corrections require occupied and virtual orbitals')
    gap = np.min(mo_energy[:, nocc:]) - np.max(mo_energy[:, :nocc])
    if not np.isfinite(gap) or gap <= 1e-8:
        raise NotImplementedError('Finite-size corrections require a gapped system')
    if not np.all(mo_occ[:, :nocc] > 1.0) or np.any(mo_occ[:, nocc:] > 1.0):
        raise NotImplementedError('Finite-size corrections require occupied orbitals before virtual orbitals')
    counts = np.count_nonzero(_mo_occ_frozen(gw, gw.mo_occ) > 1.0, axis=1)
    if np.any(counts != counts[0]) or counts[0] == 0 or counts[0] == gw.nmo:
        raise NotImplementedError('Finite-size corrections require a common occupied/virtual active space')


def _lattice_sum(basis, rmax, func):
    """Sum func(|v|) over nonzero lattice vectors v = n @ basis with |v| < rmax.

    basis : (3, 3) ndarray, lattice vectors as rows.
    func : vectorized function of the vector norm.
    """
    # |n_i| <= |v| * ||column i of inv(basis)||
    nmax = np.ceil(rmax * np.linalg.norm(np.linalg.inv(basis), axis=0)).astype(int)
    ny = np.arange(-nmax[1], nmax[1] + 1)
    nz = np.arange(-nmax[2], nmax[2] + 1)
    nyz = lib.cartesian_prod([ny, nz]) @ basis[1:]
    total = 0.0
    for nx in range(-nmax[0], nmax[0] + 1):
        r = np.linalg.norm(nyz + nx * basis[0], axis=1)
        r = r[(r > 1e-12) & (r < rmax)]
        total += np.sum(func(r))
    return total


def _supercell_basis(cell, nks):
    """Supercell lattice vectors and Gamma-centred supercell reciprocal vectors."""
    nks = np.asarray(nks)
    avec = cell.lattice_vectors() * nks[:, None]
    bvec = cell.reciprocal_vectors() / nks[:, None]
    return avec, bvec


def aux_constants(cell, nks, beta):
    """Gaussian auxiliary-function constants (FHI-gap Eq. B.6).

    Returns (c_head, c_inv_q) with

        c_head  = 1/sqrt(pi beta) - 1/(Nk Omega) sum'_k 4 pi exp(-beta k^2) / k^2
        c_inv_q = 1/(4 pi^2 beta) - 1/(Nk Omega) sum'_k exp(-beta k^2) / k

    where k runs over the Gamma-centred supercell reciprocal lattice (k = q + G)
    and the q = 0 term keeps only G != 0, as in Eq. B.6.  In FHI-gap notation
    c_head = 4 pi Cs2 / Omega and c_inv_q = Cs1 / Omega.  Units: 1/bohr and
    1/bohr^2.
    """
    nkpts = np.prod(nks)
    vol = cell.vol * nkpts
    bvec = _supercell_basis(cell, nks)[1]
    kmax = np.sqrt(_EXP_CUTOFF / beta)
    c_head = 1.0 / np.sqrt(np.pi * beta) - _lattice_sum(
        bvec, kmax, lambda k: 4.0 * np.pi * np.exp(-beta * k**2) / k**2) / vol
    c_inv_q = 1.0 / (4.0 * np.pi**2 * beta) - _lattice_sum(
        bvec, kmax, lambda k: np.exp(-beta * k**2) / k) / vol
    return c_head, c_inv_q


def madelung_inv_q(cell, nks, tau=None):
    """Madelung-like constant for the 1/k singularity (infinite-width limit).

    Evaluates  C1 = int d^3k/(2 pi)^3 1/k - 1/(Nk Omega) sum'_k 1/k  by the
    Ewald split 1/k = erfc(sqrt(tau) k)/k + erf(sqrt(tau) k)/k:

        C1 = 1/(8 pi^2 tau) - 1/V sum'_k erfc(sqrt(tau) k)/k
             - sum'_R exp(-R^2/(4 tau)) / (2 pi^2 R^2) + 2 sqrt(tau/pi) / V

    with V = Nk Omega and R over the supercell lattice.  The result does not
    depend on tau.  Units: 1/bohr^2.
    """
    nkpts = np.prod(nks)
    vol = cell.vol * nkpts
    avec, bvec = _supercell_basis(cell, nks)
    if tau is None:
        # balances the number of real- and reciprocal-space terms
        tau = vol**(2.0 / 3.0) / (4.0 * np.pi)
    kmax = np.sqrt(_EXP_CUTOFF / tau)
    rmax = np.sqrt(4.0 * _EXP_CUTOFF * tau)
    recip = _lattice_sum(bvec, kmax, lambda k: erfc(np.sqrt(tau) * k) / k) / vol
    real = _lattice_sum(avec, rmax, lambda r: np.exp(-r**2 / (4.0 * tau)) / (2.0 * np.pi**2 * r**2))
    return 1.0 / (8.0 * np.pi**2 * tau) - recip - real + 2.0 * np.sqrt(tau / np.pi) / vol


def get_fc_beta(gw):
    """Gaussian exponent beta (bohr^2) of the auxiliary functions exp(-beta |q+G|^2).

    Default is the FHI-gap choice beta = (Omega/6 pi^2)^(1/3), taken literally
    in atomic units (the paper derives it from beta R_BZ = 1).  A custom width
    gw.fc_sigma (1/bohr) gives beta = 1/(2 sigma^2).
    """
    if gw.fc_sigma is None:
        return (gw.mol.vol / (6.0 * np.pi**2))**(1.0 / 3.0)
    return 0.5 / gw.fc_sigma**2


def get_fc_coeffs(gw):
    """Geometric constants (c_head, c_wing) of the q -> 0 corrections.

    c_head multiplies (eps^-1_00 - 1) in the correlation head correction and is
    subtracted from the occupied diagonal of the exchange self-energy.  c_wing
    multiplies the wing term Wn_P0.  For fc_method='sphere' these are PySCF's
    2 q0/pi and sqrt(Omega/4 pi^3) q0^2 with q0 = (6 pi^2/(Nk Omega))^(1/3).
    Otherwise c_wing = sqrt(4 pi Omega) * C1, C1 being the 1/k analog of c_head.
    """
    cell = gw.mol
    nkpts = gw.nkpts
    nks = _uniform_mesh_shape(cell, gw.kpts)
    if gw.fc_method == 'sphere':
        q0 = (6.0 * np.pi**2 / cell.vol / nkpts)**(1.0 / 3.0)
        return 2.0 / np.pi * q0, np.sqrt(cell.vol / 4.0 / np.pi**3) * q0**2
    elif gw.fc_method == 'aux':
        c_head, c_inv_q = aux_constants(cell, nks, get_fc_beta(gw))
    elif gw.fc_method == 'madelung':
        c_head = tools.madelung(cell, gw.kpts)
        c_inv_q = madelung_inv_q(cell, nks)
    else:
        raise ValueError('Unknown finite-size correction method %s' % str(gw.fc_method))
    return c_head, np.sqrt(4.0 * np.pi * cell.vol) * c_inv_q


def get_qdirs(gw):
    """Unit q directions and weights for the q -> 0 dielectric head/wings.

    Returns dirs (ndir, 3) in Cartesian coordinates and weights (ndir,) summing
    to one.  fc_qdirs=None follows PySCF: the single direction b1, or with
    fc_grid the 26 positive-octant points.  An integer selects that many
    Lebedev points (6 gives +-x, +-y, +-z).
    """
    cell = gw.mol
    if gw.fc_qdirs is None:
        if not gw.fc_grid:
            q_pts = np.array([[1e-3, 0, 0]])
        else:
            nq = 3
            q_pts = lib.cartesian_prod([np.arange(nq)] * 3)[1:, ::-1] * 5e-4
        dirs = cell.get_abs_kpts(q_pts)
        dirs /= np.linalg.norm(dirs, axis=1)[:, None]
        return dirs, np.full(len(dirs), 1.0 / len(dirs))
    if gw.fc_qdirs not in LEBEDEV_NGRID[1:]:
        raise ValueError('fc_qdirs must be None or a Lebedev grid size %s' % str(LEBEDEV_NGRID[1:]))
    grid = MakeAngularGrid(gw.fc_qdirs)
    return grid[:, :3], grid[:, 3]


def get_eps_inv_q0(omega, mo_energy, Lia, Pi_inv, qij_cart, dirs, weights):
    """Direction-averaged q -> 0 inverse dielectric head and wings at freq iw.

    k.p gives qij(q) = q . d exactly linearly, so the Cartesian components fix
    eps^-1_00(q^) = 1 / (q^T M q^) for every direction, with

        M_ab = delta_ab - 4 pi P_ab - w_a^* Pi_inv w_b,
        P_ab = 4/Nk sum_k,ia eia qij_a^* qij_b,   eps_P0(q^) = sum_a q^_a w_a,

    i.e. equations 47-48 and 53-54 in 10.1021/acs.jctc.0c00704 for any q^.

    Parameters
    ----------
    omega : double
        real position of imaginary frequency
    mo_energy : double 2d array
        orbital energy
    Lia : complex 4d ndarray
        (nkpts, naux, nocc, nvir) occupied-virtual DF integrals at q = 0
    Pi_inv : complex ndarray
        (naux, naux) inverse of (1 - Pi) at q = 0
    qij_cart : complex ndarray
        (3, nkpts, nocc, nvir) k.p pair densities for unit Cartesian q
    dirs, weights : double ndarray
        (ndir, 3) unit directions and (ndir,) weights

    Returns
    -------
    eps_inv_00 : complex
        weighted average of the inverse dielectric head
    eps_inv_P0 : complex ndarray
        (naux,) weighted average of the inverse dielectric wing
    """
    nkpts, naux, nocc, nvir = Lia.shape
    P = np.zeros((3, 3), dtype=np.complex128)
    for k in range(nkpts):
        eia = mo_energy[k, :nocc, None] - mo_energy[k, None, nocc:]
        eia = eia / (omega**2 + eia**2)
        P += 4.0 / nkpts * einsum('xia,ia,yia->xy', qij_cart[:, k].conj(), eia, qij_cart[:, k])
    w = np.array([-np.sqrt(4.0 * np.pi) * get_rho_response_wing(omega, mo_energy, Lia, qij_cart[x])
                  for x in range(3)])
    Pi_inv_w = np.matmul(Pi_inv, w.T)
    M = np.eye(3) - 4.0 * np.pi * P - np.matmul(w.conj(), Pi_inv_w)
    eps_inv_00 = 1.0 / einsum('gx,xy,gy->g', dirs, M, dirs)
    eps_inv_P0 = -np.matmul(Pi_inv_w, np.matmul(weights * eps_inv_00, dirs))
    return np.dot(weights, eps_inv_00), eps_inv_P0


def get_lij_q0(gw, mo_coeff):
    """Three-center DF integrals (L|ij) at q = 0, i.e. the (k, k) pairs.

    Returns a complex ndarray of shape (nkpts, naux, nmo, nmo).
    """
    kpts = gw.kpts
    nao = mo_coeff[0].shape[0]
    nmo = mo_coeff[0].shape[1]
    cderiarr = gw.with_df.cderi_array()
    Lij = []
    for k in range(len(kpts)):
        Lpq = cderiarr.load(kpts[k], kpts[k])
        if Lpq.shape[-1] == (nao * (nao + 1)) // 2:
            Lpq = lib.unpack_tril(Lpq).reshape(-1, nao**2)
        else:
            Lpq = Lpq.reshape(-1, nao**2)
        Lpq = Lpq.astype(np.complex128)
        moij, ijslice = _conc_mos(mo_coeff[k], mo_coeff[k])[2:]
        Lij.append(_ao2mo.r_e2(Lpq, moij, ijslice, tao=[], ao_loc=None, out=None).reshape(-1, nmo, nmo))
    return np.ascontiguousarray(Lij)


def get_sigma_fc(gw, freqs, wts, omega, mo_energy, orbs=None, kptlist=None, mo_coeff=None,
                 fullsigma=False, fc_coeffs=None):
    """Head and wing q -> 0 corrections to the GW correlation self-energy.
    See equations 43-45 in 10.1021/acs.jctc.0c00704

    Only the q = 0 (Gamma) block of the response is rebuilt, so the cost is
    about 1/nkpts of get_sigma.

    Parameters
    ----------
    gw : KRGWAC_ewald
        GW object
    freqs : double array
        position of imaginary frequency
    wts : double array
        weight of frequency points
    omega : complex ndarray
        imaginary frequency grids of self-energy, as returned by get_sigma
    mo_energy : double ndarray
        non-frozen orbital energy
    orbs : list, optional
        orbital index in non-frozen nmo to calculate self-energy, by default None
    kptlist : list, optional
        k-point index to calculate self-energy, by default None
    mo_coeff : complex ndarray, optional
        coefficient from AO to non-frozen MO, by default None
    fullsigma : bool, optional
        shape the correction for off-diagonal self-energy, by default False
    fc_coeffs : tuple, optional
        (c_head, c_wing) from get_fc_coeffs, by default None

    Returns
    -------
    sigma : complex ndarray
        (nklist, norbs, nw_sigma) correction, or (nklist, norbs, norbs, nw_sigma)
        if fullsigma; only the orbital diagonal is nonzero
    """
    nocc = gw.nocc
    nmo = gw.nmo
    nkpts = gw.nkpts
    if orbs is None:
        orbs = list(range(nmo))
    if kptlist is None:
        kptlist = list(range(nkpts))
    orbs = np.asarray(orbs)
    norbs = len(orbs)
    nw_sigma = len(omega)
    if mo_coeff is None:
        mo_coeff = _mo_frozen(gw, gw.mo_coeff)
    if fc_coeffs is None:
        fc_coeffs = gw.get_fc_coeffs()
    c_head, c_wing = fc_coeffs
    dirs, dir_wts = gw.get_qdirs()

    # k.p pair densities for unit Cartesian q: (3, nkpts, nocc, nvir)
    qij_cart = np.asarray([get_qij(gw, e, mo_energy, mo_coeff) for e in np.eye(3)])

    Lij = get_lij_q0(gw, mo_coeff)
    naux = Lij.shape[1]
    Lia = np.ascontiguousarray(Lij[:, :, :nocc, nocc:])
    Lnn = np.einsum('kPnn->kPn', Lij)[:, :, orbs]
    Lij = None

    emo = omega[None, None, :] - mo_energy[:, :, None]
    if fullsigma is False:
        sigma = np.zeros(shape=[len(kptlist), norbs, nw_sigma], dtype=np.complex128)
    else:
        sigma = np.zeros(shape=[len(kptlist), norbs, norbs, nw_sigma], dtype=np.complex128)
    diag = np.arange(norbs)
    for w in range(len(freqs)):
        Pi = get_rho_response(freqs[w], mo_energy, Lia, np.arange(nkpts))
        Pi_inv = np.linalg.inv(np.eye(naux) - Pi)
        eps_inv_00, eps_inv_P0 = get_eps_inv_q0(freqs[w], mo_energy, Lia, Pi_inv, qij_cart, dirs, dir_wts)
        # head correction, equation 43 in 10.1021/acs.jctc.0c00704
        Del_00 = c_head * (eps_inv_00 - 1.0)
        g0 = wts[w] * emo / (emo**2 + freqs[w]**2)
        for k, kn in enumerate(kptlist):
            # wing correction, equation 45 in 10.1021/acs.jctc.0c00704
            Wn_P0 = np.matmul(eps_inv_P0, Lnn[kn]).real * 2.0
            tmp = -(Del_00 + c_wing * Wn_P0)[:, None] * g0[kn][orbs] / np.pi
            if fullsigma is False:
                sigma[k] += tmp
            else:
                sigma[k, diag, diag] += tmp
    return sigma


def kernel(gw):
    mf = gw._scf
    nocc = gw.nocc
    nmo = gw.nmo
    nkpts = gw.nkpts

    # set frozen orbitals
    gw.set_frozen_orbs()
    orbs = gw.orbs
    orbs_frz = gw.orbs_frz
    kptlist = gw.kptlist
    if kptlist is None:
        gw.kptlist = kptlist = range(gw.nkpts)
    mo_energy_frz = _mo_energy_frozen(gw, gw.mo_energy)

    # v_xc
    mo_coeff_frz = _mo_frozen(gw, gw.mo_coeff)
    with temporary_env(mf, verbose=0), temporary_env(mf.mol, verbose=0), temporary_env(mf.with_df, verbose=0):
        dm = mf.make_rdm1()
        v_mf_ao = mf.get_veff() - mf.get_j(dm_kpts=dm)
    v_mf = np.zeros(shape=[nkpts, nmo, nmo], dtype=np.complex128)
    for k in range(nkpts):
        v_mf[k] = reduce(np.matmul, (mo_coeff_frz[k].T.conj(), v_mf_ao[k], mo_coeff_frz[k]))
    gw.vxc = v_mf

    # v_hf from DFT/HF density
    vk = gw.get_sigma_exchange()

    # finite size correction for exchange self-energy
    if gw.fc:
        fc_coeffs = gw.get_fc_coeffs()
        for k in range(nkpts):
            for i in range(nocc):
                vk[k][i, i] = vk[k][i, i] - fc_coeffs[0]
    gw.vk = vk

    # set up Fermi level
    gw.ef = ef = get_ef(kmf=mf, mo_energy=mf.mo_energy)

    # grids for integration on imaginary axis
    gw.freqs, gw.wts = freqs, wts = _get_scaled_legendre_roots(gw.nw)

    # calculate self-energy on imaginary axis without q -> 0 corrections.
    # PySCF's get_sigma mishandles non-contiguous or reordered orbital lists,
    # so evaluate the containing contiguous range and select afterwards.
    sigma_orbs = range(min(orbs_frz), max(orbs_frz) + 1)
    with temporary_env(gw, fc=False):
        if gw.outcore:
            sigmaI, omega = get_sigma_outcore(
                gw, freqs, wts, ef=ef, mo_energy=mo_energy_frz, orbs=sigma_orbs, kptlist=kptlist,
                iw_cutoff=gw.ac_iw_cutoff, fullsigma=gw.fullsigma,
            )
        else:
            sigmaI, omega = get_sigma(
                gw, freqs, wts, ef=ef, mo_energy=mo_energy_frz, orbs=sigma_orbs, kptlist=kptlist,
                iw_cutoff=gw.ac_iw_cutoff, fullsigma=gw.fullsigma,
            )
    selected = np.asarray(orbs_frz) - sigma_orbs.start
    sigmaI = np.take(sigmaI, selected, axis=1)
    if gw.fullsigma:
        sigmaI = np.take(sigmaI, selected, axis=2)

    # q -> 0 head and wing corrections
    if gw.fc:
        sigmaI += gw.get_sigma_fc(
            freqs, wts, omega, mo_energy_frz, orbs=orbs_frz, kptlist=kptlist,
            fullsigma=gw.fullsigma, fc_coeffs=fc_coeffs,
        )
    if gw.rdm:
        gw.sigmaI = sigmaI

    # analytic continuation
    if gw.ac == 'twopole':
        acobj = TwoPoleAC(list(range(nmo)), nocc)
    elif gw.ac == 'pade':
        acobj = PadeAC(npts=gw.ac_pade_npts, step_ratio=gw.ac_pade_step_ratio)
    else:
        raise ValueError('Unknown GW-AC type %s' % (str(gw.ac)))

    acobj.ac_fit(sigmaI, omega, axis=-1)

    if gw.fullsigma:
        diag_acobj = acobj.diagonal(axis1=1, axis2=2)
    else:
        diag_acobj = acobj

    mo_energy = np.zeros_like(mf.mo_energy)
    for ik, k in enumerate(kptlist):
        for ip, p in enumerate(orbs_frz):
            if gw.qpe_linearized:
                # linearized G0W0
                de = 1e-6
                ep = mf.mo_energy[k][orbs[ip]]
                sigmaR = diag_acobj[ik, ip].ac_eval(ep).real
                dsigma = diag_acobj[ik, ip].ac_eval(ep + de).real - sigmaR.real
                zn = 1.0 / (1.0 - dsigma / de)
                if gw.qpe_linearized_range is not None:
                    zn = 1.0 if zn < gw.qpe_linearized_range[0] or zn > gw.qpe_linearized_range[1] else zn
                mo_energy[k, orbs[ip]] = ep + zn * (sigmaR + vk[k, p, p] - v_mf[k, p, p]).real
            else:
                # self-consistently solve QP equation
                def quasiparticle(omega):
                    sigmaR = diag_acobj[ik, ip].ac_eval(omega)
                    return omega - mf.mo_energy[k][orbs[ip]] - (sigmaR + vk[k, p, p] - v_mf[k, p, p]).real

                try:
                    mo_energy[k, orbs[ip]] = scipy.optimize.newton(
                        quasiparticle, mf.mo_energy[k][orbs[ip]], tol=gw.qpe_tol, maxiter=gw.qpe_max_iter
                    )
                except RuntimeError:
                    logger.warn(gw, 'QPE for k=%d orbital=%d not converged!', k, orbs[ip])

    # save GW results
    gw.mo_energy = mo_energy
    gw.acobj = acobj

    with np.printoptions(threshold=len(mf.mo_energy[0])):
        for k in range(nkpts):
            logger.debug(gw, '  GW mo_energy @ k%d =\n%s', k, mo_energy[k])
    logger.warn(gw, 'GW QP energies may not be sorted from min to max')

    if gw.writefile > 0:
        with h5py.File('vxc.h5', 'w') as feri:
            feri['vk'] = np.asarray(vk)
            feri['v_mf'] = np.asarray(v_mf)

        with h5py.File('sigma_imag.h5', 'w') as feri:
            feri['sigmaI'] = np.asarray(sigmaI)
            feri['omega'] = np.asarray(omega)
            if gw.sigmaI is not None:
                feri['sigmaI_full'] = np.asarray(gw.sigmaI)

        acobj.save('ac_coeff.h5')

    return


class KRGWAC_ewald(krgw_ac.KRGWAC):
    """KRGWAC with selectable q -> 0 finite-size corrections.

    Attributes
    ----------
    fc_method : str
        'aux' (Gaussian auxiliary functions, FHI-gap App. B), 'madelung'
        (infinite Gaussian width), or 'sphere' (PySCF). Default 'aux'.
    fc_sigma : float or None
        Gaussian width (1/bohr) of the 'aux' functions, beta = 1/(2 sigma^2).
        None uses the FHI-gap beta = (Omega/6 pi^2)^(1/3).
    fc_qdirs : int or None
        Number of Lebedev directions over which the k.p dielectric head and
        wings are averaged.  None uses PySCF's direction(s).
    """
    _keys = {'fc_method', 'fc_sigma', 'fc_qdirs'}

    def __init__(self, mf, frozen=None):
        krgw_ac.KRGWAC.__init__(self, mf, frozen=frozen)
        self.fc_method = 'aux'
        self.fc_sigma = None
        self.fc_qdirs = None

    def dump_flags(self, verbose=None):
        krgw_ac.KRGWAC.dump_flags(self, verbose)
        log = logger.Logger(self.stdout, self.verbose)
        if self.fc:
            log.info('finite size correction method = %s', self.fc_method)
            if self.fc_method == 'aux':
                log.info('auxiliary function beta = %.6g bohr^2', get_fc_beta(self))
            c_head, c_wing = self.get_fc_coeffs()
            log.info('finite size coefficients: head = %.10g, wing = %.10g', c_head, c_wing)
            log.info('q -> 0 directions = %d', len(self.get_qdirs()[1]))
            log.info('')
        return

    def kernel(self, orbs=None, kptlist=None):
        """Run a G0W0 calculation.

        Parameters
        ----------
        orbs : list, optional
            orbital list to calculate self-energy, by default None
        kptlist : list, optional
            k-point list to calculate self-energy, by default None
        """
        if self.mo_energy is None:
            self.mo_energy = np.array(self._scf.mo_energy, copy=True)
        if self.mo_coeff is None:
            self.mo_coeff = np.array(self._scf.mo_coeff, copy=True)
        if self.mo_occ is None:
            self.mo_occ = np.array(self._scf.mo_occ, copy=True)

        self.orbs = orbs
        self.kptlist = kptlist

        if hasattr(self._scf, "sigma"):
            self.nw = max(400, self.nw)
            self.ac_pade_npts = 18
            self.ac_pade_step_ratio = 5.0 / 6.0
            self.fc = False
        if self.fc:
            _validate_fc_setup(self)
        if self.with_df._cderi is None or self.with_df._j_only:
            # Pure DFT often builds only Coulomb (k,k) blocks, but GW also needs
            # (k,k') blocks; build them in a private cache, preserving the SCF DF.
            self.with_df = copy(self.with_df).reset()
            self.with_df._cderi_to_save = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
            self.with_df.build(j_only=False)

        nmo = self.nmo
        naux = self.with_df.get_naoaux()
        nkpts = self.nkpts
        mem_incore = (2 * nkpts * nmo**2 * naux) * 16 / 1e6
        mem_now = lib.current_memory()[0]
        if mem_incore + mem_now > 0.99 * self.max_memory:
            logger.warn(self, 'Memory may not be enough!')

        cput0 = (time.process_time(), time.perf_counter())
        self.dump_flags()
        kernel(self)
        logger.timer(self, 'KRGWAC_ewald', *cput0)
        return

    get_fc_coeffs = get_fc_coeffs
    get_qdirs = get_qdirs
    get_sigma_fc = get_sigma_fc
