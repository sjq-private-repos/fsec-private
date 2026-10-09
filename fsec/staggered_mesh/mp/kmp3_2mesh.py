"""Restricted periodic MP3 with two staggered k-point meshes.

Based on Xin Xing's staggered-mesh formulation. The method uses either
two commensurate submeshes of the supplied SCF mesh or band orbitals from a
non-SCF extension of that mesh.
"""


import time
import numpy as np
import concurrent.futures

from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import df
from pyscf.pbc.scf import khf
from pyscf.pbc.dft import krks
from pyscf.pbc.mp.kmp2 import (get_nocc, padded_mo_energy, padded_mo_coeff,
                               padding_k_idx)
from pyscf.pbc.lib import kpts_helper

from pyscf.pbc.tools.pbc import get_monkhorst_pack_size
from pyscf.pbc.scf.khf import get_occ

from pyscf.pbc.cc.kccsd_rhf import _get_epq

from .kmp3 import KMP3
from .._kpoints import make_kpts_helper
from pyscf.pbc import tools

einsum = lib.einsum

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

def mp3_amplitude_2mesh_efficient(mp, eris):
    time0 = time1 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mp.stdout, mp.verbose)
    time0 = log.timer_debug1('start', *time0)
    time1 = log.timer_debug1('start', *time1)

    #   basic information
    nkpts = mp.nkpts_ov
    nocc = mp.nocc
    nmo = mp.nmo
    nvir = nmo - nocc
    kconserv = mp.khelper.kconserv
    mo_e_o = [e[:nocc] for e in eris.mo_energy]
    mo_e_v = [e[nocc:] for e in eris.mo_energy]
    nonzero_opadding, nonzero_vpadding = padding_k_idx(mp, kind="split")


    #   auxiliary MP2 t-amplitude
    t_mp2 = np.empty((nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir), dtype=eris.dtype)
    for iki, ikj, ika in kpts_helper.loop_kkk(nkpts):
        ki, kj, ka = mp.kpts_idx_occ[iki], mp.kpts_idx_occ[ikj], mp.kpts_idx_vir[ika]
        kb = kconserv[ki, ka, kj]
        eia = _get_epq([0,nocc,ki,mo_e_o,nonzero_opadding],
                       [0,nvir,ka,mo_e_v,nonzero_vpadding],
                       fac=[1.0,-1.0])
        ejb = _get_epq([0,nocc,kj,mo_e_o,nonzero_opadding],
                       [0,nvir,kb,mo_e_v,nonzero_vpadding],
                       fac=[1.0,-1.0])
        eijab = eia[:, None, :, None] + ejb[None, :, None, :]
        t_mp2[iki, ikj, ika] = eris.oovv[iki, ikj, ika].conj() / eijab
    mp._t2_mp2 = t_mp2
    time1 = log.timer_debug1('MP2 amp preparation', *time1)

    t_mp3= np.zeros_like(t_mp2)

    #   4h2p term:
    for iki, ikj, ika in kpts_helper.loop_kkk(nkpts):
        ki, kj, ka = mp.kpts_idx_occ[iki], mp.kpts_idx_occ[ikj], mp.kpts_idx_vir[ika]
        kb = kconserv[ki, ka, kj]
        ikb = mp.idx_vir[kb]

        kk_kl = np.asarray([ [mp.idx_occ[kconserv[kj, kl, ki]], ikl] for ikl, kl in enumerate(mp.kpts_idx_occ)])
        t2new_tmp = 0.5 * einsum('xklij,xklab->ijab', eris.oooo[kk_kl[:,0], kk_kl[:,1], iki], t_mp2[kk_kl[:,0], kk_kl[:,1], ika])
        t_mp3[iki, ikj, ika] += t2new_tmp
        t_mp3[ikj, iki, ikb] += t2new_tmp.transpose(1, 0, 3, 2)
    time1 = log.timer_debug1('4h2p', *time1)

    #   2h4p term: )
    for iki, ikj, ika in kpts_helper.loop_kkk(nkpts):
        ki, kj, ka = mp.kpts_idx_occ[iki], mp.kpts_idx_occ[ikj], mp.kpts_idx_vir[ika]
        kb = kconserv[ki, ka, kj]
        ikb = mp.idx_vir[kb]

        t2new_tmp = 0.5 * einsum('xabcd,xijcd->ijab', eris.vvvv[ika,ikb,:], t_mp2[iki, ikj, :])
        t_mp3[iki, ikj, ika] += t2new_tmp
        t_mp3[ikj, iki, ikb] += t2new_tmp.transpose(1, 0, 3, 2)
    time1 = log.timer_debug1('2h4p', *time1)

    #   3h3p term
    for iki, ikj, ika in kpts_helper.loop_kkk(nkpts):
        ki, kj, ka = mp.kpts_idx_occ[iki], mp.kpts_idx_occ[ikj], mp.kpts_idx_vir[ika]
        kb = kconserv[ki, ka, kj]
        ikb = mp.idx_vir[kb]

        t2new_tmp = np.zeros((nocc, nocc, nvir, nvir), dtype=eris.dtype)
        #    term 1 & 2
        kk_kc = np.asarray([ [ikk, mp.idx_vir[kconserv[ka, ki, kk]]] for ikk, kk in enumerate(mp.kpts_idx_occ)] )
        t2new_tmp += 2*einsum('xakic,xkjcb->ijab', eris.voov[ika, kk_kc[:,0], iki], t_mp2[kk_kc[:,0], ikj, kk_kc[:,1]])
        t2new_tmp -= einsum('xakic,xkjcb->ijab', eris.ovov[kk_kc[:,0], ika, iki].transpose(0, 2, 1, 3, 4), t_mp2[kk_kc[:,0], ikj, kk_kc[:,1]])

        #   term 3
        t2new_tmp -= einsum('xakic,xkjbc->ijab', eris.voov[ika,:,iki], t_mp2[:, ikj, ikb])

        #   term 4
        t2new_tmp -= einsum('xkbic,xkjac->ijab', eris.ovov[:,ikb,iki], t_mp2[:, ikj, ika])

        t_mp3[iki, ikj, ika] += t2new_tmp
        t_mp3[ikj, iki, ikb] += t2new_tmp.transpose(1, 0, 3, 2)
    time1 = log.timer_debug1('3h3p', *time1)

    #   Madelung correction
    if mp.madelung:
        t_mp3 += 2 * mp.madelung_constant * t_mp2
    time1 = log.timer_debug1('Madelung', *time1)

    #   MP2 terms
    for iki, ikj, ika in kpts_helper.loop_kkk(nkpts):
        ki, kj, ka = mp.kpts_idx_occ[iki], mp.kpts_idx_occ[ikj], mp.kpts_idx_vir[ika]
        kb = kconserv[ki, ka, kj]
        eia = _get_epq([0,nocc,ki,mo_e_o,nonzero_opadding],
                       [0,nvir,ka,mo_e_v,nonzero_vpadding],
                       fac=[1.0,-1.0])
        ejb = _get_epq([0,nocc,kj,mo_e_o,nonzero_opadding],
                       [0,nvir,kb,mo_e_v,nonzero_vpadding],
                       fac=[1.0,-1.0])
        eijab = eia[:, None, :, None] + ejb[None, :, None, :]
        t_mp3[iki, ikj, ika] /= eijab
        t_mp3[iki, ikj, ika] += t_mp2[iki, ikj, ika]
    time1 = log.timer_debug1('orbital energy', *time1)

    time0 = log.timer_debug1('compute MP3 t-amplitude', *time0)
    return t_mp3

def mp3_amplitude_2mesh(mp, eris):
    time0 = time1 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mp.stdout, mp.verbose)
    time0 = log.timer_debug1('start', *time0)
    time1 = log.timer_debug1('start', *time1)

    #   basic information
    nkpts = mp.nkpts_ov
    nocc = mp.nocc
    nmo = mp.nmo
    nvir = nmo - nocc
    kconserv = mp.khelper.kconserv
    mo_e_o = [e[:nocc] for e in eris.mo_energy]
    mo_e_v = [e[nocc:] for e in eris.mo_energy]
    nonzero_opadding, nonzero_vpadding = padding_k_idx(mp, kind="split")

    if mp.madelung:
        madelung_correction = tools.madelung(mp.cell, mp.kpts_vir)

    #   auxiliary MP2 t-amplitude
    t_mp2 = np.empty((nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir), dtype=eris.dtype)
    for iki, ikj, ika in kpts_helper.loop_kkk(nkpts):
        ki, kj, ka = mp.kpts_idx_occ[iki], mp.kpts_idx_occ[ikj], mp.kpts_idx_vir[ika]
        kb = kconserv[ki, ka, kj]
        eia = _get_epq([0,nocc,ki,mo_e_o,nonzero_opadding],
                       [0,nvir,ka,mo_e_v,nonzero_vpadding],
                       fac=[1.0,-1.0])
        ejb = _get_epq([0,nocc,kj,mo_e_o,nonzero_opadding],
                       [0,nvir,kb,mo_e_v,nonzero_vpadding],
                       fac=[1.0,-1.0])
        eijab = eia[:, None, :, None] + ejb[None, :, None, :]
        t_mp2[iki, ikj, ika] = eris.oovv[iki, ikj, ika].conj() / eijab
    mp._t2_mp2 = t_mp2
    time1 = log.timer_debug1('MP2 amp preparation', *time1)

    t_mp3= np.zeros_like(t_mp2)

    #   1st term (4h2p):
    #   "oooo term" X_{ij}^{kl} t_{kl}^{ab}
    for iki, ikj, ika in kpts_helper.loop_kkk(nkpts):
        ki, kj, ka = mp.kpts_idx_occ[iki], mp.kpts_idx_occ[ikj], mp.kpts_idx_vir[ika]
        kb = kconserv[ki, ka, kj]
        ikb = mp.idx_vir[kb]

        t2new_tmp = np.zeros((nocc, nocc, nvir, nvir), dtype=eris.dtype)
        for ikl, kl in enumerate(mp.kpts_idx_occ):
            kk = kconserv[kj, kl, ki]
            ikk = mp.idx_occ[kk]
            t2new_tmp += 0.5 * einsum('klij,klab->ijab', eris.oooo[ikk, ikl, iki], t_mp2[ikk, ikl, ika])
            # NOTE: 0.5 from the symmetrization below, this quantity is the same for both

        if mp.madelung:
            t2new_tmp += 0.5 * madelung_correction * t_mp2[iki, ikj, ika]

        t_mp3[iki, ikj, ika] += t2new_tmp
        t_mp3[ikj, iki, ikb] += t2new_tmp.transpose(1, 0, 3, 2)
    time1 = log.timer_debug1('4h2p', *time1)

    #   2nd term (2h4p):
    #   "vvvv term" X_{cd}^{ab} t_{ij}^{cd}
    for iki, ikj, ika in kpts_helper.loop_kkk(nkpts):
        ki, kj, ka = mp.kpts_idx_occ[iki], mp.kpts_idx_occ[ikj], mp.kpts_idx_vir[ika]
        kb = kconserv[ki, ka, kj]
        ikb = mp.idx_vir[kb]

        t2new_tmp = np.zeros((nocc, nocc, nvir, nvir), dtype=eris.dtype)
        for ikc, kc in enumerate(mp.kpts_idx_vir):
            t2new_tmp += 0.5 * einsum('abcd,ijcd->ijab', eris.vvvv[ika, ikb, ikc], t_mp2[iki, ikj, ikc])

        if mp.madelung:
            t2new_tmp += 0.5 * madelung_correction * t_mp2[iki, ikj, ika]

        t_mp3[iki, ikj, ika] += t2new_tmp
        t_mp3[ikj, iki, ikb] += t2new_tmp.transpose(1, 0, 3, 2)
    time1 = log.timer_debug1('2h4p', *time1)

    #   3rd term
    for iki, ikj, ika in kpts_helper.loop_kkk(nkpts):
        ki, kj, ka = mp.kpts_idx_occ[iki], mp.kpts_idx_occ[ikj], mp.kpts_idx_vir[ika]
        kb = kconserv[ki, ka, kj]
        ikb = mp.idx_vir[kb]

        t2new_tmp = np.zeros((nocc, nocc, nvir, nvir), dtype=eris.dtype)
        for ikk, kk in enumerate(mp.kpts_idx_occ):
            kc = kconserv[ka, ki, kk]
            ikc = mp.idx_vir[kc]
            tmp_voov = 2. * eris.voov[ika, ikk, iki] - eris.ovov[ikk, ika, iki].transpose(1, 0, 2, 3)
            t2new_tmp += einsum('akic,kjcb->ijab', tmp_voov, t_mp2[ikk, ikj, ikc])

            kc = kconserv[ka, ki, kk]
            ikc = mp.idx_vir[kc]
            t2new_tmp -= einsum('akic,kjbc->ijab', eris.voov[ika, ikk, iki], t_mp2[ikk, ikj, ikb])

            kc = kconserv[kk, ka, kj]
            ikc = mp.idx_vir[kc]
            t2new_tmp -= einsum('kbic,kjac->ijab', eris.ovov[ikk, ikb, iki], t_mp2[ikk, ikj, ika])

        if mp.madelung:
            t2new_tmp -= 2 * madelung_correction * t_mp2[iki, ikj, ika]

        t_mp3[iki, ikj, ika] += t2new_tmp
        t_mp3[ikj, iki, ikb] += t2new_tmp.transpose(1, 0, 3, 2)
    time1 = log.timer_debug1('3h3p', *time1)

    #   MP2 terms
    for iki, ikj, ika in kpts_helper.loop_kkk(nkpts):
        ki, kj, ka = mp.kpts_idx_occ[iki], mp.kpts_idx_occ[ikj], mp.kpts_idx_vir[ika]
        kb = kconserv[ki, ka, kj]
        eia = _get_epq([0,nocc,ki,mo_e_o,nonzero_opadding],
                       [0,nvir,ka,mo_e_v,nonzero_vpadding],
                       fac=[1.0,-1.0])
        ejb = _get_epq([0,nocc,kj,mo_e_o,nonzero_opadding],
                       [0,nvir,kb,mo_e_v,nonzero_vpadding],
                       fac=[1.0,-1.0])
        eijab = eia[:, None, :, None] + ejb[None, :, None, :]
        t_mp3[iki, ikj, ika] /= eijab
        t_mp3[iki, ikj, ika] += t_mp2[iki, ikj, ika]
    time1 = log.timer_debug1('orbital energy', *time1)

    time0 = log.timer_debug1('compute MP3 t-amplitude', *time0)
    return t_mp3

def mp3_energy_2mesh(mp, t2, eris):
    nkpts = mp.nkpts_ov
    kconserv = mp.khelper.kconserv
    e = 0.0 + 1j * 0.0
    for iki, ikj, ika in kpts_helper.loop_kkk(nkpts):
        ki, kj, ka = mp.kpts_idx_occ[iki], mp.kpts_idx_occ[ikj], mp.kpts_idx_vir[ika]
        kb = kconserv[ki, ka, kj]
        ikb = mp.idx_vir[kb]
        e += 2 * einsum('ijab,ijab', t2[iki, ikj, ika], eris.oovv[iki, ikj, ika])
        e += - einsum('ijab,ijba', t2[iki, ikj, ika], eris.oovv[iki, ikj, ikb])

    e /= nkpts
    if abs(e.imag) > 1e-4:
        logger.warn(mp, 'Non-zero imaginary part found in MP3 energy %s', e)
    return e.real

class KMP3_2mesh(KMP3):
    """Two-mesh staggered restricted MP3.

    ``flag_submesh=True`` partitions the reference mesh into two submeshes;
    otherwise band orbitals are evaluated on an extended mesh.  The source
    method's Madelung amplitude correction remains enabled by default.
    """

    def __init__(self, mf, frozen=None, flag_submesh=False, nks_mp=None,
                 madelung=True):
        self.cell = mf.cell
        self._scf = mf
        self.verbose = self.cell.verbose
        self.stdout = self.cell.stdout
        self.max_memory = self.cell.max_memory

        #   MP3 energy
        self.e_corr = None

        #   nocc and nmo variables
        self._nocc = None
        self._nmo = None
        self._t2_mp2 = None
        self.t2_mp3 = None
        self.e_corr_mp2 = None
        self.e_corr_mp3 = None

        #   Construction of orbitals and orbital energies on two staggered meshes
        nks = get_monkhorst_pack_size(mf.cell, mf.kpts)
        self.flag_submesh = flag_submesh
        if flag_submesh is True:
            #   Choice 1: Use two staggered submeshes of mf.kpts of half size along each dimension
            if (  (nks % 2 == 1) & (nks > 1) ).any() :
                log = logger.Logger(self.stdout, self.verbose)
                log.warn('The k-point mesh in the input mf has odd size in certain dimension. \n'
                        'Cannot use its submeshes for staggered mesh calculation!\n'
                        'You may set "flag_submesh" to False')
                raise RuntimeError

            #   Temporary solution for low-dimensional case
            nonperiodic_dir = np.where( nks == 1)
            nks_ov = np.asarray([n // 2 if n > 1 else 1 for n in nks], dtype=int)
            shift = mf.kpts[ np.argmin( np.sum(mf.kpts**2, axis=1) )]   #   shift of the original k-point mesh
            if np.any(shift[nonperiodic_dir]):
                log = logger.Logger(self.stdout, self.verbose)
                log.warn("The given k-point mesh has non-zero shift in the nonperiodic direction")
                raise RuntimeError
            half_shift = mf.cell.get_abs_kpts( [1/n if n > 1 else 0 for n in nks] )

            kpts_vir = shift + mf.cell.make_kpts(nks_ov, with_gamma_point=True)
            kpts_occ = kpts_vir + half_shift

            self.kpts = mf.kpts
            self.mo_energy = mf.mo_energy
            self.mo_coeff = mf.mo_coeff
            self.mo_occ = get_occ(mf, mo_energy_kpts=mf.mo_energy)
            self.with_df = mf.with_df
        else:
            #   Choice 2: Apply non-SCF calculation to compute orbitals and orbital energies on a
            #             staggered mesh of the same size as mf.kpts.
            if mf.cell.dimension < 3:
                # The non-SCF calculation of orbitals/orbital energies is currently only valid
                # for 3D systems in PySCF. For example, when using get_band with a reference HF
                # system of mesh 1*1*k to compute orbitals and orbital energies on a 1*1*2k mesh,
                # the obtained mo_energy and  mo_coeff are not in good agreement with those computed
                # from an SCF-HF calculation with mesh 1*1*2k.
                raise NotImplementedError

            if nks_mp is None:
                nks_base = np.asarray(nks, dtype=int)
            else:
                mesh_input = np.asarray(nks_mp)
                mesh_integer = mesh_input.astype(int)
                if (mesh_input.shape != (3,) or np.any(mesh_input != mesh_integer)
                        or np.any(mesh_integer < 1)):
                    raise ValueError('nks_mp must contain three positive integers')
                nks_base = mesh_integer
            if all([n == 1 for n in nks_base]):
                half_shift = mf.cell.get_abs_kpts( [0.5/n for n in nks_base] )
            else:
                half_shift = mf.cell.get_abs_kpts( [0.5/n if n > 1 else 0 for n in nks_base] )

            if nks_mp is None:
                kpts_vir = mf.kpts
            else:
                kpts_vir = mf.cell.make_kpts(nks_base, with_gamma_point=True)
            kpts_occ = kpts_vir + half_shift
            kpts = np.concatenate( (kpts_occ, kpts_vir), axis=0)
            if isinstance(mf, krks.KRKS):
                with lib.temporary_env(mf, with_df=df.FFTDF(mf.cell, mf.kpts)):
                    mo_energy, mo_coeff = mf.get_bands(kpts)
                    mo_occ = get_occ(mf, mo_energy_kpts=mo_energy)
            elif isinstance(mf, khf.KRHF):
                with lib.temporary_env(mf, exxdiv='vcut_sph',
                                       with_df=df.FFTDF(mf.cell, mf.kpts)):
                    mo_energy, mo_coeff = mf.get_bands(kpts)
                    mo_occ = get_occ(mf, mo_energy_kpts=mo_energy)
            else:
                raise TypeError('staggered MP3 requires a periodic KRHF or KRKS reference')

            self.kpts = kpts
            self.mo_energy = mo_energy
            self.mo_coeff = mo_coeff
            self.mo_occ = mo_occ
            self.with_df = df.GDF(self.cell, kpts).build()

        #   Basic info
        self.nkpts = len(self.kpts)
        self.frozen = frozen
        self.khelper = make_kpts_helper(self.cell, self.kpts)

        #   Mesh info for the staggered mesh method
        self.kpts_vir = kpts_vir
        self.kpts_occ = kpts_occ
        self.nkpts_ov = len(self.kpts_vir)

        #   Map kpts, kpts_vir, and kpts_occ to the first BZ for better processing
        kpts_bz = minimum_image(self.cell, self.kpts)
        kpts_occ_bz = minimum_image(self.cell, self.kpts_occ)
        kpts_vir_bz = minimum_image(self.cell, self.kpts_vir)

        #   indices of virtual k-points in self.kpts
        self.kpts_idx_vir = [ np.asarray(np.sum((kpts_bz - kvir)**2, axis=-1) < 1e-10).nonzero()[0]
                                for kvir in kpts_vir_bz]
        self.kpts_idx_vir = np.concatenate(self.kpts_idx_vir).astype(int)

        #   indices of occupied k-points in self.kpts
        self.kpts_idx_occ = [ np.asarray(np.sum((kpts_bz - kocc)**2, axis=-1) < 1e-10).nonzero()[0]
                                for kocc in kpts_occ_bz]
        self.kpts_idx_occ = np.concatenate(self.kpts_idx_occ).astype(int)

        if (len(self.kpts_idx_vir) != self.nkpts_ov or
            len(np.unique(self.kpts_idx_vir)) != self.nkpts_ov or
            len(self.kpts_idx_occ) != self.nkpts_ov or
            len(np.unique(self.kpts_idx_occ)) != self.nkpts_ov):
            log = logger.Logger(self.stdout, self.verbose)
            log.warn('Cannot locate the provided occupied/virtual submeshes in the large k-point mesh')
            raise RuntimeError
        if np.intersect1d(self.kpts_idx_occ, self.kpts_idx_vir).size:
            raise ValueError('occupied and virtual submeshes must be disjoint')


        #   kflag_occ[i]=1/0 means self.kpts[i] is a occ/vir k-point
        self.kflag_occ = -np.ones(self.nkpts, dtype=int)
        self.kflag_occ[self.kpts_idx_occ] = 1
        self.kflag_occ[self.kpts_idx_vir] = 0

        #   kpts[p] is the 'self.idx_vir[p]'-th virtual k point
        self.idx_vir = -1 * np.ones(self.nkpts, dtype=int)
        self.idx_vir[self.kpts_idx_vir] = np.arange(self.nkpts_ov)

        #   kpts[p] is the 'self.idx_occ[p]'-th occupied k point
        self.idx_occ = -1 * np.ones(self.nkpts, dtype=int)
        self.idx_occ[self.kpts_idx_occ] = np.arange(self.nkpts_ov)

        #   Madelung constant
        self.madelung = madelung
        self.madelung_constant = -tools.madelung(mf.cell, kpts_vir)


    def ao2mo(self, mo_coeff=None, mo_energy=None):
        return _ERIS_2mesh(self, mo_coeff, mo_energy)

    double_amplitude = mp3_amplitude_2mesh_efficient
    # double_amplitude = mp3_amplitude_2mesh
    correlation_energy = mp3_energy_2mesh

def _mem_usage(nkpts, nocc, nvir):
    incore = 0
    #   oovv
    incore += nkpts ** 3 * (nocc**2 * nvir**2)
    #   oooo
    incore += nkpts ** 3 * nocc**4
    #   vvvv
    incore += nkpts ** 3 * nvir**4
    #   voov/ovov
    incore += 2 * nkpts ** 3 * (nocc**2 * nvir**2)

    # Roughly, factor of two for intermediates and factor of two
    # for safety (temp arrays, copying, etc)
    incore *= 1.5
    # TODO: Improve incore estimate and add outcore estimate
    outcore = basic = incore
    return incore * 16 / 1e6, outcore * 16 / 1e6, basic * 16 / 1e6

class _ERIS_2mesh:
    def __init__(self, cc, mo_coeff=None, mo_energy=None, method='incore'):
        log = logger.Logger(cc.stdout, cc.verbose)
        cput0 = (logger.process_clock(), logger.perf_counter())

        #   Basic information
        cell = cc._scf.cell
        kpts = cc.kpts
        nkpts = cc.nkpts
        nkpts_ov = cc.nkpts_ov
        nocc = cc.nocc
        nmo = cc.nmo
        nvir = nmo - nocc

        #   NOTE: DO NOT USE mo_coeff/energy from cc._scf
        if mo_coeff is None:
            mo_coeff = cc.mo_coeff
        if mo_energy is None:
            mo_energy = cc.mo_energy
        self.dtype = mo_coeff[0].dtype

        #   Padding MOs based on frozen parameter
        mo_coeff  = self.mo_coeff  = padded_mo_coeff(cc, mo_coeff)
        mo_energy = self.mo_energy = padded_mo_energy(cc, mo_energy)

        #   Location of padded elements in occupied and virtual space.
        nocc_per_kpt = get_nocc(cc, per_kpoint=True)
        nonzero_padding = padding_k_idx(cc, kind="joint")

        #   Check direct and indirect gaps for possible issues with CCSD convergence.
        mo_e = [self.mo_energy[kp][nonzero_padding[kp]] for kp in range(nkpts)]
        mo_e = np.sort([y for x in mo_e for y in x])  # Sort de-nested array
        gap = mo_e[np.sum(nocc_per_kpt)] - mo_e[np.sum(nocc_per_kpt)-1]
        if gap < 1e-5:
            logger.warn(cc, 'HOMO-LUMO gap %s too small for KCCD. '
                            'May cause issues in convergence.', gap)

        mem_incore, mem_outcore, mem_basic = _mem_usage(nkpts_ov, nocc, nvir)
        mem_now = lib.current_memory()[0]
        fao2mo = cc.with_df.ao2mo

        kconserv = cc.khelper.kconserv
        khelper = cc.khelper

        kflag_occ = cc.kflag_occ
        idx_occ = cc.idx_occ
        idx_vir = cc.idx_vir

        orbv = np.asarray(mo_coeff[cc.kpts_idx_vir,:,nocc:], order='C')
        orbo = np.asarray(mo_coeff[cc.kpts_idx_occ,:,:nocc], order='C')

        log.info("Two-mesh ERI memory estimate %.1f MB; current use %.1f MB; "
                 "limit %.1f MB", mem_incore, mem_now, cc.max_memory)

        if (method == 'incore' and (mem_incore + mem_now < cc.max_memory)
                or cell.incore_anyway):
            log.info("Multithread incore ERI tensor calculation")

            self.oovv = np.empty((nkpts_ov,nkpts_ov,nkpts_ov,nocc,nocc,nvir,nvir), dtype=self.dtype)
            self.voov = np.empty((nkpts_ov,nkpts_ov,nkpts_ov,nvir,nocc,nocc,nvir), dtype=self.dtype)
            self.oooo = np.empty((nkpts_ov,nkpts_ov,nkpts_ov,nocc,nocc,nocc,nocc), dtype=self.dtype)
            self.ovov = np.empty((nkpts_ov,nkpts_ov,nkpts_ov,nocc,nvir,nocc,nvir), dtype=self.dtype)
            self.vvvv = np.empty((nkpts_ov,nkpts_ov,nkpts_ov,nvir,nvir,nvir,nvir), dtype=self.dtype)

            start = time.time()
            key_list = list(khelper.symm_map)
            def eri_kpt_calc(idx):
                (ikp,ikq,ikr) = key_list[idx]
                iks = kconserv[ikp,ikq,ikr]

                #   Compute ERIs with even numbers of occ and virtual orbitals
                if sum(kflag_occ[ [ikp, ikq, ikr, iks] ]) % 2:
                    return None

                #   Occ/Vir MO orbitals and number of bands
                orb_p = mo_coeff[ikp][:,:nocc] if kflag_occ[ikp] else mo_coeff[ikp][:,nocc:]
                norb_p = nocc if kflag_occ[ikp] else nvir
                orb_q = mo_coeff[ikq][:,:nocc] if kflag_occ[ikq] else mo_coeff[ikq][:,nocc:]
                norb_q = nocc if kflag_occ[ikq] else nvir
                orb_r = mo_coeff[ikr][:,:nocc] if kflag_occ[ikr] else mo_coeff[ikr][:,nocc:]
                norb_r = nocc if kflag_occ[ikr] else nvir
                orb_s = mo_coeff[iks][:,:nocc] if kflag_occ[iks] else mo_coeff[iks][:,nocc:]
                norb_s = nocc if kflag_occ[iks] else nvir

                #   Calculate the ERIs
                eri_kpt = fao2mo((orb_p,orb_q,orb_r,orb_s),
                                (kpts[ikp],kpts[ikq],kpts[ikr],kpts[iks]), compact=False)
                if self.dtype == float: eri_kpt = eri_kpt.real
                eri_kpt = eri_kpt.reshape(norb_p, norb_q, norb_r, norb_s)
                return eri_kpt

            eri_kpt = []
            with concurrent.futures.ThreadPoolExecutor(2) as executor:
                for idx in range(len(key_list)):
                    eri_kpt.append( executor.submit(eri_kpt_calc, idx) )

            for idx, (ikp, ikq, ikr) in enumerate(key_list):
                iks = kconserv[ikp,ikq,ikr]
                if sum(kflag_occ[ [ikp, ikq, ikr, iks] ]) % 2:
                    continue
                eri_result = eri_kpt[idx].result()
                for (kp, kq, kr) in khelper.symm_map[(ikp, ikq, ikr)]:
                    eri_kpt_symm = khelper.transform_symm(eri_result, kp, kq, kr).transpose(0, 2, 1, 3)

                    if np.array_equal(kflag_occ[ [kp, kr, kq] ], [1,1,1]):
                        self.oooo[idx_occ[kp], idx_occ[kr], idx_occ[kq]] = eri_kpt_symm / nkpts_ov

                    elif np.array_equal(kflag_occ[ [kp, kr, kq] ], [1,1,0]):
                        self.oovv[idx_occ[kp], idx_occ[kr], idx_vir[kq]] = eri_kpt_symm / nkpts_ov

                    elif np.array_equal(kflag_occ[ [kp, kr, kq] ], [1,0,1]):
                        self.ovov[idx_occ[kp], idx_vir[kr], idx_occ[kq]] = eri_kpt_symm / nkpts_ov

                    elif np.array_equal(kflag_occ[ [kp, kr, kq] ], [0,1,1]):
                        self.voov[idx_vir[kp], idx_occ[kr], idx_occ[kq]] = eri_kpt_symm / nkpts_ov

                    elif np.array_equal(kflag_occ[ [kp, kr, kq] ], [0,0,0]):
                        self.vvvv[idx_vir[kp], idx_vir[kr], idx_vir[kq]] = eri_kpt_symm / nkpts_ov

            log.info("ERI evaluation time: %.3f" % (time.time() - start))
        else:
            raise MemoryError(
                'In-core two-mesh KMP3 integral storage exceeds max_memory; '
                'increase the memory limit or reduce the k-point/orbital space.')

        log.timer('CCD integral transformation', *cput0)
