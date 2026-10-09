#!/usr/bin/env python
#
# Authors: Xin Xing
#


import concurrent.futures
import itertools
import time

import numpy as np
from pyscf import __config__, lib
from pyscf.cc.ccsd import kernel
from pyscf.lib import logger
from pyscf.pbc import df, dft, scf
from pyscf.pbc.cc import kccsd_rhf
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.mp.kmp2 import (get_nocc, padded_mo_coeff, padded_mo_energy,
                               padding_k_idx)
from pyscf.pbc.scf.khf import get_occ
from pyscf.pbc.tools.pbc import get_monkhorst_pack_size

from . import kintermediates_rhf_stagger as imdk
from .._kpoints import make_kpts_helper

einsum = lib.einsum
_get_epq = kccsd_rhf._get_epq
_mem_usage = kccsd_rhf._mem_usage


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

def update_amps_3mesh(cc, t1, t2, eris):
    time0 = time1 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(cc.stdout, cc.verbose)

    #   basic information
    nocc = cc.nocc
    nvir = cc.nmo - cc.nocc
    nkpts = cc.nkpts_base
    kconserv = cc.khelper.kconserv
    mo_e_o = [e[:nocc] for e in eris.mo_energy]
    mo_e_v = [e[nocc:] + cc.level_shift for e in eris.mo_energy]
    namp = len(cc.kpts_ijab)

    #   T1 equation
    t1new = np.zeros_like(t1)

    #   T2 equation
    t2new = np.empty((namp, nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir), dtype=eris.dtype)

    #   1st term: constant <ab|ij>
    for amp in range(namp):
        mi, mj, ma, mb= cc.kpts_ijab[amp]
        for iki, ikj, ika in kpts_helper.loop_kkk(nkpts):
            t2new[amp, iki, ikj, ika] = eris.oovv[cc.mesh2idx(mi,mj,ma)][iki, ikj, ika].conj()

    #   2nd term: 4h2p
    mem_now = lib.current_memory()[0]
    if 18 * (nocc ** 4 * nkpts ** 3) * 16 / 1e6 + mem_now < cc.max_memory * .9:
        Woooo = imdk.cc_Woooo_3mesh(cc, t1, t2, eris, kconserv)
    else:
        raise NotImplementedError(
            'In-core Woooo intermediate exceeds max_memory; out-of-core update is unsupported'
        )

    for amp in range(namp):
        mi, mj, ma, mb= cc.kpts_ijab[amp]
        for iki, ikj, ika in kpts_helper.loop_kkk(nkpts):
            ki, kj, ka = cc.kpts_base_idx[mi][iki], cc.kpts_base_idx[mj][ikj], cc.kpts_base_idx[ma][ika]
            kb = kconserv[ki, ka, kj]
            ikb = cc.idx_base[mb][kb]

            t2new_tmp = np.zeros((nocc, nocc, nvir, nvir), dtype=eris.dtype)
            mk, ml = cc.dependence_amp[cc.mesh2idx(mi,mj,ma)][0]
            for ikl, kl in enumerate(cc.kpts_base_idx[ml]):
                kk = kconserv[kj, kl, ki]
                ikk = cc.idx_base[mk][kk]
                t2new_tmp += einsum('klij,klab->ijab', Woooo[cc.mesh2idx(mk,ml,mi)][ikk,ikl,iki], cc.amp_3mesh(t2, (mk,ml,ma,mb), (ikk,ikl,ika,ikb)))
            t2new[amp, iki, ikj, ika] += t2new_tmp

            #   madelung correction
            if cc.madelung:
                t2new[amp, iki, ikj, ika] += cc.madelung_correction[(mi-mk)%3] * t2[amp, iki, ikj, ika]

    Woooo = None
    time1 = log.timer_debug1('t2 oooo', *time1)


    #   4th term: 4p2h
    mem_now = lib.current_memory()[0]
    if 18 * (nvir ** 4 * nkpts ** 3) * 16 / 1e6 + mem_now < cc.max_memory * .9:
        Wvvvv = imdk.cc_Wvvvv_3mesh(cc, t1, t2, eris, kconserv)
    else:
        raise NotImplementedError(
            'In-core Wvvvv intermediate exceeds max_memory; out-of-core update is unsupported'
        )

    for amp in range(namp):
        mi, mj, ma, mb= cc.kpts_ijab[amp]
        for iki, ikj, ika in kpts_helper.loop_kkk(nkpts):
            ki, kj, ka = cc.kpts_base_idx[mi][iki], cc.kpts_base_idx[mj][ikj], cc.kpts_base_idx[ma][ika]
            kb = kconserv[ki, ka, kj]
            ikb = cc.idx_base[mb][kb]

            t2new_tmp = np.zeros((nocc, nocc, nvir, nvir), dtype=eris.dtype)
            mc, md = cc.dependence_amp[cc.mesh2idx(mi,mj,ma)][1]
            for ikc, kc in enumerate(cc.kpts_base_idx[mc]):
                kd = kconserv[ka, kc, kb]
                ikd = cc.idx_base[md][kd]
                t2new_tmp += einsum('abcd,ijcd->ijab', Wvvvv[cc.mesh2idx(ma,mb,mc)][ika,ikb,ikc], cc.amp_3mesh(t2, (mi,mj,mc,md), (iki,ikj,ikc,ikd)) )
            t2new[amp, iki, ikj, ika] += t2new_tmp

            #   madelung correction
            if cc.madelung:
                t2new[amp, iki, ikj, ika] += cc.madelung_correction[(mc-ma)%3] * t2[amp, iki, ikj, ika]

    Wvvvv = None
    time1 = log.timer_debug1('t2 vvvv', *time1)


    #   6-7th terms
    #   intermediate tensor "kappa" at Eqn 37-38 (= lambda in Eqn 40-41 for CCD calculation)
    Loo = imdk.cc_Foo_3mesh(cc, t1, t2, eris, kconserv)
    Lvv = imdk.cc_Fvv_3mesh(cc, t1, t2, eris, kconserv)
    time1 = log.timer_debug1('intermediates', *time1)
    for amp in range(namp):
        mi, mj, ma, mb= cc.kpts_ijab[amp]
        for iki, ikj, ika in kpts_helper.loop_kkk(nkpts):
            ki, kj, ka = cc.kpts_base_idx[mi][iki], cc.kpts_base_idx[mj][ikj], cc.kpts_base_idx[ma][ika]
            kb = kconserv[ki, ka, kj]
            ikb = cc.idx_base[mb][kb]

            t2new_tmp =  einsum('ac,ijcb->ijab', Lvv[ma][ika], t2[amp, iki, ikj, ika])
            t2new_tmp += einsum('bc,ijac->ijab', Lvv[mb][ikb], t2[amp, iki, ikj, ika])
            t2new_tmp -= einsum('ki,kjab->ijab', Loo[mi][iki], t2[amp, iki, ikj, ika])
            t2new_tmp -= einsum('kj,ikab->ijab', Loo[mj][ikj], t2[amp, iki, ikj, ika])
            t2new[amp, iki, ikj, ika] += t2new_tmp
    Loo = None
    Lvv = None


    #   10-12 terms:
    mem_now = lib.current_memory()[0]
    if namp * (nocc ** 2 * nvir ** 2 * nkpts ** 3) * 16 / 1e6 * 2 + mem_now < cc.max_memory * .9:
        Wvoov = imdk.cc_Wvoov_3mesh(cc, t1, t2, eris, kconserv)
        Wvovo = imdk.cc_Wvovo_3mesh(cc, t1, t2, eris, kconserv)
    else:
        raise NotImplementedError(
            'In-core Wvoov/Wvovo intermediates exceed max_memory; '
            'out-of-core update is unsupported'
        )

    for amp in range(namp):
        mi, mj, ma, mb= cc.kpts_ijab[amp]
        for iki, ikj, ika in kpts_helper.loop_kkk(nkpts):
            ki, kj, ka = cc.kpts_base_idx[mi][iki], cc.kpts_base_idx[mj][ikj], cc.kpts_base_idx[ma][ika]
            kb = kconserv[ki, ka, kj]
            ikb = cc.idx_base[mb][kb]
            t2new_tmp = np.zeros((nocc, nocc, nvir, nvir), dtype=eris.dtype)

            mk, mc = cc.dependence_amp[cc.mesh2idx(mi,mj,ma)][2][:2]
            for ikk, kk in enumerate(cc.kpts_base_idx[mk]):
                #   term 10
                kc = kconserv[ka, ki, kk]
                ikc = cc.idx_base[mc][kc]
                tmp_voov = 2. * Wvoov[cc.mesh2idx(ma,mk,mi)][ika,ikk,iki] - Wvovo[cc.mesh2idx(ma,mk,mc)][ika,ikk,ikc].transpose(0, 1, 3, 2)
                t2new_tmp += einsum('akic,kjcb->ijab', tmp_voov, cc.amp_3mesh(t2, (mk,mj,mc,mb), (ikk,ikj,ikc,ikb)))

                #   term 11
                kc = kconserv[ka, ki, kk]
                ikc = cc.idx_base[mc][kc]
                t2new_tmp -= einsum('akic,kjbc->ijab', Wvoov[cc.mesh2idx(ma,mk,mi)][ika,ikk,iki], cc.amp_3mesh(t2, (mk,mj,mb,mc), (ikk,ikj,ikb,ikc)))

            #   madelung correction
            if cc.madelung:
                t2new_tmp -= cc.madelung_correction[(mc-ma)%3] * t2[amp, iki, ikj, ika]


            mk, mc = cc.dependence_amp[cc.mesh2idx(mi,mj,ma)][3]
            for ikk, kk in enumerate(cc.kpts_base_idx[mk]):
                #   term 12
                kc = kconserv[kk, ka, kj]
                ikc = cc.idx_base[mc][kc]
                t2new_tmp -= einsum('bkci,kjac->ijab', Wvovo[cc.mesh2idx(mb,mk,mc)][ikb,ikk,ikc], cc.amp_3mesh(t2, (mk,mj,ma,mc), (ikk,ikj,ika,ikc)) )

            #   madelung correction
            if cc.madelung:
                t2new_tmp -= cc.madelung_correction[(mc-mb)%3] * t2[amp, iki, ikj, ika]


            #   SYMMETRIZED TERMS with (I, A) <-> (J, B)
            mk, mc = cc.dependence_amp[cc.mesh2idx(mj,mi,mb)][2][:2]
            for ikk, kk in enumerate(cc.kpts_base_idx[mk]):
                #   term 10
                kc = kconserv[kb, kj, kk]
                ikc = cc.idx_base[mc][kc]
                tmp_voov = 2. * Wvoov[cc.mesh2idx(mb,mk,mj)][ikb,ikk,ikj] - Wvovo[cc.mesh2idx(mb,mk,mc)][ikb,ikk,ikc].transpose(0, 1, 3, 2)
                t2new_tmp += einsum('bkjc,kica->ijab', tmp_voov, cc.amp_3mesh(t2, (mk,mi,mc,ma), (ikk,iki,ikc,ika)))

                #   term 11
                kc = kconserv[kb, kj, kk]
                ikc = cc.idx_base[mc][kc]
                t2new_tmp -= einsum('bkjc,kiac->ijab', Wvoov[cc.mesh2idx(mb,mk,mj)][ikb,ikk,ikj], cc.amp_3mesh(t2, (mk,mi,ma,mc), (ikk,iki,ika,ikc)))

            #   madelung correction
            if cc.madelung:
                t2new_tmp -= cc.madelung_correction[(mc-mb)%3] * t2[amp, iki, ikj, ika]

            mk, mc = cc.dependence_amp[cc.mesh2idx(mj,mi,mb)][3]
            for ikk, kk in enumerate(cc.kpts_base_idx[mk]):
                #   term 12
                kc = kconserv[kk, kb, ki]
                ikc = cc.idx_base[mc][kc]
                t2new_tmp -= einsum('akcj,kibc->ijab', Wvovo[cc.mesh2idx(ma,mk,mc)][ika,ikk,ikc], cc.amp_3mesh(t2, (mk,mi,mb,mc), (ikk,iki,ikb,ikc)) )

            #   madelung correction
            if cc.madelung:
                t2new_tmp -= cc.madelung_correction[(mc-ma)%3] * t2[amp, iki, ikj, ika]

            t2new[amp, iki, ikj, ika] += t2new_tmp
    Wvoov = Wvovo = None
    time1 = log.timer_debug1('t2 voov', *time1)


    #   Step 3: energy denominator
    #   get location of padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = padding_k_idx(cc, kind="split")
    for amp in range(namp):
        mi, mj, ma, mb= cc.kpts_ijab[amp]
        for iki, ikj, ika in kpts_helper.loop_kkk(nkpts):
            ki, kj, ka = cc.kpts_base_idx[mi][iki], cc.kpts_base_idx[mj][ikj], cc.kpts_base_idx[ma][ika]
            kb = kconserv[ki, ka, kj]

            eia = _get_epq([0,nocc,ki,mo_e_o,nonzero_opadding],
                        [0,nvir,ka,mo_e_v,nonzero_vpadding],
                        fac=[1.0,-1.0])
            ejb = _get_epq([0,nocc,kj,mo_e_o,nonzero_opadding],
                        [0,nvir,kb,mo_e_v,nonzero_vpadding],
                        fac=[1.0,-1.0])

            eijab = eia[:, None, :, None] + ejb[None, :, None, :]
            t2new[amp, iki, ikj, ika] /= eijab
    time0 = log.timer_debug1('update t2', *time0)

    return t1new, t2new

def energy_3mesh(cc, t1, t2, eris):
    nkpts = cc.nkpts_base
    kconserv = cc.khelper.kconserv
    # amp_ijab = [(0,0,1,2), (1,1,0,2), (2,2,0,1)]
    amp_ijab = [(0,0,1,2), (1,2,0,0), (1,1,0,2), (0,2,1,1), (2,2,0,1), (0,1,2,2)]
    # amp_ijab = [(0,0,1,2), (0,0,2,1), (1,2,0,0), (2,1,0,0),
    #             (1,1,0,2), (1,1,2,0), (0,2,1,1), (2,0,1,1),
    #             (2,2,0,1), (2,2,1,0), (0,1,2,2), (1,0,2,2)]
    # amp_ijab = [(0,0,1,2), (1,2,0,0)]
    # amp_ijab = [(0,0,1,2)]
    e = 0.0 + 1j * 0.0
    for mi, mj, ma, mb in amp_ijab:
        for iki, ikj, ika in kpts_helper.loop_kkk(nkpts):
            ki, kj, ka = cc.kpts_base_idx[mi][iki], cc.kpts_base_idx[mj][ikj], cc.kpts_base_idx[ma][ika]
            kb = kconserv[ki, ka, kj]
            ikb = cc.idx_base[mb][kb]
            t2_tmp = cc.amp_3mesh(t2, (mi,mj,ma,mb), (iki,ikj,ika,ikb))
            e += 2 * einsum('ijab,ijab', t2_tmp, eris.oovv[cc.mesh2idx(mi,mj,ma)][iki, ikj, ika])
            e += - einsum('ijab,ijba', t2_tmp, eris.oovv[cc.mesh2idx(mi,mj,mb)][iki, ikj, ikb])
    e /= nkpts
    e /= len(amp_ijab)
    if abs(e.imag) > 1e-4:
        logger.warn(cc, 'Non-zero imaginary part found in KRCCSD energy %s', e)
    return e.real

def init_amps_3mesh_zero(cc, eris):
    nocc = cc.nocc
    nvir = cc.nmo - nocc
    nkpts_base = cc.nkpts_base
    namp = len(cc.kpts_ijab)

    t1 = np.zeros((namp, nkpts_base, nocc, nvir), dtype=eris.dtype)
    t2 = np.zeros((namp, nkpts_base, nkpts_base, nkpts_base, nocc, nocc, nvir, nvir), dtype=eris.dtype)
    logger.info(cc, 'Init t2 as zeros')
    return 0, t1, t2

class KRCCD_3mesh(kccsd_rhf.RCCSD):
    max_space = getattr(__config__, 'pbc_cc_kccsd_rhf_KRCCSD_max_space', 20)

    def __init__(self, mf, frozen=None, flag_submesh=False, madelung=False):
        if not isinstance(mf, scf.khf.KSCF):
            raise TypeError('KRCCD_3mesh requires a periodic KRHF or KRKS reference')
        # Keep PySCF's periodic restricted-CC attributes initialized.
        mf_hf = mf.to_hf() if isinstance(mf, dft.krks.KRKS) else mf
        kccsd_rhf.RCCSD.__init__(self, mf_hf, frozen)
        self._reference_e_tot = mf.e_tot

        #   Initialization for staggered mesh method
        self.mo_energy = None
        self.mo_occ = None
        self.cell = mf.cell

        #   Construction of orbitals and orbital energies on two staggered meshes
        nks = get_monkhorst_pack_size(mf.cell, mf.kpts)
        self.flag_submesh = flag_submesh
        if flag_submesh is True:
            #   Choice 1: Use two staggered submeshes of mf.kpts of half size along each dimension
            if ((nks % 3 != 0) & (nks > 1)).any() :
                log = logger.Logger(self.stdout, self.verbose)
                log.warn('The k-point mesh in the input mf has meshsize not divided by 3 in certain dimension. \n'
                        'Cannot use its submeshes for staggered mesh calculation!\n'
                        'You may set "flag_submesh" to False')
                raise ValueError('flag_submesh=True requires mesh sizes divisible by three')

            #   Temporary solution for low-dimensional case
            nonperiodic_dir = np.where( nks == 1)
            nks_base = np.array([ n/3 if n > 1 else 1 for n in nks]).astype(int)
            shift = mf.kpts[ np.argmin( np.sum(mf.kpts**2, axis=1) )]   #   shift of the original k-point mesh
            if np.any(shift[nonperiodic_dir]):
                log = logger.Logger(self.stdout, self.verbose)
                log.warn('The reference mesh is shifted in a nonperiodic direction')
                raise ValueError('flag_submesh=True requires an unshifted reference mesh')
            shift13 = mf.cell.get_abs_kpts( [1.0/n if n > 1 else 0 for n in nks] )
            shift23 = mf.cell.get_abs_kpts( [2.0/n if n > 1 else 0 for n in nks] )

            kpts0 = shift + mf.cell.make_kpts(nks_base, with_gamma_point=True)
            kpts1 = kpts0 + shift13
            kpts2 = kpts0 + shift23

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
                raise NotImplementedError(
                    'Band-extension orbitals are currently supported only for 3D cells'
                )

            nks_base = nks
            shift13 = mf.cell.get_abs_kpts( [1.0/3.0/n for n in nks] )
            shift23 = mf.cell.get_abs_kpts( [2.0/3.0/n for n in nks] )
            kpts0 = mf.kpts
            kpts1 = kpts0 + shift13
            kpts2 = kpts0 + shift23
            kpts = np.concatenate( (kpts0, kpts1, kpts2), axis=0)
            # KRKS subclasses KRHF, so check it first.  These public names are
            # factories in current PySCF; use the implementation classes.
            if isinstance(mf, dft.krks.KRKS):
                with lib.temporary_env(mf, with_df=df.FFTDF(mf.cell, mf.kpts)):
                    mo_energy, mo_coeff = mf.get_bands(kpts)
                    mo_occ = get_occ(mf, mo_energy_kpts=mo_energy)
            elif isinstance(mf, scf.khf.KRHF):
                with lib.temporary_env(mf, exxdiv='vcut_sph', with_df=df.FFTDF(mf.cell, mf.kpts)):
                    mo_energy, mo_coeff = mf.get_bands( kpts )
                    mo_occ = get_occ(mf, mo_energy_kpts=mo_energy)
            else:
                raise TypeError('KRCCD_3mesh requires a periodic KRHF or KRKS reference')

            self.kpts = kpts
            self.mo_energy = mo_energy
            self.mo_coeff = mo_coeff
            self.mo_occ = mo_occ
            self.with_df = df.GDF(self.cell, kpts).build()

        #   Basic info
        self.frozen = frozen
        self.khelper = make_kpts_helper(self.cell, self.kpts)

        #   Mesh info for the staggered mesh method
        self.kpts_base = [kpts0, kpts1, kpts2]
        self.nkpts_base = len(kpts0)

        #   1. Indices of submesh k points in the global k point list
        #      NOTE: the x-th kpt in the i-th submesh has its index in self.kpts
        #      to be kpts_base_idx[i][x]
        kpts_bz = minimum_image(self.cell, self.kpts)
        self.kpts_base_idx = []
        for i in range(3):
            kpts_i_bz = minimum_image(self.cell, self.kpts_base[i])
            tmp = [ np.asarray(np.sum((kpts_bz - k0)**2, axis=-1) < 1e-10).nonzero()[0] for k0 in kpts_i_bz]
            self.kpts_base_idx.append( np.concatenate(tmp).astype(int) )
            if (
                len(self.kpts_base_idx[-1]) != self.nkpts_base
                or len(np.unique(self.kpts_base_idx[-1])) != self.nkpts_base
            ):
                raise ValueError('Could not map a staggered submesh onto the k-point list')

        all_indices = np.concatenate(self.kpts_base_idx)
        if len(np.unique(all_indices)) != 3 * self.nkpts_base:
            raise ValueError('The three staggered submeshes overlap')

        #   2. Indexing info of the global k-point list in each submesh
        #      NOTE: kpts[p] is the 'self.idx_base[i][p]'-th k point in the ith submesh
        self.idx_base = []
        for i in range(3):
            tmp = -1 * np.ones(self.nkpts, dtype=int)
            tmp[self.kpts_base_idx[i]] = np.arange(self.nkpts_base)
            self.idx_base.append( tmp[:] )

        #   3. Index of the submesh where each k-point belongs to
        #      NOTE: kpts[p] is in the 'self.flag_base[p]'-th submesh
        self.flag_base = -1*np.ones(self.nkpts, dtype = int)
        for i in range(3):
            self.flag_base[ self.kpts_base_idx[i] ] = i

        #   Amplitude configuration of k-point sampling of (i,j,a,b) with i != a on the three meshes
        #   1. symmetries t_{ij}^{ab} = t_{ji}^{ba} is used such that i <= j and a <=b when i == j
        #   2. momentum conservation: b = i + j - a (mod 3)
        self.kpts_ijab = [(0,0,1,2), (0,1,1,0), (0,1,2,2), (0,2,1,1), (0,2,2,0),
                            (1,1,0,2), (1,2,0,0), (1,2,2,1), (2,2,0,1)]

        # self.dependence_kappa = [
        #                          [[(0, 1, 2)], [(1, 2, 0)]],
        #                          [[(1, 2, 0)], [(2, 0, 1)]],
        #                          [[(2, 0, 1)], [(0, 1, 2)]]
        #                         ]
        self.dependence_kappa = {}
        for mp in range(3):
            self.dependence_kappa[mp] = ccd_mesh_dependence_kappa(mp)

        self.dependence_imb = {}
        self.dependence_amp = {}
        for mp, mq, mr in itertools.product(range(3), range(3), range(3)):
            ms = (mp + mq - mr) % 3
            if mp == mr:
                continue
            self.dependence_imb[self.mesh2idx(mp, mq, mr)] = ccd_mesh_dependence_imb(mp, mq, mr, ms)
            self.dependence_amp[self.mesh2idx(mp, mq, mr)] = ccd_mesh_dependence_amp(mp, mq, mr, ms)


        #   Madelung correction
        self.madelung = madelung
        if self.madelung:
            self.madelung_correction = [
            madelung_stagger(self.cell, nks_base, np.zeros((1,3))),
            madelung_stagger(self.cell, nks_base, shift13),
            madelung_stagger(self.cell, nks_base, shift23),
            ]
        else:
            self.madelung_correction = []

    def kernel(self, t1=None, t2=None, eris=None, mbpt2=False):
        return self.ccd_3mesh(t1, t2, eris, mbpt2=mbpt2)

    def ccd_3mesh(self, t1=None, t2=None, eris=None, mbpt2=False):
        '''Ground-state restricted CCD.

        Kwargs:
            mbpt2 : bool
                Use one-shot MBPT2 approximation to CCSD.
        '''
        self.dump_flags()
        self.e_hf = (self._reference_e_tot if self._reference_e_tot is not None
                     else self.get_e_hf())

        #   ERIs
        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        self.eris = eris
        #   MP2
        if mbpt2:
            self.converged, self.e_corr, self.t1, self.t2 = \
                kernel(self, eris, t1, t2, max_cycle=1,
                       tol=self.conv_tol, tolnormt=self.conv_tol_normt,
                       verbose=self.verbose)
            return self.e_corr, self.t1, self.t2

        self.converged, self.e_corr, self.t1, self.t2 = \
            kernel(self, eris, t1, t2, max_cycle=self.max_cycle,
                   tol=self.conv_tol, tolnormt=self.conv_tol_normt,
                   verbose=self.verbose)
        self._finalize()
        return self.e_corr, self.t1, self.t2

    @property
    def ccsd_vector_desc(self):
        """Description of the ground-state vector."""
        nvir = self.nmo - self.nocc
        namp = len(self.kpts_ijab)
        return [(namp, self.nkpts_base, self.nocc, nvir), (namp,) + (self.nkpts_base,) * 3 + (self.nocc,) * 2 + (nvir,) * 2]

    def ao2mo(self, mo_coeff=None):
        return _ERIS_3stagger(self, mo_coeff)

    def mesh2idx(self, mp, mq, mr):
        return (mp%3) + 3 * (mq%3) + 9* (mr%3)

    def idx2mesh(self, idx):
        mp = idx % 3
        mq = (idx // 3) % 3
        mr = (idx // 9) % 3
        return mp, mq, mr

    def amp_3mesh(self, t2, mesh_config, kpt_idx):
        '''
        Return the amplitudes
            t_{ij}^{ab}
        with all band indices where
            i is the kpt_idx[0]-th k-point on the mesh_config[0]-th submesh
            j is the kpt_idx[1]-th k-point on the mesh_config[1]-th submesh
            a is the kpt_idx[2]-th k-point on the mesh_config[2]-th submesh
        '''
        mi, mj, ma, mb = mesh_config[0], mesh_config[1], mesh_config[2], mesh_config[3]
        iki, ikj, ika, ikb = kpt_idx[0], kpt_idx[1], kpt_idx[2], kpt_idx[3]
        kpts_ijab = self.kpts_ijab

        if (mi,mj,ma,mb) in kpts_ijab:
            amp = kpts_ijab.index( (mi,mj,ma,mb) )
            return t2[amp, iki, ikj, ika]
        else:
            if (mj,mi,mb,ma) not in kpts_ijab:
                raise ValueError(
                    f'No stored three-mesh amplitude for {(mi, mj, ma, mb)} '
                    'or its occupied/virtual exchange'
                )
            amp = kpts_ijab.index( (mj,mi,mb,ma) )
            return t2[amp, ikj, iki, ikb].transpose(1,0,3,2)

    init_amps = init_amps_3mesh_zero
    energy = energy_3mesh
    update_amps = update_amps_3mesh

    #   TODO: the following functions are not supported yet.
    ccsd_t = None
    ipccsd = None
    eaccsd = None
    ipccsd_matvec = None
    ipccsd_diag = None
    eaccsd_matvec = None
    eaccsd_diag = None

class _ERIS_3stagger:  # (pyscf.cc.ccsd._ChemistsERIs):
    #   construct ERIs <pq|rs> with all possible sampling of pqrs on the submeshes specified
    #   by "self.kpts_ijab"
    def __init__(self, cc, mo_coeff=None, mo_energy=None, method='incore'):
        log = logger.Logger(cc.stdout, cc.verbose)
        cput0 = (logger.process_clock(), logger.perf_counter())

        #   Basic information
        cell = cc._scf.cell
        kpts = cc.kpts
        nkpts = cc.nkpts
        nkpts_base = cc.nkpts_base
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

        mem_incore, mem_outcore, mem_basic = _mem_usage(nkpts_base, nocc, nvir)
        mem_now = lib.current_memory()[0]

        fao2mo = cc.with_df.ao2mo
        kconserv = cc.khelper.kconserv
        khelper = cc.khelper

        idx_base = cc.idx_base
        flag_base = cc.flag_base

        if (method == 'incore' and (18 * mem_incore + mem_now < cc.max_memory)
                or cell.incore_anyway):
            log.info('using incore ERI storage')

            #   Initialization
            self.oovv = {}
            self.voov = {}
            self.oooo = {}
            self.ovov = {}
            self.vvvv = {}
            for mp, mq, mr in itertools.product(range(3), range(3), range(3)):
                if mp == mr:
                    continue
                idx_pqr = cc.mesh2idx(mp, mq, mr)
                self.oovv[idx_pqr] = np.empty((nkpts_base,nkpts_base,nkpts_base,nocc,nocc,nvir,nvir), dtype=self.dtype)
                self.voov[idx_pqr] = np.empty((nkpts_base,nkpts_base,nkpts_base,nvir,nocc,nocc,nvir), dtype=self.dtype)
                self.oooo[idx_pqr] = np.empty((nkpts_base,nkpts_base,nkpts_base,nocc,nocc,nocc,nocc), dtype=self.dtype)
                self.ovov[idx_pqr] = np.empty((nkpts_base,nkpts_base,nkpts_base,nocc,nvir,nocc,nvir), dtype=self.dtype)
                self.vvvv[idx_pqr] = np.empty((nkpts_base,nkpts_base,nkpts_base,nvir,nvir,nvir,nvir), dtype=self.dtype)

            log.info("Multithreading ERI tensor calculation")
            start = time.time()
            key_list = list(khelper.symm_map)

            def admissible_kpts(kp, kq, kr, ks):
                if flag_base[kp] < 0 or flag_base[kq] < 0 or flag_base[kr] < 0 or flag_base[ks] < 0:
                    return False

                if flag_base[kp] == flag_base[kr]:
                    return False

                return True

            def eri_kpt_calc(idx):
                #   <kp, kq | kr, ks>
                (kp,kr,kq) = key_list[idx]
                ks = kconserv[kp,kr,kq]

                if not admissible_kpts(kp, kq, kr, ks):
                    return None

                #   Calculate the ERIs
                eri_kpt = fao2mo((mo_coeff[kp],mo_coeff[kr],mo_coeff[kq],mo_coeff[ks]),
                                (kpts[kp],kpts[kr],kpts[kq],kpts[ks]), compact=False)

                if self.dtype == np.dtype(float): eri_kpt = eri_kpt.real
                eri_kpt = eri_kpt.reshape(nmo, nmo, nmo, nmo)
                return eri_kpt

            eri_kpt = []
            with concurrent.futures.ThreadPoolExecutor(2) as executor:
                for idx in range(len(key_list)):
                    eri_kpt.append( executor.submit(eri_kpt_calc, idx) )

            for idx, (kpp, krr, kqq) in enumerate(key_list):
                kss = kconserv[kpp,krr,kqq]
                if not admissible_kpts(kpp, kqq, krr, kss):
                    continue
                eri_result = eri_kpt[idx].result()
                for (kp, kr, kq) in khelper.symm_map[(kpp, krr, kqq)]:
                    eri_kpt_symm = khelper.transform_symm(eri_result, kp, kr, kq).transpose(0, 2, 1, 3)
                    ks = kconserv[kp, kr, kq]
                    mp, mq, mr, ms = flag_base[kp], flag_base[kq], flag_base[kr], flag_base[ks]
                    ikp, ikq, ikr, iks = idx_base[mp][kp], idx_base[mq][kq], idx_base[mr][kr], idx_base[ms][ks]
                    idx_pqr = cc.mesh2idx(mp, mq, mr)

                    if mp == mr:
                        raise ValueError
                    self.oooo[idx_pqr][ikp, ikq, ikr] = eri_kpt_symm[:nocc, :nocc, :nocc, :nocc] / nkpts_base
                    self.oovv[idx_pqr][ikp, ikq, ikr] = eri_kpt_symm[:nocc, :nocc, nocc:, nocc:] / nkpts_base
                    self.ovov[idx_pqr][ikp, ikq, ikr] = eri_kpt_symm[:nocc, nocc:, :nocc, nocc:] / nkpts_base
                    self.voov[idx_pqr][ikp, ikq, ikr] = eri_kpt_symm[nocc:, :nocc, :nocc, nocc:] / nkpts_base
                    self.vvvv[idx_pqr][ikp, ikq, ikr] = eri_kpt_symm[nocc:, nocc:, nocc:, nocc:] / nkpts_base
            log.info("ERI evaluation time: %.3f" % (time.time() - start))
        else:
            raise NotImplementedError(
                'Out-of-core ERI transformation is not supported for three-mesh CCD'
            )

def ccd_mesh_dependence_amp(mi, mj, ma, mb):
    r'''
    Construct the sampling mesh configuration for intermediate momentum vectors in
    the amplitude equation for t_{ij}^{ab} with (i,j,a,b) sampled on (mi,mj,ma,mb) meshes

    Output:
    [(k,l), (c,d), (k,c), (k,c)]
    '''
    mi, mj, ma, mb = mi%3, mj%3, ma%3, mb%3
    if mi == ma or (mi+mj)%3 != (ma+mb)%3:
        raise ValueError

    #   case 1: t_{xx}^{yz}
    if mi == mj:
        x, y, z = mi, ma, mb
        ans = []
        #   1st term sum_{KL} chi_{IJ}^{KL}t_{KL}^{AB}
        ans.append((z, y))
        #   2nd term sum_{CD} <AB|CD> t_{IJ}^{CD}
        ans.append((z, y))
        #   3-5th terms sum_{KC} X_{IC}^{AK}( t_{KJ}^{CB} + t_{KJ}^{BC}) + X_{CI}^{AK}t_{KJ}^{CB}
        ans.append((y, z))
        #   6th term: sum_{KC} X_{CI}^{BK}t_{KJ}^{AC}
        ans.append((z, y))
        return ans

    #   case 2: t_{xy}^{yx}
    if mi != mj and mi == mb and mj == ma:
        x, y = mi, mj
        z = (x+y) % 3
        z = z//2 if z%2 == 0 else 2
        ans = []
        #   1st term sum_{KL} chi_{IJ}^{KL}t_{KL}^{AB}
        ans.append((z, z))
        #   2nd term sum_{CD} <AB|CD> t_{IJ}^{CD}
        ans.append((z, z))
        #   3-5th terms sum_{KC} X_{IC}^{AK}( t_{KJ}^{CB} + t_{KJ}^{BC}) + X_{CI}^{AK}t_{KJ}^{CB}
        ans.append((y, z, z, x))
        #   6th term: sum_{KC} X_{CI}^{BK}t_{KJ}^{AC}
        ans.append((z, z))
        return ans

    #   case 3: t_{xy}^{zz}
    if mi != mj and ma == mb:
        x, y, z = ma, mi, mj
        ans = []
        #   1st term sum_{KL} chi_{IJ}^{KL}t_{KL}^{AB}
        ans.append((z, y))
        #   2nd term sum_{CD} <AB|CD> t_{IJ}^{CD}
        ans.append((z, y))
        #   3-5th terms sum_{KC} X_{IC}^{AK}( t_{KJ}^{CB} + t_{KJ}^{BC}) + X_{CI}^{AK}t_{KJ}^{CB}
        ans.append((z, y))
        #   6th term: sum_{KC} X_{CI}^{BK}t_{KJ}^{AC}
        ans.append((z, y))
        return ans

def ccd_mesh_dependence_imb(mp,mq,mr,ms):
    r'''
    Construct the sampling mesh configuration for dummy momentum vectors in the
    intermediate block \chi_{rs}^{pq} calculation where r,s,p,q are sampled on
    mr, ms, mp, mq-th submeshes

    Output:
    [(c,d), None, (l,d), (l,d)]
    '''
    mp, mq, mr, ms = mp%3, mq%3, mr%3, ms%3
    if (mp+mq)%3 != (mr+ms)%3 or mp == mr:
        raise ValueError

    #   case 1: X_{xx}^{yz}
    if mr == ms:
        x, y, z = mr, mp, mq
        ans = []
        #   C, D for X_{IJ}^{KL}
        ans.append((z, y))
        #   NA for X_{AB}^{CD}
        ans.append(())
        #   L, D for X_{IC}^{AK}
        ans.append((z, y))
        #   L, D for X_{CI}^{AK}
        ans.append((z, y))
        return ans

    #   case 2: X_{xy}^{yx}
    if mr != ms and mr == mq and mp == ms:
        x, y = mr, ms
        z = (x+y) % 3
        z = z//2 if z%2 == 0 else 2
        ans = []
         #   C, D for X_{IJ}^{KL}
        ans.append((z, z))
        #   NA for X_{AB}^{CD}
        ans.append(())
        #   L, D for X_{IC}^{AK}
        ans.append(())
        #   L, D for X_{CI}^{AK}
        ans.append((z, z))
        return ans

    #   case 3: X_{yz}^{xx}
    if mr != ms and mp == mq:
        x, y, z = mp, mr, ms
        ans = []
          #   C, D for X_{IJ}^{KL}
        ans.append((z, y))
        #   NA for X_{AB}^{CD}
        ans.append(())
        #   L, D for X_{IC}^{AK}
        ans.append((y, z))
        #   L, D for X_{CI}^{AK}
        ans.append((z, y))
        return ans

def ccd_mesh_dependence_kappa(mp):
    '''
    For i, a on mesh mp, return the mesh configuration
    for (L, C, D) and (K, L, D) for kappa_I^K and kappa_C^A

    ans[0]: a list of all possible mesh sampling for (L, C, D) for kappa_I^K
    ans[1]: a list of all possible mesh sampling for (K, L, D) for kappa_C^A
    '''
    mp1 = (mp+1)%3
    mp2 = (mp+2)%3

    ans = []
    ans.append(
        [ (mp, mp1, mp2), (mp, mp2, mp1), (mp2, mp1, mp1), (mp1, mp2, mp2) ]
        # [ (mp1, mp2, mp2), (mp2, mp1, mp1)]
        # [(mp1, mp2, mp2)]
    )

    ans.append(
        [ (mp1, mp2, mp), (mp2, mp1, mp), (mp1, mp1, mp2), (mp2, mp2, mp1) ]
        # [ (mp1, mp1, mp2), (mp2, mp2, mp1)]
        # [(mp2, mp2, mp1)]
    )

    return ans

def madelung_stagger(cell, nks, mesh_shift, ew_eta=None, ew_cut=None):
    """Return the missing-Gaussian correction for a shifted reciprocal mesh.

    The staggered-mesh Madelung quadrature uses an explicit FFT grid to
    evaluate the reciprocal-space sum.
    """
    nks = np.asarray(nks, dtype=int)
    mesh = np.asarray(cell.mesh, dtype=int) * nks
    mesh_max = np.asarray(
        np.linalg.norm(cell.lattice_vectors() * nks[:, None], axis=1) * 2,
        dtype=int,
    )
    mesh_max = np.maximum(mesh_max, 80)
    mesh = np.minimum(mesh, mesh_max)

    if ew_eta is None or ew_cut is None:
        reciprocal = cell.reciprocal_vectors() / nks[:, None]
        gmax = np.min(mesh // 2 * np.linalg.norm(reciprocal, axis=1))
        log_precision = np.log(cell.precision / (4 * np.pi * (gmax + 1e-100) ** 2))
        ew_eta = np.sqrt(-gmax**2 / (4 * log_precision)) + 1e-100

    grid = [np.fft.fftfreq(n, 1.0 / n) for n in mesh]
    gv = np.dot(lib.cartesian_prod(grid), cell.reciprocal_vectors() / nks[:, None])
    gv += np.asarray(mesh_shift).reshape(-1, 3)[:1]
    g2 = np.einsum('gi,gi->g', gv, gv)
    g2[g2 < 1e-12] = 1e200
    integrand = 4 * np.pi / cell.vol / g2 * np.exp(-g2 / (4 * ew_eta**2))
    quadrature = np.sum(integrand) / np.prod(nks)
    return 2 * ew_eta / np.sqrt(np.pi) - quadrature
