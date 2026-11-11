#!/usr/bin/env python

#
# Experimental code for staggered mesh method for restricted CCD calculation using
# two staggered meshes
#
# Ref: Hirata, et al., J. Chem. Phys. 120, 2581 (2004)
#
# Authors: Xin Xing
#


import concurrent.futures
import time

import numpy as np
from pyscf import __config__, lib
from pyscf.cc.ccsd import kernel
from pyscf.lib import logger
from pyscf.pbc import df, dft, scf, tools
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

def update_amps_stagger(cc, t1, t2, eris):
    time0 = time1 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(cc.stdout, cc.verbose)

    #   basic information
    nkpts, nocc, nvir = t1.shape    #NOTE: nkpts = cc.nkpts_ov
    kconserv = cc.khelper.kconserv
    mo_e_o = [e[:nocc] for e in eris.mo_energy]
    mo_e_v = [e[nocc:] + cc.level_shift for e in eris.mo_energy]
    madelung = tools.madelung(cc.cell, cc.kpts_vir)



    #   T1 equation
    t1new = np.zeros_like(t1)

    #   T2 equation
    t2new = np.empty((nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir), dtype=eris.dtype)
    # t2new = np.zeros((nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir), dtype=eris.dtype)

    #   1st term: constant <ab|ij>
    for iki, ikj, ika in kpts_helper.loop_kkk(nkpts):
        t2new[iki, ikj, ika] = eris.oovv[iki, ikj, ika].conj()

    #   2-3 terms: 4h2p
    mem_now = lib.current_memory()[0]
    if (nocc ** 4 * nkpts ** 3) * 16 / 1e6 + mem_now < cc.max_memory * .9:
        Woooo = imdk.cc_Woooo_stagger(cc, t1, t2, eris, kconserv)
    else:
        fimd = lib.H5TmpFile()
        Woooo = fimd.create_dataset('oooo', (nkpts, nkpts, nkpts, nocc, nocc, nocc, nocc), eris.dtype.char)
        Woooo = imdk.cc_Woooo_stagger(cc, t1, t2, eris, kconserv, Woooo)

    if cc.madelung:
        for iki in range(nkpts):
            for ikj in range(nkpts):
                for i in range(nocc):
                    for j in range(nocc):
                        Woooo[iki, ikj, iki, i, j, i, j] += madelung

    for iki, ikj, ika in kpts_helper.loop_kkk(nkpts):
        ki, kj, ka = cc.kpts_idx_occ[iki], cc.kpts_idx_occ[ikj], cc.kpts_idx_vir[ika]
        kb = kconserv[ki, ka, kj]
        ikb = cc.idx_vir[kb]

        t2new_tmp = np.zeros((nocc, nocc, nvir, nvir), dtype=eris.dtype)
        for ikl, kl in enumerate(cc.kpts_idx_occ):
            kk = kconserv[kj, kl, ki]
            ikk = cc.idx_occ[kk]
            t2new_tmp += 0.5 * einsum('klij,klab->ijab', Woooo[ikk, ikl, iki], t2[ikk, ikl, ika])
        t2new[iki, ikj, ika] += t2new_tmp
        t2new[ikj, iki, ikb] += t2new_tmp.transpose(1, 0, 3, 2)

    Woooo = None
    fimd = None
    time1 = log.timer_debug1('t2 oooo', *time1)

    #   4-5 terms: 4p2h (converged)
    add_vvvv_stagger(cc, t2new, t1, t2, eris)
    time1 = log.timer_debug1('t2 vvvv', *time1)

    #   6-9 terms: involving t_{ij}^{cb}, t_{kj}^{ab} (converged)
    #   intermediate tensor "kappa" at Eqn 37-38 (= lambda in Eqn 40-41 for CCD calculation)
    Loo = imdk.cc_Foo_stagger(cc, t1, t2, eris, kconserv)
    Lvv = imdk.cc_Fvv_stagger(cc, t1, t2, eris, kconserv)
    time1 = log.timer_debug1('intermediates', *time1)

    for iki, ikj, ika in kpts_helper.loop_kkk(nkpts):
        ki, kj, ka = cc.kpts_idx_occ[iki], cc.kpts_idx_occ[ikj], cc.kpts_idx_vir[ika]
        kb = kconserv[ki, ka, kj]
        ikb = cc.idx_vir[kb]

        t2new_tmp = einsum('ac,ijcb->ijab', Lvv[ika], t2[iki, ikj, ika])
        t2new_tmp += einsum('ki,kjab->ijab', -Loo[iki], t2[iki, ikj, ika])

        t2new[iki, ikj, ika] += t2new_tmp
        t2new[ikj, iki, ikb] += t2new_tmp.transpose(1, 0, 3, 2)


    #   10-12 terms: 3h3p term involving t_{kj}^{cb}, t_{kj}^{bc}
    mem_now = lib.current_memory()[0]
    if (nocc ** 2 * nvir ** 2 * nkpts ** 3) * 16 / 1e6 * 2 + mem_now < cc.max_memory * .9:
        Wvoov = imdk.cc_Wvoov_stagger(cc, t1, t2, eris, kconserv)
        Wvovo = imdk.cc_Wvovo_stagger(cc, t1, t2, eris, kconserv)
    else:
        fimd = lib.H5TmpFile()
        Wvoov = fimd.create_dataset('voov', (nkpts, nkpts, nkpts, nvir, nocc, nocc, nvir), eris.dtype.char)
        Wvovo = fimd.create_dataset('vovo', (nkpts, nkpts, nkpts, nvir, nocc, nvir, nocc), eris.dtype.char)
        Wvoov = imdk.cc_Wvoov_stagger(cc, t1, t2, eris, kconserv, Wvoov)
        Wvovo = imdk.cc_Wvovo_stagger(cc, t1, t2, eris, kconserv, Wvovo)

    if cc.madelung:
        for ika in range(nkpts):
            for iki in range(nkpts):
                for a in range(nvir):
                    for i in range(nocc):
                        Wvovo[ika, iki, ika, a, i, a, i] += madelung

    # Wvovo = np.zeros( (nkpts, nkpts, nkpts, nvir, nocc, nvir, nocc) )
    # Wvoov = np.zeros( (nkpts, nkpts, nkpts, nvir, nocc, nocc, nvir) )

    for iki, ikj, ika in kpts_helper.loop_kkk(nkpts):
        ki, kj, ka = cc.kpts_idx_occ[iki], cc.kpts_idx_occ[ikj], cc.kpts_idx_vir[ika]
        kb = kconserv[ki, ka, kj]
        ikb = cc.idx_vir[kb]

        t2new_tmp = np.zeros((nocc, nocc, nvir, nvir), dtype=eris.dtype)
        for ikk, kk in enumerate(cc.kpts_idx_occ):
            #   term 10
            kc = kconserv[ka, ki, kk]
            ikc = cc.idx_vir[kc]
            tmp_voov = 2. * Wvoov[ika, ikk, iki] - Wvovo[ika, ikk, ikc].transpose(0, 1, 3, 2)
            t2new_tmp += einsum('akic,kjcb->ijab', tmp_voov, t2[ikk, ikj, ikc])

            #   term 11
            kc = kconserv[ka, ki, kk]
            ikc = cc.idx_vir[kc]
            t2new_tmp -= einsum('akic,kjbc->ijab', Wvoov[ika, ikk, iki], t2[ikk, ikj, ikb])

            #   term 12
            kc = kconserv[kk, ka, kj]
            ikc = cc.idx_vir[kc]
            t2new_tmp -= einsum('bkci,kjac->ijab', Wvovo[ikb, ikk, ikc], t2[ikk, ikj, ika])

        t2new[iki, ikj, ika] += t2new_tmp
        t2new[ikj, iki, ikb] += t2new_tmp.transpose(1, 0, 3, 2)
    Wvoov = Wvovo = None
    fimd = None
    time1 = log.timer_debug1('t2 voov', *time1)


    #   Step 3: energy denominator
    #   get location of padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = padding_k_idx(cc, kind="split")
    for iki, ikj, ika in kpts_helper.loop_kkk(nkpts):
        ki, kj, ka = cc.kpts_idx_occ[iki], cc.kpts_idx_occ[ikj], cc.kpts_idx_vir[ika]
        kb = kconserv[ki, ka, kj]
        eia = _get_epq([0,nocc,ki,mo_e_o,nonzero_opadding],
                       [0,nvir,ka,mo_e_v,nonzero_vpadding],
                       fac=[1.0,-1.0])

        ejb = _get_epq([0,nocc,kj,mo_e_o,nonzero_opadding],
                       [0,nvir,kb,mo_e_v,nonzero_vpadding],
                       fac=[1.0,-1.0])
        eijab = eia[:, None, :, None] + ejb[None, :, None, :]
        t2new[iki, ikj, ika] /= eijab

    time0 = log.timer_debug1('update t2', *time0)
    return t1new, t2new

def energy_stagger(cc, t1, t2, eris):
    nkpts, nocc, nvir = t1.shape    #   nkpts = cc.nkpts_ov
    kconserv = cc.khelper.kconserv
    e = 0.0 + 1j * 0.0
    for iki, ikj, ika in kpts_helper.loop_kkk(nkpts):
        ki, kj, ka = cc.kpts_idx_occ[iki], cc.kpts_idx_occ[ikj], cc.kpts_idx_vir[ika]
        kb = kconserv[ki, ka, kj]
        ikb = cc.idx_vir[kb]
        e += 2 * einsum('ijab,ijab', t2[iki, ikj, ika], eris.oovv[iki, ikj, ika])
        e += - einsum('ijab,ijba', t2[iki, ikj, ika], eris.oovv[iki, ikj, ikb])
    e /= nkpts
    if abs(e.imag) > 1e-4:
        logger.warn(cc, 'Non-zero imaginary part found in KRCCSD energy %s', e)
    return e.real

def init_amps_stagger(cc, eris):
    time0 = logger.process_clock(), logger.perf_counter()
    nocc = cc.nocc
    nvir = cc.nmo - nocc
    nkpts_ov = cc.nkpts_ov
    nkpts = cc.nkpts

    t1 = np.zeros((nkpts_ov,nocc,nvir), dtype=eris.dtype)
    t2 = np.empty((nkpts_ov,nkpts_ov,nkpts_ov,nocc,nocc,nvir,nvir), dtype=eris.dtype)
    mo_e_o = [eris.mo_energy[k][:nocc] for k in range(nkpts)]
    mo_e_v = [eris.mo_energy[k][nocc:] for k in range(nkpts)]

    #   Get location of padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = padding_k_idx(cc, kind="split")

    #   Initial guess gives the MP2 energy
    emp2 = 0
    kconserv = cc.khelper.kconserv
    touched = np.zeros((nkpts_ov, nkpts_ov, nkpts_ov), dtype=bool)
    for iki, ikj, ika in kpts_helper.loop_kkk(nkpts_ov):
        ki, kj, ka = cc.kpts_idx_occ[iki], cc.kpts_idx_occ[ikj], cc.kpts_idx_vir[ika]
        kb = kconserv[ki, ka, kj]
        ikb = cc.idx_vir[kb]
        if touched[iki, ikj, ika]:
            continue

        eia = _get_epq([0,nocc,ki,mo_e_o,nonzero_opadding],
                    [0,nvir,ka,mo_e_v,nonzero_vpadding],
                    fac=[1.0,-1.0])

        ejb = _get_epq([0,nocc,kj,mo_e_o,nonzero_opadding],
                    [0,nvir,kb,mo_e_v,nonzero_vpadding],
                    fac=[1.0,-1.0])
        eijab = eia[:, None, :, None] + ejb[None, :, None, :]

        eris_ijab = eris.oovv[iki, ikj, ika]
        eris_ijba = eris.oovv[iki, ikj, ikb]
        t2[iki, ikj, ika] = eris_ijab.conj() / eijab
        emp2 += 2 * np.einsum('ijab,ijab', t2[iki, ikj, ika], eris_ijab)
        emp2 += -np.einsum('ijab,ijba', t2[iki, ikj, ika], eris_ijba)

        if ka != kb:
            eijba = eijab.transpose(0, 1, 3, 2)
            t2[iki, ikj, ikb] = eris_ijba.conj() / eijba
            emp2 += 2 * np.einsum('ijab,ijab', t2[iki, ikj, ikb], eris_ijba)
            emp2 += -np.einsum('ijab,ijba', t2[iki, ikj, ikb], eris_ijab)

        touched[iki, ikj, ika] = touched[iki, ikj, ikb] = True

    cc.emp2 = emp2.real / nkpts_ov
    logger.info(cc, 'Init t2, MP2 energy  = %.15g', cc.emp2)
    logger.timer(cc, 'init mp2', *time0)
    return cc.emp2, t1, t2

def init_amps_stagger_zero(cc, eris):
    nocc = cc.nocc
    nvir = cc.nmo - nocc
    nkpts_ov = cc.nkpts_ov

    t1 = np.zeros((nkpts_ov,nocc,nvir), dtype=eris.dtype)
    t2 = np.zeros((nkpts_ov,nkpts_ov,nkpts_ov,nocc,nocc,nvir,nvir), dtype=eris.dtype)
    cc.emp2 = 0.0
    logger.info(cc, 'Init t2, MP2 energy  = %.15g', cc.emp2)
    return cc.emp2, t1, t2

class KRCCD_2mesh(kccsd_rhf.RCCSD):
    max_space = getattr(__config__, 'pbc_cc_kccsd_rhf_KRCCSD_max_space', 20)

    def __init__(self, mf, frozen=None, flag_submesh=False, nks_ccd = None, madelung=False):
        if not isinstance(mf, scf.khf.KSCF):
            raise TypeError('KRCCD_2mesh requires a periodic KRHF or KRKS reference')
        # Keep PySCF's periodic restricted-CC attributes initialized.
        mf_hf = mf.to_hf() if isinstance(mf, dft.krks.KRKS) else mf
        kccsd_rhf.RCCSD.__init__(self, mf_hf, frozen)
        self._reference_e_tot = mf.e_tot
        self.direct = True  # If possible, use GDF to compute Wvvvv on-the-fly
        self.keep_exxdiv = False

        #   Initialization for staggered mesh method
        self.mo_energy = None
        self.mo_occ = None
        self.cell = mf.cell
        self.madelung = madelung

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
                raise ValueError('flag_submesh=True requires even mesh sizes')

            #   Temporary solution for low-dimensional case
            nonperiodic_dir = np.where( nks == 1)
            nks_ov = np.array([n // 2 if n > 1 else 1 for n in nks], dtype=int)
            shift = mf.kpts[ np.argmin( np.sum(mf.kpts**2, axis=1) )]   #   shift of the original k-point mesh
            if np.any(shift[nonperiodic_dir]):
                log = logger.Logger(self.stdout, self.verbose)
                log.warn('The reference mesh is shifted in a nonperiodic direction')
                raise ValueError('flag_submesh=True requires an unshifted reference mesh')
            half_shift = mf.cell.get_abs_kpts( [1/n if n > 1 else 0 for n in nks] )

            # nks_ov = nks // 2
            # shift = mf.kpts[ np.argmin( np.sum(mf.kpts**2, axis=1) )]   #   shift of the original k-point mesh
            # half_shift = mf.cell.get_abs_kpts( [0.5/n for n in nks_ov] )
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
                raise NotImplementedError(
                    'Band-extension orbitals are currently supported only for 3D cells'
                )

            nks_base = np.asarray(nks if nks_ccd is None else nks_ccd, dtype=int)
            if all([n == 1 for n in nks_base]):
                half_shift = mf.cell.get_abs_kpts( [0.5/n for n in nks_base] )
            else:
                half_shift = mf.cell.get_abs_kpts( [0.5/n if n > 1 else 0 for n in nks_base] )

            if nks_ccd is None:
                kpts_vir = mf.kpts
            else:
                kpts_vir = mf.cell.make_kpts(nks_base, with_gamma_point=True)
            kpts_occ = kpts_vir + half_shift
            kpts = np.concatenate( (kpts_occ, kpts_vir), axis=0)
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
                raise TypeError('KRCCD_2mesh requires a periodic KRHF or KRKS reference')

            self.kpts = kpts
            self.mo_energy = mo_energy
            self.mo_coeff = mo_coeff
            self.mo_occ = mo_occ
            self.with_df = df.GDF(self.cell, kpts).build()


        #   Basic info
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
        self.kpts_idx_vir = [ np.asarray(np.sum((kpts_bz - kvir)**2, axis=-1) < 1e-10).nonzero()[0] for kvir in kpts_vir_bz]
        self.kpts_idx_vir = np.concatenate(self.kpts_idx_vir).astype(int)

        #   indices of occupied k-points in self.kpts
        self.kpts_idx_occ = [ np.asarray(np.sum((kpts_bz - kocc)**2, axis=-1) < 1e-10).nonzero()[0] for kocc in kpts_occ_bz]
        self.kpts_idx_occ = np.concatenate(self.kpts_idx_occ).astype(int)

        if (
            len(self.kpts_idx_vir) != self.nkpts_ov
            or len(np.unique(self.kpts_idx_vir)) != self.nkpts_ov
            or len(self.kpts_idx_occ) != self.nkpts_ov
            or len(np.unique(self.kpts_idx_occ)) != self.nkpts_ov
        ):
            raise ValueError('Could not map both staggered meshes onto the k-point list')
        if np.intersect1d(self.kpts_idx_occ, self.kpts_idx_vir).size:
            raise ValueError('The occupied and virtual staggered meshes overlap')


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

    @property
    def ccsd_vector_desc(self):
        """Description of the ground-state vector."""
        nvir = self.nmo - self.nocc
        return [(self.nkpts_ov, self.nocc, nvir), (self.nkpts_ov,) * 3 + (self.nocc,) * 2 + (nvir,) * 2]

    def ao2mo(self, mo_coeff=None):
        return _ERIS_2mesh(self, mo_coeff)

    # init_amps = init_amps_stagger
    init_amps = init_amps_stagger_zero
    energy = energy_stagger
    update_amps = update_amps_stagger

    #   TODO: the following functions are not supported yet.
    ccsd_t = None
    ipccsd = None
    eaccsd = None
    ipccsd_matvec = None
    ipccsd_diag = None
    eaccsd_matvec = None
    eaccsd_diag = None

class _ERIS_2mesh:  # (pyscf.cc.ccsd._ChemistsERIs):
    # NOTE: all occupied and virtual orbirtals lie on two staggered uniform meshes, respectively
    # NOTE: the two electron integrals are stored in different orders from
    # kccsd_uhf._ERIS.  Integrals (ab|cd) are stored as [ka,kc,kb,a,c,b,d] here
    # while the order is [ka,kb,kc,a,b,c,d] in kccsd_uhf._ERIS
    # NOTE: only consider the canonical HF MOs where the Fock operator is diagonal
    # NOTE:   no madelung correction to ERIs

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
        #   cc.mo_coeff/energy contains quantities for both occ/vir orbitals.
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
                if self.dtype == np.dtype(float): eri_kpt = eri_kpt.real
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
            raise NotImplementedError(
                'Out-of-core ERI transformation is not supported for two-mesh CCD'
            )

        log.timer('CCD integral transformation', *cput0)

def add_vvvv_stagger(cc, Ht2, t1, t2, eris):
    nkpts, nocc, nvir = t1.shape
    kconserv = cc.khelper.kconserv

    if cc.madelung:
        madelung = tools.madelung(cc.cell, cc.kpts_vir)

    #   access chi_vvvv entries
    mem_now = lib.current_memory()[0]
    if cc.direct and getattr(eris, 'Lvv', None) is not None:
        #: If memory is not enough to hold eris.Lpv
        #:def get_Wvvvv(ka, kb, kc):
        #:    kd = kconserv[ka,kc,kb]
        #:    v = cc._scf.with_df.ao2mo([eris.mo_coeff[k] for k in [ka,kc,kb,kd]],
        #:                              cc.kpts[[ka,kc,kb,kd]]).reshape([nmo]*4)
        #:    Wvvvv  = lib.einsum('kcbd,ka->abcd', v[:nocc,nocc:,nocc:,nocc:], -t1[ka])
        #:    Wvvvv += lib.einsum('ackd,kb->abcd', v[nocc:,nocc:,:nocc,nocc:], -t1[kb])
        #:    Wvvvv += v[nocc:,nocc:,nocc:,nocc:].transpose(0,2,1,3)
        #:    Wvvvv *= (1./nkpts)
        #:    return Wvvvv
        def get_Wvvvv(ika, ikb, ikc): # --> <ab|cd>
            Lvv = eris.Lvv  #NOTE: size of Lvv (nkpts_ov, nkpts_ov, naux, nvir, nvir)
            ka, kb, kc = cc.kpts_idx_vir[ika], cc.kpts_idx_vir[ikb], cc.kpts_idx_vir[ikc]
            kd = kconserv[ka, kc, kb]
            ikd = cc.idx_vir[kd]
            Wvvvv = (1. / nkpts) * lib.einsum('Lac,Lbd->abcd', Lvv[ika,ikc], Lvv[ikb,ikd])

            if ika == ikc and cc.madelung:
                for a in range(nvir):
                    for b in range(nvir):
                        Wvvvv[a, b, a, b] += madelung
            return Wvvvv

    elif (nvir ** 4 * nkpts ** 3) * 16 / 1e6 + mem_now < cc.max_memory * .9:
        _Wvvvv = imdk.cc_Wvvvv_stagger(cc, t1, t2, eris, kconserv)

        if cc.madelung:
            for ika in range(nkpts):
                for ikb in range(nkpts):
                    for a in range(nvir):
                        for b in range(nvir):
                            _Wvvvv[ika, ikb, ika, a, b, a, b] += madelung

        def get_Wvvvv(ika, ikb, ikc):
            return _Wvvvv[ika, ikb, ikc]
    else:
        fimd = lib.H5TmpFile()
        _Wvvvv = fimd.create_dataset('vvvv', (nkpts, nkpts, nkpts, nvir, nvir, nvir, nvir), eris.dtype.char)
        _Wvvvv = imdk.cc_Wvvvv_stagger(cc, t1, t2, eris, kconserv, _Wvvvv)

        if cc.madelung:
            for ika in range(nkpts):
                for ikb in range(nkpts):
                    for a in range(nvir):
                        for b in range(nvir):
                            _Wvvvv[ika, ikb, ika, a, b, a, b] += madelung

        def get_Wvvvv(ika, ikb, ikc):
            return _Wvvvv[ika, ikb, ikc]

    #   tensor contraction for the 4p2h term
    for ika, ikb, ikc in kpts_helper.loop_kkk(nkpts):
        ka, kb, kc = cc.kpts_idx_vir[ika], cc.kpts_idx_vir[ikb], cc.kpts_idx_vir[ikc]
        Wvvvv = get_Wvvvv(ika, ikb, ikc)
        for iki, ki in enumerate(cc.kpts_idx_occ):
            kj = kconserv[ka, ki, kb]
            ikj = cc.idx_occ[kj]
            Ht2[iki, ikj, ika] += lib.einsum('abcd,ijcd->ijab', Wvvvv, t2[iki, ikj, ikc])
    fimd = None
    _Wvvvv = None
    return Ht2
