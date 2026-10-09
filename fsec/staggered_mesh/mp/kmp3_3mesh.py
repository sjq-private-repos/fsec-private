"""Restricted periodic MP3 with three staggered k-point meshes.

Based on Xin Xing's staggered-mesh formulation. The default correction
uses an adaptive reciprocal-space quadrature for the shifted-mesh singularity.
"""

import time
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import df
from pyscf.pbc.scf import khf
from pyscf.pbc.dft import krks
from pyscf.pbc.mp.kmp2 import (get_nocc,
                               padded_mo_coeff, padded_mo_energy, padding_k_idx)

from pyscf.pbc.lib import kpts_helper

from pyscf.pbc.tools.pbc import get_monkhorst_pack_size
from pyscf.pbc.scf.khf import get_occ

from pyscf.pbc.cc.kccsd_rhf import _get_epq
from .kmp3 import KMP3
from .._kpoints import make_kpts_helper


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

def madelung_stagger(cell, nks, mesh_shift, ew_eta=None, ew_cut=None):
    """Evaluate the staggered Ewald correction with an explicit G-vector sum.

    ``ew_cut`` remains accepted for compatibility.  The explicit reciprocal
    mesh determines its own cutoff, so this argument is unused.
    """
    del ew_cut
    return madelung_stagger_direct(cell, nks, mesh_shift, ew_eta)

def madelung_stagger_direct(cell, nks, mesh_shift, ew_eta=None):
    #   Singularity subtraction for staggered-mesh-based quadrature
    #
    #   The target function is
    #   1/(2pi)^3 * \sum_{G} 4*pi e^{-|G+q|^2/(4*eta**2)} / |G+q|^2
    #   =  1/|Omega^*| \sum_{G} 4*pi/|Omega| * e^{-|G+q|^2/(4*eta**2)} / |G+q|^2
    #   over domain BZ

    nks = np.asarray(nks, dtype=int)
    mesh_qG = np.asarray(cell.mesh, dtype=int) * nks

    # Cap the reciprocal grid at roughly two samples per Bohr of supercell.
    mesh_max = np.asarray(np.linalg.norm(cell.lattice_vectors() * nks[:, None], axis=1) * 2,
                            dtype=int)  # roughly 2 grids per bohr
    mesh_max[mesh_max<80] = 80
    mesh_qG[mesh_qG>mesh_max] = mesh_max[mesh_qG>mesh_max]

    #   Step 2: deterimine the eta parameter using the mesh size and precision requirement
    if ew_eta is None:
        Gmax = min(np.asarray(mesh_qG)//2 * lib.norm(cell.reciprocal_vectors()/nks.reshape((3,1)), axis=1))
        precision = cell.precision
        log_precision = np.log(precision/(4*np.pi*(Gmax+1e-100)**2))
        ew_eta = np.sqrt(-Gmax**2/(4*log_precision)) + 1e-100

    #   Step 3: generate the q+G mesh by cell.get_Gv_weights(mesh_qG)
    rx = np.fft.fftfreq(mesh_qG[0], 1./mesh_qG[0])
    ry = np.fft.fftfreq(mesh_qG[1], 1./mesh_qG[1])
    rz = np.fft.fftfreq(mesh_qG[2], 1./mesh_qG[2])
    b = cell.reciprocal_vectors() / nks.reshape((3,1))
    Gvbase = (rx, ry, rz)
    Gv = np.dot(lib.cartesian_prod(Gvbase), b)
    Gv = Gv + mesh_shift

    #   Step 4: evaluate the integrand and the corresponding quadrature
    absG2 = np.einsum('gi,gi->g', Gv, Gv)
    absG2[absG2<1e-12] = 1e200
    integrand = 4*np.pi / cell.vol / absG2 * np.exp(-absG2/(4*ew_eta**2))
    val_quadrature = 1/np.prod(nks) * np.sum(integrand)
    val_integral =  2 * ew_eta / np.sqrt(np.pi)
    correction = val_integral - val_quadrature

    return correction

def madelung_stagger_adaptive(cell, nks, mesh_shift, precision=None):
    #   Singularity subtraction for staggered-mesh-based quadrature
    #
    #   The target function is
    #   1/(2pi)^3 * \sum_{G} 4*pi e^{-|G+q|^2/(4*eta**2)} / |G+q|^2
    #   =  1/|Omega^*| \sum_{G} 4*pi/|Omega| * e^{-|G+q|^2/(4*eta**2)} / |G+q|^2
    #   over domain BZ

    if precision is None:
        precision = cell.precision
    log = logger.Logger(cell.stdout, cell.verbose)
    nks = np.asarray(nks, dtype=int)
    mesh_qG = np.asarray(cell.mesh, dtype=int) * nks

    #   Step 2: deterimine the eta parameter using the mesh size and precision requirement
    Gmax = min(np.asarray(mesh_qG)//2 * lib.norm(cell.reciprocal_vectors()/nks.reshape((3,1)), axis=1))
    log_precision = np.log(precision/(4*np.pi*(Gmax+1e-100)**2))
    ew_eta = np.sqrt(-Gmax**2/(4*log_precision)) + 1e-100
    ew_eta = 0.5 * ew_eta

    correction0 = madelung_stagger_direct(cell, nks, mesh_shift, ew_eta)
    log.debug("Madelung quadrature eta=%s correction=%s", ew_eta, correction0)

    for _ in range(4):
        #   increase ew_eta
        ew_eta *= 2

        #   adapative choose mesh size
        Gmax = min(np.asarray(mesh_qG)//2 * lib.norm(cell.reciprocal_vectors()/nks.reshape((3,1)), axis=1))
        log_precision = np.log(precision/(4*np.pi*(Gmax+1e-100)**2))
        Gmin = np.sqrt(4*ew_eta**2 * (-log_precision))
        mesh_qG = np.asarray( np.ceil( 2 * Gmin / lib.norm(cell.reciprocal_vectors()/nks.reshape((3,1)), axis=1)),
                             dtype=int)

        #   generate the q+G mesh by cell.get_Gv_weights(mesh_qG)
        rx = np.fft.fftfreq(mesh_qG[0], 1./mesh_qG[0])
        ry = np.fft.fftfreq(mesh_qG[1], 1./mesh_qG[1])
        rz = np.fft.fftfreq(mesh_qG[2], 1./mesh_qG[2])
        b = cell.reciprocal_vectors() / nks.reshape((3,1))
        Gvbase = (rx, ry, rz)
        Gv = np.dot(lib.cartesian_prod(Gvbase), b)
        Gv = Gv + mesh_shift

        #   evaluate the integrand and the corresponding quadrature
        absG2 = np.einsum('gi,gi->g', Gv, Gv)
        absG2[absG2<1e-12] = 1e200
        integrand = 4*np.pi / cell.vol / absG2 * np.exp(-absG2/(4*ew_eta**2))
        val_quadrature = 1/np.prod(nks) * np.sum(integrand)
        val_integral =  2 * ew_eta / np.sqrt(np.pi)
        correction = val_integral - val_quadrature

        log.debug("Madelung quadrature eta=%s correction=%s", ew_eta, correction)

        if np.abs(correction - correction0) < precision:
            log.debug("Madelung quadrature reached precision %s", precision)
            break
        else:
            correction0 = correction
    return correction


def mp3_amplitude_3mesh_efficient(mp, eris):
    time0 = time1 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mp.stdout, mp.verbose)
    time0 = log.timer_debug1('start', *time0)
    time1 = log.timer_debug1('start', *time1)


    #   basic information
    nkpts = mp.nkpts_base
    nocc = mp.nocc
    nmo = mp.nmo
    nvir = nmo - nocc
    kconserv = mp.khelper.kconserv
    mo_e_o = [e[:nocc] for e in eris.mo_energy]
    mo_e_v = [e[nocc:] for e in eris.mo_energy]
    nonzero_opadding, nonzero_vpadding = padding_k_idx(mp, kind="split")
    mesh2idx = mp.mesh2idx
    kpts_base_idx = mp.kpts_base_idx
    idx_base = mp.idx_base

    #   auxiliary MP2 t-amplitude
    t_mp2 = {}
    for mi, mj, ma, mb in mp._mesh_mp2_amp:
        midx= mesh2idx(mi, mj, ma)
        t_mp2[midx] = np.empty((nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir), dtype=eris.dtype)

        for iki, ikj, ika in kpts_helper.loop_kkk(nkpts):
            ki, kj, ka = kpts_base_idx[mi][iki], kpts_base_idx[mj][ikj], kpts_base_idx[ma][ika]
            kb = kconserv[ki, ka, kj]
            eia = _get_epq([0,nocc,ki,mo_e_o,nonzero_opadding],
                           [0,nvir,ka,mo_e_v,nonzero_vpadding],
                           fac=[1.0,-1.0])
            ejb = _get_epq([0,nocc,kj,mo_e_o,nonzero_opadding],
                           [0,nvir,kb,mo_e_v,nonzero_vpadding],
                           fac=[1.0,-1.0])
            eijab = eia[:, None, :, None] + ejb[None, :, None, :]
            t_mp2[midx][iki, ikj, ika] = eris.oovv[midx][iki, ikj, ika].conj() / eijab
    time1 = log.timer_debug1('MP2 amp preparation', *time1)
    mp._t2_mp2 = t_mp2[mp.mesh2idx(*mp._mesh_mp3_amp[:3])]


    #   compute MP3 t-amplitude
    t_mp3 = np.empty((nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir), dtype=eris.dtype)
    mi, mj, ma, mb = mp._mesh_mp3_amp

    for iki, ikj, ika in kpts_helper.loop_kkk(nkpts):
        ki, kj, ka = kpts_base_idx[mi][iki], kpts_base_idx[mj][ikj], kpts_base_idx[ma][ika]
        kb = kconserv[ki, ka, kj]
        ikb = idx_base[mb][kb]
        t2new_tmp = np.zeros((nocc, nocc, nvir, nvir), dtype=eris.dtype)

        #   4h2p
        mk, ml = mp._mesh_term[0]
        kk_kl = np.asarray([ [idx_base[mk][kconserv[kj, kl, ki]], ikl] for ikl, kl in enumerate(kpts_base_idx[ml])])
        t2new_tmp += einsum('xklij,xklab->ijab', eris.oooo[mesh2idx(mk, ml, mi)][kk_kl[:,0],kk_kl[:,1],iki], t_mp2[mesh2idx(mk, ml, ma)][kk_kl[:,0], kk_kl[:,1], ika])

        #   2h4p
        mc, md = mp._mesh_term[1]
        t2new_tmp += einsum('xabcd,xijcd->ijab', eris.vvvv[mesh2idx(ma,mb,mc)][ika,ikb,:], t_mp2[mesh2idx(mi, mj, mc)][iki, ikj, :])

        #   3h3p
        #   term 1 & 2
        mk, mc = mp._mesh_term[2]
        kk_kc = np.asarray([ [ikk, idx_base[mc][kconserv[ka, ki, kk]]] for ikk, kk in enumerate(kpts_base_idx[mk])])
        t2new_tmp += 2*einsum('xakic,xkjcb->ijab', eris.voov[mesh2idx(ma,mk,mi)][ika, kk_kc[:,0], iki], t_mp2[mesh2idx(mk, mj, mc)][kk_kc[:,0], ikj, kk_kc[:,1]])
        t2new_tmp -= einsum('xakic,xkjcb->ijab', eris.ovov[mesh2idx(mk,ma,mi)][kk_kc[:,0], ika, iki].transpose(0, 2, 1, 3, 4), t_mp2[mesh2idx(mk, mj, mc)][kk_kc[:,0], ikj, kk_kc[:,1]])

        #   term 3
        t2new_tmp -= einsum('xakic,xkjbc->ijab', eris.voov[mesh2idx(ma,mk,mi)][ika,:,iki], t_mp2[mesh2idx(mk, mj, mb)][:, ikj, ikb])

        #   term 4
        mk, mc = mp._mesh_term[3]
        t2new_tmp -= einsum('xkbic,xkjac->ijab', eris.ovov[mesh2idx(mk,mb,mi)][:,ikb,iki], t_mp2[mesh2idx(mk, mj, ma)][:, ikj, ika])


        #   SYMMETRIZED TERMS with (I, A) <-> (J, B)
        #   term 1 & 2
        mk, mc = mp._mesh_term[4]
        kk_kc = np.asarray([ [ikk, idx_base[mc][kconserv[kb, kj, kk]]] for ikk, kk in enumerate(kpts_base_idx[mk])])
        t2new_tmp += 2*einsum('xbkjc,xkica->ijab', eris.voov[mesh2idx(mb,mk,mj)][ikb, kk_kc[:,0], ikj], t_mp2[mesh2idx(mk, mi, mc)][kk_kc[:,0], iki, kk_kc[:,1]])
        t2new_tmp -= einsum('xbkjc,xkica->ijab', eris.ovov[mesh2idx(mk,mb,mj)][kk_kc[:,0], ikb, ikj].transpose(0, 2, 1, 3, 4), t_mp2[mesh2idx(mk, mi, mc)][kk_kc[:,0], iki, kk_kc[:,1]])

        #   term 3
        t2new_tmp -= einsum('xbkjc,xkiac->ijab', eris.voov[mesh2idx(mb,mk,mj)][ikb,:,ikj], t_mp2[mesh2idx(mk, mi, ma)][:, iki, ika])

        #   term 4
        mk, mc = mp._mesh_term[5]
        t2new_tmp -= einsum('xkajc,xkibc->ijab', eris.ovov[mesh2idx(mk,ma,mj)][:,ika,ikj], t_mp2[mesh2idx(mk, mi, mb)][:, iki, ikb])


        #   Madelung correction
        if mp.madelung:
            t2new_tmp += 2 * mp.madelung_constant * t_mp2[mesh2idx(mi, mj, ma)][iki, ikj, ika]

        #   MP2 term
        eia = _get_epq([0,nocc,ki,mo_e_o,nonzero_opadding],
                    [0,nvir,ka,mo_e_v,nonzero_vpadding],
                    fac=[1.0,-1.0])
        ejb = _get_epq([0,nocc,kj,mo_e_o,nonzero_opadding],
                    [0,nvir,kb,mo_e_v,nonzero_vpadding],
                    fac=[1.0,-1.0])
        eijab = eia[:, None, :, None] + ejb[None, :, None, :]
        t2new_tmp /= eijab
        t2new_tmp += t_mp2[mesh2idx(mi, mj, ma)][iki, ikj, ika]
        t_mp3[iki, ikj, ika]  = t2new_tmp

    time1 = log.timer_debug1('compute MP3 t-amplitude', *time1)
    time0 = log.timer_debug1('compute MP3 t-amplitude', *time0)
    return t_mp3

def mp3_amplitude_3mesh(mp, eris):
    time0 = time1 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mp.stdout, mp.verbose)
    time0 = log.timer_debug1('start', *time0)
    time1 = log.timer_debug1('start', *time1)

    #   basic information
    nkpts = mp.nkpts_base
    nocc = mp.nocc
    nmo = mp.nmo
    nvir = nmo - nocc
    kconserv = mp.khelper.kconserv
    mo_e_o = [e[:nocc] for e in eris.mo_energy]
    mo_e_v = [e[nocc:] for e in eris.mo_energy]
    nonzero_opadding, nonzero_vpadding = padding_k_idx(mp, kind="split")
    mesh2idx = mp.mesh2idx
    kpts_base_idx = mp.kpts_base_idx
    idx_base = mp.idx_base

    if mp.madelung:
        madelung_correction = [
                               madelung_stagger(mp.cell, mp.nks_base, np.zeros((1, 3))),
                               madelung_stagger(mp.cell, mp.nks_base, mp.shifts[0]),
                               madelung_stagger(mp.cell, mp.nks_base, mp.shifts[1]),
                              ]

    #   auxiliary MP2 t-amplitude
    t_mp2 = {}
    for mi, mj, ma, mb in mp._mesh_mp2_amp:
        midx= mesh2idx(mi, mj, ma)
        t_mp2[midx] = np.empty((nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir), dtype=eris.dtype)

        for iki, ikj, ika in kpts_helper.loop_kkk(nkpts):
            ki, kj, ka = kpts_base_idx[mi][iki], kpts_base_idx[mj][ikj], kpts_base_idx[ma][ika]
            kb = kconserv[ki, ka, kj]
            eia = _get_epq([0,nocc,ki,mo_e_o,nonzero_opadding],
                           [0,nvir,ka,mo_e_v,nonzero_vpadding],
                           fac=[1.0,-1.0])
            ejb = _get_epq([0,nocc,kj,mo_e_o,nonzero_opadding],
                           [0,nvir,kb,mo_e_v,nonzero_vpadding],
                           fac=[1.0,-1.0])
            eijab = eia[:, None, :, None] + ejb[None, :, None, :]
            t_mp2[midx][iki, ikj, ika] = eris.oovv[midx][iki, ikj, ika].conj() / eijab
    time1 = log.timer_debug1('MP2 amp preparation', *time1)
    mp._t2_mp2 = t_mp2[mp.mesh2idx(*mp._mesh_mp3_amp[:3])]


    #   compute MP3 t-amplitude
    t_mp3 = np.empty((nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir), dtype=eris.dtype)
    mi, mj, ma, mb = mp._mesh_mp3_amp

    for iki, ikj, ika in kpts_helper.loop_kkk(nkpts):
        ki, kj, ka = kpts_base_idx[mi][iki], kpts_base_idx[mj][ikj], kpts_base_idx[ma][ika]
        kb = kconserv[ki, ka, kj]
        ikb = idx_base[mb][kb]
        t2new_tmp = np.zeros((nocc, nocc, nvir, nvir), dtype=eris.dtype)

        #   2nd term: 4h2p
        mk, ml = mp._mesh_term[0]
        for ikl, kl in enumerate(kpts_base_idx[ml]):
            kk = kconserv[kj, kl, ki]
            ikk = idx_base[mk][kk]
            t2new_tmp += einsum('klij,klab->ijab', eris.oooo[mesh2idx(mk, ml, mi)][ikk,ikl,iki], t_mp2[mesh2idx(mk, ml, ma)][ikk, ikl, ika])

        #   madelung correction
        if mp.madelung:
            t2new_tmp += madelung_correction[(mi-mk)%3] * t_mp2[mesh2idx(mi, mj, ma)][iki, ikj, ika]

        #   3rd term: 4p2h
        mc, md = mp._mesh_term[1]
        for ikc, kc in enumerate(kpts_base_idx[mc]):
            t2new_tmp += einsum('abcd,ijcd->ijab', eris.vvvv[mesh2idx(ma,mb,mc)][ika,ikb,ikc], t_mp2[mesh2idx(mi, mj, mc)][iki, ikj, ikc])

        #   madelung correction
        if mp.madelung:
            t2new_tmp += madelung_correction[(mc-ma)%3] * t_mp2[mesh2idx(mi, mj, ma)][iki, ikj, ika]

        #   4-5th term
        mk, mc = mp._mesh_term[2]
        for ikk, kk in enumerate(kpts_base_idx[mk]):
            kc = kconserv[ka, ki, kk]
            ikc = idx_base[mc][kc]
            tmp_voov = 2. * eris.voov[mesh2idx(ma,mk,mi)][ika,ikk,iki] - eris.ovov[mesh2idx(mk,ma,mi)][ikk,ika,iki].transpose(1, 0, 2, 3)
            t2new_tmp += einsum('akic,kjcb->ijab', tmp_voov, t_mp2[mesh2idx(mk, mj, mc)][ikk, ikj, ikc])

            kc = kconserv[ka, ki, kk]
            ikc = idx_base[mc][kc]
            t2new_tmp -= einsum('akic,kjbc->ijab', eris.voov[mesh2idx(ma,mk,mi)][ika,ikk,iki], t_mp2[mesh2idx(mk, mj, mb)][ikk, ikj, ikb])

        #   madelung correction
        if mp.madelung:
            t2new_tmp -= madelung_correction[(mc-ma)%3] * t_mp2[mesh2idx(mi, mj, ma)][iki, ikj, ika]


        #   6th term
        mk, mc = mp._mesh_term[3]
        for ikk, kk in enumerate(kpts_base_idx[mk]):
            t2new_tmp -= einsum('kbic,kjac->ijab', eris.ovov[mesh2idx(mk,mb,mi)][ikk,ikb,iki], t_mp2[mesh2idx(mk, mj, ma)][ikk, ikj, ika])

        #   madelung correction
        if mp.madelung:
            t2new_tmp -= madelung_correction[(mc-mb)%3] * t_mp2[mesh2idx(mi, mj, ma)][iki, ikj, ika]


        #   SYMMETRIZED TERMS with (I, A) <-> (J, B)
        mk, mc = mp._mesh_term[4]
        for ikk, kk in enumerate(kpts_base_idx[mk]):
            kc = kconserv[kb, kj, kk]
            ikc = idx_base[mc][kc]
            tmp_voov = 2. * eris.voov[mesh2idx(mb,mk,mj)][ikb,ikk,ikj] - eris.ovov[mesh2idx(mk,mb,mj)][ikk,ikb,ikj].transpose(1, 0, 2, 3)
            t2new_tmp += einsum('bkjc,kica->ijab', tmp_voov, t_mp2[mesh2idx(mk,mi,mc)][ikk, iki, ikc])

            kc = kconserv[kb, kj, kk]
            ikc = idx_base[mc][kc]
            t2new_tmp -= einsum('bkjc,kiac->ijab', eris.voov[mesh2idx(mb,mk,mj)][ikb,ikk,ikj], t_mp2[mesh2idx(mk, mi, ma)][ikk, iki, ika])

        #   madelung correction
        if mp.madelung:
            t2new_tmp -= madelung_correction[(mc-mb)%3] * t_mp2[mesh2idx(mi, mj, ma)][iki, ikj, ika]

        mk, mc = mp._mesh_term[5]
        for ikk, kk in enumerate(kpts_base_idx[mk]):
            t2new_tmp -= einsum('kajc,kibc->ijab', eris.ovov[mesh2idx(mk,ma,mj)][ikk,ika,ikj], t_mp2[mesh2idx(mk, mi, mb)][ikk, iki, ikb])

        #   madelung correction
        if mp.madelung:
            t2new_tmp -= madelung_correction[(mc-ma)%3] * t_mp2[mesh2idx(mi, mj, ma)][iki, ikj, ika]


        #   MP2 term
        eia = _get_epq([0,nocc,ki,mo_e_o,nonzero_opadding],
                    [0,nvir,ka,mo_e_v,nonzero_vpadding],
                    fac=[1.0,-1.0])
        ejb = _get_epq([0,nocc,kj,mo_e_o,nonzero_opadding],
                    [0,nvir,kb,mo_e_v,nonzero_vpadding],
                    fac=[1.0,-1.0])
        eijab = eia[:, None, :, None] + ejb[None, :, None, :]
        t2new_tmp /= eijab
        t2new_tmp += t_mp2[mesh2idx(mi, mj, ma)][iki, ikj, ika]
        t_mp3[iki, ikj, ika]  = t2new_tmp

    time1 = log.timer_debug1('compute MP3 t-amplitude', *time1)
    time0 = log.timer_debug1('compute MP3 t-amplitude', *time0)
    return t_mp3

def mp3_energy_3mesh(mp, t2, eris):
    nkpts = mp.nkpts_base
    kconserv = mp.khelper.kconserv
    e = 0.0 + 1j * 0.0
    mi, mj, ma, mb = mp._mesh_mp3_amp
    kpts_base_idx = mp.kpts_base_idx
    idx_base = mp.idx_base
    mesh2idx = mp.mesh2idx

    for iki, ikj, ika in kpts_helper.loop_kkk(nkpts):
        ki, kj, ka = kpts_base_idx[mi][iki], kpts_base_idx[mj][ikj], kpts_base_idx[ma][ika]
        kb = kconserv[ki, ka, kj]
        ikb = idx_base[mb][kb]
        e += 2 * einsum('ijab,ijab', t2[iki, ikj, ika], eris.oovv[mesh2idx(mi,mj,ma)][iki, ikj, ika])
        e += - einsum('ijab,ijba', t2[iki, ikj, ika], eris.oovv[mesh2idx(mi,mj,mb)][iki, ikj, ikb])
    e /= nkpts
    if abs(e.imag) > 1e-4:
        logger.warn(mp, 'Non-zero imaginary part found in KMP3 energy %s', e)
    return e.real

def mesh_config_mp3(mesh_mp3_amp):
    mi, mj, ma, mb = mesh_mp3_amp[0], mesh_mp3_amp[1], mesh_mp3_amp[2], mesh_mp3_amp[3]
    mi, mj, ma, mb = mi%3, mj%3, ma%3, mb%3
    if mi == ma or (mi+mj)%3 != (ma+mb)%3:
        raise ValueError

    if not (mi == mj or ma == mb):
        return ValueError

    #   case 1: t_{xx}^{yz}
    if mi == mj:
        x, y, z = mi, ma, mb

        #   PART 1: mesh for MP2 amplitudes
        mesh_mp2_amp = []
        #   term 1
        mesh_mp2_amp.append( (x, x, y, z) )
        #   term 2
        mesh_mp2_amp.append( (z, y, y, z) )
        #   term 3
        mesh_mp2_amp.append( (x, x, z, y) )
        #   term 4-5
        mesh_mp2_amp.append( (y, x, z, z) )
        #   term 4-5 symmetrized
        mesh_mp2_amp.append( (z, x, y, y) )
        # #   term 6
        # mesh_mp2_amp.append( (z, x, y, y) )
        # #   term 6 symmetrized
        # mesh_mp2_amp.append( (y, x, z, z) )

        #   PART 2: mesh for intermediate momentum vectors in MP3
        mesh_term = []
        #   term 2:
        mesh_term.append( (z, y) )
        #   term 3:
        mesh_term.append( (z, y) )
        #   term 4-5:
        mesh_term.append( (y, z) )
        #   term 6:
        mesh_term.append( (z, y) )
        #   term 4-5 symmetrized:
        mesh_term.append( (z, y) )
        #   term 6 symmetrized:
        mesh_term.append( (y, z) )


        #   PART 3: mesh for ERIs
        mesh_eri = {}
        #   term 1:
        mesh_eri["oovv"] = mesh_mp2_amp[:]
        #   term 2:
        mesh_eri["oooo"] = [(z, y, x, x)]
        #   term 3:
        mesh_eri["vvvv"] = [(y, z, z, y)]
        #   term 4-5:
        mesh_eri["voov"] = [(y, y, x, z)]
        mesh_eri["ovov"] = [(y, y, x, z)]
        #   term 4-5 symmetrized
        mesh_eri["voov"].append((z, z, x, y))
        mesh_eri["ovov"].append((z, z, x, y))
        # #   term 6:
        # mesh_eri["ovov"].append((z, z, x, y))
        # #   term 6 symmetrized
        # mesh_eri["ovov"].append((y, y, x, z))


    #   case 2: t_{yz}^{xx}
    if ma == mb:
        x, y, z = ma, mi, mj

        #   PART 1: mesh for MP2 amplitudes
        mesh_mp2_amp = []
        #   term 1
        mesh_mp2_amp.append( (y, z, x, x) )
        #   term 2
        mesh_mp2_amp.append( (z, y, x, x) )
        #   term 3
        mesh_mp2_amp.append( (y, z, z, y) )
        #   term 4-5
        mesh_mp2_amp.append( (z, z, y, x) )
        #   term 4-5 symmetrized
        mesh_mp2_amp.append( (y, y, z, x) )
        #   term 6
        mesh_mp2_amp.append( (z, z, x, y) )
        #   term 6 symmetrized
        mesh_mp2_amp.append( (y, y, x, z) )

        #   PART 2: mesh for intermediate momentum vectors in MP3
        mesh_term = []
        #   term 2:
        mesh_term.append( (z, y) )
        #   term 3:
        mesh_term.append( (z, y) )
        #   term 4-5:
        mesh_term.append( (z, y) )
        #   term 6:
        mesh_term.append( (z, y) )
        #   term 4-5 symmetrized:
        mesh_term.append( (y, z) )
        #   term 6 symmetrized:
        mesh_term.append( (y, z) )


        #   PART 3: mesh for ERIs
        mesh_eri = {}
        #   term 1:
        mesh_eri["oovv"] = mesh_mp2_amp[:]
        #   term 2:
        mesh_eri["oooo"] = [(z, y, y, z)]
        #   term 3:
        mesh_eri["vvvv"] = [(x, x, z, y)]
        #   term 4-5:
        mesh_eri["voov"] = [(x, z, y, y)]
        mesh_eri["ovov"] = [(z, x, y, y)]
        #   term 4-5 symmetrized
        mesh_eri["voov"].append((x, y, z, z))
        mesh_eri["ovov"].append((y, x, z, z))
        # #   term 6:
        # mesh_eri["ovov"].append((z, x, y, y))
        # #   term 6 symmetrized
        # mesh_eri["ovov"].append((y, x, z, z))

    return mesh_mp2_amp, mesh_term, mesh_eri

class KMP3_3mesh(KMP3):
    """Three-mesh staggered restricted MP3.

    ``flag_submesh=True`` uses three commensurate submeshes of the SCF mesh;
    otherwise the method evaluates non-SCF band orbitals on three shifted
    meshes.  The source method's adaptive Madelung correction is disabled by
    default and can be enabled with ``madelung=True``.
    """

    def __init__(self, mf, frozen=None, flag_submesh=False, nks_mp=None,
                 madelung=False):
        self.cell = mf.cell
        self._scf = mf
        self.verbose = self.cell.verbose
        self.stdout = self.cell.stdout
        self.max_memory = self.cell.max_memory
        log = logger.Logger(self.stdout, self.verbose)
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
            if ((nks % 3 != 0) & (nks > 1)).any() :
                log = logger.Logger(self.stdout, self.verbose)
                log.warn('The k-point mesh in the input mf has meshsize not divided by 3 in certain dimension. \n'
                        'Cannot use its submeshes for staggered mesh calculation!\n'
                        'You may set "flag_submesh" to False')
                raise RuntimeError

            #   Temporary solution for low-dimensional case
            nonperiodic_dir = np.where( nks == 1)
            nks_base = np.asarray([n // 3 if n > 1 else 1 for n in nks], dtype=int)
            shift = mf.kpts[ np.argmin( np.sum(mf.kpts**2, axis=1) )]   #   shift of the original k-point mesh
            if np.any(shift[nonperiodic_dir]):
                log.warn("The given k-point mesh has non-zero shift in a "
                         "nonperiodic direction")
                raise RuntimeError
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
                #   when given a 1*1*1 mesh, treat it as a 3d periodic system
                shift13 = mf.cell.get_abs_kpts( [1.0/3.0/n for n in nks_base] )
                shift23 = mf.cell.get_abs_kpts( [2.0/3.0/n for n in nks_base] )
            else:
                shift13 = mf.cell.get_abs_kpts( [1.0/3.0/n if n > 1 else 0 for n in nks_base] )
                shift23 = mf.cell.get_abs_kpts( [2.0/3.0/n if n > 1 else 0 for n in nks_base] )

            if nks_mp is None:
                kpts0 = mf.kpts
            else:
                kpts0 = mf.cell.make_kpts(nks_base, with_gamma_point=True)
            kpts1 = kpts0 + shift13
            kpts2 = kpts0 + shift23
            kpts = np.concatenate( (kpts0, kpts1, kpts2), axis=0)
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
            self.with_df = df.GDF(self.cell, self.kpts).build()

        #   Basic info
        self.frozen = frozen
        self.khelper = make_kpts_helper(self.cell, self.kpts)

        #   Mesh info for the staggered mesh method
        self.kpts_base = [kpts0, kpts1, kpts2]
        self.nkpts_base = len(kpts0)
        self.nkpts = len(self.kpts)
        self.nks_base = nks_base
        self.shifts = [shift13, shift23]

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
                len(self.kpts_base_idx[-1]) != self.nkpts_base or
                len(np.unique(self.kpts_base_idx[-1])) != self.nkpts_base
            ):
                log = logger.Logger(self.stdout, self.verbose)
                log.warn('Cannot locate the provided occupied/virtual submeshes in the large k-point mesh')
                raise RuntimeError
        for imesh in range(3):
            for jmesh in range(imesh):
                if np.intersect1d(
                    self.kpts_base_idx[imesh], self.kpts_base_idx[jmesh]
                ).size:
                    raise ValueError('three staggered submeshes must be disjoint')

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

        self._mesh_mp3_amp = (0, 0, 1, 2)
        # self._mesh_mp3_amp = (1, 2, 0, 0)
        # self._mesh_mp3_amp = (0, 0, 2, 1)
        # self._mesh_mp3_amp = (1, 1, 0, 2)
        self._mesh_mp2_amp, self._mesh_term, self._mesh_eri = \
            mesh_config_mp3( self._mesh_mp3_amp )

        #   Madelung correction
        self.madelung = madelung
        if self.madelung:
            self.madelung_constant = -madelung_stagger_adaptive(
                self.cell, self.nks_base, self.shifts[0]
            )
        else:
            self.madelung_constant = 0.0

        # c0 = -madelung_stagger(self.cell, self.nks_base, self.shifts[0])
        # c1 = -madelung_stagger_direct(self.cell, self.nks_base, self.shifts[0])
        # c2 = -madelung_stagger_adaptive(self.cell, self.nks_base, self.shifts[0])
        # logger.info(self, f"three madelung {c0}, {c1}, {c2}")




    # double_amplitude = mp3_amplitude_3mesh
    double_amplitude = mp3_amplitude_3mesh_efficient
    correlation_energy = mp3_energy_3mesh


    def ao2mo(self, mo_coeff=None, mo_energy=None):
        return _ERIS_3mesh(self, mo_coeff, mo_energy)

    def mesh2idx(self, mp, mq, mr):
        return (mp%3) + 3 * (mq%3) + 9* (mr%3)

    def idx2mesh(self, idx):
        mp = idx % 3
        mq = (idx // 3) % 3
        mr = (idx // 9) % 3
        return mp, mq, mr


def _mem_usage(nkpts, nocc, nvir):
    incore = 0
    #   MP2 amp
    incore += 5 * nkpts ** 3 * (nocc**2 * nvir**2)
    #   oovv
    incore += 5 * nkpts ** 3 * (nocc**2 * nvir**2)
    #   oooo
    incore += nkpts ** 3 * nocc**4
    #   vvvv
    incore += nkpts ** 3 * nvir**4
    #   voov/ovov
    incore += 4 * nkpts ** 3 * (nocc**2 * nvir**2)

    # Roughly, factor of two for intermediates and factor of two
    # for safety (temp arrays, copying, etc)
    incore *= 1.5
    # TODO: Improve incore estimate and add outcore estimate
    outcore = basic = incore
    return incore * 16 / 1e6, outcore * 16 / 1e6, basic * 16 / 1e6

class _ERIS_3mesh:
    #   construct ERIs <pq|rs> with all possible sampling of pqrs on the submeshes specified
    #   by "self.kpts_ijab"
    def __init__(self, cc, mo_coeff=None, mo_energy=None, method='incore'):
        log = logger.Logger(cc.stdout, cc.verbose)
        time0 = time1 = logger.process_clock(), logger.perf_counter()
        time0 = log.timer_debug1('start', *time0)
        time1 = log.timer_debug1('start', *time1)

        #   Basic information
        cell = cc._scf.cell
        kpts = cc.kpts
        nkpts = cc.nkpts
        nkpts_base = cc.nkpts_base
        nocc = cc.nocc
        nvir = cc.nmo - cc.nocc
        max_memory = cc.max_memory

        #   NOTE: DO NOT USE mo_coeff/energy from _scf
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
            logger.warn(cc, 'HOMO-LUMO gap %s too small for CC and MBPT. '
                            'May cause issues in convergence.', gap)

        mem_incore, mem_outcore, mem_basic = _mem_usage(nkpts_base, nocc, nvir)
        mem_now = lib.current_memory()[0]

        fao2mo = cc.with_df.ao2mo
        kconserv = cc.khelper.kconserv
        kpts_base_idx = cc.kpts_base_idx

        log.info("Three-mesh ERI memory estimate %.1f MB; current use %.1f MB; "
                 "limit %.1f MB", mem_incore, mem_now, max_memory)
        if (method == 'incore' and (mem_incore + mem_now < max_memory)
                or cell.incore_anyway):
            log.info('using incore ERI storage')
            start = time.time()

            self.oovv = {}
            for mp, mq, mr, ms in cc._mesh_eri["oovv"]:
                idx_pqr = cc.mesh2idx(mp, mq, mr)
                self.oovv[idx_pqr] = np.empty((nkpts_base,nkpts_base,nkpts_base,nocc,nocc,nvir,nvir), dtype=self.dtype)
                for ikp, ikq, ikr in kpts_helper.loop_kkk(nkpts_base):
                    kp, kq, kr = kpts_base_idx[mp][ikp], kpts_base_idx[mq][ikq], kpts_base_idx[mr][ikr]
                    ks = kconserv[kp, kr, kq]

                    orbo_p = mo_coeff[kp][:, :nocc]
                    orbo_q = mo_coeff[kq][:, :nocc]
                    orbv_r = mo_coeff[kr][:, nocc:]
                    orbv_s = mo_coeff[ks][:, nocc:]
                    buf_kpt = fao2mo((orbo_p, orbv_r, orbo_q, orbv_s),
                                     (kpts[kp], kpts[kr], kpts[kq], kpts[ks]), compact=False)
                    if self.dtype == float: buf_kpt = buf_kpt.real
                    buf_kpt = buf_kpt.reshape(nocc, nvir, nocc, nvir).transpose(0, 2, 1, 3)
                    self.oovv[idx_pqr][ikp, ikq, ikr, :, :, :, :] = buf_kpt / nkpts_base
            time1 = log.timer_debug1('oovv ERI', *time1)


            self.oooo = {}
            for mp, mq, mr, ms in cc._mesh_eri["oooo"]:
                idx_pqr = cc.mesh2idx(mp, mq, mr)
                self.oooo[idx_pqr] = np.empty((nkpts_base,nkpts_base,nkpts_base,nocc,nocc,nocc,nocc), dtype=self.dtype)
                for ikp, ikq, ikr in kpts_helper.loop_kkk(nkpts_base):
                    kp, kq, kr = kpts_base_idx[mp][ikp], kpts_base_idx[mq][ikq], kpts_base_idx[mr][ikr]
                    ks = kconserv[kp, kr, kq]

                    orbo_p = mo_coeff[kp][:, :nocc]
                    orbo_q = mo_coeff[kq][:, :nocc]
                    orbo_r = mo_coeff[kr][:, :nocc]
                    orbo_s = mo_coeff[ks][:, :nocc]
                    buf_kpt = fao2mo((orbo_p, orbo_r, orbo_q, orbo_s),
                                     (kpts[kp], kpts[kr], kpts[kq], kpts[ks]), compact=False)
                    if self.dtype == float: buf_kpt = buf_kpt.real
                    buf_kpt = buf_kpt.reshape(nocc, nocc, nocc, nocc).transpose(0, 2, 1, 3)
                    self.oooo[idx_pqr][ikp, ikq, ikr, :, :, :, :] = buf_kpt / nkpts_base
            time1 = log.timer_debug1('oooo ERI', *time1)



            self.ovov = {}
            for mp, mq, mr, ms in cc._mesh_eri["ovov"]:
                idx_pqr = cc.mesh2idx(mp, mq, mr)
                self.ovov[idx_pqr] = np.empty((nkpts_base,nkpts_base,nkpts_base,nocc,nvir,nocc,nvir), dtype=self.dtype)
                for ikp, ikq, ikr in kpts_helper.loop_kkk(nkpts_base):
                    kp, kq, kr = kpts_base_idx[mp][ikp], kpts_base_idx[mq][ikq], kpts_base_idx[mr][ikr]
                    ks = kconserv[kp, kr, kq]

                    orbo_p = mo_coeff[kp][:, :nocc]
                    orbv_q = mo_coeff[kq][:, nocc:]
                    orbo_r = mo_coeff[kr][:, :nocc]
                    orbv_s = mo_coeff[ks][:, nocc:]
                    buf_kpt = fao2mo((orbo_p, orbo_r, orbv_q, orbv_s),
                                     (kpts[kp], kpts[kr], kpts[kq], kpts[ks]), compact=False)
                    if self.dtype == float: buf_kpt = buf_kpt.real
                    buf_kpt = buf_kpt.reshape(nocc, nocc, nvir, nvir).transpose(0, 2, 1, 3)
                    self.ovov[idx_pqr][ikp, ikq, ikr, :, :, :, :] = buf_kpt / nkpts_base
            time1 = log.timer_debug1('ovov ERI', *time1)

            self.voov = {}
            for mp, mq, mr, ms in cc._mesh_eri["voov"]:
                idx_pqr = cc.mesh2idx(mp, mq, mr)
                self.voov[idx_pqr] = np.empty((nkpts_base,nkpts_base,nkpts_base,nvir,nocc,nocc,nvir), dtype=self.dtype)
                for ikp, ikq, ikr in kpts_helper.loop_kkk(nkpts_base):
                    kp, kq, kr = kpts_base_idx[mp][ikp], kpts_base_idx[mq][ikq], kpts_base_idx[mr][ikr]
                    ks = kconserv[kp, kr, kq]

                    orbv_p = mo_coeff[kp][:, nocc:]
                    orbo_q = mo_coeff[kq][:, :nocc]
                    orbo_r = mo_coeff[kr][:, :nocc]
                    orbv_s = mo_coeff[ks][:, nocc:]
                    buf_kpt = fao2mo((orbv_p, orbo_r, orbo_q, orbv_s),
                                     (kpts[kp], kpts[kr], kpts[kq], kpts[ks]), compact=False)
                    if self.dtype == float: buf_kpt = buf_kpt.real
                    buf_kpt = buf_kpt.reshape(nvir, nocc, nocc, nvir).transpose(0, 2, 1, 3)
                    self.voov[idx_pqr][ikp, ikq, ikr, :, :, :, :] = buf_kpt / nkpts_base
            time1 = log.timer_debug1('voov ERI', *time1)


            self.vvvv = {}
            for mp, mq, mr, ms in cc._mesh_eri["vvvv"]:
                idx_pqr = cc.mesh2idx(mp, mq, mr)
                self.vvvv[idx_pqr] = np.empty((nkpts_base,nkpts_base,nkpts_base,nvir,nvir,nvir,nvir), dtype=self.dtype)
                for ikp, ikq, ikr in kpts_helper.loop_kkk(nkpts_base):
                    kp, kq, kr = kpts_base_idx[mp][ikp], kpts_base_idx[mq][ikq], kpts_base_idx[mr][ikr]
                    ks = kconserv[kp, kr, kq]

                    orbv_p = mo_coeff[kp][:, nocc:]
                    orbv_q = mo_coeff[kq][:, nocc:]
                    orbv_r = mo_coeff[kr][:, nocc:]
                    orbv_s = mo_coeff[ks][:, nocc:]
                    buf_kpt = fao2mo((orbv_p, orbv_r, orbv_q, orbv_s),
                                     (kpts[kp], kpts[kr], kpts[kq], kpts[ks]), compact=False)
                    if self.dtype == float: buf_kpt = buf_kpt.real
                    buf_kpt = buf_kpt.reshape(nvir, nvir, nvir, nvir).transpose(0, 2, 1, 3)
                    self.vvvv[idx_pqr][ikp, ikq, ikr, :, :, :, :] = buf_kpt / nkpts_base
            time1 = log.timer_debug1('vvvv ERI', *time1)

            log.info("ERI evaluation time: %.3f" % (time.time() - start))
            log.timer('MP3 3-mesh integral transformation', *time0)
        else:
            raise MemoryError(
                'In-core three-mesh KMP3 integral storage exceeds max_memory; '
                'increase the memory limit or reduce the k-point/orbital space.')
