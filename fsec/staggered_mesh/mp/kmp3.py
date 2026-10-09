"""Canonical restricted MP3 on periodic k-point meshes.

The contraction equations are based on the implementation by Xin Xing.
"""

import time
import numpy as np
import concurrent.futures

from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.mp.kmp2 import (get_nocc, padded_mo_energy, padded_mo_coeff,
                               padding_k_idx, KMP2)
from pyscf.pbc.lib import kpts_helper
from pyscf import __config__
from pyscf.pbc import tools
from pyscf.pbc.cc.kccsd_rhf import _get_epq
from .._kpoints import make_kpts_helper
einsum = lib.einsum

WITH_T2 = getattr(__config__, 'mp_mp3_with_t2', True)


def mp3_amplitude(mp, eris):
    time0 = time1 = logger.perf_counter(), logger.process_clock()
    log = logger.Logger(mp.stdout, mp.verbose)
    time1 = log.timer_debug1('t2 start', *time1)


    #   basic information
    nkpts = mp.nkpts
    nocc = mp.nocc
    nvir = mp.nmo - mp.nocc
    kconserv = mp.khelper.kconserv
    mo_e_o = [e[:nocc] for e in eris.mo_energy]
    mo_e_v = [e[nocc:] for e in eris.mo_energy]
    nonzero_opadding, nonzero_vpadding = padding_k_idx(mp, kind="split")

    #   auxiliary MP2 t-amplitude
    t_mp2 = np.empty((nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir), dtype=eris.dtype)
    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        kb = kconserv[ki, ka, kj]
        eia = _get_epq([0,nocc,ki,mo_e_o,nonzero_opadding],
                       [0,nvir,ka,mo_e_v,nonzero_vpadding],
                       fac=[1.0,-1.0])
        ejb = _get_epq([0,nocc,kj,mo_e_o,nonzero_opadding],
                       [0,nvir,kb,mo_e_v,nonzero_vpadding],
                       fac=[1.0,-1.0])
        eijab = eia[:, None, :, None] + ejb[None, :, None, :]
        t_mp2[ki, kj, ka] = eris.oovv[ki, kj, ka].conj() / eijab
    mp._t2_mp2 = t_mp2
    time1 = log.timer_debug1('t2 MP2', *time1)

    #   MP3 t-amplitude
    t_mp3 = np.zeros_like(t_mp2)

    #   1st term (4h2p):
    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        kb = kconserv[ki, ka, kj]

        t2new_tmp = np.zeros((nocc, nocc, nvir, nvir), dtype=eris.dtype)
        for kl in range(nkpts):
            kk = kconserv[kj, kl, ki]
            t2new_tmp += 0.5 * einsum('klij,klab->ijab', eris.oooo[kk, kl, ki], t_mp2[kk, kl, ka])
            # NOTE: 0.5 from the symmetrization below, this quantity is the same for both

        t_mp3[ki, kj, ka] += t2new_tmp
        t_mp3[kj, ki, kb] += t2new_tmp.transpose(1, 0, 3, 2)
    time1 = log.timer_debug1('t2 4h2p', *time1)


    #   2nd term (2h4p):
    #   "vvvv term" X_{cd}^{ab} t_{ij}^{cd}
    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        kb = kconserv[ki, ka, kj]

        t2new_tmp = np.zeros((nocc, nocc, nvir, nvir), dtype=eris.dtype)
        for kc in range(nkpts):
            t2new_tmp += 0.5 * einsum('abcd,ijcd->ijab', eris.vvvv[ka, kb, kc], t_mp2[ki, kj, kc])

        t_mp3[ki, kj, ka] += t2new_tmp
        t_mp3[kj, ki, kb] += t2new_tmp.transpose(1, 0, 3, 2)
    time1 = log.timer_debug1('t2 2h4p', *time1)

    #   3rd term
    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        kb = kconserv[ki, ka, kj]

        t2new_tmp = np.zeros((nocc, nocc, nvir, nvir), dtype=eris.dtype)
        for kk in range(nkpts):
            kc = kconserv[ka, ki, kk]
            tmp_voov = 2. * eris.voov[ka, kk, ki] - eris.ovov[kk, ka, ki].transpose(1, 0, 2, 3)
            t2new_tmp += einsum('akic,kjcb->ijab', tmp_voov, t_mp2[kk, kj, kc])

            kc = kconserv[ka, ki, kk]
            t2new_tmp -= einsum('akic,kjbc->ijab', eris.voov[ka, kk, ki], t_mp2[kk, kj, kb])

            kc = kconserv[kk, ka, kj]
            t2new_tmp -= einsum('kbic,kjac->ijab', eris.ovov[kk, kb, ki], t_mp2[kk, kj, ka])

        t_mp3[ki, kj, ka] += t2new_tmp
        t_mp3[kj, ki, kb] += t2new_tmp.transpose(1, 0, 3, 2)
    time1 = log.timer_debug1('t2 3h3p', *time1)

    #   Madelung constant correction
    if mp.madelung:
        t_mp3 += 2 * mp.madelung_constant * t_mp2

    #   multiplication by orbital energy fraction
    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        kb = kconserv[ki, ka, kj]
        eia = _get_epq([0,nocc,ki,mo_e_o,nonzero_opadding],
                       [0,nvir,ka,mo_e_v,nonzero_vpadding],
                       fac=[1.0,-1.0])
        ejb = _get_epq([0,nocc,kj,mo_e_o,nonzero_opadding],
                       [0,nvir,kb,mo_e_v,nonzero_vpadding],
                       fac=[1.0,-1.0])
        eijab = eia[:, None, :, None] + ejb[None, :, None, :]
        t_mp3[ki, kj, ka] /= eijab
    time1 = log.timer_debug1('t2 orbital', *time1)

    #   MP2 term
    t_mp3 += t_mp2
    time1 = log.timer_debug1('t2 MP2', *time1)

    time0 = log.timer_debug1('compute MP3 t-amplitude', *time0)
    return t_mp3

def mp3_amplitude_efficient(mp, eris):
    time0 = time1 = logger.perf_counter(), logger.process_clock()
    log = logger.Logger(mp.stdout, mp.verbose)
    time1 = log.timer_debug1('t2 start', *time1)


    #   basic information
    nkpts = mp.nkpts
    nocc = mp.nocc
    nvir = mp.nmo - mp.nocc
    kconserv = mp.khelper.kconserv
    mo_e_o = [e[:nocc] for e in eris.mo_energy]
    mo_e_v = [e[nocc:] for e in eris.mo_energy]
    nonzero_opadding, nonzero_vpadding = padding_k_idx(mp, kind="split")

    #   auxiliary MP2 t-amplitude
    t_mp2 = np.empty((nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir), dtype=eris.dtype)
    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        kb = kconserv[ki, ka, kj]
        eia = _get_epq([0,nocc,ki,mo_e_o,nonzero_opadding],
                       [0,nvir,ka,mo_e_v,nonzero_vpadding],
                       fac=[1.0,-1.0])
        ejb = _get_epq([0,nocc,kj,mo_e_o,nonzero_opadding],
                       [0,nvir,kb,mo_e_v,nonzero_vpadding],
                       fac=[1.0,-1.0])
        eijab = eia[:, None, :, None] + ejb[None, :, None, :]
        t_mp2[ki, kj, ka] = eris.oovv[ki, kj, ka].conj() / eijab
    mp._t2_mp2 = t_mp2
    time1 = log.timer_debug1('t2 MP2', *time1)

    #   MP3 t-amplitude
    t_mp3 = np.zeros_like(t_mp2)

    #   1: 4h2p term
    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        kb = kconserv[ki, ka, kj]
        kk_kl = np.asarray([ [kconserv[kj,kl,ki], kl] for kl in range(nkpts)])
        t2new_tmp = 0.5 * einsum('xklij,xklab->ijab', eris.oooo[kk_kl[:,0], kk_kl[:,1], ki], t_mp2[kk_kl[:,0], kk_kl[:,1], ka])
        t_mp3[ki, kj, ka] += t2new_tmp
        t_mp3[kj, ki, kb] += t2new_tmp.transpose(1, 0, 3, 2)
    time1 = log.timer_debug1('t2 4h2p', *time1)


    #   2: 2h4p term
    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        kb = kconserv[ki, ka, kj]
        t2new_tmp = 0.5 * einsum('xabcd,xijcd->ijab', eris.vvvv[ka,kb,:], t_mp2[ki, kj, :])  # 0.5 from the symmetrization below, this quantity is the same for both
        t_mp3[ki, kj, ka] += t2new_tmp
        t_mp3[kj, ki, kb] += t2new_tmp.transpose(1, 0, 3, 2)
    time1 = log.timer_debug1('t2 2h4p', *time1)

    #   3: 3h3p terms
    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        kb = kconserv[ki, ka, kj]
        t2new_tmp = np.zeros((nocc, nocc, nvir, nvir), dtype=eris.dtype)

        #   term 1 & 2
        kk_kc = np.asarray([ [kk,kconserv[ka, ki, kk]] for kk in range(nkpts) ])
        t2new_tmp = 2*einsum('xakic,xkjcb->ijab', eris.voov[ka, kk_kc[:,0], ki], t_mp2[kk_kc[:,0], kj, kk_kc[:,1]])
        t2new_tmp -= einsum('xakic,xkjcb->ijab', eris.ovov[kk_kc[:,0], ka, ki].transpose(0, 2, 1, 3, 4), t_mp2[kk_kc[:,0], kj, kk_kc[:,1]])

        #   term 3
        t2new_tmp -= einsum('xakic,xkjbc->ijab', eris.voov[ka, :, ki], t_mp2[:, kj, kb])

        #   term 4
        t2new_tmp -= einsum('xkbic,xkjac->ijab', eris.ovov[:, kb, ki], t_mp2[:, kj, ka])

        t_mp3[ki, kj, ka] += t2new_tmp
        t_mp3[kj, ki, kb] += t2new_tmp.transpose(1, 0, 3, 2)
    time1 = log.timer_debug1('t2 3h3p', *time1)

    #   Madelung constant correction
    if mp.madelung:
        t_mp3 += 2 * mp.madelung_constant * t_mp2

    #   multiplication by orbital energy fraction
    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        kb = kconserv[ki, ka, kj]
        eia = _get_epq([0,nocc,ki,mo_e_o,nonzero_opadding],
                       [0,nvir,ka,mo_e_v,nonzero_vpadding],
                       fac=[1.0,-1.0])
        ejb = _get_epq([0,nocc,kj,mo_e_o,nonzero_opadding],
                       [0,nvir,kb,mo_e_v,nonzero_vpadding],
                       fac=[1.0,-1.0])
        eijab = eia[:, None, :, None] + ejb[None, :, None, :]
        t_mp3[ki, kj, ka] /= eijab
    time1 = log.timer_debug1('t2 orbital', *time1)

    #   MP2 term
    t_mp3 += t_mp2
    time1 = log.timer_debug1('t2 MP2', *time1)
    time0 = log.timer_debug1('compute MP3 t-amplitude', *time0)
    return t_mp3

def mp3_energy(mp, t2, eris):
    nkpts = mp.nkpts
    kconserv = mp.khelper.kconserv
    e = 0.0 + 1j * 0.0
    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        kb = kconserv[ki, ka, kj]
        e += 2 * einsum('ijab,ijab', t2[ki, kj, ka], eris.oovv[ki, kj, ka])
        e += -einsum('ijab,ijba', t2[ki, kj, ka], eris.oovv[ki, kj, kb])
    e /= nkpts
    if abs(e.imag) > 1e-4:
        logger.warn(mp, 'Non-zero imaginary part found in MP3 energy %s', e)
    return e.real

class KMP3(KMP2):
    """Restricted periodic MP3 for an ordinary uniform k-point mesh.

    ``e_corr_mp2`` is the second-order energy, ``e_corr_mp3`` is the third-
    order increment, and ``e_corr`` is their sum.  ``t2`` contains the first-
    order doubles amplitudes, while ``t2_mp3`` retains the combined amplitudes
    used to evaluate the energy.
    """

    def __init__(self, mf, frozen=None, mo_energy=None,
                 mo_coeff=None, mo_occ=None, madelung=None):

        super(KMP3, self).__init__(mf, frozen, mo_coeff, mo_occ)

        # Retain the actual k-point ordering even when it is a complete but
        # noncanonical mesh (as in concatenated shifted extension meshes).
        self.khelper = make_kpts_helper(self._scf.cell, self.kpts)

        if mo_energy is None: mo_energy = mf.mo_energy
        self.mo_energy = mo_energy
        # This correction removes the singularity shift used by the HF
        # reference and is therefore enabled by default only for Ewald HF.
        if madelung is None:
            madelung = getattr(mf, 'exxdiv', None) == 'ewald'
        self.madelung = bool(madelung)
        self.madelung_constant = -tools.madelung(self._scf.cell, self.kpts)
        self.e_corr_mp2 = None
        self.e_corr_mp3 = None
        self.t2_mp3 = None
        self._t2_mp2 = None

    # double_amplitude = mp3_amplitude
    double_amplitude = mp3_amplitude_efficient
    correlation_energy = mp3_energy

    @property
    def emp2(self):
        """Return the second-order component, as in the PySCF MP3 API."""
        return self.e_corr_mp2

    def _finalize(self):
        """Log MP3 components without invoking MP2 SCS finalization."""
        log = logger.new_logger(self)
        log.note('E(%s) = %.15g  E_corr = %.15g',
                 self.__class__.__name__, self.e_tot, self.e_corr)
        log.note('E_corr(MP2) = %.15g  E_corr(MP3 increment) = %.15g',
                 self.e_corr_mp2, self.e_corr_mp3)
        return self

    def kernel(self, mo_energy=None, mo_coeff=None, eris=None,
               with_t2=WITH_T2):
        """Compute the MP2 and MP3 correlation energies.

        Returns ``(e_corr, t2)``.  ``t2`` is the first-order MP2 amplitude;
        the combined first- plus second-order amplitude is available as
        ``t2_mp3`` for inspection.
        """
        self.e_hf = self._scf.e_tot

        if mo_energy is None:
            mo_energy = self.mo_energy
        if mo_coeff is None:
            mo_coeff = self.mo_coeff

        #   construct required integrals
        if eris is None:
            eris = self.ao2mo(mo_coeff, mo_energy)
        self.eris = eris

        #   MP3 energy calculation
        self.dump_flags()
        t2_mp3 = self.double_amplitude(eris)
        t2_mp2 = self._t2_mp2
        self.e_corr_mp2 = self.correlation_energy(t2_mp2, eris)
        self.e_corr = self.correlation_energy(t2_mp3, eris)
        self.e_corr_mp3 = self.e_corr - self.e_corr_mp2
        self.e_corr = self.e_corr_mp2 + self.e_corr_mp3

        if with_t2:
            self.t2 = t2_mp2
            self.t2_mp3 = t2_mp3
        else:
            self.t2 = None
            self.t2_mp3 = None
            self._t2_mp2 = None

        self._finalize()
        return self.e_corr, self.t2

    def ao2mo(self, mo_coeff=None, mo_energy=None):
        return _ERIS_MP3(self, mo_coeff, mo_energy)

    def dump_flags(self):
        logger.info(self, "")
        logger.info(self, "******** %s ********", self.__class__)
        logger.info(self, "nkpts = %d", self.nkpts)
        logger.info(self, "nocc = %d", self.nocc)
        logger.info(self, "nmo = %d", self.nmo)
        if self.frozen is not None:
            logger.info(self, "frozen orbitals = %s", self.frozen)
        logger.info(
            self,
            "max_memory %d MB (current use %d MB)",
            self.max_memory,
            lib.current_memory()[0],
        )
        return self

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

class _ERIS_MP3:
    def __init__(self, cc, mo_coeff=None, mo_energy=None, method='incore'):
        from pyscf.pbc import df

        log = logger.Logger(cc.stdout, cc.verbose)
        cput0 = (logger.process_clock(), logger.perf_counter())
        cell = cc._scf.cell
        kpts = cc.kpts
        nkpts = cc.nkpts
        nocc = cc.nocc
        nmo = cc.nmo
        nvir = nmo - nocc
        max_memory = cc.max_memory

        # Get mo_coeff and mo_energy
        if mo_coeff is None:
            mo_coeff = cc.mo_coeff
        if mo_energy is None:
            mo_energy = cc.mo_energy
        mo_coeff = self.mo_coeff = padded_mo_coeff(cc, mo_coeff)
        mo_energy = self.mo_energy = padded_mo_energy(cc, mo_energy)
        self.dtype = mo_coeff[0].dtype

        # Get location of padded elements in occupied and virtual space.
        nocc_per_kpt = get_nocc(cc, per_kpoint=True)
        nonzero_padding = padding_k_idx(cc, kind="joint")

        # Check direct and indirect gaps for possible issues with MP3.
        mo_e = [self.mo_energy[kp][nonzero_padding[kp]] for kp in range(nkpts)]
        mo_e = np.sort([y for x in mo_e for y in x])  # Sort de-nested array
        gap = mo_e[np.sum(nocc_per_kpt)] - mo_e[np.sum(nocc_per_kpt)-1]
        if gap < 1e-5:
            logger.warn(cc, 'HOMO-LUMO gap %s too small for MP3.', gap)

        mem_incore, mem_outcore, mem_basic = _mem_usage(nkpts, nocc, nvir)
        mem_now = lib.current_memory()[0]
        fao2mo = cc._scf.with_df.ao2mo

        kconserv = cc.khelper.kconserv
        khelper = cc.khelper

        log.info("ERI memory estimate %.1f MB; current use %.1f MB; limit %.1f MB",
                 mem_incore, mem_now, max_memory)
        if (method == 'incore' and (mem_incore + mem_now < max_memory)
                or cell.incore_anyway):
            log.info('Using incore ERI storage')
            self.oooo = np.empty((nkpts, nkpts, nkpts, nocc, nocc, nocc, nocc), dtype=self.dtype)
            self.oovv = np.empty((nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir), dtype=self.dtype)
            self.ovov = np.empty((nkpts, nkpts, nkpts, nocc, nvir, nocc, nvir), dtype=self.dtype)
            self.voov = np.empty((nkpts, nkpts, nkpts, nvir, nocc, nocc, nvir), dtype=self.dtype)
            self.vvvv = np.empty((nkpts, nkpts, nkpts, nvir, nvir, nvir, nvir), dtype=self.dtype)

            start = time.time()
            key_list = list(khelper.symm_map.keys())
            def eri_kpt_calc(idx):
                (kp, kr, kq) = key_list[idx]
                ks = kconserv[kp, kr, kq]

                #   Calculate the ERIs
                eri_kpt = fao2mo((mo_coeff[kp],mo_coeff[kr],mo_coeff[kq],mo_coeff[ks]),
                                (kpts[kp],kpts[kr],kpts[kq],kpts[ks]), compact=False)

                if self.dtype == float: eri_kpt = eri_kpt.real
                eri_kpt = eri_kpt.reshape(nmo, nmo, nmo, nmo)
                return eri_kpt

            eri_kpt = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for idx in range(len(key_list)):
                    eri_kpt.append( executor.submit(eri_kpt_calc, idx) )

            for idx, (kpp, krr, kqq) in enumerate(key_list):
                eri_result = eri_kpt[idx].result()
                for (kp, kr, kq) in khelper.symm_map[(kpp, krr, kqq)]:
                    eri_kpt_symm = khelper.transform_symm(eri_result, kp, kr, kq).transpose(0, 2, 1, 3)
                    ks = kconserv[kp, kr, kq]
                    self.oooo[kp, kq, kr] = eri_kpt_symm[:nocc, :nocc, :nocc, :nocc] / nkpts
                    self.oovv[kp, kq, kr] = eri_kpt_symm[:nocc, :nocc, nocc:, nocc:] / nkpts
                    self.ovov[kp, kq, kr] = eri_kpt_symm[:nocc, nocc:, :nocc, nocc:] / nkpts
                    self.voov[kp, kq, kr] = eri_kpt_symm[nocc:, :nocc, :nocc, nocc:] / nkpts
                    self.vvvv[kp, kq, kr] = eri_kpt_symm[nocc:, nocc:, nocc:, nocc:] / nkpts

            log.info("ERI evaluation time: %.3f" % (time.time() - start))

        else:
            raise MemoryError(
                'In-core KMP3 integral storage exceeds max_memory; increase '
                'the memory limit or reduce the k-point/orbital space.')

        log.timer('MP3 integral transformation', *cput0)
