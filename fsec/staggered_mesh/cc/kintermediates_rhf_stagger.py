#!/usr/bin/env python

# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
# Authors: Xin Xing

import itertools
import numpy as np

from itertools import product
from pyscf import lib
from pyscf.lib import logger
from pyscf.lib.parameters import LOOSE_ZERO_TOL, LARGE_DENOM  # noqa
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.mp.kmp2 import (get_frozen_mask, get_nocc, get_nmo,
                               padded_mo_coeff, padding_k_idx)  # noqa

einsum = lib.einsum

# This is restricted (R)CCD for staggered mesh method
# Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004)


### Eqs. (37)-(38) "kappa" == "lambda" in CCD
#   NOTE: (1) assume diagonal Fock matrix
#         (2) Fock diagonal entries are not included in kappa

def cc_Foo_stagger(cc, t1, t2, eris, kconserv):
    nkpts_ov, nocc, nvir = t1.shape
    Foo = np.zeros((nkpts_ov,nocc,nocc),dtype=eris.dtype)
    for iki, ikl, ikc in kpts_helper.loop_kkk(nkpts_ov):
        ki, kl, kc = cc.kpts_idx_occ[iki], cc.kpts_idx_occ[ikl], cc.kpts_idx_vir[ikc]
        ikk, kk = iki, ki
        kd = kconserv[kk,kc,kl]
        ikd = cc.idx_vir[kd]

        Soovv = 2*eris.oovv[ikk,ikl,ikc] - eris.oovv[ikk,ikl,ikd].transpose(0,1,3,2)
        Foo[iki] += einsum('klcd,ilcd->ki', Soovv, t2[iki,ikl,ikc])
    return Foo

def cc_Fvv_stagger(cc, t1, t2, eris, kconserv):
    nkpts_ov, nocc, nvir = t1.shape
    Fvv = np.zeros((nkpts_ov,nvir,nvir),dtype=eris.dtype)
    for ika, ikl, ikk in kpts_helper.loop_kkk(nkpts_ov):
        ka, kl, kk = cc.kpts_idx_vir[ika], cc.kpts_idx_occ[ikl], cc.kpts_idx_occ[ikk]
        ikc, kc = ika, ka
        kd = kconserv[kk,kc,kl]
        ikd = cc.idx_vir[kd]

        Soovv = 2*eris.oovv[ikk,ikl,ikc] - eris.oovv[ikk,ikl,ikd].transpose(0,1,3,2)
        Fvv[ika] += -einsum('klcd,klad->ac', Soovv, t2[ikk,ikl,ika])
    return Fvv


### Eqs. (42)-(45) "chi"
def cc_Woooo_stagger(cc, t1, t2, eris, kconserv, out=None):
    nkpts, nocc, nvir = t1.shape
    Wklij = _new(eris.oooo.shape, eris.dtype, out)
    for ikk in range(nkpts):
        for ikl in range(ikk+1):
            for iki in range(nkpts):
                kk, kl, ki = cc.kpts_idx_occ[ikk], cc.kpts_idx_occ[ikl], cc.kpts_idx_occ[iki]
                kj = kconserv[kk,ki,kl]
                ikj = cc.idx_occ[kj]

                tmp_oooo = eris.oooo[ikk,ikl,iki].copy()
                vvoo = eris.oovv[ikk,ikl].transpose(0,3,4,1,2).reshape(nkpts*nvir,nvir,nocc,nocc)
                t2t  = t2[iki,ikj].transpose(0,3,4,1,2).reshape(nkpts*nvir,nvir,nocc,nocc)
                tmp_oooo += einsum('cdkl,cdij->klij', vvoo, t2t)
                # for ikc in range(nkpts):
                #     tmp_oooo += einsum('klcd,ijcd->klij',eris.oovv[ikk,ikl,ikc],t2[iki,ikj,ikc])

                Wklij[ikk,ikl,iki] = np.copy(tmp_oooo)
                # Wklij[ikk,ikl,iki] = eris.oooo[ikk,ikl,iki].copy()

        for ikl in range(ikk+1):
            for iki in range(nkpts):
                kk, kl, ki = cc.kpts_idx_occ[ikk], cc.kpts_idx_occ[ikl], cc.kpts_idx_occ[iki]
                kj = kconserv[kk,ki,kl]
                ikj = cc.idx_occ[kj]
                Wklij[ikl,ikk,ikj] = Wklij[ikk,ikl,iki].transpose(1,0,3,2)
    return Wklij

def cc_Wvvvv_stagger(cc, t1, t2, eris, kconserv, out=None):
    Wabcd = _new(eris.vvvv.shape, eris.dtype, out)
    Wabcd = eris.vvvv.copy()

    return Wabcd

def cc_Wvoov_stagger(cc, t1, t2, eris, kconserv, out=None):
    nkpts, nocc, nvir = t1.shape
    Wakic = _new(eris.voov.shape, eris.dtype, out)
    for ika, ka in enumerate(cc.kpts_idx_vir):
        for ikk, kk in enumerate(cc.kpts_idx_occ):
            voov_i = np.copy(eris.voov[ika,ikk,:])
            for iki, ki in enumerate(cc.kpts_idx_occ):
                kc = kconserv[ka,ki,kk]
                ikc = cc.idx_vir[kc]

                oovv_tmp = eris.oovv[ikk,:,ikc] #<kl|cd>
                voov_i[iki] -= 0.5*einsum('xklcd,xliad->akic', oovv_tmp, t2[:,iki,ika])

                Soovv_tmp = 2*oovv_tmp - eris.oovv[:,ikk,ikc].transpose(0,2,1,3,4)
                voov_i[iki] += 0.5*einsum('xklcd,xilad->akic', Soovv_tmp, t2[iki,:,ika])

            Wakic[ika,ikk,:] = np.copy(voov_i)
            # Wakic[ika,ikk,:] = np.copy(eris.voov[ika,ikk,:])
    return Wakic

def cc_Wvovo_stagger(cc, t1, t2, eris, kconserv, out=None):
    nkpts, nocc, nvir = t1.shape
    Wakci = _new((nkpts,nkpts,nkpts,nvir,nocc,nvir,nocc), eris.dtype, out)
    for ika, ikk, ikc in kpts_helper.loop_kkk(nkpts):
        ka, kk, kc = cc.kpts_idx_vir[ika], cc.kpts_idx_occ[ikk], cc.kpts_idx_vir[ikc]
        ki = kconserv[ka,kc,kk]
        iki = cc.idx_occ[ki]

        # vovo_tmp = np.asarray(eris.ovov[ikk,ika,iki]).transpose(1,0,3,2)
        # vovo_tmp = eris.ovov[ikk,ika,iki].transpose(1,0,3,2)
        vovo_tmp = np.copy( eris.ovov[ikk,ika,iki].transpose(1,0,3,2) )
        oovvf = eris.oovv[:,ikk,ikc].reshape(nkpts*nocc,nocc,nvir,nvir)
        t2f   = t2[:,iki,ika].reshape(nkpts*nocc,nocc,nvir,nvir)
        vovo_tmp -= 0.5*einsum('lkcd,liad->akci',oovvf,t2f)
        # vovo_tmp += eris.ovov[ikk,ika,iki].transpose(1,0,3,2)
        # vovo_tmp = eris.ovov[ikk,ika,iki].transpose(1,0,3,2) - 0.5*einsum('lkcd,liad->akci',oovvf,t2f)
        Wakci[ika,ikk,ikc] = np.copy( vovo_tmp)
        # Wakci[ika,ikk,ikc] = np.copy( eris.ovov[ikk,ika,iki].transpose(1,0,3,2) )
    return Wakci



#   3mesh version
def cc_Foo_3mesh(cc, t1, t2, eris, kconserv):
    _, nkpts, nocc, nvir = t1.shape
    Foo = {}
    for mi in range(3):
        Foo[mi] = np.zeros((nkpts,nocc,nocc),dtype=eris.dtype)
        for ml, mc, md in cc.dependence_kappa[mi][0]:
            for iki, ikl, ikc in kpts_helper.loop_kkk(nkpts):
                ki, kl, kc = cc.kpts_base_idx[mi][iki], cc.kpts_base_idx[ml][ikl], cc.kpts_base_idx[mc][ikc]
                mk, ikk, kk = mi, iki, ki
                kd = kconserv[kk,kc,kl]
                ikd = cc.idx_base[md][kd]

                Soovv = 2*eris.oovv[cc.mesh2idx(mk, ml, mc)][ikk,ikl,ikc] - eris.oovv[cc.mesh2idx(mk, ml, md)][ikk,ikl,ikd].transpose(0,1,3,2)
                Foo[mi][iki] += einsum('klcd,ilcd->ki', Soovv, cc.amp_3mesh(t2, (mi,ml,mc,md), (iki,ikl,ikc,ikd)) )
        Foo[mi] /= len(cc.dependence_kappa[mi][0])
    return Foo

def cc_Fvv_3mesh(cc, t1, t2, eris, kconserv):
    _, nkpts, nocc, nvir = t1.shape
    Fvv = {}
    for ma in range(3):
        Fvv[ma] = np.zeros((nkpts,nvir, nvir),dtype=eris.dtype)
        for mk, ml, md in cc.dependence_kappa[ma][1]:
            for ikk, ikl, ika in kpts_helper.loop_kkk(nkpts):
                kk, kl, ka = cc.kpts_base_idx[mk][ikk], cc.kpts_base_idx[ml][ikl], cc.kpts_base_idx[ma][ika]
                mc, ikc, kc = ma, ika, ka
                kd = kconserv[kk,kc,kl]
                ikd = cc.idx_base[md][kd]

                Soovv = 2*eris.oovv[cc.mesh2idx(mk, ml, mc)][ikk,ikl,ikc] - eris.oovv[cc.mesh2idx(mk, ml, md)][ikk,ikl,ikd].transpose(0,1,3,2)
                Fvv[ma][ika] += -einsum('klcd,klad->ac', Soovv, cc.amp_3mesh(t2, (mk,ml,ma,md), (ikk,ikl,ika,ikd)) )
        Fvv[ma] /= len(cc.dependence_kappa[ma][1])
    return Fvv

def cc_Woooo_3mesh(cc, t1, t2, eris, kconserv):
    namp, nkpts, nocc, nvir = t1.shape
    Wklij = {}
    for mk, ml, mi in itertools.product(range(3), range(3), range(3)):
        mj = (mk + ml - mi) % 3
        if mk == mi or mk > ml or (mk == ml and mi > mj):
            continue
        tmp_klij = np.empty( (nkpts, nkpts, nkpts, nocc, nocc, nocc, nocc), dtype=eris.dtype)
        tmp_lkji = np.empty( (nkpts, nkpts, nkpts, nocc, nocc, nocc, nocc), dtype=eris.dtype)
        mc, md = cc.dependence_imb[cc.mesh2idx(mk, ml, mi)][0]

        for ikk, ikl, iki in itertools.product(range(nkpts), range(nkpts), range(nkpts)):
            kk, kl, ki = cc.kpts_base_idx[mk][ikk], cc.kpts_base_idx[ml][ikl], cc.kpts_base_idx[mi][iki]
            kj = kconserv[kk,ki,kl]
            ikj = cc.idx_base[mj][kj]

            tmp_oooo = eris.oooo[cc.mesh2idx(mk, ml, mi)][ikk,ikl,iki].copy()
            for ikc, kc in enumerate(cc.kpts_base_idx[mc]):
                kd = kconserv[ki,kc,kj]
                ikd = cc.idx_base[md][kd]
                tmp_oooo += einsum(
                    'klcd,ijcd->klij',
                    eris.oovv[cc.mesh2idx(mk, ml, mc)][ikk, ikl, ikc],
                    cc.amp_3mesh(t2, (mi, mj, mc, md), (iki, ikj, ikc, ikd)),
                )

            tmp_klij[ikk,ikl,iki] = np.copy(tmp_oooo)
            tmp_lkji[ikl,ikk,ikj] = np.copy(tmp_oooo.transpose(1,0,3,2))
        Wklij[cc.mesh2idx(mk, ml, mi)] = tmp_klij
        Wklij[cc.mesh2idx(ml, mk, mj)] = tmp_lkji
    return Wklij

def cc_Wvvvv_3mesh(cc, t1, t2, eris, kconserv):
    Wabcd = eris.vvvv
    return Wabcd

def cc_Wvoov_3mesh(cc, t1, t2, eris, kconserv):
    _, nkpts, nocc, nvir = t1.shape
    Wakic = {}
    for ma, mk, mi in itertools.product(range(3), range(3), range(3)):
        if mk == mi or ma == mi:
            continue
        mc = (ma + mk - mi) % 3
        ml, md = cc.dependence_imb[cc.mesh2idx(ma,mk,mi)][2]

        tmp_akic = np.empty( (nkpts, nkpts, nkpts, nvir, nocc, nocc, nvir), dtype=eris.dtype)
        for ika, ikk, iki in kpts_helper.loop_kkk(nkpts):
            ka, kk, ki = cc.kpts_base_idx[ma][ika], cc.kpts_base_idx[mk][ikk], cc.kpts_base_idx[mi][iki]
            kc = kconserv[ka,ki,kk]
            ikc = cc.idx_base[mc][kc]

            voov_tmp = np.copy(eris.voov[cc.mesh2idx(ma, mk, mi)][ika,ikk,iki])
            for ikl, kl in enumerate(cc.kpts_base_idx[ml]):
                kd = kconserv[kk, kc, kl]
                ikd = cc.idx_base[md][kd]

                oovv_tmp = eris.oovv[cc.mesh2idx(mk,ml,mc)][ikk, ikl, ikc]
                voov_tmp -= 0.5*einsum('klcd,liad->akic', oovv_tmp, cc.amp_3mesh(t2, (ml,mi,ma,md), (ikl,iki,ika,ikd)) )
                Soovv_tmp = 2*oovv_tmp - eris.oovv[cc.mesh2idx(ml,mk,mc)][ikl,ikk,ikc].transpose(1,0,2,3)
                voov_tmp += 0.5*einsum('klcd,ilad->akic', Soovv_tmp, cc.amp_3mesh(t2, (mi,ml,ma,md), (iki,ikl,ika,ikd)) )
            tmp_akic[ika,ikk,iki] = np.copy(voov_tmp)
        Wakic[cc.mesh2idx(ma, mk, mi)] = tmp_akic
        # Wakic[cc.mesh2idx(ma, mk, mc)] = np.zeros( (nkpts, nkpts, nkpts, nvir, nocc, nocc, nvir), dtype=eris.dtype)
    return Wakic

def cc_Wvovo_3mesh(cc, t1, t2, eris, kconserv):
    _, nkpts, nocc, nvir = t1.shape
    Wakci = {}

    for ma, mk, mc in itertools.product(range(3), range(3), range(3)):
        if ma == mc:
            continue
        mi = (ma + mk - mc) % 3
        ml, md = cc.dependence_imb[cc.mesh2idx(ma,mk,mc)][3]

        tmp_akci = np.empty( (nkpts, nkpts, nkpts, nvir, nocc, nvir, nocc), dtype=eris.dtype)
        for ika, ikk, ikc in kpts_helper.loop_kkk(nkpts):
            ka, kk, kc = cc.kpts_base_idx[ma][ika], cc.kpts_base_idx[mk][ikk], cc.kpts_base_idx[mc][ikc]
            ki = kconserv[ka,kc,kk]
            iki = cc.idx_base[mi][ki]

            vovo_tmp = np.copy(eris.ovov[cc.mesh2idx(mk, ma, mi)][ikk,ika,iki].transpose(1,0,3,2))
            for ikl, kl in enumerate(cc.kpts_base_idx[ml]):
                kd = kconserv[kl, kc, kk]
                ikd = cc.idx_base[md][kd]
                vovo_tmp -= 0.5*einsum('lkcd,ilda->akci', eris.oovv[cc.mesh2idx(ml,mk,mc)][ikl,ikk,ikc], cc.amp_3mesh(t2, (mi,ml,md,ma), (iki,ikl,ikd,ika)) )
            tmp_akci[ika,ikk,ikc] = np.copy(vovo_tmp)
        Wakci[cc.mesh2idx(ma, mk, mc)] = tmp_akci
        # Wakci[cc.mesh2idx(ma, mk, mc)] = np.zeros( (nkpts, nkpts, nkpts, nvir, nocc, nvir, nocc), dtype=eris.dtype)
    return Wakci


def _new(shape, dtype, out):
    if out is None: # Incore:
        out = np.empty(shape, dtype=dtype)
    else:
        assert(out.shape == shape)
        assert(out.dtype == dtype)
    return out
