#!/usr/bin/env python
# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
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
# Adapted from pyscf.pbc.mp.kmp2_stagger by Xin Xing (xxing@berkeley.edu).

"""Restricted periodic MP2 with two staggered k-point meshes.

Reference: Staggered Mesh Method for Correlation Energy Calculations of Solids:
Second-Order Møller–Plesset Perturbation Theory, J. Chem. Theory Comput. 2021,
17, 8, 4733-4745.

This matches ``pyscf.pbc.mp.kmp2_stagger`` by default, except that (L|pq) is read
with ``sr_loop``. ``CDERIArray`` unpacks every k-point pair with the auxiliary
count of the first pair, which corrupts the integrals when linear dependency
makes that count vary with q.

It can also evaluate the fixed-density band Fock matrix on the shifted mesh with
density-fitted smoothed truncated Coulomb (sTC) exchange and ordinary GDF
Hartree, instead of a temporary FFTDF with hard ``vcut_sph`` truncation. sTC
applies only to the orbitals and orbital energies: the correlation integrals
stay bare. The sTC path requires the custom ``pyscf.pbc.df.rsdf_stc`` module.
"""

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.lib.parameters import LARGE_DENOM
from pyscf.ao2mo import _ao2mo
from pyscf.pbc import df, scf
from pyscf.pbc.dft import krks
from pyscf.pbc.scf.khf import get_occ
from pyscf.pbc.mp import kmp2
from pyscf.pbc.mp.kmp2 import padding_k_idx, _add_padding
from pyscf.pbc.lib.kpts_helper import round_to_fbz
from pyscf.pbc.tools.pbc import get_monkhorst_pack_size

from .._kpoints import make_kpts_helper

#   Real-space cutoff that pairs with each sTC exchange kernel
_STC_RC_TYPE = {'vcut_ws': 'ws', 'vcut_sph': 'sph'}


#   Kernel function for computing the MP2 energy
def kernel(mp, mo_energy, mo_coeff, verbose=logger.NOTE):
    """Computes k-point RMP2 energy using the staggered mesh method

    Args:
        mp (KMP2_stagger): an instance of KMP2_stagger
        mo_energy (list): a list of numpy.ndarray. Each array contains MO energies of
                          shape (Nmo,) for one kpt
        mo_coeff (list): a list of numpy.ndarray. Each array contains MO coefficients
                         of shape (Nao, Nmo) for one kpt
        verbose (int, optional): level of verbosity. Defaults to logger.NOTE (=3).

    Returns:
        KMP2 energy in hartree per primitive cell
    """
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mp, verbose)

    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    nkpts = mp.nkpts
    nkpts_ov = mp.nkpts_ov
    kconserv = mp.khelper.kconserv

    mem_avail = mp.max_memory - lib.current_memory()[0]
    mem_usage = (nkpts_ov * (nocc * nvir)**2) * 16 / 1e6
    if mp.with_df_ints:
        with_df = mp._scf.with_df if mp.with_df is None else mp.with_df
        naux = with_df.auxcell.nao_nr()
        mem_usage += (nkpts_ov**2 * naux * nocc * nvir) * 16 / 1e6
    if mem_usage > mem_avail:
        raise MemoryError('Insufficient memory! MP2 memory usage %d MB (currently available %d MB)'
                          % (mem_usage, mem_avail))

    if mp.with_df_ints:
        Lov = _init_mp_df_eris_stagger(mp)
    else:
        with_df = df.FFTDF(mp.cell, mp.kpts)
        fao2mo = with_df.ao2mo

    #   info of occupied and virtual meshes
    nkpts_ov     = mp.nkpts_ov          #   number of points on the occupied/virtual mesh
    kpts_idx_vir = mp.kpts_idx_vir
    kpts_idx_occ = mp.kpts_idx_occ

    eia   = np.zeros((nocc,nvir))
    eijab = np.zeros((nocc,nocc,nvir,nvir))

    emp2     = 0.0
    oovv_ij  = np.zeros((nkpts_ov,nocc,nocc,nvir,nvir), dtype=mo_coeff[0].dtype)

    mo_e_o = [mo_energy[k][:nocc] for k in range(nkpts)]
    mo_e_v = [mo_energy[k][nocc:] for k in range(nkpts)]

    #   get location of non-zero/padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = padding_k_idx(mp, kind="split")

    for iki, ki in enumerate(kpts_idx_occ):
        for ikj, kj in enumerate(kpts_idx_occ):
            for ika, ka in enumerate(kpts_idx_vir):
                kb = kconserv[ki,ka,kj]
                ikb = np.where(kpts_idx_vir == kb)[0][0]

                if mp.with_df_ints:
                    oovv_ij[ika] = (1./nkpts_ov) * lib.einsum(
                        "Lia,Ljb->iajb",
                        Lov[iki, ika], Lov[ikj, ikb]
                    ).transpose(0,2,1,3)
                else:
                    orbo_i = mo_coeff[ki][:,:nocc]
                    orbo_j = mo_coeff[kj][:,:nocc]
                    orbv_a = mo_coeff[ka][:,nocc:]
                    orbv_b = mo_coeff[kb][:,nocc:]
                    oovv_ij[ika] = fao2mo(
                        (orbo_i,orbv_a,orbo_j,orbv_b),
                        (mp.kpts[ki],mp.kpts[ka],mp.kpts[kj],mp.kpts[kb]),
                        compact=False
                    ).reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3) / nkpts_ov
            for ika, ka in enumerate(kpts_idx_vir):
                kb = kconserv[ki,ka,kj]
                ikb = np.where(kpts_idx_vir == kb)[0][0]

                #   remove zero/padded elements from denominator
                eia = LARGE_DENOM * np.ones((nocc, nvir), dtype=mo_energy[0].dtype)
                n0_ovp_ia = np.ix_(nonzero_opadding[ki], nonzero_vpadding[ka][:nvir])
                eia[n0_ovp_ia] = (mo_e_o[ki][:,None] - mo_e_v[ka])[n0_ovp_ia]

                ejb = LARGE_DENOM * np.ones((nocc, nvir), dtype=mo_energy[0].dtype)
                n0_ovp_jb = np.ix_(nonzero_opadding[kj], nonzero_vpadding[kb][:nvir])
                ejb[n0_ovp_jb] = (mo_e_o[kj][:,None] - mo_e_v[kb])[n0_ovp_jb]

                #   energy calculation
                eijab = lib.direct_sum('ia,jb->ijab',eia,ejb)
                t2_ijab = np.conj(oovv_ij[ika]/eijab)
                woovv = 2*oovv_ij[ika] - oovv_ij[ikb].transpose(0,1,3,2)
                emp2 += np.einsum('ijab,ijab', t2_ijab, woovv).real

    log.timer("KMP2_stagger", *cput0)
    emp2 /= nkpts_ov
    return emp2


def _load_Lpq_ao(with_df, kpti, kptj):
    """Load (L|pq) for one k-point pair, of shape (naux, nao**2).

    naux depends on the pair if linear dependency is found in the auxiliary
    basis. Only the positive part of the metric is read, which is complete
    outside 2D, where the caller raises.
    """
    kpti_kptj = np.asarray((kpti, kptj))
    return np.vstack([LpqR + 1j * LpqI for LpqR, LpqI, _ in
                      with_df.sr_loop(kpti_kptj, compact=False)])


def _init_mp_df_eris_stagger(mp):
    """
    Compute 3-center electron repulsion integrals, i.e. (L|ov),
    where `L` denotes DF auxiliary basis functions and `o` and `v` occupied and virtual
    canonical crystalline orbitals that lie on two staggered uniform meshes.
    Note that `o` and `v` contain kpt indices `ko` and `kv`,
    and the third kpt index `kL` is determined by the conservation of momentum.

    Arguments:
        mp (KMP2_stagger) -- A KMP2_stagger instance

    Returns:
        Lov (numpy.ndarray) -- 3-center DF ints, with shape (nkpts_ov, nkpts_ov, naux, nocc, nvir)
    """

    log = logger.Logger(mp.stdout, mp.verbose)

    cell = mp._scf.cell
    if cell.dimension == 2:
        # 2D ERIs are not positive definite. The 3-index tensors are stored in
        # two part. One corresponds to the positive part and one corresponds
        # to the negative part. The negative part is not considered in the
        # DF-driven CCSD implementation.
        raise NotImplementedError

    if mp.with_df is not None:
        #   Bare integrals prepared by the constructor.
        with_df = mp.with_df

    elif mp._scf.with_df._cderi is not None and mp.flag_submesh is True:
        #   When using submeshes for staggered mesh calculation, the 3c DF tensor from mean-field
        #   calculation can be used directly.
        with_df = mp._scf.with_df

    else:
        with_df = df.GDF(mp.cell, mp.kpts).build()

    nocc = mp.nocc
    nmo = mp.nmo
    nvir = nmo - nocc

    #   info of occupied and virtual meshes
    nkpts_ov     = mp.nkpts_ov          #   number of points on the occupied/virtual mesh
    kpts_idx_vir = mp.kpts_idx_vir
    kpts_idx_occ = mp.kpts_idx_occ

    mo_coeff = _add_padding(mp, mp.mo_coeff, mp.mo_energy)[0]
    dtype = np.result_type(np.complex128, *mo_coeff)

    Lov = np.empty((nkpts_ov, nkpts_ov), dtype=object)

    cput0 = (logger.process_clock(), logger.perf_counter())
    bra_start = 0
    bra_end = nocc
    ket_start = nmo+nocc
    ket_end = ket_start + nvir
    tao = []
    ao_loc = None
    for iki, ki in enumerate(kpts_idx_occ):
        for ika, ka in enumerate(kpts_idx_vir):
            Lpq_ao = _load_Lpq_ao(with_df, mp.kpts[ki], mp.kpts[ka])

            mo = np.hstack((mo_coeff[ki], mo_coeff[ka]))
            mo = np.asarray(mo, dtype=dtype, order='F')

            out = _ao2mo.r_e2(Lpq_ao, mo, (bra_start, bra_end, ket_start, ket_end), tao, ao_loc)
            Lov[iki, ika] = out.reshape(-1, nocc, nvir)

    log.timer_debug1("transforming DF-MP2 integrals", *cput0)

    return Lov


def _is_stc_reference(mf):
    """Return whether ``mf`` uses density-fitted, exchange-only sTC integrals."""
    with_df = mf.with_df
    return (isinstance(with_df, df.df.GDF)
            and getattr(with_df, 'supports_post_hf', True) is False)


def _get_bands_stc(mp, mf, kpts_occ, kpts):
    """Evaluate fixed-density bands with sTC exchange and ordinary GDF Hartree.

    The density stays on ``mf.kpts``, which also defines the sTC cutoff. The
    reference ``mf.with_df`` is not used or modified. As a side effect this sets
    ``mp.with_df`` to the bare GDF on ``kpts``, which supplies both the band
    Hartree potential and the correlation integrals.

    Args:
        mp (KMP2_stagger): The instance under construction.
        mf: Converged periodic KRHF reference.
        kpts_occ (ndarray): Shifted k-points of shape (nkpts_ov, 3), absent from
            ``mf.kpts``.
        kpts (ndarray): Band k-points of shape (2*nkpts_ov, 3).

    Returns:
        mo_energy, mo_coeff, mo_occ on ``kpts``. Energies are in hartree.
    """
    #   Imported here so the package still loads with a stock PySCF.
    from pyscf.pbc.df.rsdf_stc import RSGDF_STC

    mp.with_df = mp._make_bare_gdf(kpts)

    #   The k-points here must be the SCF mesh: they define the cutoff, and the
    #   union with the shifted mesh is not a regular mesh.
    stc_df = RSGDF_STC(mf.cell, mf.kpts, eta=mp.eta, exxdiv=mp.exxdiv,
                       rc_type=_STC_RC_TYPE[mp.exxdiv])
    stc_df.auxbasis = mp.auxbasis
    stc_df.with_df_j = mp.with_df
    #   Include the band points now so get_k_kpts does not rebuild the integrals.
    stc_df.build(j_only=False, kpts_band=kpts_occ)

    #   The sTC integrals hold the complete kernel: no Ewald correction applies.
    with lib.temporary_env(mf, exxdiv=None, with_df=stc_df):
        mo_energy, mo_coeff = mf.get_bands(kpts)
        mo_occ = get_occ(mf, mo_energy_kpts=mo_energy)
    return mo_energy, mo_coeff, mo_occ


#   Class for staggered mesh MP2
class KMP2_stagger(kmp2.KMP2):
    """Staggered-mesh restricted periodic MP2.

    ``flag_submesh=True`` partitions the reference mesh into two half-sized
    submeshes. ``flag_submesh=False`` evaluates band orbitals on a mesh shifted
    from ``mf.kpts``, at the fixed reference density. ``kernel`` returns the
    correlation energy in hartree per primitive cell.

    Args:
        mf: Converged periodic mean-field reference.
        frozen: Frozen orbital specification accepted by PySCF kMP2.
        flag_submesh: Select the submesh or band-extension workflow.
        band_df: How to build the band Fock matrix for ``flag_submesh=False``.
            ``'fftdf'`` is the PySCF path: a temporary FFTDF with hard
            ``vcut_sph``, then a default-auxiliary GDF for correlation.
            ``'stc'`` uses ``RSGDF_STC`` exchange and ordinary GDF Hartree, and
            requires a GDF or RSGDF KRHF reference. ``None`` selects ``'stc'``
            for a reference from ``rsdf_stc.density_fit`` and ``'fftdf'``
            otherwise.
        eta: sTC smoothing parameter. Defaults to the sTC reference's value,
            otherwise 4.0. Only valid with ``band_df='stc'``.
        exxdiv: sTC cutoff, ``'vcut_ws'`` or ``'vcut_sph'``. Defaults to the sTC
            reference's value, otherwise ``'vcut_ws'``. Only valid with
            ``band_df='stc'``.
        auxbasis: Auxiliary basis for the sTC exchange, band Hartree, and
            correlation integrals. Defaults to ``mf.with_df.auxbasis``. Not used
            by the ``'fftdf'`` path, which keeps PySCF's default auxiliary basis.
    """

    _keys = kmp2.KMP2._keys.union({
        'cell', 'with_df', 'with_df_ints', 'flag_submesh', 'band_df', 'eta',
        'exxdiv', 'auxbasis', 'kpts_vir', 'kpts_occ', 'nkpts_ov',
        'kpts_idx_vir', 'kpts_idx_occ',
    })

    def __init__(self, mf, frozen=None, flag_submesh=False, band_df=None,
                 eta=None, exxdiv=None, auxbasis=None):
        self.cell = mf.cell
        self._scf = mf
        self.verbose = self.cell.verbose
        self.stdout = self.cell.stdout
        self.max_memory = self.cell.max_memory

        #   MP2 energy
        self.e_corr = None

        #   nocc and nmo variables
        self._nocc = None
        self._nmo = None

        #   Bare integrals; None leaves the choice to _init_mp_df_eris_stagger
        self.with_df = None

        stc_reference = _is_stc_reference(mf)
        self.band_df = self._check_band_df(mf, band_df, flag_submesh, stc_reference,
                                           eta, exxdiv)
        self.eta = self.exxdiv = self.auxbasis = None
        if self.band_df == 'stc':
            default_eta, default_exxdiv = 4.0, 'vcut_ws'
            if stc_reference:
                default_eta, default_exxdiv = mf.with_df.eta, mf.with_df.exxdiv
            self.eta = float(default_eta if eta is None else eta)
            self.exxdiv = str(default_exxdiv if exxdiv is None else exxdiv).lower()
            if self.exxdiv not in _STC_RC_TYPE:
                raise ValueError("exxdiv must be 'vcut_ws' or 'vcut_sph'")
        if self.band_df == 'stc' or (stc_reference and flag_submesh):
            self.auxbasis = mf.with_df.auxbasis if auxbasis is None else auxbasis

        #   Construction of orbitals and orbital energies on two staggered meshes
        nks = get_monkhorst_pack_size(mf.cell, mf.kpts)  #   mesh size
        self.flag_submesh = flag_submesh
        if flag_submesh is True:
            #   Choice 1: Use two staggered submeshes of mf.kpts of half size along each dimension
            if np.any( nks % 2) :
                log = logger.Logger(self.stdout, self.verbose)
                log.warn(
                    'The k-point mesh in the input mf has odd size in certain dimension. \n'
                    'Cannot use its submeshes for staggered mesh calculation!\n'
                    'You may set "flag_submesh" to False')
                raise RuntimeError

            nks_ov = nks // 2
            shift = mf.kpts[ np.argmin( np.sum(mf.kpts**2, axis=1) )]   #   shift of the original k-point mesh
            half_shift = mf.cell.get_abs_kpts( [0.5/n for n in nks_ov] )
            kpts_vir = shift + mf.cell.make_kpts(nks_ov, with_gamma_point=True)
            kpts_occ = kpts_vir + half_shift

            self.kpts = mf.kpts
            self.mo_energy = mf.mo_energy
            self.mo_coeff = mf.mo_coeff
            self.mo_occ = get_occ(mf, mo_energy_kpts=mf.mo_energy)

            if stc_reference:
                #   The reference CDERIs hold the truncated interaction, and its
                #   Hartree builder stores only diagonal k-pairs.
                self.with_df = self._make_bare_gdf(mf.kpts)

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

            half_shift = mf.cell.get_abs_kpts( [0.5/n for n in nks] )
            kpts_vir = mf.kpts
            kpts_occ = kpts_vir + half_shift
            kpts = np.concatenate( (kpts_occ, kpts_vir), axis=0)
            if self.band_df == 'stc':
                mo_energy, mo_coeff, mo_occ = _get_bands_stc(self, mf, kpts_occ, kpts)
            elif isinstance(mf, scf.khf.KRHF):
                with lib.temporary_env(mf, exxdiv='vcut_sph', with_df=df.FFTDF(mf.cell, mf.kpts)):
                    mo_energy, mo_coeff = mf.get_bands(kpts)
                    mo_occ = get_occ(mf, mo_energy_kpts=mo_energy)

            self.kpts = kpts
            self.mo_energy = mo_energy
            self.mo_coeff = mo_coeff
            self.mo_occ = mo_occ

        if isinstance(self._scf.with_df, df.df.GDF):
            self.with_df_ints = True
        else:
            self.with_df_ints = False

        #   Basic info
        self.nkpts = len(self.kpts)
        self.frozen = frozen
        self.khelper = make_kpts_helper(self.cell, self.kpts)

        #   Mesh info for the staggered mesh method
        self.kpts_vir = kpts_vir
        self.kpts_occ = kpts_occ
        self.nkpts_ov = len(self.kpts_vir)

        #   Map kpts, kpts_vir, and kpts_occ to the first BZ for better processing
        kpts_scaled = self.cell.get_scaled_kpts(self.kpts)
        kpts_occ_scaled = self.cell.get_scaled_kpts(self.kpts_occ)
        kpts_vir_scaled = self.cell.get_scaled_kpts(self.kpts_vir)
        kpts_bz = round_to_fbz(kpts_scaled, wrap_around=True, tol=1e-8)
        kpts_occ_bz = round_to_fbz(kpts_occ_scaled, wrap_around=True, tol=1e-8)
        kpts_vir_bz = round_to_fbz(kpts_vir_scaled, wrap_around=True, tol=1e-8)

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

    @staticmethod
    def _check_band_df(mf, band_df, flag_submesh, stc_reference, eta, exxdiv):
        """Resolve ``band_df`` and reject combinations the sTC path cannot run."""
        if band_df is None:
            band_df = 'stc' if stc_reference else 'fftdf'
        if band_df not in ('fftdf', 'stc'):
            raise ValueError("band_df must be 'fftdf', 'stc', or None")

        if band_df == 'fftdf':
            if eta is not None or exxdiv is not None:
                raise ValueError("eta and exxdiv require band_df='stc'")
            return band_df

        if flag_submesh and not stc_reference:
            raise ValueError("band_df='stc' requires flag_submesh=False, "
                             "since submesh orbitals come from the reference")
        if not isinstance(mf.with_df, df.df.GDF):
            raise TypeError("band_df='stc' requires a GDF or RSGDF reference")
        if not isinstance(mf, scf.khf.KRHF) or isinstance(mf, krks.KRKS):
            raise TypeError("band_df='stc' requires a KRHF reference")
        return band_df

    def _make_bare_gdf(self, kpts):
        """Build ordinary GDF integrals on ``kpts`` with the resolved aux basis."""
        with_df = df.GDF(self.cell, kpts)
        with_df.auxbasis = self.auxbasis
        return with_df.build()

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        nocc = self.nocc
        nvir = self.nmo - nocc
        log.info('Staggerd mesh method for MP2 nocc = %d, nvir = %d', nocc, nvir)

        nks_ov = get_monkhorst_pack_size(self.cell, self.kpts_vir)  #   mesh size
        if self.flag_submesh is True:
            log.info('Two %d*%d*%d-sized submeshes of mf.kpts are used', nks_ov[0], nks_ov[1], nks_ov[2])
        else:
            log.info('Two %d*%d*%d-sized meshes are used based on non-SCF calculation', nks_ov[0], nks_ov[1], nks_ov[2])

        log.info('band_df = %s', self.band_df)
        if self.band_df == 'stc':
            log.info('sTC cutoff = %s, eta = %.15g', self.exxdiv, self.eta)
            log.info('auxbasis = %s', self.auxbasis)

        if self.frozen is not None:
            log.info('frozen orbitals = %s', str(self.frozen))
        log.info('KMP2 energy = %.15g' % (self.e_corr))
        return self

    def kernel(self):
        if self.mo_energy is None or self.mo_coeff is None:
            log = logger.Logger(self.stdout, self.verbose)
            log.warn('mo_coeff, mo_energy are not provided.')
            raise RuntimeError

        mo_coeff, mo_energy = _add_padding(self, self.mo_coeff, self.mo_energy)

        self.e_corr = kernel(self, mo_energy, mo_coeff)
        self.dump_flags()
        return self.e_corr
