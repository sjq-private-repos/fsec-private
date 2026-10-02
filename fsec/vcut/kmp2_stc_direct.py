#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
# Copyright 2026 Stephen Quiton. All Rights Reserved.
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

"""Periodic MP2 with on-demand direct sTC density-fitting factors.

The contractions reuse the orbital preparation and public behavior of the
stored sTC calculators, while transforming occupied orbitals in blocks. No
AO three-center integrals are written to a CDERI file.
"""

import copy

import numpy as np
import scipy.linalg

from pyscf import lib
from pyscf.lib import einsum, logger
from pyscf.lib.parameters import LARGE_DENOM
from pyscf.pbc.df import rsdf
from pyscf.pbc.lib import kpts_helper

from .kmp2_stc import KMP2_STC as _KMP2_STC_STORED
from .kmp2_stc import _prepare_orbitals

try:
    from pyscf.pbc.df import fft_stc
except ImportError as err:
    raise ImportError(
        "Direct sTC MP2 requires the custom PySCF fork with "
        "pyscf.pbc.df.fft_stc support"
    ) from err


def _direct_support():
    """Load the custom direct-transform API only when this path is used."""
    try:
        from pyscf.pbc.df import rsdf_direct_helper
        from pyscf.pbc.mp import kmp2_direct
    except ImportError as err:
        raise ImportError(
            "Direct sTC MP2 requires custom PySCF direct-RSDF support "
            "(pyscf.pbc.df.rsdf_direct_ao2mo and "
            "pyscf.pbc.mp.kmp2_direct)"
        ) from err
    if not hasattr(kmp2_direct, "_init_mp_df_eris"):
        raise ImportError(
            "Direct sTC MP2 requires the custom PySCF "
            "kmp2_direct._init_mp_df_eris API"
        )
    return rsdf_direct_helper, kmp2_direct._init_mp_df_eris


class _RSGDF_STC_DIRECT(rsdf.RSGDF):
    """Private direct RSDF builder using the complete smoothed cutoff."""

    supports_post_hf = False
    supports_ao_eri = False
    _keys = rsdf.RSGDF._keys | {
        "eta", "exxdiv", "rc_type", "omega_stc", "_scf_kpts",
        "_stc_rin", "_stc_kmesh", "_ws_exx",
    }

    def __init__(self, cell, kpts, eta=4.0, exxdiv="vcut_ws", rc_type="ws"):
        super().__init__(cell, kpts)
        self._scf_kpts = np.asarray(kpts, dtype=float).reshape(-1, 3).copy()
        self.eta = float(eta)
        self.exxdiv, self.rc_type = fft_stc._validate_cutoff(exxdiv, rc_type)
        self.omega_stc = None
        self._stc_rin = None
        self._stc_kmesh = None
        self._ws_exx = None
        self.direct = True
        self.semidirect = False
        self.ksym = "s2"

    def _rs_build(self):
        fft_stc._prepare_stc(self)
        self.omega = self.omega_j2c = self.omega_stc
        super()._rs_build()

    def weighted_coulG(self, kpt=np.zeros(3), exx=False, mesh=None, omega=None):
        if mesh is None:
            mesh = self.mesh
        if omega is None:
            omega = self.omega
        coulG = fft_stc.get_coulG(self, kpt, mesh, long_range=True)
        if not np.isclose(omega, self.omega_stc):
            raise RuntimeError("inconsistent sTC range-separation parameter")
        return coulG * self.cell.get_Gv_weights(mesh)[2]

    def _unsupported(self, *args, **kwargs):
        raise NotImplementedError(
            "The direct sTC builder is reserved for the fsec.vcut MP2 "
            "calculator; general SCF and post-HF APIs are unsupported"
        )

    get_jk = get_eri = get_ao_eri = ao2mo = get_mo_eri = _unsupported
    ao2mo_7d = loop = update_mp = _unsupported


class KMP2_STC_DIRECT(_KMP2_STC_STORED):
    """Blockwise direct implementation of sTC periodic MP2.

    The default is exchange-only; ``with_direct=True`` also computes the sTC
    direct contribution. ``rsdf_occ_block_size`` requests a padded occupied
    block size. If it is ``None``, the largest estimated block fitting within
    80 percent of available memory is selected.
    """

    _keys = _KMP2_STC_STORED._keys | {"rsdf_occ_block_size"}

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None, auxbasis=None,
                 eta=4.0, exxdiv="vcut_ws", rc_type="ws", *,
                 rsdf_occ_block_size=None):
        super().__init__(
            mf, frozen=frozen, mo_coeff=mo_coeff, mo_occ=mo_occ,
            auxbasis=auxbasis, eta=eta, exxdiv=exxdiv, rc_type=rc_type,
        )
        self.with_df = _RSGDF_STC_DIRECT(
            self.cell, self.kpts, eta=eta, exxdiv=exxdiv, rc_type=rc_type
        )
        self.with_df.auxbasis = auxbasis
        if auxbasis is None:
            self.with_df.auxbasis = getattr(
                getattr(mf, "with_df", None), "auxbasis", None
            )
        self.with_df.max_memory = self.max_memory
        self.with_df.verbose = self.verbose
        self.with_df.stdout = self.stdout
        self.rsdf_occ_block_size = rsdf_occ_block_size

    @property
    def rsdf_occ_block_size(self):
        return self._rsdf_occ_block_size

    @rsdf_occ_block_size.setter
    def rsdf_occ_block_size(self, value):
        if value is not None:
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer)) or value <= 0):
                raise ValueError("rsdf_occ_block_size must be a positive integer or None")
            value = int(value)
        self._rsdf_occ_block_size = value

    def kernel(self, mo_energy=None, mo_coeff=None, with_t2=False,
               with_direct=False):
        """Return exchange energy, optionally with same-interaction sTC direct.

        Energies are Hartree per unit cell. If requested, amplitudes retain
        the padded ``(nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir)`` shape.
        """
        cput0 = (logger.process_clock(), logger.perf_counter())
        rsdf_direct_helper, transform = _direct_support()
        context, orbitals = _prepare_orbitals(
            self, mo_energy=mo_energy, mo_coeff=mo_coeff
        )
        context.dump_flags()
        _prepare_direct_builder(self, self.with_df)
        block_size, estimate, available = _choose_block_size(
            self, orbitals, [self.with_df], retained_t2=int(with_t2),
            helper=rsdf_direct_helper,
        )
        logger.info(
            self,
            "direct sTC occupied block size = %d (estimated peak %.0f MB; "
            "available %.0f MB)",
            block_size, estimate, available,
        )
        direct, exchange, t2 = _contract_components(
            self, context, orbitals, self.with_df, with_t2=with_t2,
            do_direct=with_direct, do_exchange=True, block_size=block_size,
            transform=transform, helper=rsdf_direct_helper,
        )

        self.e_corr_direct = float(direct / self.nkpts) if with_direct else None
        self.e_corr_exchange = float(exchange / self.nkpts)
        self.e_corr = self.e_corr_exchange + (self.e_corr_direct or 0.0)
        self.t2 = t2
        logger.new_logger(self).timer("KMP2_STC_DIRECT", *cput0)
        if with_direct:
            logger.new_logger(self).info(
                "sTC MP2 direct energy = %.15g Hartree per unit cell",
                self.e_corr_direct,
            )
        logger.new_logger(self).info(
            "sTC MP2 exchange energy = %.15g Hartree per unit cell",
            self.e_corr_exchange,
        )
        return self.e_corr, t2


class KMP2_HYBRID_DIRECT(KMP2_STC_DIRECT):
    """Blockwise bare direct plus on-demand sTC exchange periodic MP2."""

    _keys = KMP2_STC_DIRECT._keys | {"with_df_direct"}

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None, auxbasis=None,
                 eta=4.0, exxdiv="vcut_ws", rc_type="ws", *,
                 rsdf_occ_block_size=None):
        super().__init__(
            mf, frozen=frozen, mo_coeff=mo_coeff, mo_occ=mo_occ,
            auxbasis=auxbasis, eta=eta, exxdiv=exxdiv, rc_type=rc_type,
            rsdf_occ_block_size=rsdf_occ_block_size,
        )
        self.with_df_direct = rsdf.RSGDF(self.cell, self.kpts)
        self.with_df_direct.auxbasis = self.with_df.auxbasis
        self.with_df_direct.max_memory = self.max_memory
        self.with_df_direct.verbose = self.verbose
        self.with_df_direct.stdout = self.stdout
        self.with_df_direct.direct = True
        self.with_df_direct.semidirect = False
        self.with_df_direct.ksym = "s2"

    def kernel(self, mo_energy=None, mo_coeff=None, with_t2=False):
        """Return bare direct plus sTC exchange and optional padded amplitudes."""
        cput0 = (logger.process_clock(), logger.perf_counter())
        rsdf_direct_helper, transform = _direct_support()
        context, orbitals = _prepare_orbitals(
            self, mo_energy=mo_energy, mo_coeff=mo_coeff
        )
        context.dump_flags()
        builders = [self.with_df_direct, self.with_df]
        for builder in builders:
            _prepare_direct_builder(self, builder)
        block_size, estimate, available = _choose_block_size(
            self, orbitals, builders, retained_t2=2 if with_t2 else 0,
            helper=rsdf_direct_helper,
        )
        logger.info(
            self,
            "direct hybrid occupied block size = %d (estimated peak %.0f MB; "
            "available %.0f MB)",
            block_size, estimate, available,
        )

        # Run each interaction independently; only requested amplitudes remain
        # resident while the second factor transformation proceeds.
        direct, _, t2_direct = _contract_components(
            self, context, orbitals, self.with_df_direct,
            with_t2=with_t2, do_direct=True, do_exchange=False,
            block_size=block_size, transform=transform,
            helper=rsdf_direct_helper,
        )
        _, exchange, t2_exchange = _contract_components(
            self, context, orbitals, self.with_df,
            with_t2=with_t2, do_direct=False, do_exchange=True,
            block_size=block_size, transform=transform,
            helper=rsdf_direct_helper,
        )

        self.e_corr_direct = float(direct / self.nkpts)
        self.e_corr_exchange = float(exchange / self.nkpts)
        self.e_corr = self.e_corr_direct + self.e_corr_exchange
        self.t2 = ({"direct": t2_direct, "exchange": t2_exchange}
                   if with_t2 else None)
        logger.new_logger(self).timer("KMP2_HYBRID_DIRECT", *cput0)
        logger.new_logger(self).info(
            "Hybrid MP2 bare direct energy = %.15g Hartree per unit cell",
            self.e_corr_direct,
        )
        logger.new_logger(self).info(
            "Hybrid MP2 sTC exchange energy = %.15g Hartree per unit cell",
            self.e_corr_exchange,
        )
        return self.e_corr, self.t2


def _prepare_direct_builder(mp, builder):
    """Build integral metadata without ever creating stored CDERIs."""
    if not getattr(builder, "direct", False) or getattr(builder, "semidirect", False):
        raise NotImplementedError(
            "Direct MP2 requires direct=True and semidirect=False; stored and "
            "semidirect CDERI modes are unsupported"
        )
    if builder._cderi is not None:
        raise RuntimeError("Direct MP2 builders must not contain stored CDERIs")
    builder.max_memory = mp.max_memory
    builder.verbose = mp.verbose
    builder.stdout = mp.stdout
    if not np.any(np.linalg.norm(mp.kpts, axis=1) < 1e-9):
        # Native kmesh inference can misidentify commensurately shifted meshes.
        builder.use_bvk = False
    builder.build()
    if builder._cderi is not None:
        raise RuntimeError("Direct RSDF build unexpectedly created stored CDERIs")


def _unique_q_points(builder, helper):
    kptij_lst = helper.get_kptij_lst(builder.kpts, ksym="s2")
    return [entry[0] for entry in helper.loop_uniq_q(
        builder, kptij_lst=kptij_lst, verbose=0
    )]


def _memory_costs(mp, orbitals, builder, helper, current_memory):
    """Estimate fixed, linear, and quadratic memory for one direct pass."""
    nkpts = mp.nkpts
    nvir = orbitals["nvir"]
    nao = mp.cell.nao_nr()
    naux = builder.auxcell.nao_nr()
    itemsize = np.dtype(orbitals["eri_dtype"]).itemsize
    energy_itemsize = np.dtype(orbitals["energy_dtype"]).itemsize
    qpoints = _unique_q_points(builder, helper)
    nq = len(qpoints)
    mesh_size = int(np.prod(builder.mesh_compact))
    kptij_count = nkpts * (nkpts + 1) // 2

    # The native ijL kernel reserves (1 + nkpts**2) transformed tables,
    # including its intermediate. A second output table remains live while
    # transforming the next occupied block.
    native_factor_per_occ = (
        nkpts**2 * (nkpts**2 + 1) * nvir * naux * itemsize
    )
    retained_table_per_occ = nkpts**2 * nvir * naux * itemsize
    linear_mb = (native_factor_per_occ + retained_table_per_occ) / 1e6

    # Two four-index ERIs, denominator, quotient scratch, and amplitude block.
    block_itemsize = max(itemsize, energy_itemsize)
    quadratic_mb = 5 * nvir**2 * block_itemsize / 1e6

    q_metric_mb = (nq + 1) * naux**2 * 16 / 1e6
    reciprocal_aux_mb = (
        nq * 2 * mesh_size * naux * 8 + mesh_size * naux * 16
    ) / 1e6
    lr_memory_mb = max(2000.0, builder.max_memory - current_memory)
    lr_block = max(2048, int(lr_memory_mb * 0.5e6 / (16 * naux)))
    # The native buffer floor is a block-length limit, not an allocation:
    # Fourier transforms only materialize the reciprocal points that exist.
    metric_grid_size = int(np.prod(builder.mesh_j2c))
    lr_block = min(metric_grid_size, lr_block)
    j2c_lr_mb = 3 * lr_block * naux * 16 / 1e6
    j3c_rows = (kptij_count + nkpts + 1) * naux
    # AO rows are already shell-blocked by the native transform. Requiring
    # the full nao**2 buffer would defeat that blocking during preflight.
    # Its 70-percent AO allocation must fit at least the largest AO shell.
    shell_width = int(np.max(np.diff(mp.cell.ao_loc_nr())))
    j3c_buffer_mb = j3c_rows * shell_width * nao * itemsize / (0.7e6)
    fixed_mb = q_metric_mb + reciprocal_aux_mb + j2c_lr_mb + j3c_buffer_mb
    return fixed_mb, linear_mb, quadratic_mb


def _choose_block_size(mp, orbitals, builders, retained_t2, helper):
    """Choose the largest occupied block under the 80-percent memory budget."""
    nocc = orbitals["nocc"]
    nvir = orbitals["nvir"]
    nkpts = mp.nkpts
    t2_itemsize = np.dtype(
        np.result_type(orbitals["eri_dtype"], np.complex128)
    ).itemsize
    retained_mb = (
        retained_t2 * nkpts**3 * (nocc * nvir) ** 2 * t2_itemsize / 1e6
    )
    current_memory = lib.current_memory()[0]
    available = float(mp.max_memory) - current_memory
    target = max(0.0, 0.8 * available)
    costs = [
        _memory_costs(mp, orbitals, builder, helper, current_memory)
        for builder in builders
    ]
    requested = mp.rsdf_occ_block_size
    max_block = min(nocc, requested) if requested is not None else nocc

    candidates = [max_block] if requested is not None else range(max_block, 0, -1)
    for block_size in candidates:
        peak = retained_mb + max(
            fixed + block_size * linear + block_size**2 * quadratic
            for fixed, linear, quadratic in costs
        )
        if peak <= target:
            return block_size, peak, available

    if requested is not None:
        raise MemoryError(
            "Requested rsdf_occ_block_size=%d is estimated to need %.0f MB, "
            "above the 80%% memory budget of %.0f MB"
            % (max_block, peak, target)
        )
    raise MemoryError(
        "Direct MP2 occupied block size 1 is estimated to need %.0f MB, "
        "above the 80%% memory budget of %.0f MB "
        "(%.0f MB currently available)"
        % (peak, target, available)
    )


def _preflight_cholesky_metrics(builder, helper):
    """Reject eigendecomposition or non-positive metrics before AO2MO."""
    if getattr(builder, "j2c_eig_always", False):
        raise NotImplementedError(
            "Direct MP2 requires Cholesky-decomposable RSDF metrics; "
            "j2c_eig_always=True is unsupported"
        )
    for qpt in _unique_q_points(builder, helper):
        metric = helper.get_j2c(
            builder, kpts=np.asarray([qpt]), verbose=0
        )[0]
        try:
            scipy.linalg.cholesky(
                metric, lower=True, overwrite_a=True, check_finite=False
            )
        except scipy.linalg.LinAlgError as err:
            raise NotImplementedError(
                "Direct MP2 requires Cholesky-decomposable RSDF metrics; "
                "the metric for q=%s is not positive definite" % qpt
            ) from err
        del metric


def _transform_occupied_block(context, orbitals, builder, transform, i0, i1):
    """Transform one padded occupied slice to (occupied, virtual, auxiliary)."""
    df_context = copy.copy(context)
    df_context._scf = copy.copy(context._scf)
    df_context._scf.with_df = builder
    df_context.kernel_ao2mo = 1
    return transform(
        df_context, mo_coeff=orbitals["mo_coeff"], j3c_order="ijL",
        oslice=(i0, i1),
    )


def _contract_components(mp, context, orbitals, builder, with_t2,
                         do_direct, do_exchange, block_size, transform, helper):
    """Contract selected terms from one interaction using two sliced tables."""
    _preflight_cholesky_metrics(builder, helper)
    nkpts = mp.nkpts
    nocc = orbitals["nocc"]
    nvir = orbitals["nvir"]
    mo_energy = orbitals["mo_energy"]
    opadding = orbitals["opadding"]
    vpadding = orbitals["vpadding"]
    eri_dtype = orbitals["eri_dtype"]
    kconserv = mp.khelper.kconserv
    t2 = None
    if with_t2:
        t2 = np.zeros(
            (nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir),
            dtype=np.result_type(eri_dtype, np.complex128),
        )

    energy_o = [np.asarray(mo_energy[k][:nocc]) for k in range(nkpts)]
    energy_v = [np.asarray(mo_energy[k][nocc:]) for k in range(nkpts)]
    direct_energy = 0.0
    exchange_energy = 0.0
    blocks = [(i0, min(i0 + block_size, nocc))
              for i0 in range(0, nocc, block_size)]

    for iblock, (i0, i1) in enumerate(blocks):
        left = _transform_occupied_block(
            context, orbitals, builder, transform, i0, i1
        )
        for j0, j1 in blocks[iblock:]:
            if (j0, j1) == (i0, i1):
                right = left
            else:
                right = _transform_occupied_block(
                    context, orbitals, builder, transform, j0, j1
                )
            direct_energy, exchange_energy = _contract_block_orientation(
                mp, orbitals, left, right, (i0, i1), (j0, j1),
                energy_o, energy_v, opadding, vpadding, kconserv,
                direct_energy, exchange_energy, do_direct, do_exchange, t2,
            )
            if (j0, j1) != (i0, i1):
                direct_energy, exchange_energy = _contract_block_orientation(
                    mp, orbitals, right, left, (j0, j1), (i0, i1),
                    energy_o, energy_v, opadding, vpadding, kconserv,
                    direct_energy, exchange_energy, do_direct, do_exchange, t2,
                )
            del right
        del left

    return direct_energy, exchange_energy, t2


def _contract_block_orientation(mp, orbitals, left, right, ibounds, jbounds,
                                energy_o, energy_v, opadding, vpadding,
                                kconserv, direct_energy, exchange_energy,
                                do_direct, do_exchange, t2):
    """Accumulate all ordered k triples for one occupied-block orientation."""
    i0, i1 = ibounds
    j0, j1 = jbounds
    nkpts = mp.nkpts
    energy_dtype = orbitals["energy_dtype"]

    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[ki, ka, kj]
                eia = _get_eia(
                    ki, ka, i0, i1, energy_o, energy_v, opadding,
                    vpadding, energy_dtype,
                )
                ejb = _get_eia(
                    kj, kb, j0, j1, energy_o, energy_v, opadding,
                    vpadding, energy_dtype,
                )
                eri = np.einsum(
                    "iaL,jbL->iajb", left[ki, ka], right[kj, kb],
                    optimize=True,
                ).transpose(0, 2, 1, 3)
                eri /= nkpts
                denominator = lib.direct_sum(
                    "ia,jb->ijab", eia, ejb
                )
                # A real ndarray may return itself from conjugation; copy so
                # division cannot overwrite the ERI needed for the energy.
                amplitudes = eri.copy()
                np.conjugate(amplitudes, out=amplitudes)
                amplitudes /= denominator
                if t2 is not None:
                    t2[ki, kj, ka, i0:i1, j0:j1] = amplitudes
                if do_direct:
                    direct_energy += 2.0 * einsum(
                        "ijab,ijab->", amplitudes, eri
                    ).real
                if do_exchange:
                    exchange_eri = np.einsum(
                        "iaL,jbL->iajb", left[ki, kb], right[kj, ka],
                        optimize=True,
                    ).transpose(0, 2, 1, 3)
                    exchange_eri /= nkpts
                    exchange_energy -= einsum(
                        "ijab,ijba->", amplitudes, exchange_eri
                    ).real
                del eri, denominator, amplitudes
                if do_exchange:
                    del exchange_eri
    return direct_energy, exchange_energy


def _get_eia(occupied_kpoint, virtual_kpoint, i0, i1, energy_o, energy_v,
             opadding, vpadding, dtype):
    """Build a padded occupied-virtual energy-difference block."""
    nvir = len(energy_v[0])
    eia = LARGE_DENOM * np.ones((i1 - i0, nvir), dtype=dtype)
    occupied = np.asarray(opadding[occupied_kpoint])
    global_occ = occupied[(occupied >= i0) & (occupied < i1)]
    occ_valid = global_occ - i0
    vir_valid = np.asarray(vpadding[virtual_kpoint])
    if occ_valid.size and vir_valid.size:
        eia[np.ix_(occ_valid, vir_valid)] = (
            energy_o[occupied_kpoint][global_occ, None]
            - energy_v[virtual_kpoint][None, vir_valid]
        )
    return eia


KMP2_STC = KMP2_STC_DIRECT
KMP2_HYBRID = KMP2_HYBRID_DIRECT
KMP2 = KMP2_STC
