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
#
# The CDERI-to-MO transformation and padding conventions below are adapted
# from pyscf.pbc.mp.kmp2 by the PySCF Developers.

"""Periodic MP2 with stored sTC and hybrid Coulomb interactions.

This module requires the custom ``pyscf.pbc.df.rsdf_stc`` module. Both classes
keep the reference SCF orbitals and density-fitting object unchanged. Each
owns a ``with_df`` builder for sTC; the hybrid also owns a bare direct builder.
"""

import copy

import numpy as np

from pyscf import lib
from pyscf.lib import einsum, logger
from pyscf.lib.parameters import LARGE_DENOM
from pyscf.pbc.df import df, rsdf
from pyscf.pbc.lib import kpts as libkpts
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.mp import kmp2 as pyscf_kmp2

try:
    from pyscf.pbc.df.rsdf_stc import RSGDF_STC
    from pyscf.pbc.df.fft_stc import _regular_kmesh
except ImportError as err:
    raise ImportError(
        "KMP2_STC requires custom PySCF stored sTC support "
        "(pyscf.pbc.df.rsdf_stc and pyscf.pbc.df.fft_stc)"
    ) from err


def _as_full_kpts(mf):
    """Return and validate a complete, regular 3D k-point mesh."""
    stored_kpts = getattr(getattr(mf, "with_df", None), "_kpts", None)
    if (isinstance(stored_kpts, libkpts.KPoints)
            and stored_kpts.nkpts_ibz < stored_kpts.nkpts):
        raise NotImplementedError(
            "KMP2_STC requires full k-point sampling; symmetry-reduced "
            "KPoints objects are unsupported"
        )

    kpts = mf.kpts
    if isinstance(kpts, libkpts.KPoints):
        raise NotImplementedError(
            "KMP2_STC requires full k-point sampling; symmetry-reduced "
            "KPoints objects are unsupported"
        )

    kpts = np.asarray(kpts, dtype=float).reshape(-1, 3)
    try:
        _regular_kmesh(mf.cell, kpts)
    except RuntimeError as err:
        raise NotImplementedError(
            "KMP2_STC requires a complete regular full k-point mesh"
        ) from err
    return kpts


def _validate_reference(mf, mo_occ=None):
    from pyscf.pbc.scf.khf import KRHF

    if not isinstance(mf, KRHF):
        raise TypeError("KMP2_STC requires a restricted periodic KRHF reference")
    if mf.cell.dimension != 3:
        raise NotImplementedError("KMP2_STC is implemented for three-dimensional cells")
    if mf.mo_energy is None or mf.mo_coeff is None or mf.mo_occ is None:
        raise RuntimeError("Run KRHF before constructing KMP2_STC")

    occupations = mf.mo_occ if mo_occ is None else mo_occ
    if any(not np.all(np.isclose(occ, 0.0) | np.isclose(occ, 2.0))
           for occ in occupations):
        raise NotImplementedError("KMP2_STC requires closed-shell integer occupations")
    return _as_full_kpts(mf)


class KMP2_STC(lib.StreamObject):
    """Compute periodic MP2 exchange, optionally with the sTC direct term.

    By default, the returned energy is exchange-only. With ``with_direct=True``
    the sTC interaction is also used in both factors of the direct term. The
    latest returned energy is stored in ``e_corr``; ``e_corr_exchange`` always
    stores exchange, while ``e_corr_direct`` is ``None`` when the direct term
    was not requested. Energies are in Hartree per unit cell. The independent
    ``with_df`` builder can be configured before its first build.

    Args:
        mf: A converged, canonical, closed-shell periodic KRHF reference.
        frozen: Frozen orbital specification accepted by PySCF kMP2.
        mo_coeff: Optional reference MO coefficients.
        mo_occ: Optional reference occupations.
        auxbasis: Auxiliary basis for the independent sTC builder.  If omitted,
            the reference SCF density-fitting builder's auxiliary basis is used.
        eta: sTC smoothing parameter passed to ``RSGDF_STC``.
        exxdiv: sTC cutoff, either ``"vcut_ws"`` or ``"vcut_sph"``.
        rc_type: Matching real-space cutoff, either ``"ws"`` or ``"sph"``.
    """

    _keys = {
        "mol", "cell", "_scf", "frozen", "mo_coeff", "mo_occ", "mo_energy",
        "kpts", "nkpts", "khelper", "with_df", "e_corr", "e_corr_direct",
        "e_corr_exchange", "t2", "_nocc", "_nmo", "max_memory",
    }

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None, auxbasis=None,
                 eta=4.0, exxdiv="vcut_ws", rc_type="ws"):
        kpts = _validate_reference(mf, mo_occ)
        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        if mo_occ is None:
            mo_occ = mf.mo_occ

        self.mol = mf.mol
        self.cell = mf.cell
        self._scf = mf
        self.verbose = mf.verbose
        self.stdout = mf.stdout
        self.max_memory = mf.max_memory
        self.frozen = frozen
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.mo_energy = mf.mo_energy
        self.kpts = kpts
        self.nkpts = len(kpts)
        self.khelper = kpts_helper.KptsHelper(mf.cell, kpts)

        if auxbasis is None:
            auxbasis = getattr(getattr(mf, "with_df", None), "auxbasis", None)
        self.with_df = RSGDF_STC(
            mf.cell, kpts, eta=float(eta), exxdiv=exxdiv, rc_type=rc_type
        )
        self.with_df.auxbasis = auxbasis
        self.with_df.max_memory = self.max_memory
        self.with_df.verbose = self.verbose
        self.with_df.stdout = self.stdout

        self._nocc = None
        self._nmo = None
        self.e_corr = None
        self.e_corr_direct = None
        self.e_corr_exchange = None
        self.t2 = None

    get_nocc = pyscf_kmp2.get_nocc
    get_nmo = pyscf_kmp2.get_nmo
    get_frozen_mask = pyscf_kmp2.get_frozen_mask

    @property
    def nocc(self):
        return self.get_nocc()

    @nocc.setter
    def nocc(self, value):
        self._nocc = value

    @property
    def nmo(self):
        return self.get_nmo()

    @nmo.setter
    def nmo(self, value):
        self._nmo = value

    def dump_flags(self):
        logger.info(self, "")
        logger.info(self, "******** %s ********", self.__class__)
        logger.info(self, "nkpts = %d", self.nkpts)
        logger.info(self, "nocc = %d", self.nocc)
        logger.info(self, "nmo = %d", self.nmo)
        logger.info(self, "sTC cutoff = %s, rc_type = %s",
                    self.with_df.exxdiv, self.with_df.rc_type)
        logger.info(self, "sTC eta = %.15g", self.with_df.eta)
        if hasattr(self, "with_df_direct"):
            logger.info(self, "bare direct builder = %s", self.with_df_direct)
        logger.info(
            self,
            "max_memory %d MB (current use %d MB)",
            self.max_memory,
            lib.current_memory()[0],
        )
        return self

    def kernel(self, mo_energy=None, mo_coeff=None, with_t2=False,
               with_direct=False):
        """Return exchange energy, or sTC direct plus exchange when requested.

        Each contribution uses its interaction in both ERI factors, including
        the amplitude numerator. Energies are in Hartree per unit cell. The
        optional amplitudes retain the padded shape
        ``(nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir)``. ``e_corr_direct`` is
        ``None`` when ``with_direct`` is false; ``e_corr`` matches the returned
        energy.
        """
        cput0 = (logger.process_clock(), logger.perf_counter())
        context, orbitals = _prepare_orbitals(
            self, mo_energy=mo_energy, mo_coeff=mo_coeff
        )
        context.dump_flags()
        builder = self.with_df
        _ensure_stored_builder(builder)
        _check_memory(self, orbitals, [builder], retained_t2=int(with_t2))
        direct, exchange, t2 = _contract_components(
            self, context, orbitals, builder, with_t2=with_t2,
            do_direct=with_direct, do_exchange=True,
        )

        self.e_corr_direct = float(direct / self.nkpts) if with_direct else None
        self.e_corr_exchange = float(exchange / self.nkpts)
        self.e_corr = self.e_corr_exchange + (self.e_corr_direct or 0.0)
        self.t2 = t2
        logger.new_logger(self).timer("KMP2_STC", *cput0)
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


class KMP2_HYBRID(KMP2_STC):
    """Bare direct plus sTC exchange periodic MP2 on fixed KRHF orbitals.

    A separate ordinary RSDF builder supplies the bare direct interaction;
    the inherited ``with_df`` builder supplies the sTC exchange interaction.
    Both builders are independent and may be configured before first use. The
    energy is in Hartree per unit cell and uses the same orbitals and energy
    denominators for both terms.
    """

    _keys = KMP2_STC._keys | {"with_df_direct"}

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None, auxbasis=None,
                 eta=4.0, exxdiv="vcut_ws", rc_type="ws"):
        super().__init__(
            mf, frozen=frozen, mo_coeff=mo_coeff, mo_occ=mo_occ,
            auxbasis=auxbasis, eta=eta, exxdiv=exxdiv, rc_type=rc_type,
        )
        self.with_df_direct = rsdf.RSGDF(self.cell, self.kpts)
        self.with_df_direct.auxbasis = self.with_df.auxbasis
        self.with_df_direct.max_memory = self.max_memory
        self.with_df_direct.verbose = self.verbose
        self.with_df_direct.stdout = self.stdout

    def kernel(self, mo_energy=None, mo_coeff=None, with_t2=False):
        """Return bare direct plus sTC exchange and optional amplitudes.

        When ``with_t2=True``, the second return value and ``self.t2`` are a
        dictionary with ``direct`` and ``exchange`` padded amplitude arrays.
        Otherwise both are ``None``. Component energies are available in
        ``e_corr_direct`` and ``e_corr_exchange``; ``e_corr`` matches the
        returned sum. Energies are in Hartree per unit cell.
        """
        cput0 = (logger.process_clock(), logger.perf_counter())
        context, orbitals = _prepare_orbitals(
            self, mo_energy=mo_energy, mo_coeff=mo_coeff
        )
        context.dump_flags()
        _ensure_stored_builder(self.with_df_direct)
        _ensure_stored_builder(self.with_df)
        _check_memory(
            self, orbitals, [self.with_df_direct, self.with_df],
            retained_t2=2 if with_t2 else 0,
        )

        # Each pass owns its transformed factors and work arrays. Only its
        # requested amplitudes survive into the next pass.
        direct_raw, _, t2_direct = _contract_components(
            self, context, orbitals, self.with_df_direct, with_t2=with_t2,
            do_direct=True, do_exchange=False,
        )
        _, exchange_raw, t2_exchange = _contract_components(
            self, context, orbitals, self.with_df, with_t2=with_t2,
            do_direct=False, do_exchange=True,
        )

        self.e_corr_direct = float(direct_raw / self.nkpts)
        self.e_corr_exchange = float(exchange_raw / self.nkpts)
        self.e_corr = self.e_corr_direct + self.e_corr_exchange
        if with_t2:
            self.t2 = {"direct": t2_direct, "exchange": t2_exchange}
        else:
            self.t2 = None
        logger.new_logger(self).timer("KMP2_HYBRID", *cput0)
        logger.new_logger(self).info(
            "Hybrid MP2 bare direct energy = %.15g Hartree per unit cell",
            self.e_corr_direct,
        )
        logger.new_logger(self).info(
            "Hybrid MP2 sTC exchange energy = %.15g Hartree per unit cell",
            self.e_corr_exchange,
        )
        return self.e_corr, self.t2


def _prepare_orbitals(mp, mo_energy=None, mo_coeff=None):
    """Pad the selected orbitals once and collect contraction dimensions."""
    if mo_energy is None:
        mo_energy = mp.mo_energy
    if mo_coeff is None:
        mo_coeff = mp.mo_coeff
    if mo_energy is None or mo_coeff is None:
        raise RuntimeError("mo_coeff and mo_energy must be initialized")

    context = copy.copy(mp)
    context.mo_energy = mo_energy
    coeffs = list(mo_coeff)
    coeff_dtype = np.result_type(*(np.asarray(coeff).dtype for coeff in coeffs))
    coeffs = [np.asarray(coeff, dtype=coeff_dtype) for coeff in coeffs]
    coeffs, mo_energy = pyscf_kmp2._add_padding(context, coeffs, mo_energy)
    coeffs = np.asarray(coeffs, dtype=coeff_dtype)
    context.mo_coeff = coeffs

    nocc = context.get_nocc()
    nmo = context.get_nmo()
    nvir = nmo - nocc
    dtype = (np.double if kpts_helper.gamma_point(mp.kpts)
             else np.complex128)
    eri_dtype = np.result_type(dtype, coeffs.dtype)
    energy_dtype = np.result_type(*(np.asarray(e).dtype for e in mo_energy))
    opadding, vpadding = pyscf_kmp2.padding_k_idx(context, kind="split")
    return context, {
        "mo_coeff": coeffs,
        "mo_energy": mo_energy,
        "nocc": nocc,
        "nmo": nmo,
        "nvir": nvir,
        "eri_dtype": eri_dtype,
        "energy_dtype": energy_dtype,
        "opadding": opadding,
        "vpadding": vpadding,
    }


def _ensure_stored_builder(mydf):
    """Build a selected density-fitting object and require stored CDERIs."""
    if getattr(mydf, "direct", False) or getattr(mydf, "semidirect", False):
        raise NotImplementedError(
            "KMP2_STC requires stored CDERIs; direct and semidirect DF are unsupported"
        )
    if mydf._cderi is None:
        mydf.build()


def _check_memory(mp, orbitals, builders, retained_t2):
    """Preflight each selected pass and all amplitudes retained across passes."""
    nocc = orbitals["nocc"]
    nvir = orbitals["nvir"]
    nkpts = mp.nkpts
    block_size = nocc * nocc * nvir * nvir
    itemsize = np.dtype(orbitals["eri_dtype"]).itemsize
    energy_itemsize = np.dtype(orbitals["energy_dtype"]).itemsize
    nao = mp.cell.nao_nr()
    pass_estimates = []
    for builder in builders:
        if builder.auxcell is not None:
            naux = builder.auxcell.nao_nr()
        else:
            naux = builder.get_naoaux()
        pass_estimates.append(
            nkpts**2 * naux * nocc * nvir * 16
            + (nkpts + 3) * itemsize * block_size
            + energy_itemsize * block_size
            + 3 * naux * nao**2 * 16
        )

    t2_itemsize = np.dtype(
        np.result_type(orbitals["eri_dtype"], np.complex128)
    ).itemsize
    retained_bytes = retained_t2 * nkpts**3 * (nocc * nvir) ** 2 * t2_itemsize
    mem_usage = (max(pass_estimates, default=0) + retained_bytes) / 1e6
    mem_avail = mp.max_memory - lib.current_memory()[0]
    if mem_usage > mem_avail:
        raise MemoryError(
            "Insufficient memory for %s: estimated %.0f MB "
            "(currently available %.0f MB)"
            % (mp.__class__.__name__, mem_usage, mem_avail)
        )


def _contract_components(mp, context, orbitals, builder, with_t2,
                         do_direct, do_exchange):
    """Contract selected direct and exchange terms using one DF interaction."""
    nocc = orbitals["nocc"]
    nvir = orbitals["nvir"]
    mo_coeff = orbitals["mo_coeff"]
    mo_energy = orbitals["mo_energy"]
    eri_dtype = orbitals["eri_dtype"]
    energy_dtype = orbitals["energy_dtype"]
    opadding = orbitals["opadding"]
    vpadding = orbitals["vpadding"]

    Lov = _init_mp_df_eris(context, mo_coeff, with_df=builder)
    t2_dtype = np.result_type(eri_dtype, np.complex128)
    if with_t2:
        t2 = np.zeros(
            (mp.nkpts, mp.nkpts, mp.nkpts,
             nocc, nocc, nvir, nvir), dtype=t2_dtype,
        )
    else:
        t2 = None

    mo_e_o = [np.asarray(mo_energy[k][:nocc]) for k in range(mp.nkpts)]
    mo_e_v = [np.asarray(mo_energy[k][nocc:]) for k in range(mp.nkpts)]
    direct_energy = 0.0
    exchange_energy = 0.0
    oovv = np.empty(
        (mp.nkpts, nocc, nocc, nvir, nvir), dtype=eri_dtype
    )
    for ki in range(mp.nkpts):
        for kj in range(mp.nkpts):
            for ka in range(mp.nkpts):
                kb = mp.khelper.kconserv[ki, ka, kj]
                oovv[ka] = einsum(
                    "Lia,Ljb->iajb", Lov[ki, ka], Lov[kj, kb]
                ).transpose(0, 2, 1, 3)
                oovv[ka] /= mp.nkpts

            for ka in range(mp.nkpts):
                kb = mp.khelper.kconserv[ki, ka, kj]
                eia = LARGE_DENOM * np.ones(
                    (nocc, nvir), dtype=energy_dtype
                )
                idx = np.ix_(opadding[ki], vpadding[ka])
                eia[idx] = (mo_e_o[ki][:, None] - mo_e_v[ka])[idx]
                ejb = LARGE_DENOM * np.ones(
                    (nocc, nvir), dtype=energy_dtype
                )
                idx = np.ix_(opadding[kj], vpadding[kb])
                ejb[idx] = (mo_e_o[kj][:, None] - mo_e_v[kb])[idx]
                denominator = lib.direct_sum("ia,jb->ijab", eia, ejb)
                t2_ijab = np.conj(oovv[ka] / denominator)
                if with_t2:
                    t2[ki, kj, ka] = t2_ijab
                if do_direct:
                    direct_energy += 2.0 * einsum(
                        "ijab,ijab", t2_ijab, oovv[ka]
                    ).real
                if do_exchange:
                    exchange_energy -= einsum(
                        "ijab,ijba", t2_ijab, oovv[kb]
                    ).real

    return direct_energy, exchange_energy, t2


def _init_mp_df_eris(mp, mo_coeff, with_df=None):
    """Transform a selected stored DF builder's CDERIs to the padded MOs."""
    from pyscf.ao2mo import _ao2mo

    mydf = mp.with_df if with_df is None else with_df
    if mydf._cderi is None:
        mydf.build()
    if getattr(mydf, "direct", False) or getattr(mydf, "semidirect", False):
        raise NotImplementedError(
            "KMP2_STC requires stored CDERIs; direct and semidirect DF are unsupported"
        )

    nocc = mp.nocc
    nmo = mp.nmo
    nvir = nmo - nocc
    nao = mp.cell.nao_nr()
    kpts = np.asarray(mp.kpts)
    dtype = np.double if kpts_helper.gamma_point(kpts) else np.complex128
    dtype = np.result_type(dtype, mo_coeff.dtype)
    Lov = np.empty((mp.nkpts, mp.nkpts), dtype=object)
    bra_start, bra_end = 0, nocc
    ket_start, ket_end = nmo + nocc, nmo + nocc + nvir
    tao = []

    with df.CDERIArray(mydf._cderi, mydf._dataname) as cderi_array:
        for ki in range(mp.nkpts):
            for kj in range(mp.nkpts):
                # Lookup by vectors; stored CDERI ordering need not match the
                # order of the SCF k-point list.
                Lpq_ao = cderi_array.load(kpts[ki], kpts[kj])
                mo = np.asarray(
                    np.hstack((mo_coeff[ki], mo_coeff[kj])),
                    dtype=dtype,
                    order="F",
                )
                if dtype == np.double:
                    out = _ao2mo.nr_e2(
                        Lpq_ao,
                        mo,
                        (bra_start, bra_end, ket_start, ket_end),
                        aosym=("s1" if Lpq_ao.shape[-1] == nao**2 else "s2"),
                    )
                else:
                    if Lpq_ao[0].size != nao**2:
                        Lpq_ao = lib.unpack_tril(Lpq_ao)
                    Lpq_ao = np.asarray(
                        Lpq_ao, dtype=np.complex128, order="C"
                    )
                    out = _ao2mo.r_e2(
                        Lpq_ao,
                        mo,
                        (bra_start, bra_end, ket_start, ket_end),
                        tao,
                        None,
                    )
                Lov[ki, kj] = out.reshape(-1, nocc, nvir)
    return Lov


KMP2 = KMP2_STC
