"""Contraction kernels for the MP2 structure factor."""

import numpy as np
from pyscf.lib.parameters import LARGE_DENOM

from fsec.singularity_subtraction.structure_factor.laplace_minimax import (
    make_minimax_laplace_grid,
)


def build_trs_unique_pairs(trs_map, nkpts=None):
    r"""Build canonical k-point pairs that exploit TRS symmetry.

    The unique-pair contraction uses the conjugation symmetries

    .. math::

        \left[r_{ijab\mathbf{k}_i\mathbf{k}_j}(\mathbf{q}+\mathbf{G})\right]^*
        = r_{jiba(-\mathbf{k}_j)(-\mathbf{k}_i)}(\mathbf{q}+\mathbf{G})

    and

    .. math::

        \left[T^{(2)}_{ijab\mathbf{k}_i\mathbf{k}_j}(\mathbf{q})\right]^*
        = T^{(2)}_{jiba(-\mathbf{k}_j)(-\mathbf{k}_i)}(\mathbf{q}).

    Thus, the pair ``(m, n)`` and its TRS partner
    ``(trs_map[n], trs_map[m])`` contribute equally to the real contraction.
    Self-partner pairs receive a factor of one; all other canonical pairs
    receive a factor of two.
    """
    trs_map = np.asarray(trs_map, dtype=int)
    if nkpts is None:
        nkpts = len(trs_map)
    unique_pairs = []
    for m in range(nkpts):
        for n in range(nkpts):
            flat = m * nkpts + n
            partner_m = trs_map[n]
            partner_n = trs_map[m]
            partner_flat = partner_m * nkpts + partner_n
            if flat <= partner_flat:
                factor = 1 if flat == partner_flat else 2
                unique_pairs.append((m, n, factor))
    return unique_pairs


def contract_trs_unique_pair_rijab(
        rho_ia_full, rho_jb_full, trs_unique_pairs,
        direct_t2=None, exchange_t2=None, eijab_recip=None, profile=None):
    """Contract rijab/t2 over TRS-unique pairs without full materialization."""
    direct_value = 0.0
    exchange_value = 0.0
    q4_weighted_norm = 0.0

    for m, n, factor in trs_unique_pairs:
        if direct_t2 is not None:
            region_t0 = profile.start() if profile is not None else None
            tmp_jb = np.einsum(
                'ia,ijab->jb',
                rho_ia_full[m],
                direct_t2[m, n],
                optimize=True,
            )
            direct_block = np.einsum(
                'jb,bj->',
                tmp_jb,
                rho_jb_full[n].conj(),
                optimize=True,
            )
            direct_value += factor * direct_block.real
            if profile is not None:
                profile.stop("TRS direct rho/t2 contraction", region_t0)
        if exchange_t2 is not None:
            region_t0 = profile.start() if profile is not None else None
            tmp_jb = np.einsum(
                'ia,ijba->jb',
                rho_ia_full[m],
                exchange_t2[m, n],
                optimize=True,
            )
            exchange_block = np.einsum(
                'jb,bj->',
                tmp_jb,
                rho_jb_full[n].conj(),
                optimize=True,
            )
            exchange_value += factor * exchange_block.real
            if profile is not None:
                profile.stop("TRS exchange rho/t2 contraction", region_t0)
        if eijab_recip is not None:
            region_t0 = profile.start() if profile is not None else None
            tmp_jb = np.einsum(
                'ia,ijab->jb',
                np.abs(rho_ia_full[m])**2,
                np.abs(eijab_recip[m, n]),
                optimize=True,
            )
            q4_weighted_norm += factor * np.einsum(
                'jb,bj->',
                tmp_jb,
                np.abs(rho_jb_full[n])**2,
                optimize=True,
            )
            if profile is not None:
                profile.stop("TRS dG0 rho/denominator contraction", region_t0)

    return direct_value, exchange_value, q4_weighted_norm


def lov_block(Lov, ko, kv):
    """Return a (naux, nocc, nvir) Lov block for supported layouts."""
    Lov = np.asarray(Lov)
    if Lov.ndim == 2:
        return np.asarray(Lov[ko, kv])
    if Lov.ndim == 4:
        return np.asarray(Lov[ko])
    raise ValueError("Lov must be a 2D object array or a dense 4D array")


def build_eov_pair(
        ko, kv, mo_e_o, mo_e_v, nonzero_opadding, nonzero_vpadding):
    """Build a padded occupied-virtual energy difference block."""
    nocc = mo_e_o.shape[1]
    nvir = mo_e_v.shape[1]
    eov = LARGE_DENOM * np.ones((nocc, nvir), dtype=mo_e_o.dtype)
    nonzero_idx = np.ix_(nonzero_opadding[ko], nonzero_vpadding[kv])
    eov[nonzero_idx] = (mo_e_o[ko][:, None] - mo_e_v[kv])[nonzero_idx]
    return eov


def contract_exchange_lov_laplace(
        rho_ia, rho_jb, Lov_mkb, Lov_nka, eia, ejb,
        active_ia, active_jb, tolerance, max_points):
    """Contract one exchange block using a separable minimax denominator.

    The signed MP2 denominator is ``-(gap_ia + gap_jb)``.  Its minimax
    expansion separates the virtual labels, allowing two ``O x V`` by
    ``V x O`` products per auxiliary index and Laplace point instead of
    constructing an ``(OV) x (OV)`` ERI/denominator matrix.

    Returns
    -------
    tuple or None
        ``(exchange_block, grid)`` or ``None`` if the minimax tables do
        not cover the requested denominator range and tolerance.
    """
    active_ia = np.asarray(active_ia, dtype=bool)
    active_jb = np.asarray(active_jb, dtype=bool)
    if not np.any(active_ia) or not np.any(active_jb):
        return 0.0j, None

    gap_ia = np.full(eia.shape, np.inf, dtype=eia.dtype)
    gap_jb = np.full(ejb.shape, np.inf, dtype=ejb.dtype)
    gap_ia[active_ia] = -eia[active_ia]
    gap_jb[active_jb] = -ejb[active_jb]
    active_gaps_ia = gap_ia[active_ia]
    active_gaps_jb = gap_jb[active_jb]
    if np.any(active_gaps_ia <= 0) or np.any(active_gaps_jb <= 0):
        return None

    denominator_min = active_gaps_ia.min() + active_gaps_jb.min()
    denominator_max = active_gaps_ia.max() + active_gaps_jb.max()
    grid = make_minimax_laplace_grid(
        denominator_min,
        denominator_max,
        tolerance=tolerance,
        max_points=max_points,
    )
    if grid is None:
        return None

    exchange_block = 0.0j
    Lov_nka_t = Lov_nka.transpose(0, 2, 1)
    for point, weight in zip(grid.points, grid.weights):
        scaled_rho_ia = rho_ia * np.exp(-point * gap_ia)
        scaled_rho_bj = rho_jb.conj() * np.exp(-point * gap_jb).T

        # Both intermediates have shape (naux, nocc, nocc).  The work is
        # O(naux * nocc**2 * nvir) per Laplace point and the peak scratch
        # is O(naux * nocc**2), independent of nvir**2.
        left_lij = Lov_mkb @ scaled_rho_bj
        right_lij = np.matmul(scaled_rho_ia, Lov_nka_t)
        exchange_block -= weight * np.einsum(
            'Lij,Lij->', left_lij, right_lij, optimize=True)

    return exchange_block, grid


def contract_direct_q4_lov_laplace(
        rho_ia, rho_jb, Lov_mka, Lov_nkb, eia, ejb,
        active_ia, active_jb, compute_direct, compute_q4,
        tolerance, max_points):
    """Contract direct and q4 blocks with a separable minimax denominator.

    The direct contraction is evaluated as two batched ``A x OV`` by
    ``OV x n`` products.  The q4 contraction reuses the same exponential
    factors, so neither component materializes an ``(OV) x (OV)``
    reciprocal-denominator matrix.

    Returns
    -------
    tuple or None
        ``(direct_block, q4_block, grid)`` or ``None`` when the active
        gaps or minimax tables require the exact fallback.
    """
    active_ia = np.asarray(active_ia, dtype=bool)
    active_jb = np.asarray(active_jb, dtype=bool)
    if not np.any(active_ia) or not np.any(active_jb):
        return 0.0j, 0.0, None

    gap_ia = np.full(eia.shape, np.inf, dtype=eia.dtype)
    gap_jb = np.full(ejb.shape, np.inf, dtype=ejb.dtype)
    gap_ia[active_ia] = -eia[active_ia]
    gap_jb[active_jb] = -ejb[active_jb]
    active_gaps_ia = gap_ia[active_ia]
    active_gaps_jb = gap_jb[active_jb]
    if np.any(active_gaps_ia <= 0) or np.any(active_gaps_jb <= 0):
        return None

    grid = make_minimax_laplace_grid(
        active_gaps_ia.min() + active_gaps_jb.min(),
        active_gaps_ia.max() + active_gaps_jb.max(),
        tolerance=tolerance,
        max_points=max_points,
    )
    if grid is None:
        return None

    exp_ia = np.exp(-grid.points[:, None] * gap_ia.ravel()[None, :])
    exp_jb = np.exp(-grid.points[:, None] * gap_jb.ravel()[None, :])

    direct_block = 0.0j
    if compute_direct:
        rho_ia_weighted = exp_ia * rho_ia.ravel()[None, :]
        rho_jb_weighted = exp_jb * rho_jb.conj().T.ravel()[None, :]
        left_ln = Lov_mka.reshape(Lov_mka.shape[0], -1) @ rho_ia_weighted.T
        right_ln = Lov_nkb.reshape(Lov_nkb.shape[0], -1) @ rho_jb_weighted.T
        direct_by_point = np.einsum(
            'Ln,Ln->n', left_ln, right_ln, optimize=True)
        direct_block = -np.dot(grid.weights, direct_by_point)

    q4_block = 0.0
    if compute_q4:
        rho_ia_abs = np.abs(rho_ia.ravel())**2
        rho_jb_abs = np.abs(rho_jb.conj().T.ravel())**2
        q4_ia = exp_ia @ rho_ia_abs
        q4_jb = exp_jb @ rho_jb_abs
        q4_block = np.dot(grid.weights, q4_ia * q4_jb)

    return direct_block, q4_block, grid


def contract_trs_unique_pair_rijab_lov(
        rho_ia_full, rho_jb_full, trs_unique_pairs,
        Lov, Lov_b, kas_at_qi, kbs_at_qi, mo_e_o, mo_e_v, mo_e_v_b,
        nonzero_opadding, nonzero_vpadding, nkpts,
        compute_direct=False, compute_exchange=False, compute_q4=False,
        profile=None):
    """Contract TRS-unique pairs exactly from DF Lov blocks."""
    direct_value = 0.0
    exchange_value = 0.0
    q4_weighted_norm = 0.0

    for m, n, factor in trs_unique_pairs:
        ka = int(kas_at_qi[m])
        kb = int(kbs_at_qi[n])

        if compute_direct or compute_exchange or compute_q4:
            region_t0 = profile.start() if profile is not None else None
            eia = build_eov_pair(
                m, ka, mo_e_o, mo_e_v, nonzero_opadding, nonzero_vpadding)
            ejb = build_eov_pair(
                n, kb, mo_e_o, mo_e_v_b, nonzero_opadding, nonzero_vpadding)
            if profile is not None:
                profile.stop("TRS Lov denominator eov build", region_t0)

        Lov_mka = None
        Lov_nkb = None
        if compute_direct:
            region_t0 = profile.start() if profile is not None else None
            Lov_mka = lov_block(Lov, m, ka).conj()
            Lov_nkb = lov_block(Lov_b, n, kb).conj()
            if profile is not None:
                profile.stop("TRS Lov direct Lov fetch/conj", region_t0)

        edenom_matrix = None
        if compute_direct or compute_q4:
            region_t0 = profile.start() if profile is not None else None
            edenom_matrix = np.add.outer(eia.ravel(), ejb.ravel())
            np.reciprocal(edenom_matrix, out=edenom_matrix)
            if profile is not None:
                profile.stop("TRS Lov direct/q4 denominator build", region_t0)

        if compute_direct:
            region_t0 = profile.start() if profile is not None else None
            x_lia = Lov_mka * rho_ia_full[m][None, :, :]
            y_ljb = Lov_nkb * rho_jb_full[n].conj().T[None, :, :]
            if profile is not None:
                profile.stop("TRS Lov direct rho scale", region_t0)

            region_t0 = profile.start() if profile is not None else None
            naux = x_lia.shape[0]
            x_lia = x_lia.reshape(naux, -1)
            y_ljb = y_ljb.reshape(naux, -1)
            if profile is not None:
                profile.stop("TRS Lov direct reshape", region_t0)

            region_t0 = profile.start() if profile is not None else None
            xE_ljb = x_lia @ edenom_matrix
            if profile is not None:
                profile.stop("TRS Lov direct X@denom", region_t0)

            region_t0 = profile.start() if profile is not None else None
            direct_block = np.sum(xE_ljb * y_ljb)
            direct_value += factor * (direct_block / nkpts).real
            if profile is not None:
                profile.stop("TRS Lov direct reduce", region_t0)

        if compute_exchange:
            region_t0 = profile.start() if profile is not None else None
            Lov_mkb = lov_block(Lov_b, m, kb).conj()
            Lov_nka = lov_block(Lov, n, ka).conj()
            if profile is not None:
                profile.stop("TRS Lov exchange Lov fetch/conj", region_t0)

            region_t0 = profile.start() if profile is not None else None
            naux, nocc, nvir = Lov_mkb.shape
            eris_ijba = (
                Lov_mkb.transpose(1, 2, 0).reshape(nocc * nvir, naux)
                @ Lov_nka.reshape(naux, nocc * nvir)
            )
            if profile is not None:
                profile.stop("TRS Lov exchange ERI matmul", region_t0)

            region_t0 = profile.start() if profile is not None else None
            exchange_edenom = np.empty(
                (nocc, nvir, nocc, nvir), dtype=eia.dtype)
            np.add(
                eia[:, None, None, :],
                ejb.T[None, :, :, None],
                out=exchange_edenom,
            )
            np.reciprocal(exchange_edenom, out=exchange_edenom)
            if profile is not None:
                profile.stop("TRS Lov exchange denominator build", region_t0)

            region_t0 = profile.start() if profile is not None else None
            eris_ib_ja = eris_ijba.reshape(nocc, nvir, nocc, nvir)
            np.multiply(eris_ib_ja, exchange_edenom, out=eris_ib_ja)
            if profile is not None:
                profile.stop("TRS Lov exchange denom scale", region_t0)

            region_t0 = profile.start() if profile is not None else None
            tmp_ia = np.einsum(
                'ibja,bj->ia',
                eris_ib_ja,
                rho_jb_full[n].conj(),
                optimize=True,
            )
            exchange_block = np.einsum(
                'ia,ia->', rho_ia_full[m], tmp_ia, optimize=True)
            if profile is not None:
                profile.stop("TRS Lov exchange exact contraction", region_t0)

            region_t0 = profile.start() if profile is not None else None
            exchange_value += factor * (exchange_block / nkpts).real
            if profile is not None:
                profile.stop("TRS Lov exchange rho dot", region_t0)

        if compute_q4:
            region_t0 = profile.start() if profile is not None else None
            rho_ia_abs = np.abs(rho_ia_full[m])**2
            rho_jb_abs = np.abs(rho_jb_full[n])**2
            # No later contraction needs the signed denominator, so reuse
            # its storage for the absolute value required by q4.
            np.abs(edenom_matrix, out=edenom_matrix)
            tmp_jb = rho_ia_abs.reshape(-1) @ edenom_matrix
            q4_weighted_norm += factor * np.dot(
                tmp_jb, rho_jb_abs.T.reshape(-1))
            if profile is not None:
                profile.stop(
                    "TRS Lov dG0 rho/denominator contraction", region_t0)

    return direct_value, exchange_value, q4_weighted_norm


def contract_trs_unique_pair_rijab_lov_laplace(
        rho_ia_full, rho_jb_full, trs_unique_pairs,
        Lov, Lov_b, kas_at_qi, kbs_at_qi, mo_e_o, mo_e_v, mo_e_v_b,
        nonzero_opadding, nonzero_vpadding, nkpts,
        compute_direct=False, compute_exchange=False, compute_q4=False,
        profile=None,
        direct_tolerance=1e-8, direct_max_points=16,
        exchange_tolerance=1e-8, exchange_max_points=16):
    """Contract TRS-unique pairs with Laplace denominators.

    Any unique pair that is not covered by the minimax grid is passed
    to the exact contraction for the affected components.
    """
    direct_value = 0.0
    exchange_value = 0.0
    q4_weighted_norm = 0.0

    for m, n, factor in trs_unique_pairs:
        ka = int(kas_at_qi[m])
        kb = int(kbs_at_qi[n])

        region_t0 = profile.start() if profile is not None else None
        eia = build_eov_pair(
            m, ka, mo_e_o, mo_e_v, nonzero_opadding, nonzero_vpadding)
        ejb = build_eov_pair(
            n, kb, mo_e_o, mo_e_v_b,
            nonzero_opadding, nonzero_vpadding)
        if profile is not None:
            profile.stop("TRS Lov denominator eov build", region_t0)

        active_ia = np.zeros(eia.shape, dtype=bool)
        active_jb = np.zeros(ejb.shape, dtype=bool)
        active_ia[np.ix_(
            nonzero_opadding[m], nonzero_vpadding[ka])] = True
        active_jb[np.ix_(
            nonzero_opadding[n], nonzero_vpadding[kb])] = True

        direct_result = None
        if compute_direct or compute_q4:
            Lov_mka = None
            Lov_nkb = None
            if compute_direct:
                region_t0 = profile.start() if profile is not None else None
                Lov_mka = lov_block(
                    Lov, m, ka).conj()
                Lov_nkb = lov_block(
                    Lov_b, n, kb).conj()
                if profile is not None:
                    profile.stop(
                        "TRS Lov direct Lov fetch/conj", region_t0)

            region_t0 = profile.start() if profile is not None else None
            direct_result = (
                contract_direct_q4_lov_laplace(
                    rho_ia_full[m],
                    rho_jb_full[n],
                    Lov_mka,
                    Lov_nkb,
                    eia,
                    ejb,
                    active_ia,
                    active_jb,
                    compute_direct,
                    compute_q4,
                    direct_tolerance,
                    direct_max_points,
                )
            )
            if profile is not None:
                label = (
                    "TRS Lov direct/q4 Laplace contraction"
                    if direct_result is not None
                    else "TRS Lov direct/q4 Laplace fallback"
                )
                profile.stop(label, region_t0)

            if direct_result is not None:
                direct_block, q4_block, _ = direct_result
                if compute_direct:
                    direct_value += factor * (direct_block / nkpts).real
                if compute_q4:
                    q4_weighted_norm += factor * q4_block

        exchange_result = None
        if compute_exchange:
            region_t0 = profile.start() if profile is not None else None
            Lov_mkb = lov_block(
                Lov_b, m, kb).conj()
            Lov_nka = lov_block(Lov, n, ka).conj()
            if profile is not None:
                profile.stop("TRS Lov exchange Lov fetch/conj", region_t0)

            region_t0 = profile.start() if profile is not None else None
            exchange_result = (
                contract_exchange_lov_laplace(
                    rho_ia_full[m],
                    rho_jb_full[n],
                    Lov_mkb,
                    Lov_nka,
                    eia,
                    ejb,
                    active_ia,
                    active_jb,
                    exchange_tolerance,
                    exchange_max_points,
                )
            )
            if profile is not None:
                label = (
                    "TRS Lov exchange Laplace contraction"
                    if exchange_result is not None
                    else "TRS Lov exchange Laplace fallback"
                )
                profile.stop(label, region_t0)

            if exchange_result is not None:
                exchange_value += factor * (
                    exchange_result[0] / nkpts).real

        fallback_direct_q4 = (
            (compute_direct or compute_q4) and direct_result is None)
        fallback_exchange = compute_exchange and exchange_result is None
        if fallback_direct_q4 or fallback_exchange:
            fallback = (
                contract_trs_unique_pair_rijab_lov(
                    rho_ia_full,
                    rho_jb_full,
                    [(m, n, factor)],
                    Lov,
                    Lov_b,
                    kas_at_qi,
                    kbs_at_qi,
                    mo_e_o,
                    mo_e_v,
                    mo_e_v_b,
                    nonzero_opadding,
                    nonzero_vpadding,
                    nkpts,
                    compute_direct=compute_direct and fallback_direct_q4,
                    compute_exchange=fallback_exchange,
                    compute_q4=compute_q4 and fallback_direct_q4,
                    profile=profile,
                )
            )
            direct_value += fallback[0]
            exchange_value += fallback[1]
            q4_weighted_norm += fallback[2]

    return direct_value, exchange_value, q4_weighted_norm
