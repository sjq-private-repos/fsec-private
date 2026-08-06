"""Helpers for occupied-block direct-RSDF structure-factor construction."""

import numpy as np

from pyscf import lib
from pyscf.pbc import df
from pyscf.pbc.mp import kmp2

from fsec.singularity_subtraction.grids import minimum_image
from fsec.singularity_subtraction.structure_factor.helpers import (
    should_compute_line_sample,
    update_line_sampling_decay_mask,
)
from fsec.singularity_subtraction.structure_factor.mp2_contractions import (
    contract_trs_unique_pair_rijab_lov,
    contract_trs_unique_pair_rijab_lov_laplace,
)
from fsec.singularity_subtraction.mp2_variants import MP2Variant, get_mp2_variant


DIRECT_RSDF_TARGET_COMMIT = "5843306770f1df0ba9fb3f5a95c9b267683f2f86"


def is_direct_rsdf(kmp):
    """Return whether *kmp* is backed by the experimental direct RSGDF."""
    mydf = getattr(getattr(kmp, "_scf", None), "with_df", None)
    return isinstance(mydf, df.RSDF) and bool(getattr(mydf, "direct", False))


def validate_direct_rsdf_structure_factor(kmp, grids=None):
    """Validate the deliberately narrow first direct-RSDF implementation."""
    mydf = kmp._scf.with_df
    if kmp._scf.cell.dimension != 3:
        raise NotImplementedError(
            "The occupied-block direct-RSDF structure factor currently "
            "supports only three-dimensional cells")
    if bool(getattr(mydf, "semidirect", False)):
        raise NotImplementedError(
            "The occupied-block direct-RSDF structure factor requires "
            "semidirect=False")
    if str(getattr(mydf, "ksym", "s2")).lower() != "s2":
        raise NotImplementedError(
            "The occupied-block direct-RSDF structure factor currently "
            "requires ksym='s2'")
    kpts = np.asarray(kmp.kpts)
    if kpts.ndim != 2 or kpts.shape[1] != 3:
        raise NotImplementedError(
            "The occupied-block direct-RSDF structure factor requires an "
            "explicit regular k-point array")
    if grids is not None and grids.kGrid3_neq_kGrid2:
        raise NotImplementedError(
            "The occupied-block direct-RSDF structure factor is not "
            "implemented for kGrid3_neq_kGrid2")


def build_direct_rsdf_lov_block(kmp, occ_slice):
    """Build one occupied-sliced Lov table through the PySCF fork API."""
    try:
        from pyscf.pbc.mp import kmp2_direct
    except ImportError as error:
        raise ImportError(
            "Direct RSDF structure-factor construction requires "
            "pyscf.pbc.mp.kmp2_direct from the PySCF fork at commit "
            f"{DIRECT_RSDF_TARGET_COMMIT}"
        ) from error
    initializer = getattr(kmp2_direct, "_init_mp_df_eris", None)
    if initializer is None:
        raise ImportError(
            "The installed pyscf.pbc.mp.kmp2_direct does not provide "
            "_init_mp_df_eris(..., oslice=...). Use PySCF fork commit "
            f"{DIRECT_RSDF_TARGET_COMMIT}"
        )
    mo_coeff_padded, _ = kmp2._add_padding(
        kmp, kmp.mo_coeff, kmp.mo_energy)
    return initializer(
        kmp,
        mo_coeff=mo_coeff_padded,
        j3c_order="Lij",
        oslice=tuple(int(value) for value in occ_slice),
    )


def estimate_direct_rsdf_block_memory_bytes(
        nkpts, naux, nvir, block_size, ngrid,
        nocc=None, laplace=True, max_laplace_points=16):
    """Conservative incremental peak estimate for two occupied Lov blocks.

    The orbital grids have already been constructed when this estimate is
    evaluated, so they are represented by the process's current memory rather
    than counted a second time here.
    """
    nkpts = int(nkpts)
    naux = int(naux)
    nvir = int(nvir)
    block_size = int(block_size)
    ngrid = int(ngrid)
    nocc = block_size if nocc is None else int(nocc)
    complex_bytes = np.dtype(np.complex128).itemsize
    real_bytes = np.dtype(np.float64).itemsize

    nlove_tables = 1 if block_size >= nocc else 2
    lov = (
        nlove_tables * complex_bytes * nkpts**2
        * naux * block_size * nvir
    )
    # Occupied-grid phase product and both directed pair-density blocks.
    pair_density = complex_bytes * nkpts * block_size * (
        2 * ngrid + 4 * nvir)
    # The exact fallback holds a denominator and an exchange/direct matrix.
    exact_fallback = (
        2 * complex_bytes * (block_size * nvir)**2
    )
    # Laplace exponentials and auxiliary-by-quadrature intermediates.
    laplace_scratch = 0
    if laplace:
        nlaplace = max(1, int(max_laplace_points))
        laplace_scratch = (
            2 * real_bytes * nlaplace * block_size * nvir
            + 4 * complex_bytes * naux * nlaplace
        )
    return int(lov + pair_density + exact_fallback + laplace_scratch)


def select_direct_rsdf_occ_block_size(
        *, nocc, nkpts, naux, nvir, ngrid, available_bytes,
        requested=None, laplace=True, max_laplace_points=16):
    """Select and preflight the direct-RSDF occupied block size."""
    nocc = int(nocc)
    if nocc < 1:
        raise ValueError(
            "The direct-RSDF route requires at least one occupied orbital")
    available_bytes = max(0, int(available_bytes))
    candidates = (
        [min(nocc, int(requested))]
        if requested is not None
        else range(nocc, 0, -1)
    )
    for block_size in candidates:
        if block_size < 1:
            raise ValueError("rsdf_occ_block_size must be positive or None")
        estimate = estimate_direct_rsdf_block_memory_bytes(
            nkpts,
            naux,
            nvir,
            block_size,
            ngrid,
            nocc=nocc,
            laplace=laplace,
            max_laplace_points=max_laplace_points,
        )
        if estimate <= available_bytes:
            return block_size, estimate
    if requested is not None:
        requested_size = min(nocc, int(requested))
        requested_estimate = estimate_direct_rsdf_block_memory_bytes(
            nkpts,
            naux,
            nvir,
            requested_size,
            ngrid,
            nocc=nocc,
            laplace=laplace,
            max_laplace_points=max_laplace_points,
        )
        raise MemoryError(
            "Insufficient memory for occupied-block direct RSDF: requested "
            f"B={requested_size} requires approximately "
            f"{requested_estimate / 1024**2:.1f} MiB, but only "
            f"{available_bytes / 1024**2:.1f} MiB is available after the "
            "orbital-grid reservation."
        )
    minimum = estimate_direct_rsdf_block_memory_bytes(
        nkpts,
        naux,
        nvir,
        1,
        ngrid,
        nocc=nocc,
        laplace=laplace,
        max_laplace_points=max_laplace_points,
    )
    raise MemoryError(
        "Insufficient memory for occupied-block direct RSDF: "
        f"automatic block selection has "
        f"{available_bytes / 1024**2:.1f} MiB available; "
        f"even B=1 requires approximately {minimum / 1024**2:.1f} MiB "
        "in addition to the resident orbital grids. Reduce the pair-density "
        "grid/cutoff or increase kmp.max_memory."
    )


def accumulate_direct_rsdf_occ_blocks(
        *, kmp, kmf, qG_full, qi_map, qGrid, kGrid1, kas, kbs,
        qG_uses_outer_mesh, mesh_contexts, trs_map, trs_unique_pairs,
        mo_e_o, mo_e_v, mo_e_v_b, nonzero_opadding, nonzero_vpadding,
        omega_cell, direct, exchange, dG0, sq_inversion_symm,
        inversion_partner, requested_block_size, laplace,
        laplace_direct_tol, laplace_direct_max_points,
        laplace_exchange_tol, laplace_exchange_max_points,
        SqG_full_direct, SqG_full_exchange, SqG_full_q4,
        SqG_full_direct_mask, SqG_full_exchange_mask, SqG_full_q4_mask,
        q4_decay_state, exchange_decay_state, profile, log, variant=None):
    """Accumulate all occupied-block pairs for the fully direct RSDF route."""
    if variant is None:
        variant = get_mp2_variant()
    elif not isinstance(variant, MP2Variant):
        raise TypeError("variant must be an MP2Variant or None")
    nkpts = int(kmp.nkpts)
    nocc = int(kmp.nocc)
    nvir = int(kmp.nmo - kmp.nocc)
    mydf = kmp._scf.with_df
    auxcell = getattr(mydf, "auxcell", None)
    if auxcell is None:
        from pyscf.pbc.df import incore as pbc_df_incore
        auxcell = pbc_df_incore.make_auxcell(
            kmp._scf.cell, getattr(mydf, "auxbasis", None))
    naux = int(auxcell.nao_nr())
    ngrid = max(
        int(context["rptGrid3D"].shape[0])
        for context in mesh_contexts.values()
    )
    current_memory_mb = float(lib.current_memory()[0])
    available_bytes = int(
        max(0.0, kmp.max_memory - current_memory_mb) * 1e6 * 0.8)
    block_size, estimated_bytes = select_direct_rsdf_occ_block_size(
        nocc=nocc,
        nkpts=nkpts,
        naux=naux,
        nvir=nvir,
        ngrid=ngrid,
        available_bytes=available_bytes,
        requested=requested_block_size,
        laplace=laplace,
        max_laplace_points=max(
            laplace_direct_max_points, laplace_exchange_max_points),
    )
    occ_slices = [
        (start, min(start + block_size, nocc))
        for start in range(0, nocc, block_size)
    ]
    log.note(
        "Direct-RSDF occupied blocks: B=%d, %d blocks, estimated "
        "incremental peak %.1f MiB (80%% memory budget %.1f MiB)",
        block_size,
        len(occ_slices),
        estimated_bytes / 1024**2,
        available_bytes / 1024**2,
    )

    independent_qG = []
    mirror_qG = []
    for qG_index in range(len(qG_full)):
        partner = (
            int(inversion_partner[qG_index])
            if inversion_partner is not None else len(qG_full)
        )
        if sq_inversion_symm and partner < qG_index:
            mirror_qG.append((qG_index, partner))
        else:
            independent_qG.append(qG_index)

    contraction = (
        contract_trs_unique_pair_rijab_lov_laplace
        if laplace else contract_trs_unique_pair_rijab_lov
    )
    laplace_kwargs = (
        {
            "direct_tolerance": laplace_direct_tol,
            "direct_max_points": laplace_direct_max_points,
            "exchange_tolerance": laplace_exchange_tol,
            "exchange_max_points": laplace_exchange_max_points,
        }
        if laplace else {}
    )

    def pair_densities(qG_index, occ_slice):
        mesh_context = (
            mesh_contexts["outer"]
            if qG_uses_outer_mesh[qG_index]
            else mesh_contexts["inner"]
        )
        qGpt = qG_full[qG_index]
        qi = int(qi_map[qG_index])
        kptas = kGrid1 + qGpt
        kptas_BZ = minimum_image(kmf.cell, kptas)
        kGdiffas = kptas - kptas_BZ
        exp_term_as = np.exp(
            -1j * (mesh_context["rptGrid3D"] @ kGdiffas.T)).T
        start, stop = occ_slice
        phased_conj_ui = (
            np.conj(mesh_context["uKpts_i"][:, start:stop, :])
            * exp_term_as[:, None, :]
        )
        weights = mesh_context["quadrature_weights"]
        if weights is not None:
            phased_conj_ui *= weights[None, None, :]
        rho_ia = (
            phased_conj_ui
            @ mesh_context["uKpts_a"][kas[qi]].transpose(0, 2, 1)
        )
        rho_jb = rho_ia[trs_map].transpose(0, 2, 1)
        return (
            rho_ia,
            rho_jb,
            qi,
            mesh_context["quadrature_product_scale"],
        )

    build_count = 0
    for block_i, occ_slice_i in enumerate(occ_slices):
        build_t0 = profile.start()
        Lov_i = build_direct_rsdf_lov_block(kmp, occ_slice_i)
        build_count += 1
        profile.stop("direct-RSDF occupied Lov build", build_t0)
        for block_j in range(block_i + 1):
            occ_slice_j = occ_slices[block_j]
            if block_j == block_i:
                Lov_j = Lov_i
            else:
                build_t0 = profile.start()
                Lov_j = build_direct_rsdf_lov_block(kmp, occ_slice_j)
                build_count += 1
                profile.stop("direct-RSDF occupied Lov build", build_t0)

            for qG_index in independent_qG:
                rho_i, rho_i_b, qi, quadrature_scale = pair_densities(
                    qG_index, occ_slice_i)
                if block_i == block_j:
                    rho_j, rho_j_b = rho_i, rho_i_b
                else:
                    rho_j, rho_j_b, qi_j, quadrature_scale_j = (
                        pair_densities(qG_index, occ_slice_j)
                    )
                    if qi_j != qi or quadrature_scale_j != quadrature_scale:
                        raise RuntimeError(
                            "Inconsistent q-point pair-density block context")

                contraction_t0 = profile.start()
                directed_values = contraction(
                    rho_i,
                    rho_j_b,
                    trs_unique_pairs,
                    Lov_i,
                    Lov_j,
                    kas[qi],
                    kbs[qi],
                    mo_e_o,
                    mo_e_v,
                    mo_e_v_b,
                    nonzero_opadding,
                    nonzero_vpadding,
                    nkpts,
                    compute_direct=direct,
                    compute_exchange=exchange,
                    compute_q4=dG0,
                    profile=profile,
                    occ_slice_i=occ_slice_i,
                    occ_slice_j=occ_slice_j,
                    Lov_exchange_i=Lov_i,
                    Lov_exchange_j=Lov_j,
                    variant=variant,
                    **laplace_kwargs,
                )
                if block_i != block_j:
                    reverse_values = contraction(
                        rho_j,
                        rho_i_b,
                        trs_unique_pairs,
                        Lov_j,
                        Lov_i,
                        kas[qi],
                        kbs[qi],
                        mo_e_o,
                        mo_e_v,
                        mo_e_v_b,
                        nonzero_opadding,
                        nonzero_vpadding,
                        nkpts,
                        compute_direct=direct,
                        compute_exchange=exchange,
                        compute_q4=dG0,
                        profile=profile,
                        occ_slice_i=occ_slice_j,
                        occ_slice_j=occ_slice_i,
                        Lov_exchange_i=Lov_j,
                        Lov_exchange_j=Lov_i,
                        variant=variant,
                        **laplace_kwargs,
                    )
                    directed_values = tuple(
                        left + right
                        for left, right in zip(
                            directed_values, reverse_values)
                    )
                profile.stop(
                    "direct-RSDF occupied-pair contraction", contraction_t0)

                direct_sum, exchange_sum, q4_weighted_norm = directed_values
                if direct:
                    SqG_full_direct[qG_index] += (
                        2 * quadrature_scale * direct_sum
                        / (omega_cell * nkpts**2)
                    )
                if exchange:
                    SqG_full_exchange[qG_index] += (
                        -quadrature_scale * exchange_sum
                        / (omega_cell * nkpts**2)
                    )
                if dG0:
                    scale = quadrature_scale / (nkpts * omega_cell)
                    SqG_full_q4[qG_index] += (
                        -2 * scale**2 * q4_weighted_norm / nkpts
                    ).real

            if block_j != block_i:
                Lov_j = None
        Lov_j = None
        Lov_i = None

    for qG_index, partner in mirror_qG:
        SqG_full_direct[qG_index] = SqG_full_direct[partner]
        SqG_full_exchange[qG_index] = SqG_full_exchange[partner]
        SqG_full_q4[qG_index] = SqG_full_q4[partner]

    # Decay decisions require the fully accumulated structure factor in the
    # blocked route, so they are deliberately applied only after all blocks.
    for qG_index, qGpt in enumerate(qG_full):
        if (
            exchange
            and should_compute_line_sample(qG_index, exchange_decay_state)
        ):
            update_line_sampling_decay_mask(
                SqG_full_exchange_mask,
                qG_index,
                SqG_full_exchange[qG_index],
                np.linalg.norm(qGpt),
                exchange_decay_state,
                qGpt=qGpt,
            )
        if dG0 and should_compute_line_sample(qG_index, q4_decay_state):
            update_line_sampling_decay_mask(
                SqG_full_q4_mask,
                qG_index,
                SqG_full_q4[qG_index],
                np.linalg.norm(qGpt),
                q4_decay_state,
                qGpt=qGpt,
            )
            if direct and q4_decay_state is not None:
                SqG_full_direct_mask[qG_index] = (
                    SqG_full_q4_mask[qG_index])

    expected_build_count = len(occ_slices) * (len(occ_slices) + 1) // 2
    if build_count != expected_build_count:
        raise RuntimeError(
            "Unexpected direct-RSDF Lov build count: "
            f"{build_count} != {expected_build_count}")
    return len(mirror_qG), block_size, estimated_bytes, build_count
