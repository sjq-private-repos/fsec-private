"""Focused numerical checks for stored sTC MP2 exchange."""

import copy

import numpy as np
import pytest

from pyscf import lib
from pyscf.lib.parameters import LARGE_DENOM
from pyscf.pbc import df, gto, scf
from pyscf.pbc.df import df_ao2mo
from pyscf.pbc.lib.kpts import KPoints
from pyscf.pbc.mp import kmp2 as pyscf_kmp2

from fsec.vcut import KMP2_HYBRID, KMP2_STC


def _make_reference(kmesh, scaled_center=None, two_he=False):
    cell = gto.Cell()
    cell.unit = "Angstrom"
    cell.atom = "He 0.0 0.0 0.0"
    if two_he:
        cell.atom += "; He 1.1 0.6 0.8"
    cell.a = np.eye(3) * 4.0
    cell.basis = "gth-dzvp"
    cell.pseudo = "gth-pade"
    cell.precision = 1e-7
    cell.mesh = [17, 17, 17]
    cell.verbose = 0
    cell.build()
    kpts = cell.make_kpts(kmesh, scaled_center=scaled_center)
    mf = scf.KRHF(cell, kpts)
    mf.verbose = 0
    mf.exxdiv = None
    mf.with_df = df.GDF(cell, kpts)
    mf.with_df.auxbasis = "weigend"
    mf.kernel()
    assert mf.converged
    return mf


@pytest.fixture(scope="module")
def gamma_reference():
    return _make_reference([1, 1, 1])


@pytest.fixture(scope="module")
def shifted_reference():
    return _make_reference(
        [3, 1, 1], scaled_center=[0.17, 0.0, 0.0], two_he=True
    )


def _explicit_components(mp, builder, mo_energy=None, mo_coeff=None):
    """Build V with AO2MO and contract both same-interaction components."""
    if mo_energy is None:
        mo_energy = mp.mo_energy
    if mo_coeff is None:
        mo_coeff = mp.mo_coeff

    helper = copy.copy(mp)
    helper.mo_energy = mo_energy
    coeffs = list(mo_coeff)
    dtype = np.result_type(*(np.asarray(coeff).dtype for coeff in coeffs))
    coeffs = [np.asarray(coeff, dtype=dtype) for coeff in coeffs]
    coeffs, energies = pyscf_kmp2._add_padding(helper, coeffs, mo_energy)
    coeffs = np.asarray(coeffs, dtype=dtype)
    nocc = helper.nocc
    nmo = helper.nmo
    nvir = nmo - nocc
    opad, vpad = pyscf_kmp2.padding_k_idx(helper, kind="split")
    kpts = np.asarray(mp.kpts)
    nkpts = mp.nkpts
    kconserv = mp.khelper.kconserv
    direct = 0.0
    exchange = 0.0
    t2_ref = np.zeros(
        (nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir), dtype=complex
    )

    for ki in range(nkpts):
        for kj in range(nkpts):
            eri_ij = np.empty((nkpts, nocc, nocc, nvir, nvir), dtype=complex)
            for ka in range(nkpts):
                kb = kconserv[ki, ka, kj]
                raw = df_ao2mo.general(
                    builder,
                    (
                        coeffs[ki, :, :nocc],
                        coeffs[ka, :, nocc:],
                        coeffs[kj, :, :nocc],
                        coeffs[kb, :, nocc:],
                    ),
                    (kpts[ki], kpts[ka], kpts[kj], kpts[kb]),
                    compact=False,
                )
                eri_ij[ka] = raw.reshape(
                    nocc, nvir, nocc, nvir
                ).transpose(0, 2, 1, 3) / nkpts

            for ka in range(nkpts):
                kb = kconserv[ki, ka, kj]
                eia = LARGE_DENOM * np.ones((nocc, nvir))
                idx = np.ix_(opad[ki], vpad[ka])
                eia[idx] = (energies[ki][:nocc, None] - energies[ka][None, nocc:])[idx]
                ejb = LARGE_DENOM * np.ones((nocc, nvir))
                idx = np.ix_(opad[kj], vpad[kb])
                ejb[idx] = (energies[kj][:nocc, None] - energies[kb][None, nocc:])[idx]
                denom = lib.direct_sum("ia,jb->ijab", eia, ejb)
                t2_ref[ki, kj, ka] = np.conj(eri_ij[ka] / denom)
                direct += 2.0 * np.einsum(
                    "ijab,ijab", t2_ref[ki, kj, ka], eri_ij[ka]
                ).real
                exchange -= np.einsum(
                    "ijab,ijba", t2_ref[ki, kj, ka], eri_ij[kb]
                ).real
    return direct / nkpts, exchange / nkpts, t2_ref


def _explicit_exchange(mp, mo_energy=None, mo_coeff=None):
    """Build sTC V with AO2MO and contract the exchange term."""
    _, exchange, t2 = _explicit_components(
        mp, mp.with_df, mo_energy=mo_energy, mo_coeff=mo_coeff
    )
    return exchange, t2


def _full_kernel_exchange(mf, stc_df, frozen=None):
    """Use installed kMP2 energy decomposition with the same stored factors."""
    _, exchange, t2 = _full_kernel_components(mf, stc_df, frozen=frozen)
    return exchange, t2


def _full_kernel_components(mf, builder, frozen=None):
    """Return PySCF's direct/exchange split using a selected DF builder."""
    ref = pyscf_kmp2.KMP2(mf, frozen=frozen)
    ref._scf = copy.copy(ref._scf)
    ref._scf.with_df = builder
    ref.kernel(with_t2=True)
    return 2.0 * ref.e_corr_os, ref.e_corr_ss - ref.e_corr_os, ref.t2


def test_gamma_exchange_matches_explicit_four_index_reference(gamma_reference):
    """The Gamma exchange contraction matches AO2MO four-index ERIs."""
    before_coeff = gamma_reference.mo_coeff.copy()
    before_energy = gamma_reference.mo_energy.copy()
    before_occ = gamma_reference.mo_occ.copy()
    original_df = gamma_reference.with_df

    mp = KMP2_STC(gamma_reference)
    assert mp.with_df is not original_df
    assert mp.with_df.auxbasis == original_df.auxbasis
    energy, t2 = mp.kernel(with_t2=True)
    explicit_energy, explicit_t2 = _explicit_exchange(mp)
    energy_no_t2, no_t2 = mp.kernel(with_t2=False)

    assert energy == pytest.approx(explicit_energy, abs=1e-10, rel=0)
    np.testing.assert_allclose(t2, explicit_t2, atol=1e-10, rtol=0)
    assert energy_no_t2 == pytest.approx(energy, abs=1e-12)
    assert no_t2 is None
    np.testing.assert_array_equal(gamma_reference.mo_coeff, before_coeff)
    np.testing.assert_array_equal(gamma_reference.mo_energy, before_energy)
    np.testing.assert_array_equal(gamma_reference.mo_occ, before_occ)
    assert gamma_reference.with_df is original_df


def test_shifted_complex_mesh_matches_explicit_and_full_kernel(shifted_reference):
    """Shifted complex Bloch factors match both explicit ERIs and kMP2."""
    mp = KMP2_STC(shifted_reference)
    energy, t2 = mp.kernel(with_t2=True)
    explicit_energy, explicit_t2 = _explicit_exchange(mp)
    decomposed_energy, decomposed_t2 = _full_kernel_exchange(
        shifted_reference, mp.with_df
    )

    assert any(np.iscomplexobj(coeff) for coeff in shifted_reference.mo_coeff)
    assert energy == pytest.approx(explicit_energy, abs=1e-10, rel=0)
    assert energy == pytest.approx(decomposed_energy, abs=1e-10, rel=0)
    np.testing.assert_allclose(t2, explicit_t2, atol=1e-10, rtol=0)
    np.testing.assert_allclose(t2, decomposed_t2, atol=1e-10, rtol=0)


def test_frozen_padding_and_kernel_orbital_overrides(shifted_reference):
    """Frozen virtual ranks and kernel-level orbital overrides are honored."""
    frozen = [[9], [8, 9], [9]]
    coeff_override = np.array(shifted_reference.mo_coeff, dtype=complex, copy=True)
    phase = np.exp(1j * np.array([0.0, 0.31, -0.23]))
    coeff_override[:, :, 2] *= phase[:, None]
    energy_override = np.array(shifted_reference.mo_energy, copy=True)
    energy_override[:, 2] += np.array([0.02, 0.04, 0.03])

    mp = KMP2_STC(shifted_reference, frozen=frozen)
    energy, t2 = mp.kernel(
        mo_energy=energy_override, mo_coeff=coeff_override, with_t2=True
    )
    explicit_energy, explicit_t2 = _explicit_exchange(
        mp, mo_energy=energy_override, mo_coeff=coeff_override
    )

    assert t2.shape[3:5] == (2, 2)
    assert t2.shape[-1] == t2.shape[-2]
    assert energy == pytest.approx(explicit_energy, abs=1e-10, rel=0)
    np.testing.assert_allclose(t2, explicit_t2, atol=1e-10, rtol=0)
    baseline = KMP2_STC(shifted_reference, frozen=frozen)
    _, baseline_t2 = baseline.kernel(with_t2=True)
    assert np.max(np.abs(t2 - baseline_t2)) > 1e-8

    occupied_frozen = KMP2_STC(shifted_reference, frozen=[0])
    occupied_energy, occupied_t2 = occupied_frozen.kernel(with_t2=True)
    reference_energy, reference_t2 = _full_kernel_exchange(
        shifted_reference, occupied_frozen.with_df, frozen=[0]
    )
    assert occupied_t2.shape[3:5] == (1, 1)
    assert occupied_energy == pytest.approx(reference_energy, abs=1e-10, rel=0)
    np.testing.assert_allclose(occupied_t2, reference_t2, atol=1e-10, rtol=0)


def test_cutoff_and_eta_change_exchange(shifted_reference):
    """Both supported cutoffs work, and eta changes the sTC factors."""
    configs = [
        ("vcut_ws", "ws", 4.0),
        ("vcut_ws", "ws", 3.0),
        ("vcut_sph", "sph", 4.0),
    ]
    energies = []
    for exxdiv, rc_type, eta in configs:
        mp = KMP2_STC(
            shifted_reference, exxdiv=exxdiv, rc_type=rc_type, eta=eta
        )
        energies.append(mp.kernel()[0])

    assert np.all(np.isfinite(energies))
    assert abs(energies[0] - energies[1]) > 1e-8
    assert abs(energies[0] - energies[2]) > 1e-8


@pytest.mark.parametrize("fixture_name", ["gamma_reference", "shifted_reference"])
def test_stc_direct_option_matches_explicit_and_pyscf(request, fixture_name):
    """The optional sTC direct term uses sTC in both ERI factors."""
    mf = request.getfixturevalue(fixture_name)
    mp = KMP2_STC(mf)
    energy, t2 = mp.kernel(with_t2=True, with_direct=True)
    direct, exchange, explicit_t2 = _explicit_components(mp, mp.with_df)
    full_direct, full_exchange, full_t2 = _full_kernel_components(
        mf, mp.with_df
    )

    assert mp.e_corr_direct == pytest.approx(direct, abs=1e-10, rel=0)
    assert mp.e_corr_exchange == pytest.approx(exchange, abs=1e-10, rel=0)
    assert mp.e_corr_direct == pytest.approx(full_direct, abs=1e-10, rel=0)
    assert mp.e_corr_exchange == pytest.approx(full_exchange, abs=1e-10, rel=0)
    assert energy == pytest.approx(direct + exchange, abs=1e-10, rel=0)
    np.testing.assert_allclose(t2, explicit_t2, atol=1e-10, rtol=0)
    np.testing.assert_allclose(t2, full_t2, atol=1e-10, rtol=0)


@pytest.mark.parametrize("fixture_name", ["gamma_reference", "shifted_reference"])
def test_hybrid_matches_bare_direct_and_stc_exchange(request, fixture_name):
    """Hybrid components and amplitudes use their independent interactions."""
    mf = request.getfixturevalue(fixture_name)
    before_coeff = mf.mo_coeff.copy()
    before_energy = mf.mo_energy.copy()
    before_occ = mf.mo_occ.copy()
    original_df = mf.with_df
    hybrid = KMP2_HYBRID(mf)
    energy, amplitudes = hybrid.kernel(with_t2=True)
    direct, _, explicit_direct_t2 = _explicit_components(
        hybrid, hybrid.with_df_direct
    )
    full_direct, _, direct_t2 = _full_kernel_components(
        mf, hybrid.with_df_direct
    )
    exchange_mp = KMP2_STC(mf)
    exchange, exchange_t2 = exchange_mp.kernel(with_t2=True)

    assert hybrid.with_df is not hybrid.with_df_direct
    assert hybrid.with_df is not original_df
    assert hybrid.with_df_direct is not original_df
    assert hybrid.e_corr_direct == pytest.approx(direct, abs=1e-10, rel=0)
    assert hybrid.e_corr_direct == pytest.approx(full_direct, abs=1e-10, rel=0)
    assert hybrid.e_corr_exchange == pytest.approx(exchange, abs=1e-10, rel=0)
    assert energy == pytest.approx(direct + exchange, abs=1e-10, rel=0)
    assert set(amplitudes) == {"direct", "exchange"}
    np.testing.assert_allclose(amplitudes["direct"], direct_t2, atol=1e-10, rtol=0)
    np.testing.assert_allclose(amplitudes["direct"], explicit_direct_t2, atol=1e-10, rtol=0)
    np.testing.assert_allclose(amplitudes["exchange"], exchange_t2, atol=1e-10, rtol=0)
    assert hybrid.t2 is amplitudes
    np.testing.assert_array_equal(mf.mo_coeff, before_coeff)
    np.testing.assert_array_equal(mf.mo_energy, before_energy)
    np.testing.assert_array_equal(mf.mo_occ, before_occ)
    assert mf.with_df is original_df


def test_hybrid_frozen_padding_and_orbital_overrides(shifted_reference):
    """Both hybrid components honor the same padded frozen-MO context."""
    coeff_override = np.array(shifted_reference.mo_coeff, dtype=complex, copy=True)
    coeff_override[:, :, 2] *= np.exp(1j * np.array([0.0, 0.31, -0.23]))[:, None]
    energy_override = np.array(shifted_reference.mo_energy, copy=True)
    energy_override[:, 2] += np.array([0.02, 0.04, 0.03])
    hybrid = KMP2_HYBRID(shifted_reference, frozen=[[9], [8, 9], [9]])
    energy, amplitudes = hybrid.kernel(
        mo_energy=energy_override, mo_coeff=coeff_override, with_t2=True
    )
    direct, _, direct_t2 = _explicit_components(
        hybrid, hybrid.with_df_direct,
        mo_energy=energy_override, mo_coeff=coeff_override,
    )
    _, exchange, exchange_t2 = _explicit_components(
        hybrid, hybrid.with_df,
        mo_energy=energy_override, mo_coeff=coeff_override,
    )

    assert amplitudes["direct"].shape[3:5] == (2, 2)
    assert amplitudes["direct"].shape == amplitudes["exchange"].shape
    np.testing.assert_allclose(amplitudes["direct"], direct_t2, atol=1e-10, rtol=0)
    np.testing.assert_allclose(amplitudes["exchange"], exchange_t2, atol=1e-10, rtol=0)
    assert hybrid.e_corr_direct == pytest.approx(direct, abs=1e-10, rel=0)
    assert hybrid.e_corr_exchange == pytest.approx(exchange, abs=1e-10, rel=0)
    assert energy == pytest.approx(direct + exchange, abs=1e-10, rel=0)


def test_hybrid_direct_is_independent_of_stc_settings(shifted_reference):
    """Changing sTC settings changes exchange while bare direct is fixed."""
    configs = [
        KMP2_HYBRID(shifted_reference, eta=4.0),
        KMP2_HYBRID(shifted_reference, eta=3.0),
        KMP2_HYBRID(
            shifted_reference, eta=4.0, exxdiv="vcut_sph", rc_type="sph"
        ),
    ]
    for mp in configs:
        mp.kernel()
    direct = [mp.e_corr_direct for mp in configs]
    exchange = [mp.e_corr_exchange for mp in configs]
    assert np.all(np.isfinite(direct + exchange))
    np.testing.assert_allclose(direct, direct[0], atol=1e-10, rtol=0)
    assert abs(exchange[0] - exchange[1]) > 1e-8
    assert abs(exchange[0] - exchange[2]) > 1e-8


def test_repeated_kernels_clear_disabled_results(gamma_reference):
    """Results from the latest successful call replace stored components."""
    mp = KMP2_STC(gamma_reference)
    mp.kernel(with_t2=True, with_direct=True)
    exchange, _ = mp.kernel(with_t2=False, with_direct=False)
    assert mp.e_corr_direct is None
    assert mp.e_corr == pytest.approx(exchange, abs=1e-12)
    assert mp.t2 is None

    hybrid = KMP2_HYBRID(gamma_reference)
    hybrid.kernel(with_t2=True)
    hybrid.kernel(with_t2=False)
    assert hybrid.t2 is None


def test_reference_validation_rejects_unsupported_methods(gamma_reference):
    """The wrapper rejects non-KRHF, symmetry-reduced, and low-D references."""
    from pyscf.pbc import scf as pbc_scf
    with pytest.raises(TypeError, match="KRHF"):
        KMP2_STC(pbc_scf.KUHF(gamma_reference.cell, gamma_reference.kpts))

    symmetry_reduced = copy.copy(gamma_reference)
    symmetry_reduced.with_df = copy.copy(gamma_reference.with_df)
    symmetry_reduced.__dict__.pop("kpts", None)
    full_grid = gamma_reference.cell.make_kpts([4, 1, 1])
    symmetry_reduced.with_df.kpts = KPoints(gamma_reference.cell, full_grid).build(
        time_reversal_symmetry=True
    )
    with pytest.raises(NotImplementedError, match="symmetry-reduced"):
        KMP2_STC(symmetry_reduced)

    low_dim = copy.copy(gamma_reference)
    low_dim.cell = copy.copy(gamma_reference.cell)
    low_dim.cell.dimension = 2
    with pytest.raises(NotImplementedError, match="three-dimensional"):
        KMP2_STC(low_dim)
