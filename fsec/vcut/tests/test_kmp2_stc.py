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

from fsec.vcut import KMP2_STC


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


def _explicit_exchange(mp, mo_energy=None, mo_coeff=None):
    """Build V with the public AO2MO path and contract the exchange term."""
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
                    mp.with_df,
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
                exchange -= np.einsum(
                    "ijab,ijba", t2_ref[ki, kj, ka], eri_ij[kb]
                ).real
    return exchange / nkpts, t2_ref


def _full_kernel_exchange(mf, stc_df, frozen=None):
    """Use installed kMP2 energy decomposition with the same stored factors."""
    ref = pyscf_kmp2.KMP2(mf, frozen=frozen)
    ref._scf = copy.copy(ref._scf)
    ref._scf.with_df = stc_df
    ref.kernel(with_t2=True)
    return ref.e_corr_ss - ref.e_corr_os, ref.t2


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
