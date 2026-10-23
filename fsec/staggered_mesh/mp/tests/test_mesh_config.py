"""Zero-momentum-transfer audit of the staggered MP3 mesh assignments."""

import pytest

from fsec.staggered_mesh.mp.kmp3_2mesh import KMP3_2mesh_mixed
from fsec.staggered_mesh.mp.kmp3_3mesh import mesh_blocks_mp3, mesh_config_mp3


def zero_transfer_pairs(mesh_eri):
    """Return ``(block, meshes, pair_kind)`` for each unstaggered pair.

    A block ``<PQ|RS>`` samples zero momentum transfer when ``P, R`` (and
    hence ``Q, S``) share a mesh.  Occupied-occupied and virtual-virtual pair
    densities make that point a ``1/|q|^2`` singularity; occupied-virtual
    pairs vanish there and leave only a bounded discontinuity.
    """
    found = []
    for block, tuples in mesh_eri.items():
        for meshes in tuples:
            for p, r in ((0, 2), (1, 3)):
                if meshes[p] == meshes[r]:
                    kind = 'same' if block[p] == block[r] else 'occ-vir'
                    found.append((block, meshes, kind))
    return found


@pytest.mark.parametrize('mesh_mp3_amp', [(0, 0, 1, 2), (1, 2, 0, 0)])
def test_three_mesh_schemes_avoid_all_zero_transfer(mesh_mp3_amp):
    """Both three-mesh energy forms keep every ERI and MP2 amplitude staggered."""
    mesh_term = mesh_config_mp3(mesh_mp3_amp)
    assert zero_transfer_pairs(mesh_blocks_mp3(mesh_mp3_amp, mesh_term, 3)) == []


def test_mixed_two_mesh_staggers_every_singular_vertex():
    """Two-mesh 2b leaves zero transfer only in occupied-virtual MP2 amplitudes.

    The external ``oovv`` blocks used by the energy must stay staggered.
    """
    mesh_mp3_amp = KMP3_2mesh_mixed._mesh_mp3_amp_mixed
    mesh_eri = mesh_blocks_mp3(
        mesh_mp3_amp, KMP3_2mesh_mixed._mesh_term_mixed, 2
    )
    pairs = zero_transfer_pairs(mesh_eri)
    assert pairs
    assert {(block, kind) for block, _, kind in pairs} == {('oovv', 'occ-vir')}
    energy_blocks = {meshes[:3] for block, meshes, _ in pairs}
    assert tuple(mesh_mp3_amp[:3]) not in energy_blocks
