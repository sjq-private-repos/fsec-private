# G0W0-AC with auxiliary-function and Madelung q → 0 corrections

`fsec.standard.KRGWAC_ewald` is PySCF's `pyscf.pbc.gw.krgw_ac.KRGWAC`, but with
a choice of how the integrable q → 0 Coulomb singularities of the exchange and
correlation self-energies are corrected. Everything else (DF integrals,
frequency grid, analytic continuation, QP solver) is PySCF's code. See
[Examples and checks](#examples-and-checks) for runnable comparisons.

```python
from fsec.standard import KRGWAC_ewald

gw = KRGWAC_ewald(kmf)      # kmf: converged KRKS/KRHF with a GDF/RSDF with_df
gw.fc_method = 'aux'        # 'aux' (default), 'madelung', or 'sphere'
gw.fc_sigma = None          # Gaussian width for 'aux'; None = FHI-gap choice
gw.fc_qdirs = None          # None = PySCF's q direction; or a Lebedev size, e.g. 26
gw.kernel(orbs=..., kptlist=...)
```

With `fc_method='sphere'` and `fc_qdirs=None`, the QP energies agree with
PySCF's `KRGWAC(fc=True)`. They differ only where PySCF is wrong: with
`fc_grid=True` (the accumulation bug described below), and for non-contiguous
or reordered `orbs` (see [Implementation notes](#implementation-notes)).

## Geometric constants

The singular q → 0 parts of the self-energy are corrected by two scalars:

- `c_head` multiplies `ε⁻¹₀₀(0, iν) − 1` in the correlation head correction
  (Zhu & Chan Eq. 43). It is also subtracted from the occupied diagonal of Σx.
- `c_wing` multiplies the wing term `Wn_P0` (Zhu & Chan Eq. 45).

In FHI-gap notation (Jiang et al. 2013, App. B, Eqs. B.5–B.6) these are
`c_head = 4π Cs2/Ω` and `c_wing = sqrt(4π/Ω) Cs1`. Here Ω is the unit-cell
volume, V = N_k Ω, and k runs over the Γ-centred supercell reciprocal lattice
(k = q + G).

| `fc_method` | `c_head` | `c_wing / sqrt(4πΩ)` |
| --- | --- | --- |
| `'sphere'` (PySCF) | `2 q0/π` | `q0²/(4π²)` |
| `'aux'` | `1/sqrt(πβ) − (1/V) Σ'_k 4π e^{−βk²}/k²` | `1/(4π²β) − (1/V) Σ'_k e^{−βk²}/k` |
| `'madelung'` (β → 0) | `tools.madelung(cell, kpts)` | Ewald-split 1/k constant (below) |

where `q0 = (6π²/V)^{1/3}`.

- **Default β.** `fc_sigma=None` uses FHI-gap's `β = (Ω/6π²)^{1/3}`, taken
  literally in atomic units. The paper derives it from `βR_BZ = 1`, which is
  dimensionally loose because β multiplies `|q+G|²`. A custom width `fc_sigma`
  (bohr⁻¹) sets `β = 1/(2σ²)`, so the Gaussian is `e^{−q²/2σ²}`.
- **q = 0 term.** As in Eq. B.6, the q = 0 term of the lattice sum keeps only
  G ≠ 0. The constant term of `e^{−βq²}/q²` at q → 0 is therefore dropped, and
  the 'aux' head is exactly `madelung − 4πβ/V`, up to an exponentially small
  real-space tail. So σ affects the head and exchange only at O(1/N_k), and
  σ → ∞ recovers the Madelung constant.
- **1/k Madelung constant.** It is evaluated by splitting
  `1/k = erfc(√τ k)/k + erf(√τ k)/k`:

  ```
  C1 = 1/(8π²τ) − (1/V) Σ'_k erfc(√τ k)/k − Σ'_R e^{−R²/4τ}/(2π²R²) + 2 sqrt(τ/π)/V
  ```

  R runs over the supercell lattice. C1 does not depend on τ, and the 'aux'
  value tends to it as β → 0.
- **Cost of large σ.** The 'aux' lattice sums are cut off at `βk² = 40`, so the
  number of terms grows as `σ³`. For very large σ, use `'madelung'`.

## Direction-averaged k·p dielectric head and wings

PySCF builds `ε⁻¹₀₀` and `ε⁻¹_P0` from k·p pair densities along a single
direction: b₁, or with `fc_grid=True` 26 positive-octant points. In k·p,
`qij(q)` is exactly linear in q. So the three Cartesian components give
`ε⁻¹₀₀(q̂) = 1/(q̂ᵀ M q̂)` for any direction, with

```
M_ab = δ_ab − 4π P_ab − w_a* (1 − Π)⁻¹ w_b,    P_ab = 4/N_k Σ eia qij_a* qij_b,
ε_P0(q̂) = Σ_a q̂_a w_a,    ε⁻¹_P0(q̂) = −ε⁻¹₀₀(q̂) (1 − Π)⁻¹ ε_P0(q̂).
```

`fc_qdirs = n` averages these over an n-point Lebedev grid from
`pyscf.dft.LebedevGrid`; 6 gives ±x, ±y, ±z. The extra cost is negligible.

- **Wings vanish.** `ε⁻¹_P0(q̂)` is odd in q̂. Its 1/q singularity integrates
  to zero over the sphere, and it sums to zero on an inversion-symmetric k-mesh.
  Any Lebedev average therefore gives a zero wing correction, up to rounding.
  PySCF's nonzero wing term comes from sampling only one direction.
- **Anisotropic head.** The averaged head `⟨ε⁻¹₀₀⟩` multiplies the isotropic
  `c_head`. That is exact for the Brillouin-zone integral and approximate for
  the anisotropic part of the lattice sum. An exact treatment would place
  `1/(k̂ᵀMk̂)` inside the auxiliary sums; this is not implemented.
- **PySCF `fc_grid` bug.** With `fc_grid=True`, PySCF accumulates `ε⁻¹_P0`
  using the running sum of `ε⁻¹₀₀` rather than the per-point value. Here the
  same 26 directions are averaged with per-point values.
- **Pseudopotential term.** The k·p matrix elements use only ∇. They omit the
  nonlocal-pseudopotential commutator, as in PySCF.

## Supported scope

With `fc=True`, `kernel` checks that the system is in scope and raises
otherwise. It accepts:

- 3D cells with the full Coulomb interaction (`cell.omega == 0`);
- GDF or RSDF integrals (not MDF);
- restricted scalar orbitals with integer occupations;
- a gap, with the same occupied/virtual active space at every k-point;
- a complete uniform Monkhorst–Pack mesh, which may be shifted or reordered.

An incomplete or duplicated mesh raises `ValueError`; the other cases raise
`NotImplementedError`. As in PySCF, metallic (smearing) references switch the
correction off. `kptlist` selects output k-points only: the constants always
use the full mean-field mesh.

## Implementation notes

- **Raw self-energy.** `kernel` calls PySCF's `get_sigma`/`get_sigma_outcore`
  with `fc` off. `get_sigma_fc` then adds the head and wing corrections, which
  need only the q = 0 DF block. The finite-size correction therefore also works
  with `outcore=True`.
- **Private DF cache.** Pure-DFT SCF often builds only the (k,k) Coulomb
  blocks, but GW also needs the (k,k') blocks. If the DF object is unbuilt or
  J-only, `kernel` builds a full copy in a private temporary file and leaves the
  SCF's DF object unchanged.
- **Orbital selection.** PySCF's `get_sigma` mishandles non-contiguous or
  reordered `orbs`. `kernel` therefore evaluates the containing contiguous range
  and selects the requested order afterwards. Frozen-orbital labels and output
  k-points otherwise follow upstream conventions.
- **`fullsigma`.** With `fullsigma=True`, only the diagonal receives the
  correlation correction. The stored `sigmaI` (when `rdm=True`) and the
  `writefile` output contain the corrected values.

## Examples and checks

- [krgwac_ewald_comparison.py](../examples/krgwac_ewald_comparison.py)
  compares `'aux'` (with and without a 26-point direction average),
  `'madelung'`, `'sphere'` and the uncorrected QP energies for a small H₂
  cell on one mesh.
- [krgwac_ewald.py](../examples/krgwac_ewald.py) prints the diamond Γ-point
  gap on 2³ and 3³ meshes for every `fc_method` × `fc_qdirs ∈ {None, 26}`,
  together with `c_head` and `c_wing`.

The tests are in `fsec/standard/tests/test_krgwac_ewald.py`. The kernel-level
checks of the full self-energy are marked slow:

```bash
python -m pytest fsec/standard/tests/test_krgwac_ewald.py            # fast checks
python -m pytest -m slow fsec/standard/tests/test_krgwac_ewald.py    # full self-energy checks
```

## References

- T. Zhu and G. K.-L. Chan, J. Chem. Theory Comput. 17, 727 (2021).
- H. Jiang et al., Comput. Phys. Commun. 184, 348 (2013), Appendix B.
- S. Massidda, M. Posternak, and A. Baldereschi, Phys. Rev. B 48, 5058 (1993).
- C.-N. Yeh, J. Shee, Q. Sun, E. Gull, and D. Zgid, Phys. Rev. B 106, 085121 (2022).
- C. Freysoldt et al., Comput. Phys. Commun. 176, 1 (2007).
