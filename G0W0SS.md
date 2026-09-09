# Gaussian singularity subtraction for G0W0-AC

## Usage and defaults

For a three-dimensional, spin-restricted insulating calculation on a complete
uniform k-point mesh, `G0W0SS` applies a geometry-derived Gaussian correction
to the imaginary-axis self-energy before analytic continuation.  It uses
PySCF's GDF integrals and has the same orbital, k-point, continuation, and
quasiparticle controls as `KRGWAC`.  The API is tested with PySCF 2.14.

```python
from fsec.singularity_subtraction import G0W0SS

gw = G0W0SS(kmf)
gw.nw = 80
gw.kernel(orbs=[0, 1], kptlist=[0])
print("G0W0 quasiparticle energies:", gw.mo_energy[0, [0, 1]])
```

The default `fc_grid=False` follows the PySCF small-q direction; set it to
`True` to invert and average all small-q directions separately.  Set
`gw.fc = False` to retain PySCF's uncorrected self-energy path.  The Gaussian
width is `sigma=(6*pi**2/(V*Nk))**(1/3)`, and the separate analytic and
reciprocal-mesh head and wing coefficients are available as
`gw.gaussian_coefficients`.  Its quadrature fields include the subtraction
sign, so each total is `integral + quadrature`.  The Gaussian sum includes
nonzero reciprocal-supercell vectors through `8*sigma`; there is no auxiliary
function fitting.  Standard GW analytic-continuation fitting is retained.
Frequency-dependent dielectric limits and the
separate head and wing self-energy additions are available as
`gw.fc_eps_inv_00`, `gw.fc_eps_inv_p0`, `gw.fc_sigma_head`, and
`gw.fc_sigma_wing`.  The PySCF exchange finite-size term (Eq. 46) is retained.
Corrected calculations reject metallic or smeared occupations and out-of-core
self-energy evaluation.  A complete runnable setup is in
[`examples/h2_g0w0ss.py`](examples/h2_g0w0ss.py).

## Implementation plan

Replace the correlation head and wing corrections in Eqs. 44–45 of the
Zhu–Chan G0W0 paper with Gaussian integral-minus-quadrature corrections,
applied before analytic continuation. Retain the exchange correction in
Eq. 46.

### Gaussian coefficients

Let $V=\Omega_{\mathrm{cell}}$ and $N_k$ be the number of k-points. Use the
EXX Gaussian convention and the geometry-derived volume rule:

$$
h(Q)=e^{-|Q|^2/(2\sigma^2)},\qquad
\sigma=\left(\frac{6\pi^2}{VN_k}\right)^{1/3}.
$$

For the periodized $Q=q+G$ mesh, define

$$
C_{\mathrm{head}}
=\sqrt{\frac{2}{\pi}}\sigma
-\frac{4\pi}{VN_k}\sum_{Q\ne0}\frac{h(Q)}{|Q|^2},
$$

$$
C_{\mathrm{wing}}
=\sqrt{4\pi V}\left[
\frac{\sigma^2}{2\pi^2}
-\frac{1}{VN_k}\sum_{Q\ne0}\frac{h(Q)}{|Q|}
\right].
$$

These coefficients include the all-space analytic integrals and subtraction
of sampled auxiliary values. Replace the screened-interaction corrections
with

$$
\Delta_{00}(i\omega)
=C_{\mathrm{head}}\,[\epsilon^{-1}_{00}(0,i\omega)-1],
$$

$$
\Delta_{\mathrm{wing},nk}(i\omega)
=C_{\mathrm{wing}}\,2\operatorname{Re}
\sum_P L_{Pnn}(k,k)\epsilon^{-1}_{P0}(0,i\omega).
$$

Feed these terms into the existing imaginary-frequency contraction,
retaining its sign, Green's-function factor, frequency weights, and
$1/\pi$ normalization. All quantities use atomic units.

### PySCF integration and lattice sums

- Subclass the installed PySCF 2.14.0 `KRGWAC` with a local kernel and
  in-core self-energy specialization. Reuse PySCF response helpers, orbital
  transformations, continuation utilities, and quasiparticle-solving
  machinery; preserve attribution for adapted code.
- Obtain inverse dielectric head and wing elements through the full
  dielectric matrix's Schur complement, including coupling to the body.
  Keep `fc_grid=False` and its existing small-q direction by default. With
  `fc_grid=True`, invert each direction independently before averaging both
  inverse elements, correcting the upstream cumulative-head averaging error.
- Infer the complete Monkhorst–Pack mesh shape independently of a common
  shift or ordering. Construct its unshifted reciprocal-supercell lattice
  and include every nonzero vector within `8*sigma`. Derive enumeration
  bounds from the inverse reciprocal basis for skewed cells and evaluate
  sums in bounded chunks.
- Apply corrections only to diagonal self-energy entries, including with
  `fullsigma=True`. Preserve orbital and k-point subsets, frozen-orbital
  mappings, and continuation settings. Explicit orbital indexing also
  preserves reordered selections; two-pole fitting uses the orbital axis
  rather than the k-point axis.
- Export `G0W0SS` lazily and provide usage documentation and a runnable
  example. Retain diagnostics for the width, separate integral/quadrature
  coefficients, frequency-dependent inverse dielectric elements, and
  separate head/wing self-energy additions.

### Support and diagnostics

Initial corrected support is three-dimensional, spin-restricted, gapped
systems with a complete uniform k mesh and pure Gaussian density fitting.
Reject unsupported corrected calculations explicitly, including metallic or
smeared occupations, mixed/FFT density fitting, and out-of-core self-energy
evaluation. Frozen masks must be the same at every k-point.

`fc=True` selects Gaussian correlation corrections plus Eq. 46;
`fc=False` delegates to PySCF's uncorrected kernel. The Gaussian width is
always geometry-derived; only standard analytic-continuation fitting remains.

| Diagnostic | Contents |
| --- | --- |
| `gaussian_coefficients` | Width, separate signed integral/quadrature terms, and total head/wing coefficients |
| `fc_eps_inv_00` | Inverse dielectric head, shape `(nw,)`, on `gw.freqs` |
| `fc_eps_inv_p0` | Inverse dielectric wing, shape `(nw, naux)`, on `gw.freqs` |
| `fc_sigma_head`, `fc_sigma_wing` | Separate diagonal additions, shape `(len(kptlist), len(orbs), nw_sigma)`, even with `fullsigma=True` |

### Validation plan

- Verify both analytic integrals by independent radial integration,
  including volume, Coulomb, and two-wing factors.
- Check cubic and skewed lattice sums, shifted and reordered meshes,
  Gamma-only sampling, complete vector enumeration, and convergence from
  `8*sigma` to `10*sigma`.
- Compare Schur-complement elements with direct dielectric inversion and
  verify that direction averaging is independent of iteration order.
- On a small periodic insulator, substitute the original spherical
  coefficients and reproduce PySCF's corrected self-energy. Then compare
  the Gaussian result with the uncorrected body plus independently
  reconstructed head and wing additions.
- Check frozen orbitals, selected states, diagonal/full-matrix consistency,
  unchanged exchange correction, continuation, finite quasiparticle
  energies, and rejection of unsupported corrected systems. Run relevant
  existing regressions in `fsec-312`.

## Implementation summary and verification

Implemented in
[`fsec/singularity_subtraction/g0w0ss.py`](fsec/singularity_subtraction/g0w0ss.py),
with a lazy package export, the usage above, and the runnable
[`H2 example`](examples/h2_g0w0ss.py). Numerical regression coverage is in
[`test_g0w0ss.py`](fsec/singularity_subtraction/tests/test_g0w0ss.py).

Verification completed on September 9, 2026, using `fsec-312` and PySCF 2.14.0:

- All **13 new tests passed**, together with **27 existing tests and
  18 subtests**. The runnable example completed with finite quasiparticle
  energies.
- Independent comparisons on a shifted, reordered k mesh in a skewed cell
  reproduced the uncorrected body exactly and the original spherical
  correction within `4.4e-19` hartree. Direct full-dielectric inversion
  reconstructed the Gaussian additions within `5e-15` hartree, with nonzero
  head and wing contributions.
- Integer and list frozen-core selections preserved mappings and
  diagonal/full-matrix consistency, retained Eq. 46, and produced finite
  quasiparticle energies. Extending the Gaussian cutoff from `8*sigma` to
  `10*sigma` changed the checked coefficients by less than `2e-14`.
- Two existing fitting-reference failures were reproduced in a clean
  `HEAD` checkout, independently of these changes, in
  [`test_function_fitting.py`](fsec/singularity_subtraction/tests/test_function_fitting.py):
  `test_exx_scipy_least_squares_quartic_exponential` and
  `test_mp2_scipy_minimize_exchange_xngauss_stacked_singularity_exchange`.

Run the focused suite with the `fsec-312` Python environment:

```bash
python -m pytest -q fsec/singularity_subtraction/tests/test_g0w0ss.py
```

The validation environment required its `lib` directory on
`LD_LIBRARY_PATH`; OpenMP, MKL, and OpenBLAS thread counts were set to one.

## Reference

- T. Zhu and G. K.-L. Chan, “All-Electron Gaussian-Based G0W0 for Valence and
  Core Excitation Energies of Periodic Systems,” *Journal of Chemical Theory
  and Computation* **17**, 727–741 (2021).
  [doi:10.1021/acs.jctc.0c00704](https://doi.org/10.1021/acs.jctc.0c00704)
