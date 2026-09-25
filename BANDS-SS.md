# Orbital-resolved singularity subtraction for occupied bands

Implement `BandsSS` for 3D, closed-shell HF and global hybrids, with one Gaussian fit per occupied orbital. Support both generated and supplied band data, apply scalar corrections at fixed orbitals, and use a common near-zero cutoff in the bare exchange and auxiliary quadrature.

## 1. Derivation

Let $V$ be the primitive-cell volume, $\mathcal K$ the occupied sampling mesh, and $N_k=|\mathcal K|$. Fix the target orbital $n\mathbf k$, where $\mathbf k$ is an arbitrary band k-point. Write

$$
\mathbf q=\mathbf k-\mathbf k_i,\qquad
\mathbf p=\mathbf q+\mathbf G,
$$

and define the pair-density Fourier coefficient using cell-normalized periodic orbitals:

$$
\rho_{j\mathbf k_i,n\mathbf k}(\mathbf G)
=
\int_\Omega
u_{j\mathbf k_i}^{*}(\mathbf r)
u_{n\mathbf k}(\mathbf r)
e^{-i\mathbf G\cdot\mathbf r}\,d\mathbf r.
$$

The orbital-resolved structure factor is

$$
\boxed{
S_{n\mathbf k}(\mathbf q+\mathbf G)
=
\sum_{j\in\mathrm{occ}}
\left|\rho_{j,\mathbf k-\mathbf q;\,n\mathbf k}(\mathbf G)\right|^2.
}
$$

This is the orbital-resolved version of Eq. 7: remove the outer orbital sum and k-point average, retaining the sum over exchange partners. The transfer convention here follows your $\mathbf q=\mathbf k_{\rm band}-\mathbf k_i$. [Quiton et al., Eq. 7](https://doi.org/10.1021/acs.jctc.5c01066)

For a consistent occupied manifold, orthonormality gives

$$
S_{n\mathbf k}(0)=
\begin{cases}
1,&n\text{ occupied},\\
0,&n\text{ virtual}.
\end{cases}
$$

Thus each occupied orbital has its own structure factor but the same leading singular coefficient. This connects directly to the occupied-orbital correction in Remark 5.3. [Xing et al.](https://arxiv.org/abs/2108.00206)

Define the bare quadrature

$$
Q_{\mathbf k}[f]
=
\frac{1}{VN_k}
\sum_{\mathbf k_i\in\mathcal K,\mathbf G}^{\prime}
\frac{4\pi}{|\mathbf k-\mathbf k_i+\mathbf G|^2}
f(\mathbf k-\mathbf k_i+\mathbf G),
$$

where the prime initially excludes only zero. The exchange contribution to a restricted HF orbital energy is

$$
\epsilon^{x,0}_{n\mathbf k}=-Q_{\mathbf k}[S_{n\mathbf k}].
$$

There is **no additional factor of $1/2$**: the restricted density's occupation factor of two cancels the $1/2$ multiplying PySCF's exchange matrix.

Choose an auxiliary function $h_{n\mathbf k}(0)=1$, fitted independently to $S_{n\mathbf k}$, and define

$$
I[h]=\frac{1}{(2\pi)^3}
\int_{\mathbb R^3}\frac{4\pi}{|\mathbf p|^2}h(\mathbf p)\,d\mathbf p.
$$

Singularity subtraction gives

$$
\epsilon^{x,\mathrm{SS}}_{n\mathbf k}
=
-Q_{\mathbf k}[S_{n\mathbf k}-h_{n\mathbf k}]
-I[h_{n\mathbf k}],
$$

hence

$$
\boxed{
\xi_{n\mathbf k}
=
Q_{\mathbf k}[h_{n\mathbf k}]-I[h_{n\mathbf k}],
\qquad
\epsilon^{\mathrm{SS}}_{n\mathbf k}
=
\epsilon^{0}_{n\mathbf k}+\alpha_x\xi_{n\mathbf k}.
}
$$

Here $\alpha_x=1$ for HF and is the exact-exchange fraction for a global hybrid.

For v1,

$$
h_{n\mathbf k}(\mathbf p)
=
e^{-|\mathbf p|^2/(2\sigma_{n\mathbf k}^2)},
\qquad
I[h_{n\mathbf k}]
=
\sqrt{\frac{2}{\pi}}\sigma_{n\mathbf k}.
$$

Therefore,

$$
\boxed{
\xi_{n\mathbf k}
=
\frac{4\pi}{VN_k}
\sum_{\mathbf k_i,\mathbf G}^{\prime}
\frac{
e^{-|\mathbf k-\mathbf k_i+\mathbf G|^2/(2\sigma_{n\mathbf k}^2)}
}{
|\mathbf k-\mathbf k_i+\mathbf G|^2
}
-
\sqrt{\frac{2}{\pi}}\sigma_{n\mathbf k}.
}
$$

The quadrature uses the **actual shifted grid** $\mathbf k-\mathcal K+\mathbf G$, whether or not zero is sampled. Its correction can become large and positive near an occupied sampling point, canceling the negative bare-exchange pole.

In repository conventions, this corresponds to an orbital-specific `-chi`. On the original mesh, using a common auxiliary function recovers `ExxSS.correction / nocc`. Independently optimized orbital fits need not reproduce the single global fit.

## 2. Common near-zero cutoff

Introduce `q_zero_tol`, in inverse bohr, with an initial default of $10^{-4}$. Define $Q_{\mathbf k}^{\tau}$ by retaining only transfers satisfying $|\mathbf q+\mathbf G|>\tau$.

Use that identical mask in:

- The exchange operator used to generate the baseline bands.
- The auxiliary-function quadrature.
- Coulomb-weighted fitting samples.

The corrected expression becomes

$$
\epsilon^{\mathrm{SS}}_{n\mathbf k}
=
\epsilon^{0,\tau}_{n\mathbf k}
+
\alpha_x\left(
Q_{\mathbf k}^{\tau}[h_{n\mathbf k}]
-I[h_{n\mathbf k}]
\right).
$$

Keep the auxiliary integral over all space. Enforce $h(0)=1$ analytically; no sampled zero is required.

Apply the exclusion **before constructing and diagonalizing bare exchange**. Removing only the auxiliary term would leave any pole already present in the baseline energy.

This implements the requested ordinary-SS fallback. Gaussian curvature mismatch can still leave a finite directional residual or a finite change across the cutoff boundary. The cutoff is an adjustable numerical approximation, and exact continuity is outside v1.

## 3. Implementation and interface

Add `BandsSS` in `fsec/singularity_subtraction/bands_ss.py`, reusing the existing Gaussian model and fitting machinery without inheriting the total-energy normalization or unshifted quadrature assumptions.

The public interface will be:

```python
ss = BandsSS(
    kmf,
    kpts_band,
    mo_energy_band=None,
    mo_coeff_band=None,
    q_zero_tol=1e-4,
    # Existing applicable Gaussian fitting options.
)

correction = ss.compute_correction()
energies_ss, coefficients = ss.get_bands()
```

`kpts_band` may be a single `(3,)` point or an `(nband_k, 3)` array. The band
energy and coefficient results retain the leading `nband_k` dimension.
`compute_correction()` returns a full `(nband_k, nmo)` array in hartree;
`xi`, `sigma`, `integral_term`, and `quadrature_term` each have shape
`(nband_k, nocc)`. `xi` is unscaled, `sigma` is in inverse bohr, and the
quadrature and integral terms are in hartree. Virtual corrections are zero.

The supported fitting options are `N_local`, `sq_ke_cutoff`,
`qG_norm_cutoff_sigma`, `qG_norm_cutoff`, `min_points`, `fit_with_coul`, and
`initial_guess`, following the existing one-Gaussian fitting conventions. A
fixed normalized Gaussian may be supplied as `sigma=value` or
`model_params=[1, value]`. Fourier exchange uses `cell.mesh` by default;
`exchange_mesh` allows independent convergence of this integration.

- Accept both band arrays together, or generate both. Supplied energies must use bare exchange with the same exclusion convention; document this requirement explicitly.
- Store unscaled occupied `xi`, fitted `sigma`, integral and quadrature terms, fitting diagnostics, and baseline/corrected energies.
- Return `correction` with the full band-energy shape, scaled by $\alpha_x$, with zero SS shifts for virtual bands. Preserve orbital ordering and coefficients.

**Structure factors and fitting**

- Add an orbital-resolved structure-factor helper using PySCF's analytic Fourier AO-pair integrals. Contract source occupied orbitals with all target occupied orbitals together.
- Preserve source-k-point and reciprocal-vector identities throughout; do not assume the band path is a Monkhorst–Pack mesh.
- Do not reuse inversion symmetry for individual orbital structure factors.
- Fit one positive Gaussian width per occupied target orbital. Use the existing nearest-sample pilot fit and sigma-based fitting radius, retaining at least six nonexcluded samples.
- Evaluate the auxiliary quadrature independently of the fitting radius, including Gaussian samples through $8\sigma$.

**Baseline band generation**

- Evaluate exchange through blocked AO-pair Fourier contractions with $4\pi/|\mathbf p|^2$, applying the common exclusion mask before contraction.
- On an isolated mean-field copy, route exchange through this evaluator while retaining the original Hartree, one-electron, and semilocal DFT contributions.
- Reuse PySCF's band diagonalization. Keep the source density fixed and preserve the caller's settings.
- Cover the local PySCF tiny-k real/complex casting issue through complex-safe batched nonexchange evaluation.

Document the derivation and provide an example containing on-mesh, off-mesh, and near-mesh band points.

## 4. Validation and boundaries

Use a small, focused numerical suite:

1. Verify occupied $S(0)=1$, normalization, and the exchange spin factor.
2. Recover the existing total structure factor by averaging orbital-resolved factors on the source mesh.
3. Recover the `ExxSS` correction with a shared Gaussian and identical numerical grids.
4. For an analytic Gaussian structure factor, recover the same corrected energy on shifted grids, including grids without zero.
5. Verify identical exclusions in baseline exchange and auxiliary quadrature below, at, and above `q_zero_tol`.
6. Compare the new Fourier exchange matrix with PySCF AFT exchange away from excluded transfers.
7. Verify independent orbital fits, hybrid scaling, zero virtual SS shifts, and agreement between generated and supplied inputs.
8. Check reciprocal-grid and cutoff sensitivity along a short band path, including single-point tiny-q calls.

Read-only feasibility checks already confirmed the Fourier exchange contraction against PySCF AFT and produced bounded baseline energies below the proposed cutoff.

V1 assumes a full, uniformly weighted occupied mesh and a 3D closed-shell insulator. Unrestricted, fractional-occupation, and range-separated methods are excluded. Corrections are diagonal, fixed-orbital estimates; resolving arbitrary rotations within degenerate target subspaces requires a later matrix-valued extension.
