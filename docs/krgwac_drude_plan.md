# Drude long-wavelength limit for metallic KRGWAC finite-size corrections

## Context

Metals get no finite-size correction today. `KRGWAC_ewald.kernel` sets `fc = False` for smearing
references (`fsec/standard/krgwac_ewald.py:622`), as PySCF does, and `_validate_fc_setup` rejects
fractional occupations. The goal is to make `fc=True` valid for metals by adding the Drude
(intraband) term to the $\mathbf q \to 0$ head.

**Zhu & Chan give no formula for this.** Sec. 3.6 only says that intraband transitions leave a
non-vanishing Drude term that could be a finite-size correction, and leaves it to future work. Their
Cu results are uncorrected. They cite Liu, Kaltak, Klimeš, Kresse, PRB 94, 165109 (2016), and Eqs.
79-80 there supply the formula below. Combining it with the Zhu & Chan head/wing scheme, and handling
the Fermi level, is new work here rather than a published recipe.

**Target.** `KRGWAC_ewald`. With `fc_method='sphere'` it is PySCF's regular KRGWAC, so every
`fc_method` and `fc_qdirs` gets the extension. The staggered route is out of scope.

## Theory

Let $f \in [0,1]$, $p^a_{nm} = -i\langle n|\partial_a|m\rangle$, $\epsilon_{nm} = \epsilon_n - \epsilon_m$,
and $D_{nm} = (f_n - f_m)/\epsilon_{nm}$, with $D_{nm} \to \partial f/\partial\epsilon$ for
$\lvert\epsilon_{nm}\rvert \lt 10^{-6}$. The degenerate pairs
$\mathcal D = \lbrace \lvert\epsilon_{nm}\rvert \lt 10^{-6} \rbrace$ (including $n=m$) form the Drude
weight, which is invariant within a degenerate subspace:

$$
\Omega^2_{p,ab} = \frac{8\pi}{N_k\Omega}\sum_{\mathbf k}\sum_{nm\in\mathcal D}(-D_{nm})\,p^{a*}_{nm}p^{b}_{nm},
\qquad
\epsilon^{\rm intra}_{ab}(i\omega) = \frac{\Omega^2_{p,ab}}{\omega^2}.
$$

The interband blocks generalize Eqs. 47-54 to all ordered pairs, with no energy denominators:

$$
S_{ab}(i\omega) = \frac{2}{N_k\Omega}\sum_{\mathbf k}\sum_{nm\notin\mathcal D}
\frac{D_{nm}\,p^{a*}_{nm}p^{b}_{nm}}{\omega^2+\epsilon_{nm}^2},
\qquad
w^a_P(i\omega) = \sqrt{4\pi}\,\frac{2}{N_k\sqrt\Omega}\sum_{\mathbf k}\sum_{n\ne m}
\frac{(f_n-f_m)\,L^P_{nm}\,p^{a*}_{nm}}{\omega^2+\epsilon_{nm}^2},
$$

$$
M_{ab}(i\omega) = \delta_{ab} - 4\pi S_{ab} + \frac{\Omega^2_{p,ab}}{\omega^2} - w^{a*}(1-\Pi)^{-1}w^{b},
\qquad
\epsilon^{-1}_{00}(\hat{\mathbf q}) = \frac{1}{\hat{\mathbf q}^T M \hat{\mathbf q}}.
$$

$\Pi$ is `get_rho_response_metal` at $\mathbf q = 0$. Intraband wings and body vanish at this order
(Liu et al.), so only the head changes.

**Self-energy.** With $H_n(\omega) = c_{\rm head}[\epsilon^{-1}_{00}(i\omega)-1] + c_{\rm wing}W_{n,P0}(i\omega)$
and $y_{nl} = \epsilon_F + i\omega_l - \epsilon_n$, the metal correction for $l \ge 1$ is

$$
\Sigma^{\rm fc}_n(i\omega_l) = -\frac{1}{\pi}\left[\frac{\pi}{2}(2f_n-1)H_n(\omega_l)
+ \sum_{j\ne l} w_j\,[H_n(\omega_j)-H_n(\omega_l)]\frac{y_{nl}}{y_{nl}^2+\omega_j^2}\right],
\qquad
\Sigma^{x,\rm fc}_{nn} = -c_{\rm head}\,f_n.
$$

Two separate problems drive this form:

1. **Near-pole quadrature.** $\omega_l$ is itself a quadrature node, so near $\epsilon_F$ the plain sum
   fails. With a Drude model at `nw=400`, the error in the integral (exact magnitude about 1.57) is
   1.4 at $\Delta = 10^{-3}$ and 19 at $\Delta = 10^{-4}$. Node subtraction gives at most $4\times10^{-3}$.
2. **Branch inside the smearing window.** PySCF's $\Sigma_c$ is a $T=0$ integral with a sharp branch,
   while its $\Sigma_x$ uses $f$. Keeping both as they are leaves a spurious on-shell jump of order
   $c$ at $\epsilon_F$. The $(2f_n-1)$ weight is the finite-temperature pole weighting, reduces to
   $\mathrm{sgn}(\epsilon_F-\epsilon_n)$ away from the window, and stays continuous even when the mesh
   Drude weight is poor.

**Expected physics.** On shell ($y=0$) with a Drude-dominated head, every band gets a rigid $-c_{\rm head}/2$,
plus a $Z$ renormalization with
$\partial_z\Sigma^{\rm head} = -c_{\rm head}/(2\sqrt{\epsilon_\infty}\Omega_p)$. This is the
$\epsilon_\infty \to \infty$ limit of the insulator result, where the $c\epsilon^{-1}$ gap correction
vanishes. Interband-only screening would instead open a spurious gap at $\epsilon_F$, which is why
the correction was switched off for metals. The Drude head remains valid at low frequency, where
$\mathbf q \to 0$ and $\omega \to 0$ do not commute, if $q_0 \ll k_{TF}$, because both limits then
give $\epsilon^{-1}_{00} \approx 0$.

## Source of the plasma frequency

Measured at $\sigma = 0.005$ / $0.02$ Ha, in eV, with FFTDF PBE. The momentum rows are non-SCF bands
from a fixed density ($6^3$, or $4^3$ for Cu); the interpolated row uses the SCF Fock on the $6^3$ mesh.

| Route | Na (dzvp) | Al (dzvp) | Cu (dzvp-sr) |
| --- | --- | --- | --- |
| On-mesh momentum, $4^3$ | 10.4 / 5.7 | 20.8 / 12.7 | 6.1 / 6.9 |
| On-mesh momentum, $6^3$ | 8.2 / 5.7 | 8.1 / 10.4 | 6.1 / 5.6 |
| Dense $16^3$ momentum ($8^3$ for Cu) | 6.05 / 5.9 | 12.3 / 13.3 | 6.2 / 5.1 |
| Fock interpolated $6^3 \to 32^3$ (true $\partial\epsilon/\partial k$) | 6.1 / 6.0 | 35 / 44 (failed) | 8.8 / 8.9 |

Conclusions:

- The Drude weight on GW-sized meshes is noisy, and exactly zero on Γ-centred $2^3$, where every
  k-point is time-reversal invariant.
- Momentum-only is close to the finite-difference band derivative for Na (0.6%) and Al dzvp (3.5%).
  For Cu it is about 35% low against the interpolated value. The likely cause is the omitted nonlocal
  pseudopotential term, which the existing k·p also omits.
- Fock interpolation is rejected: it is not robust (Al fails).
- The correction tolerates this. The rigid $-c/2$ does not depend on $\Omega_p$ as long as it is
  nonzero; only the $Z$ term scales as $1/\Omega_p$.

**Decision:** default to the on-mesh momentum value, add a `gw.fc_plasma` override for a better value,
and warn when the mesh value cannot be trusted.

## Code changes

All in `fsec/standard/krgwac_ewald.py`. The insulator path stays byte-identical, so the PySCF-match
tests keep their meaning.

- **`_is_metallic(gw)`**: `hasattr(gw._scf, 'sigma')`, as PySCF does.
- **`_validate_fc_setup`** (line 106): metal branch skips the integer, gap, and common-`nocc` checks
  (lines 112-128). It requires truthy `sigma` and `smearing_method`, `mu0 is None` (since `get_ef`
  ignores it), occupations in $[0,2]$, and `gw.ac == 'pade'` (`twopole` weights `omega[0]`, which is on
  the wrong branch). Warn when $N_k N_e$ is odd, because upstream fills $N+1$ electrons. The existing
  error messages stay.
- **`_get_occ_deriv(gw, mo_energy)`**: $\partial f/\partial\epsilon$ at `gw.ef` for `fermi` and `gauss`.
- **`get_pij(gw, mo_coeff)`**: `(3, nkpts, nmo, nmo)` Hermitized $p$ from
  `cell.pbc_intor('int1e_ipovlp', hermi=0, kpts=kpts, comp=3)`, using
  $\langle\mu|\nabla|\nu\rangle = -(\nabla\mu|\nu)$. Verified against `get_qij` for the occ-vir block
  to about $10^{-5}$.
- **`get_plasma_tensor(gw, ...)`**: returns $\Omega_p^2$, or `gw.fc_plasma` when set. Logs eigenvalues
  in eV, $k_{TF}^2 = \frac{8\pi}{N_k\Omega}\sum(-\partial f/\partial\epsilon)$, and $q_0$. Warns if the
  smallest eigenvalue is about 0 (suggest `fc_plasma`) or if $q_0 \ge k_{TF}$.
- **`get_eps_inv_q0_metal(...)`**: new sibling of `get_eps_inv_q0`, whose signature the tests use.
- **`get_sigma_fc`**: dispatch to a new `_get_sigma_fc_metal`, adding only keyword arguments (a test
  calls it with four positionals). Pass 1 builds `H` on all `nw` frequencies. Pass 2 applies the
  formula above. Keep plain `g0` at $l=0$, drop node $j=l$ (avoids 0/0 at $\Delta=0$), and assert
  `omega[1:].imag == freqs[:nw_sigma-1]`. This covers `ac_iw_cutoff` and `fullsigma`. Bypass `gw.nocc`,
  which is 0 for Na.
- **`kernel`** (lines 454-458): metal branch subtracts $c_{\rm head}f_n$ on the whole diagonal, with
  $f$ from `_mo_occ_frozen(gw, gw.mo_occ) / 2`. `get_rho_response_metal` halves `mo_occ` itself, so
  pass it the raw values.
- **`KRGWAC_ewald`**: drop `self.fc = False` (line 622) and keep the `nw`/Padé metal settings. Add
  attribute `fc_plasma` (`None`, a scalar, or a `(3, 3)` tensor in $\mathrm{hartree}^2$) and log it in
  `dump_flags`.
- **Behaviour change:** `fc=True` now applies to smearing references that previously matched PySCF;
  `fc=False` recovers the old result.

**Docs and example.** Add a metals section to `docs/krgwac_ewald.md` and update lines 19, 102 and 107
plus the module docstring (line 29). Add `examples/krgwac_ewald_metal.py` for Al, with a short helper
that computes a dense-mesh $\Omega_p^2$ via FFTDF `get_bands` for use as `fc_plasma` (pure functionals).

## Tests

Added to `fsec/standard/tests/test_krgwac_ewald.py`, each with a docstring:

- **Lindhard limit (fast):** NumPy tight-binding band, finite-$\mathbf q$ head vs $\Omega_p^2/\omega^2$.
- **Momentum (fast):** `get_pij` occ-vir matches `get_qij`; the diagonal matches central-difference
  $\partial\epsilon/\partial k$ on an H cell, which has no nonlocal term.
- **Insulator reduction (fast):** on a Γ-centred mesh with integer occupations, the metal route equals
  `get_eps_inv_q0` and $\Omega_p^2 \approx 0$.
- **Head assembly (fast):** Drude model matches the closed form
  $I_\pm(y) = \mp\frac{\pi}{2}\omega_p/(\omega_p \pm y)$, on-shell total is $-c/2$ across the window,
  and $\Delta=0$ gives no NaN.
- **Validation (fast):** accepts smearing; rejects `mu0`, `sigma=0`, and `twopole`.
- **Kernel (slow):** `vk - raw` equals $-c f_n$ on the diagonal, $\Sigma - \Sigma_{\rm raw}$ equals
  `get_sigma_fc`, and $-1 \le \epsilon^{-1}_{00}-1 \le 0$ on the grid.

## Verification

Use the `fsec-312` environment.

```bash
python -m pytest fsec/standard/tests/test_krgwac_ewald.py
python -m pytest -m slow fsec/standard/tests/test_krgwac_ewald.py
```

End-to-end study with `examples/krgwac_ewald_metal.py`: Al and Na `gth-dzvp` on even meshes (avoiding
the $N+1$ filling), Fermi smearing, with `fc=False`, `fc=True`, and `fc=True` with a fixed dense
`fc_plasma`. Note Γ-centred $2^3$ needs `fc_plasma`.

Acceptance criteria:

1. `fc=False` and `fc=True` extrapolate to the same $N_k\to\infty$ limit. This is the main check.
2. The $N_k^{-1/3}$ slope of absolute QP energies and $Z$ is reduced with `fc=True`.
3. No jump in $E^{\rm QP}-\epsilon$ across $\epsilon_F$.
4. QP energies are stable between `nw=400` and `800`, and across $\sigma$.

## Known limits and flagged issues

- **Raw $\Sigma_c$ keeps the near-pole problem.** PySCF's `get_sigma` puts the same pole at every
  $\epsilon_{m,\mathbf k-\mathbf q}$ near $\epsilon_F$, and the subtraction here only fixes the head.
  Padé input may stay noisy on unlucky meshes. If criterion 4 fails, the follow-up is separating the
  $\Sigma$ grid from the quadrature grid, which means forking `get_sigma`. I would stop and report
  rather than do that inside this change.
- **Raw finite-T mismatch.** Raw $\Sigma_x(f)$ and raw $\Sigma_c$(sharp) still disagree at small
  $\mathbf q \ne 0$. A head correction cannot fix that. It matters when $\sigma \gg v_F q_0$.
- **Singular $\Omega_p^2$ tensors** (1D or layered metals, or 1D k-meshes) are not resolved by Lebedev
  averaging at low $\omega$. 3D metals are fine.
- **Not yet validated on a real metal.** The checks so far are model-level, plus $\Omega_p$ on real
  cells. No metallic G0W0 has been run with this correction.
- **Existing bugs, out of scope here:** `get_sigma_outcore` ignores `kptlist` (`krgw_ac.py:827`), so
  `outcore=True` with a subset of k-points mismatches raw and head, for insulators too. A second
  `kernel` call reuses QP `mo_energy`. `make_rdm1_linear` assumes $T=0$.

## References

- T. Zhu and G. K.-L. Chan, JCTC 17, 727 (2021), Sec. 3.6.
- P. Liu, M. Kaltak, J. Klimeš, G. Kresse, PRB 94, 165109 (2016), Eqs. 79-80.
- J. Yan et al., PRB 83, 245122 (2011).
