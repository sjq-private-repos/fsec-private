# CCDSS time-reversal and gauge equations

This note collects the equations discussed while diagnosing complex CCDSS
structure factors beyond the Gamma point.

## Current aggregate origin

At zero momentum transfer, the occupied and virtual transition densities
reduce to identity matrices. The current scalar aggregation therefore gives
the same origin for all six channels:

$$
S_n(0)
=
\sum_{k_i k_j k_a}\sum_{ijab}
t_{ij}^{ab}(k_i,k_j,k_a),
\qquad n=1,\ldots,6,
$$

where the fourth momentum is fixed by momentum conservation,

$$
k_b = k_i+k_j-k_a \pmod{G}.
$$

The fitted curve is currently normalized channel by channel as

$$
\widetilde S_n(Q)=\frac{S_n(Q)}{S_n(0)}.
$$

## Time-reversal symmetry of T2

In a time-reversal-aligned orbital gauge, the amplitudes can be written
schematically as

$$
t_{ij}^{ab}(k_i,k_j,k_a)
=
\left[
t_{\bar i\bar j}^{\bar a\bar b}
(-k_i,-k_j,-k_a)
\right]^*.
$$

In a general Bloch-orbital gauge, time reversal also contains occupied and
virtual sewing matrices. The tensor relation is therefore of the form

$$
T(k_i,k_j,k_a)
=
(U_o\otimes U_o)\,
T(-k_i,-k_j,-k_a)^*\,
(U_v^\dagger\otimes U_v^\dagger).
$$

Consequently, time-reversal symmetry of the T2 operator does not require the
stored amplitudes to obey an elementwise complex-conjugation relation before
the orbital gauges at opposite momenta are aligned.

## Orbital rephasing

For the uniform rephasing

$$
\phi_i\longrightarrow e^{i\theta_o}\phi_i,
\qquad
\phi_a\longrightarrow e^{i\theta_v}\phi_a,
$$

the doubles amplitudes transform as

$$
t_{ij}^{ab}
\longrightarrow
e^{2i(\theta_o-\theta_v)}t_{ij}^{ab}.
$$

The unweighted origin therefore acquires the same arbitrary phase:

$$
S_n(0)
\longrightarrow
e^{2i(\theta_o-\theta_v)}S_n(0).
$$

Thus, a requirement that the raw aggregate $S_n(0)$ be real is not gauge
invariant, even though the underlying T2 operator continues to respect
time-reversal symmetry.

## Opposite-momentum relation

For an appropriately gauge-invariant scalar structure factor, time reversal
relates opposite momentum transfers:

$$
S_n(-Q)=S_n(Q)^*.
$$

The inversion-even contribution relevant to fitting a real auxiliary
function is consequently

$$
S_n^{\mathrm{even}}(Q)
=
\frac{S_n(Q)+S_n(-Q)}{2}
=
\operatorname{Re}S_n(Q),
$$

when the conjugation relation is satisfied.

## Gauge-invariant scalar projection

If one scalar curve per channel is desired, a gauge-invariant projection of
the tensor-valued channel structure factor onto the current T2 amplitudes is

$$
\overline S_n(Q)
=
\frac{\displaystyle\sum_X T_X^*[S_n(Q)]_X}
     {\displaystyle\sum_X |T_X|^2},
$$

with the composite external index

$$
X=(k_i,k_j,k_a,i,j,a,b).
$$

Because $[S_n(0)]_X=T_X$, this projection has the exact real origin

$$
\overline S_n(0)=1.
$$

It is also the least-squares scalar coefficient that projects the channel
tensor onto the T2 direction.

## Origin transition-density validation

When the zero-transfer transition densities are evaluated numerically rather
than assigned by hand, their active occupied and virtual blocks should obey

$$
\rho_{oo}(0,k)\approx I_{n_{\mathrm{occ}}(k)},
\qquad
\rho_{vv}(0,k)\approx I_{n_{\mathrm{vir}}(k)}.
$$

The planned validation measures the elementwise maximum deviation,

$$
\Delta_s(k)
=
\max_{pq}
\left|
[\rho_s(0,k)]_{pq}-[I_s(k)]_{pq}
\right|,
\qquad s\in\{oo,vv\},
$$

and requires

$$
\Delta_s(k)\leq 10^{-3}
$$

for every k-point, considering active orbitals only.

## Orbital covariance of T2 beyond MP2

The transformation law of the doubles amplitudes follows from the invariance
of the cluster operator, not from the MP2 expression for the amplitudes.  Write

$$
\hat T_2
=
\frac{1}{4}\sum_{ijab}
t_{ij}^{ab}\,
a_a^\dagger a_b^\dagger a_j a_i.
$$

Under independent unitary rotations within the occupied and virtual spaces,

$$
|\widetilde i\rangle=\sum_j |j\rangle U^o_{ji},
\qquad
|\widetilde a\rangle=\sum_b |b\rangle U^v_{ba},
$$

the same operator is represented by amplitudes that transform
contragrediently:

$$
\widetilde t_{i'j'}^{a'b'}
=
\sum_{ijab}
U^o_{ii'}U^o_{jj'}
(U^v_{aa'})^*(U^v_{bb'})^*
t_{ij}^{ab}.
$$

For orbital phase rotations this reduces to

$$
\widetilde t_{ij}^{ab}
=
e^{i(\theta_i+\theta_j-\theta_a-\theta_b)}t_{ij}^{ab}.
$$

This result does not use the MP2 formula

$$
t_{ij}^{ab,\mathrm{MP2}}
=
\frac{\langle ab\Vert ij\rangle}
{\epsilon_i+\epsilon_j-\epsilon_a-\epsilon_b}.
$$

It therefore also applies to converged nonlinear CCD amplitudes.  The CCD
residual equations are

$$
R_{ij}^{ab}(T_2)
=
\langle\Phi_{ij}^{ab}|
e^{-T_2}He^{T_2}
|\Phi_0\rangle
=0.
$$

Occupied--occupied and virtual--virtual rotations leave the reference
determinant invariant up to an overall phase and map the complete doubles
excitation space onto itself.  The residual consequently transforms as a
tensor in that space,

$$
\widetilde{\boldsymbol R}(\widetilde{\boldsymbol t})
=M^\dagger\boldsymbol R(\boldsymbol t),
$$

so that

$$
\boldsymbol R(\boldsymbol t)=0
\quad\Longrightarrow\quad
\widetilde{\boldsymbol R}(\widetilde{\boldsymbol t})=0.
$$

The nonlinear contractions do not alter this covariance: unitary factors on
contracted internal indices cancel pairwise.  Equivalently,
$e^{-T_2}He^{T_2}$ is a basis-independent operator.

For periodic amplitudes, rotations that retain the primitive-cell momentum
representation are block diagonal in momentum.  With

$$
k_b=k_i+k_j-k_a\pmod G,
$$

the doubles tensor transforms as

$$
\begin{aligned}
\widetilde t_{i'j'}^{a'b'}(k_i,k_j,k_a)
={}&\sum_{ijab}
U_o(k_i)_{ii'}U_o(k_j)_{jj'}\\
&\times U_v(k_a)_{aa'}^*U_v(k_b)_{bb'}^*
t_{ij}^{ab}(k_i,k_j,k_a).
\end{aligned}
$$

Time reversal is antiunitary and maps $k$ to $-k$.  In a general Bloch gauge,
the relation therefore also contains occupied and virtual sewing matrices:

$$
\begin{aligned}
t(k_i,k_j,k_a)
={}&B_o(k_i)B_o(k_j)
t(-k_i,-k_j,-k_a)^*\\
&\times B_v^\dagger(k_a)B_v^\dagger(k_b).
\end{aligned}
$$

The simpler elementwise relation

$$
t(k_i,k_j,k_a)=t(-k_i,-k_j,-k_a)^*
$$

holds only in a compatible time-reversal gauge.  Even when the converged CCD
operator obeys time reversal exactly, the unweighted coordinate sum

$$
\sum_{ijab}t_{ij}^{ab}
$$

is not an orbital-gauge-invariant scalar: the implicit all-ones contraction
does not transform along with the amplitudes.  It may consequently have a
nonzero imaginary part away from Gamma.  Invariant scalars instead require
contraction with tensors that transform covariantly.

These statements assume separate rotations that preserve the occupied and
virtual subspaces and a symmetry-preserving coupled-cluster root.
Occupied--virtual rotations change the reference; truncated local or selected
orbital spaces can break the invariance.  Likewise, after a noncanonical
rotation, retaining a canonical diagonal-denominator approximation rather
than solving the correctly transformed residual equations does not represent
the same CCD problem.

### References

- Høyvik *et al.*, "The divide-expand-consolidate family of coupled cluster
  methods: Numerical illustrations using second order Møller-Plesset
  perturbation theory," *J. Chem. Phys.* **136**, 014105 (2012), Appendix B.
  [DOI and author manuscript](https://pure.au.dk/ws/files/114486753/ERC1.pdf)
- T. Fleig, "Time-reversal symmetry in general coupled cluster theory,"
  *Phys. Rev. A* **77**, 062503 (2008).
  [DOI](https://doi.org/10.1103/PhysRevA.77.062503)
- B. Jeziorski, J. Paldus, and K. Jankowski, "Unitary group approach to
  spin-adapted open-shell coupled cluster theory," *Int. J. Quantum Chem.*
  **56**, 129--155 (1995).
  [DOI](https://doi.org/10.1002/qua.560560302)
- J. McClain *et al.*, "Gaussian-based coupled-cluster theory for the ground
  state and band structure of solids," *J. Chem. Theory Comput.* **13**,
  1209--1218 (2017). [Preprint](https://arxiv.org/abs/1701.04832)
