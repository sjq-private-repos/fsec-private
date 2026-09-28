# MP2SS FFTDF exchange-ordering bug and correction

## Bug and fix

The historical bug is in the memory-reduced FFTDF path used to construct the
MP2SS exchange structure factor with `t2_store_type="kikj"`. For exchange,
momentum conservation assigns the virtual orbitals to \((k_b,k_a)\), and
`FFTDF.ao2mo` was called with k-point labels
\((k_i,k_b,k_j,k_a)\). The corresponding MO coefficients, however, remained
in direct-term order \((C_i,C_a,C_j,C_b)\). The virtual coefficient matrices
therefore did not match their k-point labels. The error is invisible at
Gamma, where the relevant k points coincide, but changes the exchange
structure factor and MP2SS exchange correction on multi-k-point meshes; the
direct term is unaffected.

The corrected path supplies matching coefficients and labels,
\((C_i,C_b,C_j,C_a)\) with \((k_i,k_b,k_j,k_a)\). This is now the default.
`legacy_fft_exchange_orbital_order=True` deliberately reproduces the old
mismatch for FFTDF/`kikj` comparisons. A regression test checks that corrected
`kikj` exchange agrees with the independent `ki` implementation on a
\(1\times1\times3\) mesh (`rtol=1e-7`, `atol=1e-10`) and that the legacy result
is distinct.

## Diamond DZ validation

The validation system is the two-atom primitive diamond cell with lattice
vectors formed from permutations of \((0,3.3703265454,3.3703265454)\) bohr
and atoms at \((0,0,0)\) and
\((1.6851632727,1.6851632727,1.6851632727)\) bohr. Calculations used GTH-DZVP,
GTH-PBE, PySCF 2.14 FFTDF, a \(2\times2\times2\) k mesh, WS-truncated HF
(`exxdiv="vcut_ws"`), bare-Coulomb KMP2 correlation integrals, and an SCF
tolerance of \(10^{-10}\) Ha. The highest tested setting used a 150-Ha plane-
wave cutoff (FFT mesh \(29^3\)) and \(|q+G|\leq6\ \mathrm{bohr}^{-1}\).

| Energy (Ha per primitive cell) | Corrected | Legacy bug |
|---|---:|---:|
| WS-HF | -10.910504228 | -10.910504228 |
| MP2SS correlation | -0.206838552 | -0.182783483 |
| HF + MP2SS | -11.117342779 | -11.093287711 |

At identical settings, the legacy result is **24.055068 mEh less negative**
than the corrected result, entirely because of the exchange correction. The
previously observed **about 17 mEh** difference is the bug's spurious grid
sensitivity: changing the plane-wave cutoff from 120 Ha (FFT \(25^3\)) to
150 Ha (FFT \(29^3\)) changes the legacy MP2SS correlation energy by
**-17.089640 mEh**, whereas the corrected result changes by only
**-0.020148 mEh**.
