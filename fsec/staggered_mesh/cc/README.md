# Periodic restricted CCD

`KRCCD` is a small restricted k-point CCD layer built on the PySCF 2.14
`pyscf.pbc.cc.kccsd_rhf.RCCSD` implementation.  It is intended for closed-shell,
insulating, three-dimensional periodic systems sampled on a uniform k-point
mesh.  Staggered or multi-mesh methods are not implemented here.

The public constructor is:

```python
KRCCD(
    mf,
    frozen=None,
    mo_coeff=None,
    mo_occ=None,
    madelung_orbital=False,
    madelung_eri=False,
)
```

The signed constant follows the paper convention

```text
xi = -pyscf.pbc.tools.madelung(cell, kpts).
```

It is available as `cc.madelung_constant`.  The PySCF value itself is positive
for the usual three-dimensional cells.  With `madelung_orbital=True`, `+xi` is
added only to occupied orbital energies; virtual energies are unchanged.  The
occupied diagonal is changed consistently in both `eris.mo_energy` and
`eris.fock`: current PySCF uses the former in denominators and the latter in
residual intermediates.  With `madelung_eri=True`, the doubles residual receives
`2*xi*T2`.  Because PySCF returns denominator-divided updates, the implementation
adds the equivalent `(2*xi*T2)/eijab` term block by block using PySCF's padding
masks and `level_shift` convention.  The final CCD energy contraction is
unchanged.

Do not combine inherited `keep_exxdiv=True` with
`madelung_orbital=True`: the retained SCF exchange-divergence treatment may
already contain the occupied-orbital correction.  `KRCCD` warns about this
combination.  Set `keep_exxdiv=False` to use the explicit KRCCD correction.

`kernel_n(n)` starts both amplitudes at zero and performs exactly `n` raw,
undamped fixed-point updates.  Thus `kernel_n(1)` is the CCD(1)/MP2 starting
point and the ERI correction is inert on that first update.  The inherited
`kernel()` remains available for ordinary converged DIIS CCD; when both
Madelung corrections are enabled, the converged correlation energy should
match the uncorrected result within numerical convergence tolerance, while
finite-iteration CCD(n) energies generally differ.

Only `fsec.staggered_mesh.cc` exports `KRCCD` and its `KCCD` alias:

```python
from fsec.staggered_mesh.cc import KCCD, KRCCD
```

PySCF 2.14 does not provide a separate periodic CCD class.  This layer uses its
current restricted k-point CCSD ERI builder, padding, density-fitting, and
iteration APIs while fixing the singles amplitudes to zero.

The exact archived FFTDF regression is intentionally opt-in because it is much
slower than the focused algebra tests:

```bash
FSEC_RUN_ARCHIVED_REFERENCE=1 pytest \
    fsec/staggered_mesh/cc/tests/test_krccd.py -m slow
```

Method reference: X. Xing and L. Lin, “Inverse volume scaling of finite-size
error in periodic coupled cluster calculations,” *Phys. Rev. X* **14**,
011059 (2024).
