"""Compare Gamma-point periodic KMP3 with molecular CCpy MP3 for H2 in a box.

Requires CCpy, which is not a dependency of fsec. Its PyPI metadata lists PySCF,
so ``--no-deps`` keeps an existing PySCF installation in place::

    pip install --no-deps coupled-cluster-py cclib periodictable psutil

As the box length L grows, periodic MP3 for this neutral molecule approaches the
molecular value. Three periodic treatments of the G = 0 Coulomb term are shown:

    ewald        exxdiv='ewald' with the Madelung counterterm (the KMP3 default)
    ewald-nomad  exxdiv='ewald' without the counterterm
    none         exxdiv=None (the KMP3 default has no counterterm here)

Near the origin the periodic Coulomb kernel with G = 0 removed is
1/r - xi + O(r^2 / L^3), where xi = tools.madelung(cell, kpts) ~ 2.837 / L. The
differences from the molecule should therefore fall off as

    ewald        L^-3 in E2 and E3
    ewald-nomad  L^-3 in E2, and L^-1 in E3 from the uncancelled numerator term
    none         L^-1 in E2 and E3, from shifted occupied energies

For ``none`` the L^-1 terms from the amplitude norm cancel between E2 and E3,
but the sum keeps the term xi * dE3/ds, where s shifts the occupied energies.
Its prefactor is set by E3, so E2 + E3 also falls off as L^-1, only more slowly.
L^-2 terms are still visible for ``none`` at these boxes, so its E3 and E2 + E3
slopes are not yet asymptotic.

The integrals use FFTDF because GDF leaves a nearly constant error of several
1e-6 Ha in E2 and E3, which hides the L^-3 tail at large L. Tightening the
auxiliary basis barely changes that error.

All energies are in Hartree and all lengths in Bohr.
"""

import numpy as np
from ccpy.drivers.driver import Driver
from pyscf import gto, mp, scf
from pyscf.pbc import gto as pbc_gto, scf as pbc_scf

import fsec
from fsec.staggered_mesh.mp import KMP3


BOND_LENGTH = 1.8
BASIS = 'gth-dzvp'
PSEUDO = 'gth-pade'
BOX_LENGTHS = (10.0, 15.0, 20.0, 30.0, 40.0)
# Log-log slopes use boxes at least this large. Below L = 20 the most diffuse
# basis function overlaps its periodic images.
FIT_MIN_LENGTH = 20.0
LABELS = ('ewald', 'ewald-nomad', 'none')


def h2_atoms(length):
    """Return H2 centred in a cubic box, with its bond along z."""
    center = length / 2
    half_bond = BOND_LENGTH / 2
    return [('H', (center, center, center - half_bond)),
            ('H', (center, center, center + half_bond))]


def make_cell(length):
    """Build a cubic cell of side ``length`` containing one H2 molecule."""
    cell = pbc_gto.Cell()
    cell.build(
        unit='B',
        a=np.eye(3) * length,
        atom=h2_atoms(length),
        basis=BASIS,
        pseudo=PSEUDO,
        ke_cutoff=200,
        precision=1e-9,
        verbose=0,
    )
    # FFTDF buffers scale with the mesh and fail at the default 4 GB for L = 40.
    cell.max_memory = 30000
    return cell


def molecular_reference():
    """Return molecular ``(E_HF, E2, E3)`` from RHF and CCpy MP3.

    CCpy rebuilds exact ERIs from the Mole, so this RHF must not use density
    fitting. It also labels orbitals unconditionally, hence ``symmetry='C1'``.
    """
    mol = gto.M(
        unit='B',
        atom=h2_atoms(0.0),
        basis=BASIS,
        pseudo=PSEUDO,
        symmetry='C1',
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    if not mf.converged:
        raise RuntimeError('The molecular RHF calculation did not converge')

    driver = Driver.from_pyscf(mf, nfrozen=0)
    driver.run_mbpt('mp2')
    e2 = driver.correlation_energy
    driver.run_mbpt('mp3')
    # CCpy reports E2 + E3, whereas KMP3.e_corr_mp3 is the increment alone.
    e3 = driver.correlation_energy - e2

    e2_pyscf = mp.MP2(mf).kernel()[0]
    if abs(e2 - e2_pyscf) > 1e-8:
        raise RuntimeError(f'CCpy E2 {e2} differs from PySCF MP2 {e2_pyscf}')
    return mf.e_tot, e2, e3


def periodic_krhf(cell, exxdiv):
    """Run Gamma-point KRHF with the default FFTDF integrals."""
    kmf = pbc_scf.KRHF(cell, cell.make_kpts([1, 1, 1]), exxdiv=exxdiv)
    kmf.conv_tol = 1e-12
    kmf.conv_tol_grad = 1e-9
    kmf.kernel()
    if not kmf.converged:
        raise RuntimeError('The periodic KRHF calculation did not converge')
    return kmf


def periodic_energies(length):
    """Return ``{label: (E_HF, E2, E3)}`` for one box length."""
    cell = make_cell(length)

    kmf = periodic_krhf(cell, 'ewald')
    standard = KMP3(kmf)
    standard.kernel(with_t2=False)
    nomad = KMP3(kmf, madelung=False)
    nomad.kernel(eris=standard.eris, with_t2=False)

    kmf_none = periodic_krhf(cell, None)
    unshifted = KMP3(kmf_none)
    unshifted.kernel(with_t2=False)

    runs = {'ewald': standard, 'ewald-nomad': nomad, 'none': unshifted}
    return {label: (run.e_hf, run.e_corr_mp2, run.e_corr_mp3)
            for label, run in runs.items()}


def error_table(results, reference):
    """Return ``{(label, length): errors}`` relative to the molecule.

    ``errors`` holds the differences in E_HF, E2, E3, and E2 + E3.
    """
    e_hf_mol, e2_mol, e3_mol = reference
    errors = {}
    for length, energies in results.items():
        for label, (e_hf, e2, e3) in energies.items():
            errors[label, length] = (e_hf - e_hf_mol, e2 - e2_mol, e3 - e3_mol,
                                     e2 + e3 - e2_mol - e3_mol)
    return errors


def fit_slope(lengths, values):
    """Return the log-log slope of ``abs(values)`` against ``lengths``."""
    return np.polyfit(np.log(lengths), np.log(np.abs(values)), 1)[0]


def print_results(results, reference):
    """Print the differences from the molecule and their fitted slopes."""
    print(f'\nfsec: {fsec.__file__}')
    print('Molecule: E_HF = %.10f  E2 = %.10f  E3 = %.10f' % reference)

    errors = error_table(results, reference)
    print('\n%6s %-12s %14s %14s %14s %14s' %
          ('L', 'method', 'dE_HF', 'dE2', 'dE3', 'd(E2+E3)'))
    for label in LABELS:
        for length in results:
            print('%6.1f %-12s %14.6e %14.6e %14.6e %14.6e'
                  % (length, label, *errors[label, length]))

    fit_lengths = [length for length in results if length >= FIT_MIN_LENGTH]
    if len(fit_lengths) < 2:
        return
    print(f'\nLog-log slopes for L >= {FIT_MIN_LENGTH}:')
    print('%-12s %8s %8s %8s %8s' % ('method', 'dE_HF', 'dE2', 'dE3', 'd(E2+E3)'))
    for label in LABELS:
        columns = np.array([errors[label, length] for length in fit_lengths])
        slopes = [fit_slope(fit_lengths, column) for column in columns.T]
        print('%-12s %8.2f %8.2f %8.2f %8.2f' % (label, *slopes))


if __name__ == '__main__':
    reference = molecular_reference()
    results = {length: periodic_energies(length) for length in BOX_LENGTHS}
    print_results(results, reference)
