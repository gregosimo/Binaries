import numpy as np


def aspcap_dwarf_teff_correction(m_h, uncor_logg, cor_logg):
    '''Return the temperature correction for dwarfs of the given metallicity.

    First there is a check of whether the uncorrected loggs are greater than or
    less than 4.0. If greater, then a subgiant correction is used. If lower,
    then a dwarf correction is used.'''
    subgiant_correction = np.poly1d(np.array(
        [ 26.09507071, -13.16135675, 36.38213935]))
    dwarf_correction = np.poly1d(np.array(
        [ -7.17563374, -61.47743494,  51.59026248]))

    if np.any(cor_logg > 0):
        raise ValueError("Giant calibration not included yet.")

    dwarf_indices = uncor_logg <= 4.0
    subgiant_indices = uncor_logg > 4.0

    assert np.all(np.logical_or(dwarf_indices, subgiant_indices))

    corrections = np.zeros(len(m_h))
    corrections[dwarf_indices] = dwarf_correction(m_h[dwarf_indices])
    corrections[subgiant_indices] = subgiant_correction(m_h[subgiant_indices])

    return corrections

    

