import numpy as np

def flag_aspcap_giants(apogee_logg, apogee_teff):
    '''Select the giants as classified by ASPCAP for calibration purposes.

    This flags targets that ASPCAP would classify as giants and dwarfs for
    calibration. The relations are given in Holtzman et al (2018).'''
    giants = apogee_logg < np.minimum(2 + 2 / 1300 * (apogee_teff - 3500), 4.0)
    return giants

def correct_aspcap_teff(uncor_teff, uncor_mh, uncor_logg):
    '''Correct raw ASPCAP temperatures to be calibrated.'''
    giants = flag_aspcap_giants(uncor_logg, uncor_teff)

    tempcors = aspcap_teff_correction(uncor_mh, giants)

    corrected_temps = uncor_teff - tempcors
    return corrected_temps

def aspcap_teff_correction(uncor_mh, giants):
    '''Calculate appropriate temperature corrections to calibrate.'''
    temp_cors = np.zeros(len(uncor_mh))

    giant_corrections = aspcap_giant_teff_correction(uncor_mh[giants])
    dwarf_corrections = aspcap_dwarf_teff_correction(uncor_mh[~giants])
    
    temp_cors[giants] = giant_corrections
    temp_cors[~giants] = dwarf_corrections

    return temp_cors

def aspcap_dwarf_teff_correction(m_h):
    '''Return the temperature correction for dwarfs of the given metallicity.

    Calculate the temperature correction as a function of metallicity as
    reported in Holtzman et al (2018).'''
    dwarf_correction = np.poly1d(np.array(
        [ -26.0953, 13.1614, -36.3822]))

    corrections = dwarf_correction(m_h)

    return corrections

def aspcap_giant_teff_correction(m_h):
    '''Return the temperature correction for giants of a given metallicity

    Calculate the temperature correction as a function of metallicity as
    reported in Holtzman et al (2018).'''
    giant_correction = np.poly1d(np.array(
        [ 7.17561, 61.4774,  -51.5903]))

    corrections = giant_correction(m_h)

    return corrections

#############################
# Temperature Uncertainties #
#############################


def calc_aspcap_teff_uncertainties(uncor_teff, sn, uncor_mh, uncor_logg):
    '''Calculate the uncertainties in Teff.
    
    Use the relation in Holtzman et al (2018) to calculate the uncertainties in
    APOGEE temperature.'''
    giants = flag_aspcap_giants(uncor_logg, uncor_teff)

    sigtemps = aspcap_teff_uncertainties(uncor_teff, sn, uncor_mh, giants)

    return sigtemps

def aspcap_teff_uncertainties(uncor_teff, sn, uncor_mh, giants):
    '''Calculate appropriate temperature corrections to calibrate.'''
    temp_sigs = np.zeros(len(uncor_teff))

    giant_sigs = aspcap_giant_teff_uncertainties(
        uncor_teff[giants], sn[giants], uncor_mh[giants])
    dwarf_sigs = aspcap_dwarf_teff_uncertainties(
        uncor_teff[~giants], sn[~giants], uncor_mh[~giants])
    
    temp_sigs[giants] = giant_sigs
    temp_sigs[~giants] = dwarf_sigs

    return temp_sigs

def aspcap_dwarf_teff_uncertainties(uncor_teff, sn, uncor_mh):
    '''Calculate the teff uncertainty for dwarfs.

    Calculate the error in Teff using the dwarf relation reported in Holtzman
    et al (2018).'''
    Ateff = 4.583
    Bteff = 9.000290
    Cteff = -0.00130
    Dteff = -0.243

    logsig = (
        Ateff + Bteff * (uncor_teff - 4500) + 
        Cteff * (np.minimum(sn, 200) - 100) + Dteff * uncor_mh)
    sigTeff = np.exp(logsig)
    return sigTeff
    
def aspcap_giant_teff_uncertainties(uncor_teff, sn, uncor_mh):
    '''Calculate the Teff uncertainty for giants.

    Calculate the error in Teff using the giant relation reported in Holtzman
    et al (2018).'''
    Ateff = 4.361
    Bteff = 0.000604
    Cteff = -0.00196
    Dteff = -0.0659

    logsig = (
        Ateff + Bteff * (uncor_teff - 4500) + 
        Cteff * (np.minimum(sn, 200) - 100) + Dteff * uncor_mh)
    sigTeff = np.exp(logsig)
    return sigTeff
