import sed



def redden_mag(band, truemag, EB_V, Rv=3.1):
    '''Redden the true magnitude value given E(B-V).
    
    Uses calculated values from CCM to calculate the reddened magnitude given
    an E(B-V) value.
    '''
    AV = EBV_to_AV(EB_V, Rv)
    extinction = AV_to_band_extinction(AV, band)
    extinctedmag = sed.calc_abs_magnitude(truemag, 10, extinction)

    return extinctedmag

def deredden_mag(band, extinctedmag, EB_V, Rv=3.1):
    '''Deredden an observed magnitude given E(B-V).

    Uses calculated values from CCM to obtain the dereddened magnitude given an
    E(B-V) value.
    '''
    extinction = redden_mag(band, 0, EB_V, Rv=Rv)
    truemag = extinctedmag - extinction

    return truemag

def redden_color(color, truecolor, EB_V, Rv=3.1):
    '''Redden the true color given E(B-V).

    Use calculated values from CCM to calculate the reddened color given an
    E(B-V) value.
    '''
    blueband, redband = split_color(color)
    blue_extinction = redden_mag(blueband, 0, EB_V, Rv=Rv)
    red_extinction = redden_mag(redband, 0, EB_V, Rv=Rv)
    reddened_color = truecolor + (blue_extinction - red_extinction)

    return reddened_color

def deredden_color(color, extincted_color, EB_V, Rv=3.1):
    '''Deredden the observed colro given E(B-V).

    Use calculated values from CCM to deredden a cover given an E(B-V) value.
    '''
    reddening_coeff = redden_color(color, 0, EB_V, Rv=Rv)
    dereddened_color = extincted_color - reddening_coeff

    return dereddened_color

###############################################################################
# Basic reddening functions #
###############################################################################

def EBV_to_AV(EB_V, Rv=3.1):
    '''Convert a differential extinction to a total extinction.

    This is essentially a function which multiplies EB_V times Rv. The Rv
    keyword is set to the canonical Milky-Way value of 3.1.'''
    Av = EB_V * Rv
    return Av

# These are coefficients that are given by the IRSA dust map service.
CCM_reddening = {"B": 1.337, "V": 1.000, "I": 0.479, "J": 0.282, "H": 0.190,
                 "K": 0.114, "Ks": 0.114, "Kp": 0.9}

def AV_to_Aband(AV, band, system="CCM"):
    '''Convert V-band extinction to extinction in a different band.

    This essentially converts AV to a different band by looking up ratios of
    extinction from a given reference. Currently the only reference implemented
    is that from Cardelli, Clayton, and Mathis (1989) (CCM).'''
    if system.lower() == "ccm":
        banddict = CCM_reddening
    Aband = banddict[band] * AV
    return Aband
    
def AV_to_Aband_err(AV_lower, AV_upper, band, system="CCM"):
    '''Convert the uncertainty from AV to the Aband.
    
    This will take a lower and upper bound of AV, and return a tuple containing
    the lower and upper bound of Aband.'''
    Aband_lower = AV_to_Aband(AV_lower, band, system)
    Aband_upper = AV_to_Aband(AV_upper, band, system)
    return (Aband_lower, Aband_upper)
