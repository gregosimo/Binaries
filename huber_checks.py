'''
Functions which check the KIC parameters as measured by Huber et al (2013).
'''

import matplotlib.pyplot as plt

import hrplots as hr

def compare_DSEP_radii_to_KIC_radii(
    kic_radii, DSEP_radii, kic_radii_err_high, kic_radii_err_low, teff):
    '''Plot the fractional difference between DSEP radii and KIc radii.

    Calculate the fractional difference between the radii of the KIC and that
    calculated by DSEP. This will include the uncertainties of the KIC
    radius.'''
    frac = (kic_radii - DSEP_radii) / DSEP_radii
    fracerrs = [kic_radii_err_low / DSEP_radii, 
                kic_radii_err_high / DSEP_radii]

    print(frac)
    print(fracerrs)

    plt.errorbar(teff, frac, yerr=fracerrs, marker='o', color="b", ls="")
    hr.invert_x_axis()
    plt.xlabel("APOGEE Teff (K)")
    plt.ylabel("Fractional KIC - DSEP radius")
