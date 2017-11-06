'''
Functions probing consistency between spectoscopic and photometric rotation.

Spectroscopic rotation is usually in the form of a measured vsini, while
photometric rotation is generally in the form of a period. In order to
transform between them, a stellar radius is needed, as well as a statistical
treatment of sin(i). These functions give rough examples of where the two
measures of vsini and rotation period agree or not, and if the problem lies
with either the measurement of vsini, period, or the stellar radius.

The discrepancy between vsini and period is explored in:

apogee_vsini_distribution:
    Label objects with consistent and inconsistent vsinis and periods.

rotation_radial_velocity_variation:
    Label RV (non)variable to distinguish between consistent vsinis.

period_velocity_apogee:
    Display the apogee quality for targets in period/velocity space.

The radius is investigated with these functions:

plot_APOGEE_KIC_teff_DSEP_KIC_radius:
    Compare the DSEP and KIC radii based on the APOGEE and KIC Teff scales.

'''
import numpy as np
import matplotlib.pyplot as plt


def apogee_vsini_distribution(period, radius, vsini, vsini_floor=5):
    '''Plot the distribution of vsinis for an APOGEE sample.

    Will determine sini by calculating vsini * P / (2 * pi * R). For objects
    with vsini < vsini_floor, they will be treated as upper limits (ignored in
    this case).'''
    invalid_vsini_indices = vsini < 0
    vsini_limit_indices = np.logical_and(vsini >= 0, vsini <= vsini_floor)
    vsini_detections = np.logical_not(np.logical_or(
        invalid_vsini_indices, vsini_limit_indices))
    num_invalid = np.count_nonzero(invalid_vsini_indices)
    num_limits = np.count_nonzero(vsini_limit_indices)
    valid_period = period[np.where(vsini_detections)]
    valid_radius = radius[np.where(vsini_detections)]
    valid_vsini = vsini[np.where(vsini_detections)]
    print("Invalid vsinis: {0:d}".format(num_invalid))
    print("Vsini nondetections: {0:d}".format(num_limits))


    eq_vel = period_to_velocities(valid_period, valid_radius)

    impossible_vsini = valid_vsini > eq_vel
    photometric_cont = valid_vsini < eq_vel / 2
    contaminants = np.logical_or(impossible_vsini, photometric_cont)
    num_cont = np.count_nonzero(contaminants)
    print("Contaminants: {0:d}/{1:d}".format(num_cont, len(contaminants)))

    uncontam_eqvel = eq_vel[np.where(np.logical_not(contaminants))]
    uncontam_vsini = valid_vsini[np.where(np.logical_not(contaminants))]

    sini = valid_vsini / eq_vel

    plt.hist(sini, bins=15, range=(0, 1.5))
    plt.xlabel("Sin (i)")
    plt.ylabel("N")

def rotation_radial_velocity_variation(
    rv_nonvar, rv_var, vsini_colname="VSINI", Prot_colname="Prot"):
    '''Create a plot showing RV-variable/nonvariable objects.'''

    plt.semilogy(rv_nonvar[vsini_colname], rv_nonvar[Prot_colname], 'b*', 
                 ms=12, label="RV Variable")
    plt.semilogy(rv_var[vsini_colname], rv_var[Prot_colname], 'r*', ms=12,
                 label="RV Variable")
    plt.xlabel("v sin i (km/s)")
    plt.ylabel("Prot (day)")
    plt.title("Multiepoch with rotation")
    plt.legend(loc="upper right")

def period_velocity_apogee(
    periods, vsinis, apogee_flags):
    '''Plot the relationship between period & vsini for rapid rotators.

    This will put the rapid rotators which have been observed in APOGEE on a
    plot relating period and vsini.'''
    bad_indices = apogee_flags & 2**23 != 0
    # The 2**14 is a flag called VSINI_WARN. It does not trigger the STAR_BAD
    # or STAR_WARN flags.
    warn_indices = np.logical_and(apogee_flags & (2**7+2**14) != 0,
                                  np.logical_not(bad_indices))
    good_indices = np.logical_not(np.logical_or(bad_indices, warn_indices))

    plt.scatter(periods[good_indices], vsinis[good_indices], s=50, c="g",
                marker="o", label="good")
    plt.scatter(periods[warn_indices], vsinis[warn_indices], s=15, c="m",
                marker="s", label="warn")
    plt.scatter(periods[bad_indices], vsinis[bad_indices], s=15, c="r",
                marker="D", label="bad")
    plt.plot([1, 5], [51, 10], 'k-', label="Rsun")
    plt.plot([1, 5], [51/2.0, 10/2.0], 'k--', label="Rsun (min)")
    plt.plot([1, 5], [51*0.66, 10*0.66], 'b-', label="0.66 Rsun")
    plt.plot([1, 5], [51*0.66/2.0, 10*0.66/2.0], 'b--', label="0.66 Rsun (min)")

    plt.xlabel("Period (day)")
    plt.ylabel("vsini (km/s)")
    plt.ylim(0, 100)
    plt.xlim(1, 5)
#   plt.legend(loc="upper right")

################################################################################
# Radius #
################################################################################

def plot_APOGEE_KIC_teff_DSEP_KIC_radius(
    apogee_teff, kic_teff, dsep_radius, kic_radius, apogee_logg):
    '''Plot the radius of objects with respect to teff.'''
    giant_indices = apogee_logg < 3.5
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, sharex='all', sharey='all')
    ax1.plot(apogee_teff, dsep_radius, 'r*')
    ax1.plot(apogee_teff[giant_indices], dsep_radius[giant_indices], 'bo',
             label="APOGEE giants")
    ax2.plot(kic_teff, dsep_radius, 'r*')
    ax2.plot(kic_teff[giant_indices], dsep_radius[giant_indices], 'bo')
    ax3.plot(apogee_teff, kic_radius, 'r*')
    ax3.plot(apogee_teff[giant_indices], kic_radius[giant_indices], 'bo')
    ax4.plot(kic_teff, kic_radius, 'r*')
    ax4.plot(kic_teff[giant_indices], kic_radius[giant_indices], 'bo')
    ax1.set_xlim((7000, 3500))
    ax3.set_xlabel("APOGEE Teff (K)")
    ax4.set_xlabel("KIC Teff (K)")
    ax1.set_ylabel("DSEP radius (Rsun)")
    ax3.set_ylabel("KIC radius (Rsun)")
    ax1.legend()

    f.suptitle("APOGEE-McQuillan Good Fits")
