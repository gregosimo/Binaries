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
from scipy.stats import uniform
from scipy.special import erf
import astropy_util as au

import eclipsing_binaries as ebs
import catalog
import hrplots as hr
import biovis_colors as bc

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

def teff_velocity_apogee(
    teffs, vsinis, apogee_flags):
    '''Plot the relationship between period & vsini for rapid rotators.

    This will put the rapid rotators whic hhave been observed in APOGEE on a
    plot relating teff and vsini. We'll see if the slowly-rotating objects are
    cool. If so, it's possible they are significantly smaller.
    '''
    bad_indices = apogee_flags & 2**23 != 0
    # The 2**14 is a flag called VSINI_WARN. It does not trigger the STAR_BAD
    # or STAR_WARN flags.
    warn_indices = np.logical_and(apogee_flags & (2**7+2**14) != 0,
                                  np.logical_not(bad_indices))
    good_indices = np.logical_not(np.logical_or(bad_indices, warn_indices))

    plt.scatter(teffs[good_indices], vsinis[good_indices], s=50, c="g",
                marker="o", label="good")
    plt.scatter(teffs[warn_indices], vsinis[warn_indices], s=15, c="m",
                marker="s", label="warn")
    plt.scatter(teffs[bad_indices], vsinis[bad_indices], s=15, c="r",
                marker="D", label="bad")
    hr.invert_x_axis()

    plt.xlabel("Teff (K)")
    plt.ylabel("vsini (km/s)")
    plt.ylim(0, 80)
    plt.xlim(5700, 4000)
    plt.legend(loc="upper right")

def compare_rapid_rotator_period_vsini(
        vsini, period, radii, teff, xvalues, rapidperiod=5, apogee_ids=None, 
        xlim=()):
    '''Compare the expected period from vsini to the photometric period.

    This plots the actual photometric period against the period expected for
    the given vsini at the star's predicted DSEP radius. Radii are expected to
    be given in solar radii. 
    
    It will also plot the objects which are not expected to be rapid rotators.

    A list of apogee_ids will separate the double-lined spectroscopic binaries
    to the non double-lined spectroscopic binaries.
    '''
    if apogee_ids is not None:
        DLSB_indices = mark_DLSB_indices(apogee_ids)
        valid_indices = au.get_complement_indices(
            DLSB_indices, len(apogee_ids))

        DLSB_vsini = vsini[DLSB_indices]
        DLSB_period = period[DLSB_indices]
        DLSB_radii = radii[DLSB_indices]
        DLSB_teff = teff[DLSB_indices]
        DLSB_xvalues = xvalues[DLSB_indices]
        vsini = vsini[valid_indices]
        period = period[valid_indices]
        radii = radii[valid_indices]
        teff = teff[valid_indices]
        xvalues = xvalues[valid_indices]

        DLSB_representative_vsini_period = representative_norm * (
            2 * np.pi * DLSB_radii / (
                DLSB_vsini * km_per_sec_to_solRad_per_day))
        DLSB_period_ratio = DLSB_representative_vsini_period / DLSB_period
        plt.plot(DLSB_xvalues, DLSB_period_ratio, 'mo', label="DLSBs")

    definite_rapid_rotators = definite_rapid_rotator_vsini_indices(
        vsini, radii, highperiod=rapidperiod)
    possible_rapid_rotators = possible_rapid_rotator_vsini_indices(
        vsini, radii, highperiod=rapidperiod)
    non_rapid = np.logical_not(np.logical_or(definite_rapid_rotators,
                                             possible_rapid_rotators))

    max_vsini_period, rep_vsini_period, min_vsini_period = vsini_to_spot_period_range(
        vsini, radii)

    max_period_ratio = max_vsini_period / period
    period_ratio = rep_vsini_period / period
    min_period_ratio = min_vsini_period / period

    plt.errorbar(
        xvalues[non_rapid], period_ratio[non_rapid], 
        yerr=(max_period_ratio[non_rapid], 
              min_period_ratio[non_rapid]), 
        c="k", marker=".", ls="", label="Slow vsini rotators")
    plt.errorbar(
        xvalues[possible_rapid_rotators], period_ratio[possible_rapid_rotators], 
        yerr=(max_period_ratio[possible_rapid_rotators], 
              min_period_ratio[possible_rapid_rotators]), 
        c='b', marker="o", ls="", label="Possible rapid vsini rotators")
    plt.errorbar(
        xvalues[definite_rapid_rotators], period_ratio[definite_rapid_rotators], 
        yerr=(max_period_ratio[definite_rapid_rotators], 
              min_period_ratio[definite_rapid_rotators]),  
        c='r', marker="o", ls="", label="Definite rapid vsini rotators")
    plt.ylabel("Vsini Period / Photometric Period")
    plt.yscale("log")

def plot_Stauffer_APOGEE_vsini_comparison():
    '''Compare the vsini values from Stauffer & Hartmann to APOGEE.

    The vsini values are from select targets from the Pleiades.'''
    targets = catin.Stauffer_APOGEE_overlap()
    good_targets = good_aspcap_fits(targets)
    nondetections = np.logical_and(
        good_targets["vsini lim"] == stat.LOWER, good_targets["VSINI"] < 7)
    detected_targets = good_targets[~nondetections]

    Stauffer_errors = (detected_targets["vsini"] / 2 / 
                       (1 + detected_targets["R"])).filled(0)
    apogee_errors = 0.1 * detected_targets["VSINI"]

    plt.errorbar(detected_targets["vsini"], detected_targets["VSINI"],
                 apogee_errors, Stauffer_errors, 'b*')
    plt.plot([0, 25], [0, 25])
    plt.plot([0, 10, 10], [7, 7, 0], 'r--')
    plt.xlabel("Stauffer & Hartmann VSINI")
    plt.ylabel("APOGEE VSINI")
    plt.title("Good VSINI comparison")

def rapid_rotation_vsini_comparison_histogram(
        vsini, period, radii, xvalues, rapidperiod=5, xlim=None, nbins=10):
    '''Make a histogram of how concordant vsinis and periods are distributed.

    Creates a histogram which marks the percentage of consistent vsinis and
    periods over the xvalues distribution.'''
    assert(len(period) == len(xvalues))

    high_period, rep_period, low_period = vsini_to_spot_period_range(vsini, radii)

    high_period_ratio = high_period / period
    rep_period_ratio = rep_period / period
    low_period_ratio = low_period / period

    consistent_indices = np.where(np.logical_and(high_period_ratio > 1,
                                                 low_period_ratio <= 1))
    consistent_hist, bins = np.histogram(xvalues[consistent_indices], 
                                         bins=nbins, range=xlim)
    full_hist, bins = np.histogram(xvalues, bins=bins)

    frac = np.nan_to_num(consistent_hist / full_hist)
    plt.step(bins[:-1], frac, where="post")
    plt.ylabel("Fraction of consistent rapid rotators")
    plt.xlim(xlim)

def plot_velocity_vsini(max_vel, vsini, xvalue, vsini_lim=7):
    '''Plot the expected velocities and the measured vsini.

    Plot the velocity expected from the radius and period of objects, along
    with the measured vsini.'''
    sub_vsini = vsini.copy()
    sub_vsini[vsini < 0] = 0
    vsini = sub_vsini
    valid_vsini_indices = np.logical_or(
        vsini >= vsini_lim, max_vel >= vsini_lim)
    valid_vel = max_vel[np.where(valid_vsini_indices)]
    valid_vsini = vsini[np.where(valid_vsini_indices)]
    valid_xvalue = xvalue[np.where(valid_vsini_indices)]
    num_invalid = len(vsini) - np.count_nonzero(valid_vsini_indices)
    print("Invalid vsinis: {0:d}".format(num_invalid))

    plt.plot(valid_xvalue, valid_vel, 'bo', ms=6, label="Predicted V")
    plt.plot(valid_xvalue, valid_vsini, 'rd', label="V sin(i)", ms=4)
    for i in range(len(valid_xvalue)):
        if valid_vel[i] >= valid_vsini[i]:
            lc='k'
        else:
            lc='r'
        plt.plot([valid_xvalue[i]]*2, [valid_vel[i], valid_vsini[i]], ls='-', 
                 c=lc)
    plt.plot(plt.xlim(), [vsini_lim, vsini_lim], 'r--', 
             label="Detection Threshold")
    plt.ylabel("Rotational Velocity (km/s)")

def plot_velocity_with_errorbars(vsini, lowdiff, medvels, highdiff):
    '''Plot the predicted velocity against vsini.

    This will have error bars for the vsinis as well as the predicted
    velocities which should originate from the radii errors.'''
    sub_vsini = vsini.copy()
    sub_vsini[vsini < 0] = 0.0
    vsini = sub_vsini
    goodvels = np.logical_or(vsini > 7, medvels > 7)
    
    plt.errorbar(medvels[goodvels], vsini[goodvels], yerr=0.1*vsini[goodvels], 
                 xerr=[-lowdiff[goodvels], highdiff[goodvels]], fmt="b*")
    plt.plot([0, 80], [0, 80], 'k-', lw=3)
    plt.plot([0, 7, 7], [7, 7, 0], 'r--')
    plt.xlabel("Predicted velocity")
    plt.ylabel("V sini")

def rotation_radius(vsini, prot, vsini_mask=catalog.APOGEE_NULL):
    '''Calculate the maximum radius of a star with rotation period and vsini.

    This function will essentially calculate VSINI * Prot. It's assumed that
    vsini is given in km/s and prot is given in days. For targets which do not
    have a vsini value, this will recognize the APOGEE mask and propagate
    it to the output.
    '''
    masked_vsini = np.ma.masked_equal(vsini, -9999.0)
    conv = 24*60*60*1e5/6.96e10/2/np.pi

    # This is the radius estimated by rotation.
    rotation_radius = masked_vsini * prot* conv

    filled_radii = rotation_radius.filled(-9999.0)

    return filled_radii

def plot_rapid_rotation_vsini(
    vsinis, radii, teffs, highperiod=5, apogee_ids=None):
    '''Select out the rapid rotators based on vsinis.

    This will select out those objects with vsinis that can be a part of a
    rapidly-rotating population, which is defined to be in the given period
    range. The conversion between vsini and teff will be done using the radii
    given.

    If a list of apogee IDs is given, then the objects which are flagged as
    double-lined spectroscopic binaries will be marked separately on the
    figure.
    '''
    if apogee_ids is not None:
        DLSB_indices = mark_DLSB_indices(apogee_ids)
        valid_indices = au.get_complement_indices(
            DLSB_indices, len(apogee_ids))

        DLSB_vsinis = vsinis[DLSB_indices]
        DLSB_radii = radii[DLSB_indices]
        DLSB_teffs = teffs[DLSB_indices]
        vsinis = vsinis[valid_indices]
        radii = radii[valid_indices]
        teffs = teffs[valid_indices]

        plt.plot(DLSB_teffs, DLSB_vsinis, 'mo', label="DLSBs")

    print(vsinis)
    definite_rapid_rotators = definite_rapid_rotator_vsini_indices(
        vsinis, radii, highperiod=highperiod)
    possible_rapid_rotators = possible_rapid_rotator_vsini_indices(
        vsinis, radii, highperiod=highperiod)
    non_rapid = np.logical_not(np.logical_or(definite_rapid_rotators,
                                             possible_rapid_rotators))

    plt.plot(teffs[non_rapid], vsinis[non_rapid], 'k.', label="Non-rapid")
    plt.plot(teffs[possible_rapid_rotators], vsinis[possible_rapid_rotators], 
             'bo', label="Possible Rapid Rotators")
    plt.plot(teffs[definite_rapid_rotators], vsinis[definite_rapid_rotators], 'ro', 
        label="Definite Rapid Rotators")
    hr.invert_x_axis()
    plt.xlabel("Teff (K)")
    plt.ylabel("V sin i (km/s)")
    plt.ylim(0, 80)
    plt.xlim(7000, 3200)

def cool_dwarf_subgiants_comparison(
    dwarf_teff, dwarf_logg, dwarf_vsini, subgiant_teff, subgiant_logg,
    subgiant_vsini):
    '''Highlight high-vsini dwarfs and subgiants on an HR diagram.

    Plot the Teff and Log(g) for dwarf and subgiant groups. It will highlight
    targets with high vsinis.
    '''
    vsini_dwarf_detections = dwarf_vsini >= 7
    vsini_subgiant_detections = subgiant_vsini >= 7
    vsini_dwarf_nondetections = dwarf_vsini < 7
    vsini_subgiant_nondetections = subgiant_vsini < 7

    hr.logg_teff_plot(dwarf_teff[vsini_dwarf_nondetections], 
                      dwarf_logg[vsini_dwarf_nondetections], 
                      style="bx", label="Huber dwarfs")
    hr.logg_teff_plot(subgiant_teff[vsini_subgiant_nondetections], 
                      subgiant_logg[vsini_subgiant_nondetections], 
                      style="rd", label="Huber subgiants")
    hr.logg_teff_plot(dwarf_teff[vsini_dwarf_detections], 
                      dwarf_logg[vsini_dwarf_detections],
                      style="ws", label="Dwarf vsini")
    hr.logg_teff_plot(subgiant_teff[vsini_subgiant_detections], 
                      subgiant_logg[vsini_subgiant_detections],
                      style="ms", label="Subgiant vsini")
    plt.xlabel("APOGEE Teff (K)")
    plt.ylabel("APOGEE logg (uncalibrated)")
    plt.ylim((4.8, 3.2))

def apokasc_logg_rotation_trend(apogee_logg, asteroseismic_logg, vsini):
    '''Plot the log(g) comparison against rotation.

    Plot the log(g) measured from asteroseismology against the log(g)
    determined spectroscopically against vsini. One thing that may explain why
    the cool stars look completely off is if rotation causes log(g) values to
    be off for spectroscopic parameters.'''
    filled_vsini = vsini.copy()
    filled_vsini[np.where(vsini < 0)] = 0.0
    vsini_detections = vsini > 7
    logg_diff = (apogee_logg - asteroseismic_logg)
    subgiants = asteroseismic_logg < 4.1
    plt.plot(filled_vsini[~subgiants], logg_diff[~subgiants], 'ko',
             label="dwarfs")
    plt.plot(filled_vsini[subgiants], logg_diff[subgiants], 'ro',
             label="subgiants")
    print("Slow scatter: {0:.2f}".format(np.std(logg_diff[~vsini_detections])))
    print("Fast scatter: {0:.2f}".format(np.std(logg_diff[vsini_detections])))
    plt.plot([7, 7], [-0.1, 0.5], 'b--')

#####################################
# Vsini, period, radius conversions #
#####################################

def vsini_to_spot_period_range(vsini, radii):
    '''Converts vsinis to predicted periods using radii.
    
    Returns a 3-tuple containing the high-limit to the period, the 
    representative period, and the low-limit to the period assuming that
    starspots are not seen at sin(i) < 0.5.'''
    representative_norm = np.sin(np.pi/4)

    max_vsini_period = vsini_to_max_period(vsini, radii)
    representative_vsini_period = representative_norm * max_vsini_period
    vsini_period_range = 0.5 * max_vsini_period

    return max_vsini_period, representative_vsini_period, vsini_period_range

def vsini_to_max_period(vsini, radii):
    '''Converts vsinis to maximum periods using radii.

    The given period is a maximum period because if the period was smaller, the
    same vsini value could be accounted for by having a less favorable
    inclination. The returned periods are given in days.'''
    km_per_sec_to_solRad_per_day = 1e5 / 7e10 *60*60*24

    max_period = 2 * np.pi * radii / (vsini * km_per_sec_to_solRad_per_day)

    return max_period

def period_to_velocities(period, radii):
    '''Convert periods to predicted velocities.

    Return the quantity 2 * pi * radii / period, but in units of km/s if period
    and radii are given in days and solar radii.'''
    solRad_per_day_to_km_per_sec = 7e10 / (1e5 * 60 * 60 * 24)
    velocity = 2 * np.pi * radii / period * solRad_per_day_to_km_per_sec

    return velocity

def period_to_velocities_uncertainties(period, radii, radius_up, radius_down):
    '''Convert periods to predicted velocities with uncertainties.

    Return the quantity 2 * pi * radii / period in terms of km/s if period and
    radii are given in days and solar radii. It also takes upper and lower
    limits of the radii error bars. This will return a 3-tuple with the lower
    limit, most probable value, and the upper value.'''
    solRad_per_day_to_km_per_sec = 7e10 / (1e5 * 60 * 60 * 24)
    velocity = 2 * np.pi * radii / period * solRad_per_day_to_km_per_sec
    velocity_up = 2 * np.pi * (radii + radius_up) / period * solRad_per_day_to_km_per_sec
    velocity_down = 2 * np.pi * (radii + radius_down) / period * solRad_per_day_to_km_per_sec
    print(np.any(velocity_down < 0))

    updiff = velocity_up - velocity
    downdiff = velocity_down - velocity

    return (downdiff, velocity, updiff)

#########################
# Select rapid rotators #
#########################

def rapid_rotator_vsini_indices(vsinis, radii, lowperiod=0, highperiod=np.inf):
    '''Get indices with vsinis corresponding to photometric period range.

    Given a set of vsinis and radii, select those which ought to show a
    photometric period between lowperiod and highperiod. The radii should be
    given in terms of solar radii.'''
    # Convert solar radii / day to km/s
    solRad_per_day_to_km_per_s = 7e10 / 1e5 / 24 / 60 / 60
    highvel = 2 * np.pi * np.array(radii) / lowperiod * solRad_per_day_to_km_per_s
    lowvel = np.pi * radii / highperiod * solRad_per_day_to_km_per_s
    indices = np.where(np.logical_and(vsinis < highvel, vsinis > lowvel))

    return indices
                                      
def definite_rapid_rotator_vsini_indices(vsinis, radii, highperiod=np.inf):
    '''Get indices of vsinis definitely corresponding to short periods.

    Given a set of vsinis and radii, select those which definitely ought to 
    show a photometric period less than highperiod. That's because vsini >
    vcrit, which is the circular velocity of a spot on the surface of a star
    rotating at highperiod.'''
    solRad_per_day_to_km_per_s = 7e10 / 1e5 / 24 / 60 / 60
    lowvel = (2 * np.pi * np.array(radii) / highperiod * 
               solRad_per_day_to_km_per_s)
    indices = vsinis >= lowvel
    return indices

def possible_rapid_rotator_vsini_indices(vsinis, radii, highperiod=np.inf):
    '''Get indicies of vsinis possibly corresponding to short periods.

    Select those vsinis where if the sin i is unfavorable, then the object could
    possibly be a rapid rotator. Otherwise, it's likely a slow rotator with a
    favorable inclination.'''
    solRad_per_day_to_km_per_s = 7e10 / 1e5 / 24 / 60 / 60
    highvel = (2 * np.pi * np.array(radii) / highperiod * 
               solRad_per_day_to_km_per_s)
    lowvel = (np.pi * np.array(radii) / highperiod * 
              solRad_per_day_to_km_per_s)
    indices = np.logical_and(vsinis > lowvel, vsinis <= highvel)
    return indices

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

def teff_radius_apogee(
    teffs, vsinis, periods):
    '''Plots the inferred radius vs teff for rapid rotators.

    The inferred radius will basically be vsini * P. Typical bounds on sini
    will also be displayed for clarity. If cool objects have a large radius,
    then this may be indicative of subgiant contamination.'''
    conv = 24*60*60*1e5/6.96e10/2/np.pi

    good_indices = vsinis > 7
    plt.scatter(
        teffs[good_indices], vsinis[good_indices]*periods[good_indices]*conv, 
        s=50, c="g", marker="o", label="good")
    hr.invert_x_axis()

    # Use the isochrones to determine the radius as a function of Teff.
    isochrone = sed.read_DSEP_isochrone(0.0, 2)
    isoteff = isochrone[np.where(np.logical_and(
        isochrone["LogTeff"] > np.log10(4000), 
        isochrone["EEP"] < 73))]
    teffs = 10**isoteff["LogTeff"]
    radii = 10**(isoteff["LogL/Lo"] / 2 - 
                 2 * (isoteff["LogTeff"] - np.log10(5777)))

    plt.plot(teffs, radii, 'k-', label="DSEP")
    plt.plot(teffs, radii*0.5, 'k--', label="Min.")

    plt.xlim(6600, 4000)
    plt.ylim(0, 5)
    plt.xlabel("Teff (K)")
    plt.ylabel("vsini * P (Rsun)")

def rotation_radius_comparison(
    asteroseismic_radii, vsinis, periods):
    '''Plots the asteroseismic radius vs R sini from rotation.

    The inferred radius will basically be vsini * P.'''
    good_indices = vsinis > 7
    inferred_radii = rotation_radius(
        vsinis[good_indices], periods[good_indices])
    plt.plot(
        asteroseismic_radii[good_indices], inferred_radii, c="g", marker="o",
        ls="None")
    plt.plot([0, 4], [0, 4], 'k-')
    plt.xlabel("Asteroseismic Radius (Rsun)")
    plt.ylabel("Inferred R sini (Rsun)")

def compare_rotation_velocity_radius(
    vsini, period, radii, raderr_below, raderr_above, vsini_lim=10):
    '''Evaluate rotation quality in velocity and radius space.

    Create a double-paneled figure that plots the same data in velocity space
    and radius space for clarity of understanding.'''
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    valid_indices = vsini > vsini_lim
    valid_vsini = vsini[valid_indices]
    valid_period = period[valid_indices]
    valid_radii = radii[valid_indices]
    valid_raderr_above = raderr_above[valid_indices]
    valid_raderr_below = raderr_below[valid_indices]

    downvel, infvel, upvel = period_to_velocities_uncertainties(
        valid_period, valid_radii, valid_raderr_below, valid_raderr_above)

    ax1.errorbar(
        infvel, valid_vsini, xerr=[-downvel, upvel], yerr=0.1*valid_vsini, 
        fmt='b*')
    ax1.plot([0, 80], [0, 80], 'k-')
    ax1.plot([0, 80], [vsini_lim, vsini_lim], 'r--', label="Detection Limit")
    plt.sca(ax1)
#    plt.legend(loc="upper right")
    ax1.set_xlabel("Inferred equatorial velocity (km/s)")
    ax1.set_ylabel("V sini (km/s)")

    inferred_radii = rotation_radius(valid_vsini, valid_period)
    ax2.errorbar(
        valid_radii, inferred_radii, yerr=0.1*inferred_radii, 
        xerr=[-valid_raderr_below, valid_raderr_above], fmt='b*')
    ax2.plot([0, 4.0], [0, 4.0], 'k-')
    ax2.set_xlabel("Radius (Rsun)")
    ax2.set_ylabel("Inferred R sini (Rsun)")

    inferred_period = vsini_to_spot_period_range(valid_vsini, valid_radii)[1]
    # Dealing with errors is difficult because the vsini and radius errors have
    # a unspecified interplay, especially since radius is asymmetric. Instead
    # of trying to calculate some form, I'll take the maximum of either the
    # vsini error or the radius error.
    radius_fractional_errors = ((valid_raderr_above + valid_raderr_below) / 
                                valid_radii)
    vsini_fractional_errors = 0.1
    inferred_period_err_up = np.where(
        radius_fractional_errors >= vsini_fractional_errors, 
        vsini_to_spot_period_range(
            valid_vsini, valid_radii + valid_raderr_above)[1] - inferred_period, 
        vsini_to_spot_period_range(
            valid_vsini*(1-vsini_fractional_errors), valid_radii)[1] - inferred_period)
    inferred_period_err_down = np.where(
        radius_fractional_errors >= vsini_fractional_errors, 
        vsini_to_spot_period_range(
            valid_vsini, valid_radii + valid_raderr_below)[1] - inferred_period, 
        vsini_to_spot_period_range(
            valid_vsini*(1+vsini_fractional_errors), valid_radii)[1] - inferred_period)
    ax3.errorbar(
        valid_period, inferred_period, 
        yerr=[-inferred_period_err_down, inferred_period_err_up], fmt='b*')

    ax3.plot([0, 15.0], [0, 15.0], 'k-')
    ax3.set_xlabel("McQuillan Period (day)")
    ax3.set_ylabel("Inferred P / sin(i) (day)")

def plot_vsini_velocity(
    vsini, period, radii, raderr_below, raderr_above, color='k', ax=None, 
        vsini_fracerr=0.15, label="", xticks=10, yticks=10, sini_label=True):
    '''Make a plot of vsini vs velocity.

    Vsini will be on the y-axis while velocity will be on the x-axis. The
    bottom right should be a region of allowed space while the top left is
    disallowed.'''
    if not ax:
        ax = plt.subplots(111, figsize=(5,5))

    downvel, infvel, upvel = period_to_velocities_uncertainties(
        period, radii, raderr_below, raderr_above)

    ax.errorbar(
        infvel, vsini, xerr=[-downvel, upvel], yerr=vsini_fracerr*vsini/2, 
        color=color, ls="None", marker="*")
    au.adjust_axes(ax, 0, infvel+upvel, 0, vsini*(1+vsini_fracerr/2),
                   xticks, yticks)
    min_x, max_x = ax.get_xlim()
    min_y, max_y = ax.get_ylim()
    # Show sini = 1 and sini = 1/2.
    med_x = min_x + 0.8*(max_x - min_x)
    med_y = min_y + 0.8*(max_y - min_y)
    if max_y > max_x:
        bound_point = max_x, max_x
    else:
        bound_point = max_y, max_y
    ax.plot([0, bound_point[0]], [0, bound_point[1]], 'k-', lw=3)
    ax.fill_between([0, bound_point[0]], [max_y, max_y], 
                    [0, bound_point[1]], hatch="\\", facecolor="white",
                    edgecolor="gray")

    if max_y/2 > max_x:
        half_bound = max_x, max_x/2
    else:
        half_bound = 2*max_y, max_y
    ax.plot([0, half_bound[0]], [0, half_bound[1]], 'k--')

    if sini_label:
        rotangle = np.arctan(
            (bound_point[1] - min_y) / (max_y-min_y) * (max_x - min_x) /
            (bound_point[0] - min_x))/np.pi*180
        half_rotangle = np.arctan(
            (half_bound[1] - min_y) / (max_y-min_y) * (max_x - min_x) /
            (half_bound[0] - min_x))/np.pi*180
        med = min(med_x, med_y)
        ax.text(med, med, r"$\sin i = 1$", rotation=rotangle)
        ax.text(med, med/2, r"$\sin i = 0.5$", rotation=half_rotangle)

    ax.annotate(label, xy=(0.55, 0.05), xycoords="axes fraction",
                 fontsize=14, color="white")

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_xlabel(r"$v_{eq}$ (km/s) from $R$ and $P_{rot}$")
    ax.set_ylabel("APOGEE $v \sin i$ (km/s)")

def plot_rotation_radius(
        vsini, period, radii, raderr_below, raderr_above, color="k", ax=None,
        vsini_fracerr=0.15, label="", xticks=0.2, yticks=1, sini_label=True):
    '''Plot the radius against inferred radius.
    
    Radius will be on the y-axis while inferred radius will be on the x-axis.'''
    if not ax:
        ax = plt.subplots(111, figsize=(5,5))

    inferred_radii = rotation_radius(vsini, period)

    ax.errorbar(
        inferred_radii, radii, yerr=[-raderr_below, raderr_above],
        xerr=vsini_fracerr*inferred_radii/2, color=color, ls="None", marker="*")
    au.adjust_axes(ax, 0, inferred_radii+raderr_above, 0, 
                   radii*(1+vsini_fracerr/2), xticks, yticks)
    min_x, max_x = ax.get_xlim()
    min_y, max_y = ax.get_ylim()
    # Show sini = 1 and sini = 1/2.
    med_x = min_x + 0.8*(max_x - min_x)
    med_y = min_y + 0.8*(max_y - min_y)
    if max_y > max_x:
        bound_point = max_x, max_x
    else:
        bound_point = max_y, max_y
    ax.plot([0, bound_point[0]], [0, bound_point[1]], 'k-', lw=3)

    if max_y > max_x/2:
        half_bound = max_x/2, max_x
    else:
        half_bound = max_y, max_y*2
    ax.plot([0, half_bound[0]], [0, half_bound[1]], 'k--')

    if sini_label:
        rotangle = np.arctan(
            (bound_point[1] - min_y) / (max_y-min_y) * (max_x - min_x) /
            (bound_point[0] - min_x))/np.pi*180
        half_rotangle = np.arctan(
            (half_bound[1] - min_y) / (max_y-min_y) * (max_x - min_x) /
            (half_bound[0] - min_x))/np.pi*180
        med = min(med_x, med_y)
        ax.text(med, med, r"$\sin i = 1$", rotation=rotangle)
#       ax.text(med, med/2, r"$\sin i = 0.5$", rotation=half_rotangle)

    ax.annotate(label, xy=(0.55, 0.05), xycoords="axes fraction",
                 fontsize=14, color="white")

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_xlabel("$R \sin i$ (Rsun) from $v \sin i$ and $P_{rot}$")
    ax.set_ylabel("DSEP Radius (Rsun)")

def plot_rotation_period(
        vsini, period, radii, raderr_below, raderr_above, color="k", ax=None,
        vsini_fracerr=0.15, label="", xticks=0.2, yticks=1, sini_label=True):
    '''Plot the radius against inferred radius.
    
    Radius will be on the y-axis while inferred radius will be on the x-axis.'''
    if not ax:
        ax = plt.subplots(111, figsize=(5,5))

    inferred_period = vsini_to_spot_period_range(vsini, radii)[0]
    # Dealing with errors is difficult because the vsini and radius errors have
    # a unspecified interplay, especially since radius is asymmetric. Instead
    # of trying to calculate some form, I'll take the maximum of either the
    # vsini error or the radius error.
    radius_fractional_errors = ((raderr_above + raderr_below) / 
                                radii)
    inferred_period_err_up = np.where(
        radius_fractional_errors >= vsini_fracerr/2, 
        vsini_to_spot_period_range(
            vsini, radii + raderr_above)[0] - inferred_period, 
        vsini_to_spot_period_range(
            vsini*(1-vsini_fracerr/2), radii)[0] - inferred_period)
    inferred_period_err_down = np.where(
        radius_fractional_errors >= vsini_fracerr/2, 
        vsini_to_spot_period_range(
            vsini, radii + raderr_below)[0] - inferred_period, 
        vsini_to_spot_period_range(
            vsini*(1+vsini_fracerr/2), radii)[0] - inferred_period)

    ax.errorbar(
        inferred_period, period, 
        xerr=[-inferred_period_err_down, inferred_period_err_up], color=color,
        ls="None", marker="*")
    au.adjust_axes(
        ax, 0, inferred_period+inferred_period_err_up, 0,
        inferred_period*(1+vsini_fracerr/2), xticks, yticks)
    min_x, max_x = ax.get_xlim()
    min_y, max_y = ax.get_ylim()
    # Show sini = 1 and sini = 1/2.
    med_x = min_x + 0.8*(max_x - min_x)
    med_y = min_y + 0.8*(max_y - min_y)
    if max_y > max_x:
        bound_point = max_x, max_x
    else:
        bound_point = max_y, max_y
    ax.plot([0, bound_point[0]], [0, bound_point[1]], 'k-', lw=3)

    if max_y/2 > max_x:
        half_bound = max_x, max_x/2
    else:
        half_bound = 2*max_y, max_y
    ax.plot([0, half_bound[0]], [0, half_bound[1]], 'k--')

    if sini_label:
        rotangle = np.arctan(
            (bound_point[1] - min_y) / (max_y-min_y) * (max_x - min_x) /
            (bound_point[0] - min_x))/np.pi*180
        half_rotangle = np.arctan(
            (half_bound[1] - min_y) / (max_y-min_y) * (max_x - min_x) /
            (half_bound[0] - min_x))/np.pi*180
        med = min(med_x, med_y)
        ax.text(med, med, r"$\sin i = 1$", rotation=rotangle)
#       ax.text(med, med/2, r"$\sin i = 0.5$", rotation=half_rotangle)

    ax.annotate(label, xy=(0.55, 0.05), xycoords="axes fraction",
                 fontsize=14, color="white")

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_xlabel("$P / {\sin i}$ (day) from $v \sin i$ and $R$")
    ax.set_ylabel("McQuillan Period (day)")

def plot_rotation_velocity_radius(
        vsini, period, radii, raderr_below, raderr_above, color="k",
        subplot_tup=None, label=""):
    '''Make a 3-paneled figure comparing measured and inferred radii.

    This function plots vsini vs inferred v, radius vs inferred rsini, and
    period vs inferred P/sini. It assumes the inputted data already contain the
    necessary cuts.'''
    if not subplot_tup:
        subplot_tup = plt.subplots(1, 3, figsize=(15, 5))
        label_sini = True
    else:
        label_sini = False
    f, (ax1, ax2, ax3) = subplot_tup

    vsini_fractional_error = 0.15
    plot_vsini_velocity(
        vsini, period, radii, raderr_below, raderr_above, ax=ax1, 
        vsini_fracerr=vsini_fractional_error, xticks=10, yticks=10,
        color=color, sini_label=label_sini)

    plot_rotation_radius(
        vsini, period, radii, raderr_below, raderr_above, ax=ax2, 
        vsini_fracerr=vsini_fractional_error, xticks=1, yticks=0.2,
        color=color, sini_label=False)

    plot_rotation_period(
        vsini, period, radii, raderr_below, raderr_above, ax=ax3, 
        vsini_fracerr=vsini_fractional_error, xticks=0.5, yticks=5,
        color=color, sini_label=False)

    return subplot_tup

def rotation_teff_test(
    vsini, period, teff, metallicity, alpha, apogee_flags, age=2.0):
    '''Test the subgiant status using displacement in radius.

    This will see if the subgiants with large radius displacments also have
    lower logg, which correspond to the rotational modulation and period
    matching the atmosphere.
    '''
    compare_rotation_DSEP_radius_ratio(
        vsini, period, teff, metallicity, alpha, apogee_flags, teff, 
        (6600, 4000))
    plt.xlabel("Teff (K)")
    plt.title("MS Displacement")

def rotation_subgiant_test(
    vsini, period, teff, logg, metallicity, alpha, apogee_flags, age=2.0):
    '''Test the subgiant status using displacement in radius.

    This will see if the subgiants with large radius displacments also have
    lower logg, which correspond to the rotational modulation and period
    matching the atmosphere.
    '''
    compare_rotation_DSEP_radius_ratio(
        vsini, period, teff, metallicity, alpha, apogee_flags, logg, (2.9, 5.0))
    plt.xlabel("log (g)")
    plt.title("MS Displacement")

def rotation_period_test(
    vsini, period, teff, metallicity, alpha, apogee_flags, age=2.0):
    '''Test the subgiant status using displacement in radius.

    This will see if the subgiants with large radius displacments also have
    lower logg, which correspond to the rotational modulation and period
    matching the atmosphere.
    '''
    compare_rotation_DSEP_radius_ratio(
        vsini, period, teff, metallicity, alpha, apogee_flags, period, (1, 5))
    plt.xlabel("Period (day)")
    plt.title("MS Displacement")

def compare_rotation_DSEP_radius_ratio(
    vsini, period, radii, apogee_flags, xvalue, xlimits):
    '''Plot the dependence of rotation radius ratio to another value.

    The rotation radius requires vsini, the radius and period.

    The ratio of radii will be plotted against xvalue. The points with "good"
    APOGEE flags will be plotted as large green circles while those with "warn"
    will be lotted as magenta squares. Generally bad points will not have valid
    Teff values and won't be visible, but if they are, they will be red
    circles. A radius ratio of 1 is marked on the figure.'''

    usable_indices = au.multi_logical_and(
        vsini != catalog.APOGEE_NULL, teff != catalog.APOGEE_NULL, 
        xvalue != catalog.APOGEE_NULL, metallicity != catalog.APOGEE_NULL, 
        alpha != catalog.APOGEE_NULL)
    usable_flags = apogee_flags[usable_indices]
    usable_vsini = vsini[usable_indices]
    usable_period = period[usable_indices]
    usable_xvalues = xvalue[usable_indices]
    usable_radii = radii[usable_indices]


    # This is the radius estimated by rotation.
    min_rotation_radius = rotation_radius(usable_vsini, usable_period)
    average_rotation_radius = 1.5 * max_rotation_radius
    rotation_radius_range = 1 * max_rotation_radius

    radius_displacement_fraction = (average_rotation_radius / radii)
    displacement_range = rotation_radius_range / radii

    plot_by_ASPCAP_quality(
        usable_xvalues, radius_displacement_fraction, usable_flags,
        yerr=displacement_range)
    

    plt.plot(list(xlimits), [1, 1], 'k-')
    plt.ylabel("Rotation radius / MS radius")
    plt.xlim(xlimits[0], xlimits[1])

###############################################################################
# Period rate comparisons #
###############################################################################

def compare_Rafa_McQuillan_tidsync_params(
    rafa_logg, rafa_teff, mcq_logg, mcq_teff):
    '''Plot the HR diagram for both Rafa's and McQuillan's loggs.'''
    plt.subplot(1, 2, 1)
    hr.logg_teff_plot(rafa_teff, rafa_logg, 'r.')
    plt.title("Rafa rapid rotators")
    plt.xlim(7000, 3200)
    plt.ylim(5.15, 3.5)
    plt.subplot(1, 2, 2)
    hr.logg_teff_plot(mcq_teff, mcq_logg, 'g.')
    plt.ylabel("")
    plt.title("McQuillan rapid rotators")
    plt.xlim(7000, 3200)
    plt.ylim(5.15, 3.5)

def compare_Rafa_McQuillan_tidsync_params_to_APOGEE(
    rafa_logg, rafa_teff, rafa_apo_logg, rafa_apo_teff, rafa_flags, mcq_logg, 
    mcq_teff, mcq_apo_logg, mcq_apo_teff, mcq_flags):
    '''Compare the Teff and log(g) for APOGEE targets.'''
    rafa_logg_diff = rafa_logg - rafa_apo_logg
    mcq_logg_diff = mcq_logg - mcq_apo_logg
    rafa_teff_diff = rafa_teff - rafa_apo_teff
    mcq_teff_diff = mcq_teff - mcq_apo_teff

    plt.subplot(2, 2, 1)
    catalog.plot_by_ASPCAP_quality(rafa_apo_teff, rafa_logg_diff, rafa_flags)
    plt.ylabel("Log(g) diff (KIC - APOGEE)")
    plt.xlim(7000, 3200)
    plt.ylim(-1.0, 3.0)
    plt.title("Rafa APOGEE observations")
    plt.subplot(2, 2, 2)
    catalog.plot_by_ASPCAP_quality(mcq_apo_teff, mcq_logg_diff, mcq_flags)
    plt.ylabel("Log(g) diff (KIC - APOGEE)")
    plt.xlim(7000, 3200)
    plt.ylim(-1.0, 3.0)
    plt.title("McQuillan APOGEE Observations")
    plt.subplot(2, 2, 3)
    catalog.plot_by_ASPCAP_quality(rafa_apo_teff, rafa_teff_diff, rafa_flags)
    plt.ylabel("Teff diff (K) (KIC - APOGEE)")
    plt.xlim(7000, 3200)
    plt.ylim(-1000, 1000)
    plt.xlabel("Teff (K) (APOGEE)")
    plt.subplot(2, 2, 4)
    catalog.plot_by_ASPCAP_quality(mcq_apo_teff, mcq_teff_diff, mcq_flags)
    plt.ylabel("Teff diff (K) (KIC - APOGEE)")
    plt.xlabel("Teff (K) (APOGEE)")
    plt.xlim(7000, 3200)
    plt.ylim(-1000, 1000)

###############################################################################
# Simulating vsini from v #
###############################################################################

def generate_sini_distribution(npoints=10000):
    '''Generate a distribution of sin(i)s from randomly inclined orbits.

    Note that "randomly inclined" does not mean uniform in inclinations. It
    turns out that the distribution of inclinations goes as sin(i). Oddly
    enough, the distribution of sin(i)s seems to go as tan(i), which diverges
    at edge-on inclinations, which makes no mathematical sense.

    To get around the weird result, I will generate sin(i) distributions by
    hand.'''
    randvar = uniform.rvs(size=npoints)
    incs = np.arccos(2 * randvar - 1)
    sinincs = np.sin(incs)
    return sinincs

def vsini_convolution_table(velbins, binvalues, mcpoints=10000):
    '''Create a table allowing velocities to be convolved on a grid.

    Generate a single table that holds the sin(i) convolution of each velocity
    bin. In particular, the [i,:]th entry of the table contains the
    vsini distribution of objects with true velocity velbins[i].
    '''
    sini_points = generate_sini_distribution(npoints=mcpoints)
    vsini_weights = binvalues[:,np.newaxis] * sini_points
    fullhist = np.zeros(shape=(len(binvalues), len(binvalues)))
    for i in range(len(binvalues)):
        hist, bins = np.histogram(vsini_weights[i,:], bins=velbins)
        fullhist[i,:] = hist / mcpoints
    return fullhist

def error_convolution_table(velbins, fractional_uncert=0.1):
    '''Creates an error convolution table.

    In particular, this creates a square where row i is the a Gaussian profile 
    with center of (velbins[i+1] + velbins[i])/2 and dispersion of 
    fractional_uncert * (velbins[i+1] + velbins[i])/2. 

    The Gaussian will be truncated at cutoff, and all probability less than the
    cutoff value will be distributed as uniform. Right now, cutoff needs to
    coincide with a value in velbins.
    '''
    dv = velbins[1] - velbins[0]
    centers = (velbins[:-1] + velbins[1:])/2
    disp = fractional_uncert * centers

    # I don't want things to be approximate for the uncertainties. So I'll use
    # a more accurate expression for the area between the bins.
    normalized_bins = (velbins - centers[:,np.newaxis]) / (
        np.sqrt(2) * disp[:,np.newaxis])
    erfs = erf(normalized_bins)
    gaussian_table = (erfs[:,1:] - erfs[:,:-1])/2

    return gaussian_table

def vsini_convolution_table_test(velbins, velocities):
    '''Create a table allowing velocities to be convolved on a grid.

    Generate a single table that holds the sin(i) convolution of each velocity
    bin. In particular, the [i,:]th entry of the table contains the
    vsini distribution of objects with true velocity velbins[i].
    '''
    # Since the pdf is actually analytically integrable, we'll make the
    # histogram by simply integrating in each bin.
    # Transform velocity bins to sin(i) bins. 
    scaled_vels = velbins / velocities[:,np.newaxis]
    profiles = np.nan_to_num(
        np.sqrt(1-scaled_vels[:,:-1]**2) - np.sqrt(1-scaled_vels[:,1:]**2))
    # Determine the values in the bins where v/vc > 1
    edge_indices = np.argmin(scaled_vels < 1, axis=1)-1
    # I need data indices to take advantage of the advanced indexing.
    data_indices = np.arange(len(velocities))
    profiles[data_indices, edge_indices] = np.sqrt(
        1-scaled_vels[data_indices, edge_indices]**2)
    # This is the table of histograms. profiles[i,:] will be the histogram of
    # the ith profile. The sum along the 2nd axis should be 1. However, the
    # boundaries will be incorrect without further corrections.
    # The area of each histogram will be sqrt(1-sin^2 i_1) - sqrt(1-sin^2 i_2)
    profiles = np.nan_to_num(
        np.sqrt(1-scaled_vels[:,:-1]**2) - np.sqrt(1-scaled_vels[:,1:]**2))
    # Determine the values in the bins where v/vc > 1
    edge_indices = np.argmin(scaled_vels < 1, axis=1)-1
    # I need data indices to take advantage of the advanced indexing.
    data_indices = np.arange(len(velocities))
    profiles[data_indices, edge_indices] = np.sqrt(
        1-scaled_vels[data_indices, edge_indices]**2)
    np.testing.assert_allclose(np.sum(profiles, axis=1), 1.0)
    return profiles

def compare_sini_distribution(velocities, vsinis, vsini_cutoff=7, nbins=20,
                              frac_uncertainty=0.1):
    '''Compare the inferred sin(i) distribution to a random one.

    Derive a sin(i) distribution from a given velocity and observed vsin(i). In
    order to decrease the amount of noise, objects with velocities less than
    vsini_cutoff will be ignored.'''
    # This will be the same as before. Except non-detections will be removed.
    vel_bins = np.linspace(0, 100, nbins+1, endpoint=True)
    vel_hist, bins = np.histogram(velocities, bins=vel_bins)
    dv = vel_bins[2] - vel_bins[1]
    binvalues = (vel_bins[1:] + vel_bins[:-1])/2

    # First I want the noiseless vsin(i) distribution.
    scaled_vels = vel_bins / velocities[:,np.newaxis]
    profiles = (np.sqrt(1-scaled_vels[:,:-1]**2) -
                np.sqrt(1-scaled_vels[:,1:]**2)).filled(0.0)
    # Determine the values in the bins where v/vc > 1
    edge_indices = np.argmin(scaled_vels < 1, axis=1)-1
    # I need data indices to take advantage of the advanced indexing.
    data_indices = np.arange(len(velocities))
    profiles[data_indices, edge_indices] = np.sqrt(
        1-scaled_vels[data_indices, edge_indices]**2)

    # Now convolve with noise
    noise_array = error_convolution_table(vel_bins, frac_uncertainty)
    convolved_profile = profiles[:, np.newaxis, :] * noise_array
    summed_profile = np.sum(convolved_profile, axis=1)

    # Now add up all of the entries again.
    data_dist = np.sum(convolutions, axis=1)

    # Now get the sini distributions for each object.
    sini_dists = convolutions / velocities[:,np.newaxis,np.newaxis]
    plt.step(sini_dists[0])

    return
    detection_indices = np.where(vsinis > vsini_cutoff)
    sini = vsinis[detection_indices] / velocities[detection_indices]
    art_sini = generate_sini_distribution(30000)

    bins = np.linspace(0, 1.1, nbins+1)
    sini_bins, bins = np.histogram(sini, bins=bins) 
    sini_bins = sini_bins 
    art_sini_bins, bins = np.histogram(art_sini, bins=bins) 
    art_sini_bins = art_sini_bins / len(art_sini) * len(sini)

    plt.step(bins[1:], art_sini_bins, where="pre", lw=2, label="Random")
    plt.step(bins[1:], sini_bins, where="pre", lw=1, label="Observed")
    plt.xlim((1.1, 0.0))
    plt.xlabel("sin(i)")
    plt.ylabel("N(sini)")
    
def compare_vsini_distribution(velocities, vsinis, vsini_percent=0.1,
                               vsini_cutoff=5, nbins=70, maxv=70):
    '''Compare the observed vsin(i) distribution to that inferred from vrot.

    This will reconstruct a vsin(i) distribution using the provided velocity
    distribution. The reconstruction involves convolving with a fractional vsini
    uncertainty, and then convolving with a population of random
    inclinations.'''
    vel_bins = np.linspace(0, maxv*(1+1/nbins), nbins+1, endpoint=False)
    vel_hist, bins = np.histogram(velocities, bins=vel_bins)
    dv = vel_bins[2] - vel_bins[1]
    binvalues = (vel_bins[1:] + vel_bins[:-1])/2

    dispersions = np.reshape(vsini_percent * velocities, (len(velocities), 1))
    # I'll do one data point now, but more will be on the way.
    dist = 1/np.sqrt(2*np.pi*dispersions**2) * np.exp(-(
        binvalues - velocities[:,np.newaxis])**2 / (2 * dispersions**2))*dv
    # Now make a data square that contains sin(i) convolution profiles for all
    # velocity bin values.
    fullhist = vsini_convolution_table_test(vel_bins, binvalues)

    # Now make a cube for all data points
    convolutions = dist[:,:,np.newaxis] * (fullhist)

    # Now add up all of the entries
    data_dist = np.sum(convolutions, axis=1)

    # And now all of the data points
    vsini_dist = np.sum(data_dist, axis=0)

    # Pick out the upper limits.
    upper_index = np.argmin(binvalues<vsini_cutoff)
    num_upper = np.sum(vsini_dist[:upper_index])
    vsini_dist[:upper_index] = 0
    # I want to display the raw numbers in text.
    
    # Done modeling. Now do vsinis.
    vsini_hist, bins = np.histogram(vsinis, bins=vel_bins)
    num_upper_vsinis = np.sum(vsini_hist[:upper_index])
    vsini_hist[:upper_index] = 0

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    # Plot the pdf
    modelcolor = "#000000"
    rotcolor = "#377eb8"
    aspcapcolor = "#e41a1c"
    ax1.step(vel_bins[:-1], vsini_hist, where="post", lw=3, 
             label="ASPCAP vsini", c=aspcapcolor)
    ax1.step(vel_bins[:-1], vsini_dist, where="post", lw=4, label="Model vsini", 
             c=modelcolor)
    ax1.step(vel_bins[:-1], vel_hist, where="post", lw=1, label="Vrot", 
             c=rotcolor)
    ax1.set_xlim(0, vel_bins[-1])
    ax1.set_ylabel("N (vsini)")
    ax1.legend(loc="upper right")
    ax1.text(0.3, 0.8, "{0:d} Total".format(len(velocities)),
             transform=ax1.transAxes, color=modelcolor)
    ax1.text(0.3, 0.7, "{0:d} Nondetections".format(int(num_upper_vsinis)),
             transform=ax1.transAxes, color=aspcapcolor)
    ax1.text(0.3, 0.6, "{0:d} Nondetections".format(int(num_upper)),
             transform=ax1.transAxes, color=modelcolor)


    # Plot the cdf
    # This is just moving around the upper limits for display purposes.
    vsini_dist[0] = num_upper
    vsini_hist[0] = num_upper_vsinis
    dist_cum = np.cumsum(vsini_dist)
    dist_df = dist_cum / dist_cum[-1]
    hist_cum = np.cumsum(vsini_hist)
    hist_df = hist_cum / hist_cum[-1]
    ax2.step(vel_bins[:-1], dist_df, where="post", lw=3, label="Model", 
             c="#000000")
    ax2.step(vel_bins[:-1], hist_df, where="post", lw=2, label="ASPCAP", 
             c="#e41a1c")
    print("The integral error of the distribution is {0:.3f}%.".format(
        (1-np.sum(vsini_dist)/(len(velocities)))*100))
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.set_xlabel("V sin(i) (km/s)")
    ax2.set_ylabel("f (< vsini)")
    ax2.set_ylim(0, 1)
    print(dv)

    # Calculate the significance.
    # I am using a chi-squared test (Numerical Recipes pg 731) since I have
    # what should be a distribution compared to a binned dataset.
    nonzero_indices = np.where(vsini_dist > 0)
    print(nonzero_indices)
    reduced_dist = vsini_dist[nonzero_indices]
    reduced_hist = vsini_hist[nonzero_indices]
    chisq = np.sum((reduced_hist - reduced_dist)**2 / reduced_dist)
    dof = len(reduced_dist) - 1
    # Since most of the bins are empty or close to empty, this is a way to
    # compensate for that (Numerical Recipes pg 734)
    lucy_Ysq = dof + np.sqrt(
        2*dof / (2 * dof + np.sum(1/reduced_dist))) * (chisq - dof)
    prob = gammaincc(0.5*dof, 0.5*lucy_Ysq)
    print("Calculated Chi-squared with {1:d} dof: {0:f}".format(lucy_Ysq, dof))
    print("Probability of data is {0:.4f}".format(prob))

def temperature_diff_vsini_comparison(
    hot_velocities, hot_vsinis, cool_velocities, cool_vsinis,
    vsini_percent=0.1, vsini_cutoff=7, nbins=100, maxv=100):
    '''Compare the vsini/period distributions for hot and cool stars.'''
    vel_bins = np.linspace(0, maxv*(1+1/nbins), nbins+1, endpoint=False)
    hot_vel_hist, bins = np.histogram(hot_velocities, bins=vel_bins)
    cool_vel_hist, bins = np.histogram(cool_velocities, bins=vel_bins)
    dv = vel_bins[2] - vel_bins[1]
    binvalues = (vel_bins[1:] + vel_bins[:-1])/2

    hot_dispersions = np.reshape(vsini_percent * hot_velocities, 
                                 (len(hot_velocities), 1))
    cool_dispersions = np.reshape(vsini_percent * cool_velocities, 
                                 (len(cool_velocities), 1))
    # I'll do one data point now, but more will be on the way.
    hot_dist = (1/np.sqrt(2*np.pi*hot_dispersions**2) * 
                np.exp(-(binvalues - hot_velocities[:,np.newaxis])**2 / 
                       (2 * hot_dispersions**2))*dv)
    cool_dist = (1/np.sqrt(2*np.pi*cool_dispersions**2) * 
                np.exp(-(binvalues - cool_velocities[:,np.newaxis])**2 / 
                       (2 * cool_dispersions**2))*dv)
    # Now make a data square that contains sin(i) convolution profiles for all
    # velocity bin values.
    fullhist = vsini_convolution_table_test(vel_bins, binvalues)

    # Now make a cube for all data points
    hot_convolutions = hot_dist[:,:,np.newaxis] * (fullhist)
    cool_convolutions = cool_dist[:,:,np.newaxis] * (fullhist)

    # Now add up all of the entries
    hot_data_dist = np.sum(hot_convolutions, axis=1)
    cool_data_dist = np.sum(cool_convolutions, axis=1)

    # And now all of the data points
    hot_vsini_dist = np.sum(hot_data_dist, axis=0)
    cool_vsini_dist = np.sum(cool_data_dist, axis=0)

    # Pick out the upper limits.
    upper_index = np.argmin(binvalues<vsini_cutoff)
    hot_num_upper = np.sum(hot_vsini_dist[:upper_index])
    cool_num_upper = np.sum(cool_vsini_dist[:upper_index])
    hot_vsini_dist[:upper_index] = 0
    cool_vsini_dist[:upper_index] = 0
    # I want to display the raw numbers in text.
    
    # Done modeling. Now do vsinis.
    hot_vsini_hist, bins = np.histogram(hot_vsinis, bins=vel_bins)
    cool_vsini_hist, bins = np.histogram(cool_vsinis, bins=vel_bins)
    hot_num_upper_vsinis = np.sum(hot_vsini_hist[:upper_index])
    cool_num_upper_vsinis = np.sum(cool_vsini_hist[:upper_index])
    hot_vsini_hist[:upper_index] = 0
    cool_vsini_hist[:upper_index] = 0

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    # Plot the pdf
    hotcolor = "#377eb8"
    coolcolor = "#e41a1c"
    ax1.step(vel_bins[:-1], hot_vsini_hist, where="post", lw=4, 
             label="Hot ASPCAP vsini", c=hotcolor, ls="-")
    ax1.step(vel_bins[:-1], hot_vsini_dist, where="post", lw=3, 
             label="Hot model vsini", c=hotcolor, ls="--")
    ax1.step(vel_bins[:-1], hot_vel_hist, where="post", lw=1, label="Hot Vrot", 
             c=hotcolor, ls="-")
    ax1.step(vel_bins[:-1], cool_vsini_hist, where="post", lw=4, 
             label="Cool ASPCAP vsini", c=coolcolor, ls="-")
    ax1.step(vel_bins[:-1], cool_vsini_dist, where="post", lw=3, 
             label="Cool model vsini", c=coolcolor, ls="--")
    ax1.step(vel_bins[:-1], cool_vel_hist, where="post", lw=1, label="Cool Vrot", 
             c=coolcolor, ls="-")
    ax1.set_xlim(0, vel_bins[-1])
    ax1.set_ylabel("N (vsini)")
    ax1.legend(loc="upper right")
    ax1.text(0.3, 0.8, "{0:d} Total Hot".format(len(hot_velocities)),
             transform=ax1.transAxes, color=hotcolor)
    ax1.text(0.3, 0.7, "{0:d} Total Cool".format(len(cool_velocities)),
             transform=ax1.transAxes, color=coolcolor)

    # Plot the cdf
    # This is just moving around the upper limits for display purposes.
    hot_vsini_dist[0] = hot_num_upper
    hot_vsini_hist[0] = hot_num_upper_vsinis
    hot_dist_cum = np.cumsum(hot_vsini_dist)
    hot_dist_df = hot_dist_cum / hot_dist_cum[-1]
    hot_hist_cum = np.cumsum(hot_vsini_hist)
    hot_hist_df = hot_hist_cum / hot_hist_cum[-1]
    ax2.step(vel_bins[:-1], hot_dist_df, where="post", lw=3, label="Hot Model", 
             c=hotcolor, ls="--")
    ax2.step(vel_bins[:-1], hot_hist_df, where="post", lw=2, label="Hot ASPCAP", 
             c=hotcolor, ls="-")
    cool_vsini_dist[0] = cool_num_upper
    cool_vsini_hist[0] = cool_num_upper_vsinis
    cool_dist_cum = np.cumsum(cool_vsini_dist)
    cool_dist_df = cool_dist_cum / cool_dist_cum[-1]
    cool_hist_cum = np.cumsum(cool_vsini_hist)
    cool_hist_df = cool_hist_cum / cool_hist_cum[-1]
    ax2.step(vel_bins[:-1], cool_dist_df, where="post", lw=3, label="Cool Model", 
             c=coolcolor, ls="--")
    ax2.step(vel_bins[:-1], cool_hist_df, where="post", lw=2, label="Cool ASPCAP", 
             c=coolcolor, ls="-")
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.set_xlabel("V sin(i) (km/s)")
    ax2.set_ylabel("f (< vsini)")
    ax2.set_ylim(0, 1)

    # Calculate the significance.
    # I am using a chi-squared test (Numerical Recipes pg 731) since I have
    # what should be a distribution compared to a binned dataset.
    hot_nonzero_indices = np.where(hot_vsini_dist > 0)
    cool_nonzero_indices = np.where(cool_vsini_dist > 0)
    hot_reduced_dist = hot_vsini_dist[hot_nonzero_indices]
    hot_reduced_hist = hot_vsini_hist[hot_nonzero_indices]
    hot_chisq = np.sum((hot_reduced_hist - hot_reduced_dist)**2 / hot_reduced_dist)
    hot_dof = len(hot_reduced_dist) - 1
    # Since most of the bins are empty or close to empty, this is a way to
    # compensate for that (Numerical Recipes pg 734)
    hot_lucy_Ysq = hot_dof + np.sqrt(
        2*hot_dof / (2 * hot_dof + np.sum(1/hot_reduced_dist))) * (hot_chisq -
                                                                   hot_dof)
    hot_prob = gammaincc(0.5*hot_dof, 0.5*hot_lucy_Ysq)
    print("Calculated Chi-squared with {1:d} dof: {0:f}".format(hot_lucy_Ysq,
                                                                hot_dof))
    print("Probability of data is {0:.4f}".format(hot_prob))

    cool_reduced_dist = cool_vsini_dist[cool_nonzero_indices]
    cool_reduced_hist = cool_vsini_hist[cool_nonzero_indices]
    cool_chisq = np.sum((cool_reduced_hist - cool_reduced_dist)**2 / cool_reduced_dist)
    cool_dof = len(cool_reduced_dist) - 1
    # Since most of the bins are empty or close to empty, this is a way to
    # compensate for that (Numerical Recipes pg 734)
    cool_lucy_Ysq = cool_dof + np.sqrt(
        2*cool_dof / (2 * cool_dof + np.sum(1/cool_reduced_dist))) * (cool_chisq -
                                                                   cool_dof)
    cool_prob = gammaincc(0.5*cool_dof, 0.5*cool_lucy_Ysq)
    print("Calculated Chi-squared with {1:d} dof: {0:f}".format(cool_lucy_Ysq,
                                                                cool_dof))
    print("Probability of data is {0:.4f}".format(cool_prob))

###############################################################################
# Comparing to Eclipsing Binaries #
###############################################################################

def rotation_modulation_fraction():
    '''Calculate the fraction of binaries showing rotational modulation.

    This function assumes that the binaries are randomly distributed.'''
    fract = np.sqrt(3)/2
    return fract

def num_rotating_binaries_from_EBs(periods):
    '''Calculate an expected number of rotating binaries from EBs.

    This function calculates the number of expected rotating binaries from a
    set of eclipsing binaries. This differs from num_missing_binaries in that
    it incorporates a correction factor for the fact that only stars with high
    enough inclination will have observed rotation modulation.

    Note that this function will not work in the extremely short-period regime
    where all objects that should exchibit rotational modulation should also
    exhibit eclipses..'''
    total_bins = ebs.num_missing_binaries(periods)
    correction = ebs.correct_for_rotation(total_bins)
    return correction

def expected_rotation_fraction_hist(ebperiods, obsperiods, nbins=20,
                                    binrange=(1, 5)):
    '''Compare the EB periods with expected and observed period distribution.

    This function will show the eb distribution, the expected rotation
    modulation distribution based on the EB distribution, and the observed
    rotation modulation distribution for easy comparison.
    '''
    eb_values, eb_binedges = np.histogram(ebperiods, bins=nbins,
                                          range=binrange)
    eb_errors = np.sqrt(eb_values)
    rotvalues = ebs.EB_histogram_to_rotation_histogram(eb_values, eb_binedges)
    rot_errors = eb_errors * rotvalues / eb_values
    bin_starts = eb_binedges[:-1]
    nobs = plt.hist(obsperiods, bins=eb_binedges, label="Observed Rotation")
    plt.bar(bin_starts, rotvalues, label="Predicted Rotation",
            width=bin_starts[1] - bin_starts[0], yerr=rot_errors)
    plt.step(eb_binedges, np.concatenate([eb_values, [0]]), label="Eclipsing Binaries", where="post")
    print("Starts")
    print(bin_starts)
    print("Values")
    print(nobs[0])
    plt.xlabel("Period (day)")
    plt.ylabel("Number")
    plt.xlim(binrange)
    
################################################################################
# Map rotation cuts #
################################################################################

def vsini_cut_to_period_space(rad_from_teff, vsini_cut=10):
    '''Plot the effect of a vini cut in period space.

    Show how a flat vsini cut translates to to period space as a function of
    stellar effective temperature.'''
    tefflims = np.linspace(4250, 5500, 100)

    radii = rad_from_teff(tefflims)
    max_periods = vsini_to_max_period(vsini_cut, radii)

    plt.plot(tefflims, max_periods, 'k-')
    plt.xlabel("Teff (K)")
    hr.invert_x_axis()
    plt.ylabel("Maximum period (day)")
    plt.ylim([0, 7])
    plt.title("Vsini = {0:.1f} cut in period space".format(vsini_cut))

def plot_velocity_limit_in_period_space(rad_from_teff, period_cut=3):
    '''Plot the effect of a period_cut in vsini space'''
    tefflims = np.linspace(4250, 5500, 100)
    
    radii = rad_from_teff(tefflims)
    eq_vels = period_to_velocities(period_cut, radii)

    plt.plot(tefflims, eq_vels, 'k-')
    plt.xlabel("Teff (K)")
    hr.invert_x_axis()
    plt.ylabel("Predicted V_eq (km/s)")
    plt.ylim([0, 20])
    plt.title("P = {0:.1f} cut in velocity space".format(period_cut))
