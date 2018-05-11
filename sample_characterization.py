"""
A set of routines to understand samples.

These functions help understand the distribution of samples with teff, rotation
period, vsini, and other quantities. These functions are grouped according to
different types of quantities they help understand:

Functions to determine the underlying temperature distribution:

number_binned_by_temperature:
    Generate the binned distribution of the sample with temperature

number_histogram:
    Plot the binned distribution of a sample with temperature

cumulative_number_histogram:
    Plot the cumulative binned distribution of a sample with temperature

Functions investigating the rapid rotation distribution with temperature

McQuillan_plot:
    Make a plot like McQuillan with Teff on the x-axis and Period on the
    y-axis.

rapid_fraction_histogram:
    Plot the fraction of rapid rotators with temperature

rapid_fraction_multiple_limits:
    Show how the temperature distribution changes with period.


"""
import subprocess

import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
from astropy.modeling.polynomial import Polynomial1D
import astropy_util as au

import catalog
import hrplots as hr
import biovis_colors as bc
import data_splitting as data
import sed
import rotation_consistency as rot
import observations as obs
import read_catalog as catin

################################################################################
# Generate binned distributions #
################################################################################

def number_binned_by_temperature(
    mcquillan, hightemp=6500, lowtemp=3000, dtemp=50, teffcol="Teff"):
    '''Return array with number as a function of temperature.'''
    # Add dtemp because hist wants the rightmost edge.
    tempbins = np.arange(lowtemp, hightemp+dtemp, dtemp)
    hist, binedges = np.histogram(
        mcquillan[teffcol], bins=tempbins, range=(lowtemp, hightemp))
    return (hist, binedges)

################################################################################
# Plot binned distributions #
################################################################################


def number_histogram(
    mcquillan, hightemp=6500, lowtemp=3000, dtemp=50, teffcol="Teff", label=""):
    '''Plot a histogram of the number of McQuillan objects in temperature bins.

    Uses the matplotlib hist function to make the histogram plot.'''
    tempbins = np.arange(lowtemp, hightemp+dtemp, dtemp)
    plt.hist(mcquillan[teffcol], bins=tempbins, range=(lowtemp, hightemp),
             histtype="step", label=label)
    hr.invert_x_axis()
    plt.xlabel("Teff (K)")
    plt.ylabel("Number in Teff bin")

def cumulative_number_histogram(
    mcquillan, hightemp=6500, lowtemp=3000, dtemp=50, teffcol="Teff", label=""):
    '''Plots a cumulative histogram of number based on temperature.'''
    tempbins = np.arange(lowtemp, hightemp+dtemp, dtemp)
    plt.hist(mcquillan[teffcol], bins=tempbins, range=(lowtemp, hightemp),
             cumulative=True, histtype="step", label=label)
    hr.invert_x_axis()
    plt.xlabel("Teff (K)")
    plt.ylabel("Number cooler than Teff bin")

################################################################################
# Rotation Distribution #
################################################################################

def McQuillan_plot(sample, Teff_colname="Teff", Prot_colname="Prot", color="c",
                   marker=".", label="", ms=2.0):
    '''Creates a plot like in McQuillan.

    Takes the sample in McQuillan and plots the rotation period, given in
    Prot_colname, versus the temperature given in Teff_colname. The rotation
    period is plotted on a log scale.'''
    plt.semilogy(
        sample[Teff_colname], sample[Prot_colname], color=color, 
        marker=marker, label=label, ms=ms, linestyle="")
    hr.invert_x_axis()
    plt.xlabel("Teff (K)")
    plt.ylabel("Prot (day)")

###############################################################################
# Rapid rotation fraction #
###############################################################################

def rapid_fraction_histogram(
    mcquillan, hightemp=6500, lowtemp=3000, dtemp=50, maxper=5, teffcol="Teff",
    periodcol="Prot", label=""):
    '''Plot a histogram of the fraction of rapid rotators in McQuillan sample.
    '''
    totalhist, totbins = number_binned_by_temperature(
        mcquillan, hightemp=hightemp, lowtemp=lowtemp, dtemp=dtemp,
        teffcol=teffcol)
    rapid_mcquillan = catalog.perform_period_cut(
        mcquillan, highperiod=maxper, periodcol=periodcol)
    print("Rapid Rotator Number: " + rapid_mcquillan)
    rapidhist, rapidbins = number_binned_by_temperature(
        rapid_mcquillan, hightemp=hightemp, lowtemp=lowtemp, dtemp=dtemp,
        teffcol=teffcol)
    plt.step(totbins[:-1], rapidhist/totalhist, where="post", label=label)
    plt.xlim(plt.xlim()[::-1])
    plt.xlabel("Teff (K)")
    plt.ylabel("Fraction of Rapid Rotators in Teff bin")
    plt.title("Fraction of rotators with P < {0} day".format(maxper))

def vsini_rapid_fraction_histogram(
        aposet, hightemp=5450, lowtemp=4250, dtemp=100, vsini_det=10,
        teffcol="TEFF", vsini_col="VSINI", label=""):
    '''Plot a histogram of the fraction of vsini rapid rotators.'''
    totalhist, totbins = number_binned_by_temperature(
        aposet, hightemp=hightemp, lowtemp=lowtemp, dtemp=dtemp,
        teffcol=teffcol)
    rapid_vsini = catalog.perform_vsini_cut(
        aposet, lowv=vsini_det, vcol=vsini_col)
    print("Rapid Rotator Number: " + str(len(rapid_vsini)))
    rapidhist, rapidbins = number_binned_by_temperature(
        rapid_vsini, hightemp=hightemp, lowtemp=lowtemp, dtemp=dtemp,
        teffcol=teffcol)
    fracs = rapidhist / totalhist
    fracerrs = np.sqrt(rapidhist) / totalhist
#   plt.step(totbins[:-1], rapidhist/totalhist, where="post", label=label)
    plt.errorbar(totbins[:-1], fracs, yerr=fracerrs, marker="o", ls="",
                 label=label)
    totalfrac = len(rapid_vsini) / len(aposet)
    plt.plot([hightemp, lowtemp], [totalfrac, totalfrac], 'k--')
    plt.xlim(plt.xlim()[::-1])
    plt.xlabel("Teff (K)")
    plt.ylabel("Fraction of Rapid Rotatiors in Teff bin")
    plt.title(
        "Fraction of rotators with Vsini > {0:.1f} km/s".format(vsini_det))

def rapid_fraction_multiple_limits(
    mcquillan, hightemp=6500, lowtemp=3000, dtemp=50, maxper=5, dper=1, 
    teffcol="Teff", periodcol="Prot"):
    '''Plot histograms of rapid rotator fraction for different max periods.

    Bin the McQuillan sample by temperature, and then note the fraction of
    rapid rotators in each temerature bin for different criteria for rapid
    rotation. The maximum period for rapid rotators will start at maxper, and
    decrement by dper until reaching zero.'''
    period_boundaries = np.arange(maxper, 0, -dper)
    for bound in period_boundaries:
        perlabel = "P < {0} day".format(bound)
        rapid_fraction_histogram(
            mcquillan, hightemp=hightemp, lowtemp=lowtemp, dtemp=dtemp, 
            maxper=bound, teffcol=teffcol, periodcol=periodcol, label=perlabel)
    plt.title("Rapid Rotator Fraction up to {0} day".format(maxper))
    plt.legend(loc="upper center")

def rapid_rotation_fraction_detection_limits(vsinis, vsini_limits):
    '''Return the rapid rotator fraction for each of the given limits.
    
    Returns an array that has the fraction of rapid rotators in the vsinis
    array given each detection limit in vsini_limits.'''
    # How it works when numpy is upgraded to >=1.12.
    #fracs = (np.count_nonzero(vsinis[:,np.newaxis] >= vsini_limits, axis=0) / 
    #         len(vsinis))
    fracs = np.zeros(len(vsini_limits))
    for i in range(len(vsini_limits)):
        fracs[i] = np.count_nonzero(vsinis >= vsini_limits[i]) / len(vsinis)
    return fracs

def phot_rapid_rotator_fraction_spec_detection(periods, radii, vsini_limits):
    '''Plot period fraction of rapid rotators given vsini limits.

    This function will basically plot the number of targets expected to have
    vsini > vsini_limits for each of the elements in vsini_limits. Note that
    this function does not include the pi/4 correction for vsini.
    '''
    est_v = rot.period_to_velocities(periods, radii)
    frac = rapid_rotation_fraction_detection_limits(est_v, vsini_limits)
    return frac

def spectroscopic_photometric_rotation_fraction_comparison_plot(
        vsinis, periods, radii, min_limit=5, max_limit=20):
    '''Plot spectroscopic and corresponding photometric rapid rotator fractions.

    Plot the spectroscopic rapid rotator fraction accepting detection limits
    ranging from min_limit to max_limit, inclusive. And then plot the
    photometric rapid rotator fraction corresponding to each detection limit,
    including the inclination correction.'''
    vsini_bins = np.arange(min_limit, max_limit+2)
    # I want to make sure the whole sample is counted within the histogram. So
    # I set the bin edges to be an interval which encompasses the full sample. 
    bin_edges = np.insert(
        vsini_bins, [0, len(vsini_bins)], [min(vsinis), max(vsinis)])
    # I haven't fully planned what behavior would occur if the vsini bin range
    # is larger than the range of vsinis.
    assert vsini_bins[0] > min(vsinis)
    assert vsini_bins[-1] < max(vsinis)

    bindiff = vsini_bins[1] - vsini_bins[0]
    assert np.all(np.diff(vsini_bins) == bindiff)
    ax1 = plt.gca()
    spec_hist, bins = np.histogram(vsinis, bins=bin_edges, density=False)
    spec_cumhist = np.cumsum(spec_hist) / len(vsinis)
    ax1.step(bins[:-1]+bindiff/2, spec_cumhist, where="post", color=bc.black,
             label="Spectroscopic")
        
    upper_rapid_frac = (au.binomial_upper(
        spec_cumhist*len(vsinis), len(vsinis)) - spec_cumhist)
    lower_rapid_frac = (spec_cumhist - au.binomial_lower(
        spec_cumhist*len(vsinis), len(vsinis)))
    rapid_frac_errs = np.array([lower_rapid_frac, upper_rapid_frac])

    vel_periods = rot.period_to_velocities(periods, radii)
    phot_hist, bins = np.histogram(vel_periods, bins=bin_edges, density=False)
    phot_cumhist = np.cumsum(phot_hist) / len(vel_periods)
    ax1.step(bins[:-1]+bindiff/2, phot_cumhist, where="post", color=bc.red,
             label="Photometric")

    upper_phot_rapid_frac = (au.binomial_upper(
        phot_cumhist*len(vel_periods), len(vel_periods)) - phot_cumhist)
    lower_phot_rapid_frac = (phot_cumhist - au.binomial_lower(
        phot_cumhist*len(vel_periods), len(vel_periods))) 
    phot_rapid_frac_errs = np.array([
        lower_phot_rapid_frac, upper_phot_rapid_frac])

    bin_mean = vsini_bins
    ax1.errorbar(bin_mean, spec_cumhist[:-1], yerr=rapid_frac_errs[:,:-1], 
                 color=bc.black, linestyle="None", capsize=4) 
    ax1.errorbar(bin_mean, phot_cumhist[:-1], yerr=phot_rapid_frac_errs[:,:-1], 
                 color=bc.red, capsize=4, linestyle="None")
    ax2 = ax1.twinx()
    ticks = ax1.get_yticks()
    print(ticks)
    fracticks = 1 - ticks
    print(fracticks)
    ax2.set_yticks(ticks)
    ax2.set_yticklabels(fracticks)
    ax1.set_xlim(min_limit-bindiff/2, max_limit+bindiff/2)
    ax2.set_xlim(min_limit-bindiff/2, max_limit+bindiff/2)
    ax1.set_ylabel(r"$N (< v \sin i) / N$")
    ax1.set_xlabel(r"$v \sin i$")
    ax2.set_ylabel("Rapid Rotator Fraction")
    ax2.set_ylim(0.7, 1.0)
    plt.sca(ax1)

def plot_rapid_rotation_detection_limits(
        vsinis, min_limit=5, max_limit=20, color=bc.black, label="", ls="-"):
    '''Plot rapid rotator fraction for vsinis as a function of detection limits.

    For each of the vsini measurements in vsinis, plot the rapid rotator
    fraction spanning vsinis from min_limit to max_limit. For the purposes of
    this plot, this will span integer values between min_limit and
    max_limit.
    
    Offsets can be provided if many of these plots are shown at the same time.'''
    vsini_bins = np.arange(min_limit, max_limit+2)
    # I want to make sure the whole sample is counted within the histogram. So
    # I set the bin edges to be an interval which encompasses the full sample. 
    bin_edges = np.insert(
        vsini_bins, [0, len(vsini_bins)], [min(vsinis), max(vsinis)])
    # I haven't fully planned what behavior would occur if the vsini bin range
    # is larger than the range of vsinis.
    assert vsini_bins[0] > min(vsinis)
    assert vsini_bins[-1] < max(vsinis)

    bindiff = vsini_bins[1] - vsini_bins[0]
    assert np.all(np.diff(vsini_bins) == bindiff)
    ax1 = plt.gca()
    spec_hist, bins = np.histogram(vsinis, bins=bin_edges, density=False)
    spec_cumhist = np.cumsum(spec_hist) / len(vsinis)
    ax1.step(bins[:-1]+bindiff/2, spec_cumhist, where="post", color=color,
             linestyle=ls, label=label)
        
    upper_rapid_frac = (au.binomial_upper(
        spec_cumhist*len(vsinis), len(vsinis)) - spec_cumhist)
    lower_rapid_frac = (spec_cumhist - au.binomial_lower(
        spec_cumhist*len(vsinis), len(vsinis)))
    rapid_frac_errs = np.array([lower_rapid_frac, upper_rapid_frac])


    bin_mean = vsini_bins
    ax1.errorbar(bin_mean, spec_cumhist[:-1], yerr=rapid_frac_errs[:,:-1], 
                 color=color, linestyle="None", capsize=4) 

###############################################################################
# Comparing KIC values #
###############################################################################
def compare_inferred_flicker_loggs(
    flicker_logg, kic_logg, dsep_logg, apogee_logg, xvalue):
    '''Compare the flicker, Huber, and DSEP-inferred loggs.

    A plot will compare the three different values of logg. They will be
    plotted with respect to the given xvalue. The flicker logg will be plotted 
    with a blue diamond, kic loggs with a black diamond, and dsep loggs with a 
    red diamond.
    '''
    apogee_giants = np.where(np.logical_and(
        apogee_logg > 0, apogee_logg < 3.5))
    plt.plot(xvalue, flicker_logg, 'bo', ms=6, label="Flicker")
    plt.plot(xvalue, kic_logg, 'rd', label="Huber", ms=6)
    plt.plot(xvalue, dsep_logg, 'kx', label="DSEP", ms=4)
    plt.plot(xvalue[apogee_giants], flicker_logg[apogee_giants], 'sm',
             label="APOGEE GIANT", ms=6)
    for i in range(len(xvalue)):
        fkdiff = abs(flicker_logg[i] - kic_logg[i])
        kddiff = abs(kic_logg[i] - dsep_logg[i])
        fddiff = abs(flicker_logg[i] - dsep_logg[i])
        maxdiff = max([fkdiff, kddiff, fddiff])
        print(maxdiff)
        if fkdiff < 1:
            lc='k'
        else:
            lc='r'
        plt.plot([xvalue[i]]*2, [flicker_logg[i], kic_logg[i]], ls=':', c=lc)
        plt.plot([xvalue[i]]*2, [kic_logg[i], dsep_logg[i]], ls=':', c=lc)
    plt.ylabel("Log(g)")
    hr.invert_y_axis()

def compare_Huber_APOGEE_loggs(
    apogee_logg, huber_logg, huber_logg_low, huber_logg_high):
    '''Compare APOGEE log(g) to Huber log(g) with uncertainties.

    Will basically make a One-to-one plot with the asymmetric Huber
    uncertainties taken into account, to see if objects that scatter into the
    dwarf regime are uncertain subgiants.'''
    plt.errorbar(
        apogee_logg, huber_logg, yerr=[-huber_logg_low, huber_logg_high],
        fmt="go")
    plt.plot([2, 5], [2, 5], 'k-')
    plt.plot([2, 5], [4.2, 4.2], 'b--')
    plt.plot([4.2, 4.2], [2, 5], 'b--')
    plt.xlabel("Uncalibrated APOGEE log(g)")
    plt.ylabel("Huber log(g)")

###############################################################################
# Exploring subsamples #
###############################################################################

def check_Jen_subsample(jensplitter=None):
    '''Looks around at the subsample that is Jen's.'''
    # First, get Jen's subsample
    if not jensplitter:
        jen_data = catalog.build_cool_dwarf_sample()
        # The KIC Teff and logg automatically are masked arrays.
        # In order to meet the specifications of the splitter, I want them to not
        # be.
        assert(np.all(~jen_data["KIC Teff"].mask))
        assert(np.all(~jen_data["KIC logg"].mask))
        jen_data["KIC Teff"] = jen_data["KIC Teff"].filled()
        jen_data["KIC logg"] = jen_data["KIC logg"].filled()
        # Make an APOGEESplitter from it.
        jensplitter = apo.APOGEESplitter(jen_data)

    # Split off the McQuillan targets.
    jensplitter.split_McQuillan_periods(kiccol="kepid")

    # Split the giants from the dwarfs.
    # There are none!
    jensplitter.split_Ciardi_logg(
        "DC logg", "DC Teff", logg_crit="DC Ciardi logg", 
        splitnames=("DC Ciardi Giant", "DC Ciardi Dwarf"))
    num_DC_misclassified = jensplitter.subsample_len(["DC Ciardi Giant"])
    print("{0:d} objects were classified as giants by Dressing and "
          "Charbonneau".format(num_DC_misclassified))

    # Now split based on J-H Color.
    jensplitter.split_Ciardi_Color()
    print("Number of objects not analyzed by McQuillan that have J-H > 0.75: "
          "{0}".format(jensplitter.subsample_len(
              ["Unknown Mcq", "Color Giant"])))
    print("Number of objects analyzed by McQuillan that have J-H <= 0.75: "
          "{0}".format(jensplitter.subsample_len(
              ["~Unknown Mcq", "Color Giant"])))
    print("Remaining missing McQuillan: {0}".format(
        jensplitter.subsample_len(["Unknown Mcq", "Color Dwarf"])))

    # Remove the EBs.
    jensplitter.split_eclipsing_binaries()
    print("Number of objects not analyzed by Mcquillan that are EBs: "
          "{0}.".format(jensplitter.subsample_len(
              ["Unknown Mcq", "Color Dwarf", "Kepler EB"])))
    print("Number of objects analyzed by Mcquillan that are EBs: "
          "{0}.".format(jensplitter.subsample_len(
              ["~Unknown Mcq", "Color Dwarf", "Kepler EB"])))
    print("Remaining missing McQuillan: {0}".format(
        jensplitter.subsample_len(["Unknown Mcq", "Color Dwarf", "Not EB"])))

    # Remove the KOIs.
    jensplitter.split_KOIs()
    print("Number of objects not analyzed by Mcquillan that are KOIs: "
          "{0}.".format(jensplitter.subsample_len(
              ["Unknown Mcq", "Color Dwarf", "Not EB", "KOI"])))
    print("Number of objects analyzed by Mcquillan that are KOI: "
          "{0}.".format(jensplitter.subsample_len(
              ["~Unknown Mcq", "Color Dwarf", "Not EB", "KOI"])))
    print("Remaining missing McQuillan: {0}".format(
        jensplitter.subsample_len([
            "Unknown Mcq", "Color Dwarf", "Not EB", "Not KOI"])))

    # Remove targets without adequate observations.
    jensplitter.split_sufficient_quarter_obs()
    print("Number of objects not analyzed by Mcquillan with fewer than 8 "
          "quarters observed: {0}.".format(jensplitter.subsample_len(
              ["Unknown Mcq", "Color Dwarf", "Not EB", "Not KOI", 
               "Low Quarter Fraction"])))
    print("Number of objects analyzed by Mcquillan with fewer than 8 "
          "quarters observed: {0}.".format(jensplitter.subsample_len(
              ["~Unknown Mcq", "Color Dwarf", "Not EB", "Not KOI", 
               "Low Quarter Fraction"])))
    print("Remaining missing McQuillan: {0}".format(
        jensplitter.subsample_len([
            "Unknown Mcq", "Color Dwarf", "Not EB", "Not KOI", 
            "OK Quarter Fraction"])))




    withmcq = jensplitter.subsample(["~Unknown Mcq"])
    colorgiants = jensplitter.subsample(["Unknown Mcq", "Color Giant"])
    ebs = jensplitter.subsample(["Unknown Mcq", "Color Dwarf", "Kepler EB"])
    kois = jensplitter.subsample([
        "Unknown Mcq", "Color Dwarf", "Not EB", "KOI"])
    fewquarters = jensplitter.subsample([
        "Unknown Mcq", "Color Dwarf", "Not EB", "Not KOI", 
        "Low Quarter Fraction"])
    unknownmcq = jensplitter.subsample([
        "Unknown Mcq", "Color Dwarf", "Not EB", "Not KOI", 
        "OK Quarter Fraction"])
    plot_teffcol="KIC Teff"
    plot_loggcol="KIC logg"
    plt.figure()
    hr.logg_teff_plot(
        withmcq[plot_teffcol], withmcq[plot_loggcol], ls="", marker=".", 
        color=(0, 0, 0), label="Analyzed")
    hr.logg_teff_plot(
        colorgiants[plot_teffcol], colorgiants[plot_loggcol], ls="", 
        marker=".", color=(36/255, 255/255, 36/255), label="Color Giant")
    hr.logg_teff_plot(
        ebs[plot_teffcol], ebs[plot_loggcol], ls="", marker=".", 
        color=(0, 109/255, 219/255), label="EB")
    hr.logg_teff_plot(
        kois[plot_teffcol], kois[plot_loggcol], ls="", marker=".",
        color=(109/255, 182/255, 255/255), label="KOI")
    hr.logg_teff_plot(
        fewquarters[plot_teffcol], fewquarters[plot_loggcol], ls="",
        marker=".", color=(146/255, 0, 0), 
        label="Too few quarters")
    hr.logg_teff_plot(
        unknownmcq[plot_teffcol], unknownmcq[plot_loggcol], ls="", marker=".",
        color=(255/255, 109/255, 182/255), label="Unaccounted")
    plt.xlim([6400, 3200])
    plt.ylim([5.1, 1.4])
    plt.xlabel("KIC Teff")
    plt.ylabel("KIC logg")
    plt.legend(loc="upper left")

    return jensplitter

def plot_APOGEE_quality(splitter, teffcol, loggcol):
    '''Make a plot showing the locations of APOGEE qualities.

    The splitter should have the APOGEE quality splits already done.'''

    good_objs = splitter.subsample(["Good"])
    warn_objs = splitter.subsample(["Warn"])
    vsini_objs = splitter.subsample(["vsini"])
    bad_objs = splitter.subsample(["Bad"])

    plot_teffcol = "teff"
    plot_loggcol = "logg"
    hr.logg_teff_plot(good_objs[plot_teffcol], good_objs[plot_loggcol], 'g.',
                      label="Good")
    hr.logg_teff_plot(warn_objs[plot_teffcol], warn_objs[plot_loggcol], ls="",
                      marker=".", color=(219/255, 209/255, 0), label="Warn")
    hr.logg_teff_plot(vsini_objs[plot_teffcol], vsini_objs[plot_loggcol], 'y.',
                      label="vsini")
    hr.logg_teff_plot(bad_objs[plot_teffcol], bad_objs[plot_loggcol], 'r.',
                      label="bad")

    plt.legend(loc="upper left")

###############################################################################
# Radius determination #
###############################################################################

def fit_teff_radius_relation(teffs, radii):
    '''Fit a relationship between the Teff and Radius of a sample.
    
    The form of the relationship as currently used is linear.'''
    init_model = models.Polynomial1D(3)
    fit_model = fitting.LevMarLSQFitter()
    new_model = fit_model(init_model, teffs, radii)

    return new_model

def huber_dwarf_radius_relation():
    '''Return a relationship between Teff and Radius calibrated to Huber.

    This is essentially a hard-coded polynomial that was already fit to the
    region of parameters space of interest for APOGEE.'''
    # In order to generate the polynomial down below, a procedure similar to
    # this should be followed:
    # 
    # apogee = data.APOGEESplitter()
    # data.initialize_general_APOGEE()
    # cool = data.general_to_cool_sample(apogee)
    # data.initialize_cool_KICs(cool)
    # cool.split_teff("teff", 5450, ["Unevolved", "Age evolved"],
    #                 teff_crit="Age evolution split")
    # cool_dwarfs = cool.subsample(["Huber dwarfs", "Unevolved"])
    # teff_rad = fit_teff_radius_relation(cool_dwarfs["teff"],
    #                                     cool_dwarfs["radius"])
    teff_rad = Polynomial1D(
        3, c0=-14.308397276533105, c1=0.009841383596450669, 
        c2=-2.1820621757986364e-06, c3=1.6294313395008525e-10)
    return teff_rad

def plot_teff_radius_relation(dwarf_teff, dwarf_radius, subgiant_teff,
                              subgiant_radius, dwarf_metallicity):
    relation = fit_teff_radius_relation(dwarf_teff, dwarf_radius)
    fig, ax = plt.subplots()
    cax = ax.scatter(
        dwarf_teff, dwarf_radius, c=dwarf_metallicity,
        cmap=plt.cm.get_cmap("winter"), label="Dwarfs")
    hr.radius_teff_plot(subgiant_teff, subgiant_radius, 'r.',
                        label="Subgiants", axis=ax)

    teff_boundaries = np.array(np.sort(dwarf_teff))
    radius_boundaries = relation(teff_boundaries)
    hr.radius_teff_plot(teff_boundaries, radius_boundaries, style='k-', lw=3,
                        label="Fit", axis=ax) 

    cbar = fig.colorbar(cax)
    cbar.ax.set_xlabel("[Fe/H]")
    print(np.std(dwarf_radius - relation(dwarf_teff)))
    plt.xlabel("Huber Teff")
    plt.ylabel("Huber Radius")
    plt.legend(loc="upper right")

def generate_radius_column(
    apotable, teff_rad_conv, teffcol="TEFF", radcol="APOGEE radius"):
    '''Add a radius column to apotable.

    The column to convert teff to radius should be given as Teffcol. The radius
    will be stored in the radcol column.'''
    radius_column = teff_rad_conv(apotable[teffcol])
    try:
        colmask = radius_column.mask
    except AttributeError:
        pass
    else:
        if isinstance(colmask, np.bool_):
            colmask = np.ones(len(radius_column)) * colmask
            radius_column.mask = colmask

    apotable[radcol] = radius_column

def generate_DSEP_radius_column(
    apotable, teffcol="TEFF", fehcol="FE_H", age=3, radcol="APOGEE radius"):
    '''Add a radius column to apotable.

    The column to convert teff to radius should be given as Teffcol. The radius
    will be stored in the radcol column.'''
    radiusarr = np.zeros(len(apotable))
    # A dictionary referencing DSEP models according to metallicity.
    DSEP_models = {}
    rounded_metallicities = np.round(apotable[fehcol]*2, 1)/2
    # What to do about -9999 or masked arrays
    for i in range(len(rounded_metallicities)):
        try:
            dsep_interper = DSEP_models[rounded_metallicities[i]]
        except KeyError:
            dsep_interper = sed.DSEPInterpolator(age, rounded_metallicities[i])

        teffpoint = apotable[teffcol][i]
        try:
            radiusarr[i] = dsep_interper.teff_to_radius_interpolation_sb(teffpoint)
        # This will be called if the DSEP interpolator has one of the values
        # being out of bounds.
        except subprocess.CalledProcessError:
            radiusarr[i] = np.nan
        else:
            assert teffpoint > 0

    apotable[radcol] = radiusarr

def generate_DSEP_radius_column_with_errors(
        apotable, teffcol="TEFF", fehcol="FE_H", ageval=3, oldage=10,
        youngage=1, radcol="DSEP radius", topraderrcol="DSEP radius upper",
        bottomraderrcol="DSEP radius lower"):
    '''Add radius and error columns to apotable.

    The columns that will be added are a column for the radius, the upper limit
    and the lower limit. The ages corresponding to the representative, old, and
    young limits should also be specified.'''
    generate_DSEP_radius_column(
        apotable, teffcol=teffcol, fehcol=fehcol, age=ageval, radcol=radcol)
    generate_DSEP_radius_column(
        apotable, teffcol=teffcol, fehcol=fehcol, age=youngage,
        radcol=bottomraderrcol)
    apotable[bottomraderrcol] = apotable[radcol] - apotable[bottomraderrcol]
    generate_DSEP_radius_column(
        apotable, teffcol=teffcol, fehcol=fehcol, age=oldage,
        radcol=topraderrcol)
    apotable[topraderrcol] = apotable[radcol] - apotable[topraderrcol]

###############################################################################
# Asteroseismic log(g) determination #
###############################################################################

def asteroseismic_logg_check(apokasc_splitter):
    '''Use the APOKASCSplitter functions to plot the log(g) uncertainty.

    This function will plot the difference between the ASPCAP and asteroseismic
    log(g) for the dwarf sample.'''
    formats = {"LOGG_WARN": {"marker": "s"}, "COLORTE_WARN": {"markersize": 4},
               "VSINI_WARN": {"marker": "o"}, "TEFF_WARN": {"color": "r"},
               "VMICRO_WARN": {"alpha": 1}}

    good_targets = apokasc_splitter.subsample(["Asteroseismic Dwarfs", "Good"])
    warn_targets = apokasc_splitter.subsample(["Asteroseismic Dwarfs", "Warn"])

    warnindices = {
        k: catalog.search_in_ASPCAPFLAGS(warn_targets["ASPCAPFLAGS"], k) for k 
        in formats.keys()}
    for flags in au.powerset(formats.keys()):
        if not flags:
            good_loggdiff = good_targets["LOGG_FIT"] - good_targets["LOGG_DW"]
            plt.plot(good_targets["TEFF_COR"], good_loggdiff, 'k.', 
                     label="Good APOKASC Dwarf")
        else:
            # Since I want the single-entries for labeling, even though most of
            # them don't occur by themselves, I will make a flag that should
            # trigger if the criterion is specified, but there are elements in
            # the array that fit it.
            at_least_teff_or_colorte = False
            not_both_logg_and_vsini = False
            # To denote the TEFF_WARN flag, set the color to 0.0 of the
            # colormap. To denote the COLORTE_WARN flag, set the color to 1.0
            # of the colormap. For both, set the color to 0.5.
            cmap = plt.cm.get_cmap("OrRd")
            if "TEFF_WARN" in flags and "COLORTE_WARN" in flags:
                markerfacecolor = cmap(0.6)
            elif "TEFF_WARN" in flags:
                markerfacecolor = cmap(0.2)
            elif "COLORTE_WARN" in flags:
                markerfacecolor = cmap(1.0)
            else:
                at_least_teff_or_colorte = True

            if "VMICRO_WARN" in flags:
                markeredgewidth=2
            else:
                markeredgewidth=1

            if "LOGG_WARN" in flags and "VSINI_WARN" in flags:
                not_both_logg_and_vsini = True
            elif "LOGG_WARN" in flags:
                marker="s"
            elif "VSINI_WARN" in flags:
                marker="^"
            else:
                marker="o"

            # Now set the label in the correct place
            if len(flags) == 1:
                label = flags[0]
            else:
                label=""


            indexlist = np.ones(len(warn_targets))
            for k, v in warnindices.items():
                if k in flags:
                    indexlist = np.logical_and(indexlist, v)
                else:
                    indexlist = np.logical_and(indexlist, np.logical_not(v))

            smallwarn = warn_targets[indexlist]
            warn_loggdiff = smallwarn["LOGG_FIT"] - smallwarn["LOGG_DW"]

            if len(smallwarn) > 0:
                if at_least_teff_or_colorte:
                    raise ValueError(
                        "Need at least TEFF_WARN or COLORTE_WARN.")
                if not_both_logg_and_vsini:
                    raise ValueError(
                        "Can't have both LOGG_WARN and VSINI_WARN.")

            plt.plot(smallwarn["TEFF_COR"], warn_loggdiff, 
                     markerfacecolor=markerfacecolor, marker=marker, 
                     label=label, ls="none", markeredgewidth=markeredgewidth, 
                     mec="k", markersize=9)

    hr.invert_x_axis()
    plt.xlabel("APOGEE Teff (K)")
    plt.ylabel("ASPCAP - Asteroseismic log(g) Difference")
    plt.legend()

def asteroseismic_hr_check(apokascsplitter):
    '''Compare HR diagram location for APOGEE and asteroseismic logg.

    Plot the asteroseismic and spectroscopic log(g) on the same scale and see
    if the asteroseismic targets follow the track that they ought in
    spectroscopic space as well as in asteroseismic space.
    
    This requires the splitter to have quality determinations, as well as
    spectroscopic determinations of dwarfs.'''
    full_dwarfs = apokascsplitter.subsample(["~Bad", "~Spectroscopic Giants"])
    astero_dwarfs = apokascsplitter.subsample(["Asteroseismic Dwarfs", "~Bad"])

    hr.logg_teff_plot(
        full_dwarfs["TEFF_COR"], full_dwarfs["LOGG_FIT"], color=bc.black, 
        linestyle="", marker=".", label="APOGEE")
    hr.logg_teff_plot(
        astero_dwarfs["TEFF_COR"], astero_dwarfs["LOGG_FIT"], color=bc.violet, 
        linestyle="", marker="o", label="Spectroscopic log(g)")
    hr.logg_teff_plot(
        astero_dwarfs["TEFF_COR"], astero_dwarfs["LOGG_DW"], color=bc.green,
        linestyle="", marker="*", ms=9, label="Asteroseismic log(g)")

    plt.xlabel("APOGEE Teff (K)")
    plt.ylabel("Log(g)")
    plt.legend(loc="lower right")

###############################################################################
# Check evolutionary state classifications #
###############################################################################

def check_evstate_classifications(teff, logg, bollum, kmag):
    '''Show dwarf/subgiant classification agreements for three methods.

    Compare the regions where the APOGEE, bolometric luminosity, and Absolute K
    magnitude methods agree and disagree.'''

    logg_points = [(4640, 3.72), (5690, 4.43)]
    bollum_points = [(4970, 0.25), (5750, 0.24)]
    kmag_points = [(5015, 2.34), (5625, 2.46)]

    logg_slope = ((logg_points[0][1] - logg_points[1][1]) / (
        logg_points[0][0] - logg_points[1][0]))
    bollum_slope = ((bollum_points[0][1] - bollum_points[1][1]) / (
        bollum_points[0][0] - bollum_points[1][0]))
    kmag_slope = ((kmag_points[0][1] - kmag_points[1][1]) / (
        kmag_points[0][0] - kmag_points[1][0]))

    logg_dwarf_indices = logg >= logg_points[0][1] + logg_slope * (
        teff - logg_points[0][0])
    logg_subgiant_indices = logg < logg_points[0][1] + logg_slope * (
        teff - logg_points[0][0])
    bollum_dwarf_indices = bollum <= bollum_points[0][1] + bollum_slope * (
        teff - bollum_points[0][0])
    bollum_subgiant_indices = bollum > bollum_points[0][1] + bollum_slope * (
        teff - bollum_points[0][0])
    kmag_dwarf_indices = kmag >= kmag_points[0][1] + kmag_slope * (
        teff - kmag_points[0][0])
    kmag_subgiant_indices = kmag < kmag_points[0][1] + kmag_slope * (
        teff - kmag_points[0][0])

    # Where all 3 match, let it be a green dot.
    # Where logg is off, let it be an orange star
    # Where bollum is off, let it be a purple square
    # Where kmag is off, let it be a red diamond.

    matching_targets = np.logical_or(
        au.multi_logical_and(logg_dwarf_indices, bollum_dwarf_indices, 
                             kmag_dwarf_indices),
        au.multi_logical_and(logg_subgiant_indices, bollum_subgiant_indices, 
                             kmag_subgiant_indices))
    logg_off = np.logical_or(
        au.multi_logical_and(
            np.logical_not(logg_dwarf_indices), bollum_dwarf_indices, 
            kmag_dwarf_indices),
        au.multi_logical_and(
            np.logical_not(logg_subgiant_indices), bollum_subgiant_indices, 
            kmag_subgiant_indices))
    bollum_off = np.logical_or(
        au.multi_logical_and(
            logg_dwarf_indices, np.logical_not(bollum_dwarf_indices), 
            kmag_dwarf_indices),
        au.multi_logical_and(
            logg_subgiant_indices, np.logical_not(bollum_subgiant_indices), 
            kmag_subgiant_indices))
    kmag_off = np.logical_or(
        au.multi_logical_and(
            logg_dwarf_indices, bollum_dwarf_indices, 
            np.logical_not(kmag_dwarf_indices)),
        au.multi_logical_and(
            logg_subgiant_indices, bollum_subgiant_indices, 
            np.logical_not(kmag_subgiant_indices)))

    # All of the targets in logg/teff space.
    plt.figure()
    ax=plt.gca()
    hr.logg_teff_plot(
        teff[matching_targets], logg[matching_targets], ls="", marker=".",
        color=bc.green, axis=ax, label="Match")
    hr.logg_teff_plot(
        teff[logg_off], logg[logg_off], ls="", marker="*",
        color=bc.orange, axis=ax, label="Log(g) disagrees")
    hr.logg_teff_plot(
        teff[bollum_off], logg[bollum_off], ls="", marker="s",
        color=bc.purple, axis=ax, label="Lbol disagrees" )
    hr.logg_teff_plot(
        teff[kmag_off], logg[kmag_off], ls="", marker="d",
        color=bc.red, axis=ax, label="M_K disagrees" )
    ax.plot([logg_points[0][0], logg_points[1][0]], 
             [logg_points[0][1], logg_points[1][1]], 'k-')
    ax.set_xlabel("APOGEE Teff (K)")
    ax.set_xlim(6500, 3500)
    hr.invert_x_axis(ax)
    hr.invert_y_axis(ax)
    plt.legend(loc="upper left")

    # All of the targets in bollum/teff space.
    plt.figure()
    ax=plt.gca()
    hr.logL_teff_plot(
        teff[matching_targets], bollum[matching_targets], ls="", marker=".",
        color=bc.green, axis=ax, label="Match")
    hr.logL_teff_plot(
        teff[logg_off], bollum[logg_off], ls="", marker="*",
        color=bc.orange, axis=ax, label="Log(g) disagrees" )
    hr.logL_teff_plot(
        teff[bollum_off], bollum[bollum_off], ls="", marker="s",
        color=bc.purple, axis=ax, label="Lbol disagrees" )
    hr.logL_teff_plot(
        teff[kmag_off], bollum[kmag_off], ls="", marker="d",
        color=bc.red, axis=ax, label="M_K disagrees" )
    ax.plot([bollum_points[0][0], bollum_points[1][0]], 
             [bollum_points[0][1], bollum_points[1][1]], 'k-')
    ax.set_xlabel("APOGEE Teff (K)")
    ax.set_xlim(6500, 3500)
    hr.invert_x_axis(ax)
    plt.legend(loc="upper left")



    # All of the targets in kmag/teff space.
    plt.figure()
    ax=plt.gca()
    hr.absmag_teff_plot(
        teff[matching_targets], kmag[matching_targets], ls="", marker=".",
        color=bc.green, axis=ax, label="Match" )
    hr.absmag_teff_plot(
        teff[logg_off], kmag[logg_off], ls="", marker="*",
        color=bc.orange, axis=ax, label="Log(g) disagrees" )
    hr.absmag_teff_plot(
        teff[bollum_off], kmag[bollum_off], ls="", marker="s",
        color=bc.purple, axis=ax, label="Lbol disagrees" )
    hr.absmag_teff_plot(
        teff[kmag_off], kmag[kmag_off], ls="", marker="d",
        color=bc.red, axis=ax, label="M_K disagrees" )
    ax.plot([kmag_points[0][0], kmag_points[1][0]], 
             [kmag_points[0][1], kmag_points[1][1]], 'k-')
    ax.set_xlabel("APOGEE Teff (K)")
    ax.set_ylabel("M_K")
    ax.set_xlim(6500, 3500)
    hr.invert_x_axis(ax)
    hr.invert_y_axis(ax)
    plt.legend(loc="upper left")
    print(kmag[kmag_off])

###############################################################################
# Photometric binarity #
###############################################################################

def calc_DSEP_model_mags(teffs, fehs, mag, age=3):
    '''A predicted absolute magnitude given teff.

    For a variety of temperatures with associated metallicities, calculate the
    absolute magnitude in a given band for each of those temperatures. The age
    of the distribution can also be specified.'''
    magarr = np.zeros(len(teffs))
    # A dictionary referencing DSEP models according to metallicity.
    DSEP_models = {}
    rounded_metallicities = np.round(fehs*2, 1)/2
    # What to do about -9999 or masked arrays
    for i in range(len(rounded_metallicities)):
        try:
            dsep_interper = DSEP_models[rounded_metallicities[i]]
        except KeyError:
            dsep_interper = sed.DSEPInterpolator(
                age, rounded_metallicities[i], highT=7000)

        teffpoint = teffs[i]
        try:
            magarr[i] = dsep_interper.teff_to_abs_mag(teffpoint, mag)
            assert fehs[i] - rounded_metallicities[i] < 0.05
        # This will be called if the DSEP interpolator has one of the values
        # being out of bounds.
        except subprocess.CalledProcessError:
            magarr[i] = np.nan
        else:
            assert teffpoint > 0

    return magarr

def calc_photometric_excess(teffs, fehs, mag, photvals, age=3):
    '''Calculate the photometric excess above a given isochrone.

    Calculate the magnitude difference between photvals and an isochrone
    solution for the given teff, [Fe/H] and age for the given mag.'''
    DSEPmags = calc_DSEP_model_mags(teffs, fehs, mag, age=age)
    magdiff = photvals - DSEPmags

    return magdiff

def plot_photometric_binary_excess(teffs, fehs, mag, photvals, age=3):
    '''Plot the photometric excess for a sample of main-sequence targets.

    Plot the magnitude difference between photvals and an isochrone solution
    for the given teff, [Fe/H], and age for the given absolute magnitude.'''
    magdiff = calc_photometric_excess(teffs, fehs, mag, photvals, age=age)
    phot_binary_div_points = [(5427, -0.60), (3946, -0.14)]
    dividing_line = (phot_binary_div_points[0][1] + 
        (phot_binary_div_points[0][1] - phot_binary_div_points[1][1]) /
        (phot_binary_div_points[0][0] - phot_binary_div_points[1][0]) *
        (teffs - phot_binary_div_points[0][0]))
    phot_binary_indices = magdiff < dividing_line

    hr.absmag_teff_plot(
        teffs[~phot_binary_indices], magdiff[~phot_binary_indices], marker=".", 
        color=bc.black, ls="", label="Single Stellar locus")
    hr.absmag_teff_plot(
        teffs[phot_binary_indices], magdiff[phot_binary_indices], marker=".", 
        color=bc.red, ls="", label="Photometric Binaries")
    hr.absmag_teff_plot(teffs, dividing_line, ls="-", color=bc.black, marker="")
    plt.xlim(5500, 3500)
    plt.ylim(0.3, -2.2)
    plt.xlabel("Teff (K)")
    plt.ylabel("M_{0}-DSEP M_{0}".format(mag))
    plt.legend(loc="upper right")

def mcquillan_rapid_rotator_binarity():
    '''Plot rapid rotators in a CMD.'''
    mcq = catin.mcquillan_with_stelparms()
    mcq["M_K"] = mcq["kmag"] - 5 * np.log10(mcq["dis"]/10)

    rapid_rotators = mcq[np.logical_and(mcq["Prot"] < 5, mcq["Prot"] > 1)]
    very_rapid = mcq[mcq["Prot"] < 1]

    mdm_rv_var = [11819949, 12736892, 3248885]
    mdm_targets = obs.select_observing_targets(30)
    mdm_mcq = au.extract_subtable_from_column(
        mcq, "kepid", mdm_targets["kepid"])
    mdm_rvvar = au.extract_subtable_from_column(mdm_mcq, "kepid", mdm_rv_var)

    # Include the APOGEE variable-nonvariable targets.
    mcq_observing = catalog.select_tidally_synchronized_binaries(
        mcq, pcut=5, lowperiod=1, lowtemp=4850, hightemp=5600, logg=3.5,
        teffcol="teff", pcol="Prot", loggcol="logg")

    mcq_observing = catalog.filter_pulsators(mcq_observing, KICcol="KIC")

    apogee = catin.mcquillan_dr14_overlap()
    mcq_observing = catalog.join_by_2MASS_key(
        mcq_observing, apogee, "tm_designation", "tm_designation", 
        join_type="left", conflict_suffixes=("_KIC", "_APOGEE"))
    del(apogee)
    
    # Remove APOGEE giants
    autodwarfs = mcq_observing["LOGG"] < 0
    mcq_observing["LOGG"][mcq_observing["LOGG"] < 0] = 9999.0
    mcq_observing = catalog.perform_logg_cut(
        mcq_observing, lowlogg=3.5, loggcol="LOGG")
    mcq_observing["LOGG"][mcq_observing["LOGG"] == 9999.0] = -9999.0

    # Remove objects which are already observed to be RV variable
    mcq_observing["VSCATTER"] = mcq_observing["VSCATTER"].filled(-9999.0)
    apo_var = catalog.perform_vscatter_cut(
        mcq_observing, lowv=1, vcol="VSCATTER")
    mcq_observing = catalog.perform_vscatter_cut(
        mcq_observing, highv=1, vcol="VSCATTER")
    print(len(apo_var))
    mcq_observing["VSCATTER"] = np.ma.masked_values(mcq_observing["VSCATTER"],
                                                 -9999.0)
    
    # Remove objects which have been observed enough to indicate non
    # RV-variability.
    mcq_observing["NVISITS"] = mcq_observing["NVISITS"].filled(0)
    apo_nonvar = catalog.perform_cut(mcq_observing, "NVISITS", lowval=4,
                                     invert_inequality=True)
    mcq_observing = catalog.perform_cut(mcq_observing, "NVISITS", highval=4)
    mcq_observing["NVISITS"] = np.ma.masked_values(mcq_observing["NVISITS"], 0)

    hr.absmag_teff_plot(
        mcq["teff"], mcq["M_K"], color=bc.black, marker=".", ls="", 
        label="Full McQuillan")
    hr.absmag_teff_plot(
        rapid_rotators["teff"], rapid_rotators["M_K"], color=bc.pink, 
        marker="d", ls="", label="1 day < Prot < 5 day")
    hr.absmag_teff_plot(
        very_rapid["teff"], very_rapid["M_K"], color=bc.purple,
        marker="d", ls="", label="Prot < 1 day")

    hr.absmag_teff_plot(
        mdm_mcq["teff"], mdm_mcq["M_K"], color=bc.blue, marker="*", ls="",
        label="MDM Nonvariable", ms=9)
    hr.absmag_teff_plot(
        mdm_rvvar["teff"], mdm_rvvar["M_K"], color=bc.sky_blue, marker="*", 
        ls="", label="MDM Variable", ms=9)
    hr.absmag_teff_plot(
        apo_nonvar["teff"], apo_nonvar["M_K"], color=bc.blue, marker="^", ls="",
        label="APOGEE Nonvariable", ms=9)
    hr.absmag_teff_plot(
        apo_var["teff"], apo_var["M_K"], color=bc.sky_blue, marker="^", 
        ls="", label="APOGEE Variable", ms=9)

    plt.xlabel("Huber Teff (K)")
    plt.ylabel("M_K")
    plt.legend(loc="upper right")

###############################################################################
# Check uncertainties #
###############################################################################

def compare_abs_mag_composite_uncertainties(
        teffs, sigK, highdist, lowdist, dist):
    '''Compare the uncertainties of k-band photometry gaia distances.
    
    Plot the k-band variance vs the variance due to the uncertainty in the
    distance.'''
    plt.plot(teffs, (sigK)**2, color=bc.orange, marker="o", ls="")
    plt.plot(teffs, (5*(highdist + lowdist)/dist/np.log(10)/2)**2,
             color=bc.purple, marker="d", ls="")
    hr.invert_x_axis()
    plt.xlabel("Teff (K)")
    plt.ylabel("Variance")

def compare_observed_age_uncertainties(
        MKhi, MKlow, teffs, lowage=1, highage=10):
    plt.plot(teffs, (MKhi + MKlow)/2, color=bc.black, marker="o", ls="")
    highmag = calc_DSEP_model_mags(teffs, np.zeros(len(teffs)), "Ks", age=highage)
    lowmag = calc_DSEP_model_mags(teffs, np.zeros(len(teffs)), "Ks", age=lowage)
    ageerr = lowmag - highmag
    sortindices = np.argsort(teffs)
    plt.plot(teffs[sortindices], ageerr[sortindices], color=bc.red, marker="", 
             ls="-")

    
    
