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
import numpy as np
import matplotlib.pyplot as plt

import catalog
import hrplots as hr
import APOGEE_spectroscopy as apo

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
