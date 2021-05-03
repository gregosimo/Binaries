import read_catalog as catin
import hrplots as hr

import matplotlib.pyplot as plt
import numpy as np

def plot_tayar_rapid_rotators():
    '''Make the original plot of Tayar rapid rotators.'''
    tayarcat = catin.Tayar_updated_APOGEE()

    rapid_rots = np.ma.getmaskarray(tayarcat["f_vsini"])

    median_Teff_err = np.ma.median(tayarcat["Tayar e_Teff"])
    median_logg_err = np.ma.median(tayarcat["Tayar e_logg"])

    f, ax = plt.subplots(1, 1)

    hr.logg_teff_plot(
        tayarcat["Tayar Teff"][~rapid_rots], tayarcat["Tayar logg"][~rapid_rots], 
        style="k.", axis=ax, label="")
    hr.logg_teff_plot(
        tayarcat["Tayar Teff"][rapid_rots], tayarcat["Tayar logg"][rapid_rots], 
        style="ro", axis=ax, label="Tayar vsini > 5 km/s")
    hr.logg_teff_plot(
        4000, 3.0, style="k.", yerr=median_logg_err, xerr=median_Teff_err, 
        axis=ax)

    ax.set_xlabel("Tayar Teff (K)")
    ax.set_ylabel("Tayar logg (K)")
    ax.legend(loc="upper left")

def compare_tayar_rapid_rotators_gaia():
    '''Compare the rapid rotators with Tayar parameters and Gaia parameters.'''
    tayarcat = catin.Tayar_updated_APOGEE()

    rapid_rots = np.ma.getmaskarray(tayarcat["f_vsini"])

    f, axes = plt.subplots(1, 2)

    median_tayar_teff_err = np.ma.median(tayarcat["Tayar e_Teff"])
    median_tayar_logg_err = np.ma.median(tayarcat["Tayar e_logg"])
    median_apogee_teff_err = np.ma.median(tayarcat["TEFF_ERR"])
    median_gaia_M_K_err1 = np.ma.median(tayarcat["M_K_err1"])
    median_gaia_M_K_err2 = np.ma.median(tayarcat["M_K_err2"])

    teffs = [tayarcat["Tayar Teff"], tayarcat["TEFF"]]
    yvals = [tayarcat["Tayar logg"], tayarcat["M_K"]]
    xerrs = [median_tayar_teff_err, median_apogee_teff_err]
    yerrs = [median_tayar_logg_err, ((
        median_gaia_M_K_err1 + median_gaia_M_K_err2) / 2)]
    errlocs = [(4000, 3.0), (4000, 0)]
    plotters = [hr.logg_teff_plot, hr.absmag_teff_plot]
    xlabels = ["Tayar Teff (K)", "APOGEE Teff (K)"]
    ylabels = ["Tayar logg", "M_K"]
   
    for (ax, teff, yval, xerr, yerr, plotter, errloc, xlabel, ylabel) in zip(
            axes, teffs, yvals, xerrs, yerrs, plotters, errlocs, xlabels,
            ylabels):
        plotter(
            teff[~rapid_rots], yval[~rapid_rots], ls="", marker=".", 
            color="black", axis=ax, label="")
        plotter(
            teff[rapid_rots], yval[rapid_rots], ls="", marker="o", 
            color="red", axis=ax, label="Tayar vsini > 5 km/s")
        plotter(
            errloc[0], errloc[1], yerr=yerr, xerr=xerr, ls="", marker=".",
            color="black", axis=ax)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()

def compare_tayar_apogee_rapid_rotators():
    '''Compare the rapid rotators flagged by Tayar and APOGEE.'''
    tayarcat = catin.Tayar_updated_APOGEE()

    tayar_rapid_rots = np.logical_and(
        np.logical_not(np.ma.getmaskarray(tayarcat["f_vsini"])), 
        tayarcat["Tayar vsini"] > 10])
    apogee_rapid_rots = (tayarcat["VSINI"] > 10).filled(False)

    f, axes = plt.subplots(1, 2)




