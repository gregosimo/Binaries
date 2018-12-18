import numpy as np
import astropy_util as au
import matplotlib.pyplot as plt


import read_catalog as catin
import catalog
import biovis_colors as bc
import sample_characterization as samp
import hrplots as hr


def Noyes_convective_overturn_timescale(bv):
    '''Calculate the convective overturn timescale from Noyes.

    The expression for the timescale from Noyes et al (1984) is a
    piecewise-defined expression as a function of B-V. The convective overturn
    timescale is given in days.'''
    x = 1-bv
    posx = 1.362 - 0.166 * x + 0.025 * x**2 -5.323 * x**3
    negx = 1.362 - 0.14 * x
    logt = np.where(x > 0, posx, negx)

    return 10**logt

def Activity_Populations():
    '''Classify McQuillan targets by activity.'''
    # I haven't correctly separated the subgiants. So for now I'm going to only
    # use the sample with 1.5 > B-V > 1.
    mcq = catin.McQuillan_EHK()
    clean_mcq = mcq[np.logical_not(au.multi_logical_or(
        mcq["teff"].mask, mcq["M_K"].mask))]
    mcq_cool = clean_mcq[clean_mcq["teff"] < 5250]
    mcq_cool_mean_teff_error = np.mean([
        mcq_cool["teff_err1"], -mcq_cool["teff_err2"]])
    mcq_cool_mean_mk_error = np.mean([
        mcq_cool["M_K_err1"], mcq_cool["M_K_err2"]])
    mcq_cool["MIST K"] = samp.calc_model_mag_fixed_age_feh_alpha(
        mcq_cool["teff"], 0.0, "Ks", age=1e9, model="MIST v1.2")
    mcq_cool["MIST K Error"] = samp.calc_model_mag_err_fixed_age_feh_alpha(
        mcq_cool["teff"], 0.0, "Ks", teff_err=mcq_cool_mean_teff_error, age=1e9,
        model="MIST v1.2")
    mcq_cool["K Excess"] = mcq_cool["M_K"] - mcq_cool["MIST K"]
    mcq_cool["K Excess Err"] = np.sqrt(
        mcq_cool_mean_mk_error**2 + mcq_cool["MIST K Error"]**2)
    active_sample = mcq_cool[mcq_cool["K Excess"] > -0.3]

    cot = Noyes_convective_overturn_timescale(active_sample["B-V"])
    active_sample["Rossby"] = active_sample["Prot"] / cot

    return active_sample

def McQuillan_Activity_Rossby_Relation():
    '''Plot the McQuillan activity/Rossby number relationship.'''
    samp = Activity_Populations()

    stand_pop = np.log10(samp["Rper"]) < np.log10(1.8e4) + (
        (np.log10(1.8e4) - np.log10(3.8e3)) / 
        (np.log10(0.64) - np.log10(0.95)) * (
            np.log10(samp["Rossby"]) - np.log10(0.64)))
    new_pop = np.log10(samp["Rper"]) > np.log10(1.8e4) + (
        (np.log10(1.8e4) - np.log10(3.8e3)) / 
        (np.log10(0.64) - np.log10(0.95)) * (
            np.log10(samp["Rossby"]) - np.log10(0.64)))

    f, ax = plt.subplots(1, 1)
    ax.errorbar(
        samp["Rossby"][stand_pop], samp["Rper"][stand_pop], marker=".",
        color='r', ls="")
    ax.errorbar(
        samp["Rossby"][new_pop], samp["Rper"][new_pop], marker=".",
        color='b', ls="")
#   hr.absmag_teff_plot(mcq_cool["teff"], mcq_cool["M_K"],
#                       marker=".", color=bc.black, ls="")
    ax.set_xlabel("Rossby Number")
    ax.set_ylabel("amplitude (ppm)")
    ax.set_xscale("log")
    ax.set_yscale("log")

def McQuillan_Activity_Rossby_HR_Location():
    '''Plot the targets used for McQuillan activity/Rossby number relationship.'''
    fullmcq = catin.McQuillan_EHK()
    clean_mcq = fullmcq[np.logical_not(au.multi_logical_or(
        fullmcq["teff"].mask, fullmcq["M_K"].mask))]
    samp = Activity_Populations()

    f, ax = plt.subplots(1, 1)
    hr.absmag_teff_plot(
        fullmcq["teff"], fullmcq["M_K"], marker=".", color=bc.black, ls="",
        axis=ax)
    hr.absmag_teff_plot(
        samp["teff"], samp["M_K"], marker=".", color='r', ls="", axis=ax)

    ax.set_xlabel("TEFF")
    ax.set_ylabel("M_K")
