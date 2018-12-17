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

def McQuillan_Activity_Rossby_Relation():
    '''Plot the McQuillan activity/Rossby number relationship.'''
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
    rossby_num = active_sample["Prot"] / cot

    f, ax = plt.subplots(1, 1)
    ax.errorbar(rossby_num, active_sample["Rper"], marker=".", color=bc.black,
                ls="")
#   hr.absmag_teff_plot(mcq_cool["teff"], mcq_cool["M_K"],
#                       marker=".", color=bc.black, ls="")
    ax.set_xlabel("Rossby Number")
    ax.set_ylabel("amplitude (ppm)")
    ax.set_xscale("log")
    ax.set_yscale("log")



