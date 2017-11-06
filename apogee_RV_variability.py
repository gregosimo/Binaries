'''
Module investigating APOGEE RV Variability.

This module has plots which look at how the APOGEE RV variable and non-variable
samples differ from each other.

velocity evolution:
    Plots an RV curve of RV variable and RV non-variable objects.

A common part of the module are standout plots, which plot the objects with and 
without RV variability against a backdrop of the total sample. The standout
plots are:

McQuillan_Standout_Plot:
    Plot which part of Teff-Prot space is occupied by RV variable and
    nonvvariable objects.

VIM_effect_on_McQuillan_standout_plot:
    Plot whether RV variability is correlated with VIM
    
'''
import matplotlib.pyplot as plt

import catalog
import sample_characterization as samp

def velocity_evolution(variable, nonvariable):
    '''Automatically generate the velocity evolution of potential binaries.
    
    This function will plot RV-variable objects in red and RV-nonvariable
    objects in blue.'''

    sampcolors = ["r", "b"]

    tablelist = []
    for samp, col in zip([variable, nonvariable], sampcolors):
        for obj in samp:
            obj_id = obj["2MASS_ID"]
            obj_loc_id = obj["LOC_ID"]
            eisit_table = catalog.get_APOGEE_visit_info(obj_id, obj_loc_id)
            visit_dates = visit_table["MJD"]
            visit_velocities = visit_table["V_LSR"]
            
            visit_table["2MASS_ID"] = obj_id
            tablelist.append(visit_table)

            plt.plot(
                visit_dates-visit_dates[0], visit_velocities, col+"-", 
                label=obj_id)

    plt.legend()
    plt.xlabel("MJD - First MJD")
    plt.ylabel("V_LSR (km/s)")

    fulltable = vstack(tablelist)
    return fulltable

def McQuillan_standout_plot(
    fullsample, rv_nonvar, rv_var, Teff_colname="TEFF_FIT",
    Prot_colname="Prot", data_label="McQuillan/APOKASC"):
    '''Creates a plot like McQuillan et al. but overplots RV samples.

    Takes the full McQuillan sample and overplots the RV-variable and
    RV-nonvariable samples on top in red and blue.
    '''
    samp.McQuillan_plot(fullsample, Teff_colname=Teff_colname,
                   Prot_colname=Prot_colname, color="c", marker=".",
                   label=data_label)
    samp.McQuillan_plot(rv_var, Teff_colname=Teff_colname,
                   Prot_colname=Prot_colname, color="r", marker="*",
                   label="RV Variable", ms=12)
    samp.McQuillan_plot(rv_nonvar, Teff_colname=Teff_colname,
                   Prot_colname=Prot_colname, color="b", marker="*",
                   label="RV Nonvariable", ms=12)
    hr.invert_x_axis()
    plt.xlabel("Teff (K)")
    plt.ylabel("Prot (day)")
    plt.title("Jen van Saders-cut sample (Multiepoch)")
    plt.legend(loc="lower left")

def VIM_effect_on_McQuillan_standout_plot(
    fullsample, rv_nonvar, rv_var, minvim=3, Teff_colname="TEFF_FIT",
    Prot_colname="Prot", KIC_colname="KEPLER_INT"):
    '''Creates a plot showing what the effects of VIM are with APOKASC data.'''
    vimtable = catin.read_KepVIM_catalog()
    quartertable = kepVIM_quarter_table(vimtable)
    for q in range(minvim, NUM_KEPLER_QUARTERS+1):
        plt.figure()
        # These are the KICs of the objects which have at least q quarters of
        # VIM detections.
        kic_at_least_q_indices = np.unique(quartertable["KIC"][
            quartertable["Num_Q"] >= q])
        
        # Get the indices of the VIM detections.
        rv_nonvar_vim_indices = au.astropy_table_indices(
            rv_nonvar, KIC_colname, kic_at_least_q_indices)
        rv_var_vim_indices = au.astropy_table_indices(
            rv_var, KIC_colname, kic_at_least_q_indices)
        # These then are the indices of the nonVIM detections.
        rv_nonvar_nonvim_indices = au.get_complement_indices(
            rv_nonvar_vim_indices, len(rv_nonvar))
        rv_var_nonvim_indices = au.get_complement_indices(
            rv_var_vim_indices, len(rv_var))

        # Now make the tables that should be plotted
        rv_nonvar_nonvim = rv_nonvar[rv_nonvar_nonvim_indices]
        rv_var_nonvim = rv_var[rv_var_nonvim_indices]
        vims = vstack([rv_nonvar[rv_nonvar_vim_indices],
                       rv_var[rv_var_vim_indices]])
        print("Number of VIMs is {0}.".format(len(vims)))
        McQuillan_standout_plot(
            fullsample, rv_nonvar_nonvim, rv_var_nonvim, 
            Teff_colname=Teff_colname, Prot_colname=Prot_colname)
        plt.semilogy(vims[Teff_colname], vims[Prot_colname], 'm*', ms=12,
                     label="VIM blends")
        plt.legend(loc="lower left")
        plt.title("Vim cutoff {0:n} Quarters".format(q))
        return vims
