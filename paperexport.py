import os

import numpy as np
import numpy.core.defchararray as npstr
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import ascii

import observations as obs
import path_config as paths
import read_catalog as catin
import hrplots as hr
import astropy_util as au
import catalog
import sed

PAPER_PATH = paths.HOME_DIR / "papers" / "tidsync17"
TABLE_PATH = PAPER_PATH / "tables"
FIGURE_PATH = PAPER_PATH / "fig"

def build_filepath(toplevel, filename, suffix="png"):
    '''Generate a full path to save a filename.'''

    fullpath = toplevel / ".".join((filename, suffix))
    return str(fullpath)

def get_sample():
    '''Get the final sample of the observing targets.'''
    final_sample = Table.read(
        paths.HEAD_DIR / "Obs_list.txt", format="ascii.fixed_width")
    return final_sample

def create_observing_sample_table(
        dest=build_filepath(TABLE_PATH, "obsprops", "tex")):
    '''Create the observing table with relevant parameters.

    For this table, the relevant parameters are the Huber Teff and log(g), the
    McQuillan period, and the EHK V-band magnitude.'''
    sample = get_sample()

    sample_names = sample["KIC_A"]
    sample_teff = sample["teff"]
    sample_logg = sample["logg"]
    sample_period = sample["Prot"]
    sample_V = sample["V"]

    endcomments = r"""
For each Kepler target, the \(T_{eff}\) and \(\log (g)\) values are taken from 
the Kepler Stellar Parameter Pipeline (see text). The rotation periods are from 
\citet{McQuillan14}, and V-band magnitudes are from the \citet{Everett12} survey 
of the Kepler field."""

    obsdict = {"tabletype": "table*", "tablealign": "htb", 
               "col_align": "r c c c c", 
               "caption": "Observing Sample Properties\\label{tab:obsprops}",
               "preamble": "", "header_start": "", "data_start": "\\tableline",
               "data_end": "", "tablefoot": endcomments, 
               "units": {"teff": "K", "logg": "cm s\(^2\)", "Prot": "day", 
                         "V": "mag"}
               }

    output_table = Table(
        [sample_names, sample_teff, sample_logg, sample_period, sample_V], 
        names=("KIC", r"\(T_{eff}\)", r"\(\log (g)\)", r"\(P_{rot}\)", "V"))
    output_table.sort("KIC")

    column_format = {r"\(\log (g)\)": ".2f", r"\(P_{rot}\)": ".2f", "V": ".1f"}

    output_table.write(dest, format="latex", latexdict=obsdict,
                       formats=column_format)


def create_APOGEE_table(dest=build_filepath(TABLE_PATH, "apotab", "tex")):
    '''Create the table showing APOGEE parameters for objects that have them.

    This table will contain APOGEE-determined Teff, logg, vsini, and
    vscatter.'''
    full_sample = get_sample()
    good_sample = catalog.filter_bad_ASPCAP_fits(full_sample)
    sample = good_sample[~npstr.endswith(good_sample["APOGEE_ID"], "N/A")]

    kic_name = sample["KIC_A"]
    apo_apid = sample["APOGEE_ID"]
    apo_teff = sample["TEFF"]
    apo_logg = sample["LOGG_FIT"]
    apo_vsini = sample["VSINI"]
    apo_vscatter = sample["VSCATTER"]
    apo_nvisits = npstr.replace(sample["NVISITS"], "--", "1")


    endcomments = r"""APOGEE parameters are only those with good fits from DR14. 
    \log (g) are uncorrected values from FERRE. Missing VScatter objects are
    those with only one visit. """

    obsdict = {"tabletype": "table*", "tablealign": "htb", 
               "col_align": "r r c c c c", 
               "caption": "Targets with APOGEE observations\\label{tab:apogee}",
               "preamble": "", "header_start": "", "data_start": "\\tableline",
               "data_end": "", "tablefoot": endcomments
               }

    output_table = Table(
        [kic_name, apo_apid, apo_teff, apo_logg, apo_vsini, apo_vscatter,
         apo_nvisits],
        names=("KIC", "APOGEE ID",  r"APOGEE \(T_{eff}\)", r"APOGEE \(\log (g)\)", 
               r"v \sin i", "VScatter", "Num. Visits")) 
    column_format = {r"APOGEE \(\log (g)\)": ".2f", r"v \sin i": ".1f",
                     "VScatter": ".2f"}

    output_table.write(
        dest, format="latex", latexdict=obsdict, formats=column_format, 
        fill_values=[
            ('-9999.0', '--', r"v \sin i"), ('0.00', '--', 'VScatter')])

def RV_var_table(dest=build_filepath(TABLE_PATH, "vartab", "tex")):
    '''Create the table of APOGEE known RV variable objects.'''
    sample = obs.select_RV_variable_targets()

    kic_name = sample["KIC_A"]
    teff = sample["teff"]
    logg = sample["logg"]
    period = sample["Prot"]
    V = sample["V"]
    vsini = sample["VSINI"]
    vscatter = sample["VSCATTER"]
    nvisits = npstr.replace(sample["NVISITS"], "--", "1")




def create_sample_HR_diagram(dest=build_filepath(FIGURE_PATH, "sample")):
    '''HR diagram showing the location of the observing sample.

    The broadest swath of objects to be plotted are those with Kepler data.
    Then the McQuillan targets. And then there will be an inset within the Teff
    and log(g) bounds.
    '''
    pass

def create_sample_Prot_diagram(dest=build_filepath(FIGURE_PATH, "prot_sample")):
    '''Prot-Teff digram showing the observing sample.

    The broadest swatch of objects to be plotted are the McQuillan targets.  After that, various cuts will be performed. The cuts in this figure will be
    those with V > 14, and then those with APOGEE RV detections, RV
    nondetections, and those that are background pulsators.'''
    mcquillan_color = (86/255.0, 180/255.0, 233/255.0)
    bright_color = (204/255.0, 121/255.0, 167/255.0)
    pulse_color = (240/255.0, 228/255.0, 66/255.0)
    giant_color = (230/255.0, 159/255.0, 0/255.0)
    rvvar_color = (0/255.0, 114/255.0, 178/255.0)
    rvnonvar_color = (213/255.0, 94/255.0, 0/255.0)
    eb_color = (0, 0, 0)
    sample_color = (0/255.0, 158/255.0, 115/255.0)

    mcq = catin.mcquillan_with_stelparms()
    mcq_observing = catalog.select_tidally_synchronized_binaries(
        mcq, pcut=5, lowperiod=1, lowtemp=4850, hightemp=5600, logg=3.5,
        teffcol="teff", pcol="Prot", loggcol="logg")
    plt.plot(mcq_observing["teff"], mcq_observing["Prot"],
             color=mcquillan_color, marker=".", label="Full McQuillan",
             ls="None")
    # Now remove those with V > 14.
    mcq_phot = catin.mcquillan_photometry()
    mcq_observing = au.join_by_id(mcq_observing, mcq_phot, "kepid", "KIC")
    mcq_observing.rename_column("KIC_A", "KIC")
    del(mcq_phot)
    missing_Vs = mcq_observing["V"].mask
    print("{0} missing in HE catalog.".format(np.count_nonzero(missing_Vs)))
    JK_interp = sed.color_to_color_DSEP_interpolator(
        "J-Ks", "V-H", {"V": 1, "J": 1, "H": 1, "Ks": 1}, age=2, init_mass=0.9)
    missing_JKs = (mcq_observing['jmag'][missing_Vs] -
                   mcq_observing["kmag"][missing_Vs]).filled()
    mcq_observing["V"][missing_Vs] = (
        JK_interp(missing_JKs) + mcq_observing["hmag"][missing_Vs])
    mcq_observing = catalog.perform_cut(mcq_observing, "V", highval=14)
    plt.plot(mcq_observing["teff"], mcq_observing["Prot"], color=bright_color,
             marker="+", label="Bright (V<14)", ls="None", ms=14)
    # Remove pulsators
    no_pulsators = catalog.filter_pulsators(mcq_observing, KICcol="KIC")
    pulsators = au.get_complement_table(no_pulsators, mcq_observing, "KIC")
    if len(pulsators) > 0:
        plt.plot(pulsators["teff"], pulsators["Prot"], color=pulse_color,
                 marker="p", label="Pulsator", ls="None")
    else:
        print("No pulsators in sample.")

    # Remove APOGEE Giants
    apogee = catin.mcquillan_dr14_overlap()
    mcq_with_apogee = catalog.join_by_2MASS_key(
        no_pulsators, apogee, "tm_designation", "tm_designation", 
        join_type="left", conflict_suffixes=("_KIC", "_APOGEE"))
    del(apogee)
    
    # Remove APOGEE giants
    giants = catalog.perform_logg_cut(
        mcq_with_apogee, lowlogg=0, highlogg=3.5, loggcol="LOGG")
    mcq_no_giants = au.get_complement_table(giants, mcq_with_apogee, "KIC")
    if len(giants) > 0:
        plt.plot(giants["teff"], giants["Prot"], color=giant_color,
                 marker="8", label="APOGEE Giant", ls="None")
    else:
        print("No APOGEE Giants in sample")

    # Remove RV Variable targets
    mcq_no_giants["VSCATTER"] = mcq_no_giants["VSCATTER"].filled(-9999.0)
    mcq_rv_nonvar = catalog.perform_vscatter_cut(
        mcq_no_giants, highv=1, vcol="VSCATTER")
    mcq_no_giants["VSCATTER"] = np.ma.masked_values(mcq_no_giants["VSCATTER"],
                                                    -9999.0)
    mcq_rv_nonvar["VSCATTER"] = np.ma.masked_values(mcq_rv_nonvar["VSCATTER"],
                                                    -9999.0)
    rv_var = au.get_complement_table(mcq_rv_nonvar, mcq_no_giants, "KIC")
    if len(rv_var) > 0:
        plt.plot(rv_var["teff"], rv_var["Prot"], color=rvvar_color, marker="d",
                 label="APOGEE RV Variable", ls="None")
    else:
        print("No APOGEE RV Variable targets")

    # Remove RV Nonvariable targets
    mcq_rv_nonvar["NVISITS"] = mcq_rv_nonvar["NVISITS"].filled(0)
    mcq_rv_unsure = catalog.perform_cut(mcq_rv_nonvar, "NVISITS", highval=4)
    mcq_nonvar = au.get_complement_table(mcq_rv_unsure, mcq_rv_nonvar, "KIC")
    mcq_rv_unsure["NVISITS"] = np.ma.masked_values(mcq_rv_unsure["NVISITS"], 0)
    if len(mcq_rv_nonvar) > 0:
        plt.plot(mcq_nonvar["teff"], mcq_nonvar["Prot"], 
                 color=rvnonvar_color, marker="*", 
                 label="APOGEE RV Non-Variable", ls="None")
    else:
        print("No APOGEE RV Nonvariable targets")

    # Remove eclipsing binaries.
    mcq_noebs = catalog.remove_Kepler_EBs(mcq_rv_unsure, mainkiccol="kepid")
    mcq_ebs = au.get_complement_table(mcq_noebs, mcq_rv_unsure, "KIC")
    if len(mcq_ebs) > 0:
        plt.plot(mcq_ebs["teff"], mcq_ebs["Prot"], color=eb_color, 
                 marker="v", label="Eclipsing Binary", ls="None")
    else:
        print("No Eclipsing Binaries")


    # Finally the full sample
    fullsamp = obs.select_observing_targets(30)
    plt.plot(fullsamp["teff"], fullsamp["Prot"], color=sample_color,
             marker="o", label="Observing Sample", ls="None")

    hr.invert_x_axis()
    plt.legend(loc="upper right")
    plt.xlabel("Huber Teff")
    plt.ylabel("Mcquillan Period (day)")
