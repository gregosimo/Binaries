"""Functions for manipulating the APOKASC-McQuillan catalog
"""
import os
import re
import itertools

import numpy as np
import numpy.core.defchararray as npstr
from scipy.io import readsav
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import Table, join, vstack, unique
from astropy.coordinates import SkyCoord
from bs4 import BeautifulSoup
import requests


import astropy_util as au
import hrplots as hr
import binarycalcs as bc
import path_config as paths

WORKPATH = "/home/regulus/simonian/Binaries"

APOKASC_PATH = os.path.join(WORKPATH, "APOKASC_cat_v3.3.5.fits")
MCQUILLAN_PATH = os.path.join(WORKPATH, "McQuillan.fit")

SDSS3_URL = "http://data.sdss3.org"

NUM_KEPLER_QUARTERS = 17

###############################################################################
# Reading catalogs #
###############################################################################

def read_APOKASC_catalog(
    filepath=APOKASC_PATH, exclude_single_epoch=False, filter_BAD=False, 
    filter_WARN=False):
    '''Reads in the APOKASC catalog.

    The catalog should be located at filepath.
    '''
    apocat = Table.read(filepath, format="fits")
    if exclude_single_epoch:
        apocat = apocat[np.where(apocat["VSCATTER"] > 0.0)]
        add_cut_metadata(apocat, "VSCATTER > 0")
    if filter_BAD:
        apocat = filter_bad_ASPCAP_fits(apocat, filter_WARN)
    return apocat

def read_EHK_catalog(filepath=str(paths.EHK_PATH)):
    '''Reads in the UBV catalog fof the Kepler field.'''
    cat = Table.read(
        filepath, format="ascii.csv", 
        names=["RA", "Dec", "U", "U_err", "B", "B_err", "V", "V_err"] )
    return cat

def read_McQuillan_catalog(
    filepath=MCQUILLAN_PATH, Huber_KIC=True, huberpath=paths.HUBER_CATALOG):
    '''Reads in the McQuillan catalog.

    The catalog shoul be located at filepath.
    '''
    mcquillancat = Table.read(filepath, format="fits")
    if Huber_KIC:
        hubercat = read_Huber_KIC_catalog(huberpath)
        del(mcquillancat["log_g_"])
        del(mcquillancat["Teff"])
        mcquillancat = au.join_by_id(mcquillancat, hubercat, "KIC", "KIC")
    return mcquillancat

def read_Huber_KIC_catalog(huberpath=paths.HUBER_CATALOG):
    '''Read the HUBER KIC parameters.'''
    hubercat = Table.read(str(huberpath), format="ascii.cds")
    return hubercat

def read_van_Saders_file(vspath=paths.VAN_SADERS_SDSS):
    '''Read the APOGEE dwarf targets from Jen's catalog.'''
    # I don't use these columns, or know what they are, and they are 
    # 8-dimensional so they mess up the table.
    exclude_columns = ["KOIPERIOD", "TRANSITEPOCH"]
    vsidl = readsav(str(vspath))
    vsdata = vsidl["dat"]
    comments = vsidl["comments"]
    datafile = Table()
    for name in vsdata.dtype.names:
        if name not in exclude_columns:
            data_array = vsdata[name][0]
            # I want this to just be 1-d this time.
            datafile[name] = np.reshape(data_array, len(data_array))
    datafile["COMMENTS"] = vsidl["comments"]
    return datafile

def read_van_Saders_catalog(
    vspath=paths.VAN_SADERS_SDSS, mastpath=paths.VAN_SADERS_MAST):
    '''Table with relevant quantities for Jen's sample.

    This table will essentially be curated in detail to ensure that the
    returned catalog has the relevant and desired quantities.
    '''
    pass

def read_van_Saders_Kepler(vspath=paths.VAN_SADERS_MAST):
    '''Read MAST output for Jen's sample.'''
    mastdata = Table.read(
        vspath, format="ascii.csv", header_start=2, data_start=4)
    return mastdata

def read_van_Saders_APOGEE_catalog(vspath1=paths.JEN_APOGEE_1,
                                   vspath2=paths.JEN_APOGEE_2):
    '''Read the APOGEE information for Jen's sample.'''
    table1 = Table.read(str(vspath1), format="ascii.csv", comment="#")
    table2 = Table.read(str(vspath2), format="ascii.csv", comment="#")
    apogeetable = vstack([table1, table2])
    return apogeetable

###############################################################################
# Catalog Curation
###############################################################################

# def van_Saders_relevant_table(

###############################################################################
# Writing to databases #
###############################################################################

def write_MAST_files(outputtable, kiccol="KIC", outputpath=paths.HEAD_DIR,
                     output_filename="Kepler_MAST.txt"):
    '''Writes KICs so that they are able to be read by the MAST target form.

    This function will take a table and write out the KIC column to the given 
    filename. This file will then be able to be uploaded to the MAST Target
    Search (File Upload) page with the "KIC" File Contents.

    Note that since queries may occasionally be greater than the maximum file
    length allowable by the MAST target form, this function will write queries
    of over 10,000 objects to multiple files. Those files will be have the same
    basename, but will have a sequential ordering to them. For example,
    "output.txt" will become "output_1.txt", "output_2.txt", etc.
    '''
    MAST_LIMIT = 10000
    if len(outputtable) > MAST_LIMIT:
        basename, ext = output_filename.split(".")
        output_filename = ".".join([basename+"_{0:d}", ext])
    for i in range(len(outputtable) // MAST_LIMIT + 1):
        startind = MAST_LIMIT * i
        endind = min(MAST_LIMIT*(i+1), len(outputtable))
        outputfile = str(outputpath / output_filename.format(i))
        names = [kiccol]
        outputsegment = outputtable[startind:endind]
        print(outputsegment)
        outputsegment.write(outputfile, format="ascii.no_header", 
                            include_names=names)

def write_crossID_file(
    outputtable, racol="RA", deccol="DEC", outputpath=paths.HEAD_DIR, 
    output_filename="APOGEE_targets.txt"):
    '''Writes a file to submit to SDSS crossID.

    This file can be used to submit to:
    http://skyserver.sdss.org/dr13/en/tools/crossid/crossid.aspx
    '''
    APOGEE_LIMIT = 1000
    if len(outputtable) > APOGEE_LIMIT:
        basename, ext = output_filename.split(".")
        output_filename = ".".join([basename+"_{0:d}", ext])
    for i in range(len(outputtable) // APOGEE_LIMIT + 1):
        startind = APOGEE_LIMIT * i
        endind = min(APOGEE_LIMIT*(i+1), len(outputtable))
        outputfile = str(outputpath / output_filename.format(i))
        names = ["ra", "dec"]
        outputsegment = outputtable[startind:endind]
        outputsegment[[racol, deccol]].write(outputfile, format="ascii.csv", 
                                             names=names)

def create_joined_APOKASC_McQuillan_catalog(
        apocat=None, mcquillancat=None, apofile=APOKASC_PATH,
    mcquillanfile=MCQUILLAN_PATH):
    '''Creates a joint APOKASC/McQuillan catalog.

    All Kepler objects which are measured in both the McQuillan sample as well
    as in APOGEE are left in the remaining table. If the tables have already
    been read, they may be specified in the apocat and mcquillancat parameters.
    If these parameters are left as None, they will be read from apofile and
    mcquillanfile first.
    '''
    if apocat is None:
        apocat = read_APOKASC_catalog(apofile)
    if mcquillancat is None:
        mcquillancat = read_McQuillan_catalog(mcquillanfile)

    combocat = au.join_by_id(apocat, mcquillancat, "KEPLER_INT", "KIC")
    return combocat

def visit_table(obj_ids, loc_ids):
    '''Gets a table with all observations.

    This function will look up all of the observations which were taken by a
    list of given 2MASS IDs (obj_ids) and location IDs. As a result, obj_ids and
    loc_ids should be the same length. It will return a table with the obj_id,
    MJD, and V_LSR of each observation.'''
    # This will hold a table for each obj_id/loc_id.
    observation_table_list = []

    for obj_id, loc_id in zip(obj_ids, loc_ids):
        visit_table = get_APOGEE_visit_info(obj_id, loc_id)
        observation_table_list.append(visit_table)

    fulltable = vstack(observation_table_list)
    return fulltable

def calc_NOBS(object_table, visit_table, obs_col="NOBS"):
    '''Add a NOBS column to the object table using the visit table.

    This function is used to attach an NOBS column to the object table. NOBS is
    a useful quantity which may be included in future versions of APOKASC, but
    currently is not. Visit_table should be the outut of the visit_table
    function with the obj_id column from the object table.
    '''
    nobs = NOBS_array(object_table, visit_table)
    object_table[obs_col] = nobs

def NOBS_array(object_table, visit_table):
    '''Creates an array that calculates the number of observaions from visits.

    Calculate the number of observations that were given to each object in the
    object table using the visit history from the visit table.
    '''
    ids = object_table["2MASS_ID"]  # ID array used for looking up indices.
    nobs = np.zeros(object_table["2MASS_ID"].shape) # Holds NOBS
    visit_groups = visit_table.group_by("2MASS_ID")
    for object_visits in visit_groups.groups:
        object_index = np.where(ids == object_visits["2MASS_ID"][0])
        nobs[object_index] = len(object_visits)

    return nobs

def add_Everett_photometry(inputtable, racol, deccol):
    '''Adds UBV photometry from the EHK survey to table.

    Read in the EHK photometry and append that to the columns in inputtable.
    The table names will be the same as those in the photometry file.'''
    photcatalog = read_EHK_catalog()
    newcat = join_by_ra_dec(inputtable, photcatalog, racol, deccol)

def number_binned_by_temperature(
    mcquillan, hightemp=6500, lowtemp=3000, dtemp=50, teffcol="Teff"):
    '''Return array with number as a function of temperature.'''
    # Add dtemp because hist wants the rightmost edge.
    tempbins = np.arange(lowtemp, hightemp+dtemp, dtemp)
    hist, binedges = np.histogram(
        mcquillan[teffcol], bins=tempbins, range=(lowtemp, hightemp))
    return (hist, binedges)

def plot_number_bin(hist, binedges):
    '''Plot the number objects in each temperature bin.'''
    bincenters = (binedges[:-1] + binedges[1:])/2
    plt.plot(bincenters, hist)
    plt.xlabel("Teff (K)")
    plt.ylabel("Number of McQuillan objects in Temp bin")

def number_histogram(
    mcquillan, hightemp=6500, lowtemp=3000, dtemp=50, teffcol="Teff"):
    '''Plot a histogram of the number of McQuillan objects in temperature bins.

    Uses the matplotlib hist function to make the histogram plot.'''
    tempbins = np.arange(lowtemp, hightemp+dtemp, dtemp)
    plt.hist(mcquillan[teffcol], bins=tempbins, range=(lowtemp, hightemp),
             histtype="step")
    plt.xlim(plt.xlim()[::-1])
    plt.xlabel("Teff (K)")
    plt.ylabel("Number of McQuillan objects in Teff bin")

def rapid_fraction_histogram(
    mcquillan, hightemp=6500, lowtemp=3000, dtemp=50, maxper=5, teffcol="Teff",
    periodcol="Prot", label=""):
    '''Plot a histogram of the fraction of rapid rotators in McQuillan sample.
    '''
    totalhist, totbins = number_binned_by_temperature(
        mcquillan, hightemp=hightemp, lowtemp=lowtemp, dtemp=dtemp,
        teffcol=teffcol)
    rapid_mcquillan = perform_period_cut(
        mcquillan, highperiod=maxper, periodcol=periodcol)
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
            visit_table = get_APOGEE_visit_info(obj_id, obj_loc_id)
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
    

def get_APOGEE_visit_info(twomass_id, loc_id):
    '''Gets information for each visit of an APOGEE object.
    
    This function will scrape the SDSS3 web site in order to get this
    information. Currently it only gets the MJD of an observation and a
    relative velocity, but more can be added if need be.'''
    mjds = []
    vrels = []
    # Get ASPCAP page content.
    aspcap_resp = requests.post(
        SDSS3_URL+"/irSpectrumDetail", data={"apogeeid": twomass_id, "locid":
        loc_id, "commiss": 0, "show_aspcap": True})
    # If the page loads successfully, then populate MJD and Vrel. If not, then
    # return an empty table.
    if aspcap_resp.status_code < 300:
        # Find hyperlinks to individual visits.
        aspcapsoup = BeautifulSoup(aspcap_resp.content, "html.parser")
        visit_tags = aspcapsoup.find(
            string="Visit Spectra").parent.find_all_next(
                "a", href=re.compile("irSpectrum"))
        visit_urls = [SDSS3_URL + tag["href"] for tag in visit_tags]
        # Get MJD and vrel from individual pages.
        for url in visit_urls:
            visit_resp = requests.get(url)
            visitsoup = BeautifulSoup(visit_resp.content, "html.parser")
            visit_mjd = extract_mjd(visitsoup)
            mjds.append(visit_mjd)
            try:
                visit_vrel = extract_vrel(visitsoup)
            except TypeError:
                visit_vrel = np.nan
            vrels.append(visit_vrel)

    # Return table with 2MASS_ID, MJD and vrel.
    table_names = ("2MASS_ID", "MJD", "V_LSR")
    object_table = Table(
        [[twomass_id]*len(mjds), mjds, vrels], names=table_names,
        dtype=(np.str, np.int, np.float))

    return object_table

def extract_mjd(soup):
    '''Extracts the MJD value from a web page.

    The web page should be passed in as a BeautifulSoup object.
    '''
    mjd = int(soup.find(
        "span", style=re.compile("background-color:#CAF1D7")).string)
    return mjd

def extract_vrel(soup):
    '''Extracts the radial velocity from a web page.

    The web page should be passed in as a BeautifulSoup object.

    This function will count the LSR velocity as the relative velocity.
    '''
    vrad = float(soup.find(
        "sub", string="lsr").parent.next_sibling.next_sibling.string)
    return vrad

def VIM_effect_on_McQuillan_standout_plot(
    fullsample, rv_nonvar, rv_var, minvim=3, Teff_colname="TEFF_FIT",
    Prot_colname="Prot", KIC_colname="KEPLER_INT"):
    '''Creates a plot showing what the effects of VIM are with APOKASC data.'''
    vimtable = read_KepVIM_catalog()
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

def McQuillan_standout_plot(
    fullsample, rv_nonvar, rv_var, Teff_colname="TEFF_FIT",
    Prot_colname="Prot", data_label="McQuillan/APOKASC"):
    '''Creates a plot like McQuillan et al. but overplots RV samples.

    Takes the full McQuillan sample and overplots the RV-variable and
    RV-nonvariable samples on top in red and blue.
    '''
    McQuillan_plot(fullsample, Teff_colname=Teff_colname,
                   Prot_colname=Prot_colname, color="c", marker=".",
                   label=data_label)
    McQuillan_plot(rv_var, Teff_colname=Teff_colname,
                   Prot_colname=Prot_colname, color="r", marker="*",
                   label="RV Variable", ms=12)
    McQuillan_plot(rv_nonvar, Teff_colname=Teff_colname,
                   Prot_colname=Prot_colname, color="b", marker="*",
                   label="RV Nonvariable", ms=12)
    hr.invert_x_axis()
    plt.xlabel("Teff (K)")
    plt.ylabel("Prot (day)")
    plt.title("Jen van Saders-cut sample (Multiepoch)")
    plt.legend(loc="lower left")

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

def HR_standout_plot(
    fullsample, rv_nonvar, rv_var, Teff_colname="TEFF_FIT",
    logg_colname="LOGG_FIT"):
    '''Creates an HR diagram with RV samples.

    Takes the full APOKASC sample and overplots the RV-variable and
    RV-nonvariable samples on top in red and blue.
    '''
    hr.logg_teff_plot(
        fullsample[Teff_colname], fullsample[logg_colname], label="APOKASC")
    hr.logg_teff_plot(
        rv_var[Teff_colname], rv_var[logg_colname], 'r*', ms=12, 
        label="RV Variable")
    hr.logg_teff_plot(
        rv_nonvar[Teff_colname], rv_nonvar[logg_colname], 'b*', ms=12,
        label="RV Non-variable")
    plt.xlabel("Teff (K)")
    plt.ylabel("log g (cm/s^2)")
    plt.legend(loc="upper left")

def perform_cut(fullsample, col_label, lowval=None, highval=None,
                invert_inequality=False):
    """Perform a cut on the sample using a column

    This function picks out only those values in fullsample under col_label 
    where lowval <= val < highval.  It then adds a note to the metadata of the
    new table that the cut took place. If invert_inequality is specified, then
    the picked values will follow lowval < val <= highval.
    """
    colvalues = fullsample[col_label]

    if lowval is None and highval is None:
        cutsample = fullsample
        cutstring = ""
    else:
        if not invert_inequality:
            if lowval is None and highval is not None:
                cutsample = fullsample[np.where(colvalues < highval)]
                cutstring = "{0} < {1}".format(col_label, highval)
            elif lowval is not None and highval is None:
                cutsample = fullsample[np.where(colvalues >= lowval)]
                cutstring = "{0} >= {1}".format(col_label, lowval)
            elif lowval is not None and highval is not None:
                cutsample = fullsample[np.where(np.logical_and(
                    colvalues >= lowval, colvalues < highval))]
                cutstring = "{0} <= {1} < {2}".format(
                    lowval, col_label, highval)
        else:
            if lowval is None and highval is not None:
                cutsample = fullsample[np.where(colvalues <= highval)]
                cutstring = "{0} <= {1}".format(col_label, highval)
            elif lowval is not None and highval is None:
                cutsample = fullsample[np.where(colvalues > lowval)]
                cutstring = "{0} > {1}".format(col_label, lowval)
            elif lowval is not None and highval is not None:
                cutsample = fullsample[np.where(np.logical_and(
                    colvalues > lowval, colvalues <= highval))]
                cutstring = "{0} < {1} <= {2}".format(
                    lowval, col_label, highval)


    add_cut_metadata(cutsample, cutstring)
    return cutsample
    
def add_cut_metadata(sample, cutstring):
    '''Adds a string describing a cut to sample.

    The metadata entries in cut should be labeled under entries of cut_#, where
    the number ascends sequentially. If cutstring is an empty string, then no
    entry will be added to the metadata of sample.
    '''
    if cutstring:
        for i in itertools.count(1):
            metacol = "CUT_{0}".format(i)
            if metacol not in sample.meta:
                sample.meta[metacol] = cutstring
                break

def perform_teff_cut(fullsample, lowtemp=None, hightemp=None,
                     teffcol="TEFF_FIT"):
    '''Performs a Teff cut on the sample.

    This function filters the full table according to Teff values. By default
    APOGEE Teffs are used, but other values can be specified according to
    teffcol.'''
    return perform_cut(fullsample, teffcol, lowtemp, hightemp)

def perform_period_cut(fullsample, lowperiod=None, highperiod=None,
                       periodcol="Prot"):
    '''Performs a period cut on the sample.

    This function filters the full table according to measured rotation
    periods. By default, the McQuillan periods are used, but other values can
    be specified according to teffcol.'''
    return perform_cut(fullsample, periodcol, lowperiod, highperiod)

def perform_vscatter_cut(fullsample, lowv=None, highv=None, vcol="VSCATTER"):
    '''Performs a velocity scatter cut on the sample.

    This function filters the full table according to measured RV variability.
    By default, VSCATTER is used, but other values can be specified according
    to vcol.'''

    return perform_cut(fullsample, vcol, lowv, highv)

def read_pulsators(pulsatorfile=os.path.join(WORKPATH, "pulsators.kic")):
    '''Reads in a list of KIC IDs of known pulsators.'''

    pulsatortable = Table.read(
        pulsatorfile, format="ascii.no_header", names=["KIC"])
    return pulsatortable

def filter_pulsators(fulltable, quiet=False, KICcol="KEPLER_INT"):
    '''Removes known Kepler pulsators from a table of Kepler objects.

    If the quiet keyword is disabled, then this function will print the KIC IDs
    of the objects that were found to be pulsators.
    '''
    filteredtable = au.filter_column_from_subtable(
        fulltable, KICcol, read_pulsators()["KIC"])
    if not quiet:
        pulsators = au.get_complement_table(
            filteredtable, fulltable, KICcol)[KICcol]
        for kic in pulsators:
            print("KIC {0} is a Kepler Pulsator".format(kic))
    add_cut_metadata(filteredtable, "Pulsators removed")
    return filteredtable

def write_UKIRT_file(catalogtable, outputpath):
    '''Takes RA and DEC columns from catalog to make an upload file.'''
    coords = SkyCoord(
        ra=catalogtable["RA"], dec=catalogtable["DEC"], unit="deg")
    coordTable = Table(
        [coords.ra.degree, coords.dec.degree], names=("RA", "DEC"))
    coordTable.write(outputpath, format="ascii.no_header")

def read_UKIRT_file(resultfile):
    '''Reads in a file from UKIRT.'''
    results = Table.read(resultfile, format="ascii.commented_header")
    return results

def filter_good_UKIRT_observations(ukirt_table):
    '''Takes a table with UKIRT observations and only returns "good" entries.

    I'm following Jamie's lead and defining a "good" entry as one where the
    aperture magnitude is greater than -10. This is to exclude cosmic rays and
    other bright specks. Many of these do not seem to have associated error
    flags, so this will be used to define "good".
    '''
    good_indices = ukirt_table["jAperMag3"] > -10
    return ukirt_table[good_indices]

def find_UKIRT_contaminants(
    bintable, ukirt_file=os.path.join(WORKPATH, "ukirt_results.csv.gz")):
    '''Magnitude differences between stars and brightest contaminants

    This function takes a list from WFCAM, or if ukirt_file is None, will
    generate a list to be uploaded to WFCAM, and returns a list of the
    brightest contaminant for each star, with the magnitude difference
    included.
    '''
    # If ukirt_file is None, then ask the user to fetch one from the WFCAM
    # site.
    if ukirt_file is None:
        ukirt_input_path = input(
            "Please input the output path for the upload file to WFCAM:")
        write_UKIRT_file(bintable, ukirt_input_path)
        ukirt_file = input(
            "Please input the path to the WFCAM output file:")

    ukirt_catalog = read_UKIRT_file(ukirt_file)
    ukirt_catalog = filter_good_UKIRT_observations(ukirt_catalog)
    ukirt_groups = ukirt_catalog.group_by("upload_ID")

    magdiffs = np.zeros(len(ukirt_groups.groups))
    # The apogee fiber is approximately 2 arcseconds wide, so the source should
    # be in there
    apogee_window = 2 * u.arcsec
    # The optimal aperture size for Kepler photometry generally ranges from
    # 10-50 pixels. Each pixel is about 4 arcseconds long. 
    kepler_window = np.sqrt(50) * 4 * u.arcsec

    for i, contam_list in enumerate(ukirt_groups.groups):
        offsets = contam_list["distance"] * u.arcsec

        apogee_sources = contam_list[offsets < apogee_window]
        kepler_contam_sources = contam_list[np.where(np.logical_and(
            offsets > apogee_window, offsets < kepler_window))]

        target_ind = np.argmax(
            apogee_sources["jAperMag3"])
        brightest_contam_ind = np.argmax(
            kepler_contam_sources["jAperMag3"])

        magdiffs[i] = (
            apogee_sources["jAperMag3"][target_ind] - 
            kepler_contam_sources["jAperMag3"][brightest_contam_ind])

    return magdiffs

def select_tidally_synchronized_binaries(table):
    '''Cuts out the objects that are potentially TSBs.

    This function provides a standardized way to select a sample of Tidally
    Synchronized Binaries according to the prescription of Jen van Saders. This
    function may evolve as TSB selection criteria improve; however, for a
    standard, transparent selection, this will do.

    The current criteria are that TSBs have orbital periods of around 3 days,
    and effective temperatures between 5700 and 4600 K.
    '''
    period_cut = perform_period_cut(table, lowperiod=1, highperiod=5, 
                                    periodcol="Prot")
    try:
        temp_cut = perform_teff_cut(period_cut, 4850, 5600, "TEFF_FIT")
    except KeyError:
        temp_cut = perform_teff_cut(period_cut, 4850, 5600, "Teff")

    return temp_cut


def progress_plot():
    '''Plots the various subclasses of objects so that they can be easily
    figured out.

    Plots: APOKASC & McQuillan
    Eclipsing binaries?
    Plots: Overlap sample
    Plots: Teff & Period cuts
    Plots: Pulsators
    Finally plots: Sample
    '''
    apokasc = read_APOKASC_catalog()
    print("Objects in APOKASC Catalog: {0}".format(len(apokasc)))
    mcquillan = read_McQuillan_catalog()
    print("Objects in McQuillan: {0}".format(len(mcquillan)))

    hr.logg_teff_plot(apokasc["TEFF_FIT"], apokasc["LOGG_FIT"], style="k.",
                      label="Full APOKASC")
    hr.logg_teff_plot(mcquillan["Teff"], mcquillan["log_g_"], style="c.",
                      label="Full Mcquillan")

    joint_catalog = create_joined_APOKASC_McQuillan_catalog(apokasc, mcquillan)
    print("Overlap between APOKASC and McQuillan is: "
          "{0}".format(len(joint_catalog)))
    period_teff_cut = perform_period_cut(
        perform_teff_cut(joint_catalog, lowtemp=4600, hightemp=6200),
        highperiod=3)
    joint_catalog = au.get_complement_table(period_teff_cut, joint_catalog,
                                         "KEPLER_INT")
    print("Number of objects satisfying 4600 K <= Teff <= 6200 K and P < 3 "
          "days is: {0}".format(len(period_teff_cut)))
    hr.logg_teff_plot(joint_catalog["TEFF_FIT"], joint_catalog["LOGG_FIT"],
                      style="wo", ms=2, label="Overlap")

    no_pulsators = filter_pulsators(period_teff_cut, quiet=True)
    pulsators = au.get_complement_table(no_pulsators, period_teff_cut,
                                        "KEPLER_INT")
    print("After removing pulsators, the sample left contains: "
          "{0}".format(len(no_pulsators)))
    hr.logg_teff_plot(pulsators["TEFF_FIT"], pulsators["LOGG_FIT"], style="ro",
                      ms=4, label="Pulsators")
    hr.logg_teff_plot(no_pulsators["TEFF_FIT"], no_pulsators["LOGG_FIT"],
                      style="bD", ms=6, label="Candidates")

def EB_plot(eb_periods, eb_vs, eb_flags):
    """Plots the eclipsing binary period vs velocity.

    This function plots the period as determined from Kepler light curves vs
    the vscatter calculated from APOGEE. In general, sin i ~ 1 for this case. 
    This plot functions to note whether VSCATTER is a useful determinant of
    whether an object is a tidally-synchronized binary.

    VSCATTER by itself is generally not a good measure. However, I would now
    like to see whether VMAX = VSCATTER * NOBS would be.
    """
    bad_indices = bad_ASPCAP_indices(eb_flags)
    good_indices = np.logical_not(bad_indices)

    # Plot the data.
    plt.plot(eb_periods[bad_indices], eb_vs[bad_indices], 'ro')
    plt.plot(eb_periods[good_indices], eb_vs[good_indices], 'ko')

    # Plot the lines.
    periodrange = np.linspace(0.01, 12, 100) * u.day
    standard_vel = bc.calc_velocity_of_binary(1*u.solMass, periodrange, 0.5)
    highmass_vel = bc.calc_velocity_of_binary(2*u.solMass, periodrange, 0.5)
    lowmass_vel = bc.calc_velocity_of_binary(0.5*u.solMass, periodrange, 0.5)
    highratio_vel = bc.calc_velocity_of_binary(1*u.solMass, periodrange, 1.0)
    lowratio_vel = bc.calc_velocity_of_binary(1*u.solMass, periodrange, 0.1)
    plt.plot(
        periodrange.to(u.day).value, standard_vel.to(u.km/u.s).value, 'k--',
        label="M=1, q=0.5")
    plt.plot(
        periodrange.to(u.day).value, highmass_vel.to(u.km/u.s).value, 'k--')
    plt.plot(
        periodrange.to(u.day).value, lowmass_vel.to(u.km/u.s).value, 'k--')
    plt.plot(
        periodrange.to(u.day).value, lowratio_vel.to(u.km/u.s).value, 'b--',
        label="M=1, q=0.1")
    plt.plot(
        periodrange.to(u.day).value, highratio_vel.to(u.km/u.s).value, 'r--',
        label="M=1, q=1.0")

    plt.xlabel("Period (day)")
    plt.ylabel("VMAX (km/s)")

    plt.xlim(0, 12)

def radial_velocity_tides_contour(combined_mass, tidal_limit=5*u.day):
    '''Contour of RVs for given mass ratio and period.

    A combined mass has to be given as an astropy mass unit. After that, a
    contour plot will be made indicating the radial velocities corresponding to
    each pair of mass ratio and period.
    '''
    periodrange = np.linspace(0.01, 15, 100)*u.day
    ratiorange = np.linspace(0.01, 1, 100)

    velocities = bc.calc_velocity_of_binary(
        combined_mass, periodrange, ratiorange[:,np.newaxis])

    plt.figure()
    contourlevels = [1, 5, 10, 25, 50, 75, 100, 200]
    CS = plt.contour(periodrange.to(u.day).value, ratiorange,
                     velocities.to(u.km/u.s).value, levels=contourlevels,
                     colors="k")
    plt.clabel(CS, inline=1, fontsize=10)
    plt.xlabel("Period (day)")
    plt.ylabel("Mass ratio")
    plt.title("Velocities for combined mass of {0}".format(combined_mass))

def read_villanova_EBs(
    EBpath="/home/regulus/simonian/Binaries/Villanova_EB_v3.txt"):
    '''Reads in the Villanova Keler EB catalog.'''
    ebcat = Table.read(EBpath, format="ascii.commented_header",
                       header_start=-1)
    return ebcat

###############################################################################
# KepVIM #
###############################################################################

def read_KepVIM_catalog(
    KepVIMpath="/home/regulus/simonian/Binaries/KepVIM.fits"):
    '''Reads in the KepVIM catalog (Makarov & Goldin 2016).

    Catalog contains objects whose centroid positions in Kepler change with
    their variability.'''
    kepvim = Table.read(KepVIMpath, format="fits")
    return kepvim

def filter_by_quarters(kepvimtable, vimquarters, kic_col="KIC"):
    '''Remove objects with fewer than the specified quarters of VIM detections.

    Return a table with only the entries which have at least vimquarters number
    of observations.'''
    filtered_objects = []
    # Using groups takes a REALLY long time. Is there a way to avoid this?
    vimgroups = kepvimtable.group_by(kic_col)
    for kicgroup in vimgroups.groups:
        if len(kicgroup) >= vimquarters:
            filtered_objects.append(kicgroup)
    filtered_table = vstack(filtered_objects)
    return filtered_table

def KICs_with_VIM_quarters(kepvimtable, minquarters, kic_col="KIC"):
    '''Get KIC IDs for objects with VIM detections over minquarters.

    Gets the KIC IDs for all of the KIC objects which have at least minquarters
    VIM detections over the campaign.
    '''
    newvim = filter_by_quarters(kepvimtable, minquarters, kic_col=kic_col)
    kepvimgroup = newvim.group_by(kic_col)
    kictable = au.first_row_in_group(kepvimgroup)
    return kictable[kic_col]



# Also want function that returns just KIC numbers that have VIM in at least n
# quarters.

def compress_kepVIM(kepvimtable, kic_col="KIC"):
    '''Compress each KIC to a unique row.

    The raw organization of the KepVIM catalog has a row for each unique
    combination of KIC and quarter. This function moves the quarter information
    from rows to columns; that way TBD'''
    fullcolnames = kepvimtable.colnames
    vimvariable_colnames = {
        'F50', 'Xpix', 'Ypix', 'r', 'AX', 'AY', 'ds_dF', 'PA', 'Ch', 'Q'}
    remaining_colnames = [fixedcol for fixedcol in fullcolnames if fixedcol not 
                          in vimvariable_colnames]
    unique_table = unique(kepvimtable[remaining_colnames], keys=kic_col)

    for quarter in range(1, 18):
        quarter_colname = "Q{0:d}".format(quarter)
        unique_table[quarter_colname] = np.zeros(len(unique_table))

        # Find all KIC values with VIM detections in a given quarter, and set
        # the corresponding entries in quarter_column to true.
        kic_detection_in_quarter = kepvimtable[kic_col][
            au.astropy_table_index(kepvimtable, "Q", quarter)]
        unique_kic_indices = au.astropy_table_indices(
            unique_table, kic_col, kic_detection_in_quarter)
        unique_table[quarter_colname][unique_kic_indices] = 1

    return unique_table

def kepVIM_quarter_table(kepvimtable, kic_col="KIC"):
    '''Create a table indicating which in quarters each KIC object had VIM.
    
    For the sake of making things sane again, this table has one row for each
    KIC object, and columns for each quarter, indicating in which quarter the 
    KIC object had VIM observations.'''
    quarter_table = Table([kepvimtable[kic_col]])
    colcount = np.zeros(len(quarter_table))

    for quarter in range(1, 18):
        quarter_colname = "Q{0:d}".format(quarter)
        quarter_table[quarter_colname] = np.zeros(len(quarter_table))

        # Find all KIC values with VIM detections in a given quarter, and set
        # the corresponding entries in quarter_column to true.
        kic_detection_in_quarter = kepvimtable[kic_col][
            au.astropy_table_index(kepvimtable, "Q", quarter)]
        unique_kic_indices = au.astropy_table_indices(
            quarter_table, kic_col, kic_detection_in_quarter)
        quarter_table[quarter_colname][unique_kic_indices] = 1
        colcount += quarter_table[quarter_colname]

    quarter_table["Num_Q"] = colcount
    return quarter_table

###############################################################################
# ASPCAP #
###############################################################################

def filter_bad_ASPCAP_fits(apogee_table, warn=False):
    '''Removes entries which have ASPCAP flags.

    If an object has the STAR_BAD flag enabled, it will be removed. If the warn
    keyword is also specified, it will also remove the STAR_WARN flag.
    '''
    flags = apogee_table["ASPCAPFLAGS"]
    good_indices = np.logical_not(bad_ASPCAP_indices(flags, warn))
    newtable = apogee_table[good_indices]
    add_cut_metadata(newtable, "ASPCAP STAR_BAD removed")
    if warn:
        add_cut_metadata(newtable, "ASPCAP STAR_WARN removed")
    return newtable

def bad_ASPCAP_indices(aspcapflags, warn=False):
    '''Picks bad ASPCAP flags from flag array.

    Bad ASPCAP flags are defined as those with STAR_BAD in them. If the warn
    keyword is given, STAR_WARN flags are also marked as bad.
    '''
    bad_indices = npstr.find(aspcapflags, "STAR_BAD") > 0
    if warn:
        bad_indices = np.logical_and(bad_indices, npstr.find(
            aspcapflags, "STAR_WARN") > 0)

    return bad_indices

###############################################################################
# Double-Lined Spectroscopic Binaries #
###############################################################################
DLSB_LIST = [3654549, 6368779, 6381934, 6436652, 8520982]
DLSB_PATH = paths.HOME_DIR / "DLSB.kic"

def filter_double_lined_spectroscopic_binaries(apocat, kiccol="KEPLER_ID"):
    '''Remove the Double-Lined Spectroscopic Binaries stored in DLSB_LIST.'''
    filtered_cat = au.filter_column_from_subtable(apocat, kiccol, DLSB_LIST)
    add_cut_metadata(
        filtered_cat, "Removed DLSBs: see {0} for list".format( DLSB_PATH))
    return filtered_cat

def write_DLSB_list(outputpath=DLSB_PATH):
    '''Write the list of double-lined spectroscopic binaries to a file.'''
    dlsb_table = Table([DLSB_LIST])
    comments = [
        "These are KIC values of APOGEE objects which are double-lined "
        "spectroscopic binaries.", "They do not include objects with the "
        "STAR_BAD flag."]
    dlsb_table.write(
        str(DLSB_PATH), format="ascii.no_header", comment=comments)
    DLSB_PATH.chmod(0o744)

