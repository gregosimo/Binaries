"""Functions for manipulating the APOKASC-McQuillan catalog
"""
import os
import re
import itertools

import numpy as np
import numpy.core.defchararray as npstr
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import Table, join, vstack
from astropy.coordinates import SkyCoord
from bs4 import BeautifulSoup
import requests


import astropy_util as au
import hrplots as hr

WORKPATH = "/home/regulus/simonian/Binaries"

APOKASC_PATH = os.path.join(WORKPATH, "APOKASC_cat_v3.3.5.fits")
MCQUILLAN_PATH = os.path.join(WORKPATH, "McQuillan.fit")

SDSS3_URL = "http://data.sdss3.org"

def read_APOKASC_catalog(filepath=APOKASC_PATH, exclude_single_epoch=False):
    '''Reads in the APOKASC catalog.

    The catalog should be located at filepath.
    '''
    apocat = Table.read(filepath, format="fits")
    if exclude_single_epoch:
        apocat = apocat[np.where(apocat["VSCATTER"] > 0.0)]
        add_cut_metadata(apocat, "VSCATTER > 0")
    return apocat

def read_McQuillan_catalog(filepath=MCQUILLAN_PATH):
    '''Reads in the McQuillan catalog.

    The catalog shoul be located at filepath.
    '''
    mcquillancat = Table.read(filepath, format="fits")
    return mcquillancat

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
        mjds.append(extract_mjd(visitsoup))
        vrels.append(extract_vrel(visitsoup))

    # Return table with MJD and vrel.
    object_table = Table([mjds, vrels], names=("MJD", "V_LSR"))
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

def McQuillan_standout_plot(
    fullsample, rv_nonvar, rv_var, Teff_colname="TEFF_FIT",
    Prot_colname="Prot"):
    '''Creates a plot like McQuillan et al. but overplots RV samples.

    Takes the full McQuillan sample and overplots the RV-variable and
    RV-nonvariable samples on top in red and blue.
    '''
    plt.semilogy(fullsample[Teff_colname], fullsample[Prot_colname], 'c.',
                 label="McQuillan/APOKASC")
    plt.semilogy(rv_var[Teff_colname], rv_var[Prot_colname], 'r*', ms=12, 
                 label="RV Variable")
    plt.semilogy(rv_nonvar[Teff_colname], rv_nonvar[Prot_colname], 'b*', ms=12, 
                 label="RV Nonvariable")
    hr.invert_x_axis()
    plt.xlabel("Teff (K)")
    plt.ylabel("Prot (day)")
    plt.title("Jen van Saders-cut sample (Multiepoch)")
    plt.legend(loc="lower left")

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

def filter_pulsators(fulltable, quiet=False):
    '''Removes known Kepler pulsators from a table of Kepler objects.

    If the quiet keyword is disabled, then this function will print the KIC IDs
    of the objects that were found to be pulsators.
    '''
    filteredtable = au.filter_column_from_subtable(
        fulltable, "KEPLER_INT", read_pulsators()["KIC"])
    if not quiet:
        pulsators = au.get_complement_table(
            filteredtable, fulltable, "KEPLER_INT")["KEPLER_ID"]
        for kic in pulsators:
            print("KIC {0} is a Kepler Pulsator".format(kic))
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

def read_villanova_EBs(
    EBpath="/home/regulus/simonian/Binaries/Villanova_EB_v3.txt"):
    '''Reads in the Villanova Keler EB catalog.'''
    ebcat = Table.read(EBpath, format="ascii.commented_header",
                       header_start=-1)
    del(ebcat["coll11"])
    return ebcat

def filter_bad_ASPCAP_fits(apogee_table, warn=False):
    '''Removes entries which have ASPCAP flags.

    If an object has the STAR_BAD flag enabled, it will be removed. If the warn
    keyword is also specified, it will also remove the STAR_WARN flag.
    '''
    flags = apogee_table["ASPCAPFLAGS"]
    good_indices = npstr.find(flags, "STAR_BAD") == -1
    if warn:
        good_indices = np.logical_and(good_indices, npstr.find(
            flags, "STAR_WARN") == -1)
    newtable = apogee_table[good_indices]
    add_cut_metadata(newtable, "ASPCAP STAR_BAD removed")
    if warn:
        add_cut_metadata(newtable, "ASPCAP STAR_WARN removed")
    return newtable
