"""Functions for manipulating the APOKASC-McQuillan catalog
"""
import os
import re
import itertools
import io
import random

import numpy as np
import numpy.core.defchararray as npstr
import scipy
from scipy.interpolate import interp1d
from scipy.stats import norm, uniform, ks_2samp
from scipy.special import gammaincc, erf
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import astropy.units as u
from astropy.table import Table, join, vstack, unique, Column
from astropy.coordinates import SkyCoord
from astropy.convolution import convolve, Gaussian1DKernel
from bs4 import BeautifulSoup
import requests
# from apogee.tools import bitmask

import astropy_util as au
import statop as stat
import hrplots as hr
import binarycalcs as bc
import path_config as paths
import sed
import browse_APOGEE_spectra as browse
import read_catalog as catin
import rotation_consistency as rot

SDSS3_URL = "http://data.sdss3.org"

NUM_KEPLER_QUARTERS = 17

APOGEE_NULL = -9999.0

DLSB_PATH = browse.DEFAULT_DLSB_DB
NON_DLSB_PATH = browse.DEFAULT_NULL_DB

################################################################################
# Online catalog interactions #
################################################################################

################################################################################
# Table Manipulation #
################################################################################

#############
# 2MASS IDs #
#############

def join_by_2MASS_key(tbl1, tbl2, tm1, tm2, join_type="inner",
                      skip_missing=True, conflict_suffixes=("_A", "_B")):
    '''Join two tables by their 2MASS key.

    Join tables according to a 2MASS key. Since different catalogs store the
    2MASS ID in different ways ways, this will be able to distinguish between
    them transparently. Currently this can interchange the APOGEE and KIC
    formats.
    '''
    kic_prefix = "2MASS J"
    apogee_prefix = "2M"

    try:
        tbl1 = tbl1[~tbl1[tm1].mask]
    except AttributeError:
        pass

    try:
        tbl2 = tbl2[~tbl2[tm2].mask]
    except AttributeError:
        pass

    if np.any(npstr.startswith(tbl1[tm1], kic_prefix)):
        tbl1_type = "KIC"
    elif np.any(npstr.startswith(tbl1[tm1], apogee_prefix)):
        tbl1_type = "APOGEE"
    else:
        raise ValueError("Don't recognize 2MASS key: " + tbl1[tm1][0])

    if np.any(npstr.startswith(tbl2[tm2], kic_prefix)):
        tbl2_type = "KIC"
    elif np.any(npstr.startswith(tbl2[tm2], apogee_prefix)):
        tbl2_type = "APOGEE"
    else:
        raise ValueError("Don't recognize 2MASS key: " + tbl2[tm2][0])

    if tbl1_type == tbl2_type:
        new_table = au.join_by_id(
            tbl1, tbl2, tm1, tm2, join_type=join_type, idproc=npstr.strip, 
            conflict_suffixes=conflict_suffixes)
        print("Types the same")
    else:
        print("Types different")
        def transform(oldcol):
            if tbl1_type == "KIC" and tbl2_type == "APOGEE":
                newcol = npstr.replace(oldcol, apogee_prefix, kic_prefix)
            elif tbl1_type == "APOGEE" and tbl2_type == "KIC":
                newcol = npstr.replace(oldcol, kic_prefix, apogee_prefix)
            return newcol
        tbl2_oldcol = tbl2[tm2]
        random_colname = au.generate_random_string(12)
        tbl2_newcol = transform(tbl2_oldcol)
        del(tbl2[tm2])
        try:
            tbl2[tm2] = tbl2_newcol
            tbl2[random_colname] = tbl2_oldcol
            new_table = au.join_by_id(
                tbl1, tbl2, tm1, tm2, join_type=join_type, idproc=npstr.strip, 
                conflict_suffixes=conflict_suffixes)
        finally:
            del(tbl2[tm2])
            tbl2[tm2] = tbl2_oldcol

        del(new_table[tm2])
        new_table.rename_column(random_colname, tm2)

    # Find a way to get tbl2_oldcol back in the table. This may require
    # renaming tbl2 instead of deleting it.
    return new_table

def KIC_to_APOGEE_2MASS_designation(kic_desig):
    '''Function to convert KIC 2MASS designations to be APOGEE ones.

    The KIC designations are in the form of 2MASS J##########, while the apogee
    ones are 2M##########.'''
    apo_desig = npstr.replace(kic_desig, "2MASS J", "2M")
    return apo_desig

#####################
# Kepler Photometry #
#####################

def add_Everett_photometry(inputtable, racol, deccol):
    '''Adds UBV photometry from the EHK survey to table.

    Read in the EHK photometry and append that to the columns in inputtable.
    The table names will be the same as those in the photometry file.'''
    photcatalog = catin.read_EHK_catalog()
    newcat = au.join_by_ra_dec(
        inputtable, photcatalog, racol, deccol, "RA", "Dec", join_type="left")

    return newcat

###############################################################################
# Catalog Curation
###############################################################################

def select_samples_for_Rafa(
    kic_catalog=None, hightemp=5500, lowtemp=0, loggcut=3.5, teffcol="teff",
    loggcol="logg"):
    '''Select a sample of objects for Rafa to analyze.

    The default cuts for Rafa will be the requirements:
    T < 5500 K
    logg > 3.5
    '''
    if not kic_catalog:
        kic_catalog = read_KIC_DR25_catalog()
    tempcut = perform_teff_cut(kic_catalog, lowtemp=lowtemp, hightemp=hightemp,
                               teffcol=teffcol)
    giantcut = perform_logg_cut(tempcut, lowlogg=3.5, loggcol=loggcol)
    rafacat = giantcut
    return rafacat

# Default methodology to select tidally-synchronized binaries.

def select_tidally_synchronized_binaries(
    table, pcut=5, lowtemp=4850, hightemp=5600, lowperiod=1, logg=3.5, 
    teffcol="Teff", pcol="Prot", loggcol="logg"):
    '''Cuts out the objects that are potentially TSBs.

    This function provides a standardized way to select a sample of Tidally
    Synchronized Binaries according to the prescription of Jen van Saders. This
    function may evolve as TSB selection criteria improve; however, for a
    standard, transparent selection, this will do.

    The current criteria are that TSBs have orbital periods of around 5 days,
    and effective temperatures between 5700 and 4600 K.
    '''
    period_cut = perform_period_cut(
        table, lowperiod=lowperiod, highperiod=pcut, periodcol=pcol)
    temp_cut = perform_teff_cut(period_cut, lowtemp, hightemp, teffcol)
    loggcut = perform_logg_cut(temp_cut, lowlogg=logg, loggcol=loggcol)

    return loggcut

#################
# Split Catalog #
#################

def split(fullsample, col, splitpoints, invert_inequality=False):
    """Split the full sample based on values in col.

    This function partitions fullsample into tables with points in splitpoints
    serving as boundaries of the partitions. A tuple of size one larger than
    splitpoints will be returned which contain values where
    fullsample[col] < min(splitpoints), min(splitpoints) <= fullsample[col] <
    min-1(splitpoints), ... , max-1(splitpoints) <= fullsample[col] <
    max(splitpoints), fullsample[col] > max(splitpoints). 

    If invert_inequality is specified, then the picked values with follow low <
    fullsamplecol <= high.
    """
    colvalues = fullsample[col]

    try:
        ordered_splitpoints = sorted(splitpoints)
    except TypeError:
        # In this case there is only one object.
        ordered_splitpoints = [splitpoints]

    objlist = []
    # Invert_inequality basically transforms < to <= and >= to >
    if not invert_inequality:
        # First make the lowest table.
        objlist.append(fullsample[colvalues < ordered_splitpoints[0]])
        # Then make intermediate tables.
        for low, high in zip(ordered_splitpoints[:-1], ordered_splitpoints[1:]):
            objlist.append(fullsample[np.logical_and(
                colvalues >= low, colvalues < high)])
        # Now make the highest table.
        objlist.append(fullsample[colvalues >= ordered_splitpoints[-1]])
    else:
        objlist.append(fullsample[colvalues <= ordered_splitpoints[0]])
        for low, high in zip(ordered_splitpoints[:-1], ordered_splitpoints[1:]):
            objlist.append(fullsample[np.logical_and(
                colvalues > low, colvalues <= high)])
        objlist.append(fullsample[colvalues > ordered_splitpoints[-1]])

    return tuple(objlist)

def split_logg(fullsamp, loggs, loggcol="logg", invert_inequality=False):
    '''Split sample into logg bins.'''
    return split(fullsamp, loggcol, loggs, invert_inequality)

def split_teff(fullsamp, teffs, teffcol="teff", invert_inequality=False):
    '''Split sample into teff bins.'''
    return split(fullsamp, teffcol, teffs, invert_inequality)

def split_period(fullsamp, periods, periodcol="Prot", invert_inequality=False):
    '''Split sample into period bins.'''
    return split(fullsamp, periodcol, periods, invert_inequality)

def split_vscatter(fullsamp, vels, vscattercol="VSCATTER",
                   invert_inequality=False):
    '''Split sample into vscatter parts.'''
    return split(fullsamp, vscattercol, vels, invert_inequality)

def split_vsini(fullsamp, vels, vsinicol="VSINI", invert_inequality=False):
    '''Split sample into vsini parts.'''
    return split(fullsamp, vsinicol, vels, invert_inequality)

def split_spectroscopic_rapid_rotators(
        fullsamp, radii, vsinis, highperiod=5, lowperiod=1,
        invert_inequality=False):
    '''Split rapid rotators spectroscopically.
    
    Rapid rotation is defined based on the rotation period.'''
    fullsamp["TEMPPERIOD"] = vsini_to_period(vsinis, radii)[0]

    psamples = split_period(
        fullsamp, [lowperiod, highperiod], periodcol="TEMPPERIOD", 
        invert_inequality=invert_inequality)

    del(fullsamp["TEMPPERIOD"])
    for samp in psamples:
        del(samp["TEMPPERIOD"])

    return psamples

def split_dlsb(fullsamp, apid_col, dlsb_db=DLSB_PATH, nodl_db=NON_DLSB_PATH):
    '''Split sample into confirmed DLSBs, non-DLSBs, and nonconfirmed.

    This function will essentially look through the DLSB_DB and SLSB_DB to
    separate out targets which are either confirmed DLSBS, confirmed non-DLSBS,
    or objects which haven't been classified as either.'''
    dlsb_indices = mark_DLSB_indices(fullsamp[apid_col], dlsb_db=dlsb_db)
    nodl_indices = mark_non_DLSB_indices(fullsamp[apid_col], nodl_db=nodl_db)
    other_indices = np.logical_not(np.logical_or(dlsb_indices, nodl_indices))
    assert not np.any(np.logical_and(dlsb_indices, nodl_indices))

    return (fullsamp[np.where(dlsb_indices)], fullsamp[np.where(nodl_indices)],
            fullsamp[np.where(other_indices)])
    
##################
# APOGEE filters #
##################

def filter_invalid_APOGEE_entries(apotable, colname, maskvalue=APOGEE_NULL):
    '''Remove rows from apotable where column values are the mask values.

    This will filter apotable where only the rows that do not have the mask
    value in the column will be returned.'''
    filtered_table = au.filter_column_from_subtable(apotable, colname, 
                                                    [maskvalue])
    add_cut_metadata(filtered_table,
        "Removed masked entries in {0}".format(colname))
    return filtered_table

###############################################################################
# Writing to databases #
###############################################################################

def split_filename(filename):
    '''Splits the suffix from the filename.

    Will take a filename and then split it between the base name and the file
    extension. For example, foo.bar becomes ("foo", "bar") or test.tar.gz
    becomes ("test", "tar.gz"). Note that this function splits on the first
    period. So basenames should not have a period in them.'''
    return filename.split(".", maxsplit=1)

def format_split_filename(base, num, ext):
    '''Returns a filename formatted for sequential numbering.

    This filename is currently in the form {base}.{num}.{ext}.
    '''
    return "{0}.{1}.{2}".format(base, num, ext)

def delete_split_files(filename, outputpath):
    '''Delete the split files generated from filename.

    This function finds all files in outputpath that were generated by
    splitting filename and deletes them all.'''
    # Delete old split files before making new ones.
    singlefile = outputpath / filename
    previous_files = find_split_files(singlefile)
    for oldfile in previous_files:
        oldfile.unlink()


def large_table_multiple_files_split(filename, tablelen, maxlen):
    '''Formats input files into many broken files.

    This function essentially splits a filename into a list of filenames that
    are sequentially numbered. For example, if a service has a maximum limit of
    5000 rows, and the table with 12,000 entries was supposed to be saved in 
    a file called "test.txt", this function would output a list with
    [test.0.txt, test.1.txt, test.2.txt]. If the table only had 2,000 entries,
    it would return a single-item list with [test.txt]. 
    
    Note that this function splits on the first ".", so filenames periods 
    before the extension will likely not behave as intended.
    '''
    filenames = []
    if tablelen <= maxlen:
        filenames.append(filename)
    else:
        name, ext = split_filename(filename)
        for i in range((tablelen-1) // maxlen + 1):
            numberedname = format_split_filename(name, i, ext)
            filenames.append(numberedname)
    return filenames

def test_split():
    filename = "test.txt"
    assert (large_table_multiple_files_split(filename, 0, 3) == ["test.txt"])
    assert (large_table_multiple_files_split(filename, 1, 3) == ["test.txt"])
    assert (large_table_multiple_files_split(filename, 3, 3) == ["test.txt"])
    assert (large_table_multiple_files_split(filename, 4, 3) == [
        "test.0.txt", "test.1.txt"])
    assert (large_table_multiple_files_split(filename, 6, 3) == [
        "test.0.txt", "test.1.txt"])

def write_columns_for_input(outputtable, filename, maxlen, table_format,
                            output_columns=None, outputpath=paths.HEAD_DIR):
    '''Write a table so that it can be uploaded to a service.

    This function takes a table that should be written out to a (series of)
    file(s) given in filename to be uploaded to a catalog. The outputtable 
    should have only those columns that need to be written out. If the columns 
    need to be renamed, then the names should be supplied in output_columns. 
    The format of the table should be specified as table_format.
    
    If the length of of the table is larger than maxlen, then this function
    will write several files, each containing at most maxlen entries. This is
    useful for when catalogs have limits on how many entries can be given. In
    this case, the filename will be split up to incorporate counters.'''
    # Delete old split files before making new ones.
    delete_split_files(filename, outputpath)

    filenames = large_table_multiple_files_split(filename, len(outputtable),
                                                 maxlen)
    for i, outputfile in enumerate(filenames):
        startind = maxlen * i
        endind = min(maxlen * (i+1), len(outputtable))
        outputsegment = outputtable[startind:endind]
        outputsegment.write(str(outputpath / outputfile), format=table_format,
                            include_names=output_columns)

def write_KIC_Vizier_upload_list(kics, outputfile, outputpath=paths.HEAD_DIR):
    '''Write a file to be uploaded to Vizier. 

    This will make a list of KIC identifiers which should be resolved by
    Vizier.'''
    kicstr = kics.astype(np.str)
    kic_column = npstr.add("KIC ", kicstr)
    kic_table = Table([kic_column], names=["KIC"])
    # I don't know the input limit yet.
    write_columns_for_input(
        kic_table, outputfile, 99999, "ascii.no_header", output_columns=["KIC"], 
        outputpath=outputpath)
    kic_table.write(str(outputpath / outputfile), format="ascii.no_header")

def write_SIMBAD_identifier_list(
    identifiers, outputfile, outputpath=paths.HEAD_DIR):
    '''Write a list that can be uploaded to SIMBAD as a list of identifiers.'''
    ident_table = Table([identifiers], names=["Ident"])
    write_columns_for_input(
        ident_table, outputfile, 99999, "ascii.no_header",
        output_columns=["Ident"], outputpath=outputpath)

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
    write_columns_for_input(outputtable[[kiccol]], output_filename, MAST_LIMIT,
                            "ascii.no_header")

def write_SDSS_crossID_file(
    outputtable, racol="RA", deccol="DEC", outputpath=paths.HEAD_DIR, 
    output_filename="APOGEE_targets.txt"):
    '''Writes a file to submit to SDSS crossID.

    This file can be used to submit to:
    http://skyserver.sdss.org/dr13/en/tools/crossid/crossid.aspx
    '''
    APOGEE_LIMIT = 1000
    filenames = large_table_multiple_files_split(
        output_filename, len(outputtable), APOGEE_LIMIT)
    for i, outputfile in enumerate(filenames):
        startind = APOGEE_LIMIT * i
        endind = min(APOGEE_LIMIT*(i+1), len(outputtable))
        outputfile = str(outputpath / output_filename.format(i))
        names = ["ra", "dec"]
        outputsegment = outputtable[startind:endind]
        outputsegment[[racol, deccol]].write(outputfile, format="ascii.csv", 
                                             names=names)

def write_UKIRT_file(
    catalogtable, racol="RA", deccol="DEC", outputpath=paths.HEAD_DIR, 
    output_filename="ukirt_input.txt", search_radius=30):
    '''Takes RA and DEC columns from catalog to make an upload file.'''
    if search_radius >= 5:
        UKIRT_LIMIT = 5000
    else:
        UKIRT_LIMIT = 50000
    coords = SkyCoord(
        ra=catalogtable[racol], dec=catalogtable[deccol], unit="deg")
    coordTable = Table(
        [coords.ra.degree, coords.dec.degree], names=("RA", "DEC"))
    write_columns_for_input(coordTable, output_filename, UKIRT_LIMIT,
                            "ascii.no_header")

def write_CasJob_file(
    catalog, outputfile, outputpath=paths.HEAD_DIR, writefmt="ascii.csv"):
    '''Write a table that can be submitted to CasJobs.'''
    if writefmt not in ["ascii.csv", "ascii.votable"]:
        raise ValueError("CasJobs only supports CSV and VOTable formats.")
    fullfile = outputpath / outputfile
    catalog.meta["comments"] = []
    catalog.write(str(fullfile), format=writefmt)
    # This is the size in kb
    filesize = fullfile.stat().st_size / 2**10
    if ((writefmt == "ascii.csv" and filesize > 256000) or
        (writefmt == "ascii.votable" and filesize > 1048576)):
        raise ValueError("Catalog too big to write.")

def write_McQuillan_Observed_KICs_to_CasJobs(
    kiccat, outputfile="mcquillan_observed.txt", outputpath=paths.HEAD_DIR,
    writefmt="ascii.csv", kiccol="kepid"):
    '''Write the targets which have been observed in Q3-Q14.

    For a catalog from the KIC Stellar Parameter database, this will look
    through the observations and determine which subset of the catalog has been
    observed at least once between Q3 and Q14, as specified by McQuillan et al
    (2014).'''
    observed_targets = kiccat[np.logical_not(npstr.startswith(
        kiccat["st_quarters"], "000000000000", start=2))]
    write_CasJob_file(observed_targets[[kiccol]], outputfile, outputpath, 
                      writefmt)

def write_Villanova_EB_upload_list(
    kiccat, outputfile="villanova_upload.txt", outputpath=paths.HEAD_DIR,
    writefmt="ascii.no_header", kiccol="KIC"):
    '''Write KIC numbers to be uploaded to the Villanova EB catalog.'''
    output = outputpath / outputfile
    kic_targets = kiccat[[kiccol]]
    kic_targets.write(str(output), format=writefmt, comment=False)

# Interact with the SDSS database

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





###############################################################################
# APOGEE Figures #
###############################################################################

ASPCAP_STAR_BAD = 2**23
ASPCAP_STAR_WARN = 2**7
ASPCAP_VSINI_WARN = 2**14
NO_ASPCAP_RESULT = 2**31

def plot_by_ASPCAP_quality(x, y, aspcapflags, **kwargs):
    '''Distinguish between ASPCAP quality for plotting quantities.

    The arguments for this function are the x value, yvalue, and the aspcap
    flags necessary to determine which points are good, warn, and bad. 

    By default, a target is marked bad if the STAR_BAD flag is enabled, is
    marked warn if the STAR_WARN flag or VSINI_WARN flag is enabled. These
    can be overridden by passing arguments warn_flags and bad_flags. The "flag"
    should be an integer in the form of 2**digit, where the digit specified in
    the SDSS DR13 bitmask page. For convenience, some commonly-used flags are 
    included in this module prepended by ASPCAP. To specify multiple flags, the 
    flags have to be added. For example, to specify both the STAR_WARN and 
    VSINI_WARN flags, you would specify 
    warn_flags=ASPCAP_STAR_WARN+ASPCAP_VSINI_WARN. 
    '''
    # Customization of the bad and warn flags
    bad_flags = kwargs.pop("bad_flags", ASPCAP_STAR_BAD)
    warn_flags = kwargs.pop("warn_flags", ASPCAP_STAR_WARN + ASPCAP_VSINI_WARN)

    # Find which indices correspond to good, warn, and bad entries.
    bad_indices = aspcapflags & bad_flags != 0
    # The 2**14 is a flag called VSINI_WARN. It does not trigger the STAR_BAD
    # or STAR_WARN flags.
    warn_indices = np.logical_and(aspcapflags & warn_flags != 0,
                                 np.logical_not(bad_indices))
    good_indices = np.logical_not(warn_indices)

    # Dealing with errors.
    yerr = kwargs.pop("yerr", None)
    xerr = kwargs.pop("xerr", None)

    if np.any(good_indices):
        good_x = x[good_indices]
        good_y = y[good_indices]
        if yerr is not None:
            try:
                good_yerr = yerr[good_indices]
            except TypeError:
                good_yerr = None
            try:
                good_xerr = xerr[good_indices]
            except TypeError:
                good_xerr = None
        else:
            good_yerr=None
            good_xerr=None
        good_kwargs = dict(
            kwargs, yerr=good_yerr, xerr=good_xerr, ms=7, c="g", marker="o", 
            label="Good", ls="")
        plt.errorbar(good_x, good_y, **good_kwargs)

    if np.any(warn_indices):
        warn_x = x[warn_indices]
        warn_y = y[warn_indices]
        if yerr is not None:
            try:
                warn_yerr = yerr[warn_indices]
            except TypeError:
                warn_yerr = None
            try:
                warn_xerr = xerr[warn_indices]
            except TypeError:
                warn_xerr = None
        else:
            warn_yerr=None
            warn_xerr=None
        warn_kwargs = dict(
            kwargs, yerr=warn_yerr, xerr=warn_xerr, ms=4, c="m", marker="s", 
            label="Warn", ls="")
        plt.errorbar(warn_x, warn_y, **warn_kwargs)

    if np.any(bad_indices):
        bad_x = x[bad_indices]
        bad_y = y[bad_indices]
        if yerr is not None:
            try:
                bad_yerr = yerr[bad_indices]
            except TypeError:
                bad_yerr = None
            try:
                bad_xerr = xerr[bad_indices]
            except TypeError:
                bad_xerr = None
        else:
            bad_yerr=None
            bad_xerr=None
        bad_kwargs = dict(
            kwargs, yerr=bad_yerr, xerr=bad_xerr, ms=15, c="r", marker="D", 
            label="Bad", ls="")
        bad_kwargs["yerr"] = bad_yerr
        bad_kwargs["xerr"] = bad_xerr
        plt.errorbar(bad_x, bad_y, **bad_kwargs)

################################################################################
# Web Scraping
################################################################################

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

# CLEANUP

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
    vsini, period, radii, raderr_below, raderr_above, subgiant_indices):
    '''Evaluate rotation quality in velocity and radius space.

    Create a double-paneled figure that plots the same data in velocity space
    and radius space for clarity of understanding.'''
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)

    valid_indices = vsini > 7
    valid_vsini = vsini[valid_indices]
    valid_period = period[valid_indices]
    valid_radii = radii[valid_indices]
    valid_raderr_above = raderr_above[valid_indices]
    valid_raderr_below = raderr_below[valid_indices]
    valid_subgiant_indices = subgiant_indices[valid_indices]
    valid_dwarf_indices = au.get_complement_indices(
        valid_subgiant_indices, len(valid_subgiant_indices))

    downvel, infvel, upvel = period_to_velocities_uncertainties(
        valid_period, valid_radii, valid_raderr_below, valid_raderr_above)

    ax1.errorbar(
        infvel[valid_subgiant_indices], valid_vsini[valid_subgiant_indices],
        xerr=[-downvel[valid_subgiant_indices], upvel[valid_subgiant_indices]],
        yerr=0.1*valid_vsini[valid_subgiant_indices], fmt='b*',
        label="Subgiants")
    ax1.errorbar(
        infvel[valid_dwarf_indices], valid_vsini[valid_dwarf_indices],
        xerr=[-downvel[valid_dwarf_indices], upvel[valid_dwarf_indices]],
        yerr=0.1*valid_vsini[valid_dwarf_indices], fmt='ro', label="Dwarfs")
    ax1.plot([0, 80], [0, 80], 'k-')
    ax1.plot([0, 80], [7, 7], 'r--', label="Detection Limit")
    plt.sca(ax1)
#    plt.legend(loc="upper right")
    ax1.set_xlabel("Inferred equatorial velocity (km/s)")
    ax1.set_ylabel("V sini (km/s)")

    inferred_radii = rotation_radius(valid_vsini, valid_period)
    ax2.errorbar(
        valid_radii[valid_subgiant_indices], 
        inferred_radii[valid_subgiant_indices], 
        yerr=0.1*inferred_radii[valid_subgiant_indices],
        xerr=[-valid_raderr_below[valid_subgiant_indices],
              valid_raderr_above[valid_subgiant_indices]], fmt='b*')
    ax2.errorbar(
        valid_radii[valid_dwarf_indices], 
        inferred_radii[valid_dwarf_indices], 
        yerr=0.1*inferred_radii[valid_dwarf_indices],
        xerr=[-valid_raderr_below[valid_dwarf_indices],
              valid_raderr_above[valid_dwarf_indices]], fmt='ro')
    ax2.plot([0, 4.0], [0, 4.0], 'k-')
    ax2.set_xlabel("Radius (Rsun)")
    ax2.set_ylabel("Inferred R sini (Rsun)")

    inferred_period = vsini_to_period(valid_vsini, valid_radii)[1]
    # Dealing with errors is difficult because the vsini and radius errors have
    # a unspecified interplay, especially since radius is asymmetric. Instead
    # of trying to calculate some form, I'll take the maximum of either the
    # vsini error or the radius error.
    radius_fractional_errors = ((valid_raderr_above + valid_raderr_below) / 
                                valid_radii)
    vsini_fractional_errors = 0.1
    inferred_period_err_up = np.where(
        radius_fractional_errors >= vsini_fractional_errors, 
        vsini_to_period(
            valid_vsini, valid_radii + valid_raderr_above)[1] - inferred_period, 
        vsini_to_period(
            valid_vsini*(1-vsini_fractional_errors), valid_radii)[1] - inferred_period)
    inferred_period_err_down = np.where(
        radius_fractional_errors >= vsini_fractional_errors, 
        vsini_to_period(
            valid_vsini, valid_radii + valid_raderr_below)[1] - inferred_period, 
        vsini_to_period(
            valid_vsini*(1+vsini_fractional_errors), valid_radii)[1] - inferred_period)
    ax3.errorbar(
        valid_period[valid_subgiant_indices],
        inferred_period[valid_subgiant_indices],
        yerr=[-inferred_period_err_down[valid_subgiant_indices],
              inferred_period_err_up[valid_subgiant_indices]], fmt='b*')
    ax3.errorbar(
        valid_period[valid_dwarf_indices],
        inferred_period[valid_dwarf_indices],
        yerr=[-inferred_period_err_down[valid_dwarf_indices],
              inferred_period_err_up[valid_dwarf_indices]], fmt='ro')

    ax3.plot([0, 15.0], [0, 15.0], 'k-')
    ax3.set_xlabel("McQuillan Period (day)")
    ax3.set_ylabel("Inferred P / sin(i) (day)")
                                      

def write_asteroseismic_rotation_table(
        table, output_filename, title,  apid_col="APOGEE_ID", KICcol="KIC", 
        Teffcol="TEFF", logg_col="LOGG_DW", vsini_col="VSINI_DR14", 
        radius_col="RADIUS_DW", periodcol="Prot_DR14", 
        aspcapcol="ASPCAPFLAGS_DR14", starcol="STARFLAGS", 
        outputpath=paths.HEAD_DIR):
    '''Write the table with quantities relevant to rotation.'''
    output_table = table[[apid_col, KICcol, Teffcol, logg_col, vsini_col,
                          radius_col, periodcol, aspcapcol, starcol]]
    output_table.add_column(table["FPARAM"][:,1], index=3)
    names = ("APOGEE ID", "KIC", "Teff", "Spec Log(g)", "Ast. Log(g)", "vsini",
             "Ast. Radius", "Rot. Period", "ASPCAP Flags", "Star Flags")
    
    output_table.write(str(outputpath / output_filename), format="ascii.aastex",
                       names=names, latexdict={"caption": title})

def write_rotation_debug_table(
    tbl, outfile, title, label, apid_col="APOGEE_ID", KICcol="KIC",
    apogee_teff="TEFF", huber_teff="teff", huber_logg="logg", vsini="VSINI",
    radius="radius", period="Prot", aspcapcol="ASPCAPFLAGS",
    outputpath=paths.HEAD_DIR):
    '''Write a table to output rotation information.'''
    output_table = tbl[["APOGEE_ID", "KIC", "TEFF", "teff", "logg", "VSINI",
                        "radius", "Prot", "ASPCAPFLAGS"]]
    output_table.add_column(tbl["FPARAM"][:,1], index=4)
    names=("APOGEE ID", "KIC", "Spec Teff", "Huber Teff", "Spec Log(g)", 
           "Huber Log(g)", "vsini", "Huber Radius", "Rot. Period", 
           "ASPCAP Flags")
    output_table.write(
        str(outputpath / outfile), format="ascii.aastex", names=names, 
        latexdict={"caption": title + "\\\\label{{table:{0}}}".format(label), 
                   "preamble": r"\tabletypesize{\footnotesize}"})

def write_table_for_period_people(tbl, outfile, outputpath=paths.HEAD_DIR):
    '''Write a table for relevant parameters for period people.'''
    # APOGEE_ID, KIC, Huber Teff, Huber Log(g), Huber radius, McQuillan P, 
    # Spec Teff, Spec Log(g), VSINI, Predicted Pmin, Predicted Pmax
    output_table = tbl[["KIC", "APOGEE_ID", "teff", "logg", "radius", "Prot",
                        "TEFF", "VSINI"]]
    output_table.add_column(tbl["FPARAM"][:,1], index=7)
    lowp, midp, highp = vsini_to_period(
        output_table["VSINI"], output_table["radius"])
    output_table["Pmin"] = lowp                                                    
    output_table["Pmax"] = highp
    names = (
        "KIC", "APOGEE ID", "Huber Teff", "Huber Log(g)", "Huber Radius",
        "McQuillan P", "APOGEE Teff", "APOGEE Log(g)", "VSINI", "Minimum per.",
        "Maximum per.")
    output_table.write(str(outputpath / outfile), format="ascii.fixed_width",
                       names=names)

def write_table_for_spec_people(tbl, outfile, outputpath=paths.HEAD_DIR):
    '''Write a table with relevant parameters for spectroscopic people.'''
    # APOGEE_ID, LOCATION_ID, KIC, Spec Teff, Spec Log(g), VSINI, ASPCAPFLAGS,
    # STARFLAGS, McQuillan P, Equatorial V.
    output_table = tbl
    output_table.write(str(outputpath / outfile), format="ascii.basic")

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
        vsini != APOGEE_NULL, teff != APOGEE_NULL, xvalue != APOGEE_NULL,
        metallicity != APOGEE_NULL, alpha != APOGEE_NULL)
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

    max_vsini_period, rep_vsini_period, min_vsini_period = vsini_to_period(
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

    high_period, rep_period, low_period = vsini_to_period(vsini, radii)

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

def vsini_to_period(vsini, radii):
    '''Converts vsinis to predicted periods using radii.
    
    Returns a 3-tuple containing the high-limit to the period, the 
    representative period, and the low-limit to the period.'''
    km_per_sec_to_solRad_per_day = 1e5 / 7e10 *60*60*24
    representative_norm = np.sin(np.pi/4)

    max_vsini_period = 2 * np.pi * radii / (vsini  *
                                            km_per_sec_to_solRad_per_day)
    representative_vsini_period = representative_norm * max_vsini_period
    vsini_period_range = 0.5 * max_vsini_period

    return max_vsini_period, representative_vsini_period, vsini_period_range

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

def rotation_radius(vsini, prot, vsini_mask=APOGEE_NULL):
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

def DSEP_dwarf_radii(teffs, metallicities, alphas, age=2.0, lowTeff=3000,
                     highTeff=7000):
    '''Calculate MS radii predicted from DSEP.
    
    Using an isochrone of a given age, calculate the radius of a star given an
    effective temperature, metallicity, and alpha abundance.'''
    # This may be complicated, so I wanna take it slow.
    masked_teffs = np.ma.masked_equal(teffs, APOGEE_NULL)
    masked_metallicities = np.ma.masked_equal(metallicities, APOGEE_NULL)
    masked_alphas = np.ma.masked_equal(alphas, APOGEE_NULL)
    radius_mask = au.multi_logical_or(
        masked_teffs.mask, masked_metallicities.mask, masked_alphas.mask)
    model_radii = np.ma.zeros(len(masked_teffs))
    model_radii.mask = radius_mask

    # These are the alpha/Fe bins that will be fed into DSEP.
    alpha_binedges = np.arange(-0.1, 0.9, 0.2)
    # a/Fe < -0.1 corresponds to 1, and a/Fe > 0.7 corresponds to 6.
    alpha_bins = np.digitize(masked_alphas, alpha_binedges)+1
    # DSEP should crash or something if the metallicity and alpha enhancement
    # are not compatible. In particular, high alpha enhancements are only
    # available for low metallicity stars. I want to ensure that this will be
    # the case before running into weird DSEP bugs.
    assert(np.all(np.logical_or(alpha_bins < 4, np.logical_and(
        alpha_bins >= 4, masked_metallicities <= 0.0))))

    rounded_metallicities = np.around(masked_metallicities, 2)
    for i in range(len(model_radii)):
        if not model_radii.mask[i]:
            interp = sed.teff_to_radius_DSEP_interpolator(
                age=age, metallicity=rounded_metallicities[i], afe=alpha_bins[i],
                lowT=lowTeff, highT=highTeff)
            model_radii[i] = 10**interp(np.log10(masked_teffs[i]))

    return model_radii

def compare_DSEP_radii_to_KIC_radii(
    kic_radii, DSEP_radii, kic_radii_err_high, kic_radii_err_low, teff):
    '''Plot the fractional difference between DSEP radii and KIc radii.

    Calculate the fractional difference between the radii of the KIC and that
    calculated by DSEP. This will include the uncertainties of the KIC
    radius.'''
    frac = (kic_radii - DSEP_radii) / DSEP_radii
    fracerrs = [kic_radii_err_low / DSEP_radii, 
                kic_radii_err_high / DSEP_radii]

    print(frac)
    print(fracerrs)

    plt.errorbar(teff, frac, yerr=fracerrs, marker='o', color="b", ls="")
    hr.invert_x_axis()
    plt.xlabel("APOGEE Teff (K)")
    plt.ylabel("Fractional KIC - DSEP radius")

def DSEP_logg(teffs, metallicities, alphas, age=2.0, lowTeff=3000,
              highTeff=7000):
    '''Calculate logg predicted from DSEP.

    Using an isochrone of a given age, calculate the log(g) of a star given an
    effective temperature, metallicity, and alpha abundance.'''
    # This may be complicated, so I wanna take it slow.
    masked_teffs = np.ma.masked_equal(teffs, APOGEE_NULL)
    masked_metallicities = np.ma.masked_equal(metallicities, APOGEE_NULL)
    masked_alphas = np.ma.masked_equal(alphas, APOGEE_NULL)
    logg_mask = au.multi_logical_or(
        masked_teffs.mask, masked_metallicities.mask, masked_alphas.mask)
    model_logg = np.ma.zeros(len(masked_teffs))
    model_logg.mask = logg_mask

    # These are the alpha/Fe bins that will be fed into DSEP.
    alpha_binedges = np.arange(-0.1, 0.9, 0.2)
    # a/Fe < -0.1 corresponds to 1, and a/Fe > 0.7 corresponds to 6.
    alpha_bins = np.digitize(masked_alphas, alpha_binedges)+1
    # DSEP should crash or something if the metallicity and alpha enhancement
    # are not compatible. In particular, high alpha enhancements are only
    # available for low metallicity stars. I want to ensure that this will be
    # the case before running into weird DSEP bugs.
    assert(np.all(np.logical_or(alpha_bins < 4, np.logical_and(
        alpha_bins >= 4, masked_metallicities <= 0.0))))

    rounded_metallicities = np.around(masked_metallicities, 2)
    for i in range(len(model_logg)):
        if not model_logg.mask[i]:
            interp = sed.teff_to_logg_dwarf_DSEP_interpolator(
                age=age, metallicity=rounded_metallicities[i], afe=alpha_bins[i],
                lowT=lowTeff, highT=highTeff)
            model_logg[i] = interp(np.log10(masked_teffs[i]))

    return model_logg

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
    plot_by_ASPCAP_quality(rafa_apo_teff, rafa_logg_diff, rafa_flags)
    plt.ylabel("Log(g) diff (KIC - APOGEE)")
    plt.xlim(7000, 3200)
    plt.ylim(-1.0, 3.0)
    plt.title("Rafa APOGEE observations")
    plt.subplot(2, 2, 2)
    plot_by_ASPCAP_quality(mcq_apo_teff, mcq_logg_diff, mcq_flags)
    plt.ylabel("Log(g) diff (KIC - APOGEE)")
    plt.xlim(7000, 3200)
    plt.ylim(-1.0, 3.0)
    plt.title("McQuillan APOGEE Observations")
    plt.subplot(2, 2, 3)
    plot_by_ASPCAP_quality(rafa_apo_teff, rafa_teff_diff, rafa_flags)
    plt.ylabel("Teff diff (K) (KIC - APOGEE)")
    plt.xlim(7000, 3200)
    plt.ylim(-1000, 1000)
    plt.xlabel("Teff (K) (APOGEE)")
    plt.subplot(2, 2, 4)
    plot_by_ASPCAP_quality(mcq_apo_teff, mcq_teff_diff, mcq_flags)
    plt.ylabel("Teff diff (K) (KIC - APOGEE)")
    plt.xlabel("Teff (K) (APOGEE)")
    plt.xlim(7000, 3200)
    plt.ylim(-1000, 1000)

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

# Maybe make this capable of handing arrays of lowv and highv
def perform_vsini_cut(fullsample, lowv=None, highv=None, vcol="VSINI"):
    '''Perform a vsini cut on the sample.

    This function filters the full table according to the measured vsini. By
    default, the VSINI column is used, but other columns can be specified
    according to vcol.'''

    return perform_cut(fullsample, vcol, lowv, highv)

def perform_logg_cut(tbl, highlogg=None, lowlogg=None, loggcol="LogG"):
    '''Perform a cut on log g for a table sample.

    Used to restrict the range of log g for a sample.'''
    return perform_cut(tbl, loggcol, lowlogg, highlogg)

def perform_Ciardi_logg_cut(tbl, loggcol="logg", teffcol="teff"):
    '''Perform a cut on log g as advocated by Ciardi et al (2011).

    This is a proposed delineation between dwarfs and giants for Kepler
    targets. The cut is as follows:
                3.5                   if Teff >= 6000
    log(g) >= { 4.0                   if Teff <= 4250      }
                5.2 - (2.8e-4 * Teff) if 4250 < Teff < 6000

    This cut was used in McQuillan to select dwarfs.'''
    teff = tbl[teffcol]
    logg = tbl[loggcol]
    ciardi_indices = np.where(np.logical_or(np.logical_or(
        np.logical_and(teff >= 6000, logg >= 3.5), 
        np.logical_and(teff <= 4250, logg >= 4.0)),
        np.logical_and(
            np.logical_and(teff < 6000, teff > 4250),
            logg >= 5.2 - 2.8e-4 * teff)))
    cutstring = "Ciardi et al (2011) logg cut."
    cuttable = tbl[ciardi_indices]
    add_cut_metadata(cuttable, cutstring)
    return cuttable

def read_pulsators(pulsatorfile=paths.KIC_PULSATORS):
    '''Reads in a list of KIC IDs of known pulsators.'''

    pulsatortable = Table.read(
        pulsatorfile, format="ascii.no_header", names=["KIC"])
    return pulsatortable

def filter_pulsators(fulltable, quiet=False, KICcol="KIC"):
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

#############################################################################
# UKIRT info #
#############################################################################

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
    bintable, ukirt_file=paths.UKIRT_RESULTS):
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

def find_UKIRT_brightest_contaminating_object(ukirt_result):
    '''Return the row for the contaminating object from UKIRT result.

    Given a UKIRT result, find the second-brightest object within the Kepler
    PSF. This function will only contain the rows that fulfill the
    brightest-contaminant criterion.'''

    good_ukirt = filter_good_UKIRT_observations(ukirt_result)
    ukirt_groups = good_ukirt.group_by("upload_ID")

    main_contam_rows = []
    # The optimal aperture size for Kepler photometry generally ranges from
    # 10-50 pixels. Each pixel is about 4 arcseconds long.
    kepler_window = np.sqrt(50) * 4 * u.arcsec
    
    for i, contam_list in enumerate(ukirt_groups.groups):
        targetind = np.argmin(contam_list["distance"])
        sortindices = np.argsort(contam_list["jAperMag3"])

        # If the target is the brightest object, then pick the second brightest
        # in the aperture, otherwise, pick the brightest.
        if targetind == sortindices[0]:
            try:
                contamindex = sortindices[1]
            except IndexError:
                print("No object found on index {0}".format(
                    contam_list["upload_ID"]))
                contamindex = sortindices[0]
        else:
            contamindex = sortindices[0]
        main_contam_rows.append(contam_list[contamindex])

    contam_table = Table(rows=main_contam_rows, names=ukirt_result.colnames)
    return contam_table

def find_UKIRT_nearest_contaminating_object(ukirt_result):
    '''Return a table containing the nearest contaminating objects from UKIRT.

    Given a query result from UKIRT, select the rows corresponding to the
    closest contaminating object to each target.'''
    good_ukirt = filter_good_UKIRT_observations(ukirt_result)
    ukirt_groups = good_ukirt.group_by("upload_ID")

    main_contam_rows = []
    # The optimal aperture size for Kepler photometry generally ranges from
    # 10-50 pixels. Each pixel is about 4 arcseconds long.
    kepler_window = np.sqrt(50) * 4 * u.arcsec
    
    for i, contam_list in enumerate(ukirt_groups.groups):
        sortindices = np.argsort(contam_list["distance"])
        # Target should be closest to the central region
        targetind = sortindices[0]
        # Contaminant should be second-closest.
        try:
            contamindex = sortindices[1]
        except IndexError:
            print("No object found on index {0}".format(
                contam_list["upload_ID"]))
            contamindex = sortindices[0]

        main_contam_rows.append(contam_list[contamindex])

    contam_table = Table(rows=main_contam_rows, names=ukirt_result.colnames)
    return contam_table

def UKIRT_num_contaminating_objects(ukirt_result):
    '''Return array with number of contaminants in the UKIRT search window.

    Will count the number of contaminants in the ukirt_result table for each
    target. This will be useful in determining how densely populated certain
    targets are. Only objects with measured aperture magnitudes will be
    considered high enough quality to count.'''
    good_ukirt = filter_good_UKIRT_observations(ukirt_result)
    ukirt_groups = good_ukirt.group_by("upload_ID")

    num_conts = np.zeros(len(ukirt_groups.groups), dtype=np.int)
    for i, contam_list in enumerate(ukirt_groups.groups):
        num_conts[i] = len(contam_list) - 1

    return num_conts


def find_UKIRT_target_object(ukirt_result):
    '''Return the row for the target from a UKIRT result.

    Given a UKIRT result, find the object closest to the uploaded RA. This
    function will only contain the rows that are closest to the uploaded
    coordinate.'''
    good_ukirt = filter_good_UKIRT_observations(ukirt_result)
    ukirt_groups = good_ukirt.group_by("upload_ID")

    main_target_rows = []
    for target_list in ukirt_groups.groups:
        target_row = target_list[np.argmin(target_list["distance"])]

        main_target_rows.append(target_row)

    target_table = Table(rows=main_target_rows, names=ukirt_result.colnames)
    return target_table

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

def Bruntt_vsini_comparison():
    '''Plot the vsini values between APOGEE and Bruntt et al (2013).

    This is a good way to determine the uncertainty in vsini, as well as the
    cutoff vsini which is really an upper limit.'''
    brunttcat = catin.bruntt_dr14_overlap()
    aspcap_vsini = brunttcat["VSINI"]
    bad_vsini = np.where(aspcap_vsini < 0)
    good_vsini = np.where(aspcap_vsini > 0)
    aspcap_vsini[bad_vsini] = 0.0
    bruntt_vsini = brunttcat["vsini"]

    vsini_fracdiff = (bruntt_vsini - aspcap_vsini) / bruntt_vsini

#   plt.plot(bruntt_vsini[good_vsini], vsini_fracdiff[good_vsini], 'ko')
    plt.plot(bruntt_vsini[good_vsini], vsini_fracdiff[good_vsini], 'ko')
    plt.plot([7, 7], plt.ylim(), 'r-')

    detections = aspcap_vsini > 7
    print("Number of non-detections: {0:d}/{1:d}.".format(
        len(detections) - np.count_nonzero(detections), len(detections)))
    print("Vsini uncertainty is {0:.1f}%.".format(
        np.std(vsini_fracdiff[detections])*100))

    plt.xlabel("Bruntt vsini (km/s)")
    plt.ylabel("(Vsini (Bruntt) - Vsini (ASPCAP)) / Vsini(Bruntt)")

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


def EBs_missed_by_Rafa(rafa_ebs, missed_ebs, xcol, ycol, xlabel="", ylabel=""):
    '''Compares the objects which were found by Rafa's code to those missed.

    This function will plot two columns which are in both rafa_ebs and
    missed_ebs against each other.'''
    plt.plot(rafa_ebs[xcol], rafa_ebs[ycol], 'ko', label="Detected")
    plt.plot(missed_ebs[xcol], missed_ebs[ycol], 'gx', label="Missed")
    if xlabel:
        plt.xlabel(xlabel)
    else:
        plt.xlabel(xcol)
    if ylabel:
        plt.ylabel(ylabel)
    else:
        plt.ylabel(ycol)

def missed_EBs_logg_teff(rafa_ebs, missed_ebs, loggcol="logg", teffcol="teff"):
    '''Plot the missed EBs in an HR diagram.'''
    EBs_missed_by_Rafa(rafa_ebs, missed_ebs, teffcol, loggcol, "teff", "log g")
    hr.invert_x_axis()
    hr.invert_y_axis()
    plt.legend(loc="upper right")

def missed_EBs_period_depth(rafa_ebs, missed_ebs, periodcol="period",
                            depthcol="pdepth"):
    '''Plot the missed EBs with period vs. eclipse depth.'''
    EBs_missed_by_Rafa(rafa_ebs, missed_ebs, periodcol, depthcol, 
                       "Period (day)", "Primary Eclipse Depth")

def missed_EBs_teff_depth(rafa_ebs, missed_ebs, teffcol="teff", 
                          depthcol="pdepth"):
    '''Plot the missed EBs with Teff vs. eclipse depth.'''
    EBs_missed_by_Rafa(rafa_ebs, missed_ebs, teffcol, depthcol, 
                       "Teff (K)", "Primary Eclipse Depth")
    hr.invert_x_axis()

def missed_EBs_period_width(rafa_ebs, missed_ebs, periodcol="period",
                            widthcol="pwidth"):
    '''Plot the missed EBs with period vs. eclipse width.'''
    EBs_missed_by_Rafa(rafa_ebs, missed_ebs, periodcol, widthcol, 
                       "Period (day)", "Primary Eclipse Width")

def missed_EBs_teff_width(rafa_ebs, missed_ebs, teffcol="teff",
                          widthcol="pwidth"):
    '''Plot the missed EBs with Teff vs. eclipse width.'''
    EBs_missed_by_Rafa(rafa_ebs, missed_ebs, teffcol, widthcol, 
                       "Period (day)", "Primary Eclipse Width")
    hr.invert_x_axis()

def missed_EBs_compare_histogram(rafa_ebs, missed_ebs, histcol, binrange,
                                 bins=50, xlabel=""):
    '''Create a histogram of missed vs detected EBs.'''
    plt.hist(rafa_ebs[histcol], range=binrange, bins=bins, label="Missing")
    plt.hist(missed_ebs[histcol], range=binrange, bins=bins, label="Observed")
    if xlabel:
        plt.xlabel(xlabel)
    else:
        plt.xlabel(histcol)
    plt.ylabel("Number")

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
    periodrange = np.linspace(0.01, 10, 100)*u.day
    ratiorange = np.linspace(0.01, 1, 100)

    velocities = bc.calc_velocity_of_binary(
        combined_mass, periodrange, ratiorange[:,np.newaxis])

    plt.figure()
    contourlevels = [5, 10, 25, 50, 75, 100]
    CS = plt.contour(periodrange.to(u.day).value, ratiorange,
                     velocities.to(u.km/u.s).value, levels=contourlevels,
                     colors="k")
    plt.clabel(CS, inline=1, fontsize=13)
    plt.xlabel("Period (day)")
    plt.ylabel("Mass ratio")
    plt.title("Velocities for combined mass of {0}".format(combined_mass))

###############################################################################
# Eclipsing Binaries #
###############################################################################


def remove_Kepler_EBs(maincat, McQuillan=True, mainkiccol="KIC"):
    '''Filters out Kepler Eclipsing Binaries.

    Removes the KIC values corresponding to the eclipsing binaries in the
    version of the EB catalog given in paths.EB_PATH.
    '''
    if McQuillan:
        ebcat = catin.mcquillan_ebs()
    else:
        ebcat = catin.read_villanova_EBs()
    filtered_maincat = au.filter_column_from_subtable(
            maincat, mainkiccol, ebcat["KIC"])
    return filtered_maincat

def read_Kirk_geometric_correction_spline(
    splinepath="/home/regulus/simonian/Binaries/Kirk_geometric_correction_spline.csv"):
    '''Read the spline that represents the geometric correction for EBs.

    The correction was taken from Fig. 11 in Kirk et al (2016).'''
    splinepoints = Table.read(
        splinepath, format="ascii.csv", data_start=0, 
        names=("period", "efficiency"))
    periods = np.log10(splinepoints["period"])
    corrections = splinepoints["efficiency"]
    correction_interpolator = interp1d(periods, corrections)
    return correction_interpolator

def num_missing_binaries(periods):
    '''Infer the number of noneclipsing binaries from eclipsing ones.
    
    Given a distribution of orbital periods of a sample of stars, this function
    uses the geometric correction to infer the number of binaries which should
    also exist in the sample.
    '''
    logperiods = np.log10(periods)
    # This should take the log of periods as input, and the efficiency of
    # detection based on the corrections.
    geofrac_spline = read_Kirk_geometric_correction_spline()
    efficiencies = geofrac_spline(logperiods)
    # These should be the number of non-eclipsing binaries for each eclipsing
    # binary.
    inferred_missing = 1.0 / efficiencies - 1
    num_inferred_missing = np.sum(inferred_missing)
    return num_inferred_missing

def EB_histogram_to_rotation_histogram(binvalues, bins):
    '''Convert EB histogram to one of expected rotatational modulation.

    The bin values should be the number of EBs in each bin. The bins should be
    the edges of each bin. This should basically take arguments of plt.hist().
    '''
    midbins = (bins[:-1] + bins[1:]) / 2
    logmidbins = np.log10(midbins)
    geofrac_spline = read_Kirk_geometric_correction_spline()
    efficiencies = geofrac_spline(logmidbins)
    missing_fraction = 1.0 / efficiencies - 1
    missing_histogram = binvalues * missing_fraction
    print(missing_fraction)
    rotation_fraction = rotation_modulation_fraction()
    rotation_histogram = rotation_fraction * missing_histogram
    return rotation_histogram

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
    total_bins = num_missing_binaries(periods)
    correction = correct_for_rotation(total_bins)
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
    rotvalues = EB_histogram_to_rotation_histogram(eb_values, eb_binedges)
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
    

###############################################################################
# KepVIM #
###############################################################################

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
    quarter_table = Table([np.unique(kepvimtable[kic_col])])
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

def kepVIM_blending_statistics(magdiffs, offsets, quarters):
    '''Plots how contaminants affect various quarters of VIM.

    Generates a plot that shows how the number of quarters that an object
    experiences VIM is related to the magnitude and distance of the
    contaminant. The magnitude difference will be shown on the y-axis, the
    distance of the contaminant on the x-axis, and the color will reflect how
    many quarters of VIM it has.
    '''
    colormap = cm.viridis
    floatquarters = np.array(quarters, dtype=np.float)
    plt.scatter(offsets, magdiffs, c=floatquarters, cmap=colormap, s=8,
                edgecolors="face")
    cbar = plt.colorbar()
    cbar.set_label("Quarters")
    plt.xlabel("Distance from source")
    plt.ylabel("J_Target - J_Contam")


###############################################################################
# ASPCAP #
###############################################################################

def split_by_ASPCAP_flags(apogee_table, flag_col="ASPCAPFLAGS"):
    '''Splits the sample according to their ASPCAP fits.
    
    A three-tuple will be returned, which contains the bad, warn, and good
    indices, respectively. This particular implementation uses the ASPCAP flags
    instead of the bitmasks because the APOKASC catalog only has the flags
    available. Note that objects without any ASPCAP fits are classified as
    having bad fits.'''
    flags = apogee_table[flag_col]
    # Pure bad indices
    bad_indices = bad_ASPCAP_indices(flags, warn=False)
    good_indices = np.logical_not(bad_ASPCAP_indices(flags, warn=True))
    warn_indices = np.logical_not(np.logical_or(bad_indices, good_indices))
    assert np.all(
        np.logical_or(np.logical_or(good_indices, warn_indices), bad_indices) ==
        np.ones(len(flags)))

    return (apogee_table[bad_indices], apogee_table[warn_indices],
            apogee_table[good_indices])

def filter_bad_ASPCAP_fits(apogee_table, warn=False):
    '''Removes entries which have ASPCAP flags.

    If an object has the STAR_BAD flag enabled or does not have an ASPCAP fit
    at all, it will be removed. If the warn keyword is also specified, it will 
    also remove the STAR_WARN flag.
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

    Bad ASPCAP flags are defined as those with STAR_BAD in them or those with
    no ASPCAP result at all. If the warn keyword is given, STAR_WARN flags are 
    also marked as bad.
    '''
    bad_indices = npstr.find(aspcapflags, "STAR_BAD") >= 0
    bad_indices = np.logical_or(
        bad_indices, npstr.find(aspcapflags, "NO_ASPCAP_RESULT") >= 0)
    if warn:
        bad_indices = np.logical_or(bad_indices, npstr.find(
            aspcapflags, "STAR_WARN") >= 0)

    return bad_indices

def good_aspcap_fits(apotable, aspcapcol="ASPCAPFLAG"):
    '''Only return the entries with good ASPCAP fits.'''
    return apogee_filter_quality(
        apotable, quality=("bad", "warn"), aspcapcol=aspcapcol)

def apogee_filter_quality(apotable, quality=("good", "bad", "warn"), 
                          aspcapcol="ASPCAPFLAG"):
    '''Remove the entries in apotable of the given quality.
    
    If aspcapcol is given, then the objects with the given qualities for
    aspcapcol will be removed.'''
    
    bad_flags = ASPCAP_STAR_BAD + NO_ASPCAP_RESULT
    warn_flags = ASPCAP_STAR_WARN + ASPCAP_VSINI_WARN

    remaining_table = apotable
    aspcapflags = remaining_table[aspcapcol]
    bad_indices = aspcapflags & bad_flags != 0
    warn_indices = np.logical_and(aspcapflags & warn_flags != 0,
                                  np.logical_not(bad_indices))
    good_indices = np.logical_not(np.logical_or(warn_indices, bad_indices))

    if "good" in quality:
        remaining_table = remaining_table[
            np.where(np.logical_not(good_indices))]
        add_cut_metadata(remaining_table, "Removed good ASPCAP fits")
        aspcapflags = remaining_table[aspcapcol]
        bad_indices = aspcapflags & bad_flags != 0
        warn_indices = np.logical_and(aspcapflags & warn_flags != 0,
                                      np.logical_not(bad_indices))
    if "warn" in quality:
        remaining_table = remaining_table[
            np.where(np.logical_not(warn_indices))]
        add_cut_metadata(remaining_table, "Removed warn ASPCAP fits")
        aspcapflags = remaining_table[aspcapcol]
        bad_indices = aspcapflags & bad_flags != 0
    if "bad" in quality:
        remaining_table = remaining_table[
            np.where(np.logical_not(bad_indices))]
        add_cut_metadata(remaining_table, "Removed bad ASPCAP fits")

    return remaining_table

###############################################################################
# Double-Lined Spectroscopic Binaries #
###############################################################################

def mark_DLSB_indices(apogee_ids, dlsb_db=DLSB_PATH):
    '''Mark which apogee IDs correspond to known DLSBs.

    Creates an index array which marks the apogee_ids which are known
    double-line spectroscopic binaries. It looks at the file in at dlsb_db to
    have a list of APOGEE_IDs corresponding to double-lined spectroscopic
    binaries.'''
    dlsbs = browse.read_DLSB_db(db_path=dlsb_db)
    dlsb_indices = au.mark_selections_in_columns(apogee_ids, dlsbs["APOGEE_ID"])
    return dlsb_indices

def mark_non_DLSB_indices(apogee_ids, nodl_db=NON_DLSB_PATH):
    '''Mark which apogee IDs are confirmed non-DLSBs.

    Creates an index array which marks the apogee_ids which are known not to be
    double-line spectroscopic binaries. It looks at the file at nodl_db to have
    a list of APOGEE_IDs corresponding to objects which aren't doubled-lined
    spectroscopic binaries.'''
    nodls = browse.read_null_db(db_path=nodl_db)
    nodl_indices = au.mark_selections_in_columns(apogee_ids, nodls["APOGEE_ID"])
    return nodl_indices

def filter_double_lined_spectroscopic_binaries(
    apocat, apid_col="APOGEE_ID", dlsb_db=DLSB_PATH, verbose=False):
    '''Remove Double-Lined Spectroscopic Binaries by APOGEE_ID'''
    dlsb_indices = mark_DLSB_indices(apocat[apid_col], dlsb_db=dlsb_db)
    # Print out the found DLSBs
    if verbose:
        dlsb_names = apocat[apid_col][dlsb_indices]
        for dlsb in dlsb_names:
            print("{0} is a known DLSB.".format(dlsb))
    filtered_cat = apocat[au.get_complement_indices(dlsb_indices, len(apocat))]
    add_cut_metadata(
        filtered_cat, "Removed DLSBs: see {0} for list".format(dlsb_db))
    return filtered_cat


################################################################################
# Problematic Photometric Periods #
################################################################################

def select_bad_photometric_periods():
    '''Select spectroscopic rapid rotators with poor photometric agreement.

    Select cool stars with known spectroscopic rapid rotation. Then select out
    the ones with very discrepant photometric periods.
    '''
    apo = catin.read_APOKASC_catalog()
    good_apo = filter_bad_ASPCAP_fits(apo)
    mcq = catin.mcquillan_with_stelparms()
    mcq_apo = au.join_by_id(good_apo, mcq, "KEPLER_INT", "KIC")

    mcq_apo_cool = perform_teff_cut(mcq_apo, hightemp=5500, teffcol="teff")
    mcq_apo_dwarf = perform_logg_cut(mcq_apo_cool, lowlogg=4.0, loggcol="logg")

    mcq_apo_vdetect = perform_vsini_cut(mcq_apo_dwarf, lowv=10)

    high_vsini = period_to_velocities(1, mcq_apo_vdetect["radius"])
    low_vsini = period_to_velocities(5, mcq_apo_vdetect["radius"])
    mcq_apo_highv = mcq_apo_vdetect[np.where(np.logical_and(
        mcq_apo_vdetect["VSINI"] < high_vsini, 
        mcq_apo_vdetect["VSINI"] > low_vsini))]

    veq = period_to_velocities(mcq_apo_highv["Prot"], mcq_apo_highv["radius"])
    bad_periods = mcq_apo_highv[mcq_apo_highv["VSINI"] > 2 * veq]
    bad_periods_nodlsb = filter_double_lined_spectroscopic_binaries(
        bad_periods, apid_col="2MASS_ID")
    return bad_periods_nodlsb

def write_bad_photometric_periods(
        dest=paths.HEAD_DIR / "vsini_rapid_with_bad_P.txt"):
    '''Write spectroscopic rapid rotators with poor photometric agreement.

    Select cool stars with known spectroscopic rapid rotation. Then
    specifically select the ones with very discrepant photometric periods. Also
    write comments explicitly detailing how the sample was derived.
    '''
    bad = select_bad_photometric_periods()
    veq = period_to_velocities(bad["Prot"], bad["radius"])
    output_table = Table(
        [bad["KIC"], bad["2MASS_ID"], bad["teff"], bad["logg"], bad["radius"], 
         bad["TEFF_COR"], bad["LOGG_FIT"], bad["VSINI"], bad["Prot"], veq,
         bad["ASPCAPFLAGS"]],
        names=("KIC", "APOGEE_ID", "Huber Teff", "Huber Logg", "Huber Radius", 
               "APOGEE Teff", "APOGEE Logg", "VSINI", "Prot", "Veq", 
               "ASPCAPFLAGS"))
    output_table.meta["comments"] = [
        "This sample represents spectroscopic rapid rotators which have",
        "photometric periods which are highly inconsistent with the measured",
        "vsini.",
        "",
        "This sample was generated jointly from the APOKASC catalog v4.2.3",
        "and McQuillan et al (2014). The criteria to select were:",
        "",
        "  1. Select objects with Huber Teff < 5500 K.",
        "  2. Select objects with Huber logg > 4.0.",
        "  3. Select robust vsini detections (VSINI > 10 km)",
        "  4. Select objects consistent with rapid rotation:",
        "       (2 pi R)/(1 day) > VSINI > (2 pi R)/(5 day)",
        "       with R being the Huber radius.",
        "  5. Select objects with discrepant periods:",
        "       VSINI > 2 * (2 pi R)/Prot",
        "       Since (2 pi R) / Prot = edge-on equatorial velocity, any",
        "         measurement with VSINI > Veq is unphysical. The extra",
        "         factor of 2 is allowing for radius and alias uncertainties",
        "  6. Remove double-lined spectroscopic binaries.",
        "       This is achieved by checking the APOGEE spectra for the",
        "       remaining objects.",
        ""]

    format_dict = {"Veq": ".3f"}

    output_table.write(str(dest), format="ascii.fixed_width",
                       formats=format_dict)

################################################################################
# Spectroscopic Rapid Rotator Routines #
################################################################################

def select_spectroscopic_rapid_rotators(
        inputtable, lowp=1, highp=5, vsini_col="VSINI", rad_col="radius"):
    '''Select spectroscopic rapid rotators.

    These are objects whose vsinis lie between the period range given. Of
    course, because of the sin(i) ambiguity, the two samples do not match
    perfectly. There will be a few objects which scatter out of the high period
    region of the sample due to inclination, and a few that scatter into the
    sample from the low period region. Given that the number of short-period
    objects ought to be less than the number of long-period objects, the number
    scattering out should be significantly more than the number scattering in.
    '''
    low_vel_limits = period_to_velocities(5, inputtable[rad_col])
    high_vel_limits = period_to_velocities(1, inputtable[rad_col])

    vel_lim = inputtable[np.logical_and(
        inputtable['VSINI'] > low_vel_limits, inputtable["VSINI"] <
        high_vel_limits)]

    return vel_lim

def APOKASC_spectroscopic_rapid_rotators():
    '''Select rapid rotators in APOKASC.

    This function will go through the APOKASC catalog. It will make logg and
    teff cuts based on Huber et al (2014) parameters instead of APOGEE
    parameters. However, McQuillan periods won't be involved in this sample at
    all.'''
    apo = catin.APOKASC_with_KIC_stelparms()
    good_apo = filter_bad_ASPCAP_fits(apo)

    apo_dwarfs = perform_logg_cut(
        good_apo, lowlogg=3.5, loggcol="logg")
    apo_obs = perform_teff_cut(
        apo_dwarfs, lowtemp=4800, hightemp=5600, teffcol="teff")
    
    apo_detect = perform_vsini_cut(apo_obs, lowv=7)
    apo_highv = select_spectroscopic_rapid_rotators(apo_detect)
    return apo_highv
    apo_noeb = remove_Kepler_EBs(
        apo_highv, McQuillan=False, mainkiccol="KEPLER_INT")
    apo_nodlsb = filter_double_lined_spectroscopic_binaries(
        apo_noeb, apid_col="2MASS_ID")

    # Now filter out objects with V < 14
    ehk = catin.read_EHK_catalog()
    apo_photo = au.join_by_ra_dec(
        apo_nodlsb, ehk, "RA", "DEC", "RA", "Dec", join_type="left")





