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
import pyrallaxes


import astropy_util as au
import statop as stat
import hrplots as hr
import binarycalcs as bc
import path_config as paths
import sed
import browse_APOGEE_spectra as browse
import read_catalog as catin

SDSS3_URL = "http://data.sdss3.org"

NUM_KEPLER_QUARTERS = 17

DLSB_PATH = paths.DLSB_DB
NON_DLSB_PATH = paths.NODL_DB


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
    jackson_prefix = "J"
    epic_prefix = ""

    try:
        tbl1 = tbl1[~tbl1[tm1].mask]
    except AttributeError:
        pass

    try:
        tbl2 = tbl2[~tbl2[tm2].mask]
    except AttributeError:
        pass

    if np.any(npstr.startswith(tbl1[tm1], kic_prefix)):
        tbl1_prefix = kic_prefix
    elif np.any(npstr.startswith(tbl1[tm1], apogee_prefix)):
        tbl1_prefix = apogee_prefix
    elif np.any(npstr.startswith(tbl1[tm1], jackson_prefix)):
        tbl1_prefix = jackson_prefix
    elif np.any(npstr.startswith(tbl1[tm1], epic_prefix)):
        tbl1_prefix = epic_prefix
    # Given the EPIC key, I think this is redundant. But if I decide to
    # implement this differently, it may still be important.
    else:
        raise ValueError("Don't recognize 2MASS key: " + tbl1[tm1][0])

    if np.any(npstr.startswith(tbl2[tm2], kic_prefix)):
        tbl2_prefix = kic_prefix
    elif np.any(npstr.startswith(tbl2[tm2], apogee_prefix)):
        tbl2_prefix = apogee_prefix
    elif np.any(npstr.startswith(tbl2[tm2], jackson_prefix)):
        tbl2_prefix = jackson_prefix
    elif np.any(npstr.startswith(tbl2[tm2], epic_prefix)):
        tbl2_prefix = epic_prefix
    else:
        raise ValueError("Don't recognize 2MASS key: " + tbl2[tm2][0])

    if tbl1_prefix == tbl2_prefix:
        new_table = au.join_by_id(
            tbl1, tbl2, tm1, tm2, join_type=join_type, idproc=npstr.strip, 
            conflict_suffixes=conflict_suffixes)
        print("Types the same")
    else:
        print("Types different")
        def transform(oldcol):
            # The epic prefix is special because I can't search and replace an
            # empty string.
            if tbl2_prefix == epic_prefix:
                newcol = npstr.add(tbl1_prefix, oldcol)
            else:
                newcol = npstr.replace(oldcol, tbl2_prefix, tbl1_prefix)
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

###################################
# Dressing and Charbonneau (2013) #
###################################

def replace_Dressing_Charbonneau_params(
    kictable, kiccol="kepid", oldteffcol="KIC Teff", newteffcol="DC Teff",
    oldloggcol="KIC logg", newloggcol="DC logg"):
    '''Replace the Teff and logg parameters with Dressing & Charbonneau.

    For the Kepler IDs which have KIC entries in the Dressing and Charbonneau
    (2013) table, replace the temperatures and loggs in oldteffcol and
    oldloggcol with the DC13 temperatures. The new parameters are placed in the
    newteffcol and new loggcol columns.
    '''
    dctable = catin.read_Dressing_Charbonneau_table()
    kictable[newteffcol] = np.array(kictable[oldteffcol])
    kictable[newloggcol] = np.array(kictable[oldloggcol])
    fulltable = au.join_by_id(kictable, dctable, kiccol, "KIC",
                              join_type="left")
    try:
        unchangedindices = fulltable["Teff"].mask
    except KeyError:
        unchangedindices = fulltable["Teff_A"].mask

    kictable[newteffcol][~unchangedindices] = fulltable["Teff"][
        ~unchangedindices]
    try:
        kictable[newloggcol][~unchangedindices] = fulltable["logg"][
            ~unchangedindices]
    except KeyError:
        kictable[newloggcol][~unchangedindices] = fulltable["logg_A"][
            ~unchangedindices]

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

def split_asteroseismic_dwarfs(fullsamp, apid_col):
    '''Split the asteroseismic dwarfs from the rest of the sample.
    
    This function uses the current APOKASC database to determine whether an
    object is an asteroseismic dwarf or not.'''
    apokasc = catin.read_APOKASC_catalog()[["2MASS_ID", "RADIUS_DW"]]
    ast_dwarf = filter_invalid_APOGEE_entries(apokasc, "RADIUS_DW")
    match_indices = au.mark_selections_in_columns(
        fullsamp[apid_col], ast_dwarf)
    return (fullsamp[match_indices], fullsamp[~match_indices])

def split_McQuillan_periods(fullsamp, kiccol):
    '''Split sample to those with and without McQuillan periods.
    
    Note that the sample with McQuillan periods will have the McQuillan
    information.'''
    mcq = catin.read_McQuillan_catalog()
    mcq_apo = au.join_by_id(fullsamp, mcq, kiccol, "KIC")
    nonmcq_apo = au.get_complement_table(mcq_apo, fullsamp, kiccol)

    return mcq_apo, nonmcq_apo
    
##################
# APOGEE filters #
##################

def invalid_indices(apotable, colname, maskvalue=np.ma.masked):
    '''Mark the indices where the column values are mask values.'''
    badindices = au.mark_selections_in_columns(apotable[colname], [maskvalue])

    return badindices

def filter_invalid_APOGEE_entries(apotable, colname, maskvalue=np.ma.masked):
    '''Remove rows from apotable where column values are the mask values.

    This will filter apotable where only the rows that do not have the mask
    value in the column will be returned.'''
    badindices = invalid_indices(apotable, colname, maskvalue=maskvalue)
    filtered_table = apotable[~badindices]
                                         
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
    previous_files = catin.find_split_files(singlefile)
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
        outputsegment.write(
            str(outputpath / outputfile), format=table_format, 
            include_names=output_columns, comment=False, overwrite=True)

def write_Kepler_field_Vizier_upload_list(
        outputfile=paths.VIZIER_KEPLER_INPUT):
    '''Write a file resolved by Vizier with the full Kepler field.'''
    fullkepler = catin.read_KIC_DR25_catalog()
    write_KIC_Vizier_upload_list(
        fullkepler["kepid"], outputfile.name, outputpath=outputfile.parent)

def write_xMatch_upload_list(
        ra, dec, outputfile, ids=None, outputpath=paths.HEAD_DIR):
    '''Write the upload list suitable for sending to xMatch.
    
    The RA and Dec should be given in J2000 coordinates. If a set of IDs are
    associated with each coordinate, those can also be given to this file, and
    it will be added to the list. IDs are useful for identifying cases where
    xMatch didn't find an associated source, or found multiple associated
    sources.
    
    The file will be written in the directory outputpath to the file
    outputfile.'''
    xmatch_table = Table([ids, ra, dec], names=("ID", "RA", "DEC"))
    xmatch_table.write(str(outputpath / outputfile), format="ascii.csv",
                       comment=False)

def clean_cross_matched_table(tbl, uniq_col="ID", dist_col="angDist"):
    '''Ensure that a table has only one target associated with each ID.

    If any of entries under tbl[uniq_col] are duplicated, only select the one
    with the lowest value in tbl[dist_col].'''
    keep_indices = np.zeros(len(tbl), dtype="bool")
    for ident in tbl[uniq_col]:
        ident_match = np.where(tbl[uniq_col] == ident)[0]
        minval = np.argmin(tbl[dist_col][ident_match])
        keep_indices[ident_match[minval]] = 1

    return tbl[keep_indices]

def write_TAP_table(table, outputfile, outputpath=paths.HEAD_DIR):
    '''Write the table such that it can be uploaded through the TAP.
    
    This basically ensures that the table is written in a votable format.'''
    table.write(str(outputpath / outputfile), format="votable")

def clean_XMatch_file(tablepath, uniq_col="ID", tableformat="ascii.csv"):
    '''Ensure XMatch file has only one target associated with each ID.

    Go through the table located at tablepath (in the format specified in
    tableformat and ensure that all
    identifications in uniq_col are unique. If there are duplicates, chose the
    one with the lowest value in dist_col.'''
    tbl = Table.read(tablepath, format=tableformat)

    newtbl = clean_cross_matched_table(
        tbl, uniq_col=uniq_col, dist_col="angDist")

    newtbl.write(str(tablepath), format=tableformat, overwrite=True)

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

def write_KIC_to_Gaia_Archive_upload_list(kics, outputpath):
    '''Write a list of kics to be uploaded to the Gaia archive.'''
    kicstr = kics.astype(np.str)
    kic_column = npstr.add("KIC", kicstr)
    kic_table = Table([kic_column], names=["KIC"])
    kic_table.write(str(outputpath), format="ascii.no_header", overwrite=True, 
                    delimiter=",")

def write_SIMBAD_identifier_list(
    identifiers, outputfile, outputpath=paths.HEAD_DIR):
    '''Write a list that can be uploaded to SIMBAD as a list of identifiers.'''
    ident_table = Table([identifiers], names=["Ident"])
    write_columns_for_input(
        ident_table, outputfile, 99999, "ascii.no_header",
        output_columns=["Ident"], outputpath=outputpath)

def write_IPAC_identifier_list(identifiers, outputfile,
                               outputpath=paths.HEAD_DIR):
    '''Write a list to be uploaded to IPAC as a list of SIMBAD identifiers.'''
    ident_table = Table([identifiers], names=["objstr"])
    write_columns_for_input(
        ident_table, outputfile, 99999, "ascii.ipac", output_columns=["objstr"],
        outputpath=outputpath)

def write_IPAC_coord_list(ras, decs, outputfile, outputpath=paths.HEAD_DIR):
    '''Write a list to be uploaded to IPAC as a list of RA and DEC.
    
    This is preferred if RA and Dec are known because IRSA doesn't do this
    automatically.'''
    # There should be a way to validate that RA and DEC are in decimal degrees.
    coord_table = Table([ras, decs], names=("ra", "dec"))
    write_columns_for_input(
        coord_table, outputfile, 99999, "ascii.ipac", output_columns=["ra",
        "dec"], outputpath=outputpath)


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
        write_UKIRT_file(bintable, output_filename=ukirt_input_path)
        ukirt_file = paths.HEAD_DIR / input(
            "Please input the path to the WFCAM output file:")

    ukirt_catalog = catin.read_UKIRT_file(ukirt_file)
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

        target_ind = np.argmin(
            apogee_sources["jAperMag3"])
        brightest_contam_ind = np.argmin(
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

###############################################################################
# ASPCAP #
###############################################################################

def split_by_ASPCAP_flags(apogee_table, flag_col="ASPCAPFLAGS"):
    '''Splits the sample according to their ASPCAP fits.
    
    A four-tuple will be returned, which contains the bad, warn, vsini_warn, and 
    good indices, respectively. This particular implementation uses the ASPCAP 
    flags instead of the bitmasks because the APOKASC catalog only has the 
    flags available. Note that objects without any ASPCAP fits are classified as
    having bad fits. Objects with the VSINI_WARN flag are separated because
    they may need to be inspected separately from the other objects.'''
    flags = apogee_table[flag_col]
    # Pure bad indices
    bad_indices = bad_ASPCAP_indices(flags, warn=False)
    badwarn_indices = bad_ASPCAP_indices(flags, warn=True)
    warn_indices = np.logical_and(
        np.logical_not(bad_indices), badwarn_indices)
    vsini_indices = np.logical_and(
        np.logical_not(badwarn_indices), warn_VSINI_indices(flags))
    good_indices = np.logical_not(np.logical_or(
        badwarn_indices, vsini_indices))
    assert np.all(
        np.logical_or(np.logical_or(np.logical_or(
            good_indices, warn_indices), bad_indices), vsini_indices) ==
        np.ones(len(flags)))

    return (apogee_table[bad_indices], apogee_table[warn_indices],
            apogee_table[vsini_indices], apogee_table[good_indices])

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

def aspcap_quality_bitmask_indices(aspcapbitmask, quality):
    '''Return indices of objects with the given quality.

    The quality value can be either "bad", "warn", "vsini", or "good". If the 
    quality value is "bad", all objects with the STAR_BAD or NO_ASPCAP_RESULT 
    flags are marked. If the quality value is "warn", all objects with the
    STAR_WARN flag will be marked. If the quality value is "vsini", all objects
    with the VSINI_WARN flag will be marked. And lastly, a "good" quality will
    only select objects with none of the aforementioned flags. Note this
    implies that flags such as VMICRO_WARN will be classified as "good".'''
    bad_flags = ASPCAP_STAR_BAD + NO_ASPCAP_RESULT
    warn_flags = ASPCAP_STAR_WARN
    vsini_flags = ASPCAP_VSINI_WARN
    bad_indices = aspcapbitmask & bad_flags != 0
    if quality.lower() == "bad":
        return bad_indices
    # All bad_indices are accompanied by warn indices. So make sure that you
    # don't double-count the bad indices.
    warn_indices = np.logical_and(aspcapbitmask & warn_flags != 0,
                                  np.logical_not(bad_indices))
    if quality.lower() == "warn":
        return warn_indices
    vsini_indices = au.multi_logical_and(
        aspcapbitmask & vsini_flags != 0, np.logical_not(warn_indices),
        np.logical_not(bad_indices))
    if quality.lower() == "vsini":
        return vsini_indices
    good_indices = au.multi_logical_and(
        np.logical_not(vsini_indices), np.logical_not(warn_indices),
        np.logical_not(bad_indices))
    assert np.all(np.sum([bad_indices, warn_indices, vsini_indices,
                          good_indices], axis=0) == 1)
    if quality.lower() == "good":
        return good_indices
    else:
        raise ValueError("Don't understand quality: {0}".format(quality))



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

def search_in_ASPCAPFLAGS(aspcapflags, flagval):
    '''Index entries which have flagval in aspcapflags.

    Returns an index array which indicates which values in aspcapflags have the
    string given in flagval within. This helps with general manipulation of the
    aspcapflags parameter without having to pull up the bitmask array.'''
    strcol = au.byte_to_unicode_cast(aspcapflags)
    indexarr = npstr.find(strcol, flagval) >= 0
    return indexarr

def bad_ASPCAP_indices(aspcapflags, warn=False):
    '''Picks bad ASPCAP flags from flag array.

    Bad ASPCAP flags are defined as those with STAR_BAD in them or those with
    no ASPCAP result at all. If the warn keyword is given, STAR_WARN flags are 
    also marked as bad.
    '''
    badcolumns = [
        "TEFF_BAD", "LOGG_BAD", "VMICRO_BAD", "M_H_BAD", "ALPHA_M_BAD", 
        "CHI2_BAD", "SN_BAD", "COLORTE_BAD", "ATMOS_HOLE_BAD", "ROTATION_BAD", 
        "C_M_BAD", "N_M_BAD", "NO_ASPCAP_RESULT"]
    warncolumns = [
        "TEFF_WARN", "LOGG_WARN", "VMICRO_WARN", "M_H_WARN", "ALPHA_M_WARN", 
        "CHI2_WARN", "SN_WARN", "COLORTE_WARN", "ATMOS_HOLE_WARN", 
        "ROTATION_WARN", "C_M_WARN", "N_M_WARN"]
    bad_indices = au.multi_logical_or(*[search_in_ASPCAPFLAGS(
        aspcapflags, badcol) for badcol in badcolumns])
    if warn:
        warn_indices = au.multi_logical_or(*[search_in_ASPCAPFLAGS(
            aspcapflags, warncol) for warncol in warncolumns])
        bad_indices = np.logical_or(bad_indices, warn_indices)

    return bad_indices

def warn_VSINI_indices(aspcapflags):
    '''Picks ASPCAP flags which indicate a VSINI warning.

    These are objects which have the VSINI_WARN flag enabled.'''
    warn_indices = search_in_ASPCAPFLAGS(aspcapflags, "VSINI_WARN")
    return warn_indices

def good_aspcap_fits(apotable, aspcapcol="ASPCAPFLAG"):
    '''Only return the entries with good ASPCAP fits.'''
    return apogee_filter_quality(
        apotable, quality=("bad", "warn"), aspcapcol=aspcapcol)

###############################################################################
# APOGEE Targeting #
###############################################################################

# Have a lookup dictionary that has the column and bitmask information.
target_dict = {
    "APOGEE_KEPLER_COOLDWARF": ("APOGEE_TARGET2", 16), 
    "APOGEE2_APOKASC_DWARF": ("APOGEE2_TARGET1", 28),
    "APOGEE2_APOKASC_GIANT": ("APOGEE2_TARGET1", 27),
    "APOGEE2_APOKASC": ("APOGEE2_TARGET1", 30),
    "APOGEE_KEPLER_SEISMO": ("APOGEE_TARGET1", 27),
    "APOGEE_KEPLER_EB": ("APOGEE_TARGET1", 23),
    "APOGEE_KEPLER_HOST": ("APOGEE_TARGET1", 28),
    "APOGEE_RV_MONITOR_KEPLER": ("APOGEE_TARGET2", 19),
    "APOGEE2_KOI": ("APOGEE2_TARGET3", 0),
    "APOGEE2_KOI_CONTROL": ("APOGEE2_TARGET3", 2),
    "APOGEE2_EB": ("APOGEE2_TARGET3", 1),
    "APOGEE_TELLURIC": ("APOGEE_TARGET2", 9),
    "APOGEE2_TELLURIC": ("APOGEE2_TARGET2", 9),
    "APOGEE_CALIB_CLUSTER": ("APOGEE_TARGET2", 10),
    "APOGEE2_YOUNG_CLUSTER": ("APOGEE2_TARGET3", 5)
}

def target_indices(fulltable, targetlabel):
    '''Select the objects in fulltable that are specified by targetlabel.

    This function requires the full APOGEE table because it automatically
    selects the targeting column where the given targeting bitmask can be
    found.'''
    colname, exponent = target_dict[targetlabel]
    select_indices = fulltable[colname] & 2**exponent > 0
    return select_indices


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

###############################################################################
# Jen Cool Dwarfs #
###############################################################################

def build_cool_dwarf_sample(apodata=None):
    '''Build Jen's cool dwarf sample.

    This will consist of both the objects with the APOGEE_KEPLER_COOLDWARF
    targeting flag enabled, as well as objects which Jen indicated *should* be
    in the sample, but were observed through other programs.
    '''
    if not apodata:
        apodata = catin.dr14_with_KIC_stelparms()
        apodata["LOGG_FIT"] = apodata["FPARAM"][:,1]
    targeted_sample = apodata[apodata["APOGEE_TARGET2"] & 2**16 != 0]
    jen_missed = catin.read_van_Saders_missing_targets()
    found_targets = au.join_by_id(apodata, jen_missed, "kepid", "KIC",
                                  join_type="right")
    new_sample = au.extract_subtable_from_column(
        apodata, "kepid", found_targets["kepid"])

    jen_sample = vstack([targeted_sample, new_sample])
    jen_cool = perform_teff_cut(jen_sample, hightemp=5500, teffcol="KIC Teff")
    replace_Dressing_Charbonneau_params(jen_cool)

    return jen_cool

###############################################################################
# KOIs #
###############################################################################

def KOI_indices(kiccol):
    '''Get the indices of KOIs in the dataset.'''
    koicat = catin.read_KOI_list()
    koi_indices = au.mark_selections_in_columns(kiccol, koicat["kepid"])
    return koi_indices

###############################################################################
# APOGEE Binaries #
###############################################################################

def APOGEE_Binary_Classification(tmID):
    '''Classify the 2MASS IDs given according to the Binary Classification.

    Make an array which classifies the list of APOGEE objects according to the
    classes defined in the El-Badry et al (2017) study. These include the
    classes:

    * "Single" objects that show no signs of binarity
    
    * "SB1" for single-lined spectroscopic binaries which show RV variability.

    * "SB2" for double-lined spectroscopic binaries.

    * "Triple" for double-lined spectroscopic binaries that have a
        hidden third component.

    * "SB3" for triple-lined spectroscopic binaries.

    * "N/A" for objects which were not analyzed as part of the study. This will
      largely be due to the fact that they were observed after DR13.
    '''
    labelcol = np.zeros(len(tmID), dtype="U6")
    singletable = catin.read_El_Badry_Single_Stars()
    singleindices = au.mark_selections_in_columns(
        tmID, singletable["APOGEE_ID"])
    del(singletable)
    sb1table = catin.read_El_Badry_SB1()
    sb1indices = au.mark_selections_in_columns(
        tmID, sb1table["APOGEE_ID"])
    del(sb1table)
    sb2table = catin.read_El_Badry_SB2()
    sb2indices = au.mark_selections_in_columns(
        tmID, sb2table["APOGEE_ID"])
    del(sb2table)
    tripletable = catin.read_El_Badry_hidden_triples()
    tripleindices = au.mark_selections_in_columns(
        tmID, tripletable["APOGEE_ID"])
    del(tripletable)
    sb3table = catin.read_El_Badry_SB3()
    sb3indices = au.mark_selections_in_columns(
        tmID, sb3table["APOGEE_ID"])
    del(sb3table)

    labelcol[singleindices] = "Single"
    labelcol[sb1indices] = "SB1"
    labelcol[sb2indices] = "SB2"
    labelcol[tripleindices] = "Triple"
    labelcol[sb3indices] = "SB3"

    naindices = labelcol == ""
    labelcol[naindices] = "N/A"

    return labelcol

def add_APOGEE_Binary_column(datatab, apidcol="APOGEE_ID", newcol="Binarity"):
    '''Add a column corresponding to the spectral binarity of the system.

    This function will add a column to datatab with the title of newcol. The
    targets will be identified by the APOGEE_IDs in apidcol.'''
    binarray = APOGEE_Binary_Classification(datatab[apidcol])
    datatab[newcol] = binarray

###############################################################################
# Tables for Don #
###############################################################################

def write_Garcia_with_McQuillan_overlap():
    '''Write a table for Garcia with the McQuillan overlap.'''
    garcia = catin.read_Garcia_periods()
    garcia_cols = garcia[["KIC", "Prot", "e_Prot", "Sph", "e_Sph", "Type"]]
    del(garcia)
    mcq = catin.read_McQuillan_catalog()
    mcq_cols = mcq[["KIC", "Prot", "e_Prot", "Rper"]]
    del(mcq)
    joined_garcia_mcq = au.join_by_id(
        garcia_cols, mcq_cols, "KIC", "KIC", join_type="left", 
        conflict_suffixes=["_Garcia", "_McQuillan"])
    huber = catin.read_KIC_DR25_catalog()
    huber_cols = huber[[
        "kepid", "teff", "teff_err1", "teff_err2", "logg", "logg_err1",
        "logg_err2", "teff_prov", "logg_prov", "jmag", "hmag", "kmag"]]
    del(huber)
    huber_cols.rename_column("teff", "Huber teff")
    huber_cols.rename_column("logg", "Huber logg")
    joined_garcia_mcq_huber = au.join_by_id(
        joined_garcia_mcq, huber_cols, "KIC", "kepid", join_type="left")

    comment = [
        "The Garcia sample with accompanying McQuillan and Huber entries.",
        "", 
        "This table contains the rotational period and activity measure of", 
        "the spot oscillation for the Garcia sample. The activity measure in",
        'this case is labeled as "Sph", which is the average standard',
        "deviation of the light curve in 5*Prot intervals.",
        "",
        "It also contains the rotational period and activity measure of the", 
        "spot oscillation for the McQuillan overlap sample. Around 1/3 of the",
        "Garcia sample has McQuillan periods. The activity measure in this",
        "case is labeled as Rper, and is the height of the autocorrelation",
        "peak.",
        "",
        "The Teff and log(g) included in this table were derived from the",
        "Huber et al (2014) methodology, and include asymmetric",
        "uncertainties. The teff_prov and logg_prov columns describe the", 
        "inputs for the analysis: whether they use KIC photometry, J-K",
        "photometry, spectroscopy, or asteroseismology, to name a few.",
        "",
        "The codes for the provenances can be found:",
        "https://exoplanetarchive.ipac.caltech.edu/docs/API_keplerstellar_columns.html#stellar",
        "",
        "Also included are 2MASS JHK photometry for all of the targets."
    ]
    joined_garcia_mcq_huber.meta["comments"] = comment

    joined_garcia_mcq_huber.write(
        str(paths.HEAD_DIR / "Don_Garcia_Table.txt"), 
        format="ascii.fixed_width", include_names=
    ["KIC", "Prot_Garcia", "e_Prot_Garcia", "Sph", "e_Sph", "Type",
     "Prot_McQuillan", "e_Prot_McQuillan", "Rper", "Huber teff", "teff_err1", 
     "teff_err2", "Huber logg", "logg_err1", "logg_err2", "teff_prov",
     "logg_prov", "jmag", "hmag", "kmag"])

def combined_huber_apogee_table():
    '''Write a table with APOGEE targets and Huber et al parameters.'''
    huber = catin.read_KIC_DR25_catalog()
    huber_cols = huber[[
        "kepid", "tm_designation", "teff", "teff_err1", "teff_err2", "logg", 
        "logg_err1", "logg_err2", "teff_prov", "logg_prov"]]
    huber_cols.rename_column("teff", "Huber teff")
    huber_cols.rename_column("logg", "Huber logg")
    huber_cols.rename_column("kepid", "KIC")
    del(huber)
    apogee = catin.read_dr14_allStar()
    apogee["LOGG_FIT"] = apogee["FPARAM"][:,1]
    apogee_cols = apogee[[
        "APOGEE_ID", "TEFF", "LOGG_FIT", "VSINI", "VHELIO_AVG", "VERR",
        "VSCATTER", "NVISITS", "M_H", "M_H_ERR", "FE_H", "FE_H_ERR", "J", "H",
        "K", "ASPCAPFLAGS"]]
    del(apogee)
    apogee_cols.rename_column("TEFF", "APOGEE teff")
    apogee_cols.rename_column("LOGG_FIT", "APOGEE logg")
    joined_apogee_huber = join_by_2MASS_key(
        huber_cols, apogee_cols, "tm_designation", "APOGEE_ID", join_type="inner")

    comment = [
        "The APOGEE Kepler sample with Huber parameters.",
        "",
        "This is the intersection of targets with APOGEE observations and", 
        "Huber et al (2014) parameters (which should be all Kepler targets).",
        "",
        "The Huber et al (2014) parameters include the effective and log(g)",
        "including the asymmetric uncertainties. The teff_prov and logg_prov",
        "columns describe the inputs for the analysis. The codes for the",
        "provenances can be found:",
        "https://exoplanetarchive.ipac.caltech.edu/docs/API_keplerstellar_columns.html#stellar",
        "",
        "The APOGEE parameters have the Teff and log(g) as inferred from the",
        "APOGEE spectra. There is also a radial velocity (given in the",
        '"VHELIO_AVG" column) and its error ("VERR"). For targets with more',
        'than one visit, there is also a "VSCATTER" computed to reflect RV',
        'variability.',
        "",
        "Metallicity can be described in two quantities: [Fe/H] and [M/H].",
        "The usual iron abundance and its error are associated with [Fe/H].",
        "A combined metallicity (which should essentially be [Z/H]) and its",
        "error is also provided.",
        "",
        "Also included are 2MASS JHK photometry of the targets. When the",
        "Huber et al (2014) pipeline fails (in rare cases), the catalog does",
        "not include 2MASS photometry, even when it exists and is available",
        "from the APOGEE catalog. Therefore, all 2MASS photometry in this", 
        "table is taken from the APOGEE catalog instead of the Huber catalog."
    ]
    joined_apogee_huber.meta["comments"] = comment

    joined_apogee_huber.write(
        str(paths.HEAD_DIR / "Don_APOGEE_Table.txt"), 
        format="ascii.fixed_width", include_names=[
            "KIC", "APOGEE_ID", "APOGEE teff", "APOGEE logg", "VSINI",
            "VHELIO_AVG", "VERR", "VSCATTER", "NVISITS", "M_H", "M_H_ERR",
            "FE_H", "FE_H_ERR", "J", "H", "K","ASPCAPFLAGS", "Huber teff", 
            "teff_err1", "teff_err2", "Huber logg", "logg_err1", "logg_err2", 
            "teff_prov", "logg_prov"])


###############################################################################
# Supplement tables with columns #
###############################################################################

###############
# DSEP values #
###############
    
def generate_DSEP_radius_column(
    apotable, teffcol="TEFF", fehcol="FE_H", age=3, radcol="APOGEE radius"):
    '''Add a radius column to apotable.

    The column to convert teff to radius should be given as Teffcol. The radius
    will be stored in the radcol column.'''
    radiusarr = np.zeros(len(apotable))
    # A dictionary referencing DSEP models according to metallicity.
    DSEP_models = {}
    # I want to round metallicity to the nearest hundredth
    rounded_metallicities = np.round(apotable[fehcol], 2)
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

#######################
# Absolute Magnitudes #
#######################

def parallax_to_distance_modulus(parallax):
    '''Convert parallax to a distance modulus.
    
    The parallax should be given in units of milliarcseconds.'''
    return -5*np.log10(parallax/100)

def parallax_err_to_distance_modulus_err(parallax_err, parallax):
    '''Convert an error in parallax to an error in distance modulus.'''
    return 5 * parallax_err / parallax / np.log(10)

def distance_to_distance_modulus(dist):
    '''Convert distance to distance modulus.

    The distance should be given in units of parsecs.'''
    return 5 * np.log10(dist/10)

def distance_err_to_distance_modulus_err(dist_err, dist):
    '''Convert an error in distance to an error in distance modulus.'''
    return 5 * dist_err / dist / np.log(10)

def generate_abs_mag_column(
        apotable, appcol, abscol, v_to_ext, avcol="Av", parallaxcol="", 
        distcol=""):
    '''Create a extinction-corrected column of absolute magnitudes.

    For the table in apotable, use the data in appcol, avcol and either
    parallaxcol or distcol to generate absolute magnitudes, which will be stored 
    in abscol. A function which converts Av to the extinction in a given band to 
    the extinction in the desired band also needs to be passed.

    Only one of parallaxcol or distcol should be specified; if they are both
    specified, then an error will be output. Parallaxcol will be set to
    "parallax" by default.
    '''
    if parallaxcol != "" and distcol != "":
        raise ValueError("Received conflicting parallax and distance columns.")
    if parallaxcol == "" and distcol == "":
        parallaxcol = "parallax"

    if parallaxcol != "" and distcol == "":
        distance_modulus = parallax_to_distance_modulus(apotable[parallaxcol])
    elif parallaxcol == "" and distcol != "":
        distance_modulus = distance_to_distance_modulus(apotable[distcol])

    new_ext = v_to_ext(apotable[avcol])
    apotable[abscol] = sed.calc_abs_magnitude(
        apotable[appcol], distance_modulus, new_ext)

def generate_abs_mag_column_with_errors(
        apotable, appcol, apperrcol, abscol, absupcol, absdowncol, v_to_ext,
        v_err_to_ext_err, parallaxcol="", parallax_err_col="",
        parallax_offset=0.05, distcol="", dist_up_col="", dist_down_col="",  
        avcol="av", avupcol="av_err1", avdowncol="av_err2", fullgaia=True):
    '''Create absolute magnitude columns with Gaia info and photometry.

    This calculates the given K-band absolute magnitude using the usual
    relation. Either parallaxes or distance can be specified, but only one
    should be given, with the rest being empty strings. By default it will
    assume that the parallax column is "parallax" and the uncertainty column is
    "parallax_error". A zero-point offset for the parallax should be given in
    parallax_offset. It approximates errors by adding the terms in quadrature.
    For blended objects, the null value for the K-band magnitude is propagated.
    '''
    if parallaxcol != "" and distcol != "":
        raise ValueError("Received conflicting parallax and distance columns.")
    elif (parallaxcol != "" and parallax_err_col == ""):
        raise ValueError("Parallax column given without error")
    elif (parallaxcol == "" and parallax_err_col != ""):
        raise ValueError("Parallax error given without parallax.")
    elif (distcol != "" and (dist_up_col == "" or dist_down_col == "")):
        raise ValueError("Distance column given without error")
    elif (distcol == "" and (dist_up_col != "" or dist_down_col != "")):
        raise ValueError("Distance error given without distance")
    elif parallaxcol == "" and distcol == "":
        parallaxcol = "parallax"

    if parallaxcol != "" and distcol == "":
        if fullgaia:
            dmo = parallax_to_distance_modulus_fulk(
                (apotable[parallaxcol]+parallax_offset)/1000,
                apotable[parallax_err_col]/1000)
            distance_modulus_down, distance_modulus, distance_modulus_up = dmo
        else:
            distance_modulus = parallax_to_distance_modulus(
                apotable[parallaxcol]+parallax_offset)
            distance_modulus_up = parallax_err_to_distance_modulus_err(
                apotable[parallax_err_col], apotable[parallaxcol])
            distance_modulus_down = parallax_err_to_distance_modulus_err(
                apotable[parallax_err_col], apotable[parallaxcol])
    elif parallaxcol == "" and distcol != "":
        distance_modulus = distance_to_distance_modulus(apotable[distcol])
        distance_modulus_up = distance_err_to_distance_modulus_err(
            apotable[dist_up_col], apotable[distcol])
        distance_modulus_down = distance_err_to_distance_modulus_err(
            apotable[dist_down_col], apotable[distcol])

    new_ext = v_to_ext(apotable[avcol])
    new_ext_up = v_err_to_ext_err(apotable[avcol], apotable[avupcol])
    new_ext_down = v_err_to_ext_err(apotable[avcol], apotable[avdowncol])

    apotable[abscol] = sed.calc_abs_magnitude(
        apotable[appcol], distance_modulus, new_ext)
    apotable[absupcol] = sed.calc_abs_magnitude_err(
        apotable[apperrcol], distance_modulus_down, new_ext_down)
    apotable[absdowncol] = sed.calc_abs_magnitude_err(
        apotable[apperrcol], distance_modulus_up, new_ext_up)

def parallax_to_distance_modulus_fulk(parallaxes, errors, L=1350):
    '''Convert parallax to distance modulus.
    
    The parallax and error needs to be in arcseconds. The scale length of disk
    needs to be in parsecs. The function returns a 3-tuple containing the lower
    bound of the distance modulus, the mode of the distance modulus, and the
    upper bound of the distance modulus.'''
    assert len(parallaxes) == len(errors)
    parmask = parallaxes.mask
    dm = np.zeros(len(parallaxes))
    dm_upper = np.zeros(len(parallaxes))
    dm_lower = np.zeros(len(parallaxes))
    for i, (parallax, error) in enumerate(zip(parallaxes, errors)):
        # Calculate the location of the mode.
        mode = pyrallaxes.dmod_mode_exp(parallax, error, L)
        # Integrate from mode-10 to mode.
        lower_mode_integral = pyrallaxes.percentiles(
            pyrallaxes.dmpdfexp, mode-10, mode, parallax, error, L)
        normalization_factor  = pyrallaxes.normalization(
            pyrallaxes.dmpdfexp, L, parallax, error, lower_mode_integral, mode,
            mode+10)
        mode_percentile = pyrallaxes.normalized_percentile(
            lower_mode_integral, normalization_factor)

        median = pyrallaxes.dmod_median(
            pyrallaxes.dmpdfexp, parallax, error, L, mode+10, mode,
            mode_percentile, normalization_factor, -10)

        lower_dmod = pyrallaxes.distances_from_percentiles_dmod(
            pyrallaxes.dmpdfexp, normalization_factor, parallax, error, "inf",
            .5-.67/2, .5+.67/2, 0.5, median, L, mode+10, -10)
        upper_dmod = pyrallaxes.distances_from_percentiles_dmod(
            pyrallaxes.dmpdfexp, normalization_factor, parallax, error, "sup",
            .5-.67/2, .5+.67/2, 0.5, median, L, mode+10, -10)

        dm[i] = mode
        dm_upper[i] = upper_dmod
        dm_lower[i] = lower_dmod

    return dm_lower, dm, dm_upper
