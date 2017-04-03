"""Functions for manipulating the APOKASC-McQuillan catalog
"""
import os
import re
import itertools
import io

import numpy as np
import numpy.core.defchararray as npstr
import scipy
from scipy.io import readsav
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import astropy.units as u
from astropy.table import Table, join, vstack, unique, Column
from astropy.coordinates import SkyCoord
from astropy.io import fits
from bs4 import BeautifulSoup
import requests
# from apogee.tools import bitmask

import astropy_util as au
import hrplots as hr
import binarycalcs as bc
import path_config as paths
import sed
import browse_APOGEE_spectra as browse

SDSS3_URL = "http://data.sdss3.org"

NUM_KEPLER_QUARTERS = 17

APOGEE_NULL = -9999.0

###############################################################################
# Reading catalogs #
###############################################################################

def read_APOKASC_catalog(
    filepath=paths.APOKASC_PATH, exclude_single_epoch=False, filter_BAD=False, 
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
    filepath=paths.MCQUILLAN_CATALOG, Huber_KIC=True, huberpath=paths.HUBER_CATALOG):
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

def read_original_KIC_catalog(filepath=paths.ORIG_KIC):
    '''Read in the original KIC catalog.'''
    kic = Table.read(str(filepath), format="ascii.basic", delimiter="|")
    return kic

def read_Huber_KIC_catalog(huberpath=paths.HUBER_CATALOG):
    '''Read the HUBER KIC parameters.'''
    hubercat = Table.read(str(huberpath), format="ascii.cds")
    return hubercat

def read_KIC_DR25_catalog(kicpath=paths.KIC_CATALOG):
    '''Read the KIC DR2 Stellar Parameter catalog.'''
    kiccat = Table.read(str(kicpath), format="ascii.ipac")
    fix_table_coordinates_units(kiccat, "ra", "dec")
    return kiccat

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

def read_UKIRT_file(resultfile):
    '''Reads in a file from UKIRT.'''
    results = read_split_file(resultfile, "ascii.commented_header")
    return results

def read_split_file(filepath, table_format):
    '''Reads a file that has been split into multiple parts.

    This function essentially re-reads a table which has been split according
    to the large_table_multiple_files_split function. However, it uses the
    existing files in the directory instead of predicting using table
    information. For example, if filepath is /path/to/foo.txt, this will find
    foo.txt, if it exists, or foo.0.txt, foo.1.txt, foo.2.txt, etc. and read
    them all in if it doesn't.

    Since the input table isn't used, this means that when writing split files,
    care has to be taken to delete all previous queries made with them.
    '''
    try:
        inputtable = Table.read(str(filepath), format=table_format)
    except FileNotFoundError as f:
        inputfiles = find_split_files(filepath)
        table_pieces = []
        for inputfile in inputfiles:
            table_piece = Table.read(inputfile, format=table_format)
            table_pieces.append(table_piece)
        try:
            inputtable = vstack(table_pieces)
        # This means that inputfiles was empty.
        except TypeError:
            raise f
        
    return inputtable

def find_split_files(filepath):
    '''Finds the extant filenames which filepath would have if it were split.

    Given a filepath, returns a list of filepaths that match the basename being
    split in the path parent. For example, if the pathpath is /path/to/foo.txt, 
    it will return a list file paths called foo.0.txt foo.1.txt, foo.2.txt if 
    they reside in /path/to/.
    '''
    folder = filepath.parent
    filename = filepath.name
    base, ext = split_filename(filename)
    glob_pattern = format_split_filename(base, "*", ext)
    files = folder.glob(glob_pattern)
    return files

def read_APOGEE_dwarfs(apopath=paths.APOGEE_DWARF_PATH):
    '''Read a query for all APOGEE dwarfs.

    This was a query manually submitted for understanding how the RV error and
    SNR vary. The SQL query to get these objects was:

    SELECT
        a.target_id, a.apogee_id a.apogee_target1, a.apogee_target2, 
        a.extratarg, a.dec, a.mjd, a.plate, a.ra, a.snr, a.starflag, a.vhelio, 
        a.vrel, a.vrelerr, a.vtype, c.fe_h, c.fe_h_err, c.fe_h_flag, 
        c.fparam_logg, c.teff, c.teff_err, c.teff_flag, c.vsini, o.h, o.h_err, 
        o.sfd_ebv
    FROM apogeeVisit a
    JOIN aspcapstar c on a.apogee_id = c.apogee_id
    JOIN apogeeObject o on a.apogee_id = o.apogee_id
    WHERE 
        a.ra BETWEEN 277.5 and 305 AND
        a.dec BETWEEN 33.75 and 44.5 AND
        c.fparam_logg > 4.1 AND dbo.fApogeeExtraTarg('TELLURIC') > 0
    '''
    dwarfs = Table.read(str(apopath), format="ascii.csv", header_start=1)
    np.ma.masked_equal(dwarfs["vrelerr"], -9999)
    return dwarfs

def apogee_kepler_field(apogee_allvisit, apogee_allstar):
    '''Take the APOGEE allVisit file and pick out targets in the kepler field.

    This function essentially performs the same location cut as the query in
    read_APOGEE_dwarfs.
    '''
    joined_table = join_by_id(
        apogee_allvisit, apogee_allstar, "apogee_id", "apogee_id")
    kepler_field = np.logical_and(np.logical_and(np.logical_and(
        joined_table["RA"] > 277.5, joined_table["RA"] < 305), 
        joined_table["DEC"] > 33.75), joined_table["DEC"] < 44.5)
    return kepler_field



def read_UCAC4_Mcquillan_Tidsync(
    upath=paths.UCAC_TIDSYNC_PATH, kic_col="KIC"):
    '''Reads the UCAC4 table of Tidally-synchronized binaries in McQuillan.

    The UCAC-4 table was obtained from Vizier
    (http://cdsbib.u-strasbg.fr/cgi-bin/cdsbib?2012yCat.1322....0Z).'''
    pm_table = Table.read(str(upath), format="ascii.basic", delimiter=";",
                          data_start=3, header_start=0)
    pm_groups = pm_table.group_by("_1")
    # Want to find the entry in each group with the smallest distance.
    matched_rows = []
    for grp in pm_groups.groups:
        matched_rows.append(grp[np.argmin(grp["_r"])])
    UCAC_table = Table(rows=matched_rows, names=pm_table.colnames)
    kic_numbers = npstr.replace(UCAC_table["_1"], "KIC ", "")
    kic_ints = Column(kic_numbers, name=kic_col, dtype=np.int)
    del(UCAC_table["_1"])
    UCAC_table.add_column(kic_ints, index=0)
    return UCAC_table

def read_UCAC4_Rafa_Tidsync(
    upath=paths.UCAC_TIDSYNC_RAFA_PATH, kic_col="KIC"):
    '''Reads the UCAC4 table of Tidally-synchronized binaries in McQuillan.

    The UCAC-4 table was obtained from Vizier
    (http://cdsbib.u-strasbg.fr/cgi-bin/cdsbib?2012yCat.1322....0Z).'''
    pm_table = Table.read(str(upath), format="ascii.basic", delimiter=";",
                          data_start=3, header_start=0)
    pm_groups = pm_table.group_by("_1")
    # Want to find the entry in each group with the smallest distance.
    matched_rows = []
    for grp in pm_groups.groups:
        matched_rows.append(grp[np.argmin(grp["_r"])])
    UCAC_table = Table(rows=matched_rows, names=pm_table.colnames)
    kic_numbers = npstr.replace(UCAC_table["_1"], "KIC ", "")
    kic_ints = Column(kic_numbers, name=kic_col, dtype=np.int)
    del(UCAC_table["_1"])
    UCAC_table.add_column(kic_ints, index=0)
    return UCAC_table

def read_TGAS_McQuillan_APOGEE_overlap_tidsync(
    path=paths.TGAS_MCQUILLAN_APOGEE_TIDSYNC_PATH):
    '''Read the TGAS information for McQuillan/APOGEE targets.

    Reads in the information from TGAS which is available for the sample of
    targets which overlap between McQuillan and APOGEE.'''
    tgas_table = Table.read(str(path), format="fits")
    return tgas_table

def join_by_2MASS_key(tbl1, tbl2, tm1, tm2, join_type="inner",
                      skip_missing=True):
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
        new_table = au.join_by_id(tbl1, tbl2, tm1, tm2, join_type=join_type, 
                                  idproc=npstr.strip)
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
        tbl2_newcol = transform(tbl2_oldcol)
        del(tbl2[tm2])
        try:
            tbl2[tm2] = tbl2_newcol
            new_table = au.join_by_id(tbl1, tbl2, tm1, tm2, join_type=join_type,
                                      idproc=npstr.strip)
        finally:
            del(tbl2[tm2])
            tbl2[tm2] = tbl2_oldcol
            

    return new_table

def read_dr14_allVisit(allvisitpath=paths.DR14_ALLVISIT_PATH, kepleropt=True):
    '''Read the DR14 allVisit file.

    This function reads the l31c.1 version of the allVisit file. If the
    Kepleropt keyword is given, then only the objects in the RA range of Keler
    will be read, which should greatly reduce the memory requirements. Only the
    summary data table will be read, which should contain everthing necessary
    for the APOGEE pipeline.
    
    WARNING: If kepleropt is disabled, then the extremely large table may cause
    python to crash if it can't fit in memory.'''

    if kepleropt:
        allvisit_hdus = fits.open(str(allvisitpath), memmap=True)
        allvisit_indices = allvisit_hdus[2]
        index_start = allvisit_indices.data[279]
        index_end = allvisit_indices.data[302]
        # This will only have targets in the Kepler RA range.
        allvisit_kepler = allvisit_hdus[1].data[index_start:index_end]
        allvisit_hdus.close()
        # Convert from recarray to Table
        allvisit = Table(allvisit_kepler)
    else:
        allvisit = Table.read(str(allvisitpath), format="fits")
        allvisit["APOGEE_ID"] = npstr.rstrip(allvisit["APOGEE_ID"])

    return allvisit

def read_dr14_allStar(allstarpath=paths.DR14_ALLSTAR_PATH, kepleropt=True):
    '''Reads the allStar file for DR14.
    
    Reads in the allStar table for DR14. If the Kepleropt keyword is given,
    then only the objects in the RA range of Kepler will be read, which should
    greatly reduce the memory requirements. Only the Summary data table will be
    read in, which should contain everything necessary for the APOGEE pipeline.

    WARNING: If kepleropt is disabled, then extremely large table will take 
    several hours to fit into memory.
    '''
    if kepleropt:
        allstar_hdus = fits.open(str(allstarpath), memmap=True)
        allstar_indices = allstar_hdus[2]
        index_start = allstar_indices.data[279]
        index_end = allstar_indices.data[302]
        # This will only have targets in the Kepler RA range.
        allstar_kepler = allstar_hdus[1].data[index_start:index_end]
        allstar_hdus.close()
        # Convert from recarray to Table
        allstar = Table(allstar_kepler)
    else:
        allstar = Table.read(str(allstarpath), format="fits")
    return allstar

def read_Rafa_rotation(rottable=paths.RAFA_SAVITA_PERIODS):
    '''Reads in the rotation periods as determined by Rafa's pipeline.

    This function will read in a file which, at the very least, contains the
    KICs and corresponding periods of all of the stars which have detections of
    rotation periods according to Rafa's pipeline.'''
    rafa = Table.read(
        str(rottable), format="ascii.no_header", names=["KIC", "Prot"])
    # The file Rafa gave me had a bunch of giants in it. So I'm going to
    # manually limit the catalog to the same one 
    kic_catalog = read_KIC_DR25_catalog()
    rafa_kic = au.join_by_id(rafa, kic_catalog, "KIC", "kepid")
    tempcut = perform_teff_cut(rafa_kic, lowtemp=0, hightemp=5500,
                               teffcol="teff")
    giantcut = perform_logg_cut(tempcut, lowlogg=3.5, loggcol="logg")
    return giantcut
            
###############################################################################
# Catalog Curation
###############################################################################

# def van_Saders_relevant_table(

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

def select_joinable_apogee_columns(
    apotable, exclude_cols=[
        "STABLERV_CHI2", "STABLERV_RCHI2", "CHI2_THRESHOLD",
        "STABLERV_CHI2_PROB", "PARAM", "FPARAM", "PARAM_COV", "FPARAM_COV",
        "PARAMFLAG", "FELEM", "FELEM_ERR", "X_H", "X_H_ERR", "X_M", "X_M_ERR",
        "ELEM_CHI2", "ELEMFLAG", "VISIT_PK", "ALL_VISIT_PK", "FPARAM_CLASS",
        "CHI2_CLASS"]):
    new_cols = [x for x in apotable.colnames if x not in exclude_cols]
    shrunk_table = apotable[new_cols]
    return shrunk_table

def fix_table_coordinates_units(tbl, ra_col, dec_col):
    '''Fixes the units for coordinates in the table. 

    For each of the coordinate columns, check that the unit entries for 
    the coordinates are recognized by astropy.units. This has been made because
    there are non-standard representations of units in the KIC catalog (and
    possibly others), such as using "degrees" instead of "degree", which the
    units framework can automatically determine.
    '''
    if tbl[ra_col].unit == "degrees":
        tbl[ra_col].unit = u.degree
    if tbl[dec_col].unit == "degrees":
        tbl[dec_col].unit = u.degree

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

def create_joined_APOKASC_McQuillan_catalog(
        apocat=None, mcquillancat=None, apofile=paths.APOKASC_PATH,
    mcquillanfile=paths.MCQUILLAN_CATALOG):
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

def cumulative_number_binned_by_temperature(
    mcquillan, hightemp=6500, lowtemp=3000, dtemp=50, teffcol="Teff"):
    '''Plots a cumulative histogram of number based on temperature.'''
    tempbins = np.arange(lowtemp, hightemp+dtemp, dtemp)
    plt.hist(mcquillan[teffcol], bins=tempbins, range=(lowtemp, hightemp),
             cumulative=True, histtype="step")
    plt.xlim(plt.xlim()[::-1])
    plt.xlabel("Teff (K)")
    plt.ylabel("Number of cooler than Teff")

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
# APOGEE Figures #
###############################################################################

ASPCAP_STAR_BAD = 2**23
ASPCAP_STAR_WARN = 2**7
ASPCAP_VSINI_WARN = 2**14
NO_ASPCAP_RESULT = 2**31

def apogee_select_quality(apotable, quality=("good", "bad", "warn"), 
                          aspcapcol="ASPCAPFLAG"):
    '''Select the entries in apotable of the given quality.
    
    If aspcapcol is given, then the objects with the given qualities for
    aspcapcol will be selected.'''
    
    bad_flags = ASPCAP_STAR_BAD + NO_ASPCAP_RESULT
    warn_flags = ASPCAP_STAR_WARN + ASPCAP_VSINI_WARN

    aspcapflags = apotable[aspcapcol]
    bad_indices = aspcapflags & bad_flags != 0
    warn_indices = np.logical_and(aspcapflags & warn_flags != 0,
                                  np.logical_not(bad_indices))
    good_indices = np.logical_not(np.logical_or(warn_indices, bad_indices))

    indices = np.zeros(len(apotable), dtype=np.bool)
    if "good" in quality:
        indices = np.logical_or(indices, good_indices)
    if "warn" in quality:
        indices = np.logical_or(indices, warn_indices)
    if "bad" in quality:
        indices = np.logical_or(indices, bad_indices)

    return apotable[indices]


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
            eisit_table = get_APOGEE_visit_info(obj_id, obj_loc_id)
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

# Also want function that takes list of apogee fields and kic binaries and only
# returns the ones that are in the given fields.
def targets_in_APOGEE_fields(
    apogee_fields, kic_targets, kic_racol="ra", kic_raunit=u.deg, 
    kic_deccol="dec", kic_decunit=u.deg, field_col="APOGEE_Field"):
    '''Determine which KIC targets lie within the given APOGEE fields.

    Returns the subset of kic_targets which can be found in the given APOGEE
    fields. The APOGEE fields that each target can be found in will be in the
    column given by field_col.'''
    all_apogee_fields = read_APOGEE_KASC_fields()
    found_apogee_fields = unique(au.extract_subtable_from_column(
        all_apogee_fields, "NAME", apogee_fields), keys="NAME")
    field_coords = SkyCoord(
        l=found_apogee_fields["Lon"]*u.deg, b=found_apogee_fields["Lat"]*u.deg, 
        frame="galactic")
    
    try:
        object_coords = SkyCoord(
            kic_targets[kic_racol], kic_targets[kic_deccol], frame="icrs")
    except u.UnitsError:
        object_coords = SkyCoord(
            kic_targets[kic_racol], kic_targets[kic_deccol], frame="icrs",
            unit=(kic_raunit, kic_decunit))
    target_fields = APOGEE_plates(
        object_coords, found_apogee_fields["NAME"], field_coords)
    found_target_indices = np.where(target_fields != "")
    apogee_kics = kic_targets[found_target_indices]
    apogee_kics[field_col] = target_fields[found_target_indices]
    return apogee_kics

def write_APOGEE_proposal_table(
    field_targets, outputpath=paths.APOGEE_ANCILLARY_TARGETS_TABLE, 
    apogee_field_col="APOGEE_Field", twomass_col="tm_designation", ra_col="ra", 
    dec_col="dec", coord_source="KIC", hmag_col="hmag", hmag_source="2MASS", 
    pmra_col="pmRA", pm_cosdec_applied=True, pmdec_col="pmDE", 
    pm_source="UCAC-4", apokasc_visits=3, koi_visits=4, apokasc_SN=13, 
    koi_SN=100):
    '''Write the target table for the APOGEE Ancillary science proposal.

    TYPE 1 PROPOSALS: Provide a table with the following information for each target, one target per line and sorted by field:

    APOGEE-2 or MaNGA field name (e.g., "008-02", or "K12_074+15", see https://trac.sdss.org/wiki/APOGEE2/TargetingPlan for APOGEE-2 field names, and ​https://data.sdss.org/sas/mangawork/manga/target/tiles/v2_3/tilecenters_alladjusted.fits for MaNGA tile centers/names)
    target name (2MASS ID if available; if no 2MASS ID is available, proposers must provide an alternate name and separately describe the targeting/analysis plan for such targets, which present challenges for the APOGEE-2 targeting/reduction pipelines, and will most likely deliver spectra with marginal S/N);
    J2000.0 target coordinates (RA,Dec) in the format 00:00:00.0 +01:00:00
    Source of coordinates ("Gaia", "2MASS" etc.)
    H-band fiber magnitude (e.g., "10.5")
    source for H-band photometry (e.g., "2MASS"; "VVV")
    proper motion measurements, in units of mas/yr;
    source for proper motion measurements (e.g., "UCAC-4")
    minimum number of visits requested for target (e.g., "3")
    total requested S/N (e.g., "100") 
    '''
    # This is the seed output table.
    output_table = field_targets[[apogee_field_col, hmag_col]]

    # Format the 2MASS ID correctly
    output_table[twomass_col] = npstr.replace(field_targets[twomass_col],
                                              "2MASS J", "2M")

    # Now include Coordinates
    target_coordinates = SkyCoord(
        ra=field_targets[ra_col], dec=field_targets[dec_col])
    ra_strings = target_coordinates.ra.to_string(
        unit="hour", sep=":", precision=1)
    dec_strings = target_coordinates.dec.to_string(
        decimal=False, sep=":", alwayssign=True, precision=0)
    coord_strings = npstr.add(ra_strings, npstr.add(" ", dec_strings))
    output_table["Coords"] = coord_strings

    # H-band magnitude
    output_table[hmag_col].unit = None

    # Proper motions
    output_table[r"$\mu_\alpha \cos \delta$"] = np.ma.masked_invalid(
        field_targets[pmra_col])
    output_table[r"$\mu_\delta$"] = np.ma.masked_invalid(
        field_targets[pmdec_col])

    # Visits and S/N
    KASC_sources = np.logical_or(
        field_targets[apogee_field_col] == "K16_075+11",
        field_targets[apogee_field_col] == "K20_073+09")
    KOI_sources = ~KASC_sources
    minvisits = np.zeros(len(field_targets), dtype=np.int)
    minvisits[KASC_sources] = apokasc_visits
    minvisits[KOI_sources] = koi_visits
    output_table["Min. visits"] = minvisits

    requestedSN = np.zeros(len(field_targets))
    requestedSN[KASC_sources] = apokasc_SN
    requestedSN[KOI_sources] = koi_SN
    output_table["Req. S/N"] = requestedSN

    ordered_output = output_table[[
        apogee_field_col, twomass_col, "Coords", hmag_col, 
        r"$\mu_\alpha \cos \delta$", r"$\mu_\delta$", 
        "Min. visits", "Req. S/N"]]
    ordered_output.sort(apogee_field_col)

    # Comments about the dataset.
    ordered_output.meta["comments"] = [
    "Coords are from the KIC.",
    "H-band magnitudes are from 2MASS.",
    "Proper motions are from UCAC-4."]

    names = ["APOGEE field", "2MASS ID", "Coords", "H", "PM_RA", "PM_DE", 
             "Visits", "Req. S/N"]
    # Unfortunately, the LaTeX writer isn't able to handle longtable correctly.
    # Therefore, I want to write the file to a StringIO object and replace the
    # instances of tabular with those of longtable.
    ordered_output.write(
        str(outputpath), format="ascii.fixed_width", names=names)

def KIC_to_APOGEE_2MASS_designation(kic_desig):
    '''Function to convert KIC 2MASS designations to be APOGEE ones.

    The KIC designations are in the form of 2MASS J##########, while the apogee
    ones are 2M##########.'''
    apo_desig = npstr.replace(kic_desig, "2MASS J", "2M")
    return apo_desig

def read_APOGEE_KOI_fields(koifields=paths.APOGEE_KOI_FIELDS):
    '''Get a table which contains information on the APOGEE KOI fields.'''
    colnames = ["NAME", "Lon", "Lat", "DESIGN", "NVISITS", "TYPE",
                "HEMISPHERE"]
    return Table.read(str(koifields), format="ascii.basic", comment="!",
                      names=colnames, guess=False)

def read_APOGEE_KASC_fields(kascfields=paths.APOGEE_KASC_FIELDS):
    '''Get a table which contains information on the APOGEE KOI fields.'''
    colnames = ["NAME", "Lon", "Lat", "DESIGN", "NVISITS", "TYPE",
                "HEMISPHERE"]
    kasc_table = Table.read(str(kascfields), format="ascii.basic", comment="!",
                      names=colnames, guess=False)
    return kasc_table

def observable_on_plate(objcoords, fieldcoord, CENTER_EXCLUSION=1.5*u.arcmin,
                        FOV=1.5*u.degree):
    '''Returns whether an object is observable on an APOGEE plate.

    The center coordinate of the plate should be given in platecoord while the
    coordinate of a group of objects should be given as objcoords. The field of 
    view [1] should be (7\pi) degrees [2]. Each plate also has a central 
    exclusion region due to the center posts of 1.5 arcminutes [3].

    [1] There are currently three sources for the field of view. The first is the
    APOGEE web site, given in [2], with the description of the spectrograph
    having a 2 degree field of view. there is also the APOGEE technical paper
    by Majewski et al (2016; arXiv:1509.05420), which states that APO has a
    field of view of 3 degrees. Lastly, there is the targeting page [3], which
    states that the total field of view of an APOGEE plate is 7 square degrees.
    We'll take this to be the canonical value.

    [2] http://www.sdss.org/instruments/apogee_spectrograph/

    [3] http://www.sdss.org/dr13/irspec/targets/

    [4] https://trac.sdss.org/wiki/APOGEE2/PlateDesign/ExclusionRadius
    '''
    separations = fieldcoord.separation(objcoords)
    observable = np.logical_and(separations < FOV, separations >
                                CENTER_EXCLUSION)
    return observable

def APOGEE_plates(objectcoords, fieldIDs, fieldcoords):
    '''Return the APOGEE plates which the objects can be observed on.

    The object and plate coordinates should be in a SkyCoords object. The names
    of the plates should also be supplied in plateIDs.

    This function will return a string array with field names for objects
    located within an APOGEE field, or blank values if not found within an
    APOGEE field.
    '''
    object_fields = np.full_like(objectcoords, "", dtype=fieldIDs.dtype)
    ufieldIDs, unique_field_indices = np.unique(fieldIDs, return_index=True)
    # Find a relatively reasonable way to verify that unique fieldIDs
    # correspond to unique fieldcoords, and no surprises will occur. This same
    # treatment can't be done with unique_coord_indices because SkyCoords are
    # not orderable.
    # See https://github.com/numpy/numpy/issues/641
    ufieldcoords = fieldcoords[unique_field_indices]
    for i in range(len(ufieldIDs)):
        observable_indices = observable_on_plate(objectcoords, ufieldcoords[i])
        # If an object can be observed in multiple fields, I'd like to know.
        assert(np.all(object_fields[observable_indices] == ""))
        object_fields[observable_indices] = ufieldIDs[i]
    return object_fields

def APOGEE_plate_count(fieldIDs, field_array):
    '''Count the number of objects observed in each field.

    Find the number of objects which were found in each APOGEE field. This 
    function uses the output of the APOGEE_plates function in this module to
    count the number of objects in each field. The fields of interest should be 
    provided in fieldIDs.

    This function returns a dictionary mapping the fieldID to the number of
    objects in that field.
    '''
    fieldcounts = {}
    unique_fields = np.unique(fieldIDs)
    for field in unique_fields:
        field_indices = (field_array == field)
        num_objects = np.count_nonzero(field_indices)
        fieldcounts[field] = num_objects

    return fieldcounts

def extract_mjd(soup):
    '''Extracts the MJD value from a web page.

    The web page should be passed in as a BeautifulSoup object.
    '''
    mjd = int(soup.find(
        "span", style=re.compile("background-color:#CAF1D7")).string)
    return mjd

def vrel_snr_plot(apodwarfs):
    '''Plots vrelerr, SNR, and H relationship.

    Makes a double-plot showing the relationship between relative velocity
    error, signal-to-noise, and H-band magnitude. This function uses only
    APOGEE dwarfs lying within the Kepler field.
    '''
    # Set up good, warn, and bad targets.
    starflag = apodwarfs["starflag"]
#   good_dwarfs = starflag >= 0
    good_dwarfs = starflag & (2**4 + 2**9) != 0
    # These are targets with one of the: BAD_PIXELS (0), VERY_BRIGHT_NEIGHBOR
    # (3), and LOW_SNR (4) flags set.
    # http://www.sdss.org/dr12/algorithms/bitmasks/#APOGEE_TARGET2
    bad_dwarfs = starflag & (2**0 + 2**3 + 2**4) != 0
    warn_dwarfs = np.logical_not(np.logical_or(good_dwarfs, bad_dwarfs))

    Hband = apodwarfs["h"]
    snr = np.ma.masked_equal(apodwarfs["snr"], -9999)
    vel_err = np.ma.masked_equal(
        np.ma.masked_equal(
            apodwarfs["vrelerr"], 999999), -9999)

    binned_results = scipy.stats.binned_statistic(
        Hband[good_dwarfs], snr[good_dwarfs], "median", bins=19, range=(7,14))
    binned_snr = binned_results[0]
    binned_H_edges = binned_results[1]
    binned_H_values = (binned_H_edges[:-1] + 
                         (binned_H_edges[1]-binned_H_edges[0])/2)
    
    apogee_est_SNR = np.array([100, 45, 20, 10])
    apogee_est_H = np.array([11.3, 12.2, 13.3, 14.2])

    plt.subplot(2, 1, 1) 
    plt.scatter(snr[good_dwarfs], Hband[good_dwarfs], marker='x', label="",
                c='k')
#   plt.plot(snr[warn_dwarfs], Hband[warn_dwarfs], 'gx', label="")
#   plt.plot(snr[bad_dwarfs], Hband[bad_dwarfs], 'gx', label="Bad")
    plt.plot(binned_snr, binned_H_values, 'r-', lw=3, label="Median")
    plt.plot(apogee_est_SNR, apogee_est_H, 'r--', label="Wiki Est.")
    plt.title("APOGEE dwarfs in Kepler field")
    plt.ylabel("H")
    plt.xlim(0, 100)
    plt.ylim(14.3, 7)
    plt.legend(loc="lower right")

    binned_results = scipy.stats.binned_statistic(
        snr[good_dwarfs], vel_err[good_dwarfs], "median", bins=19, range=(5,100))
    binned_RV_err = binned_results[0] 
    binned_RV_edges = binned_results[1]
    binned_snr_values = (binned_RV_edges[:-1] + 
                         (binned_RV_edges[1]-binned_RV_edges[0])/2)

    plt.subplot(2, 1, 2) 
    plt.scatter(snr[good_dwarfs], vel_err[good_dwarfs], marker='x',
                c='k')
#   plt.plot(snr[warn_dwarfs], vel_err[warn_dwarfs], 'gx')
#   plt.plot(snr[bad_dwarfs], vel_err[bad_dwarfs], 'gx')
    ax = plt.gca()
    plt.ylabel("RV error (km/s)")
    plt.xlabel("SNR (single visit)")
    plt.xlim(0, 100)
    ax.plot(binned_snr_values, binned_RV_err, 'r-', lw=3)
    inax = inset_axes(ax, width="50%", height="50%", loc=1)
    inax.scatter(snr[good_dwarfs], vel_err[good_dwarfs], marker='x',
                 c='k')
#   inax.plot(snr[warn_dwarfs], vel_err[warn_dwarfs], 'gx')
#   inax.plot(snr[bad_dwarfs], vel_err[bad_dwarfs], 'gx')
    inax.plot(binned_snr_values, binned_RV_err, 'r-', lw=3)
    inax.set_ylim(0, 1.1)
    inax.set_xlim(0, 100)

    plt.figure()
    plt.plot(apodwarfs[good_dwarfs]["mjd"], vel_err[good_dwarfs], "b*")

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

def plot_APOGEE_KIC_teff_DSEP_KIC_radius(
    apogee_teff, kic_teff, dsep_radius, kic_radius, apogee_logg):
    '''Plot the radius of objects with respect to teff.'''
    giant_indices = apogee_logg < 3.5
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, sharex='all', sharey='all')
    ax1.plot(apogee_teff, dsep_radius, 'r*')
    ax1.plot(apogee_teff[giant_indices], dsep_radius[giant_indices], 'bo',
             label="APOGEE giants")
    ax2.plot(kic_teff, dsep_radius, 'r*')
    ax2.plot(kic_teff[giant_indices], dsep_radius[giant_indices], 'bo')
    ax3.plot(apogee_teff, kic_radius, 'r*')
    ax3.plot(apogee_teff[giant_indices], kic_radius[giant_indices], 'bo')
    ax4.plot(kic_teff, kic_radius, 'r*')
    ax4.plot(kic_teff[giant_indices], kic_radius[giant_indices], 'bo')
    ax1.set_xlim((7000, 3500))
    ax3.set_xlabel("APOGEE Teff (K)")
    ax4.set_xlabel("KIC Teff (K)")
    ax1.set_ylabel("DSEP radius (Rsun)")
    ax3.set_ylabel("KIC radius (Rsun)")
    ax1.legend()

    f.suptitle("APOGEE-McQuillan Good Fits")


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

def period_velocity_apogee(
    periods, vsinis, apogee_flags):
    '''Plot the relationship between period & vsini for rapid rotators.

    This will put the rapid rotators which have been observed in APOGEE on a
    plot relating period and vsini.'''
    bad_indices = apogee_flags & 2**23 != 0
    # The 2**14 is a flag called VSINI_WARN. It does not trigger the STAR_BAD
    # or STAR_WARN flags.
    warn_indices = np.logical_and(apogee_flags & (2**7+2**14) != 0,
                                  np.logical_not(bad_indices))
    good_indices = np.logical_not(np.logical_or(bad_indices, warn_indices))

    plt.scatter(periods[good_indices], vsinis[good_indices], s=50, c="g",
                marker="o", label="good")
    plt.scatter(periods[warn_indices], vsinis[warn_indices], s=15, c="m",
                marker="s", label="warn")
    plt.scatter(periods[bad_indices], vsinis[bad_indices], s=15, c="r",
                marker="D", label="bad")
    plt.plot([1, 5], [51, 10], 'k-', label="Rsun")
    plt.plot([1, 5], [51/2.0, 10/2.0], 'k--', label="Rsun (min)")
    plt.plot([1, 5], [51*0.66, 10*0.66], 'b-', label="0.66 Rsun")
    plt.plot([1, 5], [51*0.66/2.0, 10*0.66/2.0], 'b--', label="0.66 Rsun (min)")

    plt.xlabel("Period (day)")
    plt.ylabel("vsini (km/s)")
    plt.ylim(0, 100)
    plt.xlim(1, 5)
#   plt.legend(loc="upper right")

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
    teffs, vsinis, periods, apogee_flags):
    '''Plots the inferred radius vs teff for rapid rotators.

    The inferred radius will basically be vsini * P. Typical bounds on sini
    will also be displayed for clarity. If cool objects have a large radius,
    then this may be indicative of subgiant contamination.'''
    bad_indices = apogee_flags & 2**23 != 0
    # The 2**14 is a flag called VSINI_WARN. It does not trigger the STAR_BAD
    # or STAR_WARN flags.
    warn_indices = np.logical_and(apogee_flags & (2**7+2**14) != 0,
                                  np.logical_not(bad_indices))
    good_indices = np.logical_not(np.logical_or(bad_indices, warn_indices))

    conv = 24*60*60*1e5/6.96e10/2/np.pi

    plt.scatter(
        teffs[good_indices], vsinis[good_indices]*periods[good_indices]*conv, 
        s=50, c="g", marker="o", label="good")
    plt.scatter(
        teffs[warn_indices], vsinis[warn_indices]*periods[warn_indices]*conv,
        s=15, c="m", marker="s", label="warn")
    plt.scatter(
        teffs[bad_indices], vsinis[bad_indices]*periods[bad_indices]*conv, 
        s=15, c="r", marker="D", label="bad")
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
    velocity = 2 * np.pi * radii / period

    return velocity

def plot_velocity_vsini(period, radius, vsini, xvalue):
    '''Plot the expected velocities and the measured vsini.

    Plot the velocity expected from the radius and period of objects, along
    with the measured vsini.'''
    valid_vsini_indices = vsini >= 0
    num_invalid = len(vsini) - np.count_nonzero(valid_vsini_indices)
    valid_period = period[np.where(valid_vsini_indices)]
    valid_radius = radius[np.where(valid_vsini_indices)]
    valid_vsini = period[np.where(valid_vsini_indices)]
    valid_xvalue = period[np.where(valid_vsini_indices)]
    print("Invalid vsinis: {0:d}".format(num_invalid))

    max_vel = period_to_velocities(valid_period, valid_radius)
    obs_vel = valid_vsini

    plt.plot(valid_xvalue, max_vel, 'bo', ms=3, label="Predicted V")
    plt.plot(valid_xvalue, obs_vel, 'rd', label="V sin(i)")
    for i in range(len(xvalue)):
        if max_vel[i] >= obs_vel[i]:
            lc='k'
        else:
            lc='r'
        plt.plot([valid_xvalue[i]]*2, [obs_vel[i], max_vel[i]], ls='-', lc=lc)
    plt.ylabel("Rotational Velocity (km/s)")


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

def select_brightest_targets(
    tblgrp, num=2, magcol="jAperMag3"):
    '''Selects the brightest targets in the groups in his table.

    Selects the brightest objects in each table in each group. The number of
    brightest objects to reserve is given in num, and the column which holds
    the magnitudes are in magcol.'''
    for grp in tblgrp:
        pass

def select_observing_targets(ntargets=75):
    '''Selects a sample of targets that we will try to observe for our run.

    This currently pulls from the McQuillan catalog. Will remove all objects
    that don't match the current Kepler Stellar Parameter pipeline 
    log(g) > 4.25 and 5600 K > Teff > 4850 K. We will define
    tidally-synchronized as having 1 day < Prot < 5 day.
    '''
    mcq = read_McQuillan_catalog()
    kep_stelparms = read_KIC_DR25_catalog()
    mcq_stelparms = au.join_by_id(mcq, kep_stelparms, "KIC", "kepid")
    # Let's keep memory usage low, shall we?
    del(mcq)
    del(kep_stelparms)

    mcq_observing = perform_period_cut(
        perform_teff_cut(
            perform_logg_cut(mcq_stelparms, lowlogg=4.25), 
            lowtemp=4850, hightemp=5600), 
        lowperiod=1, highperiod=5)

def apogee_targets_in_observed_sample(obs, apogee):
    '''Explore the apogee targets that will be observed.

    Plot the total observation sample, as well as the subsample which already
    has APOGEE observations in an HR diagram.

    Additionally, show the objects which '''
    pass


def select_tidally_synchronized_binaries(
    table, pcut=5, lowtemp=4850, hightemp=5600, lowperiod=1, teffcol="Teff",
    pcol="Prot"):
    '''Cuts out the objects that are potentially TSBs.

    This function provides a standardized way to select a sample of Tidally
    Synchronized Binaries according to the prescription of Jen van Saders. This
    function may evolve as TSB selection criteria improve; however, for a
    standard, transparent selection, this will do.

    The current criteria are that TSBs have orbital periods of around 5 days,
    and effective temperatures between 5700 and 4600 K.
    '''
    period_cut = perform_period_cut(table, lowperiod=lowperiod, highperiod=pcut, 
                                    periodcol=pcol)
    temp_cut = perform_teff_cut(period_cut, lowtemp, hightemp, teffcol)

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

def read_villanova_EBs(EBpath=paths.EB_PATH):
    '''Reads in the Villanova Keler EB catalog.'''
    ebcat = Table.read(EBpath, format="ascii.commented_header",
                       header_start=-1)
    return ebcat

def read_synchronized_EB_details(syncpath=paths.SYNC_EB_PATH):
    '''Reads in the Villanova fit parameters for the synchronized EB Catalog.

    Synchronized in this context represents orbital periods between 1-5
    days.'''
    ebcat = Table.read(str(syncpath), format="ascii.commented_header",
                       comment="#", header_start=-1)
    del(ebcat["col16"])
    # There are false aliases in this catalog (leading to duplicate entries).
    # You can find the duplicate entries by doing:
    # [item for item, count in Counter(ebcat["KIC"]).items() if count > 1]
    # I checked the duplicate entries and these ones are the ones that are
    # incorrect.
    dupkics = [4247791, 8167938, 3832716]
    badperiods = [4.0497388, 2.5657024, 2.1701093]
    for kic, period in zip(dupkics, badperiods):
        ebcat.remove_rows(np.argwhere(np.logical_and(
            ebcat["KIC"] == kic, ebcat["period"] == period)).flatten())
    # These objects seem to have multiple periodicities. May be a
    # blended double-eclipsing binary.
    ebcat.remove_rows(np.argwhere(ebcat["KIC"] == 10091110).flatten())
    ebcat.remove_rows(np.argwhere(ebcat["KIC"] == 4150611).flatten())
    # Weird... this has a transit-like thing and a very sharp feature.
    ebcat.remove_rows(np.argwhere(ebcat["KIC"] == 7622486).flatten())
    return ebcat

def remove_Kepler_EBs(maincat, ebcat=None, mainkiccol="KIC"):
    '''Filters out Kepler Eclipsing Binaries.

    Removes the KIC values corresponding to the eclipsing binaries in the
    version of the EB catalog given in paths.EB_PATH.
    '''
    if not ebcat:
        ebcat = read_villanova_EBs()
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
# KOIs #
###############################################################################

def read_KOI_list_Mcquillan(koipath=paths.KOI_PATH):
    '''Read the list of KOIs as of Feb 16, 2017.'''
    kois = Table.read(koipath, format="ascii.csv", data_start=1, data_end=4800, comment="#")
    return kois

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
DLSB_PATH = browse.DEFAULT_DLSB_DB

def mark_DLSB_indices(apogee_ids, dlsb_db=DLSB_PATH):
    '''Mark which apogee IDs correspond to known DLSBs.

    Creates an index array which marks the apogee_ids which are known
    double-line spectroscopic binaries. It looks at the file in at dlsb_db to
    have a list of APOGEE_IDs corresponding to double-lined spectroscopic
    binaries.'''
    dlsbs = browse.read_DLSB_db(db_path=dlsb_db)
    dlsb_indices = au.mark_selections_in_columns(apogee_ids, dlsbs["APOGEE_ID"])
    return np.where(dlsb_indices)

def filter_double_lined_spectroscopic_binaries(
    apocat, apid_col="APOGEE_ID", dlsb_db=DLSB_PATH):
    '''Remove Double-Lined Spectroscopic Binaries by APOGEE_ID'''
    dlsb_indices = mark_DLSB_indices(apocat[apid_col], dlsb_db=dlsb_db)
    filtered_cat = apocat[au.get_complement_indices(dlsb_indices, len(apocat))]
    add_cut_metadata(
        filtered_cat, "Removed DLSBs: see {0} for list".format(dlsb_db))
    return filtered_cat


