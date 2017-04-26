from astropy.table import Table, vstack, Column
import astropy.units as u
from astropy.io import fits
from scipy.io import readsav
import numpy as np
import numpy.core.defchararray as npstr
import astropy_util as au

import path_config as paths
import catalog

###############################################################################
# Reading catalogs #
###############################################################################

@au.memoized
def read_APOKASC_catalog(
    filepath=paths.APOKASC_PATH):
    '''Reads in the APOKASC catalog.

    The catalog should be located at filepath.
    '''
    apocat = Table.read(filepath, format="fits")
    return apocat

def read_EHK_catalog(filepath=paths.EHK_PATH):
    '''Reads in the UBV catalog fof the Kepler field.'''
    cat = Table.read(
        str(filepath), format="ascii.csv", 
        names=["RA", "Dec", "U", "U_err", "B", "B_err", "V", "V_err"] )
    return cat

@au.memoized
def read_McQuillan_catalog(filepath=paths.MCQUILLAN_CATALOG):
    '''Reads in the McQuillan catalog.

    The catalog shoul be located at filepath.
    '''
    mcquillancat = Table.read(filepath, format="fits")
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
    '''Read the KIC DR25 Stellar Parameter catalog.'''
    kiccat = Table.read(str(kicpath), format="ascii.ipac")
    fix_table_coordinates_units(kiccat, "ra", "dec")
    return kiccat

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
    tempcut = catalog.perform_teff_cut(rafa_kic, lowtemp=0, hightemp=5500,
                               teffcol="teff")
    giantcut = catalog.perform_logg_cut(tempcut, lowlogg=3.5, loggcol="logg")
    return giantcut

def read_flicker_loggs(loggpath=paths.FLICKER_LOGG):
    '''Read in the catalog of flicker log(g)s based on Bastien et al (2016).

    This will have the KICs, and log(g)s based on the 8-hour flicker boxcar
    window.'''
    flicker_loggs = Table.read(loggpath, format="ascii.cds")
    return flicker_loggs

###################
# Joined catalogs #
###################
#
# These functions get catalogs which I use often, and are smaller than the
# individual catalogs put together. Caching these instead of the full catalogs 
# will hopefully lead to more efficient memory use.

@au.memoized
def mcquillan_with_stelparms(
    mcq_path=paths.MCQUILLAN_CATALOG, kic_path=paths.KIC_CATALOG):
    '''Read McQuillan catalog with full KIC stellar parameters.

    Read in the McQuillan detections along with the KIC DR25 stellar
    parameters.
    '''
    mcq = read_McQuillan_catalog(mcq_path)
    stellcat = read_KIC_DR25_catalog(kic_path)
    mcquillancat = au.join_by_id(
        mcq, stellcat, "KIC", "kepid", join_type="left")
    mcquillancat.remove_columns(["Teff", "log_g_", "Mass", "_RA", "_DE", "Ref"])
    return mcquillancat

@au.memoized
def mcquillan_flicker_loggs(
    mcq_path=paths.MCQUILLAN_CATALOG, flicker_path=paths.FLICKER_LOGG):
    '''Read Flicker catalog for McQuillan objects.'''
    mcq = read_McQuillan_catalog(mcq_path)[["KIC"]]
    flickercat = read_flicker_loggs(flicker_path)
    mcq_flicker = au.join_by_id(mcq, flickercat, "KIC", "KIC")
    mcq_flicker.remove_columns(["kepmag", "Teff"])
    return mcq_flicker

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
# Combine these two.            
@au.memoized
def mcquillan_apokasc_dwarfs(
    mcq_path=paths.MCQUILLAN_CATALOG, apokasc_path=paths):
    pass

@au.memoized
def mcquillan_dr14_overlap(
    mcq_path=paths.MCQUILLAN_CATALOG, apopath=paths.DR14_ALLSTAR_PATH):
    '''Read the overlap sample between McQuillan and APOGEE DR14.'''
    mcq_col = mcquillan_with_stelparms(mcq_path)[["tm_designation"]]
    dr14 = read_dr14_allStar(allstarpath=apopath)
    mcq_dr14 = catalog.join_by_2MASS_key(mcq_col, dr14, "tm_designation", "APOGEE_ID")
    return mcq_dr14

@au.memoized
def mcquillan_photometry(
    mcq_path=paths.MCQUILLAN_CATALOG, photo_path=paths.EHK_PATH):
    '''Reads in photometry from the EHK catalog for McQuillan targets.'''
    mcq = read_McQuillan_catalog(mcq_path)[["KIC", "_RA", "_DE"]]
    ehk = read_EHK_catalog()
    mcq_photo = au.join_by_ra_dec(
        mcq, ehk, "_RA", "_DE", "RA", "Dec", join_type="left")

    return mcq_photo

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

def read_KOI_list_Mcquillan(koipath=paths.KOI_PATH):
    '''Read the list of KOIs as of Feb 16, 2017.'''
    kois = Table.read(koipath, format="ascii.csv", data_start=1, data_end=4800, comment="#")
    return kois

def read_KepVIM_catalog(
    KepVIMpath="/home/regulus/simonian/Binaries/KepVIM.fits"):
    '''Reads in the KepVIM catalog (Makarov & Goldin 2016).

    Catalog contains objects whose centroid positions in Kepler change with
    their variability.'''
    kepvim = Table.read(KepVIMpath, format="fits")
    return kepvim
