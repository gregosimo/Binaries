import tempfile

from astropy.table import Table, vstack, Column
from astropy.coordinates import SkyCoord, FK5
import astropy.units as u
from astropy.io import fits
from scipy.io import readsav
import numpy as np
import numpy.core.defchararray as npstr
import astropy_util as au
import statop as stat

import path_config as paths
import catalog
import sample_characterization as samp

###############################################################################
# Reading Kepler/APOGEE catalogs #
###############################################################################

#@au.memoized
def read_APOKASC_catalog(
    filepath=paths.APOKASC_PATH):
    '''Reads in the APOKASC catalog.

    The catalog should be located at filepath.
    '''
    apocat = Table.read(filepath, format="fits", character_as_bytes=False)
    return apocat

def read_EHK_catalog(filepath=paths.EHK_PATH):
    '''Reads in the UBV catalog fof the Kepler field.'''
    cat = Table.read(
        str(filepath), format="ascii.csv", 
        names=["RA", "Dec", "U", "U_err", "B", "B_err", "V", "V_err"] )
    return cat

#@au.memoized
def read_McQuillan_catalog(filepath=paths.MCQUILLAN_CATALOG):
    '''Reads in the McQuillan catalog.

    The catalog should be located at filepath.
    '''
    mcquillancat = Table.read(filepath, format="fits")
    return mcquillancat

def read_McQuillan_nondetections(filepath=paths.MCQUILLAN_NONDETECTIONS):
    '''Read in the Mcquillan nondetection table.

    The catalog should be located at filepath.'''
    nondets = Table.read(filepath, format="fits")
    return nondets

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
    desired_cols = [
        "kepid", "tm_designation", "teff", "teff_err1", "teff_err2", "logg", 
        "logg_err1", "logg_err2", "feh", "feh_err1", "feh_err2", "mass", 
        "mass_err1", "mass_err2", "radius", "radius_err1", "radius_err2", 
        "kepmag", "dist", "dist_err1", "dist_err2", "ra", "dec", "st_quarters", 
        "teff_prov", "logg_prov", "feh_prov", "jmag", "jmag_err", "hmag", 
        "hmag_err", "kmag", "kmag_err", "av", "av_err1", "av_err2"]
    kiccat = Table.read(
        str(kicpath), format="ascii.ipac", include_names=desired_cols)
    fix_table_coordinates_units(kiccat, "ra", "dec")
    # This was necessary because of a bug in astropy where "dex" isn't
    # considered a valid flux unit.
    # https://github.com/astropy/astropy/issues/7279
    kiccat["feh"].unit = "Dex"
    kiccat["feh_err1"].unit = "Dex"
    kiccat["feh_err2"].unit = "Dex"
    au.set_numeric_fill_values(kiccat, -9999)
    return kiccat

def read_Pinsonneault_2012_catalog(pinpath=paths.PINSONNEAULT_CORRECTIONS):
    '''Read the corrected catalog from Pinsonneault et al (2012).'''
    desired_cols = [
        "KIC", "SDSS-Teff", "e_SDSS-Teff", "E_SDSS-Teff", "K-Teff", "Flag"]
    pincat = Table.read(
        str(pinpath), format="ascii.cds", include_names=desired_cols,
        fill_values=('-9999', '0'))
    au.set_numeric_fill_values(pincat, -9999)
    return pincat

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

def read_van_Saders_missing_targets(targpath=paths.VAN_SADERS_MISSING):
    '''Read the missing targets in Jen's sample.
    
    These objects were not observed in APOGEE 1. Maybe they were observed in
    APOGEE2?'''
    missing_table = Table.read(targpath, format="ascii.basic", names=(
        "Plate", "KIC", "RA", "DEC", "H", "nExp", "SNR"))
    return missing_table

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

def read_Kepler_UCAC4(upath=paths.UCAC_KEPLER_PATH):
    '''Reads the UCAC4 Table for the full Kepler field.
    
    The UCAC-4 table was obtained from the CDS XMatch service.'''
    pm_table = Table.read(
        str(upath), format="ascii.csv", include_names=("ID", "pmRA", "pmDE"))
    pm_table.rename_column("ID", "kepid")
    return pm_table

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

def read_UCAC4_EB_Tidsync(
        upath=paths.UCAC_EB_TIDSYNC_PATH, kic_col="KIC"):
    '''Reads the UCAC4 table for tidally-synchronized eclipsing binaries.

    The UCAC-4 table was obtained from Vizier
    (http://cdsbib.u-strasbg.fr/cgi-bin/cdsbib?2012yCat.1322....0Z).'''
    UCAC_table = Table.read(str(upath), format="ascii.basic", delimiter=";",
                          data_start=3, header_start=0)
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

def read_TGAS_Kepler(tgas_kep_path=paths.TGAS_KEPLER_OVERLAP):
    '''Read the TGAS table for stars overlapping with Kepler.

    If this file doesn't exist, use the astroquery package to get it from
    the Vizier xMatch service.'''
    desired_cols = ["kepid", "ra_ep2000", "dec_ep2000", "parallax",
                    "parallax_error", "pmra", "pmra_error", "pmdec",
                    "pmdec_error"]
    try:
        tgas = Table.read(tgas_kep_path, format="ascii.csv",
                          include_names=desired_cols)
    except FileNotFoundError:
        tgas = XMatch.query(cat1="J/ApJS/229/30/catalog", cat2="GAIA DR1 TGAS")
        tgas.write(str(tgas_kep_path), format="ascii.csv")

    return tgas

def read_Gaia_DR2_Kepler(gaia_dr2_kep_path=paths.GAIA_BERGER_OVERLAP):
    '''Read the Gaia DR2 table of stars overlapping with Kepler.

    If this file doesn't exist, use the astroquery package to get it from the
    Vizier xMatch service.'''
    dr2 = Table.read(gaia_dr2_kep_path, format="fits")
    au.set_numeric_fill_values(dr2, -9999)
    return dr2

def read_Berger_DR2_Kepler(berger_dr2_kep=paths.BERGER_DR2_KEPLER):
    '''Read in the Gaia parameters in Berger et al (2018).'''
    desired_cols = [
        "KIC", "source_id", "dis", "disep", "disem", "rad", "radep", "radem", 
        "class\\\\"]
    berger = Table.read(berger_dr2_kep, format="ascii.csv",
                        include_names=desired_cols, delimiter="&")
    berger.rename_column("class\\\\", "class")
    au.set_numeric_fill_values(berger, -9999)
    berger["class"] = np.asarray(npstr.replace(berger["class"], "\\\\", ""),
                                 dtype=np.int)
    return berger

def APOGEE_TGAS(tgas_kep_path=paths.TGAS_KEPLER_OVERLAP,
                apopath=paths.DR14_ALLSTAR_PATH):
    '''Read in stars observed in both APOGEE and TGAS.'''
    apo = dr14_with_KIC_stelparms()
    tgas = read_TGAS_Kepler()
    
    apo_tgas = au.join_by_id(apo, tgas, "kepid", "kepid")
    return apo_tgas


def read_TGAS_McQuillan_APOGEE_overlap_tidsync(
    path=paths.TGAS_MCQUILLAN_APOGEE_TIDSYNC_PATH):
    '''Read the TGAS information for McQuillan/APOGEE targets.

    Reads in the information from TGAS which is available for the sample of
    targets which overlap between McQuillan and APOGEE.'''
    tgas_table = Table.read(str(path), format="fits")
    return tgas_table

def read_dr14_allVisit(allvisitpath=paths.DR14_ALLVISIT_PATH, opt="kepler"):
    '''Read the DR14 allVisit file.

    This function reads the l31c.1 version of the allVisit file. The table can 
    be optimized for either Kepler targets or Pleiades targets by specifying 
    opt="kepler" or "pleiades".  Optimization means that only the objects in 
    the RA range corresponding to either the Kepler field or the Pleiades will 
    be loaded, which will significantly reduce memory usage. Only the summary 
    data table will be read, which should contain everthing necessary for the 
    APOGEE pipeline.
    
    WARNING: If opt is set to "", then full table will take 
    several hours to fit into memory.
    '''

    if opt:
        allvisit_hdus = fits.open(str(allvisitpath), memmap=True)
        allvisit_indices = allvisit_hdus[2]
        if opt.lower() == "kepler":
            index_start = allstar_indices.data[279]
            index_end = allstar_indices.data[302]
        elif opt.lower() == "pleiades":
            index_start = allstar_indices.data[55]
            index_end = allstar_indices.data[58]
        else:
            raise ValueError(
                "Don't understand optmization: {0}".format(opt))
        # This will only have targets in the optimized RA range.
        allvisit_kepler = allvisit_hdus[1].data[index_start:index_end]
        allvisit_hdus.close()
        # Convert from recarray to Table
        allvisit = Table(allvisit_kepler)
    else:
        allvisit = Table.read(str(allvisitpath), format="fits")
        allvisit["APOGEE_ID"] = npstr.rstrip(allvisit["APOGEE_ID"])

    return allvisit

def read_dr14_allStar(allstarpath=paths.DR14_ALLSTAR_PATH, opt="kepler"):
    '''Reads the allStar file for DR14.
    
    Reads in the allStar table for DR14. The table can be optimized for either
    Kepler targets or Pleiades targets by specifying opt="kepler" or "pleiades". 
    Optimization means that only the objects in the RA range corresponding to 
    either the Kepler field or the Pleiades will be loaded, which will 
    significantly reduce memory usage. Only the Summary data table will be read 
    in, which should contain everything necessary for the APOGEE pipeline.

    WARNING: If opt is set to "", then full table will take 
    several hours to fit into memory.
    '''
    desired_cols = [
        "APOGEE_ID", "LOCATION_ID", "J", "J_ERR", "H", "H_ERR", "K", "K_ERR",
        "RA", "DEC", "APOGEE_TARGET1", "APOGEE_TARGET2", "APOGEE_TARGET3",
        "TARGFLAGS", "NVISITS", "STARFLAG", "STARFLAGS", "ANDFLAG", "ANDFLAGS",
        "VHELIO_AVG", "VSCATTER", "VERR", "VERR_MED", "APOGEE2_TARGET1",
        "APOGEE2_TARGET2", "APOGEE2_TARGET3", "SNREV", "MIN_H", "MAX_H",
        "MIN_JK", "MAX_JK", "TEFF", "TEFF_ERR", "LOGG", "LOGG_ERR",
        "VMICRO", "VMACRO", "VSINI", "M_H", "M_H_ERR", "ALPHA_M",
        "ALPHA_M_ERR", "ASPCAPFLAG", "ASPCAPFLAGS", "FE_H", "PMRA", "PMDEC",
        "PM_SRC", "ALL_VISITS", "VISITS"]
    if opt:
            allstar_hdus = fits.open(str(allstarpath), memmap=True)
            allstar_indices = allstar_hdus[2]
            if opt.lower() == "kepler":
                index_start = allstar_indices.data[279]
                index_end = allstar_indices.data[302]
            elif opt.lower() == "pleiades":
                index_start = allstar_indices.data[55]
                index_end = allstar_indices.data[58]
            elif opt.lower() == "m67":
                index_start = allstar_indices.data[131]
                index_end = allstar_indices.data[134]
            else:
                raise ValueError(
                    "Don't understand optmization: {0}".format(opt))
            # This will only have targets in the optimized RA range.
            allstar_kepler = allstar_hdus[1].data[index_start:index_end]
            allstar_hdus.close()
            # Convert from recarray to Table
            allstar = Table(allstar_kepler)
    else:
        allstar = Table.read(
            str(allstarpath), format="fits", character_as_bytes=False)

    short_allstar = allstar[desired_cols]
    short_allstar["LOGG_FIT"] = allstar["FPARAM"][:,1]

    au.mask_numeric_fill_values(short_allstar, -9999)
    au.mask_numeric_fill_values(short_allstar, -9999.99)
    return short_allstar

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

def read_Garcia_periods(rottable=paths.GARCIA_PERIODS):
    '''Reads in the rotation information from Garcia et al (2014).

    This table will have KICs, rotation periods, and other assorted
    information.'''
    rafapers = Table.read(rottable, format="fits")
    return rafapers

def read_Garcia_prevsample(
    chap_sdss=paths.CHAPLIN_DWARFS_SDSS, chap_irfm=paths.CHAPLIN_DWARFS_IRFM,
    chap_bruntt=paths.CHAPLIN_DWARFS_BRUNTT, 
        chap_input=paths.CHAPLIN_DWARFS_INPUT):
    '''Reads in the Chaplin sample.

    Garcia claims to have run their period algorithm on a set of 540 targets.
    However, when checking the Chaplin et al (2014) sample, there are only 518
    targets, which implies that there were additional stars analyzed by Garcia
    et al (2014) not listed in Chaplin.

    Garcia et al (2014) note in Fig 3 that only 297 have both rotation periods 
    and asteroseismic spacings, which is what we get when we cross-match the
    Chaplin sample with the Garcia et al (2014) sample. So I don't know where
    the additional stars came from.'''
    inputtab = Table.read(chap_input, format="ascii.cds")

    return inputtab

def read_DR14_original_KIC(kicpath=paths.DR14_ORIG_KIC):
    '''Reads in the original KIC for DR14 targets.

    This table will have original KIC parameters for objects observed with
    DR14.'''
    kictable = read_MAST_file(kicpath)
    return kictable

def read_abridged_original_KIC(kicpath=paths.ORIG_KIC_ABRIDGED):
    '''Read in the original KIC for Kepler targets.

    This table will read in the original KIC parameters for all Kepler
    targets.'''
    desired_cols = ["Kepler ID", "Teff (deg K)", "Log G (cm/s/s)"]
    kictable = read_MAST_file(kicpath)
    kictable = kictable[desired_cols]
    au.set_numeric_fill_values(kictable, -9999)
    return kictable

def read_Dressing_Charbonneau_table(dcpath=paths.DRESSING_CHARBONNEAU_PROPS):
    '''Read in the cool stellar table of Dressing and Charbonneau (2013).

    This table contains revised Teff and log(g)s of a set of Kepler stars
    analyzed in Dressing and Charbonneau (2013).
    '''
    dctable = Table.read(dcpath, format="fits")
    return dctable

def read_El_Badry_Single_Stars(elb_single_path=paths.EL_BADRY_SINGLE):
    '''Read in the single stars from the APOGEE binary analysis.

    This table contains the 2MASS IDs for the stars which were determined to be
    single from the spectral analysis.'''
    singletable = Table.read(
        elb_single_path, format="ascii.csv", data_start=0, names=["APOGEE_ID"])
    return singletable

def read_El_Badry_SB1(elb_sb1=paths.EL_BADRY_SB1):
    '''Read in the SB1s from the APOGEE binary analysis.

    This table contains the 2MASS IDs for the stars which were determined to be
    single-lined spectroscopic binaries from the spectral analysis.'''
    sb1table = Table.read(elb_sb1, format="ascii.csv")
    return sb1table

def read_El_Badry_SB2(elb_sb2=paths.EL_BADRY_SB2):
    '''Read in the SB2s from the APOGEE binary analysis.

    This table contains the 2MASS IDs for the stars which were determined to be
    double-lined spectroscopic binaries from the spectral analysis.'''
    sb2table = Table.read(elb_sb2, format="ascii.csv")
    return sb2table

def read_El_Badry_hidden_triples(elb_hidden_trip=paths.EL_BADRY_HIDDEN_TRIPLE):
    '''Read in the SB2s with hidden triples from the APOGEE binary analysis.

    This table contains the 2MASS IDs for the stars which were determined to be
    double-lined spectroscopic binaries with RV trends indicating a hidden
    third component from the spectral analysis.'''
    triptable = Table.read(elb_hidden_trip, format="ascii.csv")
    return triptable

def read_El_Badry_SB3(elb_sb3=paths.EL_BADRY_SB3):
    '''Read in the SB3s from the APOGEE binary analysis.

    This table contains the 2MASS IDs for the stars which were determined to be
    triple-lined spectroscopic binaries from the spectral analysis.'''
    sb3table = Table.read(elb_sb3, format="ascii.csv")
    return sb3table

def combined_El_Badry_multiplicity(
        elb_single_path=paths.EL_BADRY_SINGLE, elb_sb1=paths.EL_BADRY_SB1,
        elb_sb2=paths.EL_BADRY_SB2,
        elb_hidden_trip=paths.EL_BADRY_HIDDEN_TRIPLE,
         elb_sb3=paths.EL_BADRY_SB3):
    '''Combine stellar parameters from all multiple El Badry papers.

    The new table will have revised APOGEE IDs, Teffs, log(g), and [Fe/H].'''
    wantedcols = ["APOGEE_ID", "T_eff [K]", "log g [dex]", "[Fe/H] [dex]"]
    sb1s = read_El_Badry_SB1()[wantedcols]
    sb2s = read_El_Badry_SB2()[wantedcols]
    hidden_triples = read_El_Badry_hidden_triples()[wantedcols]
    sb3s = read_El_Badry_SB3()[wantedcols]

    combotable = vstack([sb1s, sb2s, hidden_triples, sb3s])
    return combotable

def read_California_Kepler_Spectroscopy(
    cks=paths.CALIFORNIA_KEPLER_SPECTROSCOPY):
    '''Read in the data from the California Kepler Survey.

    This table contains spectroscopic parameters from the California Kepler
    Survey. Essentially has Teff, logg, [Fe/H] and vsini from two different
    pipelines.'''
    ckstable = Table.read(cks, format="ascii.cds")
    return ckstable

def read_Geller_M67(geller=paths.HEAD_DIR / "aj518354t2_mrt.txt"):
    '''Read in the M67 WOCS data.'''
    fulldata = Table.read(geller, format="ascii.cds")
    # I want to have regular RA/Dec columns.
    coords = SkyCoord(
        ra=fulldata["RAh"]+fulldata["RAm"]/60+fulldata["RAs"]/60/60,
        dec=fulldata["DEd"] + fulldata["DEm"]/60 + fulldata["DEs"]/60/60,
        unit=(u.hourangle, u.degree))
    fulldata["RA"] = coords.ra
    fulldata["DEC"] = coords.dec
    return fulldata

###############################################################################
# Joined catalogs #
##############################################################################
#
# These functions get catalogs which I use often, and are smaller than the
# individual catalogs put together. Caching these instead of the full catalogs 
# will hopefully lead to more efficient memory use.

@au.shortcut_file(paths.SHORTCUT_MCQUILLAN_STELLPARM)
def mcquillan_with_stelparms(
    mcq_path=paths.MCQUILLAN_CATALOG, kic_path=paths.KIC_CATALOG,
        gaia_path=paths.GAIA_DR2_KEPLER_OVERLAP,
        origpath=paths.ORIG_KIC_ABRIDGED,
        pinpath=paths.PINSONNEAULT_CORRECTIONS):
    '''Read McQuillan catalog with full KIC stellar parameters.

    Read in the McQuillan detections along with the KIC DR25 stellar
    parameters.
    '''
    mcq = read_McQuillan_catalog(mcq_path)
    stellcat = stelparms_triple_KIC(origpath, pinpath, kic_path)
    del(stellcat["kic"])
    del(stellcat["KIC"])
    mcquillancat = au.join_by_id(
        mcq, stellcat, "KIC", "kepid", join_type="left")
    trim_McQuillan_catalog(mcquillancat)
    return mcquillancat

@au.shortcut_file(paths.SHORTCUT_MCQUILLAN_NONDET_STELLPARM)
def mcquillan_nondetections_with_stelparms(
    mcq_path=paths.MCQUILLAN_NONDETECTIONS, kic_path=paths.KIC_CATALOG,
        gaia_path=paths.GAIA_DR2_KEPLER_OVERLAP):
    '''Read the McQuillan nondetections with full KIC stellar parameters.

    Read in the McQuillan nondetections along with the KIC DR25 stellar
    parameters.
    '''
    mcq = read_McQuillan_nondetections(mcq_path)
    stellcat = stelparms_triple_KIC()
    del(stellcat["kic"])
    del(stellcat["KIC"])
    mcquillancat = au.join_by_id(
        mcq, stellcat, "KIC", "kepid", join_type="left")
    mcquillancat["teff"][mcquillancat["teff"].mask] = mcquillancat["Teff"][
        mcquillancat["teff"].mask]
    mcquillancat["logg"][mcquillancat["logg"].mask] = mcquillancat["log_g_"][
        mcquillancat["logg"].mask]
    trim_McQuillan_catalog(mcquillancat)
    return mcquillancat

@au.shortcut_file(paths.SHORTCUT_MCQUILLAN_FLICKER)
def mcquillan_flicker_loggs(
    mcq_path=paths.MCQUILLAN_CATALOG, flicker_path=paths.FLICKER_LOGG):
    '''Read Flicker catalog for McQuillan objects.'''
    mcq = read_McQuillan_catalog(mcq_path)[["KIC"]]
    flickercat = read_flicker_loggs(flicker_path)
    mcq_flicker = au.join_by_id(mcq, flickercat, "KIC", "KIC")
    mcq_flicker.remove_columns(["kepmag", "Teff"])
    return mcq_flicker

def create_joined_APOKASC_McQuillan_catalog(
        apofile=paths.APOKASC_PATH, mcquillanfile=paths.MCQUILLAN_CATALOG):
    '''Creates a joint APOKASC/McQuillan catalog.

    All Kepler objects which are measured in both the McQuillan sample as well
    as in APOGEE are left in the remaining table. If the tables have already
    been read, they may be specified in the apocat and mcquillancat parameters.
    If these parameters are left as None, they will be read from apofile and
    mcquillanfile first.
    '''
    apocat = read_APOKASC_catalog(apofile)
    mcquillancat = read_McQuillan_catalog(mcquillanfile)[["KIC"]]

    combocat = au.join_by_id(apocat, mcquillancat, "KEPLER_INT", "KIC")
    return combocat

def APOKASC_Huber_KIC(
    apofile=paths.APOKASC_PATH, huberfile=paths.KIC_CATALOG):
    '''Create a joined catalog with APOKASC and McQuillan.'''
    apocat = read_APOKASC_catalog(apofile)
    kics = read_KIC_DR25_catalog(huberfile)
    apokic = au.join_by_id(
        apocat, kics, "KEPLER_INT", "kepid", join_type="left")
    return apokic

@au.shortcut_file(paths.SHORTCUT_MCQUILLAN_APOKASC)
def APOKASC_with_McQuillan_KIC(
    apofile=paths.APOKASC_PATH, huberfile=paths.KIC_CATALOG,
    mcquillanfile=paths.MCQUILLAN_CATALOG):
    '''Join the APOKASC catalog with McQuillan periods.

    This function will also have the Huber KIC stellar parameters.'''
    apocat = APOKASC_Huber_KIC(apofile, huberfile)
    mcq = read_McQuillan_catalog(mcquillanfile).copy()
    trim_McQuillan_catalog(mcq)
    combined_table = au.join_by_id(
        apocat, mcq, "KEPLER_INT", "KIC", join_type="left")

    return combined_table

def garcia_dr14(
    apofile=paths.DR14_ALLSTAR_PATH, kicfile=paths.KIC_CATALOG, 
    garciafile=paths.GARCIA_PERIODS):
    '''Created a joined Garcia/Huber/DR14 sample.

    All those objects which were observed by Garcia et al (2014) will also have
    APOGEE DR14 parameters as well as Huber et al (2014) stellar parameters.'''
    apokic = dr14_with_KIC_stelparms(apofile, kicfile)
    garcia = read_Garcia_periods(garciafile)
    garcia_apo = au.join_by_id(garcia, apokic, "KIC", "kepid")
    return garcia_apo

@au.shortcut_file(paths.SHORTCUT_MCQUILLAN_DR14)
def mcquillan_dr14_overlap(
    mcq_path=paths.MCQUILLAN_CATALOG, apopath=paths.DR14_ALLSTAR_PATH):
    '''Read the overlap sample between McQuillan and APOGEE DR14.'''
    mcq_col = mcquillan_with_stelparms(mcq_path)[["tm_designation"]]
    dr14 = read_dr14_allStar(allstarpath=apopath)
    mcq_dr14 = catalog.join_by_2MASS_key(mcq_col, dr14, "tm_designation", "APOGEE_ID")
    return mcq_dr14

@au.shortcut_file(paths.SHORTCUT_MCQUILLAN_DR14_KIC)
def mcquillan_dr14_overlap_with_stelparms(
    mcq_path=paths.MCQUILLAN_CATALOG, apopath=paths.DR14_ALLSTAR_PATH,
    kicpath=paths.KIC_CATALOG):
    '''Read in the McQuillan/APOGEE overlap with KIC parameters.'''
    dr14_stelparms = dr14_with_KIC_stelparms(apopath, kicpath)
    mcq = read_McQuillan_catalog(mcq_path)
    del(mcq["Teff"])
    del(mcq["log_g_"])
    del(mcq["Mass"])
    fullcat = au.join_by_id(mcq, dr14_stelparms, "KIC", "kepid")
    return fullcat


@au.shortcut_file(paths.SHORTCUT_MCQUILLAN_EHK)
def mcquillan_photometry(
    mcq_path=paths.MCQUILLAN_CATALOG, photo_path=paths.EHK_PATH):
    '''Reads in photometry from the EHK catalog for McQuillan targets.'''
    mcq = read_McQuillan_catalog(mcq_path)[["KIC", "_RA", "_DE"]]
    mcq_photo = catalog.add_Everett_photometry(mcq, "_RA", "_DE")

    return mcq_photo

@au.shortcut_file(paths.SHORTCUT_BRUNTT_DR14)
def bruntt_dr14_overlap(
    brunttpath=paths.BRUNTT_PATH, apopath=paths.DR14_ALLSTAR_PATH):
    '''Read in the Bruntt et al (2013) targets that are in APOGEE.

    This will also filter out the objects which don't have good ASPCAP fits.
    '''
    bruntt = read_Bruntt_catalog(brunttpath)
    kiccat = read_KIC_DR25_catalog()
    brunttkic = au.join_by_id(bruntt, kiccat, "KIC", "kepid", join_type="left",
                              conflict_suffixes=("_Bruntt", "_Huber"))
    apo = read_dr14_allStar(apopath, opt="kepler")
    fullcat = catalog.join_by_2MASS_key(
        brunttkic, apo, "tm_designation", "APOGEE_ID", 
        conflict_suffixes=("_Bruntt", "_APOGEE"))

    goodcat = catalog.good_aspcap_fits(fullcat)
    return goodcat

@au.shortcut_file(paths.SHORTCUT_MCQUILLAN_EBS)
def mcquillan_ebs(
    mcq_path=paths.MCQUILLAN_CATALOG, ebpath=paths.EB_PATH):
    '''Read in the Villanova EBs in the McQuillan catalog.'''
    mcq = read_McQuillan_catalog(mcq_path)[["KIC"]]
    ebs = read_villanova_EBs(ebpath)

    mcq_ebs = au.join_by_id(ebs, mcq, "KIC", "KIC")
    return mcq_ebs

@au.shortcut_file(paths.SHORTCUT_PLEIADES_APOGEE)
def Stauffer_APOGEE_overlap(
    stauffer_path=paths.STAUFFER_VSINI_PATH, apopath=paths.DR14_ALLSTAR_PATH):
    '''Read targets observed by both Stauffer & Hartmann (1987) and APOGEE.'''
    apo = read_dr14_allStar(apopath, opt="pleiades")
    stauffer = read_Stauffer_Pleiades(stauffer_path)
    simbad_translation = read_SIMBAD_file("simbad_Stauffer_2MASS_ids.txt")
    tmass_names = np.array([j[:23] for j in simbad_translation["identifier"]])
    translation_table = Table([simbad_translation["typed ident"], tmass_names],
                              names=("ident", "2mass"))
    stauffer_translate = au.join_by_id(
        stauffer, translation_table, "Star", "ident", join_type="left")
    joined_table = catalog.join_by_2MASS_key(
        apo, stauffer_translate, "APOGEE_ID", "2mass", join_type="inner")

    return joined_table

@au.shortcut_file(paths.SHORTCUT_APOGEE_KIC)
def dr14_with_KIC_stelparms(
    apopath=paths.DR14_ALLSTAR_PATH, kicpath=paths.KIC_CATALOG,
    origpath=paths.ORIG_KIC_ABRIDGED, pinpath=paths.PINSONNEAULT_CORRECTIONS):
    '''Read in Kepler DR14 targets with Huber stellar parameters.'''
    # Note that this table is fully cross-matched with Gaia!
    # There are no targets without matching Gaia detections.
    apo = dr14_with_ElBadry()
    kiccat = stelparms_triple_KIC(origpath, pinpath, kicpath)
    apokic = catalog.join_by_2MASS_key(
        apo, kiccat, "APOGEE_ID", "tm_designation", join_type="inner")
    return apokic

@au.shortcut_file(paths.SHORTCUT_APOKASC_KIC)
def APOKASC_with_KIC_stelparms(
    apopath=paths.APOKASC_PATH, kicpath=paths.KIC_CATALOG,
    origpath=paths.ORIG_KIC_ABRIDGED):
    '''Read in the latest APOKASC catalog with Huber stellar parameters.'''
    apo = read_APOKASC_catalog(apopath)
    kiccat = stelparms_with_original_KIC(kicpath, origpath)
    apokic = au.join_by_id(apo, kiccat, "KEPLER_INT", "kepid")
    return apokic

def stelparms_with_original_KIC(parmpath=paths.KIC_CATALOG,
                                kicpath=paths.ORIG_KIC_ABRIDGED):
    '''Read in the Huber and original KIC stellar parameters.'''
    kiccat = stelparms_with_Gaia(parmpath)
    origcat = read_abridged_original_KIC(kicpath)
    orig_subtable = Table([
        origcat["Kepler ID"], origcat["Teff (deg K)"], 
        origcat["Log G (cm/s/s)"]], names=(
            "kepid", "KIC Teff", "KIC logg"))
    newcat = au.join_by_id(kiccat, orig_subtable, "kepid", "kepid",
                           join_type="left")
    return newcat

def stelparms_triple_KIC(
    origpath=paths.ORIG_KIC_ABRIDGED, pinpath=paths.PINSONNEAULT_CORRECTIONS,
    huberpath=paths.KIC_CATALOG):
    '''Create a table with the original, Pinsonneault, and Huber parameters.

    The original KIC parameters came from the analysis in Brown et al (2011).
    The Pinsonneault et al (2012) analysis corrected the effective temperatures
    to put them on the SDSS system. Finally Huber et al (2014) reanalyzed the
    whole KIC to use the best available data and fit the parameters as well as
    uncertainties using DSEP isochrones.'''
    hubercat = stelparms_with_original_KIC(huberpath)
    pinsonneaultcat = read_Pinsonneault_2012_catalog(pinpath)
    joinedcat = au.join_by_id(
        hubercat, pinsonneaultcat, "kepid", "KIC", join_type="left")
    return joinedcat

def ebs_with_stelparms(ebpath=paths.EB_PATH, kic_path=paths.KIC_CATALOG,
                       gaia_path=paths.GAIA_DR2_KEPLER_OVERLAP):
    '''Read in Eclipsing Binaries with full stellar parameters.'''
    ebs = read_villanova_EBs(ebpath)
    ebs.remove_columns(["kmag", "Teff"])
    stellcat = stelparms_triple_KIC(kic_path)
    del(stellcat["kic"])
    del(stellcat["KIC"])
    ebcat = au.join_by_id(
        ebs, stellcat, "KIC", "kepid", join_type="left")
    return ebcat

@au.shortcut_file(paths.SHORTCUT_GAIA_KEPLER)
def stelparms_with_Gaia(
        parmpath=paths.KIC_CATALOG, gaiapath=paths.GAIA_DR2_KEPLER_OVERLAP):
    kiccat = read_KIC_DR25_catalog(parmpath)
    gaiacat = read_Gaia_DR2_Kepler()

    joinedcat = au.join_by_id(kiccat, gaiacat, "kepid", "kic", join_type="left")
    catalog.generate_abs_mag_column_with_errors(
        joinedcat, "kmag", "kmag_err", "M_K", "M_K_err1", "M_K_err2",
        samp.AV_to_AK, samp.AV_err_to_AK_err,
        null_value=np.nan)
    return joinedcat

def dr14_with_ElBadry():
    '''Add in El Badry parameters to the DR14 allStar table.'''
    apogee = read_dr14_allStar()
    wantedcols = ["APOGEE_ID", "T_eff [K]", "log g [dex]", "[Fe/H] [dex]"]
    elbadry = combined_El_Badry_multiplicity()
    singles = read_El_Badry_Single_Stars()
    singles_apo = au.extract_subtable_from_column(
        apogee, "APOGEE_ID", singles["APOGEE_ID"])
    singles_params = Table(
        singles_apo[["APOGEE_ID", "TEFF", "LOGG_FIT", "FE_H"]],
                    names=wantedcols)
    full_elbadry = vstack([singles_params, elbadry])

    combotab = catalog.join_by_2MASS_key(
        apogee, fullelbadry, "APOGEE_ID", "APOGEE_ID", join_type="left")

    return combotab

    



###########
# Helpers #
###########

def trim_McQuillan_catalog(cat):
    '''Limit McQuillan catatalog to just period information.

    Remove the KIC parameters and other not-as-important parameters from the
    McQuillan catalog. Useful if supplementing an existing table with the
    McQuillan periods.'''
    cat.remove_columns(["Teff", "log_g_", "Mass", "_RA", "_DE", "Ref"])
    


################################################################################
# Processed Catalogs #
################################################################################
# These catalogs are to easily reproduce commonly-used catalogs. They will be
# in a state of flux depending on what "commonly-used" entails, and if more
# processing will need to occur. These catalogs are to ensure
# reproducibility in case the ipython console needs to be terminated.
def cool_dwarfs():
    '''Get the sample of cool dwarfs in the McQuillan/APOGEE DR14 sample.

    These are the sample of dwarfs where we wouldn't expect stellar evolution
    to play a large role. By this point, they should be well-divided into
    massive subgiants and less-massive dwarfs without much in-between.
    '''
    mcq_parms = mcquillan_with_stelparms()
    dr14 = mcquillan_dr14_overlap()
    mcq_dr14 = catalog.join_by_2MASS_key(
        mcq_parms, dr14, "tm_designation", "tm_designation")

    good_mcq_dr14 = catalog.good_aspcap_fits(mcq_dr14)
    cool_good_mcq_dr14 = catalog.perform_teff_cut(good_mcq_dr14, hightemp=5450, 
                                                  teffcol="teff")

    cleaned = catalog.filter_double_lined_spectroscopic_binaries(
        catalog.filter_pulsators(cool_good_mcq_dr14))

    return cleaned

def asteroseismic_sample():
    '''Get the asteroseismic sample in the McQuillan/APOGEE DR14 sample.'''
    mcq_parms = mcquillan_with_stelparms()
    dr14 = mcquillan_dr14_overlap()
    mcq_dr14 = catalog.join_by_2MASS_key(
        mcq_parms, dr14, "tm_designation", "tm_designation")

    apokasc = create_joined_APOKASC_McQuillan_catalog()
    apokasc_dwarfradii = catalog.filter_invalid_APOGEE_entries(
        apokasc, "RADIUS_DW")

    mcq_apokasc_dr14 = au.join_by_id(
        mcq_dr14, apokasc_dwarfradii, "KIC", "KIC", join_type="inner",
        conflict_suffixes=("_DR14", "_APOKASC"))

    good_mcq_apokasc_dr14 = catalog.good_aspcap_fits(
        catalog.filter_double_lined_spectroscopic_binaries(
            catalog.filter_pulsators(mcq_apokasc_dr14)), "ASPCAPFLAG")

    return good_mcq_apokasc_dr14
    

###############################################################################
# Ancillary Catalogs #
###############################################################################


# Eclipsing Binaries

def read_villanova_EBs(EBpath=paths.EB_PATH):
    '''Reads in the Villanova Keler EB catalog.'''
    ebcat = Table.read(EBpath, format="ascii.commented_header",
                       header_start=-1)
    ebcat.remove_column("col11")
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

# KOIs

def read_KOI_list(koipath=paths.KOI_PATH):
    '''Read the list of KOIs as of Feb 16, 2017.'''
    kois = Table.read(koipath, format="ascii.csv", data_start=1, data_end=4800, comment="#")
    return kois

# KepVIM catalog

def read_KepVIM_catalog(
    KepVIMpath="/home/regulus/simonian/Binaries/KepVIM.fits"):
    '''Reads in the KepVIM catalog (Makarov & Goldin 2016).

    Catalog contains objects whose centroid positions in Kepler change with
    their variability.'''
    kepvim = Table.read(KepVIMpath, format="fits")
    return kepvim

# Bruntt et al (2013)

def read_Bruntt_catalog(brunttpath=paths.BRUNTT_PATH):
    '''Read in the stellar parameter table from Bruntt et al. (2013).

    This table contains the results from analyzing 93 Kepler targets, and
    comparing the stellar parameters determined through the KIC, spectroscopy,
    and asteroseismology.'''
    bruntt = Table.read(brunttpath, format="votable")
    return bruntt

# Stauffer & Hartmann (1987)

def read_Stauffer_Pleiades(vsini_file=paths.STAUFFER_VSINI_PATH):
    '''Read the vsinis from Table one of Stauffer & Hartmann 1987.'''

    tbl = Table.read(str(vsini_file), format="ascii.basic", fill_values=[
        ('---', '0'), ('', '0')])
    separate_limit(tbl, ["vsini"], eqdelim="")
    return tbl

# Stauffer (1982)

def read_Stauffer_1982_photometry(photfile=paths.STAUFFER_1982_TABLE1_PATH):
    '''Read the photometry table from Stauffer (1982).'''
    tbl = Table.read(
        str(photfile), format="ascii.no_header", guess=False, 
        fill_values=[("---", "0"), ("", "0")]) 

    staufftable = Table([
        npstr.add(tbl["col1"], tbl["col2"]), tbl["col3"],
        npstr.strip(tbl["col4"], "()").astype(np.float), tbl["col5"], 
        npstr.strip(tbl["col6"], "()").astype(np.float), tbl["col7"], 
        npstr.strip(tbl["col8"], "()").astype(np.float), tbl["col9"], 
        npstr.strip(tbl["col10"], "()").astype(np.float), tbl["col11"]], names=[
            "Star", "V", "Verr", "B-V", "B-Verr", "V-R", "V-Rerr",
            "R-I", "R-Ierr", "V-I"], masked=True)

    for initcol, stauffcol in zip(tbl.colnames[2:], staufftable.colnames[1:]):
        staufftable[stauffcol].mask = tbl[initcol].mask

    return staufftable

#################
# Service Files #
#################

# SIMBAD

def read_SIMBAD_file(simbadfile, output_path=paths.HEAD_DIR):
    '''Read in a SIMBAD table and extract the 2MASS IDs.'''
    tbl = Table.read(
        str(output_path / simbadfile), format="ascii.basic", comment="", 
        guess=False, delimiter="|", data_start=2, data_end=101)
    return tbl

def read_UKIRT_file(resultfile):
    '''Reads in a file from UKIRT.'''
    results = read_split_file(resultfile, format="ascii.commented_header")
    return results

def read_MAST_file(resultfile):
    '''Reads in a file from MAST.'''
    results = read_split_file(resultfile, format="ascii.csv", data_start=2)
    return results

# WEBDA

def read_Pleiades_WEBDA_UBV_photometry(
        photfile=paths.WEBDA_PLEIADES_UBV_PHOTOMETRY):
    '''Read in the UBV photometry archived on WEBDA for the Pleiades.'''
    tbl = Table.read(str(photfile), format="ascii.tab", data_start=2)
    return tbl

def read_Pleiades_WEBDA_VRIk_photometry(
        photfile=paths.WEBDA_PLEIADES_VRIk_PHOTOMETRY):
    '''Read in the VRIk photometry archived on WEBDA for the Pleiades.'''
    tbl = Table.read(str(photfile), format="ascii.tab", data_start=2)
    return tbl

def read_Pleiades_WEBDA_vsini(
        vsinifile=paths.WEBDA_PLEIADES_VSINI):
    '''Read in the vsini table archived on WEBDA for the Pleiades.'''
    tbl = Table.read(str(vsinifile), format="ascii.tab", data_start=2)
    return tbl

def read_Pleiades_WEBDA_coordinates(
        coofile=paths.WEBDA_PLEIADES_COORDINATES):
    '''Read in the table of Pleiades coordinates on WEBDA.'''
    tbl = Table.read(str(coofile), format="ascii.tab", data_start=2)
    coords = SkyCoord(tbl["RA"], tbl["Dec"], unit=(u.hourangle, u.degree),
                      frame="fk4", equinox="B1950")
    del(tbl["RA"])
    del(tbl["Dec"])
    modern_frame = FK5(equinox="J2000")
    modern_coords = coords.transform_to(modern_frame)
    tbl["RA"] = modern_coords.ra
    tbl["Dec"] = modern_coords.dec
    return tbl

###############################################################################
# Utilities #
###############################################################################

def separate_limit(table, limcols, updelim="<", lowdelim=">", eqdelim="=",
                   coltemplate="{0} lim"):
    '''Takes limcols from a table and splits them into limit columns.

    For all columns in the list of limcols, this function will split them into
    a limit column and a numerical value column. The column will change dtype
    to be numerical. The limit will have a column name as determined by
    coltemplate, which should be a format string which takes the column name 
    as the first argument.
    '''
    for col in limcols:
        strcol = table[col]
        valcol, limcol = split_limit_col(strcol, updelim, lowdelim, eqdelim)
        del(table[col])
        table[col] = valcol
        table[coltemplate.format(col)] = limcol

def split_limit_col(initcol, updelim="<", lowdelim=">", eqdelim="=",
                    dtype=np.float):
    '''Splits a column into a limit and numerical value column.

    One problem with table representations of limits is that the symbols for
    limits cause the columns to be represented as a string, not as a numerical
    limit. Therefore, this function splits a string column into two arrays:
    one with a limit representation, another with the numerical values.
    '''
    # If the column was not read as a string, then just return it.
    oldmask = initcol.mask
    limcol = stat.generate_limit(None, len(initcol))
    try:
        upperindices = np.where(npstr.startswith(initcol, updelim))
        lowerindices = np.where(npstr.startswith(initcol, lowdelim))
    except TypeError:
        print("{0} is not a string column. Ignoring.".format(initcol.name))
    else:
        # If initcol is not a string column, we want to skip all of these
        # string operations.
        initcol = npstr.lstrip(initcol, updelim)
        initcol = npstr.lstrip(initcol, lowdelim)
        if eqdelim is not "":
            eqindices = np.where(npstr.startswith(initcol, eqdelim))
            initcol = npstr.lstrip(initcol, eqdelim)
        limcol[upperindices] = stat.UPPER
        limcol[lowerindices] = stat.LOWER
    newcol = np.ma.asanyarray(initcol, dtype=dtype)
    newcol.mask = oldmask
    # If there is a mask, then we want to ensure that the masked values are
    # considered to be invalid data points.
    try:
        limcol[newcol.mask] = stat.NA
    except AttributeError:
        pass

    return newcol, limcol

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

# Split files for large online database queries

def read_split_file(filepath, **kwargs):
    '''Reads a file that has been split into multiple parts.

    This function essentially re-reads a table which has been split according
    to the large_table_multiple_files_split function. However, it uses the
    existing files in the directory instead of predicting using table
    information. For example, if filepath is /path/to/foo.txt, this will find
    foo.txt, if it exists, or foo.0.txt, foo.1.txt, foo.2.txt, etc. and read
    them all in if it doesn't.

    Since the input table isn't used, this means that when writing split files,
    care has to be taken to delete all previous queries made with them.

    Keyword arguments to be passed to the underlying Table.read function should
    be supplied in kwargs.
    '''
    try:
        inputtable = Table.read(str(filepath), **kwargs)
    except FileNotFoundError as f:
        inputfiles = find_split_files(filepath)
        table_pieces = []
        for inputfile in inputfiles:
            table_piece = Table.read(inputfile, **kwargs)
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
    base, ext = catalog.split_filename(filename)
    glob_pattern = catalog.format_split_filename(base, "*", ext)
    files = folder.glob(glob_pattern)
    return files
