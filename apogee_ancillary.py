'''
Module to handle items for the APOGEE ancillary proposal.

Various functions for generating the figures/analysis/tables for the ancillary
science proposals.

Functions for general analysis are:

read_APOGEE_KOI_fields
read_APOGEE_KASC_fields:
    Read in the fields of interest and their coordinates.

targets_in_APOGEE_field:
    Determine which of the targets lie within given APOGEE fields.

APOGEE_plates:
    Return plates that a given target is observable on.

observable_on_plate:
    Determine if a given target is visible on a given plate.

APOGEE_plate_count:
    Count how many objects are observable on each plate.


Functions for generating the figure are:

Functions for generating the output tables are:

APOGEE_Ancillary_table:
    Collect sample and write the table in machine-readable format.

write_APOGEE_proposal_table
    Write the proposed observing sample in machine-readable format.

'''
import random

import numpy as np
import scipy
import numpy.core.defchararray as npstr
from astropy.table import unique, Table, vstack
from astropy.coordinates import SkyCoord
from astropy.io import ascii
import astropy.units as u
import astropy_util as au

import read_catalog as catin
import catalog
import path_config as paths

###############################################################################
# Get the Ancillary sample #
###############################################################################

#########################
# Read in APOGEE fields #
#########################

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

# Also want function that takes list of apogee fields and kic binaries and only
# returns the ones that are in the given fields.
def targets_in_APOGEE_fields(
    apogee_fields, kic_targets, kic_racol="ra", kic_raunit=u.deg, 
    kic_deccol="dec", kic_decunit=u.deg, field_col="APOGEE_Field"):
    '''Determine which KIC targets lie within the given APOGEE fields.

    Returns the subset of kic_targets which can be found in the given APOGEE
    fields. The APOGEE fields that each target can be found in will be in the
    column given by aield_col.'''
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

################################################################################
# Write the figure #
################################################################################
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

###############################################################################
# Table manipulation #
###############################################################################

def mcq_table():
    '''Get the target list from the McQuillan table.
    
    This is a table which requires the APOGEE field, 2MASS designation column,
    RA column, Dec column, coordinate source, H-mag, H-mag source, proper
    motion in RA, proper motion in dec,, proper motion source, number of
    visits, and desired signal-to-noise.'''
    mcq = catin.mcquillan_with_stelparms()
    tidsync = catalog.select_tidally_synchronized_binaries(
        mcq, pcut=5, lowperiod=1, teffcol="teff")
    ancillary_fields = ["K18_070+14", "K19_076+07"]
    fieldtargs = targets_in_APOGEE_fields(ancillary_fields, tidsync)
    ucactable = catin.read_Kepler_UCAC4()
    mcq_ucac = au.join_by_id(fieldtargs, ucactable, "KIC", "kepid",
                             join_type="left")
    shared_kic_set = set(mcq_ucac["KIC"])
    full_ucac_set = set(ucactable["kepid"])
    if shared_kic_set <= full_ucac_set:
        for kepid in shared_kic_set - full_ucac_set:
            print("KIC {0:d} not in UCAC-4.")
    return mcq_ucac

def eb_table():
    '''Get the control list of eclipsing binaries.'''
    ebs = catin.read_villanova_EBs()
    stelparms = catin.read_KIC_DR25_catalog()
    ebs_parms = au.join_by_id(ebs, stelparms, "KIC", "kepid")
    tidsync = catalog.select_tidally_synchronized_binaries(
        ebs_parms, pcut=5, lowperiod=1, teffcol="teff", pcol="period")
    ancillary_fields = ["K18_070+14", "K19_076+07"]
    fieldtargs = targets_in_APOGEE_fields(ancillary_fields, tidsync)
    ucactable = catin.read_Kepler_UCAC4()
    eb_ucac = au.join_by_id(fieldtargs, ucactable, "KIC", "kepid",
                            join_type="left")
    shared_kic_set = set(eb_ucac["KIC"])
    full_ucac_set = set(ucactable["kepid"])
    if shared_kic_set <= full_ucac_set:
        for kepid in shared_kic_set - full_ucac_set:
            print("KIC {0:d} not in UCAC-4.")
    return eb_ucac

def apogee_table():
    '''Get the list of spectroscopic rapid rotators'''
    apo = catin.dr14_with_KIC_stelparms()
    tidsync = catalog.select_tidally_synchronized_binaries(
        apo, pcut=2000, lowperiod=10, teffcol="teff", pcol="VSINI")
    ancillary_fields = ["K18_070+14", "K19_076+07"]
    fieldtargs = targets_in_APOGEE_fields(ancillary_fields, tidsync)
    fieldtargs.rename_column("PMRA", "pmRA")
    fieldtargs.rename_column("PMDEC", "pmDE")
    apo_notrapid = catalog.perform_vscatter_cut(fieldtargs, highv=1)
    apo_needsrv = catalog.perform_cut(apo_notrapid, "NVISITS", highval=4)
    nodlsb = catalog.filter_double_lined_spectroscopic_binaries(apo_needsrv)
    return nodlsb


################################################################################
# Write the Ancillary Table #
################################################################################

def APOGEE_Ancillary_table(outputpath=paths.APOGEE_ANCILLARY_TARGETS_TABLE):
    '''Completely write the APOGEE Ancillary Table to a file.

    First read in the McQuillan Targets.
    Take only tidally-synchronized binary candidates.
    Get PM information from UCAC-4.
    Write out all of the necessary information to the file.
    '''
    mcq = mcq_table()
    # Remove RV variable targets from MDM
    mdmtargs = [11819949, 12736892, 3248885]
    nonobs = au.filter_column_from_subtable(mcq, "KIC", mdmtargs)
    mcq_writetable = nonobs[[
        "APOGEE_Field", "tm_designation", "ra", "dec", "hmag", "pmRA", "pmDE"]]
    mcq_writetable["Order"] = 1
    mcq_writetable["Source"] = "McQuillan"
                            
    # Add spectroscopic rapid rotators.
    apo = apogee_table()
    apo_writetable = apo[[
        "APOGEE_Field", "tm_designation", "ra", "dec", "hmag", "pmRA", "pmDE"]]
    apo_writetable["Order"] = 2
    apo_writetable["Source"] = "APOGEE"

    # Control sample.
    ebs = eb_table()
    # Remove the faint/bright control targets.
    ebs = ebs[np.logical_and(ebs["hmag"] > min(mcq["hmag"]), 
                             ebs["hmag"] < max(mcq["hmag"]))]
    # Select 2 from each field.
    eb_groups = ebs.group_by("APOGEE_Field")
    random.seed("EB CONTROL")
    tablerows = []
    for grp in eb_groups.groups:
        randindices = random.sample(range(len(grp)), 2)
        for ind in randindices:
            tablerows.append(grp[ind])
    ebs_selected = Table(rows=tablerows, names=ebs.colnames)
    eb_writetable = ebs_selected[[
        "APOGEE_Field", "tm_designation", "ra", "dec", "hmag", "pmRA", "pmDE"]]
    eb_writetable["Order"] = 3
    eb_writetable["Source"] = "EB"

    fulltable = vstack([mcq_writetable, apo_writetable, eb_writetable],
                       join_type="exact")
    fulltable.sort(["APOGEE_Field", "Order"])
#    return fulltable
    del(fulltable["Order"])
    write_APOGEE_proposal_table(fulltable)

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
#    ordered_output.sort(apogee_field_col)

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
        str(outputpath), format="ascii.fixed_width", names=names,
        overwrite=True, formats={"PM_RA": ".1f", "PM_DE": ".1f"},
        fill_values=[(ascii.masked, "0.0")], 
        fill_include_names=["PM_RA", "PM_DE"])
