import random

import numpy as np
import numpy.core.defchararray as npstr
import astropy_util as au
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
import matplotlib.pyplot as plt

import catalog
import read_catalog as catin
import sed
import path_config as paths
import hrplots as hr

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
    period_cut = catalog.perform_period_cut(
        table, lowperiod=lowperiod, highperiod=pcut, periodcol=pcol)
    temp_cut = catalog.perform_teff_cut(period_cut, lowtemp, hightemp, teffcol)
    loggcut = catalog.perform_logg_cut(temp_cut, lowlogg=logg, loggcol=loggcol)

    return loggcut

@au.shortcut_file(paths.SHORTCUT_MDM_NOMAGCUT)
def select_targets_before_magcut():
    '''Selects a sample of targets meeting the Jen van Saders criterion.

    These targets meet the criterion of Jen van Saders to have P < 5 day (with
    a floor at P=1 day to avoid ellipsoidal binaries). There is a cut of log(g)
    > 3.5 to avoid giants. And the cut of 5600 K > Teff > 4850 K to avoid
    rapidly-rotating subgiants on the hot end, and rapidly-rotating K-dwarfs
    on the cool end.

    It will also look through APOGEE to weed out known RV variable/non-variable
    systems as well as giants.
    '''
    mcq = catin.mcquillan_with_stelparms()

    mcq_observing = select_tidally_synchronized_binaries(
        mcq, pcut=5, lowperiod=1, lowtemp=4850, hightemp=5600, logg=3.5,
        teffcol="teff", pcol="Prot", loggcol="logg")

    preprocess_size = len(mcq_observing)
    print("Sample after major cuts: {0:d}".format(preprocess_size))

    mcq_observing = catalog.filter_pulsators(mcq_observing, KICcol="KIC")
    pulsators_removed = len(mcq_observing)
    print("Sample after removing pulsators: {0:d}".format(pulsators_removed))

    apogee = catin.mcquillan_dr14_overlap()
    mcq_observing = catalog.join_by_2MASS_key(
        mcq_observing, apogee, "tm_designation", "tm_designation", 
        join_type="left", conflict_suffixes=("_KIC", "_APOGEE"))
    del(apogee)
    
    # Remove APOGEE giants
    autodwarfs = mcq_observing["LOGG"] < 0
    mcq_observing["LOGG"][mcq_observing["LOGG"] < 0] = 9999.0
    mcq_observing = catalog.perform_logg_cut(
        mcq_observing, lowlogg=3.5, loggcol="LOGG")
    mcq_observing["LOGG"][mcq_observing["LOGG"] == 9999.0] = -9999.0
    giants_removed = len(mcq_observing)
    print("Sample after removing APOGEE Giants: {0:d}".format(giants_removed))

    # Remove objects which are already observed to be RV variable
    mcq_observing["VSCATTER"] = mcq_observing["VSCATTER"].filled(-9999.0)
    mcq_observing = catalog.perform_vscatter_cut(
        mcq_observing, highv=1, vcol="VSCATTER")
    mcq_observing["VSCATTER"] = np.ma.masked_values(mcq_observing["VSCATTER"],
                                                 -9999.0)
    giants_removed = len(mcq_observing)
    print("Sample after removing APOGEE giants: " + giants_removed)
    
    # Remove objects which have been observed enough to indicate non
    # RV-variability.
    mcq_observing["NVISITS"] = mcq_observing["NVISITS"].filled(0)
    mcq_observing = catalog.perform_cut(mcq_observing, "NVISITS", highval=4)
    mcq_observing["NVISITS"] = np.ma.masked_values(mcq_observing["NVISITS"], 0)

    # Also remove eclipsing binaries.
    mcq_observing = catalog.remove_Kepler_EBs(mcq_observing, mainkiccol="kepid")

    return mcq_observing

def select_observing_targets(ntargets=50, tbins=3, pbins=3, Vcut=14):
    '''Selects a sample of targets that we will try to observe for our run.

    This currently pulls from the McQuillan catalog. Will remove all objects
    that don't match the current Kepler Stellar Parameter pipeline 
    log(g) > 4.25 and 5600 K > Teff > 4850 K. We will define
    tidally-synchronized as having 1 day < Prot < 5 day.

    We'll also filter out the known Kepler pulsators.

    For targets with APOGEE observations, remove those with logg < 3.5
    '''
    mcq_observing = select_targets_before_magcut()

    # Perform a magnitude cut.
    mcq_phot = catin.mcquillan_photometry()
    mcq_observing = au.join_by_id(mcq_observing, mcq_phot, "kepid", "KIC")
    del(mcq_phot)
    missing_Vs = mcq_observing["V"].mask
    JK_interp = sed.color_to_color_DSEP_interpolator(
        "J-Ks", "V-H", {"V": 1, "J": 1, "H": 1, "Ks": 1}, age=2, init_mass=0.9)
    missing_JKs = (mcq_observing['jmag'][missing_Vs] -
                   mcq_observing["kmag"][missing_Vs]).filled()
    mcq_observing["V"][missing_Vs] = (
        JK_interp(missing_JKs) + mcq_observing["hmag"][missing_Vs])
    mcq_observing = catalog.perform_cut(mcq_observing, "V", highval=14)

    # Prioritize objects with APOGEE spectra.
    prioritytable = mcq_observing[~npstr.endswith(
        mcq_observing["APOGEE_ID"], "N/A")]
    mcq_observing = mcq_observing[npstr.endswith(
        mcq_observing["APOGEE_ID"], "N/A")]

    
    # Ensure reproducibility.
    random.seed(a="20170529")
    # Bin the sample by period to ensure that all periods are represented.
    pbins = np.trunc(mcq_observing["Prot"])
    period_groups = mcq_observing.group_by(pbins)
    # This will hold each teff subgroup for the period.
    bingroups = []
    for grp in period_groups.groups:
        teffbins = np.trunc(grp["teff"] / 150)
        teffgroups = grp.group_by(teffbins)
        for teffgrp in teffgroups.groups:
            randommag = au.random_permutation(teffgrp)
            bingroups.append(randommag)

    randgroups = au.random_permutation(bingroups)
    datarows = au.roundrobin(*randgroups)
    for row in au.take(ntargets-len(prioritytable), datarows):
        prioritytable.add_row(row)
    prioritytable.meta = mcq_observing.meta

    return prioritytable

def select_SLSB_target():
    '''Selects the single-lined spectroscopic binary to observe.'''
    mcq = catin.mcquillan_with_stelparms()

    mcq_observing = select_tidally_synchronized_binaries(
        mcq, pcut=5, lowperiod=1, lowtemp=4850, hightemp=5600, logg=3.5,
        teffcol="teff", pcol="Prot", loggcol="logg")

    mcq_observing = catalog.filter_pulsators(mcq_observing, KICcol="KIC")

    # In this case I only want APOGEE targets. So join_type="inner"
    apogee = catin.mcquillan_dr14_overlap()
    mcq_observing = catalog.join_by_2MASS_key(
        mcq_observing, apogee, "tm_designation", "tm_designation", 
        join_type="inner", conflict_suffixes=("_KIC", "_APOGEE"))
    del(apogee)
    
    # Remove APOGEE giants
    autodwarfs = mcq_observing["LOGG"] < 0
    mcq_observing["LOGG"][mcq_observing["LOGG"] < 0] = 9999.0
    mcq_observing = catalog.perform_logg_cut(
        mcq_observing, lowlogg=3.5, loggcol="LOGG")
    mcq_observing["LOGG"][mcq_observing["LOGG"] == 9999.0] = -9999.0

    # Remove double-lined spectroscopic binaries.
    mcq_observing = catalog.filter_double_lined_spectroscopic_binaries(
        mcq_observing)

    # Return random targets with RV scatter > 0.5 km/s.
    mcq_observing = catalog.perform_vscatter_cut(mcq_observing, 0.5) 

    # Perform a magnitude cut.
    mcq_phot = catin.mcquillan_photometry()
    mcq_observing = au.join_by_id(mcq_observing, mcq_phot, "kepid", "KIC")
    del(mcq_phot)
    missing_Vs = mcq_observing["V"].mask
    JK_interp = sed.color_to_color_DSEP_interpolator(
        "J-Ks", "V-H", {"V": 1, "J": 1, "H": 1, "Ks": 1}, age=2, init_mass=0.9)
    missing_JKs = (mcq_observing['jmag'][missing_Vs] -
                   mcq_observing["kmag"][missing_Vs]).filled()
    mcq_observing["V"][missing_Vs] = (
        JK_interp(missing_JKs) + mcq_observing["hmag"][missing_Vs])

    return mcq_observing
    
def select_DLSB_target():
    '''Selects the double-lined spectroscopic binary to observe.'''
    mcq = catin.mcquillan_with_stelparms()

    mcq_observing = select_tidally_synchronized_binaries(
        mcq, pcut=5, lowperiod=1, lowtemp=4850, hightemp=5600, logg=3.5,
        teffcol="teff", pcol="Prot", loggcol="logg")

    mcq_observing = catalog.filter_pulsators(mcq_observing, KICcol="KIC")

    # In this case I only want APOGEE targets. So join_type="inner"
    apogee = catin.mcquillan_dr14_overlap()
    mcq_observing = catalog.join_by_2MASS_key(
        mcq_observing, apogee, "tm_designation", "tm_designation", 
        join_type="inner", conflict_suffixes=("_KIC", "_APOGEE"))
    del(apogee)
    
    # Remove APOGEE giants
    autodwarfs = mcq_observing["LOGG"] < 0
    mcq_observing["LOGG"][mcq_observing["LOGG"] < 0] = 9999.0
    mcq_observing = catalog.perform_logg_cut(
        mcq_observing, lowlogg=3.5, loggcol="LOGG")
    mcq_observing["LOGG"][mcq_observing["LOGG"] == 9999.0] = -9999.0

    # Remove double-lined spectroscopic binaries.
    dlsb_indices = catalog.mark_DLSB_indices(mcq_observing["APOGEE_ID"])
    mcq_observing = mcq_observing[dlsb_indices]

    # Return random targets with RV scatter > 0.5 km/s.
    mcq_observing = catalog.perform_vscatter_cut(mcq_observing, 0.5) 

    # Perform a magnitude cut.
    mcq_phot = catin.mcquillan_photometry()
    mcq_observing = au.join_by_id(mcq_observing, mcq_phot, "kepid", "KIC")
    del(mcq_phot)
    missing_Vs = mcq_observing["V"].mask
    JK_interp = sed.color_to_color_DSEP_interpolator(
        "J-Ks", "V-H", {"V": 1, "J": 1, "H": 1, "Ks": 1}, age=2, init_mass=0.9)
    missing_JKs = (mcq_observing['jmag'][missing_Vs] -
                   mcq_observing["kmag"][missing_Vs]).filled()
    mcq_observing["V"][missing_Vs] = (
        JK_interp(missing_JKs) + mcq_observing["hmag"][missing_Vs])

    return mcq_observing
    

def sample_biases(prioritytable):
    '''Make figures to explore biases in sample.

    This figure will evaluate the bias of the given sample compared to a sample
    without a magnitude cut. This will look at the Teff, Logg, Rotation period,
    rotation period amplitude.'''
    base = select_targets_before_magcut()

    # Also want to set aside APOGEE targets.
    apoindices = ~npstr.endswith(prioritytable["APOGEE_ID"], "0.0")
    sample_teff = prioritytable["teff"]
    apogee_teff = prioritytable["teff"][apoindices]
    full_teff = base["teff"]

    sample_logg = prioritytable["logg"]
    apogee_logg = prioritytable["logg"][apoindices]
    full_logg = base["logg"]
     
    sample_Prot = prioritytable["Prot"]
    apogee_Prot = prioritytable["Prot"][apoindices]
    full_Prot = base["Prot"]

    sample_Rper = prioritytable["Rper"]
    apogee_Rper = prioritytable["Rper"][apoindices]
    full_Rper = base["Rper"]

    sample_ra = prioritytable["ra"]
    apogee_ra = prioritytable["ra"][apoindices]
    full_ra = base["ra"]

    sample_dec = prioritytable["dec"]
    apogee_dec = prioritytable["dec"][apoindices]
    full_dec = base["dec"]

    sample_color = "#1f78b4"
    apogee_color = "#b2df8a"
    full_color = "#a6cee3"
    plt.figure()
    plt.plot(full_teff, full_logg, marker="o", ms=4, mfc="w", mec=full_color,
             label="Full sample", ls="none")
    plt.plot(sample_teff, sample_logg, marker="o", ms=6, c=sample_color,
             label="Observing", ls="none")
    plt.plot(apogee_teff, apogee_logg, marker="o", ms=6, c=apogee_color,
             label="APOGEE Obs.", ls="none")
    plt.xlabel("Huber Teff (K)")
    plt.ylabel("Huber Log(g)")
    hr.invert_y_axis()
    hr.invert_x_axis()
    plt.legend(loc="center right")
    plt.figure()
    plt.plot(full_Prot, full_Rper, marker="o", ms=4, mfc="w", mec=full_color,
             label="Full sample", ls="none")
    plt.plot(sample_Prot, sample_Rper, marker="o", ms=6, c=sample_color,
             label="Observing", ls="none")
    plt.plot(apogee_Prot, apogee_Rper, marker="o", ms=6, c=apogee_color,
             label="APOGEE Obs.", ls="none")
    plt.xlabel("Period (day)")
    plt.ylabel("Period amplitude (ppm)")
    plt.legend(loc="upper center")
    plt.figure()
    plt.plot(full_dec, full_ra, marker="o", ms=4, mfc="w", mec=full_color,
             label="Full sample", ls="none")
    plt.plot(sample_dec, sample_ra, marker="o", ms=6, c=sample_color,
             label="Observing", ls="none")
    plt.plot(apogee_dec, apogee_ra, marker="o", ms=6, c=apogee_color,
             label="APOGEE Obs.", ls="none")
    plt.xlabel("DEC")
    plt.ylabel("RA")
    plt.legend(loc="upper right")

def write_target_list_for_MDM(target_table, filename="MDM_list.txt",
                              output_path=paths.HEAD_DIR):
    '''Write a target list containing KIC ID, ra, dec, estimated V-mag.'''
    output_table = target_table[["kepid", "ra", "dec"]]

    photometry = catin.mcquillan_photometry()
    phot_table = au.join_by_id(target_table, photometry, "kepid", "KIC",
                               join_type="left")
    Vmag = phot_table["V"]

    masked_entries = phot_table[Vmag.mask]
    DSEP_lookup = {"V": 1, "J": 1, "H": 1, "K": 1}
    JK_color = masked_entries["jmag"] - masked_entries["kmag"]
    VH_interp = sed.color_to_color_DSEP_interpolator(
        "J-K", "V-H", DSEP_lookup, lowT=3000)
    VH_color = VH_interp(JK_color.filled())
    Vmag[Vmag.mask] = VH_color + masked_entries["hmag"]
    
    output_table["EstV"] = Vmag
    output_table.write( 
        str(output_path / filename), format="ascii.csv", comment=False)
    

    assert(np.all(~phot_table["V"].mask))
    output_table["EstV"] = phot_table["V"]
    output_table.write( 
        str(output_path / filename), format="ascii.csv", comment=False)

def write_MDM_target_list(target_table, filename="MDM_list.txt",
                          output_path=paths.HEAD_DIR):
    '''Write the target table to the given filename.'''

    kics = npstr.add("KIC", target_table["kepid"].astype(np.str))
    Vmag = npstr.add("P=", target_table["Prot"].astype("U3"))
    objname = npstr.add(kics, Vmag)

    coords = SkyCoord(target_table["ra"], target_table["dec"], frame="icrs")

    write_jskycalc_file(objname, coords, filename=filename,
                        output_path=output_path)

def write_jskycalc_file(names, coords, filename="MDM_list.txt", 
                        output_path=paths.HEAD_DIR):
    '''Write the targets in the table to JSkycalc format.

    Whitespace in the name will be converted to underscores. The coordinates
    should be included in a SkyCoord object, which shoul also contain the
    equinox.'''
    objnames = npstr.replace(names, " ", "_")

    ra_h = coords.ra.hms.h.astype(np.int)
    ra_m = coords.ra.hms.m.astype(np.int)
    ra_s = coords.ra.hms.s
    dec_d = coords.dec.dms.d.astype(np.int)
    dec_m = coords.dec.dms.m.astype(np.int)
    dec_s = coords.dec.dms.s
    equinox = ["2000"]*len(objnames)

    output_table = Table([
        objnames, ra_h, ra_m, ra_s, dec_d, dec_m, dec_s, equinox], names=(
        "Name", "hh", "mm", "ss", "dd", "mm ", "ss ", "equinox"))

    output_table.write(str(output_path / filename),
                       format="ascii.commented_header")
