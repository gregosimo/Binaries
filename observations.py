import random

import numpy as np
import astropy_util as au

import catalog
import read_catalog as catin
import sed

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

    mcq_observing = catalog.filter_pulsators(mcq_observing, KICcol="KIC")

    apogee = catin.mcquillan_dr14_overlap()
    mcq_observing = catalog.join_by_2MASS_key(
        mcq_observing, apogee, "tm_designation", "tm_designation", 
        join_type="left", conflict_suffixes=("_KIC", "_APOGEE"))
    del(apogee)
    
    # Remove APOGEE giants
    mcq_observing = catalog.perform_logg_cut(
        mcq_observing, lowlogg=3.5, loggcol="TEFF")

    # Remove objects which are already observed to be RV variable
    mcq_observing["VSCATTER"] = mcq_observing["VSCATTER"].filled(-9999.0)
    mcq_observing = catalog.perform_vscatter_cut(
        mcq_observing, highv=1, vcol="VSCATTER")
    mcq_observing["VSCATTER"] = np.ma.masked_values(mcq_observing["VSCATTER"],
                                                 -9999.0)
    
    # Remove objects which have been observed enough to indicate non
    # RV-variability.
    mcq_observing["NVISITS"] = mcq_observing["NVISITS"].filled(0)
    mcq_observing = catalog.perform_cut(mcq_observing, "NVISITS", highval=4)
    mcq_observing["NVISITS"] = np.ma.masked_values(mcq_observing["NVISITS"], 0)

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
    prioritytable = mcq_observing[~mcq_observing["APOGEE_ID"].mask]
    mcq_observing = mcq_observing[mcq_observing["APOGEE_ID"].mask]

    
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
