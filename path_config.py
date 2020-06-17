from pathlib import Path
import os

HOME_DIR = Path.home()
HEAD_DIR = Path(os.environ["RESEARCH"])

CASAGRANDE_TABLE_PATH = HEAD_DIR
CASAGRANDE_TABLE_4 = CASAGRANDE_TABLE_PATH / "Casagrande_10_Table_4.txt" 
CASAGRANDE_TABLE_5 = CASAGRANDE_TABLE_PATH / "Casagrande_10_Table_5.txt"

DSEP_OUTPUT = HEAD_DIR / "DSEP_out"

DSEP_PATH = HEAD_DIR / "DSep"
DSEP_ISOCHRONES= DSEP_PATH / "isochrones"
DSEP_TRACKS = DSEP_PATH / "Tracks"
DSEP_INTERPOLATOR_EXECUTABLE = DSEP_PATH / "iso_interp_feh"
DSEP_SPLITTER_EXECUTABLE = DSEP_PATH / "isolf_split"

MIST_PATH = HEAD_DIR / "MIST"
# Because I don't need the full grid of ages, the abridged directory just has
# the ages for immediate use. Flip back to the old grid if interpolation over
# age needs to be done.
MIST_ISOCHRONES = MIST_PATH / "MIST_v1.2_abridged_vvcrit0.0_UBVRIplus"

BARAFFE_PATH = HEAD_DIR / "Baraffe"

YREC_PATH = HEAD_DIR / "YREC"

JEN_FAST_LAUNCH_SGB_PATH = HEAD_DIR / "simonian_sgb_fast_launch.dat"
JEN_SLOW_LAUNCH_SGB_PATH = HEAD_DIR / "simonian_sgb_slow_launch.dat"

CLUSTER_PATH = HEAD_DIR / "Clusters"
PLEIADES_PATH = CLUSTER_PATH / "Pleiades"
BOUY_TABLE_PATH = PLEIADES_PATH
BOUY_TABLE_2 = BOUY_TABLE_PATH / "Bouy_15_Table_2.fits"
BOUY_TABLE_6 = BOUY_TABLE_PATH / "Bouy_15_Table_6.fits"
BOUY_GOOD_MEMBERS = BOUY_TABLE_PATH / "Bouy_good_members.fits"
STAUFFER_PLEIADES_MASSES = HEAD_DIR / "ajaa2dfet3_mrt.txt"

M67_PATH = CLUSTER_PATH / "M67"
M67_TABLE = M67_PATH / "m67ensemblecal.pmem.dat"
M67_TWOMASS = M67_PATH / "2mass_photometry.ipac"
M67_EPIC = M67_PATH / "M67_WOCS_EPIC_match.txt"

MCQUILLAN_CATALOG = HEAD_DIR / "McQuillan.fit"
MCQUILLAN_NONDETECTIONS = HEAD_DIR / "McQuillan_nondet.fit"
APOKASC_PATH = HEAD_DIR / "APOKASC_cat_v4.2.4.fits"
KIC_PULSATORS = HEAD_DIR / "pulsators.kic"
UKIRT_RESULTS = HEAD_DIR / "ukirt_results.csv.gz"
SANTOS_PERIODS = HEAD_DIR / "Santos_Table_3.votable"
GARCIA_PERIODS = HEAD_DIR / "Garcia14_Periods.fit"
HUBER_CATALOG = HEAD_DIR / "huber_kic_parameters.txt"
KIC_CATALOG = HEAD_DIR / "KIC_DR25.tbl"
ORIG_KIC = HEAD_DIR / "kic.txt.gz"
PINSONNEAULT_CORRECTIONS = HEAD_DIR / "apjs479928t7_mrt.txt"
DR14_ORIG_KIC = HEAD_DIR / "DR14_orig_KIC.txt"
ORIG_KIC_ABRIDGED = HEAD_DIR / "full_kepler_KIC.txt"
KEPLER_GALEX_BEST = HEAD_DIR / "KGGoldStandard.csv.gz"
EHK_README = HEAD_DIR / "hlsp_kplrubv_readme.txt"
EHK_PATH = HEAD_DIR / "EHK2012catalog.dat"
VAN_SADERS_SDSS = HEAD_DIR / "cool_dwarfs.sav"
VAN_SADERS_MAST = HEAD_DIR / "Jen_SDSS_Kepler_MAST_output.csv"
VAN_SADERS_MISSING = HEAD_DIR / "vansaders_targets_missing.dat"
JEN_APOGEE_1 = HEAD_DIR / "Jen_APOGEE_results_0.csv"
JEN_APOGEE_2 = HEAD_DIR / "Jen_APOGEE_results_1.csv"
APOGEE_KOI_FIELDS = HEAD_DIR / "apogee_koi.txt"
APOGEE_KASC_FIELDS = HEAD_DIR / "apogee_apokasc_fields.txt"
APOGEE_DWARF_PATH = HEAD_DIR / "apogee_kepler_dwarfs_rv.csv"
UCAC_TIDSYNC_PATH = HEAD_DIR / "UCAC_4_McQuillan_Tidsync.tsv"
UCAC_EB_TIDSYNC_PATH = HEAD_DIR / "UCAC_4_ebs_Tidsync.tsv"
UCAC_TIDSYNC_RAFA_PATH = HEAD_DIR / "UCAC_4_Rafa_Tidsync.tsv"
EB_PATH = HEAD_DIR / "Villanova_EB_v3.txt"
SYNC_EB_PATH = HEAD_DIR / "sync_ebs_moredetails.csv"
KOI_PATH = HEAD_DIR / "koi.csv"
TGAS_MCQUILLAN_APOGEE_TIDSYNC_PATH = (
    HEAD_DIR / "TGAS_McQuillan_APOGEE_tidsync.fits")
TGAS_KEPLER_OVERLAP = HEAD_DIR / "TGAS_Kepler.csv"
BERGER_DR2_KEPLER = HEAD_DIR / "apjaada83t1_mrt.txt"
BERGER_KSPC_KEPLER_INPUT = HEAD_DIR / "ajab8a33t1_mrt.txt"
BERGER_KSPC_KEPLER_OUTPUT = HEAD_DIR / "ajab8a33t2_mrt.txt"
BERGER_MISSING_GAIA = HEAD_DIR / "Missing_Berger.vot"
FLICKER_LOGG = HEAD_DIR / "flicker_loggs.txt"
BRUNTT_PATH = HEAD_DIR / "Bruntt_vsini.vot"
DRESSING_CHARBONNEAU_PROPS = HEAD_DIR / "Dressing_Charbonneau13.fits"
EL_BADRY_SINGLE = HEAD_DIR / "Table_E1_all_single_star_ids.csv"
EL_BADRY_SB1 = HEAD_DIR / "Table_E2_all_SB1_labels.csv"
EL_BADRY_SB2 = HEAD_DIR / "Table_E3_all_binary_star_labels.csv"
EL_BADRY_HIDDEN_TRIPLE = (
    HEAD_DIR / "Table_E4_all_SB2s_hidden_third_component_labels.csv")
EL_BADRY_SB3 = HEAD_DIR / "Table_E5_all_SB3_labels.csv"
DR16_JOKER_BINARIES = HEAD_DIR / "datafile2.fits"
CALIFORNIA_KEPLER_SPECTROSCOPY = HEAD_DIR / "ajaa80det5_mrt.txt"
KEPLER_NAMES = HEAD_DIR / "keplernames.csv"
KOUNKEL_SB2_PATH = HEAD_DIR / "dr14_sb2.txt"
REBULL_PLEIADES_PERIOD_PATH = HEAD_DIR / "ajaa2e04t2_mrt.txt"
REBULL_EPIC_PATH = HEAD_DIR / "Rebull_EPIC_output.txt"
REBULL_CROSSID_PATH = HEAD_DIR / "ajaa2e04t7_mrt.txt"
REBULL_PLEIADES_MULTIPERIOD_PATH = HEAD_DIR / "ajaa2e05t2_mrt.txt"
RADICK_HYADES_PATH = HEAD_DIR / "Radick_1987_Table_3.txt"
MEIBOM_M34_PERIODS = HEAD_DIR / "apj389066t2_mrt.txt"
REBULL_PRAESEPE_PERIOD_PATH = HEAD_DIR / "apjaa6aa4t1_mrt.txt"
LURIE_PERIOD_PATH = HEAD_DIR / "ajaa974dt2_mrt.txt"

CHAPLIN_DWARFS_BRUNTT = HEAD_DIR / "apjs487288t6_mrt.txt"
CHAPLIN_DWARFS_IRFM = HEAD_DIR / "apjs487288t5_mrt.txt"
CHAPLIN_DWARFS_SDSS = HEAD_DIR / "apjs487288t4_mrt.txt"
CHAPLIN_DWARFS_INPUT = HEAD_DIR / "apjs487288t1_mrt.txt"

STAUFFER_VSINI_PATH = HEAD_DIR / "Stauffer_Hartmann_Table1.txt"
STAUFFER_1982_TABLE1_PATH = HEAD_DIR / "Stauffer_82_Table1.txt"
STAUFFER_PHOT_FILE = HEAD_DIR / "jrsfjv.bvi.2ma-1.dat"
STAUFFER_1984_VSINI = HEAD_DIR / "Stauffer_1984_Table_2.txt"
TERNDRUP_VSINI_KPNO_PATH = HEAD_DIR / "990412.tb1.txt"
TERNDRUP_VSINI_KECK_PATH = HEAD_DIR / "990412.tb2.txt"
QUELOZ_PLEIADES_PATH = HEAD_DIR / "Queloz_Table_3.fits"
QUELOZ_CORONA_PATH = HEAD_DIR / "Queloz_Table_4.fits"
QUELOZ_NEW_CORONA = HEAD_DIR / "Queloz_Table_5.fits"
SODERBLOM_LICK_TABLE_1 = HEAD_DIR / "Soderblom_1993b_Table_1.txt"
SODERBLOM_ADDITIONAL_TABLE_6 = HEAD_DIR / "Soderblom_1993b_Table_6.txt"
JACKSON_PLEIADES_PATH = HEAD_DIR / "MN_17_3616_MJ_Table4.csv"
ODELL_TABLE_3_PATH = HEAD_DIR / "Odell_Table_3.txt"
PLEIADES_GAIA_TARGETS = HEAD_DIR / "Pleiades_Gaia.vot.gz"

MERMILLIOD_CLUSTER_TABLE_11 = HEAD_DIR / "Mermilliod_2009_Table_11.fits"
CUMMINGS_HYADES_TABLE = HEAD_DIR / "Cummings_Hyades_Table_6.fits"

RAGHAVAN_TABLE_13 = HEAD_DIR / "apjs363042t13_mrt.txt"
RAGHAVAN_TABLE_17 = HEAD_DIR / "apjs363042t17_mrt.txt"
RAGHAVAN_TABLE_18 = HEAD_DIR / "apjs363042t18_mrt.txt"

WEBDA_DIR = HEAD_DIR / "Pleiades_WEBDA"
WEBDA_PLEIADES_UBV_PHOTOMETRY = WEBDA_DIR / "ubv.peo"
WEBDA_PLEIADES_VRIk_PHOTOMETRY = WEBDA_DIR / "vrik.mes"
WEBDA_PLEIADES_VSINI = WEBDA_DIR / "vsini.don"
WEBDA_PLEIADES_COORDINATES = WEBDA_DIR / "adel.coo"

THESIS_PATH = HEAD_DIR / "Thesis"
PROPOSAL_PATH = THESIS_PATH / "Proposals"
APOGEE_ANCILLARY_TARGETS_TABLE = PROPOSAL_PATH / "ancillary_targets.txt"

DR14_ALLVISIT_PATH = HEAD_DIR / "allVisit-l31c.2.fits"
DR14_ALLSTAR_PATH = HEAD_DIR / "allStar-l31c.2.fits"
DR16_ALLSTAR_PATH = HEAD_DIR / "allStar-r12-l33.fits"
LATEST_ALLSTAR_PATH = DR16_ALLSTAR_PATH

DLSB_DB = HEAD_DIR / "DLSB.txt"
NODL_DB = HEAD_DIR / "noDL.txt"

VIZIER_KEPLER_INPUT = HEAD_DIR / "Vizier_Kepler_Input.txt"
GAIA_KEPLER_INPUT = HEAD_DIR / "Gaia_Kepler_Input.txt"
GAIA_DR2_KEPLER_OVERLAP = HEAD_DIR / "Gaia_DR2_Kepler.csv"
GAIA_BERGER_OVERLAP = HEAD_DIR / "gaia_berger.vo"
UCAC_KEPLER_PATH = HEAD_DIR / "UCAC_Kepler.csv"

TAYAR_SAMPLE = HEAD_DIR / "apj514696t1_mrt.webarchive"


# Modspec Run
MODSPEC_FOLDER = HEAD_DIR / "Modspec"
CALIB_FOLDER = MODSPEC_FOLDER / "Modspec_Calibration"
MDM_DIR = HEAD_DIR / "MDM_targets.csv"

# RV Standards
RV_STANDARD_SIMBAD = MDM_DIR / "Standard_SIMBAD.txt"
RV_STANDARD_SOURCES = HEAD_DIR / "SIMBAD_Standard_References.txt"
LCES_STANDARD_LIST = HEAD_DIR / "ajaa66cat1_mrt.txt"
RV_STANDARD_MATRIX_FOLDER = MDM_DIR / "Standard_RVs"

# SHORTCUT PATHS
# These are for subsets of table that take a really long time to generate.
# Since disk space is cheap, I think it will be much more valuable to just
# shove these tables in a file and then read them when needed.
SHORTCUTS = HEAD_DIR / "srtct"
SHORTCUT_MCQUILLAN_STELLPARM = SHORTCUTS / "mcq_stelparms.fits"
SHORTCUT_MCQUILLAN_NONDET_STELLPARM = SHORTCUTS / "mcq_nondets.fits"
SHORTCUT_MCQUILLAN_FLICKER = SHORTCUTS / "mcq_flicker.fits"
SHORTCUT_MCQUILLAN_APOKASC = SHORTCUTS / "mcq_apokasc.fits"
SHORTCUT_MCQUILLAN_DR14 = SHORTCUTS / "mcq_dr14.fits"
SHORTCUT_MCQUILLAN_EHK = SHORTCUTS / "mcq_ehk.fits"
SHORTCUT_BRUNTT_DR14 = SHORTCUTS / "bruntt_dr14.fits"
SHORTCUT_MCQUILLAN_EBS = SHORTCUTS / "mcq_ebs.fits"
SHORTCUT_MDM_NOMAGCUT = SHORTCUTS / "obs_pretarget_may2017.fits"
SHORTCUT_APOGEE_KIC = SHORTCUTS / "kic_apogee.fits"
SHORTCUT_APOKASC_KIC = SHORTCUTS / "kic_apokasc.fits"
SHORTCUT_MCQUILLAN_APOGEE_KIC = SHORTCUTS / "mcq_kic_apogee.fits"
SHORTCUT_PLEIADES_APOGEE = SHORTCUTS / "pleiades_apogee.fits"
SHORTCUT_GAIA_KEPLER = SHORTCUTS / "Gaia_Berger.fits"
