from pathlib import Path
import os

HOME_DIR = Path.home()
HEAD_DIR = Path(os.environ["THESIS"])

CASAGRANDE_TABLE_PATH = HEAD_DIR
CASAGRANDE_TABLE_4 = CASAGRANDE_TABLE_PATH / "Casagrande_10_Table_4.txt" 
CASAGRANDE_TABLE_5 = CASAGRANDE_TABLE_PATH / "Casagrande_10_Table_5.txt"

DSEP_OUTPUT = HEAD_DIR / "DSEP"

DSEP_PATH = HOME_DIR / "DSep"
DSEP_ISOCHRONES= DSEP_PATH / "isochrones"
DSEP_INTERPOLATOR_EXECUTABLE = DSEP_PATH / "iso_interp_feh"
DSEP_SPLITTER_EXECUTABLE = DSEP_PATH / "isolf_split"

CLUSTER_PATH = HEAD_DIR / "Clusters"
PLEIADES_PATH = CLUSTER_PATH / "Pleiades"
BOUY_TABLE_PATH = PLEIADES_PATH
BOUY_TABLE_2 = BOUY_TABLE_PATH / "Bouy_15_Table_2.fits"
BOUY_TABLE_6 = BOUY_TABLE_PATH / "Bouy_15_Table_6.fits"
BOUY_GOOD_MEMBERS = BOUY_TABLE_PATH / "Bouy_good_members.fits"

M67_PATH = CLUSTER_PATH / "M67"
M67_TABLE = M67_PATH / "m67ensemblecal.pmem.dat"
M67_TWOMASS = M67_PATH / "2mass_photometry.ipac"
M67_EPIC = M67_PATH / "M67_WOCS_EPIC_match.txt"

MCQUILLAN_CATALOG = HEAD_DIR / "McQuillan.fit"
RAFA_SAVITA_PERIODS = HEAD_DIR / "Prot_OK_noCP.txt"
HUBER_CATALOG = HEAD_DIR / "huber_kic_parameters.txt"
KIC_CATALOG = HEAD_DIR / "KIC_DR25.tbl"
ORIG_KIC = HEAD_DIR / "kic.txt.gz"
KEPLER_GALEX_BEST = HEAD_DIR / "KGGoldStandard.csv.gz"
EHK_README = HEAD_DIR / "hlsp_kplrubv_readme.txt"
EHK_PATH = HEAD_DIR / "EHK2012catalog.dat"
VAN_SADERS_SDSS = HEAD_DIR / "cool_dwarfs.sav"
VAN_SADERS_MAST = HEAD_DIR / "Jen_SDSS_Kepler_MAST_output.csv"
JEN_APOGEE_1 = HEAD_DIR / "Jen_APOGEE_results_0.csv"
JEN_APOGEE_2 = HEAD_DIR / "Jen_APOGEE_results_1.csv"
APOGEE_KOI_FIELDS = HEAD_DIR / "apogee_koi.txt"
APOGEE_KASC_FIELDS = HEAD_DIR / "apogee_apokasc_fields.txt"
APOGEE_DWARF_PATH = HEAD_DIR / "apogee_kepler_dwarfs_rv.csv"
UCAC_TIDSYNC_PATH = HEAD_DIR / "UCAC_4_McQuillan_Tidsync.tsv"
UCAC_TIDSYNC_RAFA_PATH = HEAD_DIR / "UCAC_4_Rafa_Tidsync.tsv"
EB_PATH = HEAD_DIR / "Villanova_EB_v3.txt"
SYNC_EB_PATH = HEAD_DIR / "sync_ebs_moredetails.csv"
KOI_PATH = HEAD_DIR / "koi.csv"
TGAS_MCQUILLAN_APOGEE_TIDSYNC_PATH = (
    HEAD_DIR / "TGAS_McQuillan_APOGEE_tidsync.fits")

THESIS_PATH = HEAD_DIR / "Thesis"
PROPOSAL_PATH = THESIS_PATH / "Proposals"
APOGEE_ANCILLARY_TARGETS_TABLE = PROPOSAL_PATH / "ancillary_targets.txt"

DR14_ALLVISIT_PATH = HEAD_DIR / "allVisit-l31c.1.fits"
DR14_ALLSTAR_PATH = HEAD_DIR / "allStar-l31c.1.fits"
