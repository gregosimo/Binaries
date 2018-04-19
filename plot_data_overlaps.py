
import numpy as np
import astropy_util as au

import read_catalog as catin
import catalog
import biovis_colors as bc

apogee = catin.dr14_with_KIC_stelparms()
good_apogee = catalog.apogee_filter_quality(apogee, quality="bad")

mcq = catin.read_McQuillan_catalog()
nomcq = catin.read_McQuillan_nondetections()

# Since there are a bunch of stars Garcia analyzed where I don't know where
# they came from. I want just the targets that were in the Chaplin sample.
garcia_fullsamp = catin.read_Garcia_prevsample()
garcia_fullperiods = catin.read_Garcia_periods()
garcia_periods = au.extract_subtable_from_column(
    garcia_fullperiods, "KIC", garcia_fullsamp["KIC"])
garcia_noper = au.get_complement_table(garcia_periods, garcia_fullsamp, "KIC")

apo_mcq = au.join_by_id(good_apogee, mcq, "kepid", "KIC")
apo_nomcq = au.join_by_id(good_apogee, nomcq, "kepid", "KIC")
apo_garcia = au.join_by_id(good_apogee, garcia_periods, "kepid", "KIC")
apo_nogarcia = au.join_by_id(good_apogee, garcia_noper, "kepid", "KIC")

apogee_giants = good_apogee[good_apogee["LOGG"] > -100]
apogee_nongiants = good_apogee[good_apogee["LOGG"] < -100]

teffbins = np.linspace(3500, 8500, 20)
metbins = np.linspace(-1, 1, 20)

# Show overlap between APOGEE and McQuillan in Teff
n, bins, apopatches = plt.hist(
    [apogee_nongiants["TEFF"], apogee_giants["TEFF"]], bins=teffbins, 
    stacked=True, label=["APOGEE Dwarfs", "APOGEE Giants"], 
    color=[bc.pink, bc.black], histtype="bar")
#for patch in apopatches[1]:
#    patch.set_hatch("/")
n, bins, mcqpatches = plt.hist(
    [apo_mcq["TEFF"], apo_nomcq["TEFF"]], bins=teffbins, stacked=True, 
    label=["McQuillan Detections", "McQuillan Nondetections"], 
    color=[bc.red, bc.yellow], histtype="bar")
#for patch in mcqpatches[1]:
#    patch.set_hatch("/")
plt.legend(loc="upper right")
plt.xlabel("APOGEE Teff (K)")
plt.ylabel("N")
plt.title("Temperature Overlap")

# Show overlap between APOGEE and McQuillan in metallicity.
plt.hist([apogee_nongiants["M_H"], apogee_giants["M_H"]], bins=metbins,
          stacked=True, label=["APOGEE Dwarfs", "APOGEE Giants"], 
         color=[bc.pink, bc.black], histtype="bar")
plt.hist([apo_mcq["M_H"], apo_nomcq["M_H"]], bins=metbins, stacked=True, 
         label=["McQuillan Detections", "McQuillan Nondetections"],
         color=[bc.yellow, bc.red], histtype="bar")
plt.legend(loc="upper right")
plt.xlabel("APOGEE [M/H]")
plt.ylabel("N")
plt.title("Metallicity Overlap")

# Show overlap between APOGEE and Garcia in Teff
plt.hist(
    [apogee_nongiants["TEFF"], apogee_giants["TEFF"]], bins=teffbins, 
    stacked=True, label=["APOGEE Dwarfs", "APOGEE Giants"], 
    color=[bc.pink, bc.black], histtype="bar")
#for patch in apopatches[1]:
#    patch.set_hatch("/")
n, bins, mcqpatches = plt.hist(
    [apo_garcia["TEFF"], apo_nogarcia["TEFF"]], bins=teffbins, stacked=True, 
    label=["Garcia Detections", "Garcia Nondetections"], 
    color=[bc.red, bc.yellow], histtype="bar")
#for patch in mcqpatches[1]:
#    patch.set_hatch("/")
plt.legend(loc="upper right")
plt.xlabel("APOGEE Teff (K)")
plt.ylabel("N")
plt.title("Temperature Overlap")

# Show overlap between APOGEE and McQuillan in metallicity.
plt.hist([apogee_nongiants["M_H"], apogee_giants["M_H"]], bins=metbins,
          stacked=True, label=["APOGEE Dwarfs", "APOGEE Giants"], 
         color=[bc.pink, bc.black], histtype="bar")
plt.hist([apo_garcia["M_H"], apo_nogarcia["M_H"]], bins=metbins, stacked=True, 
         label=["Garcia Detections", "Garcia Nondetections"],
         color=[bc.yellow, bc.red], histtype="bar")
plt.legend(loc="upper right")
plt.xlabel("APOGEE [M/H]")
plt.ylabel("N")
plt.title("Metallicity Overlap")
