import read_catalog as catin
import catalog

apogee = catin.read_dr14_with_KIC_stelparms()
good_apogee = catalog.apogee_filter_quality(apogee, quality="bad")

mcq = catin.read_McQuillan_catalog()
nomcq = catin.read_McQuillan_nondetections()

apo_mcq = au.join_by_id(good_apogee, mcq, "kepid", "KIC")
apo_nomcq = au.join_by_id(good_apogee, nomcq, "kepid", "KIC")

apogee_giants = good_apogee[good_apogee["LOGG"] > -100]
apogee_dwarfs = good_apogee[good_apogee["LOGG"] < -100]

teffbins = np.linspace(3500, 8500, 20)

# Show overlap between APOGEE and McQuillan in Teff
plt.hist([apogee_nongiants["TEFF"], apogee_giants["TEFF"]], bins=teffbins,
          stacked=True, label=["Dwarfs", "Giants"])
plt.hist([apo_mcq["TEFF"], apo_nomcq["TEFF"]], bins=teffbins, stacked=True, 
         label=[ "McQuillan Detections", "McQuillan Nondetections"])
