
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

mcq_garcia = au.join_by_id(mcq, garcia_periods, "KIC", "KIC",
                           conflict_suffixes=("_Mcq", "_Garcia"))

apo_mcq = au.join_by_id(good_apogee, mcq, "kepid", "KIC")
apo_nomcq = au.join_by_id(good_apogee, nomcq, "kepid", "KIC")
apo_garcia = au.join_by_id(good_apogee, garcia_periods, "kepid", "KIC")
apo_nogarcia = au.join_by_id(good_apogee, garcia_noper, "kepid", "KIC")

apogee_giants = good_apogee[good_apogee["LOGG"] > -100]
apogee_nongiants = good_apogee[good_apogee["LOGG"] < -100]

teffbins = np.linspace(3500, 8500, 20)
metbins = np.linspace(-1, 1, 20)
pbins = np.linspace(0, 40, 20)

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
# Have a Cumulative histogram too
plt.hist(
    apogee_nongiants["TEFF"], bins=teffbins, normed=True, cumulative=True,
    histtype="step", linestyle="--", label="APOGEE Dwarfs", color=bc.black)
plt.hist(
    apogee_giants["TEFF"], bins=teffbins, normed=True, cumulative=True,
    histtype="step", linestyle=":", label="APOGEE Giants", color=bc.black)
plt.hist(
    [np.ma.concatenate([apogee_giants["TEFF"], apogee_nongiants["TEFF"]])], 
     bins=teffbins, normed=True, cumulative=True, histtype="step", 
     linestyle="-", label="Combined APOGEE", color=bc.black, lw=3)
plt.hist(
    apo_mcq["TEFF"], bins=teffbins, normed=True, cumulative=True,
    histtype="step", linestyle="--", label="McQuillan Detections", color=bc.red)
plt.hist(
    apo_nomcq["TEFF"], bins=teffbins, normed=True, cumulative=True,
    histtype="step", linestyle=":", label="McQuillan Nondetections", 
    color=bc.red)
plt.hist(
    [np.ma.concatenate([apo_nomcq["TEFF"], apo_mcq["TEFF"]])], 
     bins=teffbins, normed=True, cumulative=True, histtype="step", 
     linestyle="-", label="Full McQuillan", color=bc.red, lw=3)
plt.legend(loc="lower right")
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
plt.ylabel("N (< Teff) / N")
plt.title("Metallicity Overlap")
# Now the cumulative histogram
plt.hist(
    apogee_nongiants["M_H"], bins=metbins, normed=True, cumulative=True,
    histtype="step", linestyle="--", label="APOGEE Dwarfs", color=bc.black)
plt.hist(
    apogee_giants["M_H"], bins=metbins, normed=True, cumulative=True,
    histtype="step", linestyle=":", label="APOGEE Giants", color=bc.black)
plt.hist(
    [np.ma.concatenate([apogee_giants["M_H"], apogee_nongiants["M_H"]])], 
     bins=metbins, normed=True, cumulative=True, histtype="step", 
     linestyle="-", label="Combined APOGEE", color=bc.black, lw=3)
plt.hist(
    apo_mcq["M_H"], bins=metbins, normed=True, cumulative=True,
    histtype="step", linestyle="--", label="McQuillan Detections", color=bc.red)
plt.hist(
    apo_nomcq["M_H"], bins=metbins, normed=True, cumulative=True,
    histtype="step", linestyle=":", label="McQuillan Nondetections", 
    color=bc.red)
plt.hist(
    [np.ma.concatenate([apo_nomcq["M_H"], apo_mcq["M_H"]])], 
     bins=metbins, normed=True, cumulative=True, histtype="step", 
     linestyle="-", label="Full McQuillan", color=bc.red, lw=3)
plt.legend(loc="lower right")
plt.xlabel("[M/H]")
plt.ylabel("N (< [M/H]) / N")
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
# Have a Cumulative histogram too
plt.hist(
    apogee_nongiants["TEFF"], bins=teffbins, normed=True, cumulative=True,
    histtype="step", linestyle="--", label="APOGEE Dwarfs", color=bc.black)
plt.hist(
    apogee_giants["TEFF"], bins=teffbins, normed=True, cumulative=True,
    histtype="step", linestyle=":", label="APOGEE Giants", color=bc.black)
plt.hist(
    [np.ma.concatenate([apogee_giants["TEFF"], apogee_nongiants["TEFF"]])], 
     bins=teffbins, normed=True, cumulative=True, histtype="step", 
     linestyle="-", label="Combined APOGEE", color=bc.black, lw=3)
plt.hist(
    apo_garcia["TEFF"], bins=teffbins, normed=True, cumulative=True,
    histtype="step", linestyle="--", label="Garcia Detections", color=bc.red)
plt.hist(
    apo_nogarcia["TEFF"], bins=teffbins, normed=True, cumulative=True,
    histtype="step", linestyle=":", label="Garcia Nondetections", 
    color=bc.red)
plt.hist(
    [np.ma.concatenate([apo_nogarcia["TEFF"], apo_garcia["TEFF"]])], 
     bins=teffbins, normed=True, cumulative=True, histtype="step", 
     linestyle="-", label="Full Garcia", color=bc.red, lw=3)
plt.legend(loc="lower right")
plt.xlabel("APOGEE Teff (K)")
plt.ylabel("N")
plt.title("Temperature Overlap")

# Show overlap between APOGEE and Garcia in metallicity.
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
# Now the cumulative histogram
plt.hist(
    apogee_nongiants["M_H"], bins=metbins, normed=True, cumulative=True,
    histtype="step", linestyle="--", label="APOGEE Dwarfs", color=bc.black)
plt.hist(
    apogee_giants["M_H"], bins=metbins, normed=True, cumulative=True,
    histtype="step", linestyle=":", label="APOGEE Giants", color=bc.black)
plt.hist(
    [np.ma.concatenate([apogee_giants["M_H"], apogee_nongiants["M_H"]])], 
     bins=metbins, normed=True, cumulative=True, histtype="step", 
     linestyle="-", label="Combined APOGEE", color=bc.black, lw=3)
plt.hist(
    apo_garcia["M_H"], bins=metbins, normed=True, cumulative=True,
    histtype="step", linestyle="--", label="Garcia Detections", color=bc.red)
plt.hist(
    apo_nogarcia["M_H"], bins=metbins, normed=True, cumulative=True,
    histtype="step", linestyle=":", label="Garcia Nondetections", 
    color=bc.red)
plt.hist(
    [np.ma.concatenate([apo_nogarcia["M_H"], apo_garcia["M_H"]])], 
     bins=metbins, normed=True, cumulative=True, histtype="step", 
     linestyle="-", label="Full Garcia", color=bc.red, lw=3)
plt.legend(loc="lower right")
plt.xlabel("[M/H]")
plt.ylabel("N (< [M/H]) / N")
plt.title("Metallicity Overlap")

# Plot APOGEE Teff against Metallicity
# Dwarfs vs Giants
plt.plot(apogee_nongiants["TEFF"], apogee_nongiants["M_H"], color=bc.black,
         ls="None", marker=".", label="APOGEE Dwarfs")
plt.plot(apogee_giants["TEFF"], apogee_giants["M_H"], color=bc.orange,
         ls="None", marker=".", label="APOGEE Giants")
plt.xlim(8000, 3500)
plt.ylim(-2.5, 0.5)
hr.invert_x_axis()
plt.xlabel("APOGEE Teff (K)")
plt.ylabel("[M/H]")
plt.title("APOGEE Teff and Metallicity")
plt.legend(loc="lower left")

# Plot APOGEE Teff against Metallicity
# McQuillan Detection vs Nondetection
plt.plot(apogee_nongiants["TEFF"], apogee_nongiants["M_H"], color=bc.black,
         ls="None", marker=".", label="APOGEE Dwarfs")
plt.plot(apo_mcq["TEFF"], apo_mcq["M_H"], color=bc.pink,
         ls="None", marker="o", label="McQuillan Detection")
plt.plot(apo_nomcq["TEFF"], apo_nomcq["M_H"], color=bc.sky_blue,
         ls="None", marker="o", label="McQuillan Non-Detection")
plt.xlim(8000, 3500)
plt.ylim(-2.5, 0.5)
hr.invert_x_axis()
plt.xlabel("APOGEE Teff (K)")
plt.ylabel("[M/H]")
plt.title("APOGEE Teff and Metallicity")
plt.legend(loc="lower left")

# Plot APOGEE Teff against Metallicity
# Garcia Detection vs Nondetection
plt.plot(apogee_nongiants["TEFF"], apogee_nongiants["M_H"], color=bc.black,
         ls="None", marker=".", label="APOGEE Dwarfs")
plt.plot(apo_garcia["TEFF"], apo_garcia["M_H"], color=bc.pink,
         ls="None", marker="o", label="Garcia Detection")
plt.plot(apo_nogarcia["TEFF"], apo_nogarcia["M_H"], color=bc.sky_blue,
         ls="None", marker="o", label="Garcia Non-Detection")
plt.xlim(8000, 3500)
plt.ylim(-2.5, 0.5)
hr.invert_x_axis()
plt.xlabel("APOGEE Teff (K)")
plt.ylabel("[M/H]")
plt.title("APOGEE Teff and Metallicity")
plt.legend(loc="lower left")

# Show overlap between McQuillan and APOGEE in period
plt.hist(mcq["Prot"], bins=pbins, stacked=True, label="McQuillan Detections", 
         color=bc.black, histtype="bar")
plt.hist(apo_mcq["Prot"], bins=pbins, stacked=True, label="APOGEE Overlap", 
         color=bc.red, histtype="bar")
plt.legend(loc="upper right")
plt.xlabel("McQuillan Period (day)")
plt.ylabel("N")
plt.title("Period Overlap")
# Now the cumulative histogram
plt.hist(
    mcq["Prot"], bins=pbins, normed=True, cumulative=True,
    histtype="step", linestyle="--", label="McQuillan Detections", 
    color=bc.black)
plt.hist(
    apo_mcq["Prot"], bins=pbins, normed=True, cumulative=True,
    histtype="step", linestyle="--", label="APOGEE Overlap", color=bc.red)
plt.legend(loc="lower right")
plt.xlabel("Rotation Period (day)")
plt.ylabel("N (< Prot]) / N")
plt.title("Period Overlap")

# Show overlap between Garcia and APOGEE in period
plt.hist(garcia_periods["Prot"], bins=pbins, stacked=True, 
         label="Garcia Detections", color=bc.black, histtype="bar")
plt.hist(apo_garcia["Prot"], bins=pbins, stacked=True, label="APOGEE Overlap", 
         color=bc.red, histtype="bar")
plt.legend(loc="upper right")
plt.xlabel("Garcia Period (day)")
plt.ylabel("N")
plt.title("Period Overlap")
# Now the cumulative histogram
plt.hist(
    garcia_periods["Prot"], bins=pbins, normed=True, cumulative=True,
    histtype="step", linestyle="--", label="Garcia Detections", color=bc.black)
plt.hist(
    apo_garcia["Prot"], bins=pbins, normed=True, cumulative=True,
    histtype="step", linestyle="--", label="APOGEE Overlap", color=bc.red)
plt.legend(loc="lower right")
plt.xlabel("Rotation Period (day)")
plt.ylabel("N (< Prot]) / N")
plt.title("Period Overlap")

# Show overlap between Garcia and McQuillan
plt.hist(mcq_garcia["Prot_Mcq"], bins=pbins, stacked=True, 
         label="Overlap (Mcq)", color=bc.red, histtype="step")
plt.hist(mcq_garcia["Prot_Garcia"], bins=pbins, stacked=True, 
         label="Overlap (Garcia)", color=bc.black, histtype="step")
plt.legend(loc="upper right")
plt.xlabel("Rotation Period (day)")
plt.ylabel("N")
plt.title("Mcq/Garcia Period Overlap")
# Now the cumulative histogram
plt.hist(
    mcq["Prot"], bins=pbins, normed=True, cumulative=True,
    histtype="step", linestyle="-", label="McQuillan Detections", 
    color=bc.black)
plt.hist(
    garcia_periods["Prot"], bins=pbins, normed=True, cumulative=True,
    histtype="step", linestyle="-", label="Garcia Detections", color=bc.blue)
plt.hist(
    mcq_garcia["Prot_Mcq"], bins=pbins, normed=True, cumulative=True,
    histtype="step", linestyle=":", label="Overlap (Mcq)", 
    color=bc.red)
plt.hist(
    mcq_garcia["Prot_Garcia"], bins=pbins, normed=True, cumulative=True,
    histtype="step", linestyle="--", label="Overlap (Garcia)", 
    color=bc.red)
plt.legend(loc="lower right")
plt.xlabel("Rotation Period (day)")
plt.ylabel("N (< Prot]) / N")
plt.title("Mcq/Garcia Period Overlap")
