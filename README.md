# Gregory's Binary Code

> **CAUTION:** This repository is no longer actively developed!

Here is my workhorse code for doing research related to binary stars. These codes are not meant to be run directly, but are rather more like libraries specific to my research on binaries that do specific tasks. To get concrete examples of how they are implemented, see the following projects:

- **[rotletter18](https://github.com/gregosimo/rotletter18)**: The code that generated figures and tables for [Simonian et al (2019)](https://iopscience.iop.org/article/10.3847/1538-4357/aaf97c).
- **[rotation17](https://github.com/gregosimo/rotation17)**: The code that generated figures and tables for [Simonian et al (2020)](https://iopscience.iop.org/article/10.3847/1538-4357/ab9a43).

More general-purpose routines (as extensions to Astropy) can be found in my [astropy-util](https://github.com/gregosimo/astropy-utils) repository.

## File Organization

### Catalog
Files for catalog manipulation

- **catalog.py** - Routines for manipulating the internal dataset.
- **data_splitting.py** - Custom DataSplitter objects to handle splitting into subsamples based on multiple cuts with internal checks to avoid losing data (handles missing data too)
- **read_catalog.py** - Module that automatically handles reading catalogs from diverse sources. Large catalogs of which only a subsample is really useful are automatically memoized for speed and memory efficiency.
- **sample_characterization.py** - Collection of functions to characterize the different samples in the analysis
- **path_config.py** - Centralized master list of paths for all datasets.
- **cluster.py** - Functions to open cluster data and classes to manipulate them.
- **plot_data_overlaps.py** - Create lots of histograms showing the parameter distributions of different samples (e.g. APOGEE, McQuillan, etc.)

### Rotation
Modules that explicitly are dedicated to processing and analyzing stellar rotation from different measures (e.g. vsini/photometric variability)

- **rotation_consistency.py** - Routines to check whether vsini and photometric periods are consistent, and to identify/characterize outliers.
- **activity.py** - Code looking at trends with Rossby number for [McQuillan et al (2014)](https://ui.adsabs.harvard.edu/abs/2014ApJS..211...24M/abstract) stars.
- **tayar_giants.py** - A few functions checking out the rotating giants from [Tayar et al (2015)](https://ui.adsabs.harvard.edu/abs/2015ApJ...807...82T/abstract).
- **jenboundary.py** - Expected rotation envelopes for subgiants experiencing standard stellar spindown based on van Saders' results.

### RV Observations
Files for probing the effects of RV sampling

- **alias_simu.py** - Code probing the effects of aliasing on binary populations.
- **observations.py** - Code for preparing and analyzing the observing run for RV variability at MDM observatory.
- **rv_simu.py** - General functions to simulate RV curve observations, with some tailoring to MDM Modspec.

### APOGEE
Files primarily relating to analyzing APOGEE data

- **myapogee.py** - Library to download large APOGEE datasets (e.g. apStar and allData)
- **browse_APOGEE_spectra.py** - Tool to interactively pull up APOGEE spectra to label double-lined spectroscopic binaries.
- **apogee_ancillary.py** - Making plots and initial targets for the APOGEE ancillary proposal.
- **apogee_RV_variability.py** - Characterizing RV variable subsamples of the APOGEE population
- **apogee_census.py** - Characterizing the population of APOGEE targets and outputting the relative sizes of different subsamples.
- **aspcap_corrections.py** - Functions applying corrections to ASPCAP calibrations for parts of the sample.

### Special Populations
Files that contain functions for analyzing certain populations

- **eclipsing_binaries.py** - Functions to read in and filter eclipsing binary data. Also makes statistical inferences based on eclipse geometry.
- **huber_checks.py** - Perform quality checks based on overlap samples with [Huber et al (2014)](https://ui.adsabs.harvard.edu/abs/2014ApJS..211....2H/abstract).
- **kepvim.py** - Routines for analyzing the crossover sample with KepVIM.

### Isochrones
Files for manipulating isochrones

- **models.py** - Base class for isochrone
  - **baraffe.py** - Object for reading and manipulating the Baraffe isochrones
  - **dsep.py** - DSEP isochrone object
  - **mist.py** - MIST isochrone object
  - **yrec.py** - Class for reading YREC isochrones
- **sed.py** - Lots of SED-related functions. In particular related to the [Casagrande et al (2010)](https://ui.adsabs.harvard.edu/abs/2010A%26A...512A..54C/abstract) Teff scale as well as interpolating DSEP Photometry.
- **extinction.py** - Calculate reddening based on dust maps for different bands.

### Misc
Miscellaneous files

- **biovis_colors.py** - Implementing the color palette from http://mkweb.bcgsc.ca/biovis2012/, which is supposed to be good for colorblindness.
- **hrplots.py** - Routines for automating the creation of HR Plots (e.g. flipping axes and making sure they are displayed appropriately).
- **label_python.py** - Incomplete module for organizing observing run data.

