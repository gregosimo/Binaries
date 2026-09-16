=========================
= Gregory's Binary Code =
=========================

.. CAUTION::
   This repository is no longer actively developed!

Here is my workhorse code for doing research related to binary stars. These
codes are not meant to be run directly, but are rather more like libraries
specific to my research on binaries that do specific tasks. To get concrete
examples of how they are implemented, see the following projects:

:`rotletter18`: The code that generated figures and tables for `Simonian et al
                (2019)`.
:`rotation17`: The code that generated figures and tables for `Simonian et al (2020)`.

.. _rotletter18: https://github.com/gregosimo/rotletter18
.. _rotation17: https://github.com/gregosimo/rotation17
.. _Simonian et al (2019): https://iopscience.iop.org/article/10.3847/1538-4357/aaf97c
.. _Simonian et al (2020): https://iopscience.iop.org/article/10.3847/1538-4357/ab9a43

More general-purpose routines (as extensions to Astropy) can be found in my `astropy-util` repository.

.. _astropy-util: https://github.com/gregosimo/astropy-utils


The files in this project can be broadly clustered in the following groups:

:Catalog: Files for catalog manipulation
    :catalog.py: Routines for manipulating the internal dataset.
    :data_splitting.py: Custom DataSplitter objects to handle splitting into
    subsamples based on multiple cuts with internal checks to avoid losing data (handles missing data
    too)
    :cluster.py: Functions to open cluster data and classes to manipulate them.

:RV Sampling: Files for probing the effects of RV sampling.
    :alias_simu.py: Code probing the effects of aliasing on binary populations.

:Proposals: Files for analyzing and generating plots for proposals
    :apogee_ancillary.py: Making plots and initial targets for the APOGEE
    ancillary proposal.

:APOGEE: Files primarily relating to analyzing APOGEE data.
    :apogee_RV_variability.py: Characterizing RV variable subsamples of the
        APOGEE population
    :apogee_census.py: Characterizing the population of APOGEE targets and
        outputting the relative sizes of different subsamples.
    :aspcap_corrections.py: Functions applying corrections to ASPCAP
        calibrations for parts of the sample.
    :browse_APOGEE_spectra.py: Tool to interactively pull up APOGEE spectra to
        label double-lined spectroscopic binaries.
    :myapogee.py: Library to download large APOGEE datasets (e.g. apStar and
        allData)

:Special Populations: Files that contain functions for analyzing certain
    populations.
    :activity.py: Code looking at trends with Rossby number for McQuillan et al
    (2014) stars.
    :eclipsing_binaries.py: Functions to read in and filter eclipsing binary
    data. Also makes statistical inferences based on eclipse geometry.
    :huber_checks.py: Perform quality checks based on overlap samples with
    Huber et al (2014).
    :kepvim.py: Routines for analyzing the crossover sample with KepVIM.

:Photometry: Photometric routines, e.g. synthetic photometry, reddening.
    :extinction.py: Calculate reddening based on dust maps for different bands.

:Isochrones: Files for manipulating isochrones.
    :models.py: Base class for isochrone.
    :baraffe.py: Object for reading and manipulating the Baraffe isochrones:
    :dsep.py: DSEP isochrone object.
    :jenboundary.py: Expected rotation envelopes for subgiants experiencing
    standard stellar spindown based on van Saders' results.
    :mist.py: MIST isochrone object.

:Misc: Miscellaneous files
    :biovis_colors.py: Implementing the color pallete from
        http://mkweb.bcgsc.ca/biovis2012/, which is supposed to be good for
        colorblindness.
    :hrplots.py: Routines for automating the creation of HR Plots (e.g.
    flipping axes and making sure they are displayed appropriately).
    :label_python.py: Incomplete module for organizing observing run data.
observations.py
path_config.py
plot_data_overlaps.py
read_catalog.py
rotation_consistency.py
rv_simu.py
sample_characterization.py
sed.py
tayar_giants.py
yrec.py
