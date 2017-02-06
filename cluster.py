from astropy.table import Table
import astropy.units as u
from scipy.interpolate import interp1d
from scipy import stats
import numpy as np
import numpy.core.defchararray as npstr
import matplotlib.pyplot as plt

import path_config as paths
import astropy_util as au
import sed
import catalog

class Cluster_Data(object):
    '''Class to represent the data of a cluster. 

    This contains the full catalog, and can hold views of subsets of the data
    in order to make exploration more structured. For example, a subset can be
    used to explore only high-probability members, or those with effective
    temperatures in a certain range.
    '''

    def __init__(self, EBV=0.0):
        '''Read in the full data and set up view to the whole data.'''

        tbl = self._read_main_table()

        self.fulltable = tbl

        self.tablesegment = tbl

        self.available_bands = ["B", "V", "I", "J", "H", "K", "Ks"]

        self.band_colnames = {}
        self.band_errnames = {}

        self._initialize_band_dics()
        
        self.EBV = EBV
        self._deredden = False

    def _read_main_table(self):
        raise NotImplementedError()

    def _initialize_band_dics(self):
        raise NotImplementedError()

    def bandmag(self, band):
        '''Get photometry from the current view of the data for the given band.

        Returns the measured magnitudes for the cluster for the given band.
        '''
        try:
            colname = self.band_colnames[band]
        except KeyError:
            raise ValueError("{0}-band is not in this dataset.".format(band))

        cluster_mags = self.tablesegment[colname]

        if self.deredden:
            cluster_mags = sed.deredden_mag(band, cluster_mags, self.EBV)

        return cluster_mags

    def banderr(self, band):
        '''Get photometric errors from current data view for the given band.

        Returns the photometric errors for the cluster in the given band.
        '''
        try:
            colname = self.band_errnames[band]
        except KeyError:
            raise ValueError(
                "Don't have photometry errors for band {0}.".format(band))

        return self.tablesegment[colname]

    def colormag(self, color):
        '''Get colors from the current view of data for the given band.'''
        blueband, redband = sed.split_color(color)
        blue = self.bandmag(blueband)
        red = self.bandmag(redband)
        color = blue - red

        return color 

    def colorerr(self, color):
        '''Get color errors from the current view of data for the given band.'''
        blueband, redband = sed.split_color(color)

        blueerr = self.banderr(blueband)
        rederr = self.banderr(redband)
        colorerr = sed.sum_errors(blueerr, rederr)

        return colorerr

    def reset(self):
        '''Resets the view to the unprocessed data in the cluster.'''
        self.tablesegment = self.fulltable

    # This might be a good use of the @property decorator.
    @property
    def deredden(self):
        '''Set this object to automatically deredden magnitudes.

        All magnitudes and colors obtained via the "bands()" method will be
        dereddened according to the E(B-V) found by Hartmann et al (2008).
        '''
        return self._deredden

    @deredden.setter
    def deredden(self, value):
        '''Set whether cluster should be dereddened or not.
        
        This should be set to be a boolean value.'''
        if isinstance(value, bool):
            self._deredden = value
        else:
            raise ValueError("Dereddening value must be boolean.")


class M67_Cluster(Cluster_Data):
    '''Represents a workable dataset of the M67 cluster.

    This function can be used to restrict members based on probability and use
    the data through interfaces. The data can also be reset to its original
    value.
    '''
    def __init__(self, maskbad=True):
        super().__init__(0.04)
        self.mask_invalid_photometry()

        self.DSEP_lookup = {"B": 1, "V": 1, "I": 1, "J": 1, "H": 1, "Ks": 1,
                            "K": 1}

        self.distance = 879
        self.distance_unc = 20
        self.age=4.0

        self.MONTGOMERY_ERRORS = {"B": 0.024, "V": 0.023, "I": 0.027}

    def _read_main_table(self):
        '''Reads in M67 photometry and membership probabilities.

        The photometry should be in BVI and 2MASS JHKs. Membership
        probabilities are complex and should be referenced in the e-mail. In
        brief they are:

        - Z: Zhao et al (1996)
        - MMJ: Montgomery et al. 
        - N: Nardiello et al. (2016)
        - C: Casamiquela et al. (2016)
        - G: Girard
        - S: Sanders
        - Y: Yadav
        '''
        return M67_data()

    def _initialize_band_dics(self):
        '''Initialize the dictionaries for band column names.'''
        for band in self.available_bands:
            if band in ["B", "V", "I"]:
                self.band_colnames[band] = band
                self.band_errnames[band] = "sig_{0}".format(band)
            elif band in ["J", "H", "K"]:
                self.band_colnames[band] = "{0}_m".format(band.lower())
                self.band_errnames[band] = "{0}_msig".format(band.lower())
            else:
                self.band_colnames[band] = "k_m"
                self.band_errnames[band] = "k_msig"

    def banderr(self, band, montgomery=True):
        '''Get the photometric errors of the current view for a given band.

        Return an array corresponding to the errors of points in the current
        view of the dataset. 
        
        If the Montgomery flag is enabled, the errors will be the minimum of 
        the photometric errors as measured in Sandquist etal (2004) and the RMS
        between Sandquist et al (2004) and Montgomery et al (1993). This should
        provide a upper limit on the accuracy of the values.
        '''
        errors = super().banderr(band)
        if montgomery:
            try:
                mont_error = self.MONTGOMERY_ERRORS[band]
            # This will occur when bands are not B, V, or I.
            except KeyError:
                pass
            else:
                errors = np.maximum(errors, mont_error)
        return errors

    def mask_invalid_photometry(self):
        '''Masks the photometric values which are missing.

        The default value of photometry is 0.00, which turns out to be a very
        confusing way of indicating missing values because 0-magnitude objects
        are possible. This function will go through the BVI mags and mask those
        entries with 0 mag and uncertainty.
        '''
        for band in ["B", "V", "I"]:
            bandval = self.bandmag(band)
            banderr = self.banderr(band, montgomery=False)
            badindices = np.where(np.logical_or(np.logical_and(
                bandval == 0.0, banderr == 0.0), banderr > 0.5))
            self.tablesegment[self.band_colnames[band]][badindices] = \
                np.ma.masked
            self.tablesegment[self.band_errnames[band]][badindices] = \
                np.ma.masked


    def set_RV_members(self):
        '''Filter out all non-RV members in the working cluster.'''
        rvstrings = self.tablesegment["RVClass"]
        members = npstr.startswith(rvstrings, "M", start=-1)
        self.tablesegment = self.tablesegment[members]

    def cut_turnoff(self):
        '''Remove objects which have gone off the MS-turnoff.

        This function uses V=14 as the cutoff point.'''
        self.tablesegment = catalog.perform_cut(
            self.tablesegment, "V", lowval=14.9)

    def set_photometric_binaries(self, color, mag, siglimit=15):
        '''Set the current  view to be just the photometric binaries.

        Uses the select_photometric_binaries() function to do the heavy
        lifting.'''
        ids = self.select_photometric_binaries(color, mag, siglimit=siglimit)
        bintable = au.extract_subtable_from_column(
            self.tablesegment, "FanID", ids)
        self.tablesegment = bintable

    def set_photometric_singles(self, color="V-I", mag="V", siglimit=15):
        '''Set the current view to be just the photometric single stars.

        Uses the select_photometric_binaries() function to determine binaries,
        and then selects stars which are consistent with being single
        stars.'''
        binids = self.select_photometric_binaries(
            color, mag, siglimit=siglimit)
        sinidxs = au.get_complement_indices(binids, len(self.tablesegment))
        sinids = self.tablesegment["FanID"]
        self.tablesegment = au.extract_subtable_from_column(
            self.tablesegment, "FanID", sinids)

    def plot_color_magnitude(
        self, color, mag, fmt='b*', subset=[], label="", errors=False,
        lowT=4000):
        '''Plot the working data points as a CMD.

        The color and mag arguments should be strings with the bands. If the
        subset keyword is given, this will only plot the subset with FanIDs
        specified in the subset list.'''
        if len(subset) > 0:
            subset_indices = au.astropy_table_indices(
                self.tablesegment, "FanID", subset)
        else:
            # Have the indices do nothing.
            subset_indices = np.arange(len(self.tablesegment))

        ymag = self.bandmag(mag)[subset_indices]

        xblueband, xredband = sed.split_color(color)
        xblue = self.bandmag(xblueband)[subset_indices]
        xred = self.bandmag(xredband)[subset_indices]
        xcolor = xblue - xred

        if errors:
            ymagerr = self.banderr(mag)[subset_indices]
            xblueerr = self.banderr(xblueband)[subset_indices]
            xrederr = self.banderr(xredband)[subset_indices]
            xcolorerr = sed.sum_errors(xblueerr, xrederr)
            cov = sed.calc_color_mag_covariance(color, mag, ymagerr)
            sed.plot_error_ellipse(xcolor, ymag, xcolorerr, ymagerr, cov)

        plt.plot(xcolor, ymag, fmt, label=label)
        sed.color_mag_isochrone_plot(
            color, mag, self.DSEP_lookup, lowT=lowT, age=4.0, 
            redden_EBV=self.EBV, D=self.distance)

        ylimits = plt.ylim()
        if ylimits[0] < ylimits[1]:
            plt.ylim(ylimits[::-1])
        plt.xlabel(color)
        plt.ylabel(mag)

    def plot_color_mag_difference(self, color, mag, subset=[]):
        '''Plot the difference between data and an isochrone.'''

        if len(subset) > 0:
            subset_indices = au.astropy_table_indices(
                self.tablesegment, "FanID", subset)
        else:
            # Have the indices do nothing.
            subset_indices = np.arange(len(self.tablesegment))

        magval = self.bandmag(mag)[subset_indices]
        magerr = self.banderr(mag)[subset_indices]

        blueband, redband = sed.split_color(color)
        blueval = self.bandmag(blueband)[subset_indices]
        blueerr = self.banderr(blueband)[subset_indices]
        redval = self.bandmag(redband)[subset_indices]
        rederr = self.banderr(redband)[subset_indices]
        colorval = blueval - redval
        colorerr = sed.sum_errors(blueerr, rederr)

        cov = sed.calc_color_mag_covariance(color, mag, magerr)

        colormagisochrone = sed.color_to_mag_DSEP_interpolator(
            color, mag, self.DSEP_lookup, age=self.age, redden_EBV=self.EBV,
            D=self.distance, lowT=4000)

        diffvals = sed.subtract_isochrone_from_data(
            colormagisochrone, magval, colorval.filled(np.nan))

        plt.plot(colorval, diffvals, 'k.')
        sed.plot_error_ellipse(colorval, diffvals, colorerr, magerr, cov)
        plt.xlabel(color)
        plt.ylabel(mag + " Diff")


    def plot_color_difference(self, diffcolor, xcolor, subset=[]):
        '''Plot the color difference points and an isochrone.
        
        The color to be subtracted from the isochrone is diffcolor, while the
        color to be interpolated on the isochrone is xcolor.'''

        if len(subset) > 0:
            subset_indices = au.astropy_table_indices(
                self.tablesegment, "FanID", subset)
        else:
            # Have the indices do nothing.
            subset_indices = np.arange(len(self.tablesegment))
        
        diffblueband, diffredband = sed.split_color(diffcolor)
        diffblueval = self.bandmag(diffblueband)[subset_indices]
        diffblueerr = self.banderr(diffblueband)[subset_indices]
        diffredval = self.bandmag(diffredband)[subset_indices]
        diffrederr = self.banderr(diffredband)[subset_indices]
        diffcolorval = diffblueval - diffredval
        diffcolorerr = sed.sum_errors(diffblueerr, diffrederr)

        xblueband, xredband = sed.split_color(xcolor)
        xblueval = self.bandmag(xblueband)[subset_indices]
        xblueerr = self.banderr(xblueband)[subset_indices]
        xredval = self.bandmag(xredband)[subset_indices]
        xrederr = self.banderr(xredband)[subset_indices]
        xcolorval = xblueval - xredval
        xcolorerr = sed.sum_errors(xblueerr, xrederr)

        cov = sed.calc_color_color_covariance(diffcolor, xcolor, xblueerr,
                                              xrederr)

        colorcolorisochrone = sed.color_to_color_DSEP_interpolator(
            xcolor, diffcolor, self.DSEP_lookup, age=self.age,
            redden_EBV=self.EBV, lowT=4000)

        diffvals = sed.subtract_isochrone_from_data(
            colorcolorisochrone, diffcolorval, xcolorval.filled(np.nan))

        plt.plot(xcolorval, diffvals, 'k.')
        sed.plot_error_ellipse(xcolorval, diffvals, xcolorerr, diffcolorerr,
                               cov)
        plt.xlabel(xcolor)
        plt.ylabel(diffcolor + " Diff")

    def photometric_binary_color_magnitude(
        self, color, mag, sinfmt="k.", binfmt="r*", siglimit=5, lowT=4000):
        '''Plot photometric binaries vs nonbinaries in CMD.'''
        sintargets = self.select_photometric_singles(color, mag, siglimit)
        self.plot_color_magnitude(color, mag, fmt=sinfmt, subset=sintargets,
                                  lowT=lowT, errors=True)
        
        bintargets= self.select_photometric_binaries(color, mag, siglimit)
        self.plot_color_magnitude(color, mag, fmt=binfmt, subset=bintargets,
                                  lowT=lowT, errors=True)


    def binary_color_magnitude(self, color, mag, sinfmt="k.", binfmt="r*"):
        '''Plot RV binaries vs nonbinaries in color-magnitude diagram.'''

        RVbinaries = self.select_RV_binaries()
        RVsingles = self.select_RV_singles()

        binarytable = au.extract_subtable_from_column(
            self.tablesegment, "FanID", RVbinaries)
        singletable = au.extract_subtable_from_column(
            self.tablesegment, "FanID", RVsingles)

        blueband, redband = sed.split_color(color)
        orig_table = self.tablesegment

        try:
            self.tablesegment = binarytable
            binblue = self.bandmag(blueband)
            binblueerr = self.banderr(blueband)
            binred = self.bandmag(redband)
            binrederr = self.banderr(redband)
            binmag = self.bandmag(mag)
            binmagerr = self.banderr(mag)
            self.tablesegment = singletable
            sinblue = self.bandmag(blueband)
            sinblueerr = self.banderr(blueband)
            sinred = self.bandmag(redband)
            sinrederr = self.banderr(redband)
            sinmag = self.bandmag(mag)
            sinmagerr = self.banderr(mag)
        finally:
            self.tablesegment = orig_table

        bincolor = binblue - binred
        bincolorerr = np.sqrt(binblueerr**2 + binrederr**2)
        sincolor = sinblue - sinred
        sincolorerr = np.sqrt(sinblueerr**2 + sinrederr**2)

        plt.errorbar(bincolor, binmag, binmagerr, bincolorerr, binfmt,
                     label="RV Binary")
        plt.errorbar(sincolor, sinmag, sinmagerr, sincolorerr, sinfmt,
                     label="RV Single")

        sed.color_mag_isochrone_plot(
            color, mag, self.DSEP_lookup, 1.04, age=4.0, D=self.distance,
            redden_EBV=self.EBV)
        sed.color_mag_isochrone_plot(
            color, mag, self.DSEP_lookup, 1.04, age=4.0,
            D=self.distance+self.distance_unc,
            redden_EBV=self.EBV, fmt="r--")
        sed.color_mag_isochrone_plot(
            color, mag, self.DSEP_lookup, 1.04, age=4.0,
            D=self.distance-self.distance_unc,
            redden_EBV=self.EBV, fmt="r--")
        ylimits = plt.ylim()
        if ylimits[0] < ylimits[1]:
            plt.ylim(ylimits[::-1])
        plt.xlabel(color)
        plt.ylabel(mag)

    def plot_color_color(self, ycolor, xcolor, fmt='r*', subset=[], label="",
                         errors=False, lowT=4000, init_mass=0.0):
        '''Plot working data points in color-color space.'''

        if len(subset) > 0:
            subset_indices = au.astropy_table_indices(
                self.tablesegment, "FanID", subset)
        else:
            # Have the indices do nothing.
            subset_indices = np.arange(len(self.tablesegment))

        yblueband, yredband = sed.split_color(ycolor)
        yblue = self.bandmag(yblueband)[subset_indices]
        yred = self.bandmag(yredband)[subset_indices]
        ycolorvals = yblue - yred

        xblueband, xredband = sed.split_color(xcolor)
        xblue = self.bandmag(xblueband)[subset_indices]
        xred = self.bandmag(xredband)[subset_indices]
        xcolorvals = xblue - xred

        if errors:
            xblueerr = self.banderr(xblueband)[subset_indices]
            xrederr = self.banderr(xredband)[subset_indices]
            xcolorerrs = sed.sum_errors(xblueerr, xrederr)

            yblueerr = self.banderr(yblueband)[subset_indices]
            yrederr = self.banderr(yredband)[subset_indices]
            ycolorerrs = sed.sum_errors(yblueerr, yrederr)

            cov = sed.calc_color_color_covariance(
                ycolor, xcolor, yblueerr, yrederr)
            sed.plot_error_ellipse(
                xcolorvals, ycolorvals, xcolorerrs, ycolorerrs, cov)

        plt.plot(xcolorvals, ycolorvals, fmt, label=label, ms=12)
        sed.color_color_isochrone_plot(
            ycolor, xcolor, self.DSEP_lookup, lowT=lowT, age=self.age, 
            redden_EBV=self.EBV, init_mass=init_mass)

        ylimits = plt.ylim()
        if ylimits[0] < ylimits[1]:
            plt.ylim(ylimits[::-1])
        plt.xlabel(xcolor)
        plt.ylabel(ycolor)

#       sed.plot_binary_loops(
#           ycolor, xcolor, self.DSEP_lookup, [1.0, 0.8, 0.6], age=4.0,
#           minsec=0.15, redden_EBV=self.EBV)

    def plot_photometric_binary_color_color(
        self, ycolor, xcolor, fmt="b*", label="", siglimit=5, lowT=4000):
        '''Plot the difference between data and the isochrone.

        Make a plot of how far off the data points are from the isochrone.
        '''

        yblueband, yredband = sed.split_color(ycolor)
        yblue = self.bandmag(yblueband)
        yred = self.bandmag(yredband)
        yblueerr = self.banderr(yblueband)
        yrederr = self.banderr(yredband)
        ycolorvals = yblue - yred
        ycolorerrs = sed.sum_errors(yblueerr, yrederr)

        xblueband, xredband = sed.split_color(xcolor)
        xblue = self.bandmag(xblueband)
        xred = self.bandmag(xredband)
        xblueerr = self.banderr(xblueband)
        xrederr = self.banderr(xredband)
        xcolorvals = xblue - xred
        xcolorerrs = sed.sum_errors(xblueerr, xrederr)

        cov = sed.calc_color_color_covariance(
            ycolor, xcolor, yblueerr, yrederr)

        sed.plot_color_color_minimized_isochrone_distance(
            ycolor, xcolor, DSEP_lookup, ycolorvals, xcolorvals, ycolorerrs,
            xcolorerrs, cov, age=self.age, lowT=lowT, redden_EBV=self.EBV)
        bintargets = self.select_photometric_color_binaries(
            ycolor, xcolor, siglimit=siglimit)
        self.plot_color_color(ycolor, xcolor, sinfmt=sinfmt, binfmt=binfmt)

    def select_photometric_binaries(self, color="V-I", mag="V", siglimit=15, 
                                    lowT=4000):
        '''Select photometric binaries using color-magnitude diagnostic.

        Use the color and magnitude specified in order to select candidate
        binaries from the cluster. It flags stars which are 5-sigma off of the
        DSEP isochrone by returning the FanID of the stars.
        '''
        blueband, redband = sed.split_color(color)
        blueval = self.bandmag(blueband)
        redval = self.bandmag(redband)
        blueerr = self.banderr(blueband)
        rederr = self.banderr(redband)
        colorvals = blueval - redval
        colorerrs = np.sqrt(blueerr**2 + rederr**2)

        magval = self.bandmag(mag)
        magerr = self.banderr(mag)

        cov = sed.calc_color_mag_covariance(color, mag, magerr)

        isochrone = sed.color_to_mag_DSEP_interpolator(
            color, mag, self.DSEP_lookup, age=self.age, redden_EBV=self.EBV,
            D=self.distance, lowT=lowT)

        min_colors = sed.isochrone_minimum_chi_squared(
            isochrone, colorvals, magval, colorerrs, magerr, 
            cov=cov)
        min_mags = isochrone(min_colors)

        # Can there be a better way of avoiding the diagonal function using
        # broadcasting?
        chi_squareds = np.diagonal(sed.chi_squared(
            min_mags, min_colors, magval, colorvals, magerr, colorerrs, cov=cov))

        sigmas = sed.chi_squared_to_sigma(chi_squareds, 2)
        binary_indices = np.where(sigmas > siglimit)
        if np.isinf(np.all(sigmas[binary_indices])):
            raise ValueError("Reached highest limit of sigmas.")
        return self.tablesegment["FanID"][binary_indices]

    def select_photometric_singles(self, color="V-I", mag="V", siglimit=15,
                                   lowT=4000):
        '''Select photometric single stars using color-magnitude diagnostic.

        Use the color and magnitude specified to select stars which are
        consistent with being single stars in the cluster.'''
        binids = self.select_photometric_binaries(
            color, mag, siglimit=siglimit)
        binidxs = au.astropy_table_indices(self.tablesegment, "FanID", binids)
        sinidxs = au.get_complement_indices(binidxs, len(self.tablesegment))
        sinids = self.tablesegment["FanID"][sinidxs]
        return sinids
        

    def select_photometric_color_binaries(self, ycolor, xcolor, siglimit=5,
                                          lowT=4000):
        '''Select photometric binaries using color-color diagnostic.

        Use B-V vs V-I to select candidate binaries from the cluster. It flags
        stars which are 5-sigma off of the DSEP isochrone by returning the
        FanID of the stars.'''
        yblueband, yredband = sed.split_color(ycolor)
        yblue = self.bandmag(yblueband)
        yred = self.bandmag(yredband)
        yblueerr = self.banderr(yblueband)
        yrederr = self.banderr(yredband)
        ycolorvals = yblue - yred
        ycolorerrs = sed.sum_errors(yblueerr, yrederr)

        xblueband, xredband = sed.split_color(xcolor)
        xblue = self.bandmag(xblueband)
        xred = self.bandmag(xredband)
        xblueerr = self.banderr(xblueband)
        xrederr = self.banderr(xredband)
        xcolorvals = xblue - xred
        xcolorerrs = sed.sum_errors(xblueerr, xrederr)

        cov = sed.calc_color_color_covariance(
            ycolor, xcolor, yblueerr, yrederr)

        isochrone = sed.color_to_color_DSEP_interpolator(
            xcolor, ycolor, self.DSEP_lookup, init_mass=1.0, age=self.age, 
            redden_EBV=self.EBV, bound_error=False, lowT=lowT)

        min_xcolors = sed.isochrone_minimum_chi_squared(
            isochrone, xcolorvals, ycolorvals, xcolorerrs, ycolorerrs, cov=cov)
        min_ycolors = isochrone(min_xcolors)

        # Can there be a better way of avoiding the diagonal function using
        # broadcasting?
        chi_squareds = np.diagonal(sed.chi_squared(
            min_xcolors, min_ycolors, xcolorvals, ycolorvals, xcolorerrs, 
            ycolorerrs, cov=cov))

        sigmas = sed.chi_squared_to_sigma(chi_squareds, 2)
        binary_indices = np.where(sigmas > siglimit)
        if np.isinf(np.all(sigmas[binary_indices])):
            raise ValueError("Reached highest limit of sigmas.")
        return self.tablesegment["FanID"][binary_indices]


    def select_RV_binaries(self):
        '''Select RV-confirmed binaries from the cluster.

        This uses the internal RV classification to select binary members.
        '''
        binaries = np.logical_or(
            npstr.startswith(self.tablesegment["RVClass"], "B"),
            npstr.startswith(self.tablesegment["RVClass"], "B", start=1))

        return self.tablesegment["FanID"][binaries]

    def select_RV_singles(self):
        '''Select RV-confirmed single-stars from the cluster.

        This uses the internal RV classification to select single members.
        '''
        singles = np.logical_or(
            npstr.startswith(self.tablesegment["RVClass"], "S"),
            npstr.startswith(self.tablesegment["RVClass"], "S", start=1))

        return self.tablesegment["FanID"][singles]


    def plot_binary_prediction_success(self, ycolor, xcolor, lowT=4000):
        '''Plot the success at predicting binary objects in M67.

        Selects photometric binaries from the CMD, and then maps them onto the
        color-color diagram.'''
        fullset = set(self.tablesegment["FanID"])
        photometric_binaries = set(self.select_photometric_color_binaries(
            ycolor, xcolor, siglimit=3, lowT=lowT))
        true_binaries = set(self.select_photometric_binaries(
            "V-I", "V", siglimit=3, lowT=lowT))

        true_positives = list(
            true_binaries.intersection(photometric_binaries))
        false_positives = list(photometric_binaries.difference(true_positives))
        false_negatives = list(true_binaries.difference(true_positives))
        true_negatives = list(fullset.difference(
            true_positives, false_positives, false_negatives))



        self.plot_color_color(
            ycolor, xcolor, fmt='ro', subset=true_negatives, errors=True,
            label="True Negative", lowT=lowT)
        self.plot_color_color(
            ycolor, xcolor, fmt='rx', subset=false_positives, errors=True,
            label="False Positive", lowT=lowT)
        self.plot_color_color(
            ycolor, xcolor, fmt='bx', subset=false_negatives, errors=True,
            label="False Negative", lowT=lowT)
        self.plot_color_color(
            ycolor, xcolor, fmt='bo', subset=true_positives, errors=True,
            label="True Positive", lowT=lowT)

        num_true_neg = len(true_negatives)
        num_false_pos = len(false_positives)
        num_false_neg = len(false_negatives)
        num_true_pos = len(true_positives)
        print("Number of true negatives: {0}".format(num_true_neg))
        print("Number of false negatives: {0}".format(num_false_neg))
        print("Number of false positives: {0}".format(num_false_pos))
        print("Number of true positives: {0}".format(num_true_pos))

    def single_star_scatter(self, color="V-I", mag="V", siglimit=15):
        '''Calculate the scatter along the single-star locus.'''
        single_ids = self.select_photometric_singles(color, mag, siglimit)
        single_indices = au.astropy_table_indices(
            self.tablesegment, "FanID", single_ids)
        magvals = self.bandmag(mag)[single_indices]
        magerrs = self.banderr(mag)[single_indices]
        colorvals = self.colormag(color)[single_indices]
        colorerrs = self.colorerr(color)[single_indices]
        CMD_isochrone = sed.color_to_mag_DSEP_interpolator(
            color, mag, self.DSEP_lookup, age=self.age, redden_EBV=self.EBV,
            lowT=4000, D=self.distance)
        MCD_isochrone = sed.mag_to_color_DSEP_interpolator(
            mag, color, self.DSEP_lookup, age=self.age, redden_EBV=self.EBV,
            lowT=4000, D=self.distance)
        magdiffs = sed.subtract_isochrone_from_data(
            CMD_isochrone, magvals, colorvals.filled(np.nan))
        colordiffs = sed.subtract_isochrone_from_data(
            MCD_isochrone, colorvals, magvals.filled(np.nan))
        print("Mean {0} magnitude offset: {1:.3f}".format(
            mag, np.mean(magdiffs)))
        print("Mean {0} color offset: {1:.3f}".format(
            color, np.mean(colordiffs)))
        print("Scatter in magnitude: {0:.3f}".format(np.std(magdiffs)))
        print("Scatter in color: {0:.3f}".format(np.std(colordiffs)))


def read_m67_magnitudes(tblpath=paths.M67_TABLE):
    '''Reads in the table with M67 photometry.

    Table contains BVI measurements for M67 objects.'''
    tbl = Table.read(
        str(tblpath), format="ascii.fixed_width_no_header", col_ends=(
            5, 14, 23, 32, 41, 50, 59, 68, 77, 86, 92, 98, 104, 107, 110, 113,
            117, 122, 127, 130, 138, 143, 149, 155, 161, 167, 174, 180, 192,
            204, 230), names=(
            "FanID", "x_pix", "y_pix", "B", "sig_B", "V", "sig_V", "I",
            "sig_I", "chi^2", "N_B", "N_V", "N_I", "S_prob", "G_prob",
            "Z_prob", "Y_prob", "C_prob", "N_prob", "RV_prob", "RVClass", 
            "SandID", "Zhao", "MMJ", "WOCS", "Yadav", "Casa", "Nard", "RA", 
            "DEC", "misc"))
    return tbl

def M67_data(magtable=paths.M67_TABLE, twomass_table=paths.M67_TWOMASS,
             k2_table=paths.M67_EPIC):
    '''Get a giant table with all of the M67 information.''' 
    M67_mags = read_m67_magnitudes(magtable)
    M67_twomass = Table.read(str(twomass_table), format="ascii.ipac")
    M67_objects = au.join_by_ra_dec(
        M67_mags, M67_twomass, ra1="RA", dec1="DEC", ra2="ra_01",
        dec2="dec_01", ra1_unit=u.degree, dec1_unit=u.degree)

    return M67_objects
        
def read_Bouy_15_Table_6(tblpath=paths.BOUY_TABLE_6):
    '''Reads in Table 6 from Buoy et al (2015).

    This table contains the individual DANCe objects in the Pleiades along with
    the photometry.
    '''

    tbl = Table.read(str(tblpath), format="fits")

    return tbl

def read_Bouy_15_good_members(tblpath=paths.BOUY_GOOD_MEMBERS):
    '''Reads in the good members from Bouy et al (2015).

    The good members in this case are those with a membership probability of
    greater than 75%. This table is significantly more manageable to read in
    than the full table.'''

    tbl = Table.read(str(tblpath), format="fits")
    return tbl

def read_Bouy_15_Table_2(tblpath=paths.BOUY_TABLE_2):
    '''Reads in Table 2 from Buoy et al (2015).

    Contains the empirical single-star isochrones for the Pleiades for all of
    the ugrizYJHKs bands.
    '''
    tbl = Table.read(str(tblpath), format="fits")

    return tbl

def format_Bouy_isochrone_column(band):
    '''Translate the magnitude to a format that matches the Bouy isochrone.

    This function essentially puts a "mag" at the end of the band, to make it
    compatible with the Bouy isochrone table.'''

    output = band + "mag"
    return output

def format_Bouy_isochrone_error_column(band):
    '''Match the band to the column name that holds observational errors.

    This function essentially puts a "e_" in front and "mag" at the end of hte
    band, which makes it compatible with the DANCe catalog.'''
    output = "e_" + format_Bouy_isochrone_column(band)
    return output

def Bouy_isochrone_mag_interpolator(inputcolor, outputband):
    '''Use Bouy single star isochrone to interpolate magnitude from color.

    Return an interpolator that uses the single-star isochrone in Bouy et al 
    (2015) to predict a magnitude in the given band.
    '''
    isochrones = read_Bouy_15_Table_2()

    blueband, redband = sed.split_color(inputcolor)

    blueband = format_Bouy_isochrone_column(blueband)
    redband = format_Bouy_isochrone_column(redband)
    outputband = format_Bouy_isochrone_column(outputband)

    color = isochrones[blueband] - isochrones[redband]
    outputmags = interp1d(color, isochrones[outputband], bounds_error=False)

    return outputmags

def interpolate_Bouy_magnitudes(inputcolor, outputband, colorvals):
    '''Interpolate the measured magnitudes of a Pleiad based on a color.

    Using the inputcolor provided, map colorvals to the outputband using the
    single-star isochrone.
    '''
    interpolator = Bouy_isochrone_mag_interpolator(inputcolor, outputband)

    # This is because the interpolator does not like to accept NaN values.
    masked_colorvals = np.ma.masked_invalid(colorvals)
    nonan_colorvals = masked_colorvals.filled(-999)

    newmags = interpolator(nonan_colorvals)
    return newmags

def Bouy_isochrone_color_interpolator(inputcolor, outputcolor):
    '''Use Bouy single star isochrone to interpolate one color from another.

    Return an interpolator that uses the single-star isochrone in Bouy et al
    (2015) to predict a given color.'''
    blueoutput, redoutput = sed.split_color(outputcolor)

    blueinterp = Bouy_isochrone_mag_interpolator(inputcolor, blueoutput)
    redinterp = Bouy_isochrone_mag_interpolator(inputcolor, redoutput)

    color_interpolator = sed.create_color_interpolator(blueinterp, redinterp)
    return color_interpolator

def extract_Bouy_magnitude(pleiades_data, band):
    '''Extract band measurements from the data structure.

    Since there are often formatting differences between tables, this function
    will deal with them and return the DANCe catalog data for the given band.
    '''
    bouy_band = format_Bouy_isochrone_column(band)
    return pleiades_data[bouy_band]

def extract_Bouy_color(pleiades_data, color):
    '''Extract color from the DANCe catalog.

    This function will take a color along with the given table from the DANCe
    catalog and extract that color from it.
    '''
    blueband, redband = sed.split_color(color)
    blue_bouy = extract_Bouy_magnitude(pleiades_data, blueband)
    red_bouy = extract_Bouy_magnitude(pleiades_data, redband)

    color = blue_bouy - red_bouy
    return color


def interpolate_Bouy_colors(inputcolor, outputcolor, colorvals):
    '''Interpolate colors based on single-star isochrones.

    Using the inputcolor provided, map colorvals to the outputcolor using the
    single star isochrone.'''
    interpolator = Bouy_isochrone_color_interpolator(inputcolor, outputcolor)
    
    # This is because the interpolator does not like to accept NaN values.
    masked_colorvals = np.ma.masked_invalid(colorvals)
    nonan_colorvals = masked_colorvals.filled(-999)

    newcolorvals = interpolator(nonan_colorvals)
    return newcolorvals


def data_excess_plot(
    pleiades, color_excess="g-K", mag_excess="K", input_color="i-z"):
    '''Plots the color excesses for Pleiads.

    This function performs the following tasks:
    1) Using the input color, calculate a predicted value for the
    "color_excess" color and the "mag_excess" flux. 
    2) Subtract the predicted color_excess and mag_excess from the actual
    colors and magnitudes.
    3) Plot the data points. The binaries should separate from the single stars
    in this plot.
    '''
    pleiades_input_color = extract_Bouy_color(pleiades, input_color)
    pleiades_actual_color = extract_Bouy_color(pleiades, color_excess)
    pleiades_actual_mag = extract_Bouy_magnitude(pleiades, mag_excess)

    single_color = interpolate_Bouy_colors(
        input_color, color_excess, pleiades_input_color)
    single_mag = interpolate_Bouy_magnitudes(
        input_color, mag_excess, pleiades_input_color)

    pleiades_color_excess = pleiades_actual_color - single_color
    pleiades_mag_excess = pleiades_actual_mag - single_mag

    plt.plot(pleiades_color_excess, pleiades_mag_excess, 'k.')
    plt.xlabel("({0}) - ({0})(predicted)".format(color_excess))
    plt.ylabel("{0} - {0}(predicted)".format(mag_excess))
    plt.title("Base color: {0}".format(input_color))

def HR_excess_plots(
    pleiades, color_excess="g-K", mag_excess="K", input_color="i-z"):
    '''Make two plots showing excess color and excess mag.

    In order to help interpret the full excess, this function makes two plots,
    each conserving one axis of the HR diagram. That way, there still one
    dimension of familiarity.
    '''
    pleiades_input_color = extract_Bouy_color(pleiades, input_color)
    pleiades_actual_color = extract_Bouy_color(pleiades, color_excess)
    pleiades_actual_mag = extract_Bouy_magnitude(pleiades, mag_excess)

    single_color = interpolate_Bouy_colors(
        input_color, color_excess, pleiades_input_color)
    single_mag = interpolate_Bouy_magnitudes(
        input_color, mag_excess, pleiades_input_color)

    pleiades_color_excess = pleiades_actual_color - single_color
    pleiades_mag_excess = pleiades_actual_mag - single_mag

    plt.plot(pleiades_actual_color, pleiades_actual_mag, 'ko')
    plt.plot(single_color, single_mag, 'ro')
    plt.ylim(plt.ylim()[::-1])
    plt.xlabel(color_excess)
    plt.ylabel(mag_excess)

def color_color_DSEP_comparison_plot(
    pleiades, ycolor, xcolor, DSEP_lookup, massrange=(0.12, 2.0)):
    '''Compare DSEP isochrone to Pleiades data.

    Takes a data structure and two colors to put on the axes. This function
    will plot those colors, and then overplot the expected DSEP isochrone on
    top.
    '''
    pleiades_ycolor = extract_Bouy_color(pleiades, ycolor)
    pleiades_xcolor = extract_Bouy_color(pleiades, xcolor)

    pleiades_masses = np.linspace(massrange[0], massrange[1])


    ycolor_interp = sed.mass_to_color_dsep_interpolator(
        ycolor, DSEP_lookup, age=1.0)
    dsep_ycolor = ycolor_interp(pleiades_masses)

    xcolor_interp = sed.mass_to_color_dsep_interpolator(
        xcolor, DSEP_lookup, age=1.0)
    dsep_xcolor = xcolor_interp(pleiades_masses)

    # Bin the median.
    numbins = 10
    bins = np.linspace(
        np.nanmin(pleiades_xcolor), np.nanmax(pleiades_xcolor), numbins)
    delta = bins[1] - bins[0]
    idx = np.digitize(pleiades_xcolor, bins)
    running_median = [np.nanmedian(pleiades_ycolor[idx==k]) for k in
                      range(numbins)]

#   medians, binedges, binnumber = stats.binned_statistic(
#       np.ma.masked_invalid(pleiades_xcolor),
#       np.ma.masked_invalid(pleiades_ycolor), bins=10, statistic=np.nanmedian)
#   delta = binedges[1] - binedges[0]

    plt.plot(pleiades_xcolor, pleiades_ycolor, 'k.')
    plt.plot(dsep_xcolor, dsep_ycolor, 'r-')
#   plt.plot(bins-delta/2, medians, 'b-')

def color_color_m67_plot(ycolor, xcolor, minprob):
    '''Make a color-color plot of M67 data.

    Takes two colors and plots only the stars which have a probability of
    cluster membership given in minprob.
    '''
    m67_points = read_m67_magnitudes()
    m67_high_prob_members = probability_cut(m67_points, minprob, "N_prob")
    m67_ms = filter_m67_main_sequence(m67_high_prob_members)
    m67_stars = m67_ms

    yblue, yred = sed.split_color(ycolor)
    xblue, xred = sed.split_color(xcolor)

    ycolorval = (m67_stars[m67_band(yblue)] - m67_stars[m67_band(yred)])
    xcolorval = (m67_stars[m67_band(xblue)] - m67_stars[m67_band(xred)])

    plt.plot(xcolorval, ycolorval, 'k.')
    plt.xlabel(xcolor)
    plt.ylabel(ycolor)

def m67_band(band):
    '''Selects the appropriate column label for the M67 data.'''
    if band in ["J", "H", "K"]:
        return "{0}_m".format(band.lower())
    elif band is "Ks":
        return "k_m"
    elif band in ["B", "V", "I"]:
        return band
    else:
        ValueError("{0} band not available for M67 data.".format(band))

def filter_m67_main_sequence(m67_data):
    '''Remove data points that don't seem to be on the main sequence.'''

    Vmag = m67_data["V"]
    BVcolor = m67_data["B"] - m67_data["V"]

    uppercut = (18.81 - 12.86) / (2.0 - 0.64) * (BVcolor - 2) + 18.81
    lowercut = (15.0 - 18.64) / (0.565 - 1.39) * (BVcolor - 0.565) + 15.0

    ms_indices = np.where(np.logical_and(Vmag > uppercut, Vmag < lowercut))

    return m67_data[ms_indices]

def probability_cut(clusterdata, cutvalue, probcol):
    '''Only keep most probable members of a cluster.

    Returns the subset of clusterdata which has probability greater than
    cutvalue. The column containing membership probabilities should be in
    probcol.
    '''
    highprobs = catalog.perform_cut(clusterdata, probcol, lowval=cutvalue)
    return highprobs


def plot_color_correspondences(candidate_colors, compare_color="g-K"):
    '''Plot how many colors map to a specific color using isochrones.

    Use the isochrones to map the candidate colors to the color comparison.
    This is to see which colors seem to be closest to a one-to-one
    correspondence. A line with unity slope will be shown for comparison.
    '''
    colorrange = np.linspace(-15, 15, 20)

    for color in candidate_color:
        interpolated_colors = interpolate_Bouy_colors(
            color, compare_color, colorrange)

def pleiades_color_errs(pleiades, color):
    '''Extract errors from the pleaides table for a color.'''
    
    blueband, redband = sed.split_color(color)
    blueerrs = pleiades_band_errs(pleiades, blueband)
    rederrs = pleiades_band_errs(pleiades, redband)

    colorerrs = np.sqrt(blueerrs**2 + rederrs**2)
    return colorerrs

def pleiades_band_errs(pleaides, band):
    '''Extract errors from the pleiades table for a band.'''
    errcol = format_Bouy_isochrone_error_column(band)

    return pleaides[errcol]


if __name__ == "__main__":
    pleiades_members = read_Bouy_15_good_members()

    pleiades_kepler = catalog.perform_teff_cut(pleiades_members, lowtemp=3200,
                                               teffcol="Teff")

#   pleiades_members = catalog.perform_cut(
#       pleiades_table, "Pmb", lowval=0.99)

    DSEP_lookup = {"u": 11, "g": 11, "r": 11, "i": 11, "z": 11, "Y": 8, "J": 1,
                   "H": 1, "K": 1}
    band_combinations = [("g-H", "H-K"), ("g-H", "r-i"), ("g-K", "H-K"), 
                         ("g-K", "r-i"), ("g-J", "r-i"), ("i-K", "r-i")]

    for bc in band_combinations:
        plt.figure()
        color_color_DSEP_comparison_plot(
            pleiades_kepler, bc[0], bc[1], DSEP_lookup)
        sed.simulated_color_color_diagram(
            bc[0], bc[1], 
            np.ma.median(pleiades_color_errs(pleiades_kepler, bc[0])),
            np.ma.median(pleiades_color_errs(pleiades_kepler, bc[1])),
            DSEP_lookup, binary_frac=0)
        sed.plot_mass_markers(bc[0], bc[1], DSEP_lookup, 
                              [2.0, 1.5, 1.0, 0.5, 0.2])
        sed.plot_binary_loops(bc[0], bc[1], DSEP_lookup, 
                              [2.0, 1.5, 1.0, 0.5, 0.2])
        plt.xlabel(bc[1])
        plt.ylabel(bc[0])
        plt.title("Big Star: {0} Msun. Intervals of 0.5 until {1} Msun".format(
            2, .12))

#   data_excess_plot(
#       pleiades_members, input_color="r-i", color_excess="i-K", mag_excess="i")
