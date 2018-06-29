from pathlib import Path
import subprocess
import tempfile 
import shutil

import numpy as np
from scipy.interpolate import interp1d,InterpolatedUnivariateSpline
from scipy.integrate import quad
from astropy.table import Table
import astropy_util as au

import catalog
import path_config as paths
import hrplots as hr
import matplotlib.pyplot as plt
import sed
import biovis_colors as bc

DESP_PATH = "/home/regulus/simonian/DSep/"


DSEP_lookup = {"M/Mo": -1, "LogL/Lo": -1, "LogTeff": -1, "LogG": -1, "B": 1, 
               "V": 1, "I": 1, "J": 1, "H": 1, "K": 1, "Ks": 1}

###############################################################################
# An object to interface with the DSEP isochrone #
###############################################################################

class DSEPInterpolator(object):
    '''Class to automatically handle interpolation of DSEP isochrones.'''

    def __init__(self, age, feh, Y=1, afe=2, lowT=3250, highT=7000,
                 minlogG=4.2):
        '''Create DSEP Interpolator object set to a given age and metallicity.'''
        self.iso = {}
        self.age = age
        self.feh = feh
        self.Y = Y
        self.afe = afe
        self.interpdicts = {}
        self.lowT = lowT
        self.highT = highT
        self.minlogG = minlogG

    def teff_to_logg_interpolation(self, teffvals):
        '''Interpolate teffvals to corresponding log(g) on this isochrone.

        In the case where the teff and log(g) are double-valued.'''
        interper = self._load_interpdict("LogTeff", "LogG", branch="lower")

        return interper(np.log10(teffvals))

    def mass_to_radius_interpolation_logg(self, masses):
        '''Convert mass to a radius through log(g).'''
        mass_to_logg = self._load_interpdict("M/Mo", "LogG")

        loggvals = mass_to_logg(masses)
        radius = (masses / 10**(loggvals - 4.438))**0.5
        return radius

    def mass_to_radius_interpolation_sb(self, masses):
        '''Convert mass to radius through Stefan-Boltzmann Equation.'''
        mass_to_logluminosity = self._load_interpdict("M/Mo", "LogL/Lo")
        mass_to_logteff = self._load_interpdict("M/Mo", "LogTeff")

        loglumvals = mass_to_logluminosity(masses)
        logteffvals = mass_to_logteff(masses)
        radius = 10**(loglumvals/2 - 2 * (logteffvals - np.log10(5778)))
        return radius

    def teff_to_radius_interpolation_sb(self, teffs):
        '''Convert Teff to radius through Stefan-Boltzmann.'''
        logteff_to_logluminosity = self._load_interpdict("LogTeff", "LogL/Lo")

        logteffvals = np.log10(teffs)
        try:
            loglumvals = logteff_to_logluminosity(logteffvals)
        except ValueError:
            teff_inputs = _spline_x_data(logteff_to_logluminosity)
            min_teff, max_teff = teff_inputs[0], teff_inputs[-1]
            too_low = logteffvals < min_teff
            too_high = logteffvals > max_teff
            if np.count_nonzero(too_low):
                print("Included Teffs are lower than {0:1f}".format(10**min_teff))
            if np.count_nonzero(too_high):
                print("Included Teffs are higher than {0:1f}".format(10**max_teff))
            raise
        radius = 10**(loglumvals/2 - 2 * (logteffvals - np.log10(5778)))

        return radius

    def teff_to_abs_mag(self, teffs, outmag, bands=1, branch="lower"):
        '''Convert Teff to an absolute magnitude.'''
        logteff_to_mag = self._load_interpdict("LogTeff", outmag, branch=branch)
        
        logteffvals = np.log10(teffs)
        try:
            mags = logteff_to_mag(logteffvals)
        except ValueError:
            teff_inputs = _spline_x_data(logteff_to_mag)
            min_teff, max_teff = 10**teff_inputs[0], 10**teff_inputs[-1]
            too_low = teffs < min_teff
            too_high = teffs > max_teff
            if np.count_nonzero(too_low):
                print("Included Teffs are lower than {0:1f}".format(
                    min_teff))
            if np.count_nonzero(too_high):
                print("Included Teffs are higher than {0:1f}".format(
                    max_teff))
            raise
        return mags

    def teff_to_color(self, teffs, color, bands=1, branch="lower"):
        '''Convert Teff to a color.'''
        blueband, redband = sed.split_color(color)
        blue = self.teff_to_abs_mag(
            teffs, blueband, bands=bands, branch=branch)
        red = self.teff_to_abs_mag(
            teffs, redband, bands=bands, branch=branch)

        color = blue-red
        return color

    def teff_err_to_abs_mag_err(self, teffs, teff_err, outmag, bands=1,
                                branch="lower"):
        '''Convert an error in temperature to absolute magnitude.

        Note that this function only calculates the magnitude uncertainty due
        to temperature.'''
        logteff_to_mag = self._load_interpdict(
            "LogTeff", outmag, branch=branch)

        logteff_deriv = logteff_to_mag.derivative()
        try:
            err = logteff_deriv(np.log10(teffs)) * teff_err / teffs / np.log(10)
        except ValueError:
            teff_inputs = _spline_x_data(logteff_to_mag)
            min_teff, max_teff = 10**teff_inputs[0], 10**teff_inputs[-1]
            too_low = teffs < min_teff
            too_high = teffs > max_teff
            if np.count_nonzero(too_low):
                print("Included Teffs are lower than {0:1f}".format(
                    min_teff))
            if np.count_nonzero(too_high):
                print("Included Teffs are higher than {0:1f}".format(
                    max_teff))
            raise
        return mags

        return err
        

    def teff_to_mass(self, teffs, bands=1, branch="lower"):
        '''Convert Teff to a mass.'''
        logteff_to_mass = self._load_interpdict(
            "LogTeff", "M/Mo", branch=branch)

        logteffvals = np.log10(teffs)
        try:
            mass = logteff_to_mass(logteffvals)
        except ValueError:
            teff_inputs = _spline_x_data(logteff_to_mass)
            min_teff, max_teff = 10**teff_inputs[0], 10**teff_inputs[-1]
            too_low = teffs < min_teff
            too_high = teffs > max_teff
            if np.count_nonzero(too_low):
                print("Included Teffs are lower than {0:1f}".format(
                    min_teff))
            if np.count_nonzero(too_high):
                print("Included Teffs are higher than {0:1f}".format(
                    max_teff))
            raise
        return mass

    def mass_to_abs_mag(self, masses, outmag, bands=1):
        '''Convert a mass to an absolute magnitude.'''
        mass_to_mag = self._load_interpdict("M/Mo", outmag)

        try:
            mags = mass_to_mag(masses)
        except ValueError:
            mass_inputs = _spline_x_data(mass_to_mag)
            min_mass, max_mass = mass_inputs[0], mass_inputs[-1]
            too_low = masses < min_mass
            too_high = masses > max_mass
            if np.count_nonzero(too_low):
                print("Included masses are lower than {0:1f}".format(min_mass))
            if np.count_nonzero(too_high):
                print("Included masses are higher than {0:1f}".format(max_mass))
            raise
        return mags

    def mass_to_teff(self, masses):
        '''Convert a mass to an effective temperature.'''
        mass_to_logTeff = self._load_interpdict("M/Mo", "LogTeff")

        try:
            teffs = 10**mass_to_logTeff(masses)
        except ValueError:
            mass_inputs = _spline_x_data(mass_to_logTeff)
            min_mass, max_mass = mass_inputs[0], mass_inputs[-1]
            too_low = masses < min_mass
            too_high = masses > max_mass
            if np.count_nonzero(too_low):
                print("Included masses are lower than {0:1f}".format(min_mass))
            if np.count_nonzero(too_high):
                print("Included masses are higher than {0:1f}".format(max_mass))
            raise
        return teffs

    def _interp_bound_values(self, fromcol, tocol, branch="lower"):
        '''Get the boundary values the spline between the columns.'''
        spl = self._load_interpdict(fromcol, tocol, branch=branch)

        bound1, bound2 = _bounding_box_from_spline(spl)

        return bound1, bound2


    def _load_interpdict(self, fromcol, tocol, branch="lower"):
        '''Load tuple key from interpdict if available, otherwise read it in.

        Takes a tuple key for the interpdicts dictionary. If the tuple key is
        found, the interpolator in the interpdict will be returned. If the
        tuple key is not found, then the interpolator will be created from
        isochrone data and then added to interpdict.'''
        # This scaffolding is to have a standard way of storing information
        # about whether the isochrones are single or double-valued.
        singlemarker = "single"
        lowermarker = "lower"
        uppermarker = "upper"

        if branch.lower() in ["lower"]:
            branchmarker = lowermarker
        elif branch.lower() in ["upper"]:
            branchmarker = uppermarker
        else:
            raise ValueError("Can't select branch {0}.".format(branch))

        # Useful for debugging purposes.
        # If one branch is in the dictionary, make sure the other branch is as
        # well.
        if branchmarker == lowermarker:
            altmarker = uppermarker
        else:
            altmarker = lowermarker

        try:
            interper = self.interpdicts[(fromcol, tocol, singlemarker)]
        except KeyError:
            # This branch isn't a single-valued branch.
            # Check if it's double-valued.
            try:
                interper = self.interpdicts[(fromcol, tocol, branchmarker)]
            except KeyError:
                # Branch is neither single or double valued. This means we have
                # to create it.
                assert (fromcol, tocol, altmarker) not in self.interpdicts
                # Get fromdata
                try:
                    fromblue, fromred = sed.split_color(fromcol)
                except IndexError:
                    fromiso = self._get_isochrone_data(fromcol)
                    fromdata = fromiso[fromcol]
                else:
                    fromisoblue = self._get_isochrone_data(fromblue)
                    fromisored = self._get_isochrone_data(fromred)
                    fromdata = fromisoblue[fromblue] - fromisored[fromred]
                # Get todata
                try:
                    toblue, tored = sed.split_color(tocol)
                except IndexError:
                    toiso = self._get_isochrone_data(tocol)
                    todata = toiso[tocol]
                else:
                    toisoblue = self._get_isochrone_data(toblue)
                    toisored = self._get_isochrone_data(tored)
                    todata = toisoblue[toblue] - toisored[tored]

                col_precision = {"LogTeff": 4}
                fromdata = np.round(fromdata, col_precision[fromcol])

                # Check if created branch is double-valued.
                doubletup = check_sequence_double_valued(fromdata)
                if doubletup:
                    masses = self._get_isochrone_data("M/Mo")["M/Mo"]
                    lowindices = masses <= masses[doubletup[0]]
                    highindices = masses >= masses[doubletup[0]]

                    # Make sure the arrays are ordered correctly
                    fromordered_low, toordered_low = ensure_array_increasing(
                        fromdata[lowindices], todata[lowindices])
                    fromordered_high, toordered_high = ensure_array_increasing(
                        fromdata[highindices], todata[highindices])

                    # Get rid of problematic duplicate entries
                    fromfixed_low, tofixed_low = fix_duplicate_array_values(
                        fromordered_low, toordered_low) 
                    fromfixed_high, tofixed_high = fix_duplicate_array_values(
                        fromordered_high, toordered_high) 

                    # Noise on the edges sometimes causes false minima/maxima.
                    # If that seems to be the case, the edges would be really
                    # small.
                    if len(fromfixed_low) < 3:
                        branchmarker = uppermarker

                    if len(fromfixed_high) < 3:
                       branchmarker=lowermarker

                    lowspline = InterpolatedUnivariateSpline(
                        fromfixed_low, tofixed_low, ext=2, k=1)
                    highspline = InterpolatedUnivariateSpline(
                        fromfixed_high, tofixed_high, ext=2, k=1)

                    
                    self.interpdicts[(fromcol, tocol, lowermarker)] = lowspline
                    self.interpdicts[(fromcol, tocol, uppermarker)] = highspline

                    if branchmarker == lowermarker:
                        interper = lowspline
                    else:
                        interper = highspline
                else:
                    fromordered, toordered = ensure_array_increasing(
                        fromdata, todata)
                    fromfixed, tofixed = fix_duplicate_array_values(
                        fromordered, toordered)
                    interper = InterpolatedUnivariateSpline(
                        fromfixed, tofixed, ext=2, k=1)
                    self.interpdicts[(fromcol, tocol, singlemarker)] = interper
            else: 
                assert (fromcol, tocol, altmarker) in self.interpdicts
                    
        return interper

    def _get_isochrone_data(self, col):
        '''Return the Table of isochrone data that has the current col.

        This function handles caching in the self.iso dictionary. If the data
        which includes col is already in self.iso, then it will pull from that.
        If not, then it will read from the corresponding file.
        
        The specific isochrone file used is determined by passing col to
        DSEP_lookup. DSEP_lookup is a dictionary which contains the
        corresponding band number for the column. For quantities shared by all
        band numbers, DSEP_lookup will return a negative number, and an
        arbitrary isochrone of the given age and metallicity will be used.'''
        band_num = DSEP_lookup[col]
        try:
            trimmed_table = self.iso[band_num]
        except KeyError:
            if len(self.iso) > 0 and band_num <= 0:
                trimmed_table = au.nth(self.iso.values(), 0)
            else:
                if band_num <= 0:
                    band_num = 1
                isotable = read_DSEP_isochrone(
                    self.feh, self.age, Y=self.Y, afe=self.afe, 
                    bands=band_num)
                trimmed_table = restrict_interpolation_table(
                    isotable, highT=self.highT, lowT=self.lowT,
                    minlogG=self.minlogG)
                self.iso[band_num] = trimmed_table
        return trimmed_table

###############################################################################
# External helper functions for the DSEP object #
###############################################################################

def check_sequence_double_valued(vals):
    '''Perform check if the vals are sequential

    This function assumes that vals is a coordinate that ought to be
    monotonic. If the function is double-valued, then it will not be
    monotonic and either the minimum or maximum do not lie on the endpoints.'''
    max_index = np.argmax(vals)
    min_index = np.argmin(vals)
    maximum_present = not (
        vals[max_index] == vals[0] or vals[max_index] == vals[-1])
    minimum_present = not (
        vals[min_index] == vals[0] or vals[min_index] == vals[-1])
    if maximum_present and minimum_present:
        raise ValueError("Can't Interpolate")
    elif maximum_present and not minimum_present:
        return (max_index, "max")
    elif not maximum_present and minimum_present:
        return (min_index, "min")
    else:
        return False

def alpha_bin(alphas):
    '''Assign the values of alpha to that appropriate for DSEP.'''
    # These are the alpha/Fe bins that will be fed into DSEP.
    alpha_binedges = np.arange(-0.1, 0.9, 0.2)
    # a/Fe < -0.1 corresponds to 1, and a/Fe > 0.7 corresponds to 6.
    alpha_bins = np.digitize(alphas, alpha_binedges)+1
    return alpha_bins

def alpha_compatible_with_metallicity(alphas, fehs):
    '''Validate whether the alpha values are compatible with the metallicities.

    DSEP may crash if the metallicity and alpha enhancement are not compatible.
    In particular, high alpha enhancements are only available for low
    metallicity stars.'''
    # DSEP should crash or something if the metallicity and alpha enhancement
    # are not compatible. In particular, high alpha enhancements are only
    # available for low metallicity stars. I want to ensure that this will be
    # the case before running into weird DSEP bugs.
    return np.all(np.logical_or(alphas < 0.3, fehs <= 0.0))

###############################################################################
# Flexible DSEP interpolator #
###############################################################################

def build_grid_interpolator():
    '''Build a bilinear interpolator for DSEP isochrones.'''
    ages = np.concatenate([np.arange(1, 5.25, 0.25), np.arange(5, 15.5, 0.5)])
    metallicites = np.array([-2.5, -2, -1.5, -1, 0.0, 0.2, 0.3, 0.5])
    return

def integrate_teff(age, feh, alpha, obs_teff, teff_err=130, band="Ks"):
    '''Calculate the absolute magnitude over Teff convolved with uncertainties.
    
    This function assumes age, [Fe/H], and [a/Fe] are held constant.'''
    if not alpha_compatible_with_metallicity(alpha, feh):
        raise ValueError("[a/Fe] not compatible with [Fe/H]")
    
    alpha_ind = alpha_bin(alpha)
    iso = DSEPInterpolator(age, feh, afe=alpha_ind, highT=7000)
    iso_data = iso._get_isochrone_data("Ks")
    logteffs = iso_data["LogTeff"]
    kvals = iso_data["Ks"]

    interper = iso._load_interpdict("LogTeff", "Ks")
    newinterper = interp1d(
        interper._data[0], interper._data[1], kind="linear",
        bounds_error=False, fill_value=0)
    interp_teffs = np.linspace(min(logteffs), max(logteffs), 100)
    interp_ys = newinterper(interp_teffs)

    a = max(3000, 10**min(logteffs))
    b = min(7000, 10**max(logteffs))
    mean_K, err = quad(teff_error_product, a, b,
                    args=(obs_teff, teff_err, newinterper))

    return mean_K

def integrate_alpha_over_teff(
    age, feh, obs_alpha, obs_teff, alpha_err=0.013, teff_err=135, band="Ks"):
    '''Given Teff, age, and [Fe/H], integrate absolute magnitude over [a/Fe]
    
    This convolves the polynomial over alpha with a Gaussian centered on
    obs_alpha and with a given APOGEE error.'''
    if not alpha_compatible_with_metallicity(obs_alpha, feh):
        raise ValueError("[a/Fe] and [Fe/H] are incompatible!")
    alphas = np.array([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8])

    kvals = np.zeros(len(alphas))
    for i, alpha in enumerate(alphas):
        try:
            kvals[i] = integrate_teff(
                age, feh, alpha, obs_teff=obs_teff, teff_err=teff_err,
                band=band)
        except ValueError:
            print("No [Fe/H]={0:0.2f} [a/Fe]={1:0.1f}".format(feh, alpha))
            kvals[i] = np.nan

    try:
        polycoeff = fit_polynomial(alphas, kvals)
    except ValueError:
        K_expect = np.nan
    else:
        K_expect = convolve_quartic_gaussian(polycoeff, obs_alpha, alpha_err)

    return K_expect

def integrate_metallicity_and_alpha(
    teff, age, obs_feh, obs_alpha, feh_err=0.009, alpha_err=0.013, band="Ks"):
    '''Integrate over metallicity and alpha.'''
    fehs = np.array([-2.4, -2, -1.5, -1, -0.5, 0.0, 0.2, 0.3, 0.5])

    kvals = np.zeros(len(fehs))
    for i, feh in enumerate(fehs):
        kvals[i] = integrate_alpha(teff, age, feh, obs_alpha, band=band)

    try:
        polycoeff = fit_polynomial(fehs, kvals)
    except ValueError:
        K_expect = np.nan
    else:
        K_expect = convolve_quartic_gaussian(polycoeff, obs_feh, feh_err)

    return K_expect

def teff_dependence(age, feh, alpha, band="Ks", obs_teff=5000, teff_err=130):
    '''Calculate the Teff dependence of isochrones at fixed params.'''
    if not alpha_compatible_with_metallicity(alpha, feh):
        raise ValueError("[a/Fe] not compatible with [Fe/H]")

    alpha_ind = alpha_bin(alpha)
    iso = DSEPInterpolator(age, feh, afe=alpha_ind, highT=7000)
    iso_data = iso._get_isochrone_data("Ks")
    logteffs = iso_data["LogTeff"]
    kvals = iso_data["Ks"]

    interper = iso._load_interpdict("LogTeff", "Ks")
    newinterper = interp1d(
        interper._data[0], interper._data[1], kind="linear",
        bounds_error=False, fill_value=0)
    interp_teffs = np.linspace(min(logteffs), max(logteffs), 100)
    interp_ys = newinterper(interp_teffs)

    a = max(3000, 10**min(logteffs))
    b = min(7000, 10**max(logteffs))
    print(a, b)
    K_expect = quad(teff_error_product, a, b,
                    args=(obs_teff, teff_err, newinterper))
    print("Expected K at T={0:d}: {1:.3f}".format(obs_teff, K_expect[0]))
    print("Integration Error: {0:3g}".format(K_expect[1]))

    plt.plot(10**logteffs, kvals, marker="o", ls="", color=bc.blue)
    plt.plot(10**interp_teffs, interp_ys, marker="", ls="-", color=bc.red)
    plt.xlabel("Teff")
    plt.ylabel(band)
    plt.title("Teff dependence (Age: {0:.2f}, [Fe/H]: {1:.2f}, "
              "[a/Fe]: {2:.1f})".format(age, feh, alpha))
    hr.invert_x_axis()
    hr.invert_y_axis()

def alpha_dependence_over_teff(
    age, feh, obs_teff, obs_alpha, teff_err=130, alpha_err=0.013, band="Ks"):
    '''Calculate the alpha dependence of isochrones at fixed params.'''
    alphas = np.array([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8])

    kvals = np.zeros(len(alphas))
    for i, alpha in enumerate(alphas):
        try:
            kvals[i] = integrate_teff(
                age, feh, alpha, obs_teff, teff_err=teff_err,
                band=band)
        except ValueError:
            print("No [Fe/H]={0:0.2f} [a/Fe]={1:0.1f}".format(feh, alpha))
            kvals[i] = np.nan

    try:
        polycoeff = fit_polynomial(alphas, kvals)
    except ValueError:
        K_expect = np.nan
        print("Could not predict K")
        polycoeff = fit_polynomial(alphas, kvals, bad_fit_error=False)
    else:
        if alpha_compatible_with_metallicity(obs_alpha, feh):
            K_expect = convolve_quartic_gaussian(
                polycoeff, obs_alpha, alpha_err)
            print("Expected K at [a/Fe]={0:.1f}: {1:.3f}".format(
                obs_alpha, K_expect))
        else:
            print("Expected [a/Fe] incompatible with given [Fe/H]")
    polynom = np.poly1d(polycoeff)
    interp_alphas = np.linspace(min(alphas), max(alphas), 100)
    interp_ys = polynom(interp_alphas)

    plt.plot(alphas, kvals, marker="o", ls="", color=bc.blue)
    plt.plot(interp_alphas, interp_ys, marker="", ls="-", color=bc.red)
    plt.xlabel("[a/Fe]")
    plt.ylabel(band)
    plt.title("Alpha dependence (Age: {0:.2f}, [Fe/H]: {1:.2f}), "
              "integrated over Teff".format(age, feh))
    hr.invert_y_axis()

def metallicity_dependence_over_alpha_teff(
    age, obs_feh, obs_alpha, obs_teff, feh_err=0.009, alpha_err=0.013,
    teff_err=135, band="Ks"):
    '''Calculate the metallicity dependence after integrating over [a/Fe].
    
    This convolves the absolute magnitude averaged over [a/Fe] with a Gaussian
    centered on obs_feh and with a given APOGEE error.'''
    fehs = np.array([-2.4, -2, -1.5, -1, -0.5, 0.0, 0.2, 0.3, 0.5])

    kvals = np.zeros(len(fehs))
    for i, feh in enumerate(fehs):
        kvals[i] = integrate_alpha_over_teff(
            age, feh, obs_alpha=obs_alpha, obs_teff=obs_teff,
            teff_err=teff_err, band=band)

    try:
        polycoeff = fit_polynomial(fehs, kvals)
    except ValueError:
        K_expect = np.nan
        print("Could not predict K")
        polycoeff = fit_polynomial(fehs, kvals, bad_fit_error=False)
    else:
        K_expect = convolve_quartic_gaussian(polycoeff, obs_feh, feh_err)
        print("Expected K at [Fe/H]={0:.1f}: {1:.3f}".format(
            obs_feh, K_expect))
    polynom = np.poly1d(polycoeff)
    interp_fehs = np.linspace(min(fehs), max(fehs), 100)
    interp_ys = polynom(interp_fehs)


    plt.plot(fehs, kvals, marker="o", ls="", color=bc.blue)
    plt.plot(interp_fehs, interp_ys, marker="", ls="-", color=bc.red)
    hr.invert_y_axis()
    plt.xlabel("[Fe/H]")
    plt.ylabel("<{0}>".format(band))
    plt.title("Metallicity dependence (Age: {0:.2f}, integrated over [a/Fe]"
              " and Teff)".format(age))


def age_dependence_over_metallicity_alpha(
    teff, obs_feh, obs_alpha, feh_err=0.009, alpha_err=0.013, band="Ks"):
    '''Calculate the age dependence after integrating over [a/Fe] and [Fe/H].

    This convolves the absolute magnitude averaged over [a/Fe] and [Fe/H] with
    a uniform distribution between 1-10 Gyr.'''
    ages = np.concatenate([
        np.arange(1.0, 5.0, 0.25), np.arange(5, 14.5, 0.5)])

    kvals = np.zeros(len(ages))
    for i, age in enumerate(ages):
        kvals[i] = integrate_metallicity_and_alpha(
            teff, age, obs_feh, obs_alpha, feh_err=feh_err,
            alpha_err=alpha_err, band=band)

    finite_indices = np.isfinite(kvals)
    if np.count_nonzero(finite_indices) > 1:
        a, b = (max(1, min(ages[finite_indices])), 
                min(10, max(ages[finite_indices])))
        polycoeff = np.polyfit(ages[finite_indices], kvals[finite_indices], 4)
        polynom = np.poly1d(polycoeff)
        interp_ages = np.linspace(a, b, 100)
        interp_ys = polynom(interp_ages)
        K_expect = (
            polycoeff[0] * (b**5 - a**5) / 5 +
            polycoeff[1] * (b**4 - a**4) / 4 +
            polycoeff[2] * (b**3 - a**3) / 3 +
            polycoeff[3] * (b**2 - a**2) / 2 +
            polycoeff[4] * (b**1 - a**1) / 1) / (b - a)
    else:
        K_expect = np.nan

    plt.plot(ages, kvals, marker="o", ls="", color=bc.blue)
    plt.plot(interp_ages, interp_ys, marker="", ls="-", color=bc.red)
    hr.invert_y_axis()
    plt.xlabel("Age")
    plt.ylabel("<{0}>".format(band))
    plt.title("Age dependence (Teff: {0:d}, integrated over [Fe/H] and "
              "[a/Fe])".format(teff))

def metallicity_dependence(teff, age, afe, band="Ks", obs_feh=0.0):
    '''Calculate the metallicity dependence of isochrones at fixed params.'''
    fehs = np.array([-2.4, -2, -1.5, -1, -0.5, 0.0, 0.2, 0.3, 0.5])
    feh_err = 0.009

    kvals = np.zeros(len(fehs))
    for i, feh in enumerate(fehs):
        if alpha_compatible_with_metallicity(afe, feh):
            alpha_ind = alpha_bin(afe)
            iso = DSEPInterpolator(age, feh, afe=alpha_ind)
            try:
                kvals[i] = iso.teff_to_abs_mag(teff, band)
            except ValueError:
                kvals[i]= np.nan
        else:
            print(feh)
            kvals[i] = np.nan

    try:
        polycoeff = fit_polynomial(fehs, kvals)
    except ValueError:
        K_expect = np.nan
        print("Could not predict K")
        polycoeff = fit_polynomial(fehs, kvals, bad_fit_error=False)
    else:
        K_expect = convolve_quartic_gaussian(polycoeff, obs_feh, feh_err)
        print("Expected K at [Fe/H]={0:.1f}: {1:.3f}".format(
            obs_feh, K_expect))
    polynom = np.poly1d(polycoeff)
    interp_fehs = np.linspace(min(fehs), max(fehs), 100)
    interp_ys = polynom(interp_fehs)

    plt.plot(fehs, kvals, marker="o", ls="", color=bc.blue)
    plt.plot(interp_fehs, interp_ys, marker="", ls="-", color=bc.red)
    plt.xlabel("[Fe/H]")
    plt.ylabel(band)
    plt.title("Metallicity dependence (Age: {0:.2f}, Teff: {1:d}, "
              "[a/Fe]: {2:.1f})".format(age, teff, afe))
    hr.invert_y_axis()

def alpha_dependence(teff, age, feh, band="Ks", obs_alpha=0.0, alpha_err=0.013):
    '''Calculate the alpha dependence of isochrones at fixed params.'''
    alphas = np.array([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8])

    kvals = np.zeros(len(alphas))
    for i, alpha in enumerate(alphas):
        if alpha_compatible_with_metallicity(alpha, feh):
            alpha_ind = alpha_bin(alpha)
            iso = DSEPInterpolator(age, feh, afe=alpha_ind)
            try:
                kvals[i] = iso.teff_to_abs_mag(teff, band)
            except ValueError:
                print(alpha)
                kvals[i] = np.nan
        else:
            print(alpha)
            kvals[i] = np.nan

    try:
        polycoeff = fit_polynomial(alphas, kvals)
    except ValueError:
        K_expect = np.nan
        print("Could not predict K")
        polycoeff = fit_polynomial(alphas, kvals, bad_fit_error=False)
    else:
        if alpha_compatible_with_metallicity(obs_alpha, feh):
            K_expect = convolve_quartic_gaussian(
                polycoeff, obs_alpha, alpha_err)
            print("Expected K at [a/Fe]={0:.1f}: {1:.3f}".format(
                obs_alpha, K_expect))
        else:
            print("Expected [a/Fe] incompatible with given [Fe/H]")
    polynom = np.poly1d(polycoeff)
    interp_alphas = np.linspace(min(alphas), max(alphas), 100)
    interp_ys = polynom(interp_alphas)

    plt.plot(alphas, kvals, marker="o", ls="", color=bc.blue)
    plt.plot(interp_alphas, interp_ys, marker="", ls="-", color=bc.red)
    plt.xlabel("[a/Fe]")
    plt.ylabel(band)
    plt.title("Alpha dependence (Age: {0:.2f}, Teff: {1:d}, "
              "[Fe/H]: {2:.2f})".format(age, teff, feh))


def age_dependence(teff, feh, alpha, band="Ks", K_expect_point=5.5):
    '''Calculate the alpha dependence of isochrones at fixed params.'''
    ages = np.concatenate([
        np.arange(1.0, 5.0, 0.25), np.arange(5, 14.5, 0.5)])
    endpoints = (1, 10)
    a, b = endpoints

    kvals = np.zeros(len(ages))
    for i, age in enumerate(ages):
        if alpha_compatible_with_metallicity(alpha, feh):
            alpha_ind = alpha_bin(alpha)
            iso = DSEPInterpolator(age, feh, afe=alpha_ind)
            try:
                kvals[i] = iso.teff_to_abs_mag(teff, band)
            except ValueError:
                print(age)
                kvals[i] = np.nan
        else:
            print(age)
            kvals[i] = np.nan

    finite_indices = np.isfinite(kvals)
    polycoeff = np.polyfit(ages[finite_indices], kvals[finite_indices], 4)
    polynom = np.poly1d(polycoeff)
    interp_ages = np.linspace(min(ages), max(ages), 100)
    interp_ys = polynom(interp_ages)

    try:
        K_expect = (
            polycoeff[0] * (b**5 - a**5) / 5 +
            polycoeff[1] * (b**4 - a**4) / 4 +
            polycoeff[2] * (b**3 - a**3) / 3 +
            polycoeff[3] * (b**2 - a**2) / 2 +
            polycoeff[4] * (b**1 - a**1) / 1) / (b - a)
    # Happens when b = a
    except ZeroDivisionError:
        K_expect = polycoeff[4]
    print("Expected K at age={0:.1f}: {1:.3f}".format(
        K_expect_point, K_expect))

    plt.plot(ages, kvals, marker="o", ls="", color=bc.blue)
    plt.plot(interp_ages, interp_ys, marker="", ls="-", color=bc.red)
    plt.xlabel("Age")
    plt.ylabel(band)
    plt.title("Age dependence (Teff: {0:d}, "
              "[Fe/H]: {1:.2f}, [a/Fe]: {2:.1f})".format(teff, feh, alpha))
    hr.invert_y_axis()




#######################
# Polynomial Routines #
#######################

def fit_polynomial(xvals, kvals, bad_fit_error=True):
    '''Return polynomial coefficients fitting xvals and kvals.

    The polynomial will have a maximum degree of 4, with fewer degrees if the
    number of points is lower.'''
    MAX_DEGREE = 4
    finite_indices = np.isfinite(kvals)
    num_finite = np.count_nonzero(finite_indices)
    degree = min(num_finite-1, MAX_DEGREE) 
    if bad_fit_error and degree == 0:
        raise ValueError("Cannot fit polynomial to degree")
    polycoeff = np.polyfit(
        xvals[finite_indices], kvals[finite_indices], degree)
    fullcoeff = np.concatenate([
        np.zeros((MAX_DEGREE+1)-len(polycoeff)), polycoeff])

    return fullcoeff

def convolve_quartic_gaussian(coeffs, mean, error):
    '''Convolve a quartic polynomial with a Gaussian.

    General-purpose function for integrating the product of a quartic
    polynomial with a Gaussian.'''
    K_expect = (
        coeffs[0] * (3* error**4 + 6 * error**2 * mean**2 + mean**4) +
        coeffs[1] * mean * (3 * error**2 + mean**2) +
        coeffs[2] * (error**2 + mean**2) + 
        coeffs[3] * mean + 
        coeffs[4])
    return K_expect

def teff_error_product(teff, obs_teff, teff_err, teff_interper):
    '''Return the product of K and the Teff uncertainty.'''
    kmag = teff_interper(np.log10(teff))
    product = (kmag / np.sqrt(2*np.pi*teff_err**2) * 
               np.exp(-(teff-obs_teff)**2/(2*teff_err**2)))
    return product


###############################################################################
# Plot isochrones #
###############################################################################

def plot_DSEP_isochrone_mk(interp):
    '''Plot the given isochrone in teff-MK space.
    
    Arguments to be passed to the underlying hr.absmag_teff_plot function.'''
    interp_data = interp._get_isochrone_data("Ks")
    teff_range = np.linspace(10**min(interp_data["LogTeff"]),
                             10**max(interp_data["LogTeff"]), 100)
    interp_K = interp.teff_to_abs_mag(teff_range, "Ks")
    hr.absmag_teff_plot(teff_range, interp_K, marker="", color=bc.blue,
                        linestyle="-")
    hr.absmag_teff_plot(
        10**interp_data["LogTeff"], interp_data["Ks"], marker="o",
        color=bc.red, linestyle="")
    plt.ylabel("Ks")

def plot_isochrone_at_ages(feh, ages, **kwargs):
    '''Plot isochrone at the given ages.'''
    for age in ages:
        interp = DSEPInterpolator(age, feh, highT=7000, minlogG=3.5)
        plot_DSEP_isochrone_mk(interp, **kwargs)

###############################################################################
# Spline Routines #
###############################################################################

def plot_spline_test(spl, inv_x=False, inv_y=False):
    '''Make a plot showing the behavior of the spline. 
    
    This function is used for testing whether the spline is correctly
    interpolating, or if there are problems with the interpolation routines.
    The internally-stored points will be plotted as well as a smooth sampling
    of the interpolation.
    
    For cases where plotting would require flipping axies, the inv_x and inv_y
    flags can be used to flip either the x or y axes.'''
    bbox = _bounding_box_from_spline(spl)
    xdata = _spline_x_data(spl)
    ydata = _spline_y_data(spl)

    testx = np.linspace(bbox[0], bbox[1], 1000)
    testy = spl(testx)

    plt.plot(testx, testy, 'k-')
    plt.plot(xdata, ydata, 'ro')
    if inv_x:
        hr.invert_x_axis()
    if inv_y:
        hr.invert_y_axis()

def _bounding_box_from_spline(spl):
    '''Get the bounding box from within a UnivariateSpline object.

    WARNING: This function mucks around with the internals of Univariate
    Spline, so it may be subject to breakage at any point!
    '''
    x, y = spl._data[3], spl._data[4]
    assert isinstance(x, float)
    assert isinstance(y, float)

    return x, y

def _spline_x_data(spl):
    '''Get the x data from within a UnivariateSpline Object

    WARNING: This function mucks around with the internals of Univariate
    Spline, so it may be subject to breakage at any point!
    '''
    xdata = spl._data[0]
    assert isinstance(xdata, np.ndarray)
    return xdata

def _spline_y_data(spl):
    '''Get the y data from within a UnivariateSpline Object

    WARNING: This function mucks around with the internals of Univariate
    Spline, so it may be subject to breakage at any point!
    '''
    ydata = spl._data[1]
    assert isinstance(ydata, np.ndarray)
    return ydata

###############################################################################
# Fix interpolation issues with DSEP isochrones #
###############################################################################

def fix_duplicate_array_values(xvals, yvals):
    '''Remove duplicate x-values from arrays.

    One of the problems with DSEP isochrones is that occasionally, there will
    be two adjacent points that have the same x-value, but have different
    y-values. This function will attempt to find those duplicate points and fix
    them.'''
    # Note: This algorithm assumes that there are only two simultaneous
    # duplications. Doing it for n simultaneous duplications might be tricky.
    # The DSEP interpolation causes there to be very slight numerical errors in
    # the answers. As a result, quantities that should be identical can be
    # scattered above or below what they are.
    dupmask = np.abs(np.diff(xvals)) > 1.01e-5
    valarray = np.vstack([xvals, yvals])
    # If the cases of duplication are isolated:
    if np.all(np.logical_or(dupmask[1:], dupmask[:-1])):
        meanvals = np.mean([
            valarray[:, np.hstack([np.ones(1, dtype=bool), dupmask])], 
            valarray[:, np.hstack([dupmask, np.ones(1, dtype=bool)])]], axis=0)
    else:
        splitlist = np.hsplit(valarray, np.where(dupmask)[0]+1)
        meanvals = np.concatenate([
            np.mean(vals, axis=1)[:,np.newaxis] for vals in splitlist], axis=1)
    newx = meanvals[0,:]
    newy = meanvals[1,:]
    return newx, newy

def ensure_array_increasing(xvals, yvals):
    '''Ensure the provided xvalues are increasing.

    Because creating a spline requires that xvalues are strictly increasing,
    this function assumes that the provided xvals array is either strictly
    increasing or decreasing, and if decreasing, it reverses it to ensure that
    it's increasing. The yvals are also reversed if that's the case.
    '''
    if xvals[0] > xvals[-1]:
        newxvals = xvals[::-1]
        newyvals = yvals[::-1]
    else:
        newxvals = xvals
        newyvals = yvals

    # These may require a bit of resorting due to interpolation errors. One
    # quality flag I'd like to ensure is that no drastic sorting changes occur.
    sorted_xvals_indices = np.argsort(newxvals)
    sorted_xvals = newxvals[sorted_xvals_indices]
    sorted_yvals = newyvals[sorted_xvals_indices]
    xdiffs = np.diff(newxvals)
    assert abs(min(xdiffs)) < 40*min(xdiffs[xdiffs > 0])

    return sorted_xvals, sorted_yvals

###############################################################################
# Read DSEP isochrones #
###############################################################################

def read_DSEP_isochrone(
    feh, age, bands=1, Y=1, afe=2, tabledir=paths.DSEP_OUTPUT,
    isochrones=paths.DSEP_ISOCHRONES,
    interp_exec=paths.DSEP_INTERPOLATOR_EXECUTABLE,
    split_exec=paths.DSEP_SPLITTER_EXECUTABLE):
    '''Read in a DSEP isochrone at a given metallicity and age.

    The individual isochrone tables need to be found in tabledir. If the given
    table is not found, then the isochrone will be generated automatically from
    the grid if possible.
    '''
    tablepath = (tabledir / format_DSEP_age_isochrone_filename(
        age, feh, afe, Y, bands))
    try:
        age_table = read_DSEP_age_table(tablepath)
    except FileNotFoundError:
        interpolated_split_isochrone(
            feh, outputdir=tabledir, bands=bands, Y=Y, afe=afe, 
            isochrones=isochrones, interp_exec=interp_exec, 
            split_exec=split_exec)
        age_table = read_DSEP_age_table(tablepath)

    return age_table

def read_DSEP_age_table(tablepath):
    '''Reads the post-split DSEP table.

    The table should be one which has been split from the monolithic isochrone
    file, and thus should contain only one age.
    '''
    age_table = Table.read(str(tablepath), format="ascii.commented_header",
                           header_start=-1)
    age_table.sort("M/Mo")
    return age_table

def interpolated_split_isochrone(
    feh, outputdir=paths.DSEP_OUTPUT, bands=1, Y=1, afe=2, 
    isochrones=paths.DSEP_ISOCHRONES,
    interp_exec=paths.DSEP_INTERPOLATOR_EXECUTABLE,
    split_exec=paths.DSEP_SPLITTER_EXECUTABLE):
    '''Generates isochrones at the specified metallicity.

    Functions creates isochrones at the specified [Fe/H] value, and outputs
    them to outputdir, into separate files corresponding to their age.
    Therefore, each file should correspond to a single isochrone. 
    
    The files will be in the format: a?????fehp??afep?[y??].{bands}. The 
    first set of 5 digits corresponds to the age of the isochrone, the second 
    set of two digits corresponds to the metallicity, the third set of one 
    digit corresponds to the alpha abundance, and the fourth set of two 
    digits (if present) represents the initial helium abundance. The {bands} 
    value notes the photometric bands which are contained in the isochrone.
    '''
    with tempfile.TemporaryDirectory() as tempdir_object:
        tempdir = Path(tempdir_object)
        isochrone_output = tempdir / format_DSEP_isochrone_filename(
            feh, afe, Y, bands)
        DSEP_isochrone_interpolator(feh, isochrone_output, bands, Y, 
                                    afe, interp_exec, isochrones)
        DSEP_age_splitter(isochrone_output, outputdir,
                          executable=split_exec)

def DSEP_isochrone_interpolator(
    feh, output, bands=1, y=1, alpha=2, 
    executable=paths.DSEP_INTERPOLATOR_EXECUTABLE,
    isochrones=paths.DSEP_ISOCHRONES):
    '''Runs interpolator to generate DSEP isochrones of a given metallicity.

    This is used to get a set of isochrones at a given metallicity, without
    having to worry about the grid. The isochrones will be output to the
    location in output.

    [Fe/H] should be the metallicity of the star. Bands, Y, and Alpha are
    integers which stand for options in DSEP. 
    '''
    command = [str(executable), str(bands), str(y), str(alpha), str(feh), 
               str(output)]
    subprocess.run(command, cwd=str(isochrones.parent), check=True)

def DSEP_age_splitter(inputfile, outputdir,
                      executable=paths.DSEP_SPLITTER_EXECUTABLE):
    '''Calls the isochrone splitter.

    Oftentimes the isochrones can be really annoying to read in their current
    shape. Therefore, the isochrone splitter splits the isochrones into
    separate files, each corresponding to a different age on the isochrone. The
    isochrone files will be put in outputdir.

    This function expects the paths above to be pathlib.Path objects.
    '''
    # This FORTRAN program is kinda awful. It has to be run in the same
    # directory as the file. And it will output all of the new files to the
    # same directory. 
    # As a result, we may have to mess around with the files a bit under the
    # hood. Here are the steps I would like to take.
    # 1. Split the input file into the directory and the basename.
    # 2. Create a temporary directory in the same directory as the input file.
    # 3. Move the input file into the temporary directory.
    # 4. Run the splitter on the input file, with the temporary directory as
    # the current working directory.
    # 5. Move the input file back into its original directory.
    # 6. Move the contents of the temporary directory into outputdir.
    # 7. Delete the temporary directory.
    basedir, input_filename = inputfile.parent, inputfile.name
    with tempfile.TemporaryDirectory(dir=str(basedir)) as tempdir_object:
        tempdir = Path(tempdir_object)
        shutil.copy(str(inputfile), str(tempdir))
        command = [str(executable), str(input_filename)]
        subprocess.run(command, cwd=str(tempdir))
        temp_input_file = tempdir / input_filename
        temp_input_file.unlink()
        for agefile in tempdir.iterdir():
            outputdir.mkdir(exist_ok=True)
            shutil.copy(str(agefile), str(outputdir))
        
# Maybe add something to automatically download isochrones. But I don't think
# it's particularly important now.

def restrict_interpolation_table(
    isochrone, highT=6000, lowT=3000, minlogG=4.1):
    '''Remove isochrone models which lie outside of cuts.

    These restrictions are largely to make all the color relations
    well-behaved. At too low stellar temps, the relations can be double-valued.
    At too high stellar temps, we can get stars turning off the MS.
    '''
    if lowT is not None:
        lowT = np.log10(lowT)

    if highT is not None:
        highT = np.log10(highT)
    tempcut = catalog.perform_teff_cut(
        isochrone, lowtemp=lowT, hightemp=highT, teffcol="LogTeff")
    loggcut = catalog.perform_logg_cut(tempcut, lowlogg=minlogG, loggcol="LogG")
    restricted_table = loggcut
    return restricted_table

#######################
# Filename Formatting #
#######################


def assign_DSEP_sign(val):
    '''Returns p if val is positive and n if val is negative.

    If val is zero, then it will return p anyway.
    '''
    return sign_switch(val, "p", "m", 1)

def fill_y_filename_template(template, Y):
    '''Fill in the Y portion of the filename template.'''
    if Y == 1:
        ystring = ""
    elif Y == 2:
        ystring = "y33"
    elif Y == 3:
        ystring = "y40"
    else:
        raise ValueError("Y={0:.2g} not supported.".format(Y))

    return template.format(y=ystring)

def fill_feh_filename_template(template, feh):
    '''Fills in the [Fe/H] portion of the filename template.'''
    feh_sign = assign_DSEP_sign(feh)
    num_index = template.index("d")-1
    width = int(template[num_index:num_index+1])
    return template.format(feh_sign=feh_sign, feh=int(abs(feh)*10**(width-1)))

def fill_afe_filename_template(template, afe):
    '''Fills in the [a/Fe] portion of the filename template.'''
    afe_val = 0.2 * (afe - 2)
    afe_sign = assign_DSEP_sign(afe_val)

    return template.format(afe_sign=afe_sign, afe=int(abs(afe_val)*10))

def fill_band_suffix_filename_template(template, bands):
    '''Fill in the suffix which depends on the band label.'''
    # When placing extra bands, make sure the numbers line up with the values
    # in the "iso_interp_feh.f" file. 
    if bands == 1:
        suffix = "UBVRIJHKsKp"
    elif bands == 8:
        suffix = "UKIDSS"
    elif bands == 10:
        suffix = "CFHTugriz"
    elif bands == 11:
        suffix = "SDSSugriz"
    elif 0 < bands <= 15:
        raise ValueError("Band {0} not implemented yet.".format(bands))
    else:
        raise ValueError("Band number not recognized")

    return template.format(suf=suffix)

def fill_age_filename_template(template, age):
    '''Fill in the age for a template filename.'''
    return template.format(age=int(age*1000))

def format_DSEP_age_isochrone_filename(age, feh, afe, y, bands):
    '''Formats the filename of a post-split age file.

    This format is a?????feh(p|m)??afe(p|m)?[y??].{bands}. The 5 digits after a
    stand for the age in Gyr, where an implied decimal place is after the
    second digit. The two digits after feh are the metallicity, with p for 
    positive and m for negative metallicity. After that is the 
    alpha-abundance, which follows the same pattern. If the helium abundance 
    is set and not metallicity-dependent, then there will be the extra y term 
    in the filename.

    The bands is basically a suffix which contains every band that is contained
    in the isochrone. For example, one isochrone contains UBVRIJHKsKp.'''

    age_template = "a{age:05d}"
    age_str = fill_age_filename_template(age_template, age)

    nonage_str = format_DSEP_isochrone_filename(feh, afe, y, bands)

    finalstr = "{0}{1}".format(age_str, nonage_str)

    return finalstr

def format_DSEP_isochrone_filename(feh, afe, Y, bands):
    '''Creates a filename which follows the DSEP format.

    This format is feh(p|m)??afe(p|m)?[y??].{bands}. Where the two digits after
    feh are the metallicity, with p for positive and m for negative
    metallicity. After that is the alpha-abundance, which follows the same
    pattern. If the helium abundance is set and not metallicity-dependent, then
    there will be the extra y term in the filename.

    The bands is basically a suffix which contains every band that is contained
    in the isochrone. For example, one isochrone contains UBVRIJHKsKp.'''

    afe_temp = "afe{afe_sign}{afe:01d}"
    afe_str = fill_afe_filename_template(afe_temp, afe)

    y_temp = "{y}"
    y_str = fill_y_filename_template(y_temp, Y)

    feh_temp = "feh{feh_sign}{feh:03d}"
    feh_str = fill_feh_filename_template(feh_temp, feh)

    band_temp = "{suf}"
    band_str = fill_band_suffix_filename_template(band_temp, bands)

    finalstr = "{0}{1}{2}.{3}".format(feh_str, afe_str, y_str, band_str)

    return finalstr

def sign_switch(val, pos_sym, neg_sym, zero=0):
    '''Return symbol based on sign of val.

    This function will return pos_sym if val is positive, neg_sym if val is
    negative. If val is zero, then the behavior depends on the zero flag. If
    zero is 0, then an empty string is returned. If zero is positive, then the
    positive symbol will be returned. If zero is negative, then the negative
    symbol will be returned.
    '''
    if val > 0:
        sym = pos_sym
    elif val < 0:
        sym = neg_sym
    elif val == 0:
        if zero > 0:
            sym = pos_sym
        elif zero < 0:
            sym = pos_sym
        elif zero == 0:
            sym = ""
        else:
            raise ValueError("Zero argument should be a number.")
    else:
        ValueError("Value to needs to be a number.")

    return sym
