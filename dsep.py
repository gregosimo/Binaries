from pathlib import Path
import subprocess
import tempfile 
import shutil
import collections

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

band_translation = {"H": "H", "K": "Ks", "Ks": "Ks"}

def test_teff_k_interpolation(age, feh, alpha, outcol, y=1, bands=1):
    '''Test temperature interpolation by removing and predicting single points.

    Performs a rough test of interpolation by removing single points and
    predicting what the value of those points ought to be.'''
    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw = {"height_ratios":[2, 1]},
                               sharex=True)
    afe = alpha_bin(alpha)
    iso = DSEPIsochrone.isochrone_from_file(feh, afe, y=y, bands=bands)
    met_table = iso.iso_table(age)
    restricted_table = interpolation_table_increasing_stretch(
        met_table, mono_col="LogTeff")
    teffvals, colvals = restricted_table["LogTeff"], restricted_table[outcol]
    teff_ordered, col_ordered = ensure_array_increasing(teffvals, colvals)
    teff_fixed, col_fixed = fix_duplicate_array_values(
        teff_ordered, col_ordered)

    # These will hold the |predicted-actual| values for each point, except the
    # first and last.
    linear_offtable = np.zeros(len(teff_fixed)-2)
    cubic_offtable = np.zeros(len(teff_fixed)-2)
    
    # Now make interpolators with missing pieces.
    for i, (logT, outval) in enumerate(zip(teff_fixed[1:-1], col_fixed[1:-1])):
        mask = np.ones(len(teff_fixed), dtype="bool")
        mask[i+1] = 0
        assert np.count_nonzero(mask) == len(mask)-1
        assert logT == teff_fixed[~mask][0]
        masked_teff = teff_fixed[mask]
        masked_out = col_fixed[mask]

        kinterp_linear = interp1d(masked_teff, masked_out, kind="linear")
        kinterp_cubic = interp1d(masked_teff, masked_out, kind="cubic")

        linear_offtable[i] = np.abs(kinterp_linear(logT) - outval)
        cubic_offtable[i] = np.abs(kinterp_cubic(logT) - outval)

    # Show the interpolation on a finer grid.
    test_teffs = np.linspace(teff_fixed[0], teff_fixed[-1], 1000)
    kinterp_linear = interp1d(teff_fixed, col_fixed, kind="linear")
    kinterp_cubic = interp1d(teff_fixed, col_fixed, kind="cubic")
    test_linear = kinterp_linear(test_teffs)
    test_cubic = kinterp_cubic(test_teffs)

    # Plot the interpolation
    hr.absmag_teff_plot(10**test_teffs, test_linear, color=bc.red, ls="-",
                        marker="", label="Linear", axis=a0)
    hr.absmag_teff_plot(10**test_teffs, test_cubic, color=bc.black, ls="-",
                        marker="", label="Cubic", axis=a0)
    hr.absmag_teff_plot(
        10**teff_fixed, col_fixed, color=bc.blue, ls="", marker="o", axis=a0)
    a0.set_xlabel("")
    a0.set_ylabel("Ks")
    a0.legend(loc="upper right")
    a0.set_title("Age: {0:.2f} Gyr; [Fe/H]: {1:.2f}".format(age, feh))

    # Plot the difference
    a1.semilogy(10**teff_fixed[1:-1], linear_offtable, color=bc.red, ls="-",
                marker="")
    a1.semilogy(10**teff_fixed[1:-1], cubic_offtable, color=bc.black, ls="-",
                marker="")
    a1.set_xlabel("Teff (K)")
    a1.set_ylabel("Error (mag)")

def test_feh_interpolation(age, alpha, teffs, outcol, y=1, bands=1):
    '''Plot the interpolation over [Fe/H].'''
    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw = {"height_ratios":[2, 1]},
                               sharex=True)
    afe = alpha_bin(alpha)
    # Now iterate through the [Fe/H] values.
    input_fehs = np.array([-2.5, -2, -1.5, -1, -0.5, 0.0, 0.2, 0.3, 0.5])
    interp_fehs = np.zeros(len(input_fehs))
    k_array = np.ma.zeros((len(input_fehs), len(teffs)))
    for i, ifeh in enumerate(input_fehs):
        iso = DSEPIsochrone.isochrone_from_file(ifeh, afe, y=y, bands=bands)
        interp_fehs[i] = iso.feh
        k_array[i,:] = interpolate_DSEP_isochrone_cols(
            iso, age, np.log10(teffs), incol="LogTeff", outcol="Ks",
            interp_kind="linear")

    # These will hold the |predicted-actual| values for each point, except the
    # first and last.
    linear_offtable = np.ma.zeros((len(interp_fehs)-2, len(teffs)))
    cubic_offtable = np.ma.zeros((len(interp_fehs)-2, len(teffs)))
    # Now step through [Fe/H].
    for j, (intfeh, outval) in enumerate(
            zip(interp_fehs[1:-1], k_array[1:-1,0])):
        mask = np.ones(len(interp_fehs), dtype="bool")
        mask[j+1] = 0
        fullmask = np.logical_and(mask[:,np.newaxis], np.logical_not(k_array.mask))
        for i in range(len(teffs)):
            masked_feh = interp_fehs[fullmask[:,i]]
            masked_out = k_array[:,i][fullmask[:,i]]

            kinterp_linear = interp1d(
                masked_feh, masked_out, kind="linear", bounds_error=False,
            fill_value=np.nan)
            kinterp_cubic = interp1d(
                masked_feh, masked_out, kind="cubic", bounds_error=False,
                fill_value=np.nan)

            linear_offtable[j,i] = np.abs(np.ma.masked_invalid(
                kinterp_linear(intfeh)) - outval)
            cubic_offtable[j,i] = np.abs(np.ma.masked_invalid(
                kinterp_cubic(intfeh)) - outval)

    teffindex = 0
    # Show the interpolation on a finer grid.
    test_fehs = np.linspace(interp_fehs[0], interp_fehs[-1], 1000)
    masked_feh = interp_fehs[~np.ma.getmask(k_array)[:,teffindex]]
    masked_vals = np.ma.compressed(k_array[~np.ma.getmask(k_array)[:,teffindex]])
    kinterp_linear = interp1d(masked_feh, masked_vals, kind="linear",
                              bounds_error=False, fill_value=np.nan)
    kinterp_cubic = interp1d(masked_feh, masked_vals, kind="cubic",
                             bounds_error=False, fill_value=np.nan)
    test_linear = kinterp_linear(test_fehs)
    test_cubic = kinterp_cubic(test_fehs)

    # Plot the interpolation
    a0.plot(
        test_fehs, test_linear, color=bc.red, ls="-", marker="", label="Linear")
    a0.plot(
        test_fehs, test_cubic, color=bc.black, ls="-", marker="", label="Cubic")
    a0.plot(
        interp_fehs, k_array[:,teffindex], color=bc.blue, ls="", marker="o")
    hr.invert_y_axis(a0)
    a0.set_xlabel("")
    a0.set_ylabel("Ks")
    a0.legend(loc="upper left")
    a0.set_title("DSEP Age: {0:.2f} Gyr; Teff: {1:4d} K".format(
        age, teffs[teffindex]))

    # Plot the difference
    a1.semilogy(interp_fehs[1:-1], linear_offtable[:,teffindex], color=bc.red, ls="-",
                marker="")
    a1.semilogy(interp_fehs[1:-1], cubic_offtable[:,teffindex], color=bc.black, ls="-",
                marker="")
    a1.set_xlabel("[Fe/H]")
    a1.set_ylabel("Error (mag)")



###############################################################################
# Tools needed to perform my own interpolation #
###############################################################################

class DSEPIsochrone(object):
    '''This is a structure which holds the original non-interpolated isochrone.

    One DSEPIsochrone corresponds to a full file with all ages for the given
    isochrone. It also parses the header to manually read the iron and alpha
    abundances.'''

    def __init__(
        self, feh, alpha, mixing_length, Y, Z, Zeff, phot_string, iso_dict):
        '''Take in attributes needed to define a DSEP isochrone.'''
        self.mixing_length = mixing_length
        self.Y = Y
        self.Z = Z
        self.Zeff = Zeff
        self.feh = feh
        self.alpha = alpha
        self.phot_string = phot_string
        self.iso_dict = iso_dict

    def iso_table(self, age):
        '''Return a table at the specified age.
        
        This function returns a table that is at the specified grid point.'''
        return self.iso_dict[age]

    @classmethod
    def isochrone_from_file(
        cls, feh, afe=2, y=1, bands=1, dsep_root=paths.DSEP_ISOCHRONES):
        '''Read in a DSEPIsochrone from a file.

        This function automatically locates the files in the default DSEP
        isochrones at dsep_root. The iron abundance has to be one of the
        grid points; they are not interpolated. The flags for the alpha and
        helium abundances also need to be passed. A DSEPIsochrone object will
        then be returned.'''
        filename = format_DSEP_isochrone_filename(feh, afe, y, bands)
        # The suffix and parent directory are usually identical.
        parentdir = filename.split(".")[1].replace("_2", "")
        filepath = dsep_root / parentdir / filename

        with filepath.open() as filehandle:
            # Parse the header
            topline = filehandle.readline()
            splittuple = topline.split("=")
            nages = int(splittuple[1][:splittuple[1].find(" ")])
            nmags = int(splittuple[2][:-1])
            filehandle.readline() # A ------- line
            # A header containing the following information.
            filehandle.readline() 
            isoprops = filehandle.readline().split()
            mixing_length = float(isoprops[1])
            Y = float(isoprops[2])
            Z = float(isoprops[3])
            Zeff = float(isoprops[4])
            feh = float(isoprops[5])
            alpha = float(isoprops[6][:-1])
            filehandle.readline() # A ------- line
            phot_string = filehandle.readline().split(":")[1][:-1]
            filehandle.readline() # A ------- line

            # Now start reading in the isochrone tables
            iso_dict = {}
            # This will be a list of table lines.
            # Once the function hits another AGE block, then the list of
            # strings is passed to Table.read().
            table_list = []
            line = filehandle.readline()
            splittuple = line.split("=")
            age = float(splittuple[1][:splittuple[1].find(" ", 1)])
            neeps = int(splittuple[2][:-1])
            for line in filehandle:
                # We've reached a new block.
                if line.startswith("#AGE="):
                    iso_dict[age] = Table.read(
                        table_list, format="ascii.commented_header")
                    assert len(iso_dict[age]) == neeps
                    assert len(iso_dict[age].colnames) == nmags+5
                    table_list = []
                    splittuple = line.split("=")
                    age = float(splittuple[1][:splittuple[1].find(" ", 1)])
                    neeps = int(splittuple[2][:-1])
                else:
                    table_list.append(line)
            iso_dict[age] = Table.read(
                table_list, format="ascii.commented_header")
            assert len(iso_dict[age]) == neeps
            assert len(iso_dict[age].colnames) == nmags+5


        return cls(feh, alpha, mixing_length, Y, Z, Zeff, phot_string,
                   iso_dict)

    def write_isochrone(self, filepath):
        '''Write the isochrone back to a file.'''

        with filepath.open("w") as filehandle:
            division_line = (
                "#----------------------------------------------------       \n")
            line_one = "#NUMBER OF AGES={0:2d} MAGS={1:2d}\n".format(
                len(self.iso_dict), len(self.iso_dict[1.0].colnames)-5)
            filehandle.write(line_one)
            filehandle.write(division_line)
            prop_header = (
                "#MIX-LEN  Y      Z          Zeff        [Fe/H] [a/Fe]\n")
            props = "#{0:7.4f}  {1:6.4f} {2:6.4E} {3:6.4E}  {4:5.2f}  {5:5.2f}\n".format(
                self.mixing_length, self.Y, self.Z, self.Zeff, self.feh,
                self.alpha)
            filehandle.write(prop_header)
            filehandle.write(props)
            filehandle.write(division_line)
            photline = "#**PHOTOMETRIC SYSTEM**:{0}\n".format(self.phot_string)
            filehandle.write(photline)
            filehandle.write(division_line)
            agetemplate = "#AGE={0:6.3f} EEPS={1:3d}\n"
            header = (
                "#EEP   M/Mo    LogTeff  LogG   LogL/Lo U       B       V       "
                "R       I       J       H       Ks      Kp      D51     \n")
            for age, iso_table in sorted(self.iso_dict.items()):
                # Make sure the number of EEPs is equal to the length of the
                # table
                assert (iso_table["EEP"][-1] - iso_table["EEP"][0] + 1 ==
                        len(iso_table))
                filehandle.write(agetemplate.format(age, len(iso_table)))
                filehandle.write(header)
                formats = {col: "%6.04f" for col in iso_table.colnames}
                formats["EEP"] = "%1d"
                formats["M/Mo"] = "%7.06f"
                iso_table.write(
                    filehandle, format="ascii.fixed_width_no_header",
                    formats=formats, delimiter="")
                filehandle.write("\n\n")

def interpolate_DSEP_isochrone_cols(
        iso, age, interp_in, incol="LogTeff", outcol="Ks",
        interp_kind="linear", mask_outside_bounds=True):
    '''Interpolate between the columns of an isochrone object.
    
    The full DSEPIsochrone object should be passed as an argument, along with
    an age, followed by the input
    values which should be interpolated. The columns to interpolate between
    should be given as incol and outcol.
    
    The kind of interpolation to be done should be given as interp_kind, which
    by default is linear because of the high density of points.'''
    met_table = iso.iso_dict[age]
    restricted_table = interpolation_table_increasing_stretch(
        met_table, mono_col=incol)
    invals, outvals = restricted_table[incol], restricted_table[outcol]
    in_ordered, out_ordered = ensure_array_increasing(invals, outvals)
    in_fixed, out_fixed = fix_duplicate_array_values(
        in_ordered, out_ordered)

    nonmasked_in = np.ma.compressed(interp_in)
    assert len(nonmasked_in) == len(interp_in)

    if mask_outside_bounds:
        kinterp = interp1d(
            in_fixed, out_fixed, kind=interp_kind, fill_value=np.nan, 
            bounds_error=False)
        interp_out = np.ma.masked_invalid(kinterp(nonmasked_in))
    else:
        kinterp = interp1d(
            in_fixed, out_fixed, kind=interp_kind, bounds_error=True)
        interp_out = kinterp(nonmasked_in)

    return interp_out

    

def plot_DSEP_isochrone_interpolation(teff_eval, afe=2, y=1, bands=1):
    '''Make a plot illustrating the interpolation'''
    input_fehs = np.array([-2.5, -2, -1.5, -1, -0.5, 0.0, 0.2, 0.3, 0.5])
    
    a = '''
    # Check input values
    if newfeh < input_fehs[0] or newfeh > input_fehs[-1]:
        raise ValueError("Cannot extrapolate beyond outside of "
                         "{0:.1f}-{1:.1f}.".format(
                             input_fehs[0], input_fehs[-1]))
    if newfeh > 0 and afe > 3:
        raise ValueError("Incompatible combination of [Fe/H] and [alpha/Fe].")
    '''

    # Read in DSEP isochrones over all metallicities.
    all_isos = {ifeh: DSEPIsochrone.isochrone_from_file(
        ifeh, afe, y=y, bands=bands) for ifeh in input_fehs}


    # Get the actual iron abundances for the isochrones.
    interp_fehs = np.array([all_isos[ifeh].feh for ifeh in input_fehs])
    
    age=1.0
    kvals_feh = np.zeros(len(interp_fehs))
    for i, ifeh in enumerate(input_fehs):
        met_table = all_isos[ifeh].iso_dict[age]
        restricted_table = restrict_interpolation_table(met_table, highT=7000)
        teffs = restricted_table["LogTeff"]
        ks = restricted_table["Ks"]
        teff_k_spline = interp1d(teffs, ks, kind="cubic")
        kvals_feh[i] = teff_k_spline(np.log10(teff_eval))
    feh_k_spline = interp1d(interp_fehs, kvals_feh, kind="cubic")

    test_fehs = np.linspace(fehs[0], fehs[-1], 1000)
    test_ks = feh_k_spline(test_fehs)
    k_points = feh_k_spline(interp_fehs)

    plt.plot(10**test_fehs, test_ks, 'k-')
    plt.plot(10**interp_fehs, k_points, 'ro')
    plt.xlabel("[Fe/H]")
    plt.ylabel("Ks")

    a = '''
    # Let's look at what's going on at 1 Gyr.
    age = 1.0
    # Get Teffs
    logteffs = np.zeros(len(interp_fehs))
    logteff_two = np.zeros(len(interp_fehs))
    kmags = np.zeros(len(interp_fehs))
    for i, ifeh in enumerate(input_fehs):
        eep_ind = np.argmin(np.abs(
            all_isos[input_fehs[np.where(input_fehs == 0)][0]].iso_dict[age]["LogTeff"] - 
            np.log10(5500)))
        logteffs[i] = all_isos[ifeh].iso_dict[age]["LogTeff"][eep_ind]
        logteff_two[i] = all_isos[ifeh].iso_dict[age]["LogTeff"][eep_ind+1]
        kmags[i] = all_isos[ifeh].iso_dict[age]["Ks"][eep_ind]
        print(all_isos[ifeh].iso_dict[age]["EEP"][eep_ind])

    teffspl = interp1d(interp_fehs, logteffs, kind="cubic", copy=False)
    teffspl_two = interp1d(interp_fehs, logteff_two, kind="cubic", copy=False)
    kspl = interp1d(interp_fehs, kmags, kind="cubic", copy=False)

    test_fehs = np.linspace(interp_fehs[0], interp_fehs[-1], 1000)
    test_teffs = teffspl(test_fehs)
    test_teff_two = teffspl_two(test_fehs)
    test_ks = kspl(test_fehs)

    plt.plot(test_fehs, 10**test_teffs, 'k-')
    plt.plot(test_fehs, 10**test_teff_two, 'k-')
    plt.plot(interp_fehs, 10**logteffs, 'ro')
    plt.plot(interp_fehs, 10**logteff_two, 'ro')
    '''

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

def alpha_values(afe):
    '''Return the value of a given alpha bin.'''
    if np.any(afe < 0) or np.any(afe > 6):
        raise ValueError("Don't recognize alpha flag: {0:d}".format(afe))

    return 0.2*(afe-2) 

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

def plot_DSEP_isochrone_mk(iso, age, teffcol="LogTeff", outcol="Ks"):
    '''Plot the given isochrone in teff-MK space.
    
    Arguments to be passed to the underlying hr.absmag_teff_plot function.'''
    met_table = iso.iso_table(age)
    iso_teffs = met_table[teffcol]
    iso_K = met_table[outcol]
    plot_teff = np.linspace(6500, 3500, 100)
    plot_K = interpolate_DSEP_isochrone_cols(
        iso, age, np.log10(plot_teff), incol=teffcol, outcol=outcol)
    
    hr.absmag_teff_plot(plot_teff, plot_K, marker="", color=bc.blue,
                        linestyle="-")
    hr.absmag_teff_plot(
        10**iso_teffs, iso_K, marker="o",
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



###############################################################################
# Read DSEP isochrones #
###############################################################################


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

    feh_temp = "feh{feh_sign}{feh:02d}"
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
