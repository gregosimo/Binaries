import os
import glob
import subprocess
import tempfile
import shutil
import itertools
import pickle
import collections

import numpy as np
import numpy.core.defchararray as npstr
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.stats import chi2, norm, multivariate_normal
from astropy.table import Table
from pathlib import Path
import matplotlib.pyplot as plt

import path_config as paths
import catalog
import hrplots as hr

DESP_PATH = "/home/regulus/simonian/DSep/"

class TableHolder(object):
    pass

tbl_holder = TableHolder()

def read_Casagrande_10_Table_4(tblpath=paths.CASAGRANDE_TABLE_4):
    '''Read Table 4 in Casagrande et al (2010).

    This table contains the coefficients for fitting effective temperatures to
    colors (and metallicities). These will also be useful for inverting the
    relationship to get colors as a function of effective temperature.'''

    try:
        cas = tbl_holder.casagrande_10_table_4
    except AttributeError:
        cas = Table.read(
            str(tblpath), format="ascii.fixed_width_no_header", data_start=1,
            col_starts=(0, 25, 30, 49, 54, 64, 72, 80, 96, 112, 128, 144, 152),
            col_ends=(23, 28, 32, 52, 57, 71, 79, 95, 111, 127, 143, 151, 173),
            names=["Color", "Met_start", "Met_end", "Color_start", "Color_end", 
                   "a0", "a1", "a2", "a3", "a4", "a5", "N", "unc"] )
        cas["Color"] = remove_latex_subscript_formatting(cas["Color"])
        tbl_holder.casagrande_10_table_4 = cas

    return cas

def read_Casagrande_10_Table_5(tblpath= paths.CASAGRANDE_TABLE_5):
    '''Read Table 5 in Casagrande et al (2010).

    This table contains the coefficients for fitting bolometric fluxes from the
    Earth.
    '''

    try:
        cas = tbl_holder.casagrande_10_table_5
    except AttributeError:
        cas = Table.read(
            str(tblpath), format="ascii.fixed_width_no_header", data_start=1,
            col_starts=(0, 16, 41, 46, 65, 70, 80, 88, 104, 120, 136, 152, 168, 
                        184, 192),
            col_ends=(15, 39, 44, 48, 68, 73, 87, 103, 119, 135, 151, 167, 183, 
                      191, 203),
            names=["Band", "Color", "Met_start", "Met_end", "Color_start",
                   "Color_end", "b0", "b1", "b2", "b3", "b4", "b5", "b6", "N",
                   "unc"] )
        cas["Band"] = remove_latex_subscript_formatting(cas["Band"])
        cas["Color"] = remove_latex_subscript_formatting(cas["Color"])
        tbl_holder.casagrande_10_table_5 = cas
    return cas

def remove_latex_subscript_formatting(col):
    '''Removes subscript formatting from A&A columns.
    
    For now, this will by definition strip dollar signs, and turn "_{\rm X}" 
    into just "X". More fancy rules may be added if they are called for.'''

    stripped_col = npstr.strip(col, "$")
    detexedcol = npstr.replace(npstr.replace(stripped_col, "_{\\rm ", ""), 
                               "}", "")
    return detexedcol

def distribute_color_subscript(col):
    '''Distributes the subscript for a color to the full band.

    A shorthand notation for noting which system a particular color was
    observed in is to surround the bands with parentheses, and then add a
    subscript to indicate that both bands are taken with that articular
    subscript. This function removes the parentheses, and distributes the
    subscript to both fo the bands. For example, (R-I)C becomes RC-IC. This
    will be easier to disentangle.
    '''
    try:
        newcol = col.copy()
    except AttributeError:
        # This occurs when newcol is a string, not a numpy array.
        newcol = np.array(col)
    shortened_indices = npstr.startswith(newcol, "(")
    shortened_entries = newcol[shortened_indices]

    # From outide to inside:
    # Replace left parenthesis.
    # Add ending suffix to first band
    # Remove ending parenthesis.
    distributed_entries = [
        npstr.replace(npstr.replace(npstr.replace(
            shortened, ")", ""), "-", shortened[-1] + "-"), "(", "") 
        for shortened in shortened_entries]

    newcol[shortened_indices] = distributed_entries
    return newcol

def split_color(color):
    '''Split colors into component bands.

    This takes either an array of strings (or just a single string formatted) 
    such as "RC-J" or "(B-V)T" and splits them into two arrays (or just a
    2-tuple of strings) which just have the bands, such as "RC" & "J" and "BT" &
    "VT". This can handle colors which need to be distributed.
    '''
    colorcol = np.atleast_1d(color)
    distributed_colors = distribute_color_subscript(colorcol)
    bluecolor = np.array(distributed_colors)
    redcolor = np.array(distributed_colors)
    splitcolors = npstr.split(distributed_colors, "-")

    for i,lst in enumerate(splitcolors):
        bluecolor[i] = lst[0]
        redcolor[i] = lst[1]
    
    if len(colorcol) == 1:
        return bluecolor[0], redcolor[0]
    else:
        return bluecolor, redcolor

def join_color(blueband, redband):
    '''Join the bands into a single color string.

    This takes either two arrays of strings (or just two strings) and joins
    them into an array (or just a single string) which has the color. For
    example "B" and "V" becomes "B-V". This does not make factored strings. So
    "BT" and "VT" make "BT-VT" not "(B-V)T".
    '''
    # Not working because join is a bitch.
    bluearray = np.atleast_1d(blueband)
    redarray = np.atleast_1d(redband)
    colorarray = npstr.join("-", [bluearray, redarray])

    if len(bluearray) == 1 and len(redarray) == 1:
        return colorarray
    else:
        return colorarray


###############################################################################
# Casagrande paper routines
###############################################################################

class OutOfBoundsError(ValueError):
    pass

def Casagrande_Teff(color, colorvals, metallicity,
                    extrapolation_exception=True, 
                    tblpath=paths.CASAGRANDE_TABLE_4):
    '''Calculate the Teff of a star using Casagrande (2010) calibration.

    This function calculates the Teff of a star using a given color. The string
    color needs to be given as the color argument. The actual values of color
    should be given as colorvals. And the [Fe/H] values of the stars should be
    given as metallicity. 

    The color should be given in the condensed mode, not in the distributed
    mode.

    If the colors or metallicity given are outside of the range considered by
    Casagrande et al (2010), then if extrapolation_exception is enabled, this
    function will raise a ValueError. Otherwise, it will continue normally.
    '''
    casagrande_teff_table = read_Casagrande_10_Table_4(tblpath)

    casagrande_row = casagrande_teff_table[
            casagrande_teff_table["Color"] == color]

    out_of_color_bound = np.any(np.logical_or(
           colorvals < casagrande_row["Color_start"], 
           colorvals > casagrande_row["Color_end"]))
    out_of_met_bound = np.any(np.logical_or(
            metallicity < casagrande_row["Met_start"],
            metallicity > casagrande_row["Met_end"]))

    if out_of_color_bound and out_of_met_bound and extrapolation_exception:
        raise OutOfBoundsError(("{0}={1:.2f} is out of the range {0}={2:.2f}–"
                          "{3:.2f} and [Fe/H]={4:.1f} is out of the range " 
                          "[Fe/H]={5:.1f}–{6:.1f}").format(
                              color, colorvals[0],
                              casagrande_row["Color_start"][0],
                              casagrande_row["Color_end"][0], metallicity, 
                              casagrande_row["Met_start"][0], 
                              casagrande_row["Met_end"][0]))
    elif out_of_color_bound and extrapolation_exception:
        raise OutOfBoundsError(("{0}={1:.2f} is out of the range {0}={2:.2f}–"
                         "{3:.2f.}").format(
                             color, colorvals[0], casagrande_row["Color_start"],
                             casagrande_row["Color_end"]))
    elif out_of_met_bound and extrapolation_exception:
        raise OutOfBoundsError(("[Fe/H]={0:.1f} is out of the range "
                         "[Fe/H]={1:.1f}–{2:.1f}").format(
                             metallicity, casagrande_row["Met_start"][0],
                             casagrande_row["Met_end"][0]))

    theta_eff = (casagrande_row["a0"] + casagrande_row["a1"] * colorvals + 
                  casagrande_row["a2"] * colorvals**2 + 
                  casagrande_row["a3"] * colorvals * metallicity + 
                  casagrande_row["a4"] * metallicity + 
                  casagrande_row["a5"] * metallicity**2)
    teff = 5040 / theta_eff
    return teff

def Casagrande_inverted_color(color, teffs, metallicity, 
        extrapolation_exception=True, tblpath=paths.CASAGRANDE_TABLE_4):
    '''Inverts Casagrande et al (2010) to get color from Teff and [Fe/H].

    This function is used to get empirical, calibrated colors from a
    theoretical Teff and metallicity.

    If the metallicity or Teff is outside of the calibration bounds, this
    function will throw an error, unless the extrapolate_exception flag is
    turned off.
    '''
    casagrande_teff_table = read_Casagrande_10_Table_4(tblpath)

    casagrande_row = casagrande_teff_table[
            casagrande_teff_table["Color"] == color]

    teff_start = Casagrande_Teff(color, casagrande_row["Color_end"],
                                 metallicity)
    teff_end = Casagrande_Teff(color, casagrande_row["Color_start"],
                                 metallicity)

    out_of_teff_bound = np.any(np.logical_or(
           teffs < teff_start,
           teffs > teff_end))
    out_of_met_bound = np.any(np.logical_or(
            metallicity < casagrande_row["Met_start"],
            metallicity > casagrande_row["Met_end"]))

    if out_of_teff_bound and out_of_met_bound and extrapolation_exception:
        raise OutOfBoundsError(("Teff={0:.2f} is out of the range Teff={1:.2f}–"
                          "{2:.2f} and [Fe/H]={3:.1f} is out of the range " 
                          "[Fe/H]={4:.1f}–{5:.1f}").format(
                              teffs, teff_start, teff_end, metallicity, 
                              casagrande_row["Met_start"][0], 
                              casagrande_row["Met_end"][0]))
    elif out_of_teff_bound and extrapolation_exception:
        raise OutOfBoundsError(("Teff={0:.2f} is out of the range Teff={1:.2f}–"
                         "{2:.2f}").format(
                             teffs, teff_start[0], teff_end[0]))
    elif out_of_met_bound and extrapolation_exception:
        raise OutOfBoundsError(("[Fe/H]={0:.1f} is out of the range "
                         "[Fe/H]={1:.1f}–{2:.1f}").format(
                             metallicity, casagrande_row["Met_start"][0],
                             casagrande_row["Met_end"][0]))

    # In this case, I'm basically inverting Eq. 3 in Casagrande et al (2010) by
    # treating it as a quadratic equation in color. Therefore, these will be
    # the components of the solution:
    a = casagrande_row["a2"]
    b = casagrande_row["a1"] + casagrande_row["a3"] * metallicity
    theta_eff = 5060 / teffs
    c = (casagrande_row["a0"] + casagrande_row["a4"] * metallicity +
         casagrande_row["a5"] * metallicity**2 - theta_eff)

    color = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)

    return color

def Casagrande_Bolometric_Flux(
    band, mags, color, colorvals, metallicity, extrapolation_exception=True,
    tblpath=paths.CASAGRANDE_TABLE_5):
    '''Calculate the Bolometric Flux given a magnitude and color.

    Uses Equation 5 from Casagrande et al (2010) to find the bolometric flux
    from a source. This flux is metallicity dependent.'''
    casagrande_bol_table = read_Casagrande_10_Table_5(tblpath)

    casagrande_row = casagrande_bol_table[np.logical_and(
        casagrande_bol_table["Band"] == band, 
        casagrande_bol_table["Color"] == color)]

    # Maybe move bounds checking into another function.
    out_of_color_bound = np.any(np.logical_or(
           colorvals < casagrande_row["Color_start"][0], 
           colorvals > casagrande_row["Color_end"])[0])
    out_of_met_bound = np.any(np.logical_or(
            metallicity < casagrande_row["Met_start"][0],
            metallicity > casagrande_row["Met_end"][0]))

    if out_of_color_bound and out_of_met_bound and extrapolation_exception:
        raise OutOfBoundsError(("{0}={1:.2f} is out of the range {0}={2:.2f}–"
                          "{3:.2f} and [Fe/H]={4:.1f} is out of the range " 
                          "[Fe/H]={5:.1f}–{6:.1f}").format(
                              color, colorvals[0],
                              casagrande_row["Color_start"][0],
                              casagrande_row["Color_end"][0], metallicity, 
                              casagrande_row["Met_start"][0], 
                              casagrande_row["Met_end"][0]))
    elif out_of_color_bound and extrapolation_exception:
        raise OutOfBoundsError(("{0}={1:.2f} is out of the range {0}={2:.2f}–"
                         "{3:.2f}").format(
                             color, colorvals[0], 
                             casagrande_row["Color_start"][0],
                             casagrande_row["Color_end"][0]))
    elif out_of_met_bound and extrapolation_exception:
        raise OutOfBoundsError(("[Fe/H]={0:.1f} is out of the range "
                         "[Fe/H]={1:.1f}–{2:.1f}").format(
                             metallicity, casagrande_row["Met_start"][0],
                             casagrande_row["Met_end"][0]))

    # Sum of the polynomial term.
    cr = casagrande_row
    polysum = (
        cr["b0"] + cr["b1"] * colorvals + cr["b2"] * colorvals**2 +
        cr["b3"] * colorvals**3 + cr["b4"] * metallicity * colorvals + 
        cr["b5"] * metallicity + cr["b6"] * metallicity**2)
    fbol = 10**(-0.4*mags) * polysum
    return fbol

###############################################################################
# DSEP-specific routines #
###############################################################################

# Internal DSEP Routines #
##########################

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

def assign_DSEP_sign(val):
    '''Returns p if val is positive and n if val is negative.

    If val is zero, then it will return p anyway.
    '''
    return sign_switch(val, "p", "m", 1)

def format_DSEP_isochrone_filename(feh, afe, Y, bands):
    '''Creates a filename which follows the DSEP format.

    This format is feh(p|m)??afe(p|m)?[y??].{bands}. Where the two digits after
    feh are the metallicity, with p for positive and m for negative
    metallicity. After that is the alpha-abundance, which follows the same
    pattern. If the helium abundance is set and not metallicity-dependent, then
    there will be the extra y term in the filename.

    The bands is basically a suffix which contains every band that is contained
    in the isochrone. For example, one isochrone contains UBVRIJHKsKp.'''
    afe_val = 0.2 * (afe - 2)
    feh_sign = assign_DSEP_sign(feh)
    afe_sign = assign_DSEP_sign(afe_val)

    if Y == 1:
        ystring = ""
    elif Y == 2:
        ystring = "y33"
    elif Y == 3:
        ystring = "y40"
    else:
        raise ValueError("Y={0:.2g} not supported.".format(Y))
    
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

    filename_template = "feh{0}{1:02d}afe{2}{3:01d}{4}.{5}".format(
        feh_sign, int(abs(feh)*10), afe_sign, int(abs(afe_val)*10), ystring, 
        suffix)

    return filename_template

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

    age_prefix = "a{0:05d}".format(int(age*1000))
    return age_prefix + format_DSEP_isochrone_filename(feh, afe, y, bands)

def interpolate_split_multi_isochrones(
    fehs, outputdir, bands=1, Y=1, afe=2, isochrones=paths.DSEP_ISOCHRONES,
    interp_exec=paths.DSEP_INTERPOLATOR_EXECUTABLE,
    split_exec=paths.DSEP_SPLITTER_EXECUTABLE):
    '''Generates isochrones at multiple metallicities.

    Create isochrones at the specified [Fe/H] values, and output
    them to outputdir, into separate files corresponding to their age.
    Therefore, each file should correspond to a single isochrone. 
    
    The files will be in the format: a?????fehp??afep?[y??].{bands}. The 
    first set of 5 digits corresponds to the age of the isochrone, the second 
    set of two digits corresponds to the metallicity, the third set of one 
    digit corresponds to the alpha abundance, and the fourth set of two 
    digits (if present) represents the initial helium abundance. The {bands} 
    value notes the photometric bands which are contained in the isochrone.
    '''
    for feh in fehs:
        try:
            interpolated_split_isochrone(
                feh, outputdir, bands=bands, Y=Y, afe=afe, 
                isochrones=isochrones, interp_exec=interp_exec, 
                split_exec=split_exec)
        except subprocess.CalledProcessError:
            print("Could not generate isochrone for [Fe/H]={0:.1f}".format(
                feh))


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

def read_DSEP_age_table(tablepath):
    '''Reads the post-split DSEP table.

    The table should be one which has been split from the monolithic isochrone
    file, and thus should contain only one age.
    '''
    age_table = Table.read(str(tablepath), format="ascii.commented_header",
                           header_start=-1)
    return age_table

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

# Plotting without Interpolation #
##################################

def color_mag_age_evolution(ages, DSEP_lookup, metallicity=0.0, Y=1, afe=2, 
                            lowT=3000, mag="V", color="B-V"):
    '''Plots the evolution of the color-magnitude diagram.

    This shows how given masses evolve with age on the color-magnitude diagram.
    Points of a given mass will be connected.'''
    ages = np.sort(ages)
    firstiso = read_DSEP_isochrone(metallicity, ages[0], 
                                   bands=DSEP_lookup[mag], Y=Y, afe=afe)
    firstiso = restrict_interpolation_table(
        firstiso, highT=6000, lowT=lowT, minlogG=4.1)
    standard_masses = firstiso["M/Mo"]
    bluecol, redcol = split_color(color)
    firstmag = firstiso[mag]
    firstcolor = firstiso[bluecol] - firstiso[redcol]
    plt.plot(firstcolor, firstmag, marker="*", linestyle="None", ms=12,
             label="{0:.1f} Gyr".format(ages[0]))
    for i in range(1, len(ages), 1):
        print("Age: {0:.1f}".format(ages[i]))
        second_mag_interp = mass_to_band_DSEP_interpolator(
            mag, age=ages[i], metallicity=metallicity, bands=DSEP_lookup[mag], 
            Y=Y, afe=afe, lowT=lowT)
        second_masses = standard_masses[np.where(np.logical_and(
            standard_masses > np.amin(second_mag_interp.x), standard_masses
            < np.amax(second_mag_interp.x)))]
        secondmag = second_mag_interp(second_masses)
        second_color_interp = mass_to_color_DSEP_interpolator(
            color, DSEP_lookup, age=ages[i], Y=Y, afe=afe, lowT=lowT)
        test_masses = standard_masses[np.where(np.logical_and(
            standard_masses > np.amin(second_color_interp.x), standard_masses
            < np.amax(second_color_interp.x)))]
        assert(np.all(second_masses == test_masses))
        secondcolor = second_color_interp(second_masses)


        plt.plot(secondcolor, secondmag, marker="*", linestyle="None", ms=8,
                 label="{0:.1f} Gyr".format(ages[i]))
        first_ind = np.where(standard_masses >= second_masses[0])[0][0]
        for j in range(len(second_masses)):
            plt.plot(
                [firstcolor[first_ind+j], secondcolor[j]], 
                [firstmag[first_ind+j], secondmag[j]], 
                linestyle="-", color="k", marker="None")
            assert(standard_masses[first_ind+j] == second_masses[j])
        standard_masses = second_masses
        firstmag = secondmag
        firstcolor = secondcolor

    hr.invert_y_axis()
    plt.xlabel(color)
    plt.ylabel(mag)
    plt.legend(loc="lower left")

def color_mag_mass_ratio_transition(
    massratios, DSEP_lookup, metallicity=0.0, age=4.0, Y=1, afe=2, lowT=3000, 
    mag="V", color="B-V"):
    '''Plot tracks of mass-ratio for a fixed primary mass in color-mag space.

    Will demonstrate how a companion of a specified mass-ratio will be able to
    affect the position in an HR diagram. The single-star track will
    automatically be plotted, so q=0 does not need to be included in massratios
    to demonstrate this.'''
    massratios = np.sort(massratios)
    single_iso = read_DSEP_isochrone(metallicity, age, bands=DSEP_lookup[mag], afe=afe)
    single_iso = restrict_interpolation_table(
        single_iso, highT=6000, lowT=3000, minlogG=4.1)
    standard_masses = single_iso["M/Mo"]
    prev_masses = standard_masses
    bluecol, redcol = split_color(color)
    prev_mag = single_iso[mag]
    prev_color = single_iso[bluecol] - single_iso[redcol]
    plt.plot(prev_color, prev_mag, marker="*", linestyle="None", ms=12,
             label="q = {0:.2f}".format(0.0))
    for i, q in enumerate(massratios):
        secondary_masses = q * standard_masses
        # This is because the interpolation will fail for secondaries with too
        # low of a mass-ratio.
        valid_secondaries = np.logical_and(
            secondary_masses > np.amin(standard_masses), secondary_masses
            < np.amax(standard_masses))
        valid_mag = calculate_binary_star_mag_DSEP(
            mag, standard_masses[valid_secondaries],
            secondary_masses[valid_secondaries], metallicity, DSEP_lookup, 
            age=age, Y=Y, afe=afe)
        valid_color = calculate_binary_star_color_DSEP(
            color, standard_masses[valid_secondaries],
            secondary_masses[valid_secondaries], metallicity, DSEP_lookup, 
            age=age, Y=Y, afe=afe)
        next_mag = prev_mag.copy()
        next_mag[valid_secondaries] = valid_mag
        next_color = prev_color.copy()
        next_color[valid_secondaries] = valid_color

        plt.plot(next_color[valid_secondaries], next_mag[valid_secondaries], 
                 marker="*", linestyle="None", ms=8, 
                 label="q = {0:.2f}".format(q))
        # This is to control that the first time through the loop, EVERY
        # isochrone model will have a valid data point. However, once a
        # companion is added, the number of valid models are strictly
        # increasing. So the next models should have more points.
        for j in range(len(next_color)):
            plt.plot(
                [prev_color[j], next_color[j]], 
                [prev_mag[j], next_mag[j]], 
                linestyle="-", color="k", marker="None")
        prev_mag = next_mag
        prev_color = next_color

    hr.invert_y_axis()
    plt.xlabel(color)
    plt.ylabel(mag)
    plt.legend(loc="lower left")

def color_mag_metallicity_track(metallicities, DSEP_lookup, age=1.0, Y=1, afe=2, 
                                lowT=3000, mag="V", color="B-V"):
    '''Plots the effect of metallicity on the color-magnitude diagram

    This shows how the position given masses on the HR diagram depend on
    metallicity. Points of a given mass will be connected.'''
    # We always want solar metallicity to be a part of the array.
    if np.count_nonzero(metallicities == 0.0) == 0:
        np.insert(metallicities, 0, 0.0)
    metallicities = np.sort(metallicities)
    print(metallicities)
    firstiso = read_DSEP_isochrone(metallicities[0], age, 
                                   bands=DSEP_lookup[mag], Y=Y, afe=afe)
    firstiso = restrict_interpolation_table(
        firstiso, highT=6000, lowT=lowT, minlogG=4.1)
    standard_masses = firstiso["M/Mo"]
    bluecol, redcol = split_color(color)
    firstmag = firstiso[mag]
    firstcolor = firstiso[bluecol] - firstiso[redcol]
    print("{0}: {1}".format(color, firstcolor))
    # Assumes at least one subsolar point.
    plt.plot(firstcolor, firstmag, marker="*", linestyle="None", ms=8,
             label="[Fe/H]={0:.1f}".format(metallicities[0]))
    for i in range(1, len(metallicities), 1):
        print("[Fe/H]: {0:.1f}".format(metallicities[i]))
        second_mag_interp = mass_to_band_DSEP_interpolator(
            mag, age=age, metallicity=metallicities[i], 
            bands=DSEP_lookup[mag], Y=Y, afe=afe, lowT=lowT)
        second_masses = standard_masses[np.where(np.logical_and(
            standard_masses > np.amin(second_mag_interp.x), standard_masses
            < np.amax(second_mag_interp.x)))]
        secondmag = second_mag_interp(second_masses)
        second_color_interp = mass_to_color_DSEP_interpolator(
            color, DSEP_lookup, metallicity=metallicities[i], age=age, Y=Y, 
            afe=afe, lowT=lowT)
        test_masses = standard_masses[np.where(np.logical_and(
            standard_masses > np.amin(second_color_interp.x), standard_masses
            < np.amax(second_color_interp.x)))]
        assert(np.all(second_masses == test_masses))
        secondcolor = second_color_interp(second_masses)

        if metallicities[i] == 0.0:
            ms=12
        else:
            ms=8
        print("{0}: {1}".format(color, secondcolor))
        plt.plot(secondcolor, secondmag, marker="*", linestyle="None", ms=ms,
                 label="[Fe/H]={0:.1f}".format(metallicities[i]))
        first_ind = np.where(standard_masses >= second_masses[0])[0][0]
        for j in range(len(second_masses)):
            plt.plot(
                [firstcolor[first_ind+j], secondcolor[j]], 
                [firstmag[first_ind+j], secondmag[j]], 
                linestyle="-", color="k", marker="None")
            assert(standard_masses[first_ind+j] == second_masses[j])
        standard_masses = second_masses
        firstmag = secondmag
        firstcolor = secondcolor

    hr.invert_y_axis()
    plt.xlabel(color)
    plt.ylabel(mag)
    plt.legend(loc="lower left")

def color_mag_extinction(
    reddenings, DSEP_lookup, metallicity=0.0, age=4.0, Y=1, afe=2, lowT=3000, 
    mag="V", color="B-V"):
    '''Plot tracks of mass-ratio for a fixed primary mass in color-mag space.

    Will demonstrate how a companion of a specified mass-ratio will be able to
    affect the position in an HR diagram. The single-star track will
    automatically be plotted, so q=0 does not need to be included in massratios
    to demonstrate this.'''
    if np.count_nonzero(reddenings == 0.0) == 0:
        np.insert(reddenings, 0, 0.0)
    reddenings = np.sort(reddenings)
    unextincted_iso = read_DSEP_isochrone(
        metallicity, age, bands=DSEP_lookup[mag], afe=afe)
    unextincted_iso = restrict_interpolation_table(
        unextincted_iso, highT=6000, lowT=3000, minlogG=4.1)
    standard_masses = unextincted_iso["M/Mo"]
    prev_masses = standard_masses
    bluecol, redcol = split_color(color)
    prev_mag = unextincted_iso[mag]
    prev_color = unextincted_iso[bluecol] - unextincted_iso[redcol]
    plt.plot(prev_color, prev_mag, marker="*", linestyle="None", ms=12,
             label="E(B-V) = {0:.2f}".format(reddenings[0]))
    print(reddenings)
    for i, EBV in enumerate(reddenings[1:]):
        # This is because the interpolation will fail for secondaries with too
        # low of a mass-ratio.
        reddened_mag = calculate_single_star_magnitude_DSEP(
            mag, standard_masses, metallicity, bands=DSEP_lookup[mag], age=age, Y=Y, 
            afe=afe, redden_EBV=EBV)
        reddened_color = calculate_single_star_color_DSEP(
            color, standard_masses, metallicity, DSEP_lookup, age=age, Y=Y, 
            afe=afe, redden_EBV=EBV)
        next_mag = reddened_mag
        next_color = reddened_color

        plt.plot(next_color, next_mag, marker="*", linestyle="None", ms=8, 
                 label="E(B-V) = {0:.2f}".format(EBV))
        # This is to control that the first time through the loop, EVERY
        # isochrone model will have a valid data point. However, once a
        # companion is added, the number of valid models are strictly
        # increasing. So the next models should have more points.
        for j in range(len(next_color)):
            plt.plot(
                [prev_color[j], next_color[j]], 
                [prev_mag[j], next_mag[j]], 
                linestyle="-", color="k", marker="None")
        prev_mag = next_mag
        prev_color = next_color

    hr.invert_y_axis()
    plt.xlabel(color)
    plt.ylabel(mag)
    plt.legend(loc="lower left")

# DSEP Interpolation Routines #
###############################

def DSEP_interpolation(
    fromcol, tocol, age=1.5, metallicity=0.0, bands=1, Y=1, afe=2, lowT=3000, 
    highT=6000, bound_error=True, minlogg=4.1):
    '''Return an interpolator between two DSEP isochrone quantities.

    This will return a function which, given a value of fromcol which is
    covered by the DSEP isochrone, will interpolate a value of tocol. This
    interpolation uses a grid of the given age and metallicity.
    '''
    isochrone = read_DSEP_isochrone(metallicity, age, bands=bands, Y=Y, afe=afe)

    interp_isochrone = restrict_interpolation_table(
        isochrone, highT=highT, lowT=lowT, minlogG=minlogg)

    interpolator = interp1d(interp_isochrone[fromcol], interp_isochrone[tocol], 
                            kind="linear", bounds_error=bound_error)

    wrapped_interpolator = out_of_bounds_wrapper(
        interpolator, fromcol, interp_isochrone[fromcol][0],
        interp_isochrone[fromcol][-1])

    return wrapped_interpolator

def out_of_bounds_wrapper(interpolator, colname, bound1, bound2):
    '''Wraps the interpolator in a wrapper that raises an OutofBoundsError.

    For cases where the input to the interpolator extends past the
    interpolation bounds, this funtion will cause the interpolator to raise an
    OutofBoundsError instead of a ValueError. This makes them easier to debug.
    '''
    minbound = min((bound1, bound2))
    maxbound = max((bound1, bound2))

    def exception_wrapper(x):
        try:
            return interpolator(x)
        except ValueError:
            raise OutOfBoundsError(
                "{0} is out of the range of {1:.2f}-{2:.2f}.".format(
                    colname, minbound, maxbound))

    exception_wrapper.__dict__ = interpolator.__dict__

    return exception_wrapper


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
    loggcut = catalog.perform_logg_cut(tempcut, lowlogg=minlogG)
    restricted_table = loggcut
    return restricted_table


# This is a list that I decided to use in order to persistently store
# what the single-valued colors are when going from color to mass.
single_valued_colors_path = paths.HEAD_DIR / "single_valued_colors.pickle"
try:
    with open(str(single_valued_colors_path), 'rb') as svf:
        single_valued_colors = pickle.load(svf)
except FileNotFoundError:
    single_valued_colors = []

# Like single valued colors except this will be a dictionary that holds where
# the colors branch off.
multi_valued_colors_path = paths.HEAD_DIR / "multi_valued_colors.pickle"
try:
    with open(str(multi_valued_colors_path), 'rb') as svf:
        multi_valued_colors = pickle.load(svf)
except FileNotFoundError:
    multi_valued_colors = collections.defaultdict(list)

# Interpolators to and from magnitudes and colors #

# Interpolators to deal with double-valued colors. 
def color_to_mass_DSEP_interpolator(
    color, DSEP_lookup, mass_order="descending", init_mass=0.0, age=1.5, 
    metallicity=0.0, Y=1, afe=2, bound_error=True, lowT=3000, redden_EBV=0.0):
    '''Return function to interpolate mass given a color.

    This function will return an interpolator which will map colors to masses.
    Since this relation can be double-valued, special care needs to be taken in
    order to ensure the interpolator works correctly. Before this function can
    return a relation from color, it must know whether it is single-valued or
    not. This is done through the populate_double_valued_colors() function. 
    The function is automated, but needs to be manually run whenever an
    interpolator with different stellar parameters is needed. There may be an
    automated way of doing this in the future. Once that has been run, there
    are several ways the interpolator can be constructed:

    If the relation is single-valued, then the interpolator is constructed
    straightforwardly as a function.

    If the relation is double-valued, then the behavior of this function
    depends on the parameters given.

    If init_mass is given, then the branch of the double-valued function which
    contains the specified mass is used. If not, then the behavior is
    determined by the mass_order parameter. It will assume that the colors
    passed to it were from a sequence of models with either an ascending or
    descending mass, depending on what the passed value is. This mode can't be
    used for interpolations with only a single point.
    '''
    blue, red = split_color(color)
    packet = pack_color_packet(
        color, DSEP_lookup[blue], DSEP_lookup[red], age, metallicity, Y,
        afe)
    if packet in single_valued_colors:
        blue, red = split_color(color)
        blue_DSEP = DSEP_band_converter(blue, DSEP_lookup[blue])
        red_DSEP = DSEP_band_converter(red, DSEP_lookup[red])

        blue_table = restrict_interpolation_table(read_DSEP_isochrone(
            metallicity, age, bands=DSEP_lookup[blue], Y=Y,
            afe=afe), lowT=lowT)
        red_table = restrict_interpolation_table(read_DSEP_isochrone(
            metallicity, age, bands=DSEP_lookup[red], Y=Y,
            afe=afe), lowT=lowT)

        blue_masses = blue_table["M/Mo"]
        red_masses = red_table["M/Mo"]
        assert(np.all(blue_masses == red_masses))
        masses = blue_masses
        blue_col = blue_table[blue_DSEP]
        red_col = red_table[red_DSEP]
        color_val = blue_col - red_col

        reddened_color = redden_color(color, color_val, redden_EBV)

        interpolator = interp1d(reddened_color, masses, kind="linear",
                                bounds_error=bound_error)
        color_to_mass = out_of_bounds_wrapper(interpolator, color,
                                              max(color_val), min(color_val))
    elif packet in multi_valued_colors:
        if init_mass == 0:
            color_to_mass = color_to_mass_interpolator_mass_array(
                color, DSEP_lookup, mass_order=mass_order, age=age,
                metallicity=metallicity, Y=Y, afe=afe, bound_error=bound_error,
            redden_EBV=redden_EBV)
        else:
            color_to_mass = color_to_mass_interpolator_with_mass(
                color, DSEP_lookup, init_mass, age=age,
                metallicity=metallicity, Y=Y, afe=afe, bound_error=bound_error,
            redden_EBV=redden_EBV)
    else:
        raise ValueError("Don't know how to classify this color.")

    return color_to_mass

def color_to_mass_interpolator_mass_array(
    color, DSEP_lookup, mass_order="descending", age=1.5, metallicity=0.0, 
    Y=1, afe=2, bound_error=True, lowT=3000, redden_EBV=0.0):
    '''Recover masses for a sequence of colors generated by an isochrone.

    Return a function which interpolates mass values from colors that is able
    to distinguish between double-valued  colors. This is possible only if more
    than one mass is given. It assumes that the array of masses was generated
    according to mass_order. If the mass_order is descending, that means this
    function assumes that the colors were generated by models according to
    descending mass. Hence, it is able to determine which branch of the
    double-valued color is it on based on whether the colors are ascending or
    descending.
    '''
    blue, red = split_color(color)
    blue_DSEP = DSEP_band_converter(blue, DSEP_lookup[blue])
    red_DSEP = DSEP_band_converter(red, DSEP_lookup[red])

    blue_isochrone = restrict_interpolation_table(read_DSEP_isochrone(
        metallicity, age, bands=DSEP_lookup[blue], Y=Y, afe=afe), lowT=3000)
    masses = blue_isochrone["M/Mo"]
    bluevalues = blue_isochrone[blue_DSEP]
    # If they come from the same set of bands, then pick the red band from the
    # blue isochrone. Otherwise, read in the correct isochrone corresponding to
    # the red band.
    if DSEP_lookup[blue] == DSEP_lookup[red]:
        redvalues = blue_isochrone[red_DSEP]
    else:
        red_isochrone = restrict_interpolation_table(read_DSEP_isochrone(
            metallicity, age, bands=DSEP_lookup[red], Y=Y, afe=afe), lowT=3000)
        redvalues = red_isochrone[red_DSEP]

    DSEP_color = bluevalues - redvalues
    color_packet = pack_color_packet(
        color, DSEP_lookup[blue], DSEP_lookup[red], age, metallicity, Y, afe)

    # The turnoffs and whether the values are maxima or minima
    turnoff, local = multi_valued_colors[color_packet]

    DSEP_turnoff_index = np.argmin(np.abs(DSEP_color - turnoff))

    lowmass_branch_masses = masses[:DSEP_turnoff_index+1]
    lowmass_branch_colors = DSEP_color[:DSEP_turnoff_index+1]
    highmass_branch_masses = masses[DSEP_turnoff_index:]
    highmass_branch_colors = DSEP_color[DSEP_turnoff_index:]

    lowmass_reddened_colors = redden_color(
        color, lowmass_branch_colors, redden_EBV)
    highmass_reddened_colors = redden_color(
        color, highmass_branch_colors, redden_EBV)


    lowmass_branch_interp = interp1d(
        lowmass_reddened_colors, lowmass_branch_masses, kind='linear',
        bounds_error=bound_error)
    highmass_branch_interp = interp1d(
        highmass_reddened_colors, highmass_branch_masses, kind="linear",
        bounds_error=bound_error)

    # Maybe I can tell whether it's a local max or min from the interpolated
    # values.

    def better_interpolator(colorvals):
        if colorvals[0] <= colorvals[1]:
            color_order = "ascending"
        else:
            color_order = "descending"

        color_min_index = np.argmin(colorvals)
        color_max_index = np.argmax(colorvals)

        monotonic = ((color_max_index == len(colorvals)-1 and 
                      color_min_index == 0) or 
                     (color_max_index == 0 and 
                      color_min_index == len(colorvals)-1))

        # If the points are monotonic, then just decide what branch you are on.
        if monotonic:
            try:
                if ((color_order == mass_order and local == "min") or 
                        (color_order != mass_order and local == "max")):
                    masses = highmass_branch_interp(colorvals)
                else:
                    masses = lowmass_branch_interp(colorvals)
            except ValueError:
                raise OutOfBoundsError(
                    "{0} values are not in the range {1:.2f}-{2:.2f}".format(
                        color, DSEP_color[0], DSEP_color[-1]))
        else:
            if local == "min":
                color_turnoff_index = color_min_index
            else:
                color_turnoff_index = color_max_index

            if mass_order == "descending":
                highmass_colors = colorvals[:color_turnoff_index]
                lowmass_colors = colorvals[color_turnoff_index:]
            else:
                lowmass_colors = colorvals[:color_turnoff_index]
                highmass_colors = colorvals[color_turnoff_index:]
            
            try:
                lowmass_masses = lowmass_branch_interp(lowmass_colors)
                highmass_masses = highmass_branch_interp(highmass_colors)
            except ValueError:
                raise OutOfBoundsError(
                    "{0} values are not in the range {1:.2f}-{2:.2f}".format(
                        color, DSEP_color[0], DSEP_color[-1]))
            
            if mass_order == "descending":
                mass_array_order = (highmass_masses, lowmass_masses)
            else:
                mass_array_order = (lowmass_masses, highmass_masses)

            masses = np.concatenate(mass_array_order)

        return masses

    return better_interpolator

def color_to_mass_interpolator_with_mass(
    color, DSEP_lookup, initmass, age=1.5, metallicity=0.0, Y=1, afe=2,
    bound_error=True, lowT=3000, redden_EBV=0.0):
    '''Convert color to mass assuming mass is around given initmass.

    This function will essentially assume that the mass values given are around
    initmass. So it will only use the branch corresponding to initmass. 
    '''
    blue, red = split_color(color)
    blue_DSEP = DSEP_band_converter(blue, DSEP_lookup[blue])
    red_DSEP = DSEP_band_converter(red, DSEP_lookup[red])

    blue_isochrone = restrict_interpolation_table(read_DSEP_isochrone(
        metallicity, age, bands=DSEP_lookup[blue], Y=Y, afe=afe), lowT=lowT)
    masses = blue_isochrone["M/Mo"]
    bluevalues = blue_isochrone[blue_DSEP]
    # If they come from the same set of bands, then pick the red band from the
    # blue isochrone. Otherwise, read in the correct isochrone corresponding to
    # the red band.
    if DSEP_lookup[blue] == DSEP_lookup[red]:
        redvalues = blue_isochrone[red_DSEP]
    else:
        red_isochrone = restrict_interpolation_table(read_DSEP_isochrone(
            metallicity, age, bands=DSEP_lookup[red], Y=Y, afe=afe), lowT=lowT)
        redvalues = red_isochrone[red_DSEP]

    DSEP_color = bluevalues - redvalues
    color_packet = pack_color_packet(
        color, DSEP_lookup[blue], DSEP_lookup[red], age, metallicity, Y, afe)

    # The turnoffs and whether the values are maxima or minima
    turnoff, local = multi_valued_colors[color_packet]

    DSEP_turnoff_index = np.argmin(np.abs(DSEP_color - turnoff))

    lowmass_branch_masses = masses[:DSEP_turnoff_index+1]
    lowmass_branch_colors = DSEP_color[:DSEP_turnoff_index+1]
    highmass_branch_masses = masses[DSEP_turnoff_index:]
    highmass_branch_colors = DSEP_color[DSEP_turnoff_index:]

    lowmass_reddened_colors = redden_color(
        color, lowmass_branch_colors, redden_EBV)
    highmass_reddened_colors = redden_color(
        color, highmass_branch_colors, redden_EBV)

    if initmass < lowmass_branch_masses[-1]:
        masses, colors = lowmass_branch_masses, lowmass_reddened_colors
    else:
        masses, colors = highmass_branch_masses, highmass_reddened_colors

    interper = interp1d(
        colors, masses, kind="linear", bounds_error=bound_error)

    wrapped_interpolator = out_of_bounds_wrapper(
        interper, color, colors[0], colors[-1])

    return wrapped_interpolator
            
def test_color_to_mass_interpolator(
    color, DSEP_lookup, init_mass=0.0, mass_order="descending", age=1.0,
    metallicity=0.0, Y=1, afe=2, lowT=3000):
    '''Test the performance of the interpolator compared to the actual
    isochrone for a specific color'''
    blue, red = split_color(color)
    blue_DSEP = DSEP_band_converter(blue, DSEP_lookup[blue])
    red_DSEP = DSEP_band_converter(red, DSEP_lookup[red])

    blue_isochrone = read_DSEP_isochrone(
        metallicity, age, bands=DSEP_lookup[blue], Y=Y,
        afe=afe)
    red_isochrone = read_DSEP_isochrone(
        metallicity, age, bands=DSEP_lookup[red], Y=Y,
        afe=afe)

    blue_restricted = restrict_interpolation_table(blue_isochrone, lowT=lowT)
    red_restricted = restrict_interpolation_table(red_isochrone, lowT=lowT)

    blue_col = blue_restricted[blue_DSEP]
    red_col = red_restricted[red_DSEP]
    blue_mass_col = blue_restricted["M/Mo"]
    red_mass_col = red_restricted["M/Mo"]

    assert(np.all(blue_mass_col == red_mass_col))

    color_DSEP = blue_col - red_col

    interpolator = color_to_mass_DSEP_interpolator(
        color, DSEP_lookup, init_mass=init_mass,
        mass_order=mass_order, age=age, metallicity=metallicity, Y=Y, afe=afe)
    interpolated_color = np.linspace(
        min(color_DSEP), max(color_DSEP), 100)
    interpolated_masses= interpolator(interpolated_color)

    plt.plot(color_DSEP, blue_mass_col, 'r*')
    plt.plot(interpolated_color, interpolated_masses, 'k-')

def populate_double_valued_colors(
    bands, DSEP_lookup, age=1.5, metallicity=0.0, Y=1, afe=2):
    '''Populates single and double-valued colors.

    Uses an automatic algorithm to go through the bands and insert them into
    the single- and multi-valued color arrays.'''  
    for color in iterate_colors(bands):
        blue, red = split_color(color)
        color_packet = pack_color_packet(
            color, DSEP_lookup[blue], DSEP_lookup[red], age, metallicity, Y, 
            afe)
        if (color_packet not in single_valued_colors and 
                color_packet not in multi_valued_colors):
            double_params = check_if_double_valued(
                color, DSEP_lookup, age=age, metallicity=metallicity, Y=Y, 
                afe=afe)
            if double_params:
                multi_valued_colors[color_packet] = double_params
            else: 
                single_valued_colors.append(color_packet)

    with open(str(single_valued_colors_path), "wb") as svf:
        pickle.dump(single_valued_colors, svf)
    with open(str(multi_valued_colors_path), "wb") as mvf:
        pickle.dump(multi_valued_colors, mvf)
    



def check_if_double_valued(
    color, DSEP_lookup, age=1.5, metallicity=0.0, Y=1, afe=2, lowT=3000):
    '''Performs a primitive check to see if the color is double-valued.

    This function essentailly checks whether the DSEP models predict that the
    color should be double-valued. It does this by checking whether the
    maximum value is at the endpoints, in which case the color is
    single-valued. If there are multiple overlaps, then this function will
    fail.
    '''
    blue, red = split_color(color)
    blue_DSEP = DSEP_band_converter(blue, DSEP_lookup[blue])
    red_DSEP = DSEP_band_converter(red, DSEP_lookup[red])

    blue_isochrone = restrict_interpolation_table(read_DSEP_isochrone(
        metallicity, age, bands=DSEP_lookup[blue], Y=Y, afe=afe), lowT=lowT)
    masses = blue_isochrone["M/Mo"]
    bluevalues = blue_isochrone[blue_DSEP]
    # If they come from the same set of bands, then pick the red band from the
    # blue isochrone. Otherwise, read in the correct isochrone corresponding to
    # the red band.
    if DSEP_lookup[blue] == DSEP_lookup[red]:
        redvalues = blue_isochrone[red_DSEP]
    else:
        red_isochrone = restrict_interpolation_table(read_DSEP_isochrone(
            metallicity, age, bands=DSEP_lookup[red], Y=Y, afe=afe), lowT=lowT)
        redvalues = red_isochrone[red_DSEP]

    DSEP_color = bluevalues - redvalues
    max_index = np.argmax(DSEP_color)
    min_index = np.argmin(DSEP_color)

    maximum_present = not (max_index == 0 or max_index == len(DSEP_color)-1)
    minimum_present = not (min_index == 0 or min_index == len(DSEP_color)-1)
    if maximum_present and minimum_present:
        raise ValueError("Color {0} is too complicated to "
                         "interpolate".format(color))
    elif maximum_present and not minimum_present:
        return (DSEP_color[max_index], "max")
    elif not maximum_present and minimum_present:
        return (DSEP_color[min_index], "min")
    else:
        return False

def test_multi_valued_colors(
    bands, DSEP_lookup, age=1.5, metallicity=0.0, Y=1, afe=2, lowT=3000):
    '''Check whether colors are correctly predicted as double-valued.

    Take a bunch of bands and iterate through combinations of them to run the
    double-value checker, as well as plot them for verification by eye.
    '''
    for color in iterate_colors(bands):
        blue, red = split_color(color)
        blue_DSEP = DSEP_band_converter(blue, DSEP_lookup[blue])
        red_DSEP = DSEP_band_converter(red, DSEP_lookup[red])

        blue_isochrone = restrict_interpolation_table(read_DSEP_isochrone(
            metallicity, age, bands=DSEP_lookup[blue], Y=Y, afe=afe), lowT=lowT)
        masses = blue_isochrone["M/Mo"]
        bluevalues = blue_isochrone[blue_DSEP]
        # If they come from the same set of bands, then pick the red band from the
        # blue isochrone. Otherwise, read in the correct isochrone corresponding to
        # the red band.
        if DSEP_lookup[blue] == DSEP_lookup[red]:
            redvalues = blue_isochrone[red_DSEP]
        else:
            red_isochrone = restrict_interpolation_table(read_DSEP_isochrone(
                metallicity, age, bands=DSEP_lookup[red], Y=Y, afe=afe), 
                lowT=lowT)
            redvalues = red_isochrone[red_DSEP]

        DSEP_color = bluevalues - redvalues
        double_valued = check_if_double_valued(
            color, DSEP_lookup, age=age, metallicity=metallicity, Y=Y, afe=afe)

        plt.plot(masses, DSEP_color, 'r*')
        plt.xlabel("Mass")
        plt.ylabel(color)
        input("Double_valued: {0}".format(str(double_valued)))
        plt.close()
        
def pack_color_packet(color, blueband, redband, age, metallicity, Y, afe):
    '''Packs quantities needed to make a color packet.'''
    return (color, blueband, redband, age, metallicity, Y, afe)

def prompt_unknown_color(
    color_packet, DSEP_color, masses, kind="linear"):
    '''Indicate whether a color is single or double-valued.

    Has the user input where the color becomes double-valued. The function then
    saves it so that the location will be known for this particular color
    combination.
    '''
    prompt = "It's not known if this color is single or double-valued."
    print(prompt)
    plt.plot(masses, DSEP_color, 'r*')
    prompt = "Please indicate whether this color is single-valued [Y/N]: "
    single_valued = input(prompt)
    if single_valued.startswith("Y") or single_valued.startswith("y"):
        single_valued_colors.append(color_packet)
        with open(str(single_valued_colors_path), 'wb') as svf:
            pickle.dump(single_valued_colors, svf)
    elif single_valued.startswith("N") or single_valued.startswith("n"):
        prompt = ("Please input the highest-mass color where there is a "
            "turnoff: ")
        while True:
            stringvalue = input(prompt)
            if stringvalue == "" or stringvalue == "q" or stringvalue == "Q":
                break
            else:
                try:
                    value = float(stringvalue)
                except ValueError:
                    print("Entered value was not a valid number.")
            while True:
                prompt = "Is the peak a minimum or maximum? [min/max]"
                stringvalue = input(prompt)
            multi_valued_colors[color_packet].append(value)
            with open(str(multi_valued_colors_path), "wb") as mvf:
                pickle.dump(multi_valued_colors, mvf)
            prompt = ("If that's the last turnoff, just press return. If there "
                      " are more, then enter another: ")
    print("All done!")



def color_to_color_DSEP_interpolator(
    fromcolor, tocolor, DSEP_lookup, init_mass=0, mass_order="descending", 
    age=1.5, metallicity=0.0, Y=1, afe=2, bound_error=True, lowT=3000,
    redden_EBV=0.0):
    '''Return function to interpolate between colors in DSEP.

    Return a function which, given a color that can be calculated from the DSEP
    isochrone, will interpolate to another color. This interpolation uses a
    grid of the given age and metallicity.
    
    If the interpolator should silently mask the objects are outside with
    colors outside of the isochrone bounds, then bound_error should be set to
    False.'''
    blue, red = split_color(fromcolor)
    frompacket = pack_color_packet(
        fromcolor, DSEP_lookup[blue], DSEP_lookup[red], age, metallicity, Y,
        afe)
    if frompacket in single_valued_colors:
        fromblue, fromred = split_color(fromcolor)
        fromblue_DSEP = DSEP_band_converter(fromblue, DSEP_lookup[fromblue])
        fromred_DSEP = DSEP_band_converter(fromred, DSEP_lookup[fromred])

        toblue, tored = split_color(tocolor)
        toblue_DSEP = DSEP_band_converter(toblue, DSEP_lookup[toblue])
        tored_DSEP = DSEP_band_converter(tored, DSEP_lookup[tored])

        fromblue_col = restrict_interpolation_table(read_DSEP_isochrone(
            metallicity, age, bands=DSEP_lookup[fromblue], Y=Y,
            afe=afe), lowT=lowT)[fromblue_DSEP]
        fromred_col = restrict_interpolation_table(read_DSEP_isochrone(
            metallicity, age, bands=DSEP_lookup[fromred], Y=Y,
            afe=afe), lowT=lowT)[fromred_DSEP]
        toblue_col = restrict_interpolation_table(read_DSEP_isochrone(
            metallicity, age, bands=DSEP_lookup[toblue], Y=Y, 
            afe=afe), lowT=lowT)[toblue_DSEP]
        tored_col = restrict_interpolation_table(read_DSEP_isochrone(
            metallicity, age, bands=DSEP_lookup[tored], Y=Y, 
            afe=afe), lowT=lowT)[tored_DSEP]

        fromcolor_val = fromblue_col - fromred_col
        tocolor_val = toblue_col - tored_col

        reddened_fromcolor = redden_color(fromcolor, fromcolor_val, redden_EBV)
        reddened_tocolor = redden_color(tocolor, tocolor_val, redden_EBV)

        interpolator = interp1d(reddened_fromcolor, reddened_tocolor, 
                                kind="linear", bounds_error=bound_error)

        wrapped_interpolator = out_of_bounds_wrapper(
            interpolator, fromcolor, fromcolor_val[0], fromcolor_val[-1])

    elif frompacket in multi_valued_colors:
        mass_to_color = mass_to_color_DSEP_interpolator(
            tocolor, DSEP_lookup, age=age, metallicity=metallicity, Y=Y,
            afe=afe)
        if init_mass == 0:
            color_to_mass = color_to_mass_interpolator_mass_array(
                fromcolor, DSEP_lookup, mass_order=mass_order, age=age,
                metallicity=metallicity, Y=Y, afe=afe, bound_error=bound_error,
                lowT=lowT)
        else:
            color_to_mass = color_to_mass_interpolator_with_mass(
                fromcolor, DSEP_lookup, init_mass, age=age,
                metallicity=metallicity, Y=Y, afe=afe, bound_error=bound_error,
                lowT=lowT)
        # This is a bitch to debug
        wrapped_interpolator = (lambda x: mass_to_color(color_to_mass(x)))
        # Add these so that the endpoints of the interpolator can be known.
        wrapped_interpolator.x = color_to_mass.x
        wrapped_interpolator.y = mass_to_color(color_to_mass.y)
    else:
        raise ValueError("Don't know how to classify this color.")



    return wrapped_interpolator

def test_color_to_color_interpolator(
    ycolor, xcolor, DSEP_lookup, init_mass=0.0, mass_order="descending", 
    age=1.0, metallicity=0.0, Y=1, afe=2, lowT=3000):
    '''Test the performance of the interpolator compared to the actual
    isochrone.'''
    xblue, xred = split_color(xcolor)
    xblue_DSEP = DSEP_band_converter(xblue, DSEP_lookup[xblue])
    xred_DSEP = DSEP_band_converter(xred, DSEP_lookup[xred])

    yblue, yred = split_color(ycolor)
    yblue_DSEP = DSEP_band_converter(yblue, DSEP_lookup[yblue])
    yred_DSEP = DSEP_band_converter(yred, DSEP_lookup[yred])

    xblue_isochrone = read_DSEP_isochrone(
        metallicity, age, bands=DSEP_lookup[xblue], Y=Y,
        afe=afe)
    xred_isochrone = read_DSEP_isochrone(
        metallicity, age, bands=DSEP_lookup[xred], Y=Y,
        afe=afe)
    yblue_isochrone = read_DSEP_isochrone(
        metallicity, age, bands=DSEP_lookup[yblue], Y=Y, afe=afe)
    yred_isochrone = read_DSEP_isochrone(
        metallicity, age, bands=DSEP_lookup[yred], Y=Y, afe=afe)

    xblue_restricted = restrict_interpolation_table(xblue_isochrone, lowT=lowT)
    xred_restricted = restrict_interpolation_table(xred_isochrone, lowT=lowT)
    yblue_restricted = restrict_interpolation_table(yblue_isochrone, lowT=lowT)
    yred_restricted = restrict_interpolation_table(yred_isochrone, lowT=lowT)

    xblue_col = xblue_restricted[xblue_DSEP]
    xred_col = xred_restricted[xred_DSEP]
    yblue_col = yblue_restricted[yblue_DSEP]
    yred_col = yred_restricted[yred_DSEP]

    xcolor_DSEP = xblue_col - xred_col
    ycolor_DSEP = yblue_col - yred_col

    interpolator = color_to_color_DSEP_interpolator(
        xcolor, ycolor, DSEP_lookup, init_mass=init_mass,
        mass_order=mass_order, age=age, metallicity=metallicity, Y=Y, afe=afe,
        lowT=lowT)
    interpolated_xcolor = np.linspace(min(xcolor_DSEP), max(xcolor_DSEP), 1000)
    interpolated_ycolor = interpolator(interpolated_xcolor)
    
    plt.plot(xcolor_DSEP, ycolor_DSEP, 'r*')
    plt.plot(interpolated_xcolor, interpolated_ycolor, 'k-')

def convert_to_colors(
    fromcolor, tocolor, fromcolor_val, DSEP_lookup, age=1.0, metallicity=0, 
    Y=1, afe=2):
    '''Interpolate between colors using the DSEP isochrones.

    This is used to determine how the emission in one band is related to
    another using the DSEP isochrones.'''
    interpolator = color_to_color_DSEP_interpolator(
        fromcolor, tocolor, DSEP_lookup, age=age, metallicity=metallicity, Y=Y,
        afe=afe)
    interpolated_colors = interpolator(fromcolor_val)
    return interpolated_colors

def mass_to_band_DSEP_interpolator(
    band, age=1.5, metallicity=0.0, bands=1, Y=1, afe=2, lowT=3000,
    redden_EBV=0.0, D=10):
    '''Return function to interpolate a magnitude for a given mass.

    This function returns an interpolator to map mass and magnitude in the
    given band. 
    '''
    band = DSEP_band_converter(band, bands)
    try:
        interpolator = DSEP_interpolation(
            "M/Mo", band, age=age, metallicity=metallicity, bands=bands, Y=Y,
            afe=afe, lowT=lowT)
    except IndexError:
        interpolator = DSEP_interpolation(
            "M/Mo", band[0], age=age, metallicity=metallicity, bands=bands,
            Y=Y, afe=afe, lowT=lowT)

    # I hope weird bugs don't result from this.
    interpolator.y = redden_mag(band, interpolator.y, redden_EBV)
    interpolator.y = interpolator.y + 5 * np.log10(D/10)

    return interpolator

def mass_to_color_DSEP_interpolator(
    color, DSEP_lookup, age=1.5, metallicity=0.0, Y=1, afe=2,
    bound_error=True, lowT=3000, redden_EBV=0.0):
    '''Return function to interpolate color for a given mass.

    This function will return an interpolator which will map mass and color for
    the given color, for a star of the given age and metallicity.'''

    blue, red = split_color(color)
    blue_DSEP = DSEP_band_converter(blue, DSEP_lookup[blue])
    red_DSEP = DSEP_band_converter(red, DSEP_lookup[red])

    blue_table = restrict_interpolation_table(read_DSEP_isochrone(
        metallicity, age, bands=DSEP_lookup[blue], Y=Y,
        afe=afe), lowT=lowT)
    red_table = restrict_interpolation_table(read_DSEP_isochrone(
        metallicity, age, bands=DSEP_lookup[red], Y=Y,
        afe=afe), lowT=lowT)

    blue_masses = blue_table["M/Mo"]
    red_masses = red_table["M/Mo"]
    assert(np.all(blue_masses == red_masses))
    masses = blue_masses
    blue_col = blue_table[blue_DSEP]
    red_col = red_table[red_DSEP]
    color_val = blue_col - red_col

    reddened_color = redden_color(color, color_val, redden_EBV)

    interpolator = interp1d(masses, reddened_color, kind="linear",
                            bounds_error=bound_error)
    mass_to_color = out_of_bounds_wrapper(
        interpolator, color, max(reddened_color), min(reddened_color))

    return mass_to_color

def mag_to_color_DSEP_interpolator(
    mag, color, DSEP_lookup, age=1.5, metallicity=0.0, Y=1, afe=2,
    bound_error=True, lowT=3000, redden_EBV=0.0, D=10):
    '''Return a function interpolating from magnitudes to colors.

    This function will return an interpolator which will map the apparent
    magnitude of a star to its color.'''

    blue, red = split_color(color)
    blue_DSEP = DSEP_band_converter(blue, DSEP_lookup[blue])
    red_DSEP = DSEP_band_converter(red, DSEP_lookup[red])
    mag_DSEP = DSEP_band_converter(mag, DSEP_lookup[mag])

    blue_table = restrict_interpolation_table(read_DSEP_isochrone(
        metallicity, age, bands=DSEP_lookup[blue], Y=Y,
        afe=afe), lowT=lowT)
    red_table = restrict_interpolation_table(read_DSEP_isochrone(
        metallicity, age, bands=DSEP_lookup[red], Y=Y,
        afe=afe), lowT=lowT)
    mag_table = restrict_interpolation_table(read_DSEP_isochrone(
        metallicity, age, bands=DSEP_lookup[mag], Y=Y,
        afe=afe), lowT=lowT)

    blue_mags = blue_table["M/Mo"]
    red_mags = red_table["M/Mo"]
    assert(np.all(blue_mags == red_mags))
    mags = mag_table[mag_DSEP]
    appmags = mags + 5 * np.log10(D/10)
    print("Distance: " + str(D))
    reddened_mags = redden_mag(mag, appmags, redden_EBV)
    blue_col = blue_table[blue_DSEP]
    red_col = red_table[red_DSEP]
    color_val = blue_col - red_col
    reddened_color = redden_color(color, color_val, redden_EBV)

    interpolator = interp1d(reddened_mags, reddened_color, kind="linear",
                            bounds_error=bound_error)
    mags_to_color = out_of_bounds_wrapper(interpolator, color,
                                          max(color_val), min(color_val))

    return mags_to_color

def color_to_mag_DSEP_interpolator(
    fromcolor, tomag, DSEP_lookup, init_mass=0, mass_order="descending", 
    age=1.5, metallicity=0.0, Y=1, afe=2, bound_error=True, lowT=3000,
    redden_EBV=0.0, D=10):
    '''Return function to interpolate between colors in DSEP.

    Return a function which, given a color that can be calculated from the DSEP
    isochrone, will interpolate to another color. This interpolation uses a
    grid of the given age and metallicity.
    
    If the interpolator should silently mask the objects are outside with
    colors outside of the isochrone bounds, then bound_error should be set to
    False.'''
    blue, red = split_color(fromcolor)
    frompacket = pack_color_packet(
        fromcolor, DSEP_lookup[blue], DSEP_lookup[red], age, metallicity, Y,
        afe)
    if frompacket in single_valued_colors:
        fromblue, fromred = split_color(fromcolor)
        fromblue_DSEP = DSEP_band_converter(fromblue, DSEP_lookup[fromblue])
        fromred_DSEP = DSEP_band_converter(fromred, DSEP_lookup[fromred])

        tomag_DSEP = DSEP_band_converter(tomag, DSEP_lookup[tomag])

        fromblue_col = restrict_interpolation_table(read_DSEP_isochrone(
            metallicity, age, bands=DSEP_lookup[fromblue], Y=Y,
            afe=afe), lowT=lowT)[fromblue_DSEP]
        fromred_col = restrict_interpolation_table(read_DSEP_isochrone(
            metallicity, age, bands=DSEP_lookup[fromred], Y=Y,
            afe=afe), lowT=lowT)[fromred_DSEP]
        tomag_col = restrict_interpolation_table(read_DSEP_isochrone(
            metallicity, age, bands=DSEP_lookup[tomag], Y=Y, 
            afe=afe), lowT=lowT)[tomag_DSEP]

        # Convert the isochrone to apparent magnitudes.
        tomag_apparent = tomag_col + 5 * np.log10(D/10)

        fromcolor_val = fromblue_col - fromred_col
        reddened_color = redden_color(fromcolor, fromcolor_val, redden_EBV)
        reddened_mag = redden_mag(tomag, tomag_apparent, redden_EBV)

        interpolator = interp1d(reddened_color, reddened_mag, kind="linear",
                                bounds_error=bound_error)

        wrapped_interpolator = out_of_bounds_wrapper(
            interpolator, fromcolor, fromcolor_val[0], fromcolor_val[-1])

    elif frompacket in multi_valued_colors:
        mass_to_mag = mass_to_band_DSEP_interpolator(
            tomag, DSEP_lookup, age=age, metallicity=metallicity, Y=Y,
            afe=afe, redden_EBV=redden_EBV)
        if init_mass == 0:
            color_to_mass = color_to_mass_interpolator_mass_array(
                fromcolor, DSEP_lookup, mass_order=mass_order, age=age,
                metallicity=metallicity, Y=Y, afe=afe, bound_error=bound_error,
                lowT=lowT, redden_EBV=redden_EBV)
        else:
            color_to_mass = color_to_mass_interpolator_with_mass(
                fromcolor, DSEP_lookup, init_mass, age=age,
                metallicity=metallicity, Y=Y, afe=afe, bound_error=bound_error,
                lowT=lowT, redden_EBV=redden_EBV)
        # This is a bitch to debug
        wrapped_interpolator = (lambda x: mass_to_color(color_to_mass(x)))
        # Add these so that the endpoints of the interpolator can be known.
        wrapped_interpolator.x = color_to_mass.x
        wrapped_interpolator.y = mass_to_color(color_to_mass.y)
    else:
        raise ValueError("Don't know how to classify this color.")



    return wrapped_interpolator

# DSEP utility functions #

def DSEP_band_converter(band, bandno):
    '''Maps other representations of band to those outputted by DSEP.

    For example, if the Ks band is otherwise represented as KS, it would be
    converted to Ks.
    '''
    if bandno == 1:
        if band == "KS" or band == "K":
            band = "Ks"
    elif bandno == 10:
        if band == "i":
            print("Using i_new.")
            band = "i_new"
    elif bandno == 11:
        band = "sdss_" + band

    return band

def exponentify_interpolator(interp, base=10):
    '''Make function that raises the base to the result of the interpolation.

    If a function is best interpolated in log space, but it's preferable to
    have it output in linear space, this function will wrap the interpolator in
    a function that exponentifies it.
    '''
    return (lambda x: base**interp(x))

# Interpolate between intrinsic stellar values #

def mass_to_bolometric_luminosity_DSEP_interpolator(
    age=1.5, metallicity=0.0, bands=1, Y=1, afe=2, lowT=3000):
    '''Return function to interpolate bolometric luminosity for a given mass.

    This provides one of the important mappings between mass and bolometric
    luminosity using the DSEP isochrones. The interpolator depends on having a
    given age and metallicity.
    
    This interpolator interpolates log Luminosity, not luminosity itself.'''

    interpolator = DSEP_interpolation(
        "M/Mo", "LogL/Lo", age, metallicity, bands=bands, Y=Y, afe=afe,
        lowT=lowT)

    return exponentify_interpolator(interpolator)

def mass_to_teff_DSEP_interpolator(
    age=1.5, metallicity=0.0, bands=1, Y=1, afe=2, lowT=3000):
    '''Return function to interpolate effective temperature for a given mass.

    This provides one of the important mappings between mass and effective
    temperature using the DSEP isochrones. The interpolator depends on having a
    given age and metallicity.

    This interpolator interpolates log Teff, not Teff itself.'''

    interpolator = DSEP_interpolation(
        "M/Mo", "LogTeff", age, metallicity, bands=bands, Y=Y, afe=afe,
        lowT=lowT)

    return exponentify_interpolator(interpolator)

def teff_to_radius_DSEP_interpolator(
    age=1.5, metallicity=0.0, bands=1, Y=1, afe=2, lowT=3000, highT=6000,
    bound_error=True):
    '''Create an interpolator from effective temperature to radius.

    Unfortunately, radius is an inferred quantity and can't be inferred from
    columns. This function is supposed to help construct interpolators to do
    this.
    
    Note that this interpolator takes in logTeff and return logRadius'''
    star_table = restrict_interpolation_table(
        read_DSEP_isochrone(metallicity, age, bands=bands, Y=Y, afe=afe), 
        lowT=lowT, highT=highT)
    loglum = star_table["LogL/Lo"]
    logteff = star_table["LogTeff"]
    logradius= 0.5 * loglum - 2 * (logteff - np.log10(5777))

    interpolator = interp1d(logteff, logradius, kind="linear",
                            bounds_error=bound_error)
    teff_to_radius = out_of_bounds_wrapper(
        interpolator, "LogTeff", max(logteff), min(logteff))

    return teff_to_radius

def teff_to_logg_dwarf_DSEP_interpolator(
    age=1.5, metallicity=0.0, bands=1, Y=1, afe=2, lowT=3000, highT=6000,
    bound_error=True, minlogg=4.1):
    '''Create an interpolator from effective temperature to log(g).

    This maps Teff to logg for stars on the dwarf sequence. The dwarf sequence
    is defined as the sequence with log(g) > minlogg. As a result, the log(g)
    values aren't quite free. They also may be strange at high temperatures.

    Note that this interpolator takes in logTeff.
    '''
    interpolator = DSEP_interpolation(
        "LogTeff", "LogG", age, metallicity, bands=bands, Y=Y, afe=afe,
        lowT=lowT, highT=highT, bound_error=True, minlogg=minlogg)

    return interpolator
# Directly calculate fluxes and colors from mass #

def calculate_single_star_magnitude_DSEP(
    band, mass, metallicity, age=1.5, bands=1, Y=1, afe=2, redden_EBV=0.0):
    '''Calculate the magnitude of a star using DSEP.

    DSEP assumes the star is some fixed distance away, most likely, and this
    will return the magnitude that DSEP associates with the star. That will be
    useful in getting flux-related quantities such as color and flux ratios.
    '''
    mass_mag_interpolator = mass_to_band_DSEP_interpolator(
        band, age=age, metallicity=metallicity, bands=bands, Y=Y, afe=afe,
        redden_EBV=redden_EBV)
    star_mag = mass_mag_interpolator(mass)

    return star_mag

def calculate_binary_band_flux_ratio_DSEP(
    band, mass1, mass2, metallicity, age=1.5, bands=1, Y=1, afe=2):
    '''Calculate flux ratio between two stars in a band.

    An important quantity when adding colors is the in-band flux ratio. When
    using isochrones, the flux ratio can be calculated directly, and does not
    have to be calculated through bolometric corrections.
    '''
    mag1 = calculate_single_star_magnitude_DSEP(
        band, mass1, metallicity, age=age, bands=bands, Y=Y, afe=afe)
    mag2 = calculate_single_star_magnitude_DSEP(
        band, mass2, metallicity, age=age, bands=bands, Y=Y, afe=afe)

    return 10**(-0.4 * (mag1 - mag2))

def calculate_single_star_color_DSEP(
    color, mass, metallicity, DSEP_lookup, age=1.5, Y=1, afe=2, redden_EBV=0.0):
    '''Calculate the color of a star using only DSEP.

    The color will be calculated directly from DSEP using the mass-color
    relations in the isochrones.
    '''
    bluemag, redmag = split_color(color)
    bluemag = calculate_single_star_magnitude_DSEP(
        bluemag, mass, metallicity, age=age, bands=DSEP_lookup[bluemag], Y=Y, 
        afe=afe, redden_EBV=redden_EBV)
    redmag = calculate_single_star_magnitude_DSEP(
        redmag, mass, metallicity, age=age, bands=DSEP_lookup[redmag], Y=Y, 
        afe=afe, redden_EBV=redden_EBV)

    star_color = bluemag - redmag

    return star_color

def calculate_binary_star_color_DSEP(
    color, mass1, mass2, metallicity, DSEP_lookup, age=1.5, Y=1, afe=2,
    redden_EBV=0.0):
    '''Calculate the color of a binary star system using DSEP.

    The colors will be calculated directly from the DSEP isochrones.
    '''
    # Since the combined light from a binary is not additive (at least I
    # haven't shown that it isn't), it may be more straightforward to just
    # calculate an unreddened color and then redden the combined flux.
    color1 = calculate_single_star_color_DSEP(
        color, mass1, metallicity, age=age, DSEP_lookup=DSEP_lookup, Y=Y, 
        afe=afe, redden_EBV=0.0)
    color2 = calculate_single_star_color_DSEP(
        color, mass2, metallicity, age=age, DSEP_lookup=DSEP_lookup, 
        Y=Y, afe=afe, redden_EBV=0.0)

    blueband, redband = split_color(color)
    fluxratio = calculate_binary_band_flux_ratio_DSEP(
        blueband, mass1, mass2, metallicity, age=age,
        bands=DSEP_lookup[blueband], Y=Y, afe=afe)

    binary_color = sum_binary_color(color1, color2, fluxratio)
    reddened_binary = redden_color(color, binary_color, redden_EBV)
    return reddened_binary

def calculate_binary_star_mag_DSEP(
    mag, mass1, mass2, metallicity, DSEP_lookup, age=1.5, Y=1, afe=2,
    redden_EBV=0.0):
    '''Calculate the magnitude of a combined system.

    Magnitudes are calculated directly from the DSEP isochrones.
    '''
    mag1 = calculate_single_star_magnitude_DSEP(
        mag, mass1, metallicity, age=age, Y=Y, afe=afe, bands=DSEP_lookup[mag])
    mag2 = calculate_single_star_magnitude_DSEP(
        mag, mass2, metallicity, age=age, Y=Y, afe=afe, bands=DSEP_lookup[mag])

    binary_mag = sum_binary_mag(mag1, mag2)
    extincted_binary = redden_mag(mag, binary_mag, redden_EBV)

    return extincted_binary

def calculate_magnitude_difference_DSEP(
    band, mass1, mass2, metallicity, age=1.5, bands=1, Y=1, afe=2):
    '''Calculate magnitude difference between components with DSEP.

    In a binary with masses mass1 and mass2, return the difference of magnitude
    in the given band between the primary and the secondary. That is, return
    m_X,1 - m_X,2, where X is the band.'''
    fluxratio = calculate_binary_band_flux_ratio_DSEP(
        band, mass1, mass2, metallicity, age=age, bands=bands, Y=Y, afe=afe)
    return -2.5 * np.log10(fluxratio)

# Binary Loop Calculations #
############################

def color_mag_excess_chi2(
    color, mag, primary_mass, DSEP_lookup, photometric_errors={}, age=1.0,
    metallicity=0.0, Y=1, afe=2, numsecs=20, minsec=0.4, redden_EBV=0.0, 
    plot=False):
    '''Calculate a representative chi-squared figure above CMD for binaries.

    Using photometric errors, this function calculates a chi-squared statistic
    for each of the secondaries, and returns a the median chi-squared value
    from the isochrone.
    '''
    secondaries = np.linspace(primary_mass, minsec, numsecs)
    comb_color = calculate_binary_star_color_DSEP(
        color, primary_mass, secondaries, metallicity, DSEP_lookup, age=age)
    comb_mag = calculate_binary_star_color_DSEP(
        color2, primary_mass, secondaries, metallicity, DSEP_lookup, age=age, Y=Y, afe=afe)
    single_isochrone = color_to_mag_DSEP_interpolator(
        color, mag, DSEP_lookup, init_mass=primary_mass, age=age,
        metallicity=metallicity, Y=Y, afe=afe, redden_EBV=redden_EBV)
    blue, red = split_color(color)

    # Finish this.
    if photometric_errors:
        colorerr = sum_errors(photometric_errors[blue], photometric_errors[red])
        magerr = photometric_errors[mag]
        covar = calc_color_mag_covariance(color, mag, photometric_errors[mag])
    else:
        colorerr = 1
        magerr = 1
        covar = 0

    xlist = []
    for i in range(len(secondaries)):
        try:
            closest_x = isochrone_minimum_chi_squared(
                single_isochrone, comb_color, comb_mag, xerr=colorerr, 
                yerr=magerr2, cov=covar, npoints=1000)
        except OutOfBoundsError:
            print("Encountered an error?")
            print(single_isochrone.x)
            print(single_isochrone.y)
        xlist.append(closest_x)

    xintersections = np.array(xlist)
    yintersections = single_isochrone(xintersections)
    foms = chi_squared(
        yintersections, xintersections, comb_color, comb_mag, 
        colorerr, magerr, cov=covar)

    median_index =  np.argsort(foms)[len(foms)//2]

    if plot:
        xmed = xintersections[median_index]
        ymed = yintersections[median_index]
        plt.figure()
        color_mag_isochrone_plot(color, mag, DSEP_lookup, 1.0, age=age,
                                 redden_EBV=redden_EBV)
        plot_binary_loops(color2, color1, DSEP_lookup, [primary_mass], age=age,
                          minsec=0.3, numsec=100, redden_EBV=redden_EBV)
        plot_error_ellipse(comb_color[median_index], comb_mag[median_index],
                           colorerr, magerr, cov=covar)
        plt.plot([comb_color[median_index], xmed], [comb_mag[median_index],
                 ymed], 'b-')

    return foms[median_index]

def color_color_excess_chi2(
    color1, color2, primary_mass, DSEP_lookup, photometric_errors={}, age=1.0, 
    metallicity=0.0, Y=1, afe=2, numsecs=20, minsec=0.4, plot=False):
    '''Calculate a representative chi-squared displacement in color-color.

    Using photometric errors, this function calculates a chi-squared statistic
    for each of the secondaries, and returns the median chi-squared value from
    the isochrone.
    '''
    secondaries = np.linspace(primary_mass, minsec, numsecs)
    comb_color1 = calculate_binary_star_color_DSEP(
        color1, primary_mass, secondaries, metallicity, DSEP_lookup, age=age)
    comb_color2 = calculate_binary_star_color_DSEP(
        color2, primary_mass, secondaries, metallicity, DSEP_lookup, age=age, Y=Y, afe=afe)
    single_isochrone = color_to_color_DSEP_interpolator(
        color1, color2, DSEP_lookup, init_mass=primary_mass, age=age,
        metallicity=metallicity, Y=Y, afe=afe)
    blue1, red1 = split_color(color1)
    blue2, red2 = split_color(color2)

    if photometric_errors:
        colorerr1 = sum_errors(
            photometric_errors[blue1], photometric_errors[red1])
        colorerr2 = sum_errors(
            photometric_errors[blue2], photometric_errors[red2])
        covar = calc_color_color_covariance(
            color1, color2, photometric_errors[blue1],
            photometric_errors[red1])
    else:
        colorerr1 = 1
        colorerr2 = 1
        covar = 0


    xlist = []
    for i in range(len(secondaries)):
        try:
            closest_x = isochrone_minimum_chi_squared(
                single_isochrone, comb_color1[i], comb_color2[i], 
                xerr=colorerr1, yerr=colorerr2, cov=covar, npoints=1000)
        except OutOfBoundsError:
            print("Encountered an error?")
            print(single_isochrone.x)
            print(single_isochrone.y)
        xlist.append(closest_x)

    xintersections = np.array(xlist)
    yintersections = single_isochrone(xintersections)
    foms = chi_squared(
        yintersections, xintersections, comb_color2, comb_color1, 
        colorerr2, colorerr1, cov=covar)
    foms = np.diagonal(foms)

    median_index =  np.argsort(foms)[len(foms)//2]

    if plot:
        xmed = xintersections[median_index]
        ymed = yintersections[median_index]
        plt.figure()
        color_color_isochrone_plot(color2, color1, DSEP_lookup, 1.0, age=age)
        plot_binary_loops(color2, color1, DSEP_lookup, [primary_mass], age=age,
                          minsec=0.3, numsec=100)
        plot_error_ellipse(comb_color1[median_index], comb_color2[median_index],
                           colorerr1, colorerr2, cov=covar)
        plt.plot([comb_color1[median_index], xmed], [comb_color2[median_index],
                 ymed], 'b-')

    return foms[median_index]

def calc_color_color_covariance(
    color1, color2, blueerr, rederr):
    '''Calculate the Color-color covariance term.

    If it turns out that a band is chared between color1 and color2, it will
    return the covariance expected between the two colors.'''
    blue1, red1 = split_color(color1)
    blue2, red2 = split_color(color2)

    if blue1 == blue2:
        covar = blueerr**2
    elif blue1 == red2:
        covar = - blueerr**2
    elif red1 == blue2:
        covar = - rederr**2
    elif red1 == red2:
        covar = rederr**2
    else:
        covar = 0

    return covar

def calc_color_mag_covariance(
    color, mag, magerr):
    '''Calculate the covariance between a color and magnitude combination.'''
    blueband, redband = split_color(color)

    if mag == blueband:
        cov = magerr**2
    elif mag == redband:
        cov = -magerr**2
    else:
        cov = 0.0

    return cov


def chi_squared(modely, modelx, datay, datax, yerr, xerr, cov=0):
    '''Calculate the chi-squared value between two points.
    
    The model should be given as modely and modelx. The data points are datay
    and data x, along with the errors and covariance. If an array of data
    points are used, then the data, err, and covariance should be
    broadcastable. The model points do not necessarily have to be.'''
    try:
        modelx = modelx.reshape(len(modelx), 1)
        modely = modely.reshape(len(modely), 1)
    # Happens if models are not actually an array of values.
    except (TypeError, AttributeError):
        pass
    corr = cov / yerr / xerr
    xdiff = modelx - datax
    ydiff = modely - datay
    xfrac = xdiff / xerr
    yfrac = ydiff / yerr
    chisq = (xfrac**2 + yfrac**2 - 2 * corr * xfrac * yfrac) / (1 - corr**2)
    return chisq

def isochrone_minimum_chi_squared(isoc, xval, yval, xerr=1, yerr=1, cov=0,
                                  npoints=10000):
    '''Find a minimum chi-squared value for an isochrone.

    Picks the point on the isochrone which has the smallest chi-squared value
    for the point given. If the xerr and yerr values aren't given, then they
    will be assumed to be equally-weighted.
    
    Note that this function does not actually do an intelligent chi-squared
    minimization routine. It literally calculates chi-squared on a numpy array
    and returns the minimum index.
    
    This can accept multiple data points through xval and yval. In that case,
    it will return an array of x-values which minimize the chi-squared for
    those points.'''

    isox = np.linspace(isoc.x[0], isoc.x[-1], npoints)
    isoy = isoc(isox)

    chisq = chi_squared(isoy, isox, yval, xval, yerr, xerr, cov)
    minind = np.argmin(chisq, axis=0)
    return isox[minind]

def plot_minimized_isochrone_distance(
    isoc, ypoint, xpoint, yerr=1, xerr=1, cov=0):
    '''Test the isochrone minimization routine graphically.'''
    
    xmin = isochrone_minimum_chi_squared(
        isoc, xpoint, ypoint, xerr=xerr, yerr=yerr, cov=cov, npoints=10000)
    ymin = isoc(xmin)
    plt.plot([xpoint, xmin], [ypoint, ymin], 'b-')
    plot_isochrone(isoc)
    plot_error_ellipse(xpoint, ypoint, xerr, yerr, cov)
    print("Chi2: {0}".format(chi_squared(ypoint, xpoint, ymin, xmin, yerr,
                                         xerr, cov)))

def plot_color_color_minimized_isochrone_distance(
    ycolor, xcolor, DSEP_lookup, ypoint, xpoint, yerr, xerr, cov=0, age=4.0, 
    metallicity=0.0, Y=1, afe=2, lowT=3000, redden_EBV=0.0):
    '''Plot the minimized isochrone distance in a color-color plot.'''
    isoc = color_to_color_DSEP_interpolator(
        xcolor, ycolor, DSEP_lookup, age=age, metallicity=metallicity, Y=Y,
        afe=afe, lowT=lowT, redden_EBV=redden_EBV)
    plot_minimized_isochrone_distance(isoc, ypoint, xpoint, yerr, xerr, cov)

def plot_color_mag_minimized_isochrone_distance(
    color, mag, DSEP_lookup, colorpoint, magpoint, colorerr, magerr, cov=0, 
    age=4.0, metallicity=0.0, Y=1, afe=2, lowT=3000, redden_EBV=0.0, D=10):
    '''Plot the minimized isochrone distance in a color-magnitude plot. 

    Note that the magnitude should be an absolute magnitude.'''
    isoc = color_to_mag_DSEP_interpolator(
        color, mag, DSEP_lookup, age=age, metallicity=metallicity, Y=Y,
        afe=afe, lowT=lowT, redden_EBV=redden_EBV, D=D)
    plot_minimized_isochrone_distance(
        isoc, magpoint, colorpoint, magerr, colorerr, cov)
    ylim = plt.ylim()
    if ylim[0] < ylim[1]:
        plt.ylim(plt.ylim()[::-1])
    plt.ylabel(mag)
    plt.xlabel(color)

def iterate_colors(bands):
    '''Create all color combinations of bands.

    This function makes all color combinations of bands exactly once. If bands
    is sorted according to increasing wavelength, then the bluer band will 
    always be on the left, and the redder band will always be on the right, as
    is generally expected of colors.'''
    color_iterator = map(
        lambda p: "{0}-{1}".format(p[0], p[1]), 
        itertools.combinations(bands, 2))
    return color_iterator

def explore_color_combination_fom(
    bands, primary_mass, DSEP_lookup, photometric_errors, age=1.0, 
    metallicity=0.0, Y=1, afe=2, numsecs=20, minsec=0.62):
    '''Calculate figures of merit for color combinations.

    Iterate through the combinations of bands and calculate a figure of merit
    for each color combination.'''
    ycolorlist = []
    xcolorlist = []
    fomlist = []

    for xcolor, ycolor in itertools.combinations(iterate_colors(bands), 2):
        point_chi2 = color_color_excess_chi2(
            ycolor, xcolor, primary_mass, DSEP_lookup,
            photometric_errors, age=age, metallicity=metallicity, Y=Y, 
            afe=afe, numsecs=numsecs, minsec=minsec)
        print("{0} vs {1}: {2}".format(ycolor, xcolor, point_chi2))
        sigmas = chi_squared_to_sigma(point_chi2)
        ycolorlist.append(ycolor)
        xcolorlist.append(xcolor)
        fomlist.append(sigmas)

    fomtable = Table([ycolorlist, xcolorlist, fomlist], names=(
        "ycolor", "xcolor", "sigma"))
    fomtable.sort("sigma")
    fomtable.reverse()
    print(fomtable)

    return fomtable

def chi_squared_to_sigma(chi2_val, dof=2):
    '''Translates a chi-squared value to a sigma probability interval.
    '''
    prob = chi2.sf(chi2_val, dof)
    sigmas = norm.isf(prob/2)
    return sigmas

def sum_errors(err1, err2):
    '''Add errors in quadrature.'''
    return np.sqrt(err1**2 + err2**2)

def rotate_points(x, y, angle):
    '''Rotate points x and y through the given angle counterclockwise.
    
    Returns the rotated coordinates as a 2-tuple.'''
    xnew = x * np.cos(angle) - y * np.sin(angle)
    ynew = x * np.sin(angle) + y * np.cos(angle)
    return (xnew, ynew)

def plot_error_ellipse(x, y, xerr, yerr, cov=0):
    '''Plot error ellipses around a point with given errors and covariance.

    This function will plot three ellipses corresponding to 1, 2, and 3 sigma.
    The ellipses will go from darkest to lightest.
    '''
    # These are needed for indexing to work out.
    x = np.array(x)
    y = np.array(y)
    xerr = np.array(xerr)
    yerr = np.array(yerr)
    cov = np.array(cov)

    sigma1 = "#4dac26"
    sigma2 = "#b8e186"
    sigma3 = "#f7f7f7"
    sigma4 = "#f1b6da"
    sigma5 = "#d01c8b"
    colors = [sigma1, sigma2, sigma3, sigma4, sigma5] 

    sigma_scales = np.array([1.52, 2.48, 3.44, 4.40, 5.36])
    sigma_scales = sigma_scales.reshape((len(sigma_scales), 1))

    semimajor = sigma_scales * np.sqrt(((xerr**2 + yerr**2) / 2 + np.sqrt(
        (xerr**2 - yerr**2)**2 / 4 + cov**2)))
    semiminor = sigma_scales * np.sqrt(((xerr**2 + yerr**2) / 2 - np.sqrt(
        (xerr**2 - yerr**2)**2 / 4 + cov**2)))
    eccentricity = np.sqrt(1 - semiminor**2 / semimajor**2)


    # Note that the thetas start at the semimajor axis, and then go around.
    thetas = np.linspace(0, 2*np.pi, 1000)
    thetas = thetas.reshape(len(thetas), 1, 1)
    # This should now be a 1000x5xN array, where N is the number of data
    # points.
    radii = semimajor * (1 - eccentricity**2) / (1 + eccentricity *
                                                 np.cos(thetas))
    horizontal_xvals = semimajor * eccentricity + radii * np.cos(thetas)
    horizontal_yvals = radii * np.sin(thetas)

    # If the y-axis is the semimajor axis, then rotate by 90 degrees.
    vertical_xvals, vertical_yvals = rotate_points(
        horizontal_xvals, horizontal_yvals, np.pi/2)
    unrotated_xvals = np.where(yerr > xerr, vertical_xvals, horizontal_xvals)
    unrotated_yvals = np.where(yerr > xerr, vertical_yvals, horizontal_yvals)

    # 0/0 yields a nan, so I want to prevent that.
    rotation = np.nan_to_num(np.arctan(
        2 * cov / (xerr**2 - yerr**2)))/2
    rotated_xvals, rotated_yvals = rotate_points(
        unrotated_xvals, unrotated_yvals, rotation)
    xvals = x + rotated_xvals
    yvals = y + rotated_yvals

    for i in range(len(x)):
        for j in range(len(sigma_scales)):
            plt.plot(xvals[:,j,i], yvals[:,j,i], color=colors[j])

# Variate #
###########

def scattered_point(xcen, ycen, xerr, yerr, cov=0, npoints=1):
    '''Returns points representative of correlated error.

    Given a central point, this function returns an array of length npoints
    distributed as a gaussian with correlated errors.
    
    This returns an npointsx2 array. So to get all of the x values, do
    out[:,0], and out[:,1] for the y values.''' 
    # 2 
    means = np.array([xcen, ycen])
    # 2 x 2 
    covmat = [[xerr**2, cov], [cov, yerr**2]]

    values = multivariate_normal.rvs(means, covmat, npoints)
    # npoints x 2
    return values

def scatter_along_isochrone(
    ycolor, xcolor, masses, DSEP_lookup, photometric_errors={}, npoints=1,
    metallicity=0.0, age=4.0, Y=1, afe=2, lowT=4000, redden_EBV=0.0):
    '''Scatter isochrone points at masses with photometric error.

    Takes an array of masses, and returns a 2xlen(masses)xnpoints array which
    contains the scattered points along the isochrone. The first dimension
    indicates x-values or y-values.'''
    xblue, xred = split_color(xcolor)
    yblue, yred = split_color(ycolor)

    xerr = sum_errors(photometric_errors[xblue], photometric_errors[xred])
    yerr = sum_errors(photometric_errors[yblue], photometric_errors[yred])
    cov = calc_color_color_covariance(
        ycolor, xcolor, photometric_errors[xblue], photometric_errors[xred])
    mass_xcolor_isochrone = mass_to_color_DSEP_interpolator(
        xcolor, DSEP_lookup, metallicity=metallicity, age=age, Y=Y, afe=afe,
        lowT=lowT, redden_EBV=redden_EBV)
    # Since the scatter function can't take an array of masses, we'll just do
    # things according to this for loop.
    totallist = []
    for m in masses:
        xcolor_ycolor_isochrone = color_to_color_DSEP_interpolator(
            xcolor, ycolor, DSEP_lookup, init_mass=m, metallicity=metallicity, 
            age=age, Y=Y, afe=afe, lowT=lowT, redden_EBV=redden_EBV)
        xcolor_val = mass_xcolor_isochrone(m)
        ycolor_val = xcolor_ycolor_isochrone(xcolor_val)
        # Returns npoints x 2
        pointarray = scattered_point(
            xcolor_val, ycolor_val, xerr, yerr, cov, npoints)
        totallist.append(pointarray)
    
    # This will be len(masses) x npoints x 2
    masses_array = np.array(totallist)

    return masses_array

def plot_scattered_isochrone_points(
    ycolor, xcolor, DSEP_lookup, photometric_errors={}, npoints=1,
    metallicity=0.0, age=4.0, Y=1, afe=2, lowT=4000, redden_EBV=0.0):
    '''Plot an isochrone with modelled data scattered about it.

    Creates synthetic data based on the errors and plots it over the
    isochrone.'''
    massrange = np.linspace(1.0, 0.65, 400)
    scattered_data = scatter_along_isochrone(
        ycolor, xcolor, massrange, DSEP_lookup,
        photometric_errors=photometric_errors, npoints=npoints,
        metallicity=metallicity, age=age, Y=Y, afe=afe, lowT=lowT,
        redden_EBV=redden_EBV)

    # This stuff is to plot. It may need to be moved to a different place.
    xvalues = np.ravel(masses_array[:,:,0])
    yvalues = np.ravel(masses_array[:,:,1])

    plot_isochrone(xcolor_ycolor_isochrone)
    plt.plot(xvalues, yvalues, 'k.')
    plt.xlabel(xcolor)
    plt.ylabel(ycolor)

# Isochrone subtraction #
#########################

def subtract_isochrone_from_data(isochrone, ydata, xdata):
    '''Subtract the isochrone from the ydata.

    Evaluate the isochrone at xdata and then subtract that from ydata. Return
    the differences. Remember that this operation is not symmetric with x and y
    data.
    '''
    iso_y = isochrone(xdata)
    difference = ydata - iso_y
    return difference

##################################
# Forward-modeling Uncertainties #
##################################

def plot_modeled_color_color_differences(
    color1, color2, DSEP_lookup, photometric_errors={}, metallicity=0.0, 
    age=4.0, Y=1, afe=2, redden_EBV=0.0, lowT=4000, npoints=10):
    '''Plot differences between isochrone and simulated data for two colors.

    Makes two plots which show the difference between modeled data points given
    correlated errors and the isochrones. Maybe these should be Gaussian?'''
    isomasses = np.linspace(1.0, 0.61, 400)
    modelled_data = scatter_along_isochrone(
        color2, color1, isomasses, DSEP_lookup,
        photometric_errors=photometric_errors, metallicity=metallicity,
        age=age, Y=Y, afe=afe, redden_EBV=redden_EBV, lowT=lowT,
        npoints=npoints)
    xvalues = np.ravel(modelled_data[:,:,0])
    yvalues = np.ravel(modelled_data[:,:,1])

    iso1to2 = color_to_color_DSEP_interpolator(
        color1, color2, DSEP_lookup, metallicity=metallicity, age=age, Y=Y,
        afe=afe, redden_EBV=redden_EBV, lowT=lowT)
    iso2to1 = color_to_color_DSEP_interpolator(
        color2, color1, DSEP_lookup, metallicity=metallicity, age=age, Y=Y,
        afe=afe, redden_EBV=redden_EBV, lowT=lowT)

    ydiffs = subtract_isochrone_from_data(iso1to2, yvalues, xvalues)
    xdiffs = subtract_isochrone_from_data(iso2to1, xvalues, yvalues)

    plt.figure()
    plt.plot(xvalues, ydiffs, 'k.')
    plt.xlabel(color1)
    plt.ylabel(color2 + " Diff")
    plt.figure()
    plt.plot(yvalues, xdiffs, 'k.')
    plt.xlabel(color2)
    plt.ylabel(color1 + " Diff")



###############################################################################
# Casagrande-DSEP Stellar Parameter Routines #
###############################################################################

def calculate_binary_star_color_Casagrande_DSEP(
    color, mass1, mass2, metallicity, age=1.5, bolcolor="B-V", bands=1, Y=1,
    afe=2):
    '''Calculates the given color of a binary star system.

    The colors will be calculated using the empirical Casagrande relations
    between Teff and Color. The relationship between mass and Teff will be
    taken from the DSEP isochrones. Bolometric corrections will be taken from
    Casagrande as well. And the mass-luminosity relation will be taken from
    DSEP.

    The Bolometric corrections will be calculated using the single-star color
    in the color given in bolcolor. For example, 
    '''
    color1 = calculate_single_star_color_Casagrande_DSEP(
        color, mass1, metallicity, age=age, bands=bands, Y=Y, afe=afe)
    color2 = calculate_single_star_color_Casagrande_DSEP(
        color, mass2, metallicity, age=age, bands=bands, Y=Y, afe=afe)

    bluecolor, redcolor = split_color(color)

    fluxratio = calculate_binary_band_flux_ratio_Casagrande_DSEP(
        bluecolor, mass1, mass2, metallicity, bolcolor, age=age, bands=bands,
        Y=Y, afe=afe)

    bin_color = sum_binary_color(color1, color2, fluxratio)

    return bin_color

def calculate_single_star_color_Casagrande_DSEP(
    color, mass, metallicity, age=1.5, bands=1, Y=1, afe=2):
    '''Calculate the color of a star using Casagrande and DSEP.

    The color will be calculated using the empirical Casagrande relations
    between Teff and Color. The relationship between mass and Teff will be
    taken from the DSEP isochrones.
    '''
    mass_teff_interpolator = mass_to_teff_DSEP_interpolator(age=age,
        metallicity=metallicity, bands=bands, Y=Y, afe=afe)
    star_teff = mass_teff_interpolator(mass)

    color = Casagrande_inverted_color(color, star_teff, metallicity)

    return color

def calculate_binary_band_flux_ratio_Casagrande_DSEP(
    band, mass1, mass2, metallicity, bolcolor, age=1.5, bands=1, Y=1, afe=2):
    '''Calculate the flux ratio between two stars using Casagrande and DSEP.

    This function calculates the flux ratio by essentially running a
    mass-bolometric luminosity relation from DSEP, and then correcting the
    bolometric luminosity to an in-band luminosity from the Bolometric
    Corrections in Casagrande et al (2010).
    '''
    bol_color1 = calculate_single_star_color_Casagrande_DSEP(
        bolcolor, mass1, metallicity, age=age, bands=bands, Y=Y, afe=afe)
    bol_color2 = calculate_single_star_color_Casagrande_DSEP(
        bolcolor, mass2, metallicity, age=age, bands=bands, Y=Y, afe=afe)

    bolratio = (Casagrande_Bolometric_Flux(
        band, 0, bolcolor, bol_color2, metallicity) / 
                Casagrande_Bolometric_Flux(
        band, 0, bolcolor, bol_color1, metallicity))

    mass_lum_interpolator = mass_to_bolometric_luminosity_DSEP_interpolator(
        age, metallicity, bands=bands, Y=Y, afe=afe)
    lumratio = mass_lum_interpolator(mass1) / mass_lum_interpolator(mass2)

    fluxratio = lumratio * bolratio

    return fluxratio

def calculate_magnitude_difference_Casagrande_DSEP(
    band, mass1, mass2, metallicity, bolcolor, age=1.5, bands=1, Y=1, afe=2):
    '''Calculate magnitude difference for components with Casagrande and DSEP.

    Given a binary with mass1 and mass2, this will calculate the magnitude
    difference between the two in the given band. This will return the
    magnitude difference m_X,1 - m_X,2 where X is the band.
    '''
    fluxratio = calculate_binary_band_flux_ratio_Casagrande_DSEP(
        band, mass1, mass2, metallicity, bolcolor, age=age, bands=bands, Y=Y,
        afe=afe)
    return -2.5 * np.log10(fluxratio)


###############################################################################
# Plotting Routines
###############################################################################

def single_color_excess_plot(
    color, primary_mass, secondary_masses, metallicity, age=8, bolcolor="B-V",
    method="Casagrande-DSEP", magdiff_band="V", linestyle="solid",
    linecolor="black", DSEP_lookup={}, Y=1, afe=2):
    '''Plots the color excess as a function of secondary mass.

    This function only plots a single color excess as a function of secondary
    mass.'''
    
    colors=[]
    magdiffs=[]
    for smass in secondary_masses:
        try:
            if method == "Casagrande-DSEP":
                colorval = calculate_binary_star_color_Casagrande_DSEP(
                    color, primary_mass, smass, metallicity, age=age,
                    bolcolor=bolcolor)[0]
                magdiff = calculate_magnitude_difference_Casagrande_DSEP(
                    magdiff_band, primary_mass, smass, metallicity, bolcolor, 
                    age=age)[0]
            elif method == "DSEP":
                colorval = calculate_binary_star_color_DSEP(
                    color, primary_mass, smass, metallicity, DSEP_lookup, 
                    age=age, Y=Y, afe=afe)
                magdiff = calculate_magnitude_difference_DSEP(
                    magdiff_band, primary_mass, smass, metallicity, 
                    age=age, bands=DSEP_lookup[magdiff_band], Y=Y, afe=afe)
            else:
                raise ValueError("Don't recognize method {0}".format(method))
        except OutOfBoundsError:
            # This occurs when a value is out of bounds.
            break
        colors.append(colorval)
        magdiffs.append(magdiff)
    binary_colors = np.array(colors)
    binary_magdiffs = np.array(magdiffs)
    primary_masses = secondary_masses[:len(binary_colors)]

    if method == "Casagrande-DSEP":
        primary_color = calculate_single_star_color_Casagrande_DSEP(
            color, primary_mass, metallicity, age=age)
    elif method == "DSEP":
        primary_color = calculate_single_star_color_DSEP(
            color, primary_mass, metallicity, age=age, 
            DSEP_lookup=DSEP_lookup, Y=Y, afe=afe)
    color_excess = binary_colors - primary_color
    mag_excess = -2.5 * np.log10(1 + 10**(+0.4 * binary_magdiffs))

    plt.plot(primary_masses, color_excess, label=color, ls=linestyle,
             c=linecolor)
    plt.xlabel("{0}-band Magnitude Excess".format(magdiff_band))
    plt.ylabel("Color Excess over primary")

# Instead of colors, I might want to pass an object which creates the
# relationship between the colors and pands for DSEP models.
# Also *bands* is something that's a DSEP internal. So it should be a very
# minor change.

def color_excess_plot_comparison(
    colors, primary_mass, secondary_masses, metallicity, age=8, bolcolor="B-V",
    magdiff_band="V", linestyle="solid", linecolors=[], 
    method="Casagrande-DSEP", DSEP_bands=1, Y=1, afe=2):
    '''Plots multiple color excesses.

    The color excesses for all of the given colors in the colors list will be
    calculated and plotted on the same scale. That way, the most prominent
    color for differentiating secondaries from a given primary can be
    chosen.'''

    for i, color in enumerate(colors):
        if linecolors is []:
            linecolor=None
        else:
            linecolor = linecolors[i]
        single_color_excess_plot(
            color, primary_mass, secondary_masses, metallicity, age=age, 
            bolcolor=bolcolor, magdiff_band=magdiff_band, linestyle=linestyle,
            linecolor=linecolor, method=method, DSEP_lookup=DSEP_bands, Y=Y, 
            afe=afe)

def color_color_isochrone_plot(
    ycolor, xcolor, DSEP_lookup, lowT=3000, age=1.0, metallicity=0.0, Y=1, 
    afe=2, redden_EBV=0.0, init_mass=0):
    '''Plots an isochrone line in a color-color space.

    Plot the DSEP isochrone projected in the ycolor vs xcolor space. It ranges
    from high_mass to low_mass.'''
    interpolator = color_to_color_DSEP_interpolator(
        xcolor, ycolor, DSEP_lookup, age=age, metallicity=metallicity, Y=Y,
        afe=afe, redden_EBV=redden_EBV, lowT=lowT, init_mass=init_mass)

    plot_isochrone(interpolator)

def color_mag_isochrone_plot(
    color, mag, DSEP_lookup, lowT=3000, age=1.0, metallicity=0.0, Y=1, afe=2, 
    redden_EBV=0.0, D=10, fmt="r-"):
    '''Plot the isochrone line as a colormagnitude diagram.

    Plot the DSEP isochrone projected in the color-magnitude space specified.
    It will range from high_mass to low_mass. Reddening will be applied
    according to the E(B-V) given in redden_EBV. This function will also adjust
    the apparent magnitude by the appropriate distance modulus where D is given
    in parsecs (under the assumption that the DSEP isochrones are in absolute
    magnitudes).
    '''

    interpolator = color_to_mag_DSEP_interpolator(
        color, mag, DSEP_lookup, age=age, metallicity=metallicity, Y=Y,
        afe=afe, redden_EBV=redden_EBV, D=D, lowT=lowT)

    plot_isochrone(interpolator)

    ylimits = plt.ylim()
    if ylimits[0] < ylimits[1]:
        plt.ylim(ylimits[::-1])

def simulated_color_color_diagram(
    ycolor, xcolor, ycolorerr, xcolorerr, DSEP_lookup, uppermass=2, 
    lowermass=0.12, binary_frac=0.5):
    '''Create a color-color diagram of a simulated DSEP cluster with binaries.

    This function will make a color-color plot between two masses to show what
    the data should look like. As of now, this is NOT a good population
    synthesis; it is only useful for exploring the spread in data.
    '''
    single_masses = np.linspace(lowermass, uppermass, 500)

    y_singles = calculate_single_star_color_DSEP(
        ycolor, single_masses, 0.0, DSEP_lookup, age=1.0)
    x_singles = calculate_single_star_color_DSEP(
        xcolor, single_masses, 0.0, DSEP_lookup, age=1.0)

    y_single_simdata = y_singles + np.random.normal(scale=ycolorerr,
                                                    size=y_singles.size)
    x_single_simdata = x_singles + np.random.normal(scale=xcolorerr,
                                                    size=x_singles.size)

    plt.plot(x_single_simdata, y_single_simdata, 'b.')

    if binary_frac > 0:
        primary_masses = np.linspace(
            lowermass, uppermass, 
            int(single_masses.size*(1-binary_frac)/binary_frac))
        secondary_masses = (np.random.uniform(size=primary_masses.size) *
                            primary_masses)
        # This is to avoid making the secondaries out of bounds.
        secondary_masses[secondary_masses<lowermass] = lowermass
        
        y_binaries = calculate_binary_star_color_DSEP(
            ycolor, single_masses, secondary_masses, 0.0, DSEP_lookup, 
            age=1.0)
        x_binaries = calculate_binary_star_color_DSEP(
            xcolor, single_masses, secondary_masses, 0.0, DSEP_lookup, 
            age=1.0)
        y_binary_simdata = y_binaries + np.random.normal(scale=ycolorerr,
                                                       size=y_binaries.size)
        x_binary_simdata = x_binaries + np.random.normal(scale=xcolorerr,
                                                       size=x_binaries.size)
        plt.plot(x_binary_simdata, y_binary_simdata, 'm.')

def plot_binary_loops(
    ycolor, xcolor, DSEP_lookup, primary_masses, metallicity=0.0, age=1.0,
    minsec=0.62, numsec=20, redden_EBV=0.0):
    for prim in primary_masses:
        secondaries = np.linspace(prim, minsec, numsec)
        comb_ycolor = calculate_binary_star_color_DSEP(
            ycolor, prim, secondaries, metallicity, DSEP_lookup, age=age,
            redden_EBV=redden_EBV)
        comb_xcolor = calculate_binary_star_color_DSEP(
            xcolor, prim, secondaries, metallicity, DSEP_lookup, age=age,
            redden_EBV=redden_EBV)
        plt.plot(comb_xcolor, comb_ycolor, 'r-', linewidth=2)

def plot_binary_loop_excess(
    ycolor, xcolor, DSEP_lookup, primary_mass, metallicity=0.0, age=1.0, Y=1,
    afe=2, color="red", label="", minsec=0.62, redden_EBV=0.0):
    '''Plot the excess of the binary loop above the DSEP isochrone.

    Take the look for a binary and subtract the isochrone from it according to
    xcolor. This should show how displaced from the isochrone the binaries
    should be.
    '''
    secondaries = np.linspace(primary_mass, minsec, 50)
    comb_ycolor = calculate_binary_star_color_DSEP(
        ycolor, primary_mass, secondaries, metallicity, DSEP_lookup, age=age,
        redden_EBV=redden_EBV)
    comb_xcolor = calculate_binary_star_color_DSEP(
        xcolor, primary_mass, secondaries, metallicity, DSEP_lookup, age=age,
        redden_EBV=redden_EBV)
    ycolor_excess = subtract_DSEP_isochrone(
        ycolor, xcolor, comb_ycolor, comb_xcolor, DSEP_lookup, age=age,
        metallicity=metallicity, Y=Y, afe=afe, redden_EBV=redden_EBV)
    plt.plot(comb_xcolor, ycolor_excess, ls="-", c=color)
    plt.xlabel(xcolor)
    plt.ylabel(ycolor + " Excess")
    plt.title("Color excess for {0:.1f} Msun star".format(primary_mass))


def plot_mass_markers(
    ycolor, xcolor, DSEP_lookup, masses, metallicity=0.0, age=1.0,
    enlarge_first=True):
    '''Plot where stars of given masses lie in color-color space.

    Plots stars in the location of the stars with given mass in ycolor vs
    xcolor. If the enlarge_first keyword is given, then the star for the first
    mass will be enlarged.
    '''
    test_masses = np.array(masses)

    ycolor_interp = mass_to_color_DSEP_interpolator(
        ycolor, DSEP_lookup, age=age)
    xcolor_interp = mass_to_color_DSEP_interpolator(
        xcolor, DSEP_lookup, age=age)

    test_ycolor = ycolor_interp(test_masses)
    test_xcolor = xcolor_interp(test_masses)
    if enlarge_first:
        plt.plot(test_xcolor[0], test_ycolor[0], 'r*', ms=20)
        plt.plot(test_xcolor[1:], test_ycolor[1:], 'r*', ms=10)
    else:
        plt.plot(test_xcolor, test_ycolor, 'r*', ms=10)

def plot_mass_color(color, DSEP_lookup, metallicity=0.0, age=1.0):
    '''Plot the color vs. mass relation for the given color.

    This function is to verify whether a color is double-valued or not.
    '''
    # USe test_multi_valued_colors for now. Though the code there can probably
    # be transferred here.
    pass

def plot_isochrone(isochrone, npoints=1000, label=""):
    '''Take an isochrone of some kind and plot it internally.

    This function essentially takes the bounds contained within the isochrone
    and plots them in order to check that they are well-behaved.
    '''
    testpoints = np.linspace(isochrone.x[0], isochrone.x[-1], 1000)
    testvalues = isochrone(testpoints)
    plt.plot(testpoints, testvalues, 'r-', linewidth=3, label=label)

def test_isochrone(isochrone, npoints=1000):
    '''Plot actual isochrone points on top of interpolations.

    In cases where there is a double-valued isochrone, this would mean that the
    isochrone in practice would not be smooth as expected from the sequence of
    points. This should be obvious to the human eye. This function helps
    illustrate that.
    '''
    plot_isochrone(isochrone, npoints=npoints)
    plt.plot(isochrone.x, isochrone.y, 'bo')


####################
# Standalone plots #
####################
def Casagrande_Colors_plot_excesses():
    '''Make a plot with the different color excesses in Casagrande.'''
    colors=["B-V", "V-J", "V-H", "V-KS", "J-KS"]
    DSEP_lookup = {"B": 1, "V": 1, "J": 1, "H": 1, "KS": 1}
    linecolors = ["blue", "purple", "pink", "orange", "red"]
    primary_mass = 1.0
    secondary_masses = np.linspace(primary_mass, 0.1, 30)
    metallicity = 0.0
    age=6

    color_excess_plot_comparison(
        colors, primary_mass, secondary_masses, metallicity, age=age,
        bolcolor="V-KS", magdiff_band="V", linecolors=linecolors,
        DSEP_bands=DSEP_lookup)
    plt.legend(loc="upper left")
    color_excess_plot_comparison(
        colors, primary_mass, secondary_masses, metallicity, age=age,
        bolcolor="V-KS", magdiff_band="V", linestyle="dashed", method="DSEP",
        linecolors=linecolors, DSEP_bands=DSEP_lookup)
    plt.show()
 
def Johnson_2MASS_plot_excesses():
    '''Make a plot with color excess for Johnson and 2MASS using DSEP.'''
    colors=["B-V", "V-J", "V-H", "V-KS", "J-H", "H-KS"]
    DSEP_lookup = {"B": 1, "V": 1, "J": 1, "H": 1, "KS": 1}
    linecolors = ["blue", "purple", "green", "pink", "orange", "red"]
    primary_mass = 1.0
    secondary_masses = np.linspace(primary_mass, 0.1, 30)
    metallicity = 0.0
    age=6

    color_excess_plot_comparison(
        colors, primary_mass, secondary_masses, metallicity, age=age,
        magdiff_band="V", method="DSEP",
        linecolors=linecolors, DSEP_bands=DSEP_lookup)
    plt.show()
    plt.legend()
 

def Bouy_Colors_plot_excesses():
    '''Make a plot with the different color excesses in Bouy.'''
    colors = ["i-K", "g-J", "g-H", "g-K", "g-i", "i-z", "g-r", "J-H", "H-K",
              "r-i"]
    DSEP_lookup = {"u": 11, "g": 11, "r": 11, "i": 11, "z": 11, "Y": 8, "J": 1,
                   "H": 1, "K": 1}
    linecolors = ["black", "brown", "gray", "blue", "purple", "green", "pink", 
                  "yellow", "orange", "red"]
    primary_mass = 1.0
    secondary_masses = np.linspace(primary_mass, 0.1, 30)
    metallicity = 0.0
    age=6

    color_excess_plot_comparison(
        colors, primary_mass, secondary_masses, metallicity, age=age,
        magdiff_band="i", linecolors=linecolors, method="DSEP",
        DSEP_bands=DSEP_lookup)
    plt.legend(loc="upper left")
    plt.show()

def plot_DSEP_excesses():
    '''Plots to show sensitivity in most sensitive bands.'''
    DSEP_lookup = {'B': 1, 'H': 1, 'J': 1, 'K': 1, 'Kp': 1, 'U': 1, 'V': 1, 
                   'Y': 8, 'g': 11, 'i': 11, 'r': 11, 'u': 11, 'z': 11}
    band_combinations_20msun = [
        ("B-H", "H-K"), ("g-H", "H-K"), ("B-J", "H-K"), ("V-H", "H-K"), 
        ("g-J", "H-K")]
    band_combinations_15msun = [
        ("B-H", "H-K"), ("g-H", "H-K"), ("B-J", "H-K"), ("V-H", "H-K"), 
        ("g-J", "H-K")]
    band_combinations_10msun = [
        ("B-K", "Kp-r"), ("g-K", "Kp-r"), ("B-J", "Kp-r"), ("V-K", "Kp-r"), 
        ("g-J", "Kp-r")]
    band_combinations_05msun = [
        ("g-K", "J-H"), ("B-K", "J-H"), ("g-H", "J-H"), ("B-J", "J-H"),
        ("V-K", "J-H")]

    colors = ["black", "blue", "green", "orange", "red"]

    for i, lst in enumerate([
            band_combinations_20msun, band_combinations_15msun, 
            band_combinations_10msun, band_combinations_05msun]):
        for combo in lst:
            plt.figure()
            plot_binary_loop_excess(combo[0], combo[1], DSEP_lookup, 2.0)
            plt.ylabel(combo[0] + " Excess")
            plt.xlabel(combo[1])
            plt.title("{0:.1f} Msun Excess".format(2.0-0.5*i))
            plt.tight_layout()

# This plot should be broken up. But I wanna make it fast.
def triple_hr_comparison_plot(
    optical_mag, optical_color, nir_mag, nir_color, DSEP_lookup,
    ages=[1.0, 2.0, 5.0, 10.0], metallicity=0.0, Y=1, afe=2, lowT=3000):
    for age in ages:
        plt.subplot(131)
        teff_lum = DSEP_interpolation(
            "LogTeff", "LogL/Lo", age=age, metallicity=metallicity, Y=Y,
            afe=afe, lowT=lowT)
        plot_isochrone(teff_lum, label="{0:.2g} Gyr".format(age))
        hr.invert_x_axis()
        plt.subplot(132)
        color_mag_optical = color_to_mag_DSEP_interpolator(
            optical_color, optical_mag, DSEP_lookup, age=age,
            metallicity=metallicity, Y=Y, afe=afe, lowT=lowT)
        plot_isochrone(color_mag_optical, label="{0:.2g} Gyr".format(age))
        hr.invert_y_axis()
        plt.subplot(133)
        color_mag_nir = color_to_mag_DSEP_interpolator(
            nir_color, nir_mag, DSEP_lookup, age=age,
            metallicity=metallicity, Y=Y, afe=afe, lowT=lowT)
        plot_isochrone(color_mag_nir, label="{0:.2g} Gyr".format(age))
        hr.invert_y_axis()

###############################################################################
# Reddening Routines #
###############################################################################

reddening_coeffs = {"B": 1.337, "V": 1.000, "I": 0.479, "J": 0.282, "H": 0.190,
                    "K": 0.114, "Ks": 0.114, "Kp": 0.9}

# These are coefficients that are given by the IRSA dust map service.
reddening_coeffs = {"B": 1.337, "V": 1.000, "I": 0.479, "J": 0.282, "H": 0.190,
                    "K": 0.114, "Ks": 0.114, "Kp": 0.9}

def redden_mag(band, truemag, EB_V, Rv=3.1):
    '''Redden the true magnitude value given E(B-V).
    
    Uses calculated values from CCM to calculate the reddened magnitude given
    an E(B-V) value.
    '''
    extinction = Rv * reddening_coeffs[band] * EB_V
    extinctedmag = truemag + extinction

    return extinctedmag

def deredden_mag(band, extinctedmag, EB_V, Rv=3.1):
    '''Deredden an observed magnitude given E(B-V).

    Uses calculated values from CCM to obtain the dereddened magnitude given an
    E(B-V) value.
    '''
    extinction = redden_mag(band, 0, EB_V, Rv=Rv)
    truemag = extinctedmag - extinction

    return truemag

def redden_color(color, truecolor, EB_V, Rv=3.1):
    '''Redden the true color given E(B-V).

    Use calculated values from CCM to calculate the reddened color given an
    E(B-V) value.
    '''
    blueband, redband = split_color(color)
    blue_extinction = redden_mag(blueband, 0, EB_V, Rv=Rv)
    red_extinction = redden_mag(redband, 0, EB_V, Rv=Rv)
    reddened_color = truecolor + (blue_extinction - red_extinction)

    return reddened_color

def deredden_color(color, extincted_color, EB_V, Rv=3.1):
    '''Deredden the observed colro given E(B-V).

    Use calculated values from CCM to deredden a cover given an E(B-V) value.
    '''
    reddening_coeff = redden_color(color, 0, EB_V, Rv=Rv)
    dereddened_color = extincted_color - reddening_coeff

    return dereddened_color
###############################################################################
# Miscellaneous Routines
###############################################################################

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

def sum_binary_color(color1, color2, fluxratio):
    '''Calculate summed color from components.

    The three main ingredients needed to calculate the summed color are the
    colors of the individual components, and the flux ratio (in the blue band).
    The flux ratio should also have component 1 in the numerator, and component
    2 in the denominator. Isochrones generally provide the flux ratio fairly 
    straightforwardly, but bolometric corrections may be needed if empirical 
    relations are used.'''
    summed_color = color2 - 2.5 * np.log10(
        (1 + fluxratio) / (1 + fluxratio * 10**(-0.4*(color2 - color1))))
    return summed_color

def sum_binary_mag(mag1, mag2):
    '''Calculate summed magnitude from components.

    This function takes two magnitude values in a given band and adds them in
    order to make the combined magnitude in that band.'''
    summed_mag = mag1 - 2.5 * np.log10(1 + 10**(-0.4 * (mag2 - mag1)))
    return summed_mag
    
if __name__ == "__main__":

    Bouy_Colors_plot_excesses()
