import os
import glob
import subprocess
import tempfile
import shutil
import itertools

import numpy as np
import numpy.core.defchararray as npstr
from scipy.interpolate import interp1d
from astropy.table import Table
from pathlib import Path
import matplotlib.pyplot as plt

import path_config as paths

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

def dsep_isochrone_interpolator(
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

def dsep_age_splitter(inputfile, outputdir,
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

def assign_dsep_sign(val):
    '''Returns p if val is positive and n if val is negative.

    If val is zero, then it will return p anyway.
    '''
    return sign_switch(val, "p", "m", 1)

def format_dsep_isochrone_filename(feh, afe, Y, bands):
    '''Creates a filename which follows the dsep format.

    This format is feh(p|m)??afe(p|m)?[y??].{bands}. Where the two digits after
    feh are the metallicity, with p for positive and m for negative
    metallicity. After that is the alpha-abundance, which follows the same
    pattern. If the helium abundance is set and not metallicity-dependent, then
    there will be the extra y term in the filename.

    The bands is basically a suffix which contains every band that is contained
    in the isochrone. For example, one isochrone contains UBVRIJHKsKp.'''
    afe_val = 0.2 * (afe - 2)
    feh_sign = assign_dsep_sign(feh)
    afe_sign = assign_dsep_sign(afe_val)

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

def format_dsep_age_isochrone_filename(age, feh, afe, y, bands):
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
    return age_prefix + format_dsep_isochrone_filename(feh, afe, y, bands)

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
        isochrone_output = tempdir / format_dsep_isochrone_filename(
            feh, afe, Y, bands)
        dsep_isochrone_interpolator(feh, isochrone_output, bands, Y, 
                                    afe, interp_exec, isochrones)
        dsep_age_splitter(isochrone_output, outputdir,
                          executable=split_exec)

def read_dsep_age_table(tablepath):
    '''Reads the post-split dsep table.

    The table should be one which has been split from the monolithic isochrone
    file, and thus should contain only one age.
    '''
    age_table = Table.read(str(tablepath), format="ascii.commented_header",
                           header_start=-1)
    return age_table

def read_dsep_isochrone(
    feh, age, bands=1, Y=1, afe=2, tabledir=paths.DSEP_OUTPUT,
    isochrones=paths.DSEP_ISOCHRONES,
    interp_exec=paths.DSEP_INTERPOLATOR_EXECUTABLE,
    split_exec=paths.DSEP_SPLITTER_EXECUTABLE):
    '''Read in a DSEP isochrone at a given metallicity and age.

    The individual isochrone tables need to be found in tabledir. If the given
    table is not found, then the isochrone will be generated automatically from
    the grid if possible.
    '''
    tablepath = (tabledir / format_dsep_age_isochrone_filename(
        age, feh, afe, Y, bands))
    try:
        age_table = read_dsep_age_table(tablepath)
    except FileNotFoundError:
        interpolated_split_isochrone(
            feh, outputdir=tabledir, bands=bands, Y=Y, afe=afe, 
            isochrones=isochrones, interp_exec=interp_exec, 
            split_exec=split_exec)
        age_table = read_dsep_age_table(tablepath)

    return age_table

# DSEP Interpolation Routines #
###############################

def dsep_interpolation(fromcol, tocol, age=1.5, metallicity=0.0, bands=1, Y=1,
                       afe=2):
    '''Return an interpolator between two DSEP isochrone quantities.

    This will return a function which, given a value of fromcol which is
    covered by the DSEP isochrone, will interpolate a value of tocol. This
    interpolation uses a grid of the given age and metallicity.
    '''
    isochrone = read_dsep_isochrone(metallicity, age, bands=bands, Y=Y, afe=afe)

    interpolator = interp1d(isochrone[fromcol], isochrone[tocol], kind="linear")

    def exception_wrapper(x):
        try:
            return interpolator(x)
        except ValueError:
            raise OutOfBoundsError

    return exception_wrapper

def color_to_color_DSEP_interpolator(
    fromcolor, tocolor, DSEP_lookup, age=1.5, metallicity=0.0, Y=1, afe=2):
    '''Return function to interpolate between colors in DSEP.

    Return a function which, given a color that can be calculated from the DSEP
    isochrone, will interpolate to another color. This interpolation uses a
    grid of the given age and metallicity.'''
    fromblue, fromred = split_color(fromcolor)
    fromblue_DSEP = DSEP_band_converter(fromblue, DSEP_lookup[fromblue])
    fromred_DSEP = DSEP_band_converter(fromred, DSEP_lookup[fromred])

    toblue, tored = split_color(tocolor)
    toblue_DSEP = DSEP_band_converter(toblue, DSEP_lookup[toblue])
    tored_DSEP = DSEP_band_converter(tored, DSEP_lookup[tored])

    fromblue_col = read_dsep_isochrone(
        metallicity, age, bands=DSEP_lookup[fromblue], Y=Y,
        afe=afe)[fromblue_DSEP]
    fromred_col = read_dsep_isochrone(
        metallicity, age, bands=DSEP_lookup[fromred], Y=Y,
        afe=afe)[fromred_DSEP]
    toblue_col = read_dsep_isochrone(
        metallicity, age, bands=DSEP_lookup[toblue], Y=Y, afe=afe)[toblue_DSEP]
    tored_col = read_dsep_isochrone(
        metallicity, age, bands=DSEP_lookup[tored], Y=Y, afe=afe)[tored_DSEP]

    fromcolor_val = fromblue_col - fromred_col
    tocolor_val = toblue_col - tored_col

    interpolator = interp1d(fromcolor_val, tocolor_val, kind="linear")

    def exception_wrapper(x):
        try:
            return interpolator(x)
        except ValueError:
            raise OutOfBoundsError

    return exception_wrapper

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

def mass_to_bolometric_luminosity_dsep_interpolator(
    age=1.5, metallicity=0.0, bands=1, Y=1, afe=2):
    '''Return function to interpolate bolometric luminosity for a given mass.

    This provides one of the important mappings between mass and bolometric
    luminosity using the DSEP isochrones. The interpolator depends on having a
    given age and metallicity.
    
    This interpolator interpolates log Luminosity, not luminosity itself.'''

    interpolator = dsep_interpolation(
        "M/Mo", "LogL/Lo", age, metallicity, bands=bands, Y=Y, afe=afe)

    return exponentify_interpolator(interpolator)

def mass_to_teff_dsep_interpolator(
    age=1.5, metallicity=0.0, bands=1, Y=1, afe=2):
    '''Return function to interpolate effective temperature for a given mass.

    This provides one of the important mappings between mass and effective
    temperature using the DSEP isochrones. The interpolator depends on having a
    given age and metallicity.

    This interpolator interpolates log Teff, not Teff itself.'''

    interpolator = dsep_interpolation(
        "M/Mo", "LogTeff", age, metallicity, bands=bands, Y=Y, afe=afe)

    return exponentify_interpolator(interpolator)

def mass_to_band_dsep_interpolator(
    band, age=1.5, metallicity=0.0, bands=1, Y=1, afe=2):
    '''Return function to interpolate a magnitude for a given mass.

    This function returns an interpolator to map mass and magnitude in the
    given band. 
    '''
    band = DSEP_band_converter(band, bands)
    try:
        interpolator = dsep_interpolation(
            "M/Mo", band, age=age, metallicity=metallicity, bands=bands, Y=Y,
            afe=afe)
    except IndexError:
        interpolator = dsep_interpolation(
            "M/Mo", band[0], age=age, metallicity=metallicity, bands=bands,
            Y=Y, afe=afe)

    return interpolator

def create_color_interpolator(blue_interp, red_interp):
    '''Create a color interpolator from two band interpolators.

    The most straightfoward way to interpolate colors from an isochrone is to
    generate a function that uses the interpolators from each band, and
    subtracts them.'''
    return (lambda x: blue_interp(x) - red_interp(x))


def mass_to_color_dsep_interpolator(
    color, DSEP_lookup, age=1.5, metallicity=0.0, Y=1, afe=2):
    '''Return function to interpolate color for a given mass.

    This function will return an interpolator which will map mass and color for
    the given color, for a star of the given age and metallicity.'''

    band1, band2 = split_color(color)
    band1_interpolator = mass_to_band_dsep_interpolator(
        band1, age=age, metallicity=metallicity, bands=DSEP_lookup[band1], Y=Y, 
        afe=afe)
    band2_interpolator = mass_to_band_dsep_interpolator(
        band2, age=age, metallicity=metallicity, bands=DSEP_lookup[band2], Y=Y, 
        afe=afe)

    return create_color_interpolator(band1_interpolator, band2_interpolator)

def color_to_mass_dsep_interpolator(
    color, DSEP_lookup, age=1.5, metallicity=0.0, Y=1, afe=2):
    '''Return function to interpolate mass given a color.

    This function will return an interpolator which will map colors to masses.
    Unfortunately, this relation is generally double-valued. There needs to be
    a good way to deal with this.
    '''

# Stellar properties with DSEP only #
#####################################


def calculate_single_star_magnitude_DSEP(
    band, mass, metallicity, age=1.5, bands=1, Y=1, afe=2):
    '''Calculate the magnitude of a star using DSEP.

    DSEP assumes the star is some fixed distance away, most likely, and this
    will return the magnitude that DSEP associates with the star. That will be
    useful in getting flux-related quantities such as color and flux ratios.
    '''
    mass_mag_interpolator = mass_to_band_dsep_interpolator(
        band, age=age, metallicity=metallicity, bands=bands, Y=Y, afe=afe)
    star_mag = mass_mag_interpolator(mass)

    return star_mag

def calculate_binary_band_flux_ratio_DSEP(
    band, mass1, mass2, metallicity, age=1.5, bands=1, Y=1, afe=2):
    '''Calculate flux ratio between two stars in a band.

    An important quantity when adding colors is the in-band flux ratio. When
    using isochrones, the flux ratio can be calculated directly, and does not
    have to be calculated through bolometric corrections.
    '''
    band1 = calculate_single_star_magnitude_DSEP(
        band, mass1, metallicity, age=age, bands=bands, Y=Y, afe=afe)
    band2 = calculate_single_star_magnitude_DSEP(
        band, mass2, metallicity, age=age, bands=bands, Y=Y, afe=afe)

    return 10**(-0.4 * (band1 - band2))

def calculate_magnitude_difference_DSEP(
    band, mass1, mass2, metallicity, age=1.5, bands=1, Y=1, afe=2):
    '''Calculate magnitude difference between components with DSEP.

    In a binary with masses mass1 and mass2, return the difference of magnitude
    in the given band between the primary and the secondary. That is, return
    m_X,1 - m_X,2, where X is the band.'''
    fluxratio = calculate_binary_band_flux_ratio_DSEP(
        band, mass1, mass2, metallicity, age=age, bands=bands, Y=Y, afe=afe)
    return -2.5 * np.log10(fluxratio)

def calculate_single_star_color_DSEP(
    color, mass, metallicity, DSEP_lookup, age=1.5, Y=1, afe=2):
    '''Calculate the color of a star using only DSEP.

    The color will be calculated directly from DSEP using the mass-color
    relations in the isochrones.
    '''
    bluemag, redmag = split_color(color)
    bluemag = calculate_single_star_magnitude_DSEP(
        bluemag, mass, metallicity, age=age, bands=DSEP_lookup[bluemag], Y=Y, 
        afe=afe)
    redmag = calculate_single_star_magnitude_DSEP(
        redmag, mass, metallicity, age=age, bands=DSEP_lookup[redmag], Y=Y, 
        afe=afe)

    return bluemag - redmag

def calculate_binary_star_color_DSEP(
    color, mass1, mass2, metallicity, DSEP_lookup, age=1.5, Y=1, afe=2):
    '''Calculate the color of a binary star system using DSEP.

    The colors will be calculated directly from the DSEP isochrones.
    '''
    color1 = calculate_single_star_color_DSEP(
        color, mass1, metallicity, age=age, DSEP_lookup=DSEP_lookup, Y=Y, 
        afe=afe)
    color2 = calculate_single_star_color_DSEP(
        color, mass2, metallicity, age=age, DSEP_lookup=DSEP_lookup, 
        Y=Y, afe=afe)

    blueband, redband = split_color(color)
    fluxratio = calculate_binary_band_flux_ratio_DSEP(
        blueband, mass1, mass2, metallicity, age=age,
        bands=DSEP_lookup[blueband], Y=Y, afe=afe)

    binary_color = sum_binary_color(color1, color2, fluxratio)
    return binary_color

# Binary Loop Calculations #
############################

def color_color_excess_rms(
    ycolor, xcolor, primary_mass, DSEP_lookup, age=1.0, metallicity=0.0, Y=1,
    afe=2, numsecs=20):
    '''Calculate the RMS of the binary excess above the DSEP isochrone.

    This function calculates the typical excursion of the binary above the DSEP
    isochrone using the RMS excess as the figure of merit.
    '''
    secondaries = np.linspace(primary_mass, 0.12, numsecs)
    comb_ycolor = calculate_binary_star_color_DSEP(
        ycolor, primary_mass, secondaries, metallicity, DSEP_lookup, age=age)
    comb_xcolor = calculate_binary_star_color_DSEP(
        xcolor, primary_mass, secondaries, metallicity, DSEP_lookup, age=age)
    ycolor_excess = subtract_DSEP_isochrone(
        ycolor, xcolor, comb_ycolor, comb_xcolor, DSEP_lookup, age=age,
        metallicity=metallicity, Y=Y, afe=afe)
    print(ycolor_excess)
    fom = np.sqrt(np.mean(ycolor_excess**2))
    print(fom)
    return fom

def explore_color_combination_fom(
    bands, primary_mass, DSEP_lookup, age=1.0, metallicity=0.0, Y=1, afe=2,
    numsecs=20):
    '''Calculate figures of merit for color combinations.

    Iterate through the combinations of bands and calculate a figure of merit
    for each color combination.'''
    ycolorlist = []
    xcolorlist = []
    fomlist = []

    for ytuple in itertools.combinations(bands, 2):
        for xtuple in itertools.combinations(bands, 2):
            ycolor = (ytuple[0] + "-" + ytuple[1])
            xcolor = (xtuple[0] + "-" + xtuple[1])
            if ycolor != xcolor:
                fom = color_color_excess_rms(
                    ycolor, xcolor, primary_mass, DSEP_lookup, age=age,
                    metallicity=metallicity, Y=Y, afe=afe, numsecs=numsecs)
                ycolorlist.append(ycolor)
                xcolorlist.append(xcolor)
                fomlist.append(fom)

    fomtable = Table([ycolorlist, xcolorlist, fomlist], names=(
        "ycolor", "xcolor", "RMS"))
    fomtable.sort("RMS")
    fomtable.reverse()
    print(fomtable)

    return fomtable



# DSEP Tools #
##############

def subtract_DSEP_isochrone(
    color, inputcolor, vals, inputvals, DSEP_lookup, age=1.0, metallicity=0.0, 
    Y=1, afe=2):
    '''Subtract a DSEP isochrone from vals using inputvals.

    Isochrones interpolated from inputvals will be subtracted from vals and
    returned. The actual colors which vals and inputvals correspond to should 
    be indicated in color and inputcolor.
    '''
    interp = color_to_color_DSEP_interpolator(
        inputcolor, color, DSEP_lookup, age=age, metallicity=metallicity, Y=Y,
        afe=afe)
    isochrone_values = interp(inputvals)
    subtracted_values = vals - isochrone_values
    return subtracted_values


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
    mass_teff_interpolator = mass_to_teff_dsep_interpolator(age=age,
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

    mass_lum_interpolator = mass_to_bolometric_luminosity_dsep_interpolator(
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
    ycolor, xcolor, DSEP_lookup, high_mass, low_mass=0.12, age=1.0):
    '''Plots an isochrone line in a color-color space.

    Plot the DSEP isochrone projected in the ycolor vs xcolor space. It ranges
    from high_mass to low_mass.'''
    masses = np.linspace(high_mass, low_mass)

    ycolor_interp = mass_to_color_dsep_interpolator(
        ycolor, DSEP_lookup, age=age)
    xcolor_interp = mass_to_color_dsep_interpolator(
        xcolor, DSEP_lookup, age=age)

    ycolor_vals = ycolor_interp(masses)
    xcolor_vals = xcolor_interp(masses)

    plt.plot(xcolor_vals, ycolor_vals, 'r-')

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
    ycolor, xcolor, DSEP_lookup, primary_masses, metallicity=0.0, age=1.0):
    for prim in primary_masses:
        secondaries = np.linspace(prim, 0.12, 20)
        comb_ycolor = calculate_binary_star_color_DSEP(
            ycolor, prim, secondaries, metallicity, DSEP_lookup, age=age)
        comb_xcolor = calculate_binary_star_color_DSEP(
            xcolor, prim, secondaries, metallicity, DSEP_lookup, age=age)
        plt.plot(comb_xcolor, comb_ycolor, 'r-')

def plot_binary_loop_excess(
    ycolor, xcolor, DSEP_lookup, primary_mass, metallicity=0.0, age=1.0, Y=1,
    afe=2, color="red", label=""):
    '''Plot the excess of the binary loop above the DSEP isochrone.

    Take the look for a binary and subtract the isochrone from it according to
    xcolor. This should show how displaced from the isochrone the binaries
    should be.
    '''
    secondaries = np.linspace(primary_mass, 0.12, 50)
    comb_ycolor = calculate_binary_star_color_DSEP(
        ycolor, primary_mass, secondaries, metallicity, DSEP_lookup, age=age)
    comb_xcolor = calculate_binary_star_color_DSEP(
        xcolor, primary_mass, secondaries, metallicity, DSEP_lookup, age=age)
    ycolor_excess = subtract_DSEP_isochrone(
        ycolor, xcolor, comb_ycolor, comb_xcolor, DSEP_lookup, age=age,
        metallicity=metallicity, Y=Y, afe=afe)
    print(ycolor_excess)
    plt.plot(comb_xcolor, ycolor_excess, ls="-", c=color)


def plot_mass_markers(
    ycolor, xcolor, DSEP_lookup, masses, metallicity=0.0, age=1.0,
    enlarge_first=True):
    '''Plot where stars of given masses lie in color-color space.

    Plots stars in the location of the stars with given mass in ycolor vs
    xcolor. If the enlarge_first keyword is given, then the star for the first
    mass will be enlarged.
    '''
    test_masses = np.array(masses)

    ycolor_interp = mass_to_color_dsep_interpolator(
        ycolor, DSEP_lookup, age=age)
    xcolor_interp = mass_to_color_dsep_interpolator(
        xcolor, DSEP_lookup, age=age)

    test_ycolor = ycolor_interp(test_masses)
    test_xcolor = xcolor_interp(test_masses)
    if enlarge_first:
        plt.plot(test_xcolor[0], test_ycolor[0], 'r*', ms=20)
        plt.plot(test_xcolor[1:], test_ycolor[1:], 'r*', ms=10)
    else:
        plt.plot(test_xcolor, test_ycolor, 'r*', ms=10)

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
    
if __name__ == "__main__":

    Bouy_Colors_plot_excesses()
