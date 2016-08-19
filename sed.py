import os
import glob
import subprocess
import tempfile
import shutil

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

def split_color(colorcol):
    '''Splits an array of string colors into two arrays of bands.

    This takes an array of strings such as "RC-J" or "(B-V)T" and splits them
    into two arrays which just have the bands, such as "RC" & "J" and "BT" &
    "VT". This can handle colors which need to be distributed.
    '''
    distributed_colors = distribute_color_subscript(colorcol)
    bluecolor = np.array(distributed_colors)
    redcolor = np.array(distributed_colors)
    splitcolors = npstr.split(distributed_colors, "-")

    for i,lst in enumerate(splitcolors):
        bluecolor[i] = lst[0]
        redcolor[i] = lst[1]
    
    return bluecolor, redcolor

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
    
    if bands == 1:
        suffix = "UBVRIJHKsKp"
    elif 1 < bands <= 15:
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

def dsep_interpolation(fromcol, tocol, age=1.5, metallicity=0.0):
    '''Return an interpolator between two DSEP isochrone quantities.

    This will return a function which, given a value of fromcol which is
    covered by the DSEP isochrone, will interpolate a value of tocol. This
    interpolation uses a grid of the given age and metallicity.
    '''
    isochrone = read_dsep_isochrone(metallicity, age)

    interpolator = interp1d(isochrone[fromcol], isochrone[tocol], kind="linear")

    return interpolator

def exponentify_interpolator(interp, base=10):
    '''Make function that raises the base to the result of the interpolation.

    If a function is best interpolated in log space, but it's preferable to
    have it output in linear space, this function will wrap the interpolator in
    a function that exponentifies it.
    '''
    return (lambda x: base**interp(x))

def mass_to_bolometric_luminosity_dsep_interpolator(age=1.5, metallicity=0.0):
    '''Return function to interpolate bolometric luminosity for a given mass.

    This provides one of the important mappings between mass and bolometric
    luminosity using the DSEP isochrones. The interpolator depends on having a
    given age and metallicity.
    
    This interpolator interpolates log Luminosity, not luminosity itself.'''

    interpolator = dsep_interpolation("M/Mo", "LogL/Lo", age, metallicity)

    return exponentify_interpolator(interpolator)

def mass_to_teff_dsep_interpolator(age=1.5, metallicity=0.0):
    '''Return function to interpolate effective temperature for a given mass.

    This provides one of the important mappings between mass and effective
    temperature using the DSEP isochrones. The interpolator depends on having a
    given age and metallicity.

    This interpolator interpolates log Teff, not Teff itself.'''

    interpolator = dsep_interpolation("M/Mo", "LogTeff", age, metallicity)

    return exponentify_interpolator(interpolator)

def mass_to_band_dsep_interpolator(band, age=1.5, metallicity=0.0):
    '''Return function to interpolate a magnitude for a given mass.

    This function returns an interpolator to map mass and magnitude in the
    given band. 
    '''
    interpolator = dsep_interpolation("M/Mo", band, age=age,
                                      metallicity=metallicity)

    return interpolator

def create_color_interpolator(blue_interp, red_interp):
    '''Create a color interpolator from two band interpolators.

    The most straightfoward way to interpolate colors from an isochrone is to
    generate a function that uses the interpolators from each band, and
    subtracts them.'''
    return (lambda x: blue_interp(x) - red_interp(x))

def mass_to_color_dsep_interpolator(color, age=1.5, metallicity=0.0):
    '''Return function to interpolate color for a given mass.

    This function will return an interpolator which will map mass and color for
    the given color, for a star of the given age and metallicity.'''

    band1, band2 = split_color(color)
    band1_interpolator = mass_to_band_dsep_interpolator(
        band1, age=age, metallicity=metallicity)
    band2_interpolator = mass_to_band_dsep_interpolator(
        band2, age=age, metallicity=metallicity)

    return create_color_interpolator(band1_interpolator, band2_interpolator)

# Stellar properties with DSEP only #
#####################################


def calculate_single_star_magnitude_DSEP(
    band, mass, metallicity, age=1.5):
    '''Calculate the magnitude of a star using DSEP.

    DSEP assumes the star is some fixed distance away, most likely, and this
    will return the magnitude that DSEP associates with the star. That will be
    useful in getting flux-related quantities such as color and flux ratios.
    '''
    mass_mag_interpolator = mass_to_band_dsep_interpolator(
        band, age=age, metallicity=metallicity)
    star_mag = mass_mag_interpolator(mass)

    return star_mag

def calculate_binary_band_flux_ratio_DSEP(
    band, mass1, mass2, metallicity, age=1.5):
    '''Calculate flux ratio between two stars in a band.

    An important quantity when adding colors is the in-band flux ratio. When
    using isochrones, the flux ratio can be calculated directly, and does not
    have to be calculated through bolometric corrections.
    '''
    band1 = calculate_single_star_color_DSEP(band, mass1, metallicity, age=age)
    band2 = calculate_single_star_color_DSEP(band, mass2, metallicity, age=age)

    return 10**(-0.4 * (band1 - band2))


def calculate_single_star_color_DSEP(
    color, mass, metallicity, age=1.5):
    '''Calculate the color of a star using only DSEP.

    The color will be calculated directly from DSEP using the mass-color
    relations in the isochrones.
    '''
    bluemag, redmag = split_color(color)
    bluemag = calculate_single_star_magnitude_DSEP(
        bluemag, mass, metallicity, age=age)
    redmag = calculate_single_star_magnitude_DSEP(
        redmag, mass, metallicity, age=age)

    return bluemag - redmag

def calculate_binary_star_color_DSEP(
    color, mass1, mass2, metallicity, age=1.5, bolcolor="B-V"):
    '''Calculate the color of a binary star system using DSEP.

    The colors will be calculated directly from the DSEP isochrones.
    '''
    color1 = calculate_single_star_color_DSEP(
        color, mass1, metallicity, age=age)
    color2 = calculate_single_star_color_DSEP(
        color, mass2, metallicity, age=age)

    blueband, redband = split_color(color)
    fluxratio = calculate_binary_band_flux_ratio_DSEP(
        blueband, mass1, mass2, metallicity, age=age)

    binary_color = sum_binary_color(color1, color2, fluxratio)
    return binary_color

###############################################################################
# Casagrande-DSEP Stellar Parameter Routines #
###############################################################################

def calculate_binary_star_color_Casagrande_DSEP(
    color, mass1, mass2, metallicity, age=1.5, bolcolor="B-V"):
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
        color, mass1, metallicity, age=age)
    color2 = calculate_single_star_color_Casagrande_DSEP(
        color, mass2, metallicity, age=age)

    bluecolor, redcolor = split_color(color)

    fluxratio = calculate_binary_band_flux_ratio_Casagrande_DSEP(
        bluecolor, mass1, mass2, metallicity, bolcolor, age=age)

    bin_color = sum_binary_color(color1, color2, fluxratio)

    return bin_color

def calculate_single_star_color_Casagrande_DSEP(
    color, mass, metallicity, age=1.5):
    '''Calculate the color of a star using Casagrande and DSEP.

    The color will be calculated using the empirical Casagrande relations
    between Teff and Color. The relationship between mass and Teff will be
    taken from the DSEP isochrones.
    '''
    mass_teff_interpolator = mass_to_teff_dsep_interpolator(age=age,
        metallicity=metallicity)
    star_teff = mass_teff_interpolator(mass)

    color = Casagrande_inverted_color(color, star_teff, metallicity)

    return color

def calculate_binary_band_flux_ratio_Casagrande_DSEP(
    band, mass1, mass2, metallicity, bolcolor, age=1.5):
    '''Calculate the flux ratio between two stars using Casagrande and DSEP.

    This function calculates the flux ratio by essentially running a
    mass-bolometric luminosity relation from DSEP, and then correcting the
    bolometric luminosity to an in-band luminosity from the Bolometric
    Corrections in Casagrande et al (2010).
    '''
    bol_color1 = calculate_single_star_color_Casagrande_DSEP(
        bolcolor, mass1, metallicity, age=age)
    bol_color2 = calculate_single_star_color_Casagrande_DSEP(
        bolcolor, mass2, metallicity, age=age)

    bolratio = (Casagrande_Bolometric_Flux(
        band, 0, bolcolor, bol_color2, metallicity) / 
                Casagrande_Bolometric_Flux(
        band, 0, bolcolor, bol_color1, metallicity))

    mass_lum_interpolator = mass_to_bolometric_luminosity_dsep_interpolator(
        age, metallicity)
    lumratio = mass_lum_interpolator(mass1) / mass_lum_interpolator(mass2)

    fluxratio = lumratio * bolratio

    return fluxratio

###############################################################################
# Plotting Routines
###############################################################################

def single_color_excess_plot(
    color, primary_mass, secondary_masses, metallicity, age=8, bolcolor="B-V",
    method="Casagrande-DSEP"):
    '''Plots the color excess as a function of secondary mass.

    This function only plots a single color excess as a function of secondary
    mass.'''
    
    colors=[]
    for smass in secondary_masses:
        try:
            if method == "Casagrande-DSEP":
                colorval = calculate_binary_star_color_Casagrande_DSEP(
                    color, primary_mass, smass, metallicity, age=age,
                    bolcolor=bolcolor)
            elif method == "DSEP":
                colorval = calculate_binary_star_color_DSEP(
                    color, primary_mass, smass, metallicity, age=age)
            else:
                raise ValueError("Don't recognize method {0}".format(method))
        except OutOfBoundsError:
            # This occurs when a value is out of bounds.
            break
        colors.append(colorval[0])

    binary_colors = np.array(colors)
    if method == "Casagrande-DSEP":
        primary_color = calculate_single_star_color_Casagrande_DSEP(
            color, primary_mass, metallicity)
    elif method == "DSEP":
        primary_color = calculate_single_star_color_DSEP(
            color, primary_mass, metallicity)
    color_excess = binary_colors - primary_color

    plot_masses = secondary_masses[:len(binary_colors)]
    plt.plot(plot_masses, color_excess, label=color)
    plt.xlabel("Secondary Mass (Msun)")
    plt.ylabel("Color Excess over primary")

def color_excess_plot_comparison(
    colors, primary_mass, secondary_masses, metallicity, age=8, bolcolor="B-V"):
    '''Plots multiple color excesses.

    The color excesses for all of the given colors in the colors list will be
    calculated and plotted on the same scale. That way, the most prominent
    color for differentiating secondaries from a given primary can be
    chosen.'''

    for color in colors:
        single_color_excess_plot(color, primary_mass, secondary_masses,
                                 metallicity, age=age, bolcolor=bolcolor)
    plt.legend(loc="upper right")

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
    colors=["B-V", "V-J", "V-H", "V-KS", "J-KS"]
    primary_mass = 1.0
    secondary_masses = np.linspace(primary_mass, 0.1, 30)
    metallicity = 0.0

    color_excess_plot_comparison(
        colors, primary_mass, secondary_masses, metallicity, age=8,
        bolcolor="V-KS")
    plt.show()

