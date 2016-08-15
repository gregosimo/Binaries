import os
import subprocess
import tempfile
import shutil

import numpy as np
import numpy.core.defchararray as npstr
from astropy.table import Table

import path_config as paths

DESP_PATH = "/home/regulus/simonian/DSep/"

def read_Casagrande_10_Table_4(tblpath=paths.CASAGRANDE_TABLE_4):
    '''Read Table 4 in Casagrande et al (2010).

    This table contains the coefficients for fitting effective temperatures to
    colors (and metallicities). These will also be useful for inverting the
    relationship to get colors as a function of effective temperature.'''

    cas = Table.read(
        tblpath, format="ascii.fixed_width_no_header", data_start=1,
        col_starts=(0, 25, 30, 49, 54, 64, 72, 80, 96, 112, 128, 144, 152),
        col_ends=(23, 28, 32, 52, 57, 71, 79, 95, 111, 127, 143, 151, 173),
        names=["Color", "Met_start", "Met_end", "Color_start", "Color_end", 
               "a0", "a1", "a2", "a3", "a4", "a5", "N", "unc"] )
    cas["Color"] = remove_latex_subscript_formatting(cas["Color"])
    return cas

def read_Casagrande_10_Table_5(tblpath= paths.CASAGRANDE_TABLE_5):
    '''Read Table 5 in Casagrande et al (2010).

    This table contains the coefficients for fitting bolometric fluxes from the
    Earth.
    '''

    cas = Table.read(
        tblpath, format="ascii.fixed_width_no_header", data_start=1,
        col_starts=(0, 16, 41, 46, 65, 70, 80, 88, 104, 120, 136, 152, 168, 184, 192),
        col_ends=(15, 39, 44, 48, 68, 74, 87, 103, 119, 135, 151, 167, 183, 191, 203),
        names=["Band", "Color", "Met_start", "Met_end", "Color_start",
               "Color_end", "b0", "b1", "b2", "b3", "b4", "b5", "b6", "N",
               "unc"] )
    cas["Band"] = remove_latex_subscript_formatting(cas["Band"])
    cas["Color"] = remove_latex_subscript_formatting(cas["Color"])
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
    newcol = col.copy()
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
        raise ValueError("Both Metallicity and Colors are outside of the "
            "calibration bounds.")
    elif out_of_color_bound and extrapolation_exception:
        raise ValueError("Colors are outside of the calibration bounds.")
    elif out_of_met_bound and extrapolation_exception:
        raise ValueError("Metallicities are outside of the calibration "
                "bounds.")

    theta_eff = (casagrande_row["a0"] + casagrande_row["a1"] * colorvals + 
                  casagrande_row["a2"] * colorvals**2 + 
                  casagrande_row["a3"] * colorvals * metallicity + 
                  casagrande_row["a4"] * metallicity + 
                  casagrande_row["a5"] * metallicity**2)
    teff = 5040 / theta_eff
    return teff

def Casagrande_inverted_color(color, teffs, metallicity, 
        extrapolate_exception=True, tblpath=paths.CASAGRANDE_TABLE_4):
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

    out_of_teff_bound = np.any(np.logical_or(
           teffs > Casagrande_Teff(
               color, casagrande_row["Color_start"], metallicity), 
           teffs < Casagrande_Teff(
               color, casagrande_row["Color_end"], metallicity)))
    out_of_met_bound = np.any(np.logical_or(
            metallicity < casagrande_row["Met_start"],
            metallicity > casagrande_row["Met_end"]))

    if out_of_teff_bound and out_of_met_bound and extrapolate_exception:
        raise ValueError("Both Metallicity and Colors are outside of the "
            "calibration bounds.")
    elif out_of_teff_bound and extrapolate_exception:
        raise ValueError("Colors are outside of the calibration bounds.")
    elif out_of_met_bound and extrapolate_exception:
        raise ValueError("Metallicities are outside of the calibration "
                "bounds.")

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

def dsep_isochrone_interpolator(feh, output, bands=1, y=1, alpha=2,
                                executable=paths.DSEP_INTERPOLATOR_EXECUTABLE):
    '''Runs interpolator to generate DSEP isochrones of a given metallicity.

    This is used to get a set of isochrones at a given metallicity, without
    having to worry about the grid. The isochrones will be output to the
    location in output.

    [Fe/H] should be the metallicity of the star. Bands, Y, and Alpha are
    integers which stand for options in DSEP. 
    '''
    args = [executable, bands, y, alpha, feh, output]
    subprocess.call(args)

def dsep_age_splitter(inputfile, outputdir):
    '''Calls the isochrone splitter.

    Oftentimes the isochrones can be really annoying to read in their current
    shape. Therefore, the isochrone splitter splits the isochrones into
    separate files, each corresponding to a different age on the isochrone. The
    isochrone files will be put in outputdir.
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
    basedir, input_filename = os.path.split(inputfile)
    with tempfile.TemporaryDirectory(dir=basedir) as tempdir_object:
        tempdir = tempdir_object.name
        shutil.copy(inputfile, tempdir)



if __name__ == "__main__":
    pass
